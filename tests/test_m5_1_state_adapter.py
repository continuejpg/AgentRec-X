"""M5.1 adapter contract tests: validated shopping facts -> M5 state events.

The frozen adapter ``recommendation/state/shopping_adapter.py`` is the trust boundary
between the accepted multi-turn shopping runtime and the M5-MVP state engine: only
already-validated, *explicit* facts may become events, and event identity is
content-addressed so a retried logical turn is a store-level no-op.

These tests pin that contract directly.  They are deliberately lightweight: importing
``recommendation.state`` needs only Pydantic and the standard library, so the suite runs
with the no-torch interpreter as well as with the full test environment.  The adapter is
tested at its public boundary; the M5 engine itself remains covered by
``tests/test_state_*.py`` and ``tests/test_m5_state_acceptance.py``.

What is asserted here, and why it matters
-----------------------------------------
* **Turn identity is content-addressed.**  Same logical request -> same ``turn_key``;
  any logical change (message, k, sources, constraints, clarification) -> a different
  key.  That is what makes a retry idempotent without a client-supplied turn id.
* **Only three event types are constructible.**  ``TASK_CONTEXT_UPDATED``,
  ``USER_PREFERENCE_STATED`` and ``USER_CONSTRAINT_STATED``; there is no API through
  which an inference or arbitrary text-with-an-intent could become an event.
* **The event envelope cannot be spoofed.**  actor, task, session and scope are derived
  here; preference/constraint scope is always ``TASK`` (never durable ``LONG_TERM``).
* **Malformed input is refused, not repaired.**  Unknown kinds, blank values, oversized
  batches and blank identity fields raise ``ValueError``; credential-looking values are
  refused by the reused M5 payload models instead of being stored.
* **Order and ids are deterministic.**  Rebuilding the same turn (even at a different
  timestamp) produces the same ids and the same order, so replay is comparable run to
  run.
* **The events really feed the engine.**  Appending them to a real ``StateManager``
  over a temporary SQLite store replays to an identical state digest, and the three
  read-only projections work on the replayed state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state import (  # noqa: E402
    PreferenceKind,
    PreferencePolarity,
    PreferenceScope,
    SQLiteEventStore,
    StateManager,
    for_agent,
    for_constraints,
    for_ranker,
)
from recommendation.state.shopping_adapter import (  # noqa: E402
    MAX_TURN_CONSTRAINTS,
    MAX_TURN_PREFERENCES,
    ExplicitConstraint,
    ExplicitPreference,
    TurnIdentity,
    authorized_constraint_pairs,
    build_turn_events,
    events_digest,
    merge_constraint_pairs,
    request_turn_key,
)
from tests.state_fixtures import (  # noqa: E402
    constraint_event,
    hypothesis_event,
    preference_stated_event,
    temporary_database,
)

USER_KEY = "demo-session:user-1"
SESSION_ID = "session-1"
TASK_ID = "task-1"
MESSAGE = "Recommend a light jacket for a weekend hike"
K = 3
SOURCES = ("history", "catalog_search")
TS_A = 1_700_000_000.0
TS_B = 1_700_000_999.0

#: The only event types M5.1 may emit; anything else would be a trust-boundary breach.
SUPPORTED_EVENT_TYPES = frozenset(
    {"task_context_updated", "user_preference_stated", "user_constraint_stated"}
)


def _turn_key(**overrides: object) -> str:
    """A fixed logical-turn key with explicit overrides."""
    options: dict[str, object] = {
        "message": MESSAGE,
        "k": K,
        "sources": list(SOURCES),
        "constraints": (),
        "clarification_id": None,
    }
    options.update(overrides)
    return request_turn_key(**options)  # type: ignore[arg-type]


def _identity(turn_key: str | None = None) -> TurnIdentity:
    """The server-owned identity of one shopping turn."""
    return TurnIdentity(
        user_key=USER_KEY,
        session_id=SESSION_ID,
        task_id=TASK_ID,
        turn_key=turn_key if turn_key is not None else _turn_key(),
    )


def _events(
    *,
    preferences: tuple[ExplicitPreference, ...] = (),
    constraints: tuple[ExplicitConstraint, ...] = (),
    identity: TurnIdentity | None = None,
    message: str = MESSAGE,
    intent_label: str = "recommend",
    timestamp: float = TS_A,
):
    """Build the validated events for the fixed turn."""
    return build_turn_events(
        identity=identity or _identity(),
        message=message,
        intent_label=intent_label,
        preferences=preferences,
        constraints=constraints,
        timestamp=timestamp,
    )


def _payload_field(event, name: str):
    """Read one typed payload field (payloads are closed, frozen models)."""
    return getattr(event.payload, name)


# --------------------------------------------------------------------------- #
# request_turn_key: deterministic, content-addressed logical-turn identity
# --------------------------------------------------------------------------- #


def test_same_logical_request_has_the_same_turn_key() -> None:
    first = _turn_key()
    second = _turn_key()
    assert first == second
    assert first.startswith("turn_")
    assert len(first) == len("turn_") + 24
    assert all(character in "0123456789abcdef" for character in first[len("turn_") :])


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"message": MESSAGE + " please"}, id="message"),
        pytest.param({"k": K + 1}, id="k"),
        pytest.param({"sources": ["history", "find_similar"]}, id="sources"),
        pytest.param({"constraints": [("price_max", "150")]}, id="constraints"),
        pytest.param({"clarification_id": "clarification-1"}, id="clarification"),
    ],
)
def test_each_logical_component_changes_the_turn_key(overrides: dict[str, object]) -> None:
    assert _turn_key(**overrides) != _turn_key()


def test_turn_key_normalises_whitespace_case_and_constraint_order() -> None:
    """Equivalent logical content produces the same key; the key is not a byte copy."""
    base = _turn_key()

    spaced_message = _turn_key(message="  Recommend   a light   jacket for a weekend hike  ")
    assert spaced_message == base

    categorised_sources = _turn_key(sources=(" History ", "CATALOG_SEARCH"))
    assert categorised_sources == base

    # Constraints are order-insensitive (the key is content, not a request byte stream)
    # and normalised by kind case/whitespace and value case/whitespace.
    constraints_a = _turn_key(
        constraints=(("price_max", " 150"), ("brand", "Acme"))
    )
    constraints_b = _turn_key(
        constraints=((" BRAND ", "acme "), ("PRICE_MAX", "150"))
    )
    assert constraints_a == constraints_b

    # No clarification answer and an empty one are the same logical request.
    assert _turn_key(clarification_id="") == base


def test_turn_key_is_transport_independent() -> None:
    """An HTTP-shaped JSON request and a direct typed call are the same logical turn."""
    http_shaped = request_turn_key(
        message=MESSAGE,
        k=K,
        sources=["history", "catalog_search"],
        constraints=[["price_max", "150"], ["brand", "Acme"]],
        clarification_id=None,
    )
    direct = request_turn_key(
        message=MESSAGE,
        k=K,
        sources=("history", "catalog_search"),
        constraints=(("price_max", "150"), ("brand", "Acme")),
        clarification_id=None,
    )
    assert http_shaped == direct


# --------------------------------------------------------------------------- #
# build_turn_events: exactly the supported vocabulary, in deterministic order
# --------------------------------------------------------------------------- #


def test_build_emits_exactly_the_three_supported_event_types() -> None:
    events = _events(
        preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),),
        constraints=(ExplicitConstraint(PreferenceKind.PRICE_MAX, "150"),),
    )
    assert len(events) == 3
    assert tuple(event.event_type.value for event in events) == (
        "task_context_updated",
        "user_preference_stated",
        "user_constraint_stated",
    )
    assert {event.event_type.value for event in events} == SUPPORTED_EVENT_TYPES
    # The event ids are unique within one turn; only the duplicate-input case collapses.
    assert len({event.event_id for event in events}) == len(events)


def test_event_identity_actors_and_provenance_are_preserved() -> None:
    preference = ExplicitPreference(
        kind=PreferenceKind.FEATURE,
        value="lightweight",
        source_text="I want something lightweight.",
    )
    constraint = ExplicitConstraint(
        kind=PreferenceKind.PRICE_MAX,
        value="150",
        source_text="My budget is $150.",
    )
    events = _events(preferences=(preference,), constraints=(constraint,))
    context, stated, constrained = events

    for event in events:
        assert event.user_key == USER_KEY
        assert event.task_id == TASK_ID
        assert event.session_id == SESSION_ID
        assert event.source == "shopping_state_adapter"
        assert event.timestamp == TS_A
        assert event.sequence is None

    # TASK_CONTEXT_UPDATED is agent-authored; user facts are user-authored.
    assert context.actor.value == "agent"
    assert stated.actor.value == "user"
    assert constrained.actor.value == "user"

    # The typed task context carries the validated intent label and the bounded message.
    assert _payload_field(context, "intent") == "recommend"
    assert _payload_field(context, "request_text") == MESSAGE

    # Provenance spans travel with the event (they are bounded by the M5 payload model).
    assert _payload_field(stated, "source_text") == preference.source_text
    assert _payload_field(stated, "kind") is PreferenceKind.FEATURE
    assert _payload_field(stated, "value") == "lightweight"
    assert _payload_field(stated, "polarity") is PreferencePolarity.PREFER

    assert _payload_field(constrained, "source_text") == constraint.source_text
    assert _payload_field(constrained, "kind") is PreferenceKind.PRICE_MAX
    assert _payload_field(constrained, "value") == "150"


def test_scope_is_forced_to_task_and_never_long_term() -> None:
    """A normal shopping turn must never write durable long-term state."""
    events = _events(
        preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),),
        constraints=(ExplicitConstraint(PreferenceKind.PRICE_MAX, "150"),),
    )
    for event in events:
        if event.event_type.value == "task_context_updated":
            # The context payload has no scope field at all.
            assert event.scope is None
            assert event.applies_long_term is False
            continue
        assert event.scope is PreferenceScope.TASK
        assert _payload_field(event, "scope") is PreferenceScope.TASK
        assert event.applies_long_term is False


def test_rebuild_is_deterministic_including_event_ids_across_timestamps() -> None:
    """A retried logical turn re-emits the same ids, so the store treats it as a retry."""
    preferences = (ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),)
    constraints = (ExplicitConstraint(PreferenceKind.PRICE_MAX, "150"),)
    first = _events(preferences=preferences, constraints=constraints, timestamp=TS_A)
    second = _events(
        preferences=preferences,
        constraints=constraints,
        identity=_identity(),
        timestamp=TS_B,
    )

    assert tuple(event.event_id for event in first) == tuple(
        event.event_id for event in second
    )
    # ``semantic_fields`` excludes the timestamp: the store sees the same submission.
    for rebuilt, original in zip(second, first):
        assert rebuilt.is_same_submission(original)
    assert events_digest(first) == events_digest(second)


def test_duplicate_preferences_and_constraints_collapse_to_one_event() -> None:
    duplicate_preference = ExplicitPreference(PreferenceKind.FEATURE, "lightweight")
    duplicate_constraint = ExplicitConstraint(PreferenceKind.PRICE_MAX, "150")
    # The same identity stated twice, differing only in whitespace/case/provenance text.
    variants = (
        duplicate_preference,
        ExplicitPreference(PreferenceKind.FEATURE, "  LightWeight  "),
        duplicate_preference,
    )
    constraint_variants = (
        duplicate_constraint,
        ExplicitConstraint(PreferenceKind.PRICE_MAX, " 150 "),
        duplicate_constraint,
    )
    events = _events(preferences=variants, constraints=constraint_variants)
    assert [event.event_type.value for event in events] == [
        "task_context_updated",
        "user_preference_stated",
        "user_constraint_stated",
    ]


def test_event_order_is_deterministic() -> None:
    """Context first, then preferences/constraints sorted by kind/value/polarity."""
    preferences = (
        ExplicitPreference(PreferenceKind.PRICE_MAX, "100"),
        ExplicitPreference(PreferenceKind.BRAND, "Acme"),
        ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),
    )
    constraints = (
        ExplicitConstraint(PreferenceKind.COLOR, "red"),
        ExplicitConstraint(PreferenceKind.CATEGORY, "hiking boots"),
    )
    events = _events(preferences=preferences, constraints=constraints)
    ordered_kinds = [
        _payload_field(event, "kind").value
        for event in events
        if event.event_type.value != "task_context_updated"
    ]
    assert ordered_kinds == [
        "brand",
        "feature",
        "price_max",
        "category",
        "color",
    ]

    # A different input order produces the same event order (and the same ids).
    rebuilt = _events(
        preferences=tuple(reversed(preferences)),
        constraints=tuple(reversed(constraints)),
    )
    assert tuple(event.event_id for event in rebuilt) == tuple(
        event.event_id for event in events
    )


# --------------------------------------------------------------------------- #
# malformed input is refused, never repaired or stored
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_preference",
    [
        pytest.param(ExplicitPreference("not-a-kind", "x"), id="unknown-kind"),  # type: ignore[arg-type]
        pytest.param(ExplicitPreference(PreferenceKind.FEATURE, "   "), id="blank-value"),
        pytest.param(
            ExplicitPreference(PreferenceKind.FEATURE, "lightweight", "sideways"),  # type: ignore[arg-type]
            id="unknown-polarity",
        ),
    ],
)
def test_a_malformed_preference_is_refused(bad_preference: ExplicitPreference) -> None:
    with pytest.raises(ValueError):
        _events(preferences=(bad_preference,))


@pytest.mark.parametrize(
    "bad_constraint",
    [
        pytest.param(ExplicitConstraint("not-a-kind", "x"), id="unknown-kind"),  # type: ignore[arg-type]
        pytest.param(ExplicitConstraint(PreferenceKind.PRICE_MAX, ""), id="blank-value"),
    ],
)
def test_a_malformed_constraint_is_refused(bad_constraint: ExplicitConstraint) -> None:
    with pytest.raises(ValueError):
        _events(constraints=(bad_constraint,))


def test_oversized_preference_batches_are_refused() -> None:
    too_many = tuple(
        ExplicitPreference(PreferenceKind.FEATURE, f"feature-{index}")
        for index in range(MAX_TURN_PREFERENCES + 1)
    )
    with pytest.raises(ValueError):
        _events(preferences=too_many)


def test_oversized_constraint_batches_are_refused() -> None:
    too_many = tuple(
        ExplicitConstraint(PreferenceKind.PRICE_MAX, str(100 + index))
        for index in range(MAX_TURN_CONSTRAINTS + 1)
    )
    with pytest.raises(ValueError):
        _events(constraints=too_many)


@pytest.mark.parametrize("blank_field", ["user_key", "session_id", "task_id", "turn_key"])
def test_blank_identity_fields_are_refused(blank_field: str) -> None:
    values = {
        "user_key": USER_KEY,
        "session_id": SESSION_ID,
        "task_id": TASK_ID,
        "turn_key": _turn_key(),
    }
    for blank in ("", "   "):
        values[blank_field] = blank
        with pytest.raises(ValueError):
            build_turn_events(
                identity=TurnIdentity(**values),
                message=MESSAGE,
                intent_label="recommend",
                timestamp=TS_A,
            )


def test_non_turn_identity_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        build_turn_events(
            identity={"user_key": USER_KEY},  # type: ignore[arg-type]
            message=MESSAGE,
            intent_label="recommend",
            timestamp=TS_A,
        )


class TestSecretLikeValues:
    """The M5 payload models, not the adapter, own the credential rule."""

    def test_a_secret_looking_preference_value_is_refused(self) -> None:
        secret = "sk-abcdefghijklmnop"
        with pytest.raises(ValueError):
            _events(preferences=(ExplicitPreference(PreferenceKind.FEATURE, secret),))

    def test_a_secret_looking_constraint_value_is_refused(self) -> None:
        card = "4111 1111 1111 1111"
        with pytest.raises(ValueError):
            _events(constraints=(ExplicitConstraint(PreferenceKind.FREE_FORM_CONSTRAINT, card),))

    def test_a_secret_looking_provenance_span_is_redacted_not_stored(self) -> None:
        secret = "sk-abcdefghijklmnop"
        events = _events(
            preferences=(
                ExplicitPreference(
                    PreferenceKind.FEATURE,
                    "lightweight",
                    source_text=f"my key is {secret}",
                ),
            )
        )
        stated = events[1]
        assert _payload_field(stated, "source_text") == "[redacted]"
        assert secret not in stated.canonical_payload()["source_text"]


# --------------------------------------------------------------------------- #
# authorized_constraint_pairs: only authorized explicit hard constraints
# --------------------------------------------------------------------------- #


def _replay_state(*events) -> object:
    """Fold fixture events into a real replayed state over a temporary SQLite store."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            state = None
            for event in events:
                state = manager.apply_event(event)
            assert state is not None
            return state
        finally:
            store.close()


def test_replayed_budget_corrections_project_only_the_latest_effective_value() -> None:
    """SQLite replay must retain history while only the newest budget is effective."""
    state = _replay_state(
        constraint_event(event_id="evt-price-30", value="30"),
        constraint_event(event_id="evt-price-10", value="10"),
        constraint_event(event_id="evt-price-50", value="50"),
    )
    assert authorized_constraint_pairs(state) == (("price_max", "50"),)
    assert [constraint.value for constraint in state.authorized_hard_constraints] == ["50"]
    assert [constraint.status.value for constraint in state.hard_constraints] == [
        "superseded",
        "superseded",
        "active",
    ]


def test_authorized_pairs_contain_only_explicit_hard_constraints() -> None:
    state = _replay_state(
        preference_stated_event(user_key=USER_KEY, task_id=TASK_ID, session_id=SESSION_ID),
        constraint_event(user_key=USER_KEY, task_id=TASK_ID, session_id=SESSION_ID),
    )
    # The explicit preference is a ranking signal, not a constraint.
    pairs = authorized_constraint_pairs(state)
    assert pairs == (("price_max", "150"),)
    for kind, _value in pairs:
        assert kind in {member.value for member in PreferenceKind}

    projection = for_constraints(state)
    assert tuple((view.kind.value, view.value) for view in projection.constraints) == pairs
    assert projection.excluded_hypothesis_count == 0


def test_authorized_pairs_exclude_an_inferred_hypothesis() -> None:
    state = _replay_state(
        constraint_event(user_key=USER_KEY, task_id=TASK_ID, session_id=SESSION_ID),
        # An inference of the same constraint kind must never become a filter.
        hypothesis_event(
            user_key=USER_KEY,
            task_id=TASK_ID,
            session_id=SESSION_ID,
            kind="price_max",
            value="300",
        ),
    )
    pairs = authorized_constraint_pairs(state)
    assert pairs == (("price_max", "150"),)
    assert ("price_max", "300") not in pairs
    projection = for_constraints(state)
    assert projection.excluded_hypothesis_count == 1


def test_authorized_pairs_apply_task_over_long_term_precedence() -> None:
    state = _replay_state(
        constraint_event(
            user_key=USER_KEY,
            task_id=TASK_ID,
            session_id=SESSION_ID,
            kind="price_max",
            value="200",
            scope="long_term",
        ),
        constraint_event(
            user_key=USER_KEY,
            task_id=TASK_ID,
            session_id=SESSION_ID,
            kind="price_max",
            value="150",
            scope="task",
        ),
    )
    # Only the task-scope value is authorized for this task; the long-term record stays.
    assert authorized_constraint_pairs(state) == (("price_max", "150"),)
    projection = for_constraints(state)
    assert projection.suppressed_long_term_ids, "the suppressed long-term constraint is reported"
    assert len(state.hard_constraints) == 2, "nothing is deleted from the state"


# --------------------------------------------------------------------------- #
# merge_constraint_pairs: trusted order first, union, de-duplicated
# --------------------------------------------------------------------------- #


def test_merge_keeps_trusted_order_and_appends_m5_extras() -> None:
    trusted = (("price_max", "150"), ("brand", "Acme"))
    authorized = (("color", "Red"), ("category", "hiking boots"))
    merged = merge_constraint_pairs(trusted, authorized)
    assert merged == (
        ("price_max", "150"),
        ("brand", "Acme"),
        ("color", "Red"),
        ("category", "hiking boots"),
    )


def test_merge_current_kind_supersedes_a_stale_m5_value() -> None:
    """Current input wins over an older M5 value of the same kind; no over-constraint."""
    trusted = (("price_max", "10"),)
    authorized = (("price_max", "30"), ("color", "red"))
    merged = merge_constraint_pairs(
        trusted, authorized, current_turn_kinds=("price_max",)
    )
    assert merged == (("price_max", "10"), ("color", "red"))


def test_merge_dedupes_by_kind_and_casefolded_value() -> None:
    trusted = (("price_max", "150"), ("color", "Red"))
    authorized = (
        ("price_max", "150"),
        (" PRICE_MAX ", " 150 "),
        ("color", "red"),
        ("COLOR", "RED"),
        ("brand", "Acme"),
    )
    merged = merge_constraint_pairs(trusted, authorized)
    # A duplicate is one constraint, not two evaluations; the trusted record wins.
    assert merged == (("price_max", "150"), ("color", "Red"), ("brand", "Acme"))


def test_merge_skips_blank_pairs_and_normalises_what_it_keeps() -> None:
    trusted = ((" color ", "  dark   blue "), ("", "x"), ("brand", "   "))
    authorized = (("", ""),)
    assert merge_constraint_pairs(trusted, authorized) == (("color", "dark blue"),)


def test_merge_of_two_empty_sequences_is_empty() -> None:
    assert merge_constraint_pairs((), ()) == ()


# --------------------------------------------------------------------------- #
# end-to-end: adapter events feed a real StateManager and projections
# --------------------------------------------------------------------------- #


def _built_events():
    """The deterministic event batch used by the end-to-end tests."""
    turn_key = _turn_key()
    return build_turn_events(
        identity=_identity(turn_key),
        message=MESSAGE,
        intent_label="recommend",
        preferences=(
            ExplicitPreference(
                PreferenceKind.FEATURE,
                "lightweight",
                source_text="I want something lightweight.",
            ),
        ),
        constraints=(
            ExplicitConstraint(
                PreferenceKind.PRICE_MAX,
                "150",
                source_text="My budget is $150.",
            ),
        ),
        timestamp=TS_A,
    )


def test_events_feed_a_real_state_manager_end_to_end() -> None:
    events = _built_events()
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            state = None
            for event in events:
                state = manager.apply_event(event)
            assert state is not None

            assert state.user_key == USER_KEY
            assert state.task_id == TASK_ID
            assert state.interaction.event_count == len(events)
            assert state.task_context.intent == "recommend"
            assert state.task_context.request_text == MESSAGE
            assert len(state.active_preference_beliefs) == 1
            assert len(state.authorized_hard_constraints) == 1
            applied_digest = state.state_digest()
            assert store.count(USER_KEY) == len(events)
        finally:
            store.close()

        # A fresh store/manager over the same file reconstructs the identical state.
        replayed_store = SQLiteEventStore(database)
        try:
            replayed = StateManager(replayed_store).replay_state(USER_KEY, TASK_ID)
            assert replayed.state_digest() == applied_digest
            assert replayed.state_digest() == state.state_digest()

            agent_view = for_agent(replayed)
            assert agent_view.task_id == TASK_ID
            assert agent_view.intent == "recommend"
            assert [summary.kind.value for summary in agent_view.preferences] == ["feature"]
            assert [summary.value for summary in agent_view.preferences] == ["lightweight"]
            assert [summary.kind.value for summary in agent_view.constraints] == ["price_max"]
            assert [summary.value for summary in agent_view.constraints] == ["150"]

            ranker_view = for_ranker(replayed)
            assert [feature.kind.value for feature in ranker_view.features] == ["feature"]
            assert ranker_view.features[0].strength_rank == 3
            assert ranker_view.features[0].is_soft is False

            constraint_view = for_constraints(replayed)
            assert [
                (view.kind.value, view.value) for view in constraint_view.constraints
            ] == [("price_max", "150")]
            assert constraint_view.excluded_hypothesis_count == 0
            assert authorized_constraint_pairs(replayed) == (("price_max", "150"),)
        finally:
            replayed_store.close()


def test_reapplying_the_same_turn_events_is_idempotent() -> None:
    events = _built_events()
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in events:
                manager.apply_event(event)
            first = manager.replay_state(USER_KEY, TASK_ID)
            count_after_first = store.count(USER_KEY)

            # The retried turn rebuilds the same ids; the store must treat it as a retry.
            for event in _built_events():
                manager.apply_event(event)
            second = manager.replay_state(USER_KEY, TASK_ID)

            assert store.count(USER_KEY) == count_after_first == len(events)
            assert second.state_digest() == first.state_digest()
            assert second.interaction.event_count == len(events)
        finally:
            store.close()
