"""M5.2 adapter contract tests: direction, retractions, turn identity, snapshots.

This suite pins the M5.2 extension of the frozen adapter
``recommendation/state/shopping_adapter.py`` at its public boundary.  It is deliberately
**light**: importing ``recommendation.state`` needs only Pydantic plus the standard
library, so the file runs with the no-torch interpreter as well as with the full Python
3.10 test venv.

What is asserted, and why it matters
------------------------------------
* **Direction is a trusted choice, never inferred.**  A clarification option (for
  example "lower price") can carry ``MINIMIZE``/``MAXIMIZE``; the adapter validates the
  value and emits it on the ``USER_PREFERENCE_STATED`` payload (and the reducer applies it
  to the belief).  An unknown direction is refused, not guessed.
* **A retraction identifies what is withdrawn.**  ``ExplicitRetraction`` can select a kind
  (optionally narrowed by value/polarity) or a value alone; a value-only selector must be
  resolved with :func:`resolve_retraction` against the current task state *before* it can
  become an event, because the M5 payload requires a concrete kind.  An unmatched or
  ambiguous value-only removal returns ``None`` (the caller reports a bounded mismatch;
  it never guesses ``FREE_FORM_CONSTRAINT`` or any other kind).
* **Retraction events are TASK-scoped and auditable.**  They are user-authored, carry the
  accepted M5 provenance span, reuse deterministic event ids from the logical turn
  identity, de-duplicate repeated selectors, and never write long-term state.
* **Turn identity has strict precedence.**  ``idempotency_key`` (retry-safe) >
  ``server_turn_id`` (server-owned allocation) > content fallback (direct callers only),
  scoped by ``task_id``.  The reporter string makes the chosen source explicit rather than
  silent.
* **Snapshots are opaque and trusted.**  A candidate-pool reference is order-insensitive
  (it is a set), a ranking snapshot is order-sensitive (it is the presented order), and the
  emitted events are ``trusted_component`` references carrying only ref/count/digest plus
  ``snapshot_schema_version``.  Product identities stay in the trusted layer: the decision
  state stores the reference, never the rows.

Nothing here weakens an M5 or M5.1 assertion: the earlier suites remain the authority for
those contracts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state import (  # noqa: E402
    BeliefUpdater,
    PreferenceDirection,
    PreferenceKind,
    PreferenceLifecycle,
    PreferencePolarity,
    PreferenceScope,
    PreferenceStatus,
    SQLiteEventStore,
    StateManager,
    UnsupportedTransitionError,
    for_agent,
    for_ranker,
)
from recommendation.state.shopping_adapter import (  # noqa: E402
    ADAPTER_SOURCE,
    CANDIDATE_SNAPSHOT_SOURCE,
    MAX_TURN_RETRACTIONS,
    RANKING_SNAPSHOT_SOURCE,
    ExplicitConstraint,
    ExplicitPreference,
    ExplicitRetraction,
    TurnIdentity,
    build_snapshot_events,
    build_turn_events,
    candidate_pool_snapshot,
    events_digest,
    ranking_snapshot,
    resolve_retraction,
    resolve_turn_identity,
)
from tests.state_fixtures import temporary_database  # noqa: E402

USER_KEY = "demo-session:user-1"
SESSION_ID = "session-1"
TASK_ID = "task-1"
MESSAGE = "Recommend a light jacket for a weekend hike"
TS_A = 1_700_000_000.0
TS_B = 1_700_000_999.0

def _identity(turn_key: str = "turn-fixed") -> TurnIdentity:
    return TurnIdentity(
        user_key=USER_KEY,
        session_id=SESSION_ID,
        task_id=TASK_ID,
        turn_key=turn_key,
    )


def _events(
    *,
    preferences=(),
    constraints=(),
    retractions=(),
    identity: TurnIdentity | None = None,
    timestamp: float = TS_A,
):
    """Build the validated events for one fixed logical turn."""
    return build_turn_events(
        identity=identity or _identity(),
        message=MESSAGE,
        intent_label="recommend",
        preferences=tuple(preferences),
        constraints=tuple(constraints),
        retractions=tuple(retractions),
        timestamp=timestamp,
    )


def _replayed_state(*, preferences=(), constraints=()):
    """Fold a real adapter event batch into a replayed M5 state over a temporary store."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(preferences=preferences, constraints=constraints):
                manager.apply_event(event)
            return manager.replay_state(USER_KEY, TASK_ID)
        finally:
            store.close()


def _retracted(event):
    """Return the retraction payload of an emitted PREFERENCE_RETRACTED event."""
    assert event.event_type.value == "preference_retracted"
    return event.payload


# =========================================================================== #
# A. ExplicitPreference.direction: validated trusted direction, not inferred
# =========================================================================== #


def test_direction_defaults_to_none() -> None:
    """An ordinary explicit statement carries no direction (nothing is inferred)."""
    events = _events(preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),))
    stated = events[1]
    assert stated.payload.direction is None
    # The dataclass default is also ``None``, so callers never get a silent direction.
    assert ExplicitPreference(PreferenceKind.FEATURE, "lightweight").direction is None


@pytest.mark.parametrize(
    "direction",
    [PreferenceDirection.MINIMIZE, PreferenceDirection.MAXIMIZE, PreferenceDirection.EXACT],
)
def test_direction_is_emitted_on_the_stated_preference_payload(
    direction: PreferenceDirection,
) -> None:
    """A trusted directional choice survives into the event payload unchanged."""
    events = _events(
        preferences=(
            ExplicitPreference(
                PreferenceKind.FEATURE,
                "weight",
                direction=direction,
                source_text="I want the lightest one you have.",
            ),
        )
    )
    stated = events[1]
    assert stated.payload.kind is PreferenceKind.FEATURE
    assert stated.payload.value == "weight"
    assert stated.payload.direction is direction


def test_direction_accepts_its_wire_value_and_rejects_an_unknown_one() -> None:
    """The direction field is validated; a made-up direction is refused."""
    events = _events(
        preferences=(
            ExplicitPreference(PreferenceKind.FEATURE, "price", direction="minimize"),  # type: ignore[arg-type]
        )
    )
    assert events[1].payload.direction is PreferenceDirection.MINIMIZE

    with pytest.raises(ValueError):
        _events(
            preferences=(
                ExplicitPreference(PreferenceKind.FEATURE, "price", direction="sideways"),  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError):
        _events(
            preferences=(
                ExplicitPreference(PreferenceKind.FEATURE, "price", direction=3),  # type: ignore[arg-type]
            )
        )


def test_direction_reaches_the_state_and_the_ranker_projection() -> None:
    """The reducer and the read-only projections see the trusted direction."""
    state = _replayed_state(
        preferences=(
            ExplicitPreference(
                PreferenceKind.FEATURE, "weight", direction=PreferenceDirection.MINIMIZE
            ),
        )
    )
    (belief,) = state.active_preference_beliefs
    assert belief.kind is PreferenceKind.FEATURE
    assert belief.value == "weight"
    assert belief.direction is PreferenceDirection.MINIMIZE

    (feature,) = for_ranker(state).features
    assert feature.direction is PreferenceDirection.MINIMIZE
    assert feature.is_soft is False


def test_restated_preference_without_direction_keeps_the_known_direction() -> None:
    """Omitting direction means "not restated", never "forget the direction"."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                preferences=(
                    ExplicitPreference(
                        PreferenceKind.FEATURE,
                        "weight",
                        direction=PreferenceDirection.MINIMIZE,
                    ),
                )
            ):
                manager.apply_event(event)
            # A later turn restates the same preference without a direction.
            for event in _events(
                identity=_identity("turn-second"),
                preferences=(ExplicitPreference(PreferenceKind.FEATURE, "weight"),),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            (belief,) = manager.replay_state(USER_KEY, TASK_ID).active_preference_beliefs
            assert belief.direction is PreferenceDirection.MINIMIZE
        finally:
            store.close()


# =========================================================================== #
# A2. M5.2 general correction: replace_same_kind (REPLACE without a named target)
# =========================================================================== #


def test_replace_same_kind_is_emitted_as_a_correction_flag() -> None:
    """A general REPLACE carries the flag, never an invented named selector."""
    events = _events(
        preferences=(
            ExplicitPreference(
                PreferenceKind.COLOR,
                "blue",
                replace_same_kind=True,
                source_text="Actually I prefer blue instead.",
            ),
        )
    )
    stated = events[1]
    assert stated.payload.replace_same_kind is True
    assert stated.payload.replaces_value is None
    # A normal ADD carries neither selector.
    plain = _events(preferences=(ExplicitPreference(PreferenceKind.COLOR, "blue"),))[1]
    assert plain.payload.replace_same_kind is False
    assert plain.payload.replaces_value is None


def test_replace_same_kind_and_a_named_selector_are_mutually_exclusive() -> None:
    """The M5 payload refuses a statement that names a target AND claims a general replace."""
    with pytest.raises(ValueError):
        _events(
            preferences=(
                ExplicitPreference(
                    PreferenceKind.COLOR,
                    "blue",
                    replaces_value="black",
                    replace_same_kind=True,
                ),
            )
        )


def test_replace_same_kind_supersedes_the_earlier_same_kind_value() -> None:
    """A general REPLACE supersedes the old same-kind/same-polarity explicit belief."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                preferences=(ExplicitPreference(PreferenceKind.COLOR, "black"),)
            ):
                manager.apply_event(event)
            before = manager.replay_state(USER_KEY, TASK_ID)
            (old_black,) = before.active_preference_beliefs
            assert old_black.value == "black"

            for event in _events(
                identity=_identity("turn-correction"),
                preferences=(
                    ExplicitPreference(
                        PreferenceKind.COLOR, "blue", replace_same_kind=True
                    ),
                ),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            after = manager.replay_state(USER_KEY, TASK_ID)
            assert [
                (belief.kind.value, belief.value)
                for belief in after.active_preference_beliefs
            ] == [("color", "blue")]
            (old_record,) = [
                belief
                for belief in after.preference_beliefs
                if belief.value.casefold() == "black"
            ]
            assert old_record.belief_id == old_black.belief_id
            assert old_record.is_active is False
            assert old_record.status is PreferenceStatus.SUPERSEDED
            assert old_record.lifecycle is PreferenceLifecycle.EXPLICIT
            assert len(old_record.supporting_event_ids) == 2
            (new_blue,) = [
                belief
                for belief in after.active_preference_beliefs
                if belief.value.casefold() == "blue"
            ]
            assert new_blue.lifecycle is PreferenceLifecycle.EXPLICIT
            assert new_blue.status is PreferenceStatus.ACTIVE
            assert new_blue.last_event_id == old_record.last_event_id
        finally:
            store.close()


def test_a_plain_add_after_a_correction_does_not_supersede() -> None:
    """Only an explicit correction supersedes; a second independent statement adds."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                identity=_identity("turn-correction"),
                preferences=(
                    ExplicitPreference(
                        PreferenceKind.COLOR, "blue", replace_same_kind=True
                    ),
                ),
            ):
                manager.apply_event(event)
            for event in _events(
                identity=_identity("turn-add"),
                preferences=(ExplicitPreference(PreferenceKind.COLOR, "green"),),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER_KEY, TASK_ID)
            assert {
                belief.value for belief in state.active_preference_beliefs
            } == {"blue", "green"}
        finally:
            store.close()


# =========================================================================== #
# B. ExplicitRetraction: selector validity and event emission
# =========================================================================== #


def test_a_retraction_needs_a_kind_or_a_value_selector() -> None:
    """An empty retraction is malformed; kind-only and value-only are representable."""
    with pytest.raises(ValueError):
        ExplicitRetraction()
    with pytest.raises(ValueError):
        ExplicitRetraction(kind=None, value="   ")
    with pytest.raises(ValueError):
        ExplicitRetraction(kind=None, value="")
    # Kind-only (withdraw the whole kind) and kind+value are valid as constructed.
    assert ExplicitRetraction(kind=PreferenceKind.FEATURE).kind is PreferenceKind.FEATURE
    both = ExplicitRetraction(kind=PreferenceKind.FEATURE, value="lightweight")
    assert both.kind is PreferenceKind.FEATURE
    assert both.value == "lightweight"
    # A value-only selector is accepted by the dataclass, but must be resolved first.
    assert ExplicitRetraction(value="lightweight").kind is None


def test_a_value_only_retraction_must_be_resolved_before_build() -> None:
    """The M5 retraction payload requires a concrete kind: building guesses nothing."""
    with pytest.raises(ValueError):
        _events(retractions=(ExplicitRetraction(value="lightweight"),))
    # The same selectors, resolved to a kind, are accepted.
    events = _events(
        retractions=(ExplicitRetraction(kind=PreferenceKind.FEATURE, value="lightweight"),)
    )
    assert events[-1].event_type.value == "preference_retracted"


def test_kind_only_and_kind_plus_value_retraction_payloads() -> None:
    """Kind-wide retractions carry ``value=None``; narrowed ones carry the value."""
    kind_only = _events(retractions=(ExplicitRetraction(kind=PreferenceKind.FEATURE),))[-1]
    payload = _retracted(kind_only)
    assert payload.kind is PreferenceKind.FEATURE
    assert payload.value is None
    assert payload.polarity is None
    assert payload.scope is PreferenceScope.TASK

    narrowed = _events(
        retractions=(
            ExplicitRetraction(
                kind=PreferenceKind.FEATURE,
                value="lightweight",
                polarity=PreferencePolarity.PREFER,
                source_text="Forget the lightweight option.",
            ),
        )
    )[-1]
    payload = _retracted(narrowed)
    assert payload.kind is PreferenceKind.FEATURE
    assert payload.value == "lightweight"
    assert payload.polarity is PreferencePolarity.PREFER
    assert payload.source_text == "Forget the lightweight option."


def test_retraction_events_are_task_scoped_user_authored_and_auditable() -> None:
    """A retraction is a TASK-scoped user act with the accepted adapter provenance."""
    event = _events(
        retractions=(
            ExplicitRetraction(
                kind=PreferenceKind.FEATURE,
                source_text="Actually, drop the lightweight requirement.",
            ),
        )
    )[-1]
    assert event.actor.value == "user"
    assert event.source == ADAPTER_SOURCE
    assert event.user_key == USER_KEY
    assert event.task_id == TASK_ID
    assert event.session_id == SESSION_ID
    assert event.scope is PreferenceScope.TASK
    assert event.applies_long_term is False
    assert event.payload.source_text == "Actually, drop the lightweight requirement."


def test_retraction_event_ids_are_deterministic_and_deduped() -> None:
    """A retry re-emits the same id; repeated/equivalent selectors collapse to one event."""
    retraction = ExplicitRetraction(kind=PreferenceKind.FEATURE, value="lightweight")
    first = _events(retractions=(retraction,), timestamp=TS_A)
    second = _events(retractions=(retraction,), timestamp=TS_B)
    assert events_digest(first) == events_digest(second)
    assert tuple(event.event_id for event in first) == tuple(
        event.event_id for event in second
    )
    for rebuilt, original in zip(second, first):
        assert rebuilt.is_same_submission(original)

    duplicated = _events(
        retractions=(
            retraction,
            retraction,
            # Same selector with different case/whitespace is the same retraction; the
            # first occurrence is the one that is kept.
            ExplicitRetraction(kind=PreferenceKind.FEATURE, value="  LightWeight  "),
        )
    )
    assert [event.event_type.value for event in duplicated] == [
        "task_context_updated",
        "preference_retracted",
    ]
    assert duplicated[-1].payload.value == "lightweight"
    # A different logical turn produces a different event id.
    other = _events(
        identity=_identity("turn-other"), retractions=(retraction,), timestamp=TS_A
    )
    assert other[-1].event_id != first[-1].event_id


def test_retractions_are_deterministically_ordered_after_other_facts() -> None:
    """Order is context, preferences, constraints, then retractions (each sorted)."""
    events = _events(
        preferences=(ExplicitPreference(PreferenceKind.BRAND, "Acme"),),
        constraints=(ExplicitConstraint(PreferenceKind.PRICE_MAX, "150"),),
        retractions=(
            ExplicitRetraction(kind=PreferenceKind.FEATURE, value="lightweight"),
            ExplicitRetraction(kind=PreferenceKind.COLOR),
        ),
    )
    assert [event.event_type.value for event in events] == [
        "task_context_updated",
        "user_preference_stated",
        "user_constraint_stated",
        "preference_retracted",
        "preference_retracted",
    ]
    # Retraction order is by kind then value: color, then feature/lightweight.
    assert [event.payload.kind for event in events if event.event_type.value == "preference_retracted"] == [
        PreferenceKind.COLOR,
        PreferenceKind.FEATURE,
    ]


def test_oversized_retraction_batches_are_refused() -> None:
    too_many = tuple(
        ExplicitRetraction(kind=PreferenceKind.FEATURE, value=f"feature-{index}")
        for index in range(MAX_TURN_RETRACTIONS + 1)
    )
    with pytest.raises(ValueError):
        _events(retractions=too_many)


# =========================================================================== #
# C. resolve_retraction against a real M5 state
# =========================================================================== #


def test_kind_selector_is_passed_through_unchanged() -> None:
    """An explicit kind is authoritative; the state is not consulted to rewrite it."""
    state = _replayed_state(
        preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),)
    )
    retraction = ExplicitRetraction(
        kind=PreferenceKind.BRAND, value="Acme", source_text="not a fan"
    )
    resolved = resolve_retraction(state, retraction)
    assert resolved == retraction
    assert resolved is retraction


def test_kind_selector_passes_through_even_when_nothing_matches() -> None:
    """No-match detection is the reducer's job for an explicit kind; nothing is invented."""
    state = _replayed_state(
        preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),)
    )
    resolved = resolve_retraction(
        state, ExplicitRetraction(kind=PreferenceKind.BRAND, value="Acme")
    )
    assert resolved is not None
    assert resolved.kind is PreferenceKind.BRAND
    # ... and the reducer refuses a retraction that matches no active belief.
    event = build_turn_events(
        identity=_identity(),
        message=MESSAGE,
        intent_label="recommend",
        retractions=(resolved,),
        timestamp=TS_A,
    )[-1]
    with pytest.raises(UnsupportedTransitionError):
        BeliefUpdater().apply(state, event)


def test_a_unique_value_resolves_to_its_kind() -> None:
    """A value-only removal with exactly one match is resolved instead of guessed."""
    state = _replayed_state(
        preferences=(ExplicitPreference(PreferenceKind.FEATURE, "LightWeight"),)
    )
    resolved = resolve_retraction(state, ExplicitRetraction(value="  lightweight  "))
    assert resolved is not None
    assert resolved.kind is PreferenceKind.FEATURE
    assert resolved.value == "  lightweight  "  # the caller's selector text is preserved
    # The resolved retraction is now buildable and actually retracts that belief.
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),)
            ):
                manager.apply_event(event)
            for event in build_turn_events(
                identity=_identity("turn-remove"),
                message="Actually, forget lightweight.",
                intent_label="recommend",
                retractions=(resolved,),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            current = manager.replay_state(USER_KEY, TASK_ID)
            assert current.active_preference_beliefs == ()
            (belief,) = current.preference_beliefs
            assert belief.lifecycle is PreferenceLifecycle.RETRACTED
            assert belief.status is PreferenceStatus.REMOVED
        finally:
            store.close()


def test_an_unknown_value_returns_none() -> None:
    """Zero matches is a legacy/M5 disagreement, reported as ``None`` -- never a guess."""
    state = _replayed_state(
        preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),)
    )
    assert resolve_retraction(state, ExplicitRetraction(value="waterproof")) is None


def test_a_unique_value_prefix_resolves_and_survives_sqlite_replay() -> None:
    """Withdrawing a remembered phrase by its leading noun phrase is durable."""
    state = _replayed_state(
        preferences=(
            ExplicitPreference(
                PreferenceKind.FEATURE,
                "replacement mouthpiece for a water filter",
            ),
        )
    )
    resolved = resolve_retraction(
        state, ExplicitRetraction(value="replacement mouthpiece")
    )
    assert resolved is not None
    assert resolved.kind is PreferenceKind.FEATURE

    with temporary_database() as database:
        store = SQLiteEventStore(database)
        manager = StateManager(store)
        try:
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                preferences=(
                    ExplicitPreference(
                        PreferenceKind.FEATURE,
                        "replacement mouthpiece for a water filter",
                    ),
                )
            ):
                manager.apply_event(event)
            for event in _events(
                identity=_identity("turn-withdraw"),
                retractions=(resolved,),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            replayed = manager.replay_state(USER_KEY, TASK_ID)
            assert replayed.active_preference_beliefs == ()
            (belief,) = replayed.preference_beliefs
            assert belief.lifecycle is PreferenceLifecycle.RETRACTED
        finally:
            store.close()

        # A new store on the same SQLite path must not resurrect the withdrawn
        # preference from a stale projection.
        store = SQLiteEventStore(database)
        try:
            replayed = StateManager(store).replay_state(USER_KEY, TASK_ID)
            assert replayed.active_preference_beliefs == ()
            (belief,) = replayed.preference_beliefs
            assert belief.status is PreferenceStatus.REMOVED
        finally:
            store.close()


def test_a_value_matching_two_kinds_is_ambiguous_and_returns_none() -> None:
    """"lightweight" as both a feature and a material cannot be resolved by value alone."""
    state = _replayed_state(
        preferences=(
            ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),
            ExplicitPreference(PreferenceKind.MATERIAL, "lightweight"),
        )
    )
    assert len(state.active_preference_beliefs) == 2
    assert resolve_retraction(state, ExplicitRetraction(value="lightweight")) is None


def test_the_polarity_filter_disambiguates_an_otherwise_ambiguous_value() -> None:
    """Polarity narrows the match set before the "exactly one kind" rule is applied."""
    state = _replayed_state(
        preferences=(
            ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),
            ExplicitPreference(
                PreferenceKind.MATERIAL, "lightweight", polarity=PreferencePolarity.AVOID
            ),
        )
    )
    assert resolve_retraction(state, ExplicitRetraction(value="lightweight")) is None
    resolved = resolve_retraction(
        state,
        ExplicitRetraction(value="lightweight", polarity=PreferencePolarity.PREFER),
    )
    assert resolved is not None
    assert resolved.kind is PreferenceKind.FEATURE
    # The avoid-side selector resolves to the other record.
    resolved_avoid = resolve_retraction(
        state,
        ExplicitRetraction(value="lightweight", polarity=PreferencePolarity.AVOID),
    )
    assert resolved_avoid is not None
    assert resolved_avoid.kind is PreferenceKind.MATERIAL


def test_retraction_resolution_ignores_inactive_beliefs() -> None:
    """A previously retracted value is not a match: state truth is the active set."""
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                preferences=(ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),)
            ):
                manager.apply_event(event)
            for event in build_turn_events(
                identity=_identity("turn-remove"),
                message="Forget lightweight.",
                intent_label="recommend",
                retractions=(ExplicitRetraction(kind=PreferenceKind.FEATURE),),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER_KEY, TASK_ID)
            assert state.active_preference_beliefs == ()
            assert resolve_retraction(state, ExplicitRetraction(value="lightweight")) is None
        finally:
            store.close()


def test_targeted_retraction_clears_constraint_and_survives_replay() -> None:
    """A requirement cannot survive as a hard constraint after its targeted retraction."""
    phrase = "replacement mouthpiece for a water filter"
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in _events(
                preferences=(
                    ExplicitPreference(PreferenceKind.FEATURE, phrase),
                    ExplicitPreference(PreferenceKind.FEATURE, "lightweight"),
                ),
                constraints=(ExplicitConstraint(PreferenceKind.FEATURE, phrase),),
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER_KEY, TASK_ID)
            resolved = resolve_retraction(state, ExplicitRetraction(value=phrase))
            assert resolved is not None
            for event in build_turn_events(
                identity=_identity("turn-targeted-retract"),
                message="Forget the replacement mouthpiece requirement.",
                intent_label="recommend",
                retractions=(resolved,),
                timestamp=TS_B,
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER_KEY, TASK_ID)
            assert [belief.value for belief in state.active_preference_beliefs] == ["lightweight"]
            assert state.authorized_hard_constraints == ()
        finally:
            store.close()
        # Reopen the same SQLite store and replay: the retraction is durable, not just
        # an in-memory projection mutation.
        reopened = SQLiteEventStore(database)
        try:
            state = StateManager(reopened).replay_state(USER_KEY, TASK_ID)
            assert [belief.value for belief in state.active_preference_beliefs] == ["lightweight"]
            assert state.authorized_hard_constraints == ()
        finally:
            reopened.close()


# =========================================================================== #
# D. resolve_turn_identity: precedence, scoping, determinism
# =========================================================================== #


def test_identity_precedence_is_idempotency_key_then_server_then_fallback() -> None:
    """The strongest retry-safe identity wins, and the source is reported."""
    key, source = resolve_turn_identity(
        task_id=TASK_ID,
        idempotency_key="idem-1",
        server_turn_id="server-1",
        fallback_content_key="content-1",
    )
    assert source == "idempotency_key"
    assert key == f"{TASK_ID}:idem:idem-1"

    key, source = resolve_turn_identity(
        task_id=TASK_ID, server_turn_id="server-1", fallback_content_key="content-1"
    )
    assert source == "server_turn_id"
    assert key == "server-1"

    key, source = resolve_turn_identity(task_id=TASK_ID, fallback_content_key="content-1")
    assert source == "content_fallback"
    assert key == "content-1"


def test_blank_identity_inputs_fall_through_and_all_blank_is_refused() -> None:
    """Blank values are not identities: they fall through to the next source."""
    key, source = resolve_turn_identity(
        task_id=TASK_ID, idempotency_key="   ", server_turn_id="server-1"
    )
    assert (key, source) == ("server-1", "server_turn_id")

    key, source = resolve_turn_identity(
        task_id=TASK_ID, idempotency_key="", server_turn_id="  ", fallback_content_key="content-1"
    )
    assert (key, source) == ("content-1", "content_fallback")

    with pytest.raises(ValueError):
        resolve_turn_identity(task_id=TASK_ID)
    with pytest.raises(ValueError):
        resolve_turn_identity(
            task_id=TASK_ID, idempotency_key=" ", server_turn_id="  ", fallback_content_key="\t"
        )
    with pytest.raises(ValueError):
        resolve_turn_identity(task_id="   ", idempotency_key="idem-1")


def test_identity_is_scoped_by_task_id() -> None:
    """The same client key in two tasks can never collide, and is stable within a task."""
    first = resolve_turn_identity(task_id="task-a", idempotency_key="retry-1")
    second = resolve_turn_identity(task_id="task-b", idempotency_key="retry-1")
    assert first[1] == second[1] == "idempotency_key"
    assert first[0] != second[0]
    # Same key, same task: the same identity across calls (that is the retry contract).
    assert resolve_turn_identity(task_id="task-a", idempotency_key="retry-1") == first
    # Whitespace-normalised keys are the same logical key.
    assert (
        resolve_turn_identity(task_id="task-a", idempotency_key="  retry-1  ") == first
    )


def test_distinct_server_turns_get_distinct_identities() -> None:
    """Two identical messages without a client key are two distinct server turns."""
    first = resolve_turn_identity(task_id=TASK_ID, server_turn_id="server-turn-1")
    second = resolve_turn_identity(task_id=TASK_ID, server_turn_id="server-turn-2")
    assert first[0] != second[0]
    assert first[1] == second[1] == "server_turn_id"


# =========================================================================== #
# E. Snapshot references: canonical, opaque, trusted
# =========================================================================== #


def test_candidate_pool_snapshot_is_order_insensitive_and_stable() -> None:
    """A candidate pool is a set: order must not change its reference or digest."""
    forward = candidate_pool_snapshot(("cand-red", "cand-blue", "cand-black"))
    backward = candidate_pool_snapshot(("cand-black", "cand-blue", "cand-red"))
    assert forward == backward
    assert forward.ref.startswith("cp_")
    assert len(forward.ref) == len("cp_") + 24
    assert forward.count == 3
    assert len(forward.digest) == 64
    assert forward.ref != candidate_pool_snapshot(("cand-red", "cand-blue")).ref
    # Duplicates collapse; empty strings are not identities.
    deduped = candidate_pool_snapshot(("cand-red", "cand-red", "", "cand-blue"))
    assert deduped == candidate_pool_snapshot(("cand-blue", "cand-red"))


def test_ranking_snapshot_is_order_sensitive() -> None:
    """A ranking snapshot captures exactly the presented order."""
    shown = ranking_snapshot(("cand-red", "cand-blue", "cand-black"))
    reordered = ranking_snapshot(("cand-blue", "cand-red", "cand-black"))
    assert shown.ref != reordered.ref
    assert shown.digest != reordered.digest
    assert shown.count == reordered.count == 3
    assert shown.ref.startswith("rs_")
    assert ranking_snapshot(("cand-red", "cand-blue", "cand-black")) == shown


def test_build_snapshot_events_emits_exactly_two_trusted_reference_events() -> None:
    """Two opaque events, actor ``trusted_component``, payload metadata only."""
    pool = candidate_pool_snapshot(("cand-red", "cand-blue"))
    ranking = ranking_snapshot(("cand-blue", "cand-red"))
    events = build_snapshot_events(
        identity=_identity(), candidate_pool=pool, ranking=ranking, timestamp=TS_A
    )
    assert len(events) == 2
    assert [event.event_type.value for event in events] == [
        "candidate_pool_referenced",
        "ranking_snapshot_referenced",
    ]
    pool_event, ranking_event = events
    for event in events:
        assert event.actor.value == "trusted_component"
        assert event.user_key == USER_KEY
        assert event.task_id == TASK_ID
        assert event.session_id == SESSION_ID
        assert event.timestamp == TS_A
        assert event.sequence is None

    assert pool_event.source == CANDIDATE_SNAPSHOT_SOURCE
    assert pool_event.payload.ref == pool.ref
    assert pool_event.payload.source == CANDIDATE_SNAPSHOT_SOURCE
    assert pool_event.payload.candidate_count == pool.count
    assert pool_event.payload.digest == pool.digest
    assert pool_event.payload.snapshot_schema_version == 1

    assert ranking_event.source == RANKING_SNAPSHOT_SOURCE
    assert ranking_event.payload.ref == ranking.ref
    assert ranking_event.payload.source == RANKING_SNAPSHOT_SOURCE
    assert ranking_event.payload.method == "presented_order"
    assert ranking_event.payload.candidate_count == ranking.count
    assert ranking_event.payload.digest == ranking.digest
    assert ranking_event.payload.snapshot_schema_version == 1


def test_build_snapshot_events_is_deterministic_across_rebuilds() -> None:
    """A retried turn re-emits the same reference event ids (timestamp is metadata)."""
    pool = candidate_pool_snapshot(("cand-red", "cand-blue"))
    ranking = ranking_snapshot(("cand-blue", "cand-red"))
    first = build_snapshot_events(
        identity=_identity(), candidate_pool=pool, ranking=ranking, timestamp=TS_A
    )
    second = build_snapshot_events(
        identity=_identity(), candidate_pool=pool, ranking=ranking, timestamp=TS_B
    )
    assert tuple(event.event_id for event in first) == tuple(
        event.event_id for event in second
    )
    for rebuilt, original in zip(second, first):
        assert rebuilt.is_same_submission(original)


def test_absent_snapshots_emit_no_event() -> None:
    """A snapshot that was not passed is skipped; nothing is fabricated or zeroed."""
    assert build_snapshot_events(identity=_identity(), timestamp=TS_A) == ()
    pool_only = build_snapshot_events(
        identity=_identity(),
        candidate_pool=candidate_pool_snapshot(("cand-red",)),
        timestamp=TS_A,
    )
    assert [event.event_type.value for event in pool_only] == ["candidate_pool_referenced"]
    ranking_only = build_snapshot_events(
        identity=_identity(),
        ranking=ranking_snapshot(("cand-red",)),
        timestamp=TS_A,
    )
    assert [event.event_type.value for event in ranking_only] == [
        "ranking_snapshot_referenced"
    ]


def test_snapshot_labels_and_schema_version_are_caller_controlled_metadata() -> None:
    """The trusted producer labels and the snapshot schema version travel unchanged."""
    events = build_snapshot_events(
        identity=_identity(),
        candidate_pool=candidate_pool_snapshot(("cand-red",)),
        ranking=ranking_snapshot(("cand-red",)),
        candidate_source="serving_v1_candidate_plane",
        ranking_source="serving_v1_presented",
        ranking_method="rrf_rank_fusion",
        snapshot_schema_version=3,
        timestamp=TS_A,
    )
    pool_event, ranking_event = events
    assert pool_event.payload.source == "serving_v1_candidate_plane"
    assert pool_event.payload.snapshot_schema_version == 3
    assert ranking_event.payload.source == "serving_v1_presented"
    assert ranking_event.payload.method == "rrf_rank_fusion"
    assert ranking_event.payload.snapshot_schema_version == 3


# =========================================================================== #
# F. End-to-end: snapshot events feed a real StateManager
# =========================================================================== #


def test_snapshot_events_replay_into_read_only_state_references() -> None:
    """The state stores opaque read-only refs whose version and digest match the events."""
    pool = candidate_pool_snapshot(("cand-red", "cand-blue", "cand-black"))
    ranking = ranking_snapshot(("cand-black", "cand-red", "cand-blue"))
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in build_snapshot_events(
                identity=_identity(),
                candidate_pool=pool,
                ranking=ranking,
                timestamp=TS_A,
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER_KEY, TASK_ID)
            applied_digest = state.state_digest()
        finally:
            store.close()

        # A fresh manager over the same file reconstructs the identical state.
        replayed_store = SQLiteEventStore(database)
        try:
            replayed = StateManager(replayed_store).replay_state(USER_KEY, TASK_ID)
            assert replayed.state_digest() == applied_digest

            pool_ref = replayed.candidate_pool_ref
            ranking_ref = replayed.ranking_snapshot_ref
            assert pool_ref is not None and ranking_ref is not None
            assert pool_ref.ref == pool.ref
            assert pool_ref.digest == pool.digest
            assert pool_ref.candidate_count == pool.count == 3
            assert pool_ref.source == CANDIDATE_SNAPSHOT_SOURCE
            assert pool_ref.snapshot_schema_version == 1
            assert pool_ref.read_only is True
            assert pool_ref.referenced_by_event_id is not None

            assert ranking_ref.ref == ranking.ref
            assert ranking_ref.digest == ranking.digest
            assert ranking_ref.candidate_count == ranking.count == 3
            assert ranking_ref.method == "presented_order"
            assert ranking_ref.snapshot_schema_version == 1
            assert ranking_ref.read_only is True
            assert ranking_ref.referenced_by_event_id is not None

            # The projections expose the opaque refs, not product rows or scores.
            agent = for_agent(replayed)
            assert agent.candidate_pool_ref == pool.ref
            assert agent.candidate_pool_candidate_count == 3
            assert agent.ranking_snapshot_ref == ranking.ref
        finally:
            replayed_store.close()


def test_state_references_hold_no_product_identity_or_score() -> None:
    """The reference models have no field for a product identity, a rank or a score."""
    identities = ("cand-red", "cand-blue", "cand-black")
    pool = candidate_pool_snapshot(identities)
    ranking = ranking_snapshot(identities)
    with temporary_database() as database:
        store = SQLiteEventStore(database)
        try:
            manager = StateManager(store)
            manager.load_or_create(USER_KEY, SESSION_ID, TASK_ID)
            for event in build_snapshot_events(
                identity=_identity(), candidate_pool=pool, ranking=ranking, timestamp=TS_A
            ):
                manager.apply_event(event)
            state = manager.replay_state(USER_KEY, TASK_ID)
            payload = json.dumps(state.model_dump(mode="json"), sort_keys=True)
        finally:
            store.close()

    assert set(state.candidate_pool_ref.model_dump()) == {
        "ref",
        "source",
        "candidate_count",
        "digest",
        "snapshot_schema_version",
        "referenced_by_event_id",
        "read_only",
    }
    assert set(state.ranking_snapshot_ref.model_dump()) == {
        "ref",
        "source",
        "method",
        "candidate_count",
        "digest",
        "snapshot_schema_version",
        "referenced_by_event_id",
        "read_only",
    }
    # The trusted product identities are not stored in the decision state: only the
    # canonical digest crosses the boundary.
    for identity in identities:
        assert identity not in payload
    assert pool.digest in payload and ranking.digest in payload
    for forbidden in ("score", "rank", "parent_asin", "item_id", "products"):
        assert forbidden not in state.candidate_pool_ref.model_dump()
        assert forbidden not in state.ranking_snapshot_ref.model_dump()
