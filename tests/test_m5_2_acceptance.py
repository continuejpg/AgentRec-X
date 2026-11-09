"""M5.2 independent black-box acceptance suite (requirements 1-20 + follow-ups).

The suite drives the demo service over HTTP with the real offline harness
(:func:`tests.demo_fixture.build_harness`), flips ``AGENTRECX_STATE_MODE`` through the
environment, and inspects the decision-state database, the accepted M9 memory database and
the documented diagnostics surface **independently** of the integration internals: a
passing run is evidence about the service contract, not about a helper function.

Contract under test: ``docs/M5_2_INTERACTION_FEEDBACK_LOOP.md``.  Requirement map
------------------------------------------------------------------------------
1. :func:`test_initial_shopping_turn_emits_valid_events` — a normal planning turn appends
   valid, TASK-scoped, attributed M5 events and a replayable state.
2. :func:`test_clarification_option_answer_updates_m5_preference_state` — a real
   preference question (``preference_discovery=True`` + completed recommendation) is
   answered with a structured ``option_id``; the trusted option becomes one explicit
   TASK-scoped belief with the option's direction, and diagnostics record
   ``feedback_kind=clarification_option``.
3. :func:`test_declined_clarification_answer_creates_no_preference_fact` — a ``pref:none``
   / ``pref:skip`` answer creates no preference event and no belief.
4. :func:`test_explicit_replace_correction_supersedes_and_keeps_provenance` — an M9
   ``REPLACE`` correction supersedes the old belief while both records keep provenance.
5. :func:`test_m9_remove_retracts_belief_without_projection` — an accepted M9 ``REMOVE``
   produces ``PREFERENCE_RETRACTED``; the belief is RETRACTED/REMOVED and no projection
   contains it.
6. :func:`test_task_feedback_never_writes_long_term_or_changes_m9_schema` — no LONG_TERM
   belief/event is created and the accepted M9 database schema is unchanged.
7. :func:`test_retry_with_same_idempotency_key_is_idempotent` — same ``idempotency_key``
   re-emits the same ids/count/digest and adds no duplicate row.
8. :func:`test_identical_turns_without_key_get_distinct_server_identities` — two identical
   messages without a key are two distinct server turns with distinct events.
9. :func:`test_cross_user_isolation` — two users never share state/events.
10. :func:`test_cross_task_isolation` — events and diagnostics are scoped by task.
11. :func:`test_replay_after_reopening_sqlite_equals_diagnostics_digest` — a fresh
    store/manager over the same file reproduces the diagnostics digest.
12. :func:`test_off_mode_is_unchanged_and_writes_no_state` — OFF == explicit off, no state
    database, no diagnostics.
13. :func:`test_shadow_mode_is_user_visible_parity_while_recording_state` — SHADOW matches
    OFF after normalising volatile fields while state/diagnostics are produced.
14. :func:`test_apply_mode_uses_authorized_constraints_with_consistent_intent_view` —
    APPLY enforces the effective M5 constraint through the existing evaluator and
    ``response.intent.hard`` reports it with ``origin="decision_state"`` (current-turn
    ``user_stated``, remembered ``validated_existing``).
15. :func:`test_unknown_stays_unknown_not_ineligible` — a structurally unverifiable
    constraint leaves candidates UNKNOWN; ineligible_count is 0 and the three-state sum
    equals the assessments.
16. :func:`test_inferred_hypothesis_never_hard_filters` — a seeded inference cannot filter.
17. :func:`test_candidate_and_ranking_snapshots_have_trusted_provenance` — two trusted
    snapshot events with opaque refs/counts/digests/schema version and no product identity
    or score; the state refs reflect them.
18. :func:`test_invalid_event_is_not_appended` — SHADOW 200 + diagnostics
    ``state_event_invalid`` with no new row; APPLY 502 fail-closed.
19. :func:`test_store_failure_follows_mode_policy` — unopenable state DB: SHADOW 200 +
    diagnostics error, APPLY 502.
20. The existing M5/M5.1 suites are run in the final report; this file does not weaken any
    of their assertions.

Additional required checks
--------------------------
* :func:`test_pending_conflict_turn_still_records_validated_facts` — a suspended conflict
  turn records its validated constraint facts.
* :func:`test_unmatched_removal_reports_disagreement_without_inventing_events` — an
  unmatched value-only removal invents no retraction event and reports a bounded
  diagnostics disagreement.

Normalisation for parity comparisons
------------------------------------
Two independently created demo sessions cannot have byte-identical responses: the
accepted response contract carries controller-generated opaque identities (``run_id`` /
``run_ref``), session-derived provenance ids (``source_turn_id`` / ``turn_ids``) and
wall-clock ``timing``.  Those are identities and clocks, not behaviour, so
:func:`normalised_response` drops ``timing`` and masks exactly those values plus
session-derived turn ids.  Every other field must match exactly.
"""

from __future__ import annotations

import contextlib
import copy
import json
import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.memory import (  # noqa: E402
    PreferenceExtraction,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
)
from recommendation.state import (  # noqa: E402
    BeliefOrigin,
    EVENT_ACTORS,
    PreferenceDirection,
    PreferenceLifecycle,
    PreferenceScope,
    PreferenceStatus,
    SQLiteEventStore,
    StateManager,
    for_agent,
    for_constraints,
    for_ranker,
)
from tests.agent_reranking_fixture import INITIAL_ORDER  # noqa: E402
from tests.demo_fixture import (  # noqa: E402
    PROFILE_A,
    PROFILE_B,
    build_harness,
    close_harness,
)
from tests.state_fixtures import constraint_event, hypothesis_event  # noqa: E402

# --------------------------------------------------------------------------- #
# fixed contract surface
# --------------------------------------------------------------------------- #

#: Environment switches under test.
STATE_MODE_ENV = "AGENTRECX_STATE_MODE"
STATE_DB_ENV = "AGENTRECX_STATE_DB"

#: A plain recommendation request that states nothing by itself.
NEUTRAL_MESSAGE = "Recommend some hiking gear."
#: A request whose trusted prefix yields exactly one explicit colour preference.
PREFERENCE_MESSAGE = "Recommend some hiking gear. I prefer black."
#: The request used for the OFF/SHADOW parity comparison: one explicit preference and one
#: decidable body constraint, so the answer is complete and the state records both facts.
PARITY_MESSAGE = "Recommend some hiking gear. I prefer black."
PARITY_CONSTRAINTS = ({"kind": "price_max", "value": "120"},)

FIXED_K = 4
FIXED_SOURCES = ("history", "catalog_search")

#: A trusted hard constraint exercised in the APPLY / idempotency paths: only
#: ``cand-red`` ($25.00) and ``cand-black`` ($15.00) satisfy price_max=30.
BUDGET_CONSTRAINT = ({"kind": "price_max", "value": "30"},)

#: Event vocabulary M5.2 may emit on the shopping route (M5.1 superset + retraction +
#: trusted snapshot references).  Anything else is a trust-boundary breach.
SUPPORTED_EVENT_TYPES = frozenset(
    {
        "task_context_updated",
        "user_preference_stated",
        "user_constraint_stated",
        "preference_retracted",
        "candidate_pool_referenced",
        "ranking_snapshot_referenced",
    }
)

#: Server-owned turn ids look like ``demo-session:<uuid>:<hex/digits>``.
_SESSION_TURN_ID = re.compile(r"^demo-session:[0-9a-f-]{36}:[0-9a-f0-9]{1,32}$")

#: Keys holding controller-generated / session-derived identities (per-run by design).
_MASKED_KEYS = frozenset(
    {"timing", "run_id", "run_ref", "source_turn_id", "turn_ids"}
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Diagnostics keys the M5.2 contract promises: M5.1 keys plus the M5.2 additions.
DIAGNOSTIC_KEYS = frozenset(
    {
        # M5.1 (must remain)
        "mode",
        "task_id",
        "turn_key_digest",
        "events",
        "state_version",
        "state_digest",
        "event_count",
        "user_turn_count",
        "preference_count",
        "soft_hypothesis_count",
        "hard_constraint_count",
        "constraint_kinds",
        "suppressed_long_term_count",
        "agent_projection_version",
        "ranker_signal_version",
        "ranker_feature_count",
        "ranker_priority_count",
        "constraints_version",
        "excluded_hypothesis_count",
        "trusted_constraint_count",
        "constraint_match",
        "error",
        "turn_count",
        # M5.2 additions
        "turn_identity_source",
        "feedback_kind",
        "retraction_match",
        "legacy_m5",
        "snapshot",
    }
)

#: Accepted values of the documented turn-identity sources.
TURN_IDENTITY_SOURCES = frozenset({"idempotency_key", "server_turn_id", "content_fallback"})


# --------------------------------------------------------------------------- #
# harness / request helpers
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _harness(tmp_path: Path, monkeypatch, mode: str | None, label: str):
    """Build one demo harness with the given state mode; yield ``(harness, state_db)``.

    ``mode=None`` means the environment variable is unset (the default-OFF path).
    """
    root = tmp_path / label
    root.mkdir(parents=True, exist_ok=True)
    state_db = root / "decision_state.sqlite3"
    if mode is None:
        monkeypatch.delenv(STATE_MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(STATE_MODE_ENV, mode)
    monkeypatch.setenv(STATE_DB_ENV, str(state_db))
    harness = build_harness(root)
    try:
        yield harness, state_db
    finally:
        close_harness(harness)


def _create_session(harness: Any, profile: str = PROFILE_A) -> str:
    response = harness.client.post("/v1/demo/sessions", json={"profile_id": profile})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _session(harness: Any, session_id: str) -> Any:
    """The live demo session object (server-owned identities)."""
    return harness.runtime.sessions.get(session_id)


def _identity(harness: Any, session_id: str) -> tuple[str, str]:
    """The server-owned ``(user_key, task_id)`` pair of one demo session."""
    session = _session(harness, session_id)
    return session.user_key, session.session_id


def _service(harness: Any) -> Any:
    """The composed agent service (the M5.2 diagnostics surface)."""
    return harness.runtime.agent_service()


def _shopping_turn(
    harness: Any,
    session_id: str,
    *,
    message: str = NEUTRAL_MESSAGE,
    k: int = FIXED_K,
    sources=FIXED_SOURCES,
    constraints: Any = (),
    **overrides: Any,
) -> Any:
    """POST one shopping turn and return the raw response."""
    body: dict[str, Any] = {
        "message": message,
        "k": k,
        "sources": list(sources),
        "constraints": [dict(item) for item in constraints],
    }
    body.update(overrides)
    return harness.client.post(
        f"/v1/demo/sessions/{session_id}/agent/shopping", json=body
    )


def _answer_clarification(
    harness: Any,
    session_id: str,
    clarification: dict[str, Any],
    *,
    option: dict[str, Any],
    preference_discovery: bool = True,
    k: int = FIXED_K,
    idempotency_key: str | None = None,
) -> Any:
    """Answer a pending preference question with one structured option id."""
    body: dict[str, Any] = {
        "message": str(option.get("label") or ""),
        "k": k,
        "sources": list(FIXED_SOURCES),
        "constraints": [],
        "clarification_id": clarification["clarification_id"],
        "clarification_option": option["option_id"],
        "preference_discovery": preference_discovery,
    }
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return harness.client.post(
        f"/v1/demo/sessions/{session_id}/agent/shopping", json=body
    )


# --------------------------------------------------------------------------- #
# response normalisation and comparison
# --------------------------------------------------------------------------- #


def _normalise(value: Any) -> Any:
    """Recursively mask controller-generated identities and wall-clock keys."""
    if isinstance(value, dict):
        return {
            key: "<masked>" if key in _MASKED_KEYS else _normalise(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, str) and _SESSION_TURN_ID.match(value):
        return "<turn-id>"
    return value


def normalised_response(body: dict[str, Any]) -> dict[str, Any]:
    """Return the response with only non-behavioural fields normalised."""
    data = copy.deepcopy(body)
    data.pop("timing", None)
    return _normalise(data)


def _diff(left: Any, right: Any, path: str = "$") -> list[tuple[str, Any, Any]]:
    """Return the differing paths between two JSON-like values."""
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        differences: list[tuple[str, Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((f"{path}.{key}", left.get(key), right.get(key)))
            else:
                differences.extend(_diff(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list):
        differences = []
        if len(left) != len(right):
            differences.append((f"{path}.length", len(left), len(right)))
        for index, (first, second) in enumerate(zip(left, right)):
            differences.extend(_diff(first, second, f"{path}[{index}]"))
        return differences
    if left != right:
        return [(path, left, right)]
    return []


def assert_response_parity(expected: dict[str, Any], actual: dict[str, Any], *, context: str) -> None:
    """Assert two user-visible responses are identical apart from identities and timing."""
    first = normalised_response(expected)
    second = normalised_response(actual)
    if first != second:
        differences = _diff(first, second)
        pytest.fail(
            f"{context}: responses differ at {len(differences)} path(s): "
            f"{differences[:10]}"
        )


# --------------------------------------------------------------------------- #
# decision-state database helpers (independent read path)
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _open_state(db_path: Path):
    """Open the decision-state database with the frozen M5 event store."""
    store = SQLiteEventStore(db_path)
    try:
        yield store, StateManager(store)
    finally:
        store.close()


def _seed_state(db_path: Path, user_key: str, task_id: str, events: Any) -> None:
    """Append fixture events for one task through a real ``StateManager``."""
    with _open_state(db_path) as (_store, manager):
        manager.load_or_create(user_key, session_id=task_id, task_id=task_id)
        for event in events:
            manager.apply_event(event)


def _read_state(db_path: Path, user_key: str, task_id: str) -> dict[str, Any]:
    """Replay one task independently and return events/state/projections."""
    with _open_state(db_path) as (store, manager):
        events = store.get_events(task_id, user_key)
        state = manager.replay_state(user_key, task_id)
        return {
            "events": events,
            "state": state,
            "agent": for_agent(state),
            "ranker": for_ranker(state),
            "constraints": for_constraints(state),
            "digest": state.state_digest(),
            "count": store.count(user_key),
        }


def _stored_rows(db_path: Path, user_key: str, task_id: str) -> int:
    """Number of durable event rows for one user+task (direct SQL)."""
    if not db_path.exists():
        return 0
    connection = sqlite3.connect(str(db_path))
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM events WHERE user_key = ? AND task_id = ?",
            (user_key, task_id),
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def _event_type_counts(events: Any) -> dict[str, int]:
    """Count stored events by type."""
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type.value] = counts.get(event.event_type.value, 0) + 1
    return counts


def _snapshot_events(events: Any) -> list[Any]:
    return [
        event
        for event in events
        if event.event_type.value
        in {"candidate_pool_referenced", "ranking_snapshot_referenced"}
    ]


def _assert_event_batch(events: Any, *, user_key: str, task_id: str) -> None:
    """Every stored event must be a valid, attributed, TASK-scoped M5 event."""
    previous_sequence = 0
    seen_ids: set[str] = set()
    for event in events:
        assert event.user_key == user_key
        assert event.task_id == task_id
        assert event.schema_version == 1
        assert event.event_type.value in SUPPORTED_EVENT_TYPES, event.event_type.value
        assert event.actor in EVENT_ACTORS[event.event_type]
        assert event.scope is None or event.scope is PreferenceScope.TASK
        assert event.applies_long_term is False
        assert event.event_id not in seen_ids, "event ids must be unique within a task"
        seen_ids.add(event.event_id)
        assert event.sequence is not None
        assert event.sequence > previous_sequence, "store order must be strictly increasing"
        previous_sequence = event.sequence


def _assert_diagnostics(diagnostics: dict[str, Any], *, mode: str) -> None:
    """Assert the documented M5.2 diagnostics shape and its non-sensitivity bounds."""
    assert isinstance(diagnostics, dict), "diagnostics must be a mapping"
    missing = DIAGNOSTIC_KEYS - set(diagnostics)
    assert not missing, f"diagnostics missing documented keys: {sorted(missing)}"
    assert diagnostics["mode"] == mode
    assert isinstance(diagnostics["constraint_match"], dict)
    assert set(diagnostics["constraint_match"]) == {
        "matched",
        "missing_from_state",
        "extra_in_state",
    }
    assert diagnostics["turn_identity_source"] in TURN_IDENTITY_SOURCES
    assert diagnostics["feedback_kind"] is None or isinstance(
        diagnostics["feedback_kind"], str
    )
    assert diagnostics["retraction_match"] is None or isinstance(
        diagnostics["retraction_match"], dict
    )
    assert diagnostics["legacy_m5"] is None or isinstance(diagnostics["legacy_m5"], dict)
    assert diagnostics["snapshot"] is None or isinstance(diagnostics["snapshot"], dict)
    # Bounded and non-sensitive: no raw message, no user key, no unbounded blob.
    serialised = json.dumps(diagnostics, sort_keys=True)
    assert "demo-session:" not in serialised
    assert len(serialised) < 8192, "diagnostics record is not bounded"
    if diagnostics["state_digest"] is not None:
        assert _HEX64.match(diagnostics["state_digest"]), "state digest is a SHA-256 hex value"
    assert re.fullmatch(r"[0-9a-f]{12}", diagnostics["turn_key_digest"])


def _retraction_numbers(retraction_match: Any) -> dict[str, int]:
    """Flatten the bounded retraction diagnostics into ``{key: int}``.

    The exact sub-key names are an implementation detail of the bounded record; the
    *semantics* the contract pins are the counts, so the tests read them generically.
    """
    numbers: dict[str, int] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, bool):
                    continue
                if isinstance(item, int):
                    numbers[str(key)] = int(item)
                else:
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(retraction_match)
    return numbers


def _memory_db(root: Path) -> Path:
    """Path of the accepted M9 preference-memory database created by ``build_harness``."""
    return root / "demo_memory.sqlite3"


def _sqlite_schema(db_path: Path) -> dict[str, str]:
    """Return ``sqlite_master`` name -> SQL for one database file (schema only)."""
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        connection.close()
    return {str(name): str(sql or "") for name, sql in rows}


def _sqlite_row_count(db_path: Path, table: str) -> int:
    """Row count of one table in an existing SQLite database."""
    connection = sqlite3.connect(str(db_path))
    try:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])
    finally:
        connection.close()


def _preference_events(events: Any) -> list[Any]:
    return [
        event
        for event in events
        if event.event_type.value in {"user_preference_stated", "preference_retracted"}
    ]


def _active_identities(state: Any) -> set[tuple[str, str, str]]:
    return {
        (belief.kind.value, belief.polarity.value, belief.value.casefold())
        for belief in state.active_preference_beliefs
        if belief.scope is PreferenceScope.TASK
    }


# =========================================================================== #
# 1. initial shopping turn: valid events + replayable state
# =========================================================================== #


def test_initial_shopping_turn_emits_valid_events(tmp_path: Path, monkeypatch) -> None:
    """Requirement 1: a planning turn appends valid, TASK-scoped, attributed events."""
    with _harness(tmp_path, monkeypatch, "shadow", "initial-turn") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        response = _shopping_turn(
            harness, session_id, message=PREFERENCE_MESSAGE, constraints=PARITY_CONSTRAINTS
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["recommendations"], "the turn must present recommendations"

        recorded = _read_state(state_db, user_key, task_id)
        events = recorded["events"]
        assert events, "a SHADOW planning turn must record M5 events"
        _assert_event_batch(events, user_key=user_key, task_id=task_id)

        context_events = [
            event for event in events if event.event_type.value == "task_context_updated"
        ]
        assert len(context_events) == 1
        assert context_events[0].payload.intent in {"unknown", "recommend"}
        assert context_events[0].payload.request_text == PREFERENCE_MESSAGE
        assert context_events[0].actor.value == "agent"

        state = recorded["state"]
        # The trusted facts are visible in the state and its projections.
        assert {(b.kind.value, b.value) for b in state.active_preference_beliefs} == {
            ("color", "black")
        }
        assert [
            (constraint.kind.value, constraint.value)
            for constraint in state.authorized_hard_constraints
        ] == [("price_max", "120")]
        assert state.task_context.request_text == PREFERENCE_MESSAGE

        agent_view = recorded["agent"]
        assert agent_view.projection_version == 1
        assert [(item.kind.value, item.value) for item in agent_view.preferences] == [
            ("color", "black")
        ]
        assert [(item.kind.value, item.value) for item in agent_view.constraints] == [
            ("price_max", "120")
        ]
        ranker_view = recorded["ranker"]
        assert ranker_view.ranker_signal_version == 1
        assert [(feature.kind.value, feature.value) for feature in ranker_view.features] == [
            ("color", "black")
        ]
        constraint_view = recorded["constraints"]
        assert constraint_view.constraints_version == 1
        assert [
            (view.kind.value, view.value) for view in constraint_view.constraints
        ] == [("price_max", "120")]

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["state_digest"] == recorded["digest"]
        assert diagnostics["event_count"] == len(events) == _stored_rows(
            state_db, user_key, task_id
        )
        assert diagnostics["turn_identity_source"] == "server_turn_id"
        assert diagnostics["turn_count"] == 1


# =========================================================================== #
# 2-3. clarification answers: structured option -> fact; none/skip -> no fact
# =========================================================================== #


def _pending_preference_question(harness: Any, session_id: str) -> dict[str, Any]:
    """Run the turn that asks a real candidate-aware preference question."""
    response = _shopping_turn(
        harness,
        session_id,
        message=NEUTRAL_MESSAGE,
        constraints=(),
        preference_discovery=True,
    )
    assert response.status_code == 200, response.text
    clarification = response.json().get("clarification")
    assert clarification is not None, "preference_discovery should ask a question"
    assert clarification["kind"] == "preference"
    evidence = clarification.get("preference")
    assert evidence is not None
    assert evidence["options"], "the trusted prompt must expose executable options"
    return clarification


def test_clarification_option_answer_updates_m5_preference_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 2: a structured option answer becomes one explicit TASK belief."""
    with _harness(tmp_path, monkeypatch, "shadow", "clarification-answer") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        clarification = _pending_preference_question(harness, session_id)

        # Before the answer: a question must not fabricate a preference fact.
        pending_state = _read_state(state_db, user_key, task_id)["state"]
        assert pending_state.active_preference_beliefs == ()

        options = list(clarification["preference"]["options"])
        numeric = next(
            option for option in options if option["action"] in {"lower", "higher"}
        )
        attribute = str(numeric["attribute"])
        expected_direction = (
            PreferenceDirection.MINIMIZE
            if numeric["action"] == "lower"
            else PreferenceDirection.MAXIMIZE
        )

        answer = _answer_clarification(
            harness, session_id, clarification, option=numeric
        )
        assert answer.status_code == 200, answer.text
        body = answer.json()
        outcome = body.get("preference_outcome")
        assert outcome is not None
        assert outcome["selected_option_id"] == numeric["option_id"]

        recorded = _read_state(state_db, user_key, task_id)
        state = recorded["state"]
        (belief,) = state.active_preference_beliefs
        assert belief.kind is PreferenceKind.FEATURE
        assert belief.value.casefold() == attribute.casefold()
        assert belief.direction is expected_direction
        assert belief.scope is PreferenceScope.TASK
        assert belief.lifecycle is PreferenceLifecycle.EXPLICIT
        assert belief.origin is BeliefOrigin.USER_STATED
        assert belief.status is PreferenceStatus.ACTIVE

        # One trusted feedback event, user-authored, TASK-scoped, with the option label
        # as provenance.
        feedback_events = [
            event
            for event in recorded["events"]
            if event.event_type.value == "user_preference_stated"
        ]
        assert len(feedback_events) == 1
        feedback = feedback_events[0]
        assert feedback.actor.value == "user"
        assert feedback.scope is PreferenceScope.TASK
        assert feedback.payload.kind is PreferenceKind.FEATURE
        assert feedback.payload.value.casefold() == attribute.casefold()
        assert feedback.payload.direction is expected_direction
        assert feedback.payload.source_text == str(numeric["label"])

        # The projections used to build the effective state see the fact.
        assert [
            (feature.kind.value, feature.value, feature.direction)
            for feature in recorded["ranker"].features
        ] == [("feature", attribute, expected_direction)]
        assert [
            (item.kind.value, item.value)
            for item in recorded["agent"].preferences
        ] == [("feature", attribute)]

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["feedback_kind"] == "clarification_option"
        assert diagnostics["state_digest"] == recorded["digest"]


def test_declined_clarification_answer_creates_no_preference_fact(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 3: ``pref:none`` / ``pref:skip`` creates no preference event or belief."""
    with _harness(tmp_path, monkeypatch, "shadow", "clarification-declined") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        clarification = _pending_preference_question(harness, session_id)
        options = list(clarification["preference"]["options"])
        declined = next(
            option for option in options if option["action"] in {"none", "skip"}
        )

        before = _read_state(state_db, user_key, task_id)
        before_preference_ids = {
            event.event_id for event in _preference_events(before["events"])
        }

        answer = _answer_clarification(harness, session_id, clarification, option=declined)
        assert answer.status_code == 200, answer.text
        body = answer.json()
        # The accepted answer flow still returns the base recommendation list.
        assert body.get("terminal", {}).get("status") == "finished"

        after = _read_state(state_db, user_key, task_id)
        after_preference_ids = {
            event.event_id for event in _preference_events(after["events"])
        }
        assert after_preference_ids == before_preference_ids, (
            "a declined clarification must not record a preference event"
        )
        assert after["state"].active_preference_beliefs == ()
        assert not [
            event
            for event in _preference_events(after["events"])
            if event.event_id not in {item.event_id for item in before["events"]}
        ], "a declined answer must not append any preference/retraction event"
        assert after["agent"].preferences == ()
        assert after["ranker"].features == ()

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        # The turn is a recorded feedback attempt (the label is not pinned by the contract
        # for a declined answer), but it created no preference fact -- asserted above.
        assert diagnostics["feedback_kind"] is not None
        assert diagnostics["preference_count"] == 0


# =========================================================================== #
# 4. explicit REPLACE correction
# =========================================================================== #


def test_explicit_replace_correction_supersedes_and_keeps_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 4: a REPLACE correction supersedes the corrected belief, provenance kept.

    Covers both accepted M9 correction shapes: a named target ("blue instead of black",
    mapped to ``replaces_value``) and a general correction ("... instead.", mapped to
    ``replace_same_kind``), then proves that a later plain ADD is not treated as one.
    """
    with _harness(tmp_path, monkeypatch, "shadow", "correction") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)

        first = _shopping_turn(harness, session_id, message="I prefer black.", constraints=())
        assert first.status_code == 200, first.text
        before = _read_state(state_db, user_key, task_id)
        (old_black,) = before["state"].active_preference_beliefs
        assert (old_black.kind.value, old_black.value) == ("color", "black")

        # The accepted M9 rule-based extractor resolves this as REPLACE mode with the
        # explicitly corrected value ("black"), not as a second independent preference.
        correction = _shopping_turn(
            harness,
            session_id,
            message="Actually I prefer blue instead of black.",
            constraints=(),
        )
        assert correction.status_code == 200, correction.text

        after = _read_state(state_db, user_key, task_id)
        state = after["state"]
        active = [
            (belief.kind.value, belief.value)
            for belief in state.active_preference_beliefs
        ]
        assert active == [("color", "blue")], "the corrected value must be the only active one"
        # The old record is kept for audit: not active, but its provenance survives.
        old_records = [
            belief
            for belief in state.preference_beliefs
            if belief.value.casefold() == "black"
        ]
        assert len(old_records) == 1
        old = old_records[0]
        assert old.is_active is False
        assert old.status is PreferenceStatus.SUPERSEDED
        assert old.lifecycle is PreferenceLifecycle.EXPLICIT
        assert old_black.belief_id == old.belief_id, "the corrected record is updated in place"
        assert old.supporting_event_ids and old.supporting_event_ids[0] == (
            before["state"].preference_beliefs[0].supporting_event_ids[0]
        )
        assert old.supporting_event_ids == (
            before["state"].preference_beliefs[0].supporting_event_ids
        ) + (old.last_event_id,)
        assert old.last_event_id is not None and old.last_event_id != (
            before["state"].preference_beliefs[0].last_event_id
        )
        new_blue = next(
            belief
            for belief in state.active_preference_beliefs
            if belief.value.casefold() == "blue"
        )
        assert new_blue.supporting_event_ids == (new_blue.last_event_id,)
        assert new_blue.last_event_id == old.last_event_id

        # The projection that builds the effective state no longer contains the old value.
        agent_values = {item.value for item in after["agent"].preferences}
        assert "blue" in agent_values and "black" not in agent_values
        assert all(feature.value.casefold() != "black" for feature in after["ranker"].features)

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["feedback_kind"] == "correction"
        assert diagnostics["state_digest"] == after["digest"]

        # A second REPLACE that does not name the corrected value ("... instead.") must
        # still supersede the same-kind/same-polarity value instead of adding a third one.
        second_correction = _shopping_turn(
            harness,
            session_id,
            message="Actually I prefer red instead.",
            constraints=(),
        )
        assert second_correction.status_code == 200, second_correction.text
        after_general = _read_state(state_db, user_key, task_id)
        general_state = after_general["state"]
        assert [
            (belief.kind.value, belief.value)
            for belief in general_state.active_preference_beliefs
        ] == [("color", "red")]
        blue_records = [
            belief
            for belief in general_state.preference_beliefs
            if belief.value.casefold() == "blue"
        ]
        assert len(blue_records) == 1
        assert blue_records[0].is_active is False
        assert blue_records[0].status is PreferenceStatus.SUPERSEDED
        assert len(blue_records[0].supporting_event_ids) == 2
        # The earlier superseded record is untouched by the later correction.
        black_records = [
            belief
            for belief in general_state.preference_beliefs
            if belief.value.casefold() == "black"
        ]
        assert len(black_records) == 1 and black_records[0].is_active is False
        general_diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(general_diagnostics, mode="shadow")
        assert general_diagnostics["feedback_kind"] == "correction"
        assert general_diagnostics["error"] is None

        # A later plain ADD is not a correction: it adds an independent value and does not
        # supersede the active one.
        addition = _shopping_turn(
            harness, session_id, message="I prefer green.", constraints=()
        )
        assert addition.status_code == 200, addition.text
        after_addition = _read_state(state_db, user_key, task_id)
        assert {
            belief.value for belief in after_addition["state"].active_preference_beliefs
        } == {"red", "green"}
        (red_record,) = [
            belief
            for belief in after_addition["state"].active_preference_beliefs
            if belief.value.casefold() == "red"
        ]
        assert red_record.status is PreferenceStatus.ACTIVE
        # The corrected (superseded) records are still retained for audit.
        assert {
            belief.value
            for belief in after_addition["state"].preference_beliefs
            if belief.status is PreferenceStatus.SUPERSEDED
        } == {"black", "blue"}


# =========================================================================== #
# 5. M9 REMOVE -> PREFERENCE_RETRACTED
# =========================================================================== #


def test_m9_remove_retracts_belief_without_projection(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 5: an accepted M9 REMOVE becomes a TASK retraction, not a deletion."""
    with _harness(tmp_path, monkeypatch, "shadow", "removal") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)

        first = _shopping_turn(
            harness, session_id, message="I prefer lightweight gear.", constraints=()
        )
        assert first.status_code == 200, first.text
        before = _read_state(state_db, user_key, task_id)
        (old_belief,) = before["state"].active_preference_beliefs
        assert old_belief.value.casefold() == "lightweight"

        # An accepted M9 REMOVE phrasing: the value-only selector must be resolved against
        # the current task state before it can become an M5 retraction event.
        removal = _shopping_turn(
            harness,
            session_id,
            message="I don't care about the lightweight preference anymore.",
            constraints=(),
        )
        assert removal.status_code == 200, removal.text

        after = _read_state(state_db, user_key, task_id)
        state = after["state"]
        assert state.active_preference_beliefs == ()
        (record,) = [
            belief
            for belief in state.preference_beliefs
            if belief.value.casefold() == "lightweight"
        ]
        assert record.lifecycle is PreferenceLifecycle.RETRACTED
        assert record.status is PreferenceStatus.REMOVED
        assert record.is_active is False
        assert old_belief.belief_id == record.belief_id, "retraction keeps the record identity"
        assert record.last_event_id is not None

        retractions = [
            event
            for event in after["events"]
            if event.event_type.value == "preference_retracted"
        ]
        assert len(retractions) == 1
        retraction = retractions[0]
        assert retraction.actor.value == "user"
        assert retraction.scope is PreferenceScope.TASK
        assert retraction.applies_long_term is False
        assert retraction.payload.kind == old_belief.kind
        assert retraction.payload.value is not None
        assert retraction.payload.value.casefold() == "lightweight"

        # Nothing projected into the effective state still contains the retracted value.
        assert all(
            item.value.casefold() != "lightweight" for item in after["agent"].preferences
        )
        assert all(
            feature.value.casefold() != "lightweight" for feature in after["ranker"].features
        )
        assert all(
            feature.value.casefold() != "lightweight" for feature in after["ranker"].features
        )
        assert after["state"].state_digest() == after["digest"]

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        counts = _retraction_numbers(diagnostics["retraction_match"])
        assert counts, "a resolved removal must be reported in the bounded diagnostics"
        assert 1 in counts.values() or 2 in counts.values(), (
            f"retraction diagnostics should report the resolved selector: {counts}"
        )


# =========================================================================== #
# 6. task feedback never corrupts long-term data
# =========================================================================== #


def test_task_feedback_never_writes_long_term_or_changes_m9_schema(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 6: TASK facts/retractions never create LONG_TERM state or DDL changes."""
    with _harness(tmp_path, monkeypatch, "shadow", "long-term") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        memory_db = _memory_db(state_db.parent)
        schema_before = _sqlite_schema(memory_db)

        first = _shopping_turn(
            harness, session_id, message="I prefer lightweight gear.", constraints=()
        )
        assert first.status_code == 200, first.text
        second = _shopping_turn(
            harness,
            session_id,
            message="I don't care about the lightweight preference anymore.",
            constraints=(),
        )
        assert second.status_code == 200, second.text

        recorded = _read_state(state_db, user_key, task_id)
        events = recorded["events"]
        assert events
        for event in events:
            assert event.applies_long_term is False, "M5.2 must never write LONG_TERM state"
            assert event.scope is None or event.scope is PreferenceScope.TASK
        assert all(
            belief.scope is PreferenceScope.TASK
            for belief in recorded["state"].preference_beliefs
        ), "a task feedback loop must not create long-term beliefs"

        # The accepted M9 database is untouched in structure and separate from the state DB.
        memory_db = _memory_db(state_db.parent)
        assert memory_db.exists()
        assert memory_db != state_db, "decision state and M9 memory must be separate files"
        assert _sqlite_schema(memory_db) == schema_before, (
            "M5.2 must not change the accepted M9 memory schema"
        )
        # The decision-state engine still owns exactly its own two tables.
        state_schema = _sqlite_schema(state_db)
        assert "events" in state_schema and "meta" in state_schema


# =========================================================================== #
# 7-8. turn identity: idempotency key vs distinct server turns
# =========================================================================== #


def test_retry_with_same_idempotency_key_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 7: a client-keyed retry re-emits the same ids and appends nothing."""
    with _harness(tmp_path, monkeypatch, "shadow", "retry") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        body = {
            "message": PREFERENCE_MESSAGE,
            "constraints": [],
            "idempotency_key": "retry-0001",
        }

        first = _shopping_turn(harness, session_id, **body)
        assert first.status_code == 200, first.text
        first_state = _read_state(state_db, user_key, task_id)
        first_diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(first_diagnostics, mode="shadow")
        assert first_diagnostics["turn_identity_source"] == "idempotency_key"
        assert first_diagnostics["error"] is None
        first_ids = {event.event_id for event in first_state["events"]}
        first_rows = _stored_rows(state_db, user_key, task_id)
        assert first_rows == len(first_state["events"]) == first_diagnostics["event_count"]
        # The accepted M9 memory row count is part of the retry contract: a transport retry
        # must not duplicate the memory fact either.
        memory_db = _memory_db(state_db.parent)
        memory_rows_before = _sqlite_row_count(memory_db, "preference_memory")

        second = _shopping_turn(harness, session_id, **body)
        assert second.status_code == 200, second.text
        second_state = _read_state(state_db, user_key, task_id)
        second_diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(second_diagnostics, mode="shadow")
        assert second_diagnostics["turn_identity_source"] == "idempotency_key"

        assert {event.event_id for event in second_state["events"]} == first_ids
        assert len(second_state["events"]) == len(first_state["events"])
        assert second_state["digest"] == first_state["digest"]
        assert second_state["state"].state_digest() == first_state["digest"]
        assert _stored_rows(state_db, user_key, task_id) == first_rows
        assert second_diagnostics["event_count"] == first_diagnostics["event_count"]
        assert second_diagnostics["state_digest"] == first_diagnostics["state_digest"]
        # The diagnostics counter still counts the attempt: that is what makes a retry
        # observable even though the append-only store treated it as a duplicate.
        assert second_diagnostics["turn_count"] == first_diagnostics["turn_count"] + 1
        assert _sqlite_row_count(memory_db, "preference_memory") == memory_rows_before, (
            "a retried turn must not duplicate the accepted memory fact"
        )


def test_identical_turns_without_key_get_distinct_server_identities(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 8: two identical messages without a key are two distinct turns."""
    with _harness(tmp_path, monkeypatch, "shadow", "distinct-turns") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        body = {"message": PREFERENCE_MESSAGE, "constraints": []}

        first = _shopping_turn(harness, session_id, **body)
        assert first.status_code == 200, first.text
        first_state = _read_state(state_db, user_key, task_id)
        first_diagnostics = _service(harness).state_diagnostics(task_id)
        first_ids = {event.event_id for event in first_state["events"]}

        second = _shopping_turn(harness, session_id, **body)
        assert second.status_code == 200, second.text
        second_state = _read_state(state_db, user_key, task_id)
        second_diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(second_diagnostics, mode="shadow")
        assert second_diagnostics["turn_identity_source"] == "server_turn_id"

        # The server allocated two turns: the session sequence advanced, and the two
        # identical messages did NOT collapse into one logical turn.  A state read returns
        # the whole task stream, so the *second* turn's new rows are the difference.
        assert _session(harness, session_id).next_turn_sequence == 2
        first_ids = {event.event_id for event in first_state["events"]}
        all_events = second_state["events"]
        new_events = [event for event in all_events if event.event_id not in first_ids]
        assert new_events, "the second identical turn must append its own events"
        assert len(new_events) == len(first_state["events"])
        assert {event.event_id for event in new_events}.isdisjoint(first_ids), (
            "distinct turns must not reuse event ids"
        )
        assert len({event.event_id for event in all_events}) == len(all_events)
        assert len(_preference_events(all_events)) == 2
        assert second_diagnostics["event_count"] == len(all_events)
        assert second_diagnostics["state_digest"] == second_state["digest"]


# =========================================================================== #
# 9-10. isolation
# =========================================================================== #


def test_cross_user_isolation(tmp_path: Path, monkeypatch) -> None:
    """Requirement 9: two users are separate state namespaces; nothing crosses over."""
    with _harness(tmp_path, monkeypatch, "shadow", "cross-user") as (harness, state_db):
        first_session = _create_session(harness, PROFILE_A)
        second_session = _create_session(harness, PROFILE_B)
        assert first_session != second_session
        first_key, first_task = _identity(harness, first_session)
        second_key, second_task = _identity(harness, second_session)
        assert first_key != second_key and first_task != second_task

        first = _shopping_turn(
            harness, first_session, message="I prefer black.", constraints=()
        )
        second = _shopping_turn(
            harness, second_session, message="I prefer red.", constraints=()
        )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text

        first_state = _read_state(state_db, first_key, first_task)
        second_state = _read_state(state_db, second_key, second_task)
        assert _active_identities(first_state["state"]) == {
            ("color", "prefer", "black")
        }
        assert _active_identities(second_state["state"]) == {("color", "prefer", "red")}
        assert first_state["digest"] != second_state["digest"]
        _assert_event_batch(first_state["events"], user_key=first_key, task_id=first_task)
        _assert_event_batch(second_state["events"], user_key=second_key, task_id=second_task)
        with _open_state(state_db) as (store, _manager):
            assert store.get_events(second_task, first_key) == ()
            assert store.get_events(first_task, second_key) == ()
            assert store.count(first_key) == len(first_state["events"])
            assert store.count(second_key) == len(second_state["events"])
        service = _service(harness)
        assert service.state_diagnostics(first_task)["state_digest"] == first_state["digest"]
        assert (
            service.state_diagnostics(second_task)["state_digest"]
            == second_state["digest"]
        )


def test_cross_task_isolation(tmp_path: Path, monkeypatch) -> None:
    """Requirement 10: events/diagnostics are task-scoped; tasks never replay each other."""
    with _harness(tmp_path, monkeypatch, "shadow", "cross-task") as (harness, state_db):
        first_session = _create_session(harness, PROFILE_A)
        second_session = _create_session(harness, PROFILE_A)
        assert first_session != second_session
        first_key, first_task = _identity(harness, first_session)
        second_key, second_task = _identity(harness, second_session)

        first = _shopping_turn(
            harness, first_session, message="I prefer black.", constraints=()
        )
        assert first.status_code == 200, first.text
        first_state = _read_state(state_db, first_key, first_task)
        first_ids = {event.event_id for event in first_state["events"]}
        assert first_ids

        second = _shopping_turn(
            harness, second_session, message="I prefer red.", constraints=()
        )
        assert second.status_code == 200, second.text
        second_state = _read_state(state_db, second_key, second_task)
        assert {event.event_id for event in second_state["events"]}.isdisjoint(first_ids)
        assert all(event.task_id == second_task for event in second_state["events"])
        assert all(event.task_id == first_task for event in first_state["events"])
        # A task's state vanishes from every other task's replay: the task id scopes it.
        with _open_state(state_db) as (store, manager):
            assert store.get_events(second_task, first_key) == ()
            assert store.get_events(first_task, second_key) == ()
            other_task_state = manager.replay_state(first_key, "task-not-this-one")
            assert other_task_state.interaction.event_count == 0
        assert _session(harness, first_session).next_turn_sequence == 1
        assert _session(harness, second_session).next_turn_sequence == 1
        service = _service(harness)
        assert service.state_diagnostics(first_task)["task_id"] == first_task
        assert service.state_diagnostics(second_task)["task_id"] == second_task


# =========================================================================== #
# 11. replay after reopening SQLite
# =========================================================================== #


def test_replay_after_reopening_sqlite_equals_diagnostics_digest(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 11: a fresh store/manager over the same file reproduces the digest."""
    state_db: Path
    with _harness(tmp_path, monkeypatch, "shadow", "replay") as (harness, db):
        state_db = db
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        response = _shopping_turn(
            harness, session_id, message=PREFERENCE_MESSAGE, constraints=PARITY_CONSTRAINTS
        )
        assert response.status_code == 200, response.text
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        expected_digest = diagnostics["state_digest"]
        expected_count = diagnostics["event_count"]
        live_state = _read_state(state_db, user_key, task_id)

    # The app is closed; replay happens in a fresh store/manager over the same file.
    replayed = _read_state(state_db, user_key, task_id)
    replayed_again = _read_state(state_db, user_key, task_id)
    assert replayed["digest"] == replayed_again["digest"] == expected_digest
    assert replayed["state"].interaction.event_count == expected_count
    assert [event.event_id for event in replayed["events"]] == [
        event.event_id for event in live_state["events"]
    ]
    assert replayed["agent"].projection_version == 1
    assert replayed["ranker"].ranker_signal_version == 1
    assert replayed["constraints"].constraints_version == 1
    assert replayed["state"].candidate_pool_ref == live_state["state"].candidate_pool_ref
    assert replayed["state"].ranking_snapshot_ref == live_state["state"].ranking_snapshot_ref


# =========================================================================== #
# 12-13. OFF default and SHADOW parity
# =========================================================================== #


def test_off_mode_is_unchanged_and_writes_no_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 12: default OFF == explicit off; no state DB, no diagnostics."""
    with _harness(tmp_path, monkeypatch, None, "off-default") as (default, default_db):
        default_session = _create_session(default)
        default_response = _shopping_turn(
            default, default_session, message=PARITY_MESSAGE, constraints=PARITY_CONSTRAINTS
        )
        assert default_response.status_code == 200, default_response.text
        assert _service(default).state_mode == "off"
        assert _service(default).state_diagnostics(default_session) is None
        assert not default_db.exists(), "OFF opened/created the decision-state database"
        assert "state" not in default_response.json()
        assert "diagnostics" not in default_response.json()

    with _harness(tmp_path, monkeypatch, "off", "off-explicit") as (explicit, explicit_db):
        explicit_session = _create_session(explicit)
        explicit_response = _shopping_turn(
            explicit, explicit_session, message=PARITY_MESSAGE, constraints=PARITY_CONSTRAINTS
        )
        assert explicit_response.status_code == 200, explicit_response.text
        assert _service(explicit).state_mode == "off"
        assert _service(explicit).state_diagnostics(explicit_session) is None
        assert not explicit_db.exists(), "explicit off opened/created the state database"

    assert_response_parity(
        default_response.json(), explicit_response.json(), context="OFF default vs explicit"
    )
    body = default_response.json()
    # price_max=120 is decidable but excludes nobody in this catalogue, so the accepted
    # fused order is unchanged.
    assert [item["parent_asin"] for item in body["recommendations"]] == list(INITIAL_ORDER)
    assert body["terminal"]["status"] == "finished"


def test_shadow_mode_is_user_visible_parity_while_recording_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 13: SHADOW answers identically to OFF while producing state/diagnostics."""
    with _harness(tmp_path, monkeypatch, "off", "shadow-off") as (reference, _ref_db):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(
            reference, reference_session, message=PARITY_MESSAGE, constraints=PARITY_CONSTRAINTS
        )
        assert reference_response.status_code == 200, reference_response.text

    with _harness(tmp_path, monkeypatch, "shadow", "shadow-on") as (shadow, state_db):
        session_id = _create_session(shadow)
        user_key, task_id = _identity(shadow, session_id)
        response = _shopping_turn(
            shadow, session_id, message=PARITY_MESSAGE, constraints=PARITY_CONSTRAINTS
        )
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="SHADOW vs OFF"
        )
        assert state_db.exists(), "SHADOW must persist the decision state"
        recorded = _read_state(state_db, user_key, task_id)
        assert recorded["events"]
        diagnostics = _service(shadow).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["state_digest"] == recorded["digest"]
        assert diagnostics["turn_identity_source"] == "server_turn_id"


# =========================================================================== #
# 14. APPLY_CONSTRAINTS: effective constraints + consistent intent view
# =========================================================================== #


def _intent_entries(body: dict[str, Any], strength: str) -> list[dict[str, Any]]:
    intent = body.get("intent") or {}
    return list(intent.get(strength) or ())


def _find_intent_entry(
    entries: list[dict[str, Any]], *, kind: str, value: str
) -> dict[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind")) == kind and str(entry.get("value")).casefold() == value.casefold():
            return entry
    return None


def test_apply_mode_uses_authorized_constraints_with_consistent_intent_view(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 14: APPLY enforces exactly the constraints its intent view reports."""
    with _harness(tmp_path, monkeypatch, "apply_constraints", "apply-view") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)

        # Turn 1: the current-turn user statement creates the authorized constraint and a
        # remembered colour preference.  It must be reported with origin=user_stated.
        first = _shopping_turn(
            harness,
            session_id,
            message="Recommend some hiking gear. I prefer black.",
            constraints=BUDGET_CONSTRAINT,
        )
        assert first.status_code == 200, first.text
        first_body = first.json()
        first_hard = _intent_entries(first_body, "hard")
        stated_budget = _find_intent_entry(first_hard, kind="price_max", value="30")
        assert stated_budget is not None, f"current-turn constraint missing from intent: {first_hard}"
        assert stated_budget["origin"] == "user_stated"
        first_soft = _intent_entries(first_body, "soft")
        stated_colour = _find_intent_entry(first_soft, kind="color", value="black")
        assert stated_colour is not None
        assert stated_colour["origin"] == "user_stated"
        assert {"kind": "price_max", "expected": "30"} in first_body["eligibility"][
            "requirements"
        ]

        # Turn 2: no body constraint and no new preference.  The authorized M5 constraint
        # must still narrow eligibility, and the response intent view must report it as
        # decision_state; the remembered M9 colour is validated_existing, not user_stated.
        second = _shopping_turn(
            harness,
            session_id,
            message=NEUTRAL_MESSAGE,
            constraints=(),
        )
        assert second.status_code == 200, second.text
        body = second.json()

        second_hard = _intent_entries(body, "hard")
        state_budget = _find_intent_entry(second_hard, kind="price_max", value="30")
        assert state_budget is not None, (
            f"the effective M5 constraint is missing from the intent view: {second_hard}"
        )
        assert state_budget["origin"] == "decision_state"
        second_soft = _intent_entries(body, "soft")
        remembered_colour = _find_intent_entry(second_soft, kind="color", value="black")
        assert remembered_colour is not None
        assert remembered_colour["origin"] == "validated_existing"

        # The existing evaluator consumed the same constraint set: eligibility and intent
        # cannot disagree.
        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert {"kind": "price_max", "expected": "30"} in eligibility["requirements"]
        presented = [item["parent_asin"] for item in body["recommendations"]]
        assert presented == ["cand-red", "cand-black"], presented
        assert (
            eligibility["verified_eligible_count"]
            + eligibility["ineligible_count"]
            + eligibility["unresolved_count"]
            == len(eligibility["assessments"])
        )
        ineligible = {
            entry["parent_asin"]
            for entry in eligibility["assessments"]
            if entry["eligibility"] == "ineligible"
        }
        assert ineligible == {"cand-blue", "cand-green"}

        # The state and its authorized projection carry exactly that constraint.
        recorded = _read_state(state_db, user_key, task_id)
        assert [
            (constraint.kind.value, constraint.value)
            for constraint in recorded["state"].authorized_hard_constraints
        ] == [("price_max", "30")]
        assert [
            (view.kind.value, view.value) for view in recorded["constraints"].constraints
        ] == [("price_max", "30")]
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] is None
        assert diagnostics["hard_constraint_count"] == 1
        assert diagnostics["state_digest"] == recorded["digest"]


# =========================================================================== #
# 15. UNKNOWN stays UNKNOWN
# =========================================================================== #


def test_unknown_stays_unknown_not_ineligible(tmp_path: Path, monkeypatch) -> None:
    """Requirement 15: a structurally unverifiable constraint never becomes a violation."""
    with _harness(tmp_path, monkeypatch, "apply_constraints", "unknown") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        _seed_state(
            state_db,
            user_key,
            task_id,
            [
                constraint_event(
                    user_key=user_key,
                    task_id=task_id,
                    session_id=task_id,
                    kind="free_form_constraint",
                    value="extra-large",
                )
            ],
        )
        response = _shopping_turn(harness, session_id, message=NEUTRAL_MESSAGE, constraints=())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["recommendations"] == []
        assert body["terminal"]["status"] == "waiting_for_user"
        assert body["terminal"]["termination_reason"] == "awaiting_user"
        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert eligibility["ineligible_count"] == 0, (
            "no candidate may be ineligible only because a dimension is UNKNOWN"
        )
        assessments = eligibility["assessments"]
        assert eligibility["unresolved_count"] == len(assessments) == 4
        assert (
            eligibility["verified_eligible_count"]
            + eligibility["ineligible_count"]
            + eligibility["unresolved_count"]
            == len(assessments)
        )
        for assessment in assessments:
            assert assessment["eligibility"] == "unresolved"
            assert assessment["reports"]
            assert all(report["verdict"] == "unknown" for report in assessment["reports"])

        # The effective intent view names the unverifiable constraint (decision_state) and
        # eligibility consumed it: one visible requirement, three-state preserved.
        requirement = next(iter(eligibility["requirements"]))
        assert requirement["kind"] == "unverifiable"
        recorded = _read_state(state_db, user_key, task_id)
        assert [
            (constraint.kind.value, constraint.value)
            for constraint in recorded["state"].authorized_hard_constraints
        ] == [("free_form_constraint", "extra-large")]
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] is None


# =========================================================================== #
# 16. inferred hypothesis never hard-filters
# =========================================================================== #


def test_inferred_hypothesis_never_hard_filters(tmp_path: Path, monkeypatch) -> None:
    """Requirement 16: a soft inference can never narrow the candidate set, even in APPLY."""
    with _harness(tmp_path, monkeypatch, "off", "hypothesis-off") as (reference, _ref_db):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(
            reference, reference_session, message=NEUTRAL_MESSAGE, constraints=()
        )
        assert reference_response.status_code == 200, reference_response.text
        reference_ids = [
            item["parent_asin"] for item in reference_response.json()["recommendations"]
        ]

    with _harness(tmp_path, monkeypatch, "apply_constraints", "hypothesis-apply") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        _seed_state(
            state_db,
            user_key,
            task_id,
            [
                hypothesis_event(
                    user_key=user_key,
                    task_id=task_id,
                    session_id=task_id,
                    kind="price_max",
                    value="1",
                )
            ],
        )
        response = _shopping_turn(harness, session_id, message=NEUTRAL_MESSAGE, constraints=())
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["parent_asin"] for item in body["recommendations"]] == reference_ids
        assert body["eligibility"]["evaluated"] is False

        recorded = _read_state(state_db, user_key, task_id)
        assert len(recorded["state"].inferred_hypotheses) == 1
        assert [
            (belief.kind.value, belief.value)
            for belief in recorded["state"].inferred_hypotheses
        ] == [("price_max", "1")]
        assert recorded["state"].authorized_hard_constraints == ()
        assert recorded["constraints"].constraints == ()
        assert recorded["constraints"].excluded_hypothesis_count >= 1
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] is None
        assert diagnostics["hard_constraint_count"] == 0
        assert diagnostics["soft_hypothesis_count"] == 1
        assert diagnostics["excluded_hypothesis_count"] >= 1


# =========================================================================== #
# 17. candidate / ranking snapshot provenance
# =========================================================================== #


def test_candidate_and_ranking_snapshots_have_trusted_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement 17: two trusted snapshot events, opaque payload, refs in the state."""
    with _harness(tmp_path, monkeypatch, "shadow", "snapshots") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        response = _shopping_turn(harness, session_id, message=NEUTRAL_MESSAGE, constraints=())
        assert response.status_code == 200, response.text
        body = response.json()
        presented = [item["parent_asin"] for item in body["recommendations"]]
        grounded = [item["parent_asin"] for item in (body.get("grounded") or {}).get("candidates", [])]
        assert presented and grounded

        recorded = _read_state(state_db, user_key, task_id)
        snapshots = {
            event.event_type.value: event for event in _snapshot_events(recorded["events"])
        }
        assert set(snapshots) == {
            "candidate_pool_referenced",
            "ranking_snapshot_referenced",
        }, f"expected exactly two snapshot references, got {sorted(snapshots)}"
        pool_event = snapshots["candidate_pool_referenced"]
        ranking_event = snapshots["ranking_snapshot_referenced"]

        for event in (pool_event, ranking_event):
            assert event.actor.value == "trusted_component"
            assert event.payload.snapshot_schema_version == 1
            assert event.payload.candidate_count >= 0
            assert _HEX64.match(event.payload.digest or "")
            assert event.payload.ref
            serialised = json.dumps(event.payload.model_dump(mode="json"), sort_keys=True)
            for identity in presented:
                assert identity not in serialised, (
                    "a product identity must not enter the decision state"
                )
            for forbidden in ("score", "rank", "parent_asin", "item_id"):
                assert forbidden not in event.payload.model_dump(mode="json")

        assert pool_event.payload.ref.startswith("cp_")
        assert pool_event.payload.candidate_count == len(set(grounded))
        assert ranking_event.payload.ref.startswith("rs_")
        assert ranking_event.payload.method in {"presented_order", "task_preference_rerank"}
        assert ranking_event.payload.candidate_count == len(presented)

        state = recorded["state"]
        pool_ref = state.candidate_pool_ref
        ranking_ref = state.ranking_snapshot_ref
        assert pool_ref is not None and ranking_ref is not None
        assert pool_ref.read_only is True and ranking_ref.read_only is True
        assert pool_ref.ref == pool_event.payload.ref
        assert pool_ref.digest == pool_event.payload.digest
        assert pool_ref.candidate_count == pool_event.payload.candidate_count
        assert pool_ref.snapshot_schema_version == 1
        assert pool_ref.referenced_by_event_id == pool_event.event_id
        assert ranking_ref.ref == ranking_event.payload.ref
        assert ranking_ref.digest == ranking_event.payload.digest
        assert ranking_ref.candidate_count == ranking_event.payload.candidate_count
        assert ranking_ref.method == ranking_event.payload.method
        assert ranking_ref.snapshot_schema_version == 1
        assert ranking_ref.referenced_by_event_id == ranking_event.event_id

        # The read-only projections expose the refs/counts, never product rows.
        agent_view = recorded["agent"]
        assert agent_view.candidate_pool_ref == pool_ref.ref
        assert agent_view.candidate_pool_candidate_count == pool_ref.candidate_count
        assert agent_view.ranking_snapshot_ref == ranking_ref.ref
        assert recorded["ranker"].candidate_pool_ref == pool_ref.ref
        assert recorded["ranker"].ranking_snapshot_ref == ranking_ref.ref

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["snapshot"] is not None
        snapshot_blob = json.dumps(diagnostics["snapshot"], sort_keys=True)
        assert pool_ref.ref in snapshot_blob or ranking_ref.ref in snapshot_blob


# =========================================================================== #
# 18-19. failure policy: invalid event, unavailable store
# =========================================================================== #


class _InvalidExtractor:
    """An extractor whose *turn-preview* output the runtime tolerates but M5 rejects.

    The accepted runtime calls the extractor twice: once for the intent preview (whose
    output M5.2 converts) and once when the loop commits memory.  Only the first call
    returns the malformed candidate (a 200-character value, impossible under the accepted
    M9 bound), so the accepted memory write still succeeds and OFF serves the normal
    answer; M5.2 must reject the preview as an invalid event.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.value = "x" * 200

    def extract(self, user_message: str) -> Any:
        self.calls += 1
        if self.calls > 1:
            return PreferenceExtraction()
        return SimpleNamespace(
            preferences=(
                SimpleNamespace(
                    kind=PreferenceKind.BRAND,
                    value=self.value,
                    polarity=PreferencePolarity.PREFER,
                    source_text="I want this brand",
                    mode=PreferenceMode.ADD,
                    replaces=None,
                    extractor="test_invalid",
                ),
            ),
            removals=(),
        )


def test_invalid_event_is_not_appended(tmp_path: Path, monkeypatch) -> None:
    """Requirement 18: an invalid fact is not appended; per-mode failure policy holds."""
    request = dict(message=NEUTRAL_MESSAGE, constraints=())

    with _harness(tmp_path, monkeypatch, "off", "invalid-off") as (reference, _ref_db):
        reference_session = _create_session(reference)
        monkeypatch.setattr(
            reference.runtime.memory_service, "_extractor", _InvalidExtractor()
        )
        reference_response = _shopping_turn(reference, reference_session, **request)
        assert reference_response.status_code == 200, reference_response.text

    with _harness(tmp_path, monkeypatch, "shadow", "invalid-shadow") as (harness, state_db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        rows_before = _stored_rows(state_db, user_key, task_id)
        monkeypatch.setattr(
            harness.runtime.memory_service, "_extractor", _InvalidExtractor()
        )
        response = _shopping_turn(harness, session_id, **request)
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="invalid event SHADOW vs OFF"
        )
        rows_after = _stored_rows(state_db, user_key, task_id)
        assert rows_after == rows_before == 0, (
            "an invalid event batch must store nothing at all"
        )
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] == "state_event_invalid"
        assert diagnostics["events"] == 0
        assert diagnostics["event_count"] == 0
        assert diagnostics["state_digest"] is None

    with _harness(tmp_path, monkeypatch, "apply_constraints", "invalid-apply") as (
        harness,
        _state_db,
    ):
        session_id = _create_session(harness)
        _, task_id = _identity(harness, session_id)
        monkeypatch.setattr(
            harness.runtime.memory_service, "_extractor", _InvalidExtractor()
        )
        response = _shopping_turn(harness, session_id, **request)
        assert response.status_code == 502, response.text
        body = response.json()
        assert body["error"] in {"agent_failed", "state_integration_failed"}
        assert "recommendations" not in body
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] == "state_event_invalid"


def test_store_failure_follows_mode_policy(tmp_path: Path, monkeypatch) -> None:
    """Requirement 19: unopenable state DB -> SHADOW diagnostics-only, APPLY fail-closed."""
    request = dict(message=NEUTRAL_MESSAGE, constraints=())

    with _harness(tmp_path, monkeypatch, "off", "store-off") as (reference, _ref_db):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(reference, reference_session, **request)
        assert reference_response.status_code == 200, reference_response.text

    # A path whose parent is a regular file: the store cannot be opened (a realistic
    # misconfiguration the request loop must not crash on).
    blocked_root = tmp_path / "store-blocked"
    blocked_root.mkdir(parents=True, exist_ok=True)
    (blocked_root / "blocked").write_text("not a directory", encoding="utf-8")
    blocked_db = blocked_root / "blocked" / "state.sqlite3"
    monkeypatch.setenv(STATE_DB_ENV, str(blocked_db))

    monkeypatch.setenv(STATE_MODE_ENV, "shadow")
    shadow_root = blocked_root / "shadow"
    shadow_root.mkdir(parents=True, exist_ok=True)
    harness = build_harness(shadow_root)
    try:
        session_id = _create_session(harness)
        _, task_id = _identity(harness, session_id)
        response = _shopping_turn(harness, session_id, **request)
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="store unavailable SHADOW vs OFF"
        )
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] == "state_store_unavailable"
        assert diagnostics["state_digest"] is None
        assert diagnostics["event_count"] == 0
    finally:
        close_harness(harness)

    monkeypatch.setenv(STATE_MODE_ENV, "apply_constraints")
    apply_root = blocked_root / "apply"
    apply_root.mkdir(parents=True, exist_ok=True)
    harness = build_harness(apply_root)
    try:
        session_id = _create_session(harness)
        _, task_id = _identity(harness, session_id)
        response = _shopping_turn(harness, session_id, **request)
        assert response.status_code == 502, response.text
        assert response.json()["error"] in {"agent_failed", "state_integration_failed"}
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] == "state_store_unavailable"
    finally:
        close_harness(harness)


# =========================================================================== #
# additional required follow-ups
# =========================================================================== #


def test_pending_conflict_turn_still_records_validated_facts(
    tmp_path: Path, monkeypatch
) -> None:
    """A suspended (conflict) turn still records the validated facts that caused it."""
    with _harness(tmp_path, monkeypatch, "shadow", "pending-conflict") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        conflicting = (
            {"kind": "price_max", "value": "50"},
            {"kind": "price_min", "value": "100"},
        )
        response = _shopping_turn(
            harness, session_id, message=NEUTRAL_MESSAGE, constraints=conflicting
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "waiting_for_user"
        assert body["terminal"]["termination_reason"] == "awaiting_user"
        assert body["clarification"] is not None
        assert body["clarification"]["kind"] == "conflict"

        recorded = _read_state(state_db, user_key, task_id)
        events = recorded["events"]
        assert events, "a suspended turn must still record its validated facts"
        _assert_event_batch(events, user_key=user_key, task_id=task_id)
        context = [
            event for event in events if event.event_type.value == "task_context_updated"
        ]
        assert len(context) == 1
        constraints = {
            (event.payload.kind.value, event.payload.value)
            for event in events
            if event.event_type.value == "user_constraint_stated"
        }
        assert constraints == {("price_max", "50"), ("price_min", "100")}
        assert {
            (constraint.kind.value, constraint.value)
            for constraint in recorded["state"].authorized_hard_constraints
        } == {("price_max", "50"), ("price_min", "100")}
        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["state_digest"] == recorded["digest"]


def test_unmatched_removal_reports_disagreement_without_inventing_events(
    tmp_path: Path, monkeypatch
) -> None:
    """An unmatched value-only removal reports a bounded disagreement, never an event."""
    with _harness(tmp_path, monkeypatch, "shadow", "unmatched-removal") as (
        harness,
        state_db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        first = _shopping_turn(harness, session_id, message=NEUTRAL_MESSAGE, constraints=())
        assert first.status_code == 200, first.text
        before = _read_state(state_db, user_key, task_id)
        before_ids = {event.event_id for event in before["events"]}

        # No active task belief has this value: the legacy M9 removal and the M5 task state
        # disagree, and the integration must report the mismatch instead of guessing a kind.
        response = _shopping_turn(
            harness,
            session_id,
            message="I don't care about the waterproofing preference anymore.",
            constraints=(),
        )
        assert response.status_code == 200, response.text
        after = _read_state(state_db, user_key, task_id)
        new_events = [
            event for event in after["events"] if event.event_id not in before_ids
        ]
        assert not [
            event for event in new_events if event.event_type.value == "preference_retracted"
        ], "an unmatched removal must not invent a retraction event"
        assert after["state"].active_preference_beliefs == ()
        assert after["state"].state_digest() == after["digest"]

        diagnostics = _service(harness).state_diagnostics(task_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        match = diagnostics["retraction_match"]
        assert match is not None, "an unmatched removal must be reported in diagnostics"
        numbers = _retraction_numbers(match)
        assert any(number >= 1 for number in numbers.values()), (
            f"the bounded disagreement must count the unmatched selector: {numbers}"
        )
