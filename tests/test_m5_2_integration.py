"""Focused M5.2 tests: interaction feedback loop & state consistency integration.

Scope: the **demo integration**, not the M5 engine, the M9 extractor or the shopping
policy (each has its own suite).  These tests drive the real offline demo harness
(:func:`tests.demo_fixture.build_harness`) over HTTP, flip ``AGENTRECX_STATE_MODE``, and
inspect the decision-state database / documented diagnostics surface directly.

Requirement map
---------------
1. an initial shopping turn records task context + explicit preference + explicit
   constraint (and the deterministic turn identity source);
2. a preference-question round-trip: a structured answer becomes an explicit TASK-scoped
   preference belief with a trusted direction, visible in ``for_agent``/``for_ranker``,
   with no response-schema change;
3. an ambiguous/declined answer (``pref:none`` / ``pref:skip``) creates NO preference
   belief -- only task context (and the snapshot of the list it re-presents);
4. an explicit ADD turn adds a belief without deleting the earlier one;
5. an explicit REPLACE correction maps to supersede: a named target ("instead of
   black") and a general replacement with no named target ("...instead.") both
   supersede the old same-kind explicit belief, while a later ADD adds without
   superseding;
6. an explicit REMOVE maps to ``PREFERENCE_RETRACTED``; a value-only removal resolves
   uniquely or is reported in ``retraction_match`` (never guessed);
7. task-scope feedback never touches LONG_TERM state;
8. idempotency: the same ``idempotency_key`` retry reuses event ids/counts/digests;
   distinct identical messages without a key get distinct server turn ids;
9. cross-user / cross-task isolation;
10. replay after reopen equals the diagnostics digest;
11. OFF is inert and opens no state database;
12. SHADOW user-visible response equals OFF (normalising clocks/run ids/turn ids);
13. APPLY enforces a seeded M5 constraint AND shows it in ``intent.hard`` with
    ``origin="decision_state"`` while a remembered constraint keeps
    ``origin="validated_existing"``;
14. an inferred hypothesis never becomes a hard filter;
15. an UNKNOWN (free-form) constraint stays UNKNOWN, never ineligible;
16. trusted candidate/ranking snapshot references are recorded after a presented
    response: opaque refs, counts, digests, ``snapshot_schema_version == 1``, projected
    refs, no product identity or score;
17. an invalid event is not appended (SHADOW diagnostic vs APPLY 502), store unchanged;
18. store failure modes: bad state DB -> SHADOW 200 + diagnostics, APPLY 502.

The fixture catalogue is the established synthetic one: prices 25.00 / 40.00 / 15.00 /
60.00 for cand-red / cand-blue / cand-black / cand-green (see
``tests/agent_reranking_fixture.py``).
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from recommendation.demo.agent_schemas import AgentRecommendResponse
from recommendation.demo.state_integration import (
    ENV_STATE_MODE,
    SNAPSHOT_SCHEMA_VERSION,
    ShoppingStateIntegration,
    StateIntegrationError,
    StateMode,
)
from recommendation.memory import (
    PreferenceExtraction,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
)
from recommendation.state import SQLiteEventStore, StateManager
from recommendation.state.schemas import PreferenceScope
from recommendation.state.state_projector import for_agent, for_constraints, for_ranker
from tests.agent_reranking_fixture import CANDIDATE_ROWS, INITIAL_ORDER
from tests.demo_fixture import PROFILE_A, PROFILE_B, build_harness, close_harness
from tests.state_fixtures import constraint_event, hypothesis_event, preference_stated_event

BASE = "/v1/demo"
CREATE = BASE + "/sessions"
SHOPPING = BASE + "/sessions/{session_id}/agent/shopping"

ENV_STATE_DB = "AGENTRECX_STATE_DB"

#: The fixed shopping requests used across tests.  The extractor resolves "black" to an
#: explicit colour preference, so a planning turn emits a preference event as well.
PREFERENCE_MESSAGE = "Recommend some hiking gear. I prefer black shoes."
NEUTRAL_MESSAGE = "Recommend some hiking gear."

#: Fields the accepted control plane randomises per request (run ids are uuid-based),
#: that embed the concrete server-owned session id, or that are wall clocks.
_RANDOM_KEYS = frozenset(
    {
        "timing",
        "run_id",
        "run_ref",
        "source_turn_id",
        "turn_ids",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_REF = re.compile(r"^cp_[0-9a-f]{24}$")
_RANKING_REF = re.compile(r"^rs_[0-9a-f]{24}$")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


@contextmanager
def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str | None,
    name: str,
    rows: Any = CANDIDATE_ROWS,
) -> Iterator[tuple[Any, Path]]:
    """Build one offline demo harness with the given state mode; yield (harness, db)."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    database = root / "decision_state.sqlite3"
    if mode is None:
        monkeypatch.delenv(ENV_STATE_MODE, raising=False)
    else:
        monkeypatch.setenv(ENV_STATE_MODE, mode)
    monkeypatch.setenv(ENV_STATE_DB, str(database))
    harness = build_harness(root, rows=rows, memory_db=root / "memory.sqlite3")
    try:
        yield harness, database
    finally:
        close_harness(harness)


def _create_session(harness: Any, profile: str = PROFILE_A) -> str:
    response = harness.client.post(CREATE, json={"profile_id": profile})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _shopping(
    harness: Any,
    session_id: str,
    message: str = NEUTRAL_MESSAGE,
    *,
    constraints: Any = (),
    **overrides: Any,
) -> Any:
    body: dict[str, Any] = {
        "message": message,
        "k": 3,
        "sources": ["history", "catalog_search"],
        "constraints": list(constraints),
    }
    body.update(overrides)
    return harness.client.post(SHOPPING.format(session_id=session_id), json=body)


def _identity(harness: Any, session_id: str) -> tuple[str, str]:
    """The server-owned ``(user_key, task_id)`` pair of one demo session."""
    session = harness.runtime.sessions.get(session_id)
    return session.user_key, session.session_id


def _service(harness: Any) -> Any:
    return harness.runtime.agent_service()


def _read_state(database: Path, user_key: str, task_id: str) -> tuple[Any, Any]:
    """Read one task's events and replay its state with a fresh manager."""
    store = SQLiteEventStore(database)
    try:
        return store.get_events(task_id, user_key), StateManager(store).replay_state(
            user_key, task_id
        )
    finally:
        store.close()


def _stored_rows(database: Path, user_key: str) -> int:
    """Durable event rows for one user (0 when the database was never created)."""
    if not database.exists():
        return 0
    store = SQLiteEventStore(database)
    try:
        return len(store.get_all_events(user_key))
    finally:
        store.close()


def _seed_events(database: Path, events: Any) -> None:
    """Append fixture events to a standalone decision-state database."""
    store = SQLiteEventStore(database)
    try:
        manager = StateManager(store)
        for event in events:
            manager.apply_event(event)
    finally:
        store.close()


def _event_types(events: Any) -> list[str]:
    return [event.event_type.value for event in events]


def _last_event(events: Any, event_type: str) -> Any:
    matches = [event for event in events if event.event_type.value == event_type]
    assert matches, f"no {event_type} event in {[e.event_type.value for e in events]}"
    return matches[-1]


def _belief(state: Any, kind: str, value: str, *, scope: str = "task") -> Any:
    matches = [
        belief
        for belief in state.preference_beliefs
        if belief.kind.value == kind
        and belief.value.casefold() == value.casefold()
        and belief.scope.value == scope
    ]
    assert matches, (
        f"no {scope} belief {kind}={value}; have "
        f"{[(b.kind.value, b.value, b.scope.value, b.status.value) for b in state.preference_beliefs]}"
    )
    return matches[-1]


def _normalised(payload: Any) -> Any:
    """Response body with per-request random fields and session-derived turn ids dropped."""
    if isinstance(payload, dict):
        return {
            key: _normalised(value)
            for key, value in payload.items()
            if key not in _RANDOM_KEYS
        }
    if isinstance(payload, (list, tuple)):
        return [_normalised(value) for value in payload]
    return payload


def _diagnostics_json(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)


class _InvalidExtractor:
    """A preview extractor whose *first* call yields a preference M5 must reject.

    The accepted runtime calls the extractor twice per shopping turn; the malformed
    200-character value is only returned by the preview call, so the accepted M9 memory
    write (which runs later) is unaffected and OFF still serves a normal answer.  The M5
    adapter therefore rejects the fact as an invalid event -- the failure the SHADOW and
    APPLY policies are tested on.
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


def _equalise_attributes(harness: Any, identities: tuple[str, ...]) -> None:
    """Make every preference dimension except price identical for the given candidates.

    The accepted preference policy then asks (and, after the answer, finishes) on price
    alone, which is what lets a round-trip test assert a presented reranked response.
    """
    records = harness.runtime.metadata.records
    for identity in identities:
        record = records[identity]
        records[identity] = record.model_copy(
            update={
                "details": (
                    ("Color", "red"),
                    ("Item Weight", "0.8 Kilograms"),
                    ("Material", "cotton"),
                    ("Brand Name", "Acme"),
                )
            }
        )


# =========================================================================== #
# 1. initial turn: task context + explicit preference + explicit constraint
# =========================================================================== #


def test_initial_turn_records_context_preference_and_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first planning turn records its validated facts and the trusted identity source."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="initial") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        response = _shopping(
            harness,
            session_id,
            PREFERENCE_MESSAGE,
            constraints=[{"kind": "price_max", "value": "120"}],
        )
        assert response.status_code == 200, response.text
        assert response.json()["recommendations"]

        events, state = _read_state(db, user_key, task_id)
        types = _event_types(events)
        assert types[:3] == [
            "task_context_updated",
            "user_preference_stated",
            "user_constraint_stated",
        ]
        context = _last_event(events, "task_context_updated")
        assert context.payload.request_text == PREFERENCE_MESSAGE
        assert context.payload.intent in {"recommend", "unknown"}
        preference = _last_event(events, "user_preference_stated")
        assert preference.payload.kind is PreferenceKind.COLOR
        assert preference.payload.value == "black"
        assert preference.payload.scope is PreferenceScope.TASK
        constraint = _last_event(events, "user_constraint_stated")
        assert constraint.payload.kind is PreferenceKind.PRICE_MAX
        assert constraint.payload.value == "120"
        assert constraint.payload.scope is PreferenceScope.TASK
        # Every event this integration emits is TASK-scoped: no durable write, ever.
        for event in events:
            assert event.scope is None or event.scope is PreferenceScope.TASK
            assert event.applies_long_term is False
            assert event.user_key == user_key
            assert event.task_id == task_id

        assert _belief(state, "color", "black").lifecycle.value == "explicit"
        assert [c.kind.value for c in state.authorized_hard_constraints] == ["price_max"]

        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["mode"] == "shadow"
        assert diagnostics["error"] is None
        assert diagnostics["state_digest"] == state.state_digest()
        assert diagnostics["event_count"] == state.interaction.event_count == len(events)
        assert diagnostics["turn_identity_source"] == "server_turn_id"
        assert diagnostics["feedback_kind"] == "explicit_statement"
        assert diagnostics["retraction_match"] == {
            "requested": 0,
            "resolved": 0,
            "unmatched_kinds": [],
            "unmatched_value_present": False,
        }
        legacy = diagnostics["legacy_m5"]
        assert legacy["memory_active_count"] == 0
        assert legacy["state_explicit_count"] == 1
        assert legacy["delta"] == 1
        # Snapshot of exactly what was presented after the state facts.
        snapshot = diagnostics["snapshot"]
        assert snapshot is not None
        assert snapshot["candidate_pool_ref"] == state.candidate_pool_ref.ref
        assert snapshot["ranking_ref"] == state.ranking_snapshot_ref.ref
        assert snapshot["snapshot_error"] is None
        # Bounded and non-sensitive.
        blob = _diagnostics_json(diagnostics)
        assert PREFERENCE_MESSAGE not in blob
        assert user_key not in blob
        assert '"120"' not in blob


# =========================================================================== #
# 2. preference-question round-trip
# =========================================================================== #


def test_preference_answer_becomes_task_scoped_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structured answer is an explicit TASK preference (with direction), no schema change."""
    with _harness(
        tmp_path, monkeypatch, mode="shadow", name="roundtrip", rows=CANDIDATE_ROWS[:2]
    ) as (harness, db):
        _equalise_attributes(harness, ("cand-red", "cand-blue"))
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)

        question = _shopping(
            harness,
            session_id,
            "recommend gear under a tight budget",
            k=2,
            sources=["history"],
            preference_discovery=True,
        )
        assert question.status_code == 200, question.text
        question_body = question.json()
        assert question_body["terminal"]["status"] == "waiting_for_user"
        assert question_body["recommendations"] == []
        assert question_body["clarification"]["kind"] == "preference"
        assert question_body["preference"]["attribute"] == "price"

        answer = _shopping(
            harness,
            session_id,
            "lower price",
            k=2,
            sources=["history"],
            clarification_id=question_body["clarification"]["clarification_id"],
            clarification_option="pref:price:lower",
        )
        assert answer.status_code == 200, answer.text
        answer_body = answer.json()
        # No M5 field is added to the wire contract (the schema itself still validates).
        AgentRecommendResponse.model_validate(answer_body)
        assert "state" not in answer_body
        assert "diagnostics" not in answer_body
        assert answer_body["terminal"]["status"] == "finished"
        assert answer_body["recommendations"]
        assert answer_body["preference_outcome"]["selected_option_id"] == "pref:price:lower"

        events, state = _read_state(db, user_key, task_id)
        belief = _belief(state, "feature", "price")
        assert belief.lifecycle.value == "explicit"
        assert belief.origin.value == "user_stated"
        assert belief.direction is not None and belief.direction.value == "minimize"
        # The answer's fact and the presented reranked snapshot are both durable.
        answer_preference = _last_event(events, "user_preference_stated")
        assert answer_preference.payload.kind is PreferenceKind.FEATURE
        assert answer_preference.payload.value == "price"
        assert answer_preference.payload.direction is not None
        assert answer_preference.payload.direction.value == "minimize"

        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["feedback_kind"] == "clarification_option"
        assert diagnostics["snapshot"]["ranking_method"] == "task_preference_rerank"

        # Both projections the next state consumers read see the confirmed preference.
        agent_view = for_agent(state)
        assert [(item.kind.value, item.value) for item in agent_view.preferences] == [
            ("feature", "price")
        ]
        ranker_view = for_ranker(state)
        assert [(item.kind.value, item.direction.value) for item in ranker_view.features] == [
            ("feature", "minimize")
        ]

        # The next shopping turn's recorded state still holds it (nothing resets per turn).
        follow_up = _shopping(harness, session_id, "recommend something else", k=2, sources=["history"])
        assert follow_up.status_code == 200, follow_up.text
        follow_diagnostics = _service(harness).state_diagnostics(task_id)
        assert follow_diagnostics is not None
        assert follow_diagnostics["ranker_feature_count"] >= 1
        _, followed_state = _read_state(db, user_key, task_id)
        assert [(item.kind.value, item.value) for item in for_agent(followed_state).preferences] == [
            ("feature", "price")
        ]


# =========================================================================== #
# 3. ambiguous / declined answer creates no preference
# =========================================================================== #


@pytest.mark.parametrize("option_id", ["pref:none", "pref:skip"])
def test_declined_preference_answer_creates_no_preference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option_id: str,
) -> None:
    """``none``/``skip`` answer only records context: declining is not a preference fact."""
    name = "declined_" + option_id.replace(":", "_")
    with _harness(tmp_path, monkeypatch, mode="shadow", name=name) as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        question = _shopping(
            harness,
            session_id,
            "recommend gear under a tight budget",
            k=3,
            sources=["history"],
            preference_discovery=True,
        ).json()
        clarification_id = question["clarification"]["clarification_id"]
        answer = _shopping(
            harness,
            session_id,
            "skip",
            clarification_id=clarification_id,
            clarification_option=option_id,
        )
        assert answer.status_code == 200, answer.text
        assert answer.json()["terminal"]["status"] == "finished"

        events, state = _read_state(db, user_key, task_id)
        # No preference belief was created by either the question or the decline.
        assert state.preference_beliefs == ()
        assert "user_preference_stated" not in _event_types(events)
        # The answer turn still recorded its task context (and its presented snapshot).
        answer_contexts = [
            event
            for event in events
            if event.event_type.value == "task_context_updated"
        ]
        assert len(answer_contexts) == 2
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["feedback_kind"] == "clarification_option"
        assert diagnostics["legacy_m5"]["state_explicit_count"] == 0
        assert diagnostics["preference_count"] == 0


# =========================================================================== #
# 4. explicit ADD keeps the earlier preference
# =========================================================================== #


def test_explicit_add_does_not_delete_the_earlier_preference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An independent second statement ADDs a belief; it never supersedes the first."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="add") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        first = _shopping(harness, session_id, PREFERENCE_MESSAGE, sources=["history"])
        assert first.status_code == 200, first.text
        second = _shopping(
            harness, session_id, "I also prefer lightweight gear.", sources=["history"]
        )
        assert second.status_code == 200, second.text

        events, state = _read_state(db, user_key, task_id)
        colour = _belief(state, "color", "black")
        lightweight = _belief(state, "free_form_constraint", "lightweight")
        assert colour.status.value == "active" and lightweight.status.value == "active"
        assert not any(
            event.event_type.value == "preference_retracted" for event in events
        )
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["feedback_kind"] == "explicit_statement"
        assert diagnostics["retraction_match"]["requested"] == 0


# =========================================================================== #
# 5. explicit REPLACE correction maps to supersede
# =========================================================================== #


def test_general_replace_without_named_target_supersedes_same_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A general "instead" correction supersedes the old same-kind belief; a later ADD does not."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="replace-general") as (
        harness,
        db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        assert _shopping(harness, session_id, PREFERENCE_MESSAGE, sources=["history"]).status_code == 200
        # ``replaces is None`` here: the extractor recognised a correction but no target value.
        correction = _shopping(
            harness, session_id, "Actually I prefer blue instead.", sources=["history"]
        )
        assert correction.status_code == 200, correction.text
        _, state = _read_state(db, user_key, task_id)
        assert _belief(state, "color", "black").status.value == "superseded"
        assert _belief(state, "color", "blue").status.value == "active"

        # A following independent statement is an ADD: it adds its own value and must not
        # supersede the corrected one.
        add = _shopping(harness, session_id, "I prefer red shoes.", sources=["history"])
        assert add.status_code == 200, add.text
        events, state = _read_state(db, user_key, task_id)
        assert _belief(state, "color", "blue").status.value == "active"
        assert _belief(state, "color", "red").status.value == "active"
        assert _belief(state, "color", "black").status.value == "superseded"
        add_event = _last_event(events, "user_preference_stated")
        assert add_event.payload.value == "red"
        assert add_event.payload.replace_same_kind is False
        assert add_event.payload.replaces_value is None


def test_named_replace_correction_supersedes_the_named_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trusted ``REPLACE`` ("instead of") supersedes the named earlier value only."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="replace") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        assert _shopping(harness, session_id, PREFERENCE_MESSAGE, sources=["history"]).status_code == 200
        correction = _shopping(
            harness, session_id, "I prefer blue instead of black.", sources=["history"]
        )
        assert correction.status_code == 200, correction.text

        events, state = _read_state(db, user_key, task_id)
        old = _belief(state, "color", "black")
        new = _belief(state, "color", "blue")
        assert old.status.value == "superseded"
        assert new.status.value == "active"
        assert new.lifecycle.value == "explicit"
        correction_event = _last_event(events, "user_preference_stated")
        assert correction_event.payload.value == "blue"
        assert correction_event.payload.replaces_value == "black"
        assert [item.value for item in for_agent(state).preferences] == ["blue"]
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["feedback_kind"] == "correction"


# =========================================================================== #
# 6. REMOVE -> PREFERENCE_RETRACTED; value-only unmatched never guessed
# =========================================================================== #


def test_remove_maps_to_retraction_and_unmatched_value_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REMOVE retracts; a value-only removal resolves uniquely or is reported, never guessed."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="remove") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        service = _service(harness)
        assert _shopping(harness, session_id, PREFERENCE_MESSAGE, sources=["history"]).status_code == 200

        # Kind removal: "I don't care about color anymore" retracts the whole slot.
        remove_kind = _shopping(
            harness, session_id, "I don't care about color anymore.", sources=["history"]
        )
        assert remove_kind.status_code == 200, remove_kind.text
        diagnostics = service.state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["feedback_kind"] == "retraction"
        assert diagnostics["retraction_match"] == {
            "requested": 1,
            "resolved": 1,
            "unmatched_kinds": [],
            "unmatched_value_present": False,
        }
        events, state = _read_state(db, user_key, task_id)
        retracted = _belief(state, "color", "black")
        assert retracted.lifecycle.value == "retracted"
        assert retracted.status.value == "removed"
        assert not retracted.is_active
        assert for_agent(state).preferences == ()
        retraction_event = _last_event(events, "preference_retracted")
        assert retraction_event.payload.kind is PreferenceKind.COLOR
        assert retraction_event.payload.value is None

        # An unmatched value-only removal is a bounded disagreement, not a fabricated event.
        assert _shopping(
            harness, session_id, "I also prefer lightweight gear.", sources=["history"]
        ).status_code == 200
        events_before = len(_read_state(db, user_key, task_id)[0])
        unmatched = _shopping(
            harness, session_id, "I don't care about merino anymore.", sources=["history"]
        )
        assert unmatched.status_code == 200, unmatched.text
        diagnostics = service.state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["retraction_match"] == {
            "requested": 1,
            "resolved": 0,
            "unmatched_kinds": [],
            "unmatched_value_present": True,
        }
        events_after, state = _read_state(db, user_key, task_id)
        # Context + snapshot events only: no retraction event was invented.
        assert _event_types(events_after)[events_before:] == [
            "task_context_updated",
            "candidate_pool_referenced",
            "ranking_snapshot_referenced",
        ]
        assert _belief(state, "free_form_constraint", "lightweight").is_active

        # A value-only removal that resolves to exactly one kind is retracted for real.
        resolved = _shopping(
            harness, session_id, "I don't care about lightweight anymore.", sources=["history"]
        )
        assert resolved.status_code == 200, resolved.text
        diagnostics = service.state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["retraction_match"] == {
            "requested": 1,
            "resolved": 1,
            "unmatched_kinds": [],
            "unmatched_value_present": False,
        }
        events, state = _read_state(db, user_key, task_id)
        removal = _last_event(events, "preference_retracted")
        assert removal.payload.kind is PreferenceKind.FREE_FORM_CONSTRAINT
        assert removal.payload.value == "lightweight"
        assert not _belief(state, "free_form_constraint", "lightweight").is_active


# =========================================================================== #
# 7. task-scope feedback never touches LONG_TERM state
# =========================================================================== #


def test_task_scope_feedback_never_touches_long_term_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task preference/retraction is TASK-scoped and leaves a LONG_TERM belief untouched."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="scope") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        # Seed one durable long-term belief (as the read-only M9 adapter would project it).
        _seed_events(
            db,
            [
                preference_stated_event(
                    event_id="m5evt-seed-longterm",
                    user_key=user_key,
                    task_id=task_id,
                    session_id=task_id,
                    scope="long_term",
                    kind="color",
                    value="blue",
                    source="m9_legacy_adapter",
                    source_text="I always prefer blue.",
                )
            ],
        )
        assert _shopping(harness, session_id, PREFERENCE_MESSAGE, sources=["history"]).status_code == 200
        assert _shopping(
            harness, session_id, "I don't care about color anymore.", sources=["history"]
        ).status_code == 200

        events, state = _read_state(db, user_key, task_id)
        adapter_events = [
            event for event in events if event.source == "shopping_state_adapter"
        ]
        assert adapter_events, "the integration recorded the turn"
        assert all(
            event.scope is None or event.scope is PreferenceScope.TASK
            for event in adapter_events
        )
        assert not any(event.applies_long_term for event in adapter_events)
        assert not any(
            event.event_type.value == "preference_retracted"
            and event.scope is PreferenceScope.LONG_TERM
            for event in events
        )
        # The seeded durable belief survives the task-scope retraction.
        long_term = _belief(state, "color", "blue", scope="long_term")
        assert long_term.status.value == "active"
        task_colour = _belief(state, "color", "black")
        assert task_colour.status.value == "removed"
        assert len([b for b in state.preference_beliefs if b.scope is PreferenceScope.LONG_TERM]) == 1


# =========================================================================== #
# 8. turn identity: idempotent retry vs distinct identical turns
# =========================================================================== #


def test_idempotency_key_retry_is_idempotent_and_distinct_turns_do_not_collapse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same client key -> same event identities; identical distinct turns stay distinct."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="identity") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        service = _service(harness)
        body = {
            "message": PREFERENCE_MESSAGE,
            "k": 3,
            "sources": ["history"],
            "constraints": [{"kind": "price_max", "value": "120"}],
            "idempotency_key": "client-turn-1",
        }
        first = harness.client.post(SHOPPING.format(session_id=session_id), json=body)
        assert first.status_code == 200, first.text
        first_events, first_state = _read_state(db, user_key, task_id)
        first_diagnostics = service.state_diagnostics(task_id)
        assert first_diagnostics is not None
        assert first_diagnostics["turn_identity_source"] == "idempotency_key"
        first_digest = first_diagnostics["state_digest"]
        assert first_digest == first_state.state_digest()
        first_ids = [event.event_id for event in first_events]

        retry = harness.client.post(SHOPPING.format(session_id=session_id), json=body)
        assert retry.status_code == 200, retry.text
        retry_events, retry_state = _read_state(db, user_key, task_id)
        retry_diagnostics = service.state_diagnostics(task_id)
        assert retry_diagnostics is not None
        # The retry reuses the same deterministic event ids: the append-only store no-ops.
        assert [event.event_id for event in retry_events] == first_ids
        assert retry_state.state_digest() == first_digest
        assert retry_diagnostics["state_digest"] == first_digest
        assert retry_diagnostics["event_count"] == first_diagnostics["event_count"]
        assert retry_diagnostics["events"] == first_diagnostics["events"]
        assert retry_diagnostics["turn_count"] == first_diagnostics["turn_count"] + 1

        # Two identical messages with no key are two distinct server turns: new event ids.
        with _harness(
            tmp_path, monkeypatch, mode="shadow", name="identity-distinct"
        ) as (distinct, distinct_db):
            distinct_session = _create_session(distinct)
            distinct_user, distinct_task = _identity(distinct, distinct_session)
            plain = {"message": PREFERENCE_MESSAGE, "k": 3, "sources": ["history"]}
            assert distinct.client.post(
                SHOPPING.format(session_id=distinct_session), json=plain
            ).status_code == 200
            assert distinct.client.post(
                SHOPPING.format(session_id=distinct_session), json=plain
            ).status_code == 200
            events, _state = _read_state(distinct_db, distinct_user, distinct_task)
            contexts = [
                event
                for event in events
                if event.event_type.value == "task_context_updated"
            ]
            assert len(contexts) == 2
            assert len({event.event_id for event in contexts}) == 2
            assert _service(distinct).state_diagnostics(distinct_task)[
                "turn_identity_source"
            ] == "server_turn_id"
            # Each turn's full batch (context + preference + snapshot pair) was appended.
            assert len(events) == 8


# =========================================================================== #
# 9. cross-user / cross-task isolation
# =========================================================================== #


def test_state_is_isolated_between_users_and_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two demo sessions never share a decision state, diagnostics or event stream."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="isolation") as (harness, db):
        session_a = _create_session(harness, PROFILE_A)
        session_b = _create_session(harness, PROFILE_B)
        user_a, task_a = _identity(harness, session_a)
        user_b, task_b = _identity(harness, session_b)
        assert user_a != user_b and task_a != task_b
        service = _service(harness)

        assert _shopping(
            harness, session_a, PREFERENCE_MESSAGE, sources=["history"]
        ).status_code == 200
        assert service.state_diagnostics(task_a) is not None
        assert service.state_diagnostics(task_b) is None
        assert _read_state(db, user_b, task_b)[1].interaction.event_count == 0

        assert _shopping(
            harness, session_b, "I prefer blue shoes.", sources=["history"]
        ).status_code == 200
        events_a, state_a = _read_state(db, user_a, task_a)
        events_b, state_b = _read_state(db, user_b, task_b)
        for event in events_a:
            assert event.user_key == user_a and event.task_id == task_a
        for event in events_b:
            assert event.user_key == user_b and event.task_id == task_b
        assert {b.value for b in state_a.preference_beliefs if b.scope is PreferenceScope.TASK} == {"black"}
        assert {b.value for b in state_b.preference_beliefs if b.scope is PreferenceScope.TASK} == {"blue"}
        assert state_a.state_digest() != state_b.state_digest()
        diagnostics_a = service.state_diagnostics(task_a)
        diagnostics_b = service.state_diagnostics(task_b)
        assert diagnostics_a is not None and diagnostics_b is not None
        assert diagnostics_a["state_digest"] == state_a.state_digest()
        assert diagnostics_b["state_digest"] == state_b.state_digest()
        # A cross-read of the other user's task is empty.
        store = SQLiteEventStore(db)
        try:
            assert store.get_events(task_b, user_a) == ()
            assert store.get_events(task_a, user_b) == ()
        finally:
            store.close()


# =========================================================================== #
# 10. replay after reopen equals the diagnostics digest
# =========================================================================== #


def test_replay_after_reopen_equals_diagnostics_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh manager over the same database reproduces the digest and event count."""
    database: Path
    with _harness(tmp_path, monkeypatch, mode="shadow", name="replay") as (harness, db):
        database = db
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        response = _shopping(
            harness,
            session_id,
            PREFERENCE_MESSAGE,
            constraints=[{"kind": "price_max", "value": "120"}],
        )
        assert response.status_code == 200, response.text
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        expected_digest = diagnostics["state_digest"]
        expected_count = diagnostics["event_count"]
        # Release the live connection before reopening the file.
        _service(harness)._state_integration.close()

    events, state = _read_state(database, user_key, task_id)
    assert state.state_digest() == expected_digest
    assert state.interaction.event_count == expected_count == len(events)
    assert for_agent(state).projection_version == 1
    assert for_ranker(state).ranker_signal_version == 1
    assert for_constraints(state).constraints_version == 1
    # A second replay is byte-identical.
    _, again = _read_state(database, user_key, task_id)
    assert again.state_digest() == expected_digest


# =========================================================================== #
# 11 + 12. OFF inert; SHADOW user-visible parity
# =========================================================================== #


def test_off_default_is_inert_and_creates_no_state_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default (unset) mode opens nothing and reports no diagnostics."""
    with _harness(tmp_path, monkeypatch, mode=None, name="inert") as (harness, db):
        session_id = _create_session(harness)
        _, task_id = _identity(harness, session_id)
        response = _shopping(harness, session_id, PREFERENCE_MESSAGE)
        assert response.status_code == 200, response.text
        service = _service(harness)
        assert service.state_mode == "off"
        assert service.state_diagnostics(task_id) is None
        assert not db.exists(), "OFF opened the decision-state database"


def test_shadow_user_visible_response_equals_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SHADOW records the state but the presented answer is identical to OFF."""
    request = dict(
        message=PREFERENCE_MESSAGE,
        constraints=[{"kind": "price_max", "value": "120"}],
    )
    with _harness(tmp_path, monkeypatch, mode="off", name="parity-off") as (off, _off_db):
        off_session = _create_session(off)
        off_response = _shopping(off, off_session, **request)
        assert off_response.status_code == 200, off_response.text
        off_body = _normalised(off_response.json())

    with _harness(tmp_path, monkeypatch, mode="shadow", name="parity-shadow") as (
        shadow,
        shadow_db,
    ):
        shadow_session = _create_session(shadow)
        shadow_response = _shopping(shadow, shadow_session, **request)
        assert shadow_response.status_code == 200, shadow_response.text
        assert _normalised(shadow_response.json()) == off_body
        assert _service(shadow).state_mode == "shadow"
        assert shadow_db.exists()
        body = shadow_response.json()
        assert "state" not in body and "diagnostics" not in body and "snapshot" not in body


# =========================================================================== #
# 13. APPLY: effective intent view + enforcement
# =========================================================================== #


def test_apply_enforces_seeded_constraint_and_shows_true_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """APPLY enforces the M5 constraint and never presents remembered/state facts as user_stated."""
    with _harness(tmp_path, monkeypatch, mode="apply_constraints", name="apply") as (
        harness,
        db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        # A remembered M9 constraint (legacy) and an authorized M5 constraint (this task).
        harness.runtime.memory_service.process_turn(
            user_key=user_key,
            user_message="I need it under $25",
            turn_id="legacy-seed",
        )
        _seed_events(
            db,
            [
                constraint_event(
                    event_id="m5evt-seed-constraint",
                    user_key=user_key,
                    task_id=task_id,
                    session_id=task_id,
                    kind="color",
                    value="red",
                )
            ],
        )

        response = _shopping(
            harness,
            session_id,
            NEUTRAL_MESSAGE,
            k=4,
            sources=["history", "catalog_search"],
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red",
        ]
        assert body["eligibility"]["requirements"] == [
            {"kind": "price_max", "expected": "25"},
            {"kind": "color", "expected": "red"},
        ]
        hard = {
            (entry["kind"], entry["value"]): entry for entry in body["intent"]["hard"]
        }
        assert hard[("price_max", "25")]["origin"] == "validated_existing"
        assert hard[("color", "red")]["origin"] == "decision_state"
        assert hard[("color", "red")]["source_turn_id"] == "decision_state"
        assert all(
            entry["origin"] != "user_stated" for entry in body["intent"]["hard"]
        ), "a remembered/state constraint must not be presented as user_stated"

        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["legacy_m5"] == {
            "memory_active_count": 1,
            "state_explicit_count": 0,
            "delta": -1,
        }
        # This turn's trusted constraint was recorded as well, so the task state now holds
        # both the remembered one and the seeded M5-only one.
        assert diagnostics["hard_constraint_count"] == 2
        assert diagnostics["constraint_kinds"] == ["color", "price_max"]
        assert diagnostics["constraint_match"] == {
            "matched": 1,
            "missing_from_state": [],
            "extra_in_state": ["color"],
        }
        _, state = _read_state(db, user_key, task_id)
        assert [c.value for c in state.authorized_hard_constraints] == ["red", "25"]


# =========================================================================== #
# 14. inference never hard-filters
# =========================================================================== #


def test_inferred_hypothesis_never_becomes_a_hard_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inferred hypothesis is counted, excluded from constraints, and filters nothing."""
    with _harness(tmp_path, monkeypatch, mode="apply_constraints", name="hypothesis") as (
        harness,
        db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        _seed_events(
            db,
            [
                hypothesis_event(
                    event_id="m5evt-seed-hypothesis",
                    user_key=user_key,
                    task_id=task_id,
                    session_id=task_id,
                    kind="price_max",
                    value="1",
                )
            ],
        )
        response = _shopping(
            harness, session_id, NEUTRAL_MESSAGE, k=4, sources=["history"]
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["parent_asin"] for item in body["recommendations"]] == list(
            INITIAL_ORDER
        )
        assert body["eligibility"]["evaluated"] is False
        assert body["intent"]["hard"] == []

        _, state = _read_state(db, user_key, task_id)
        assert len(state.inferred_hypotheses) == 1
        assert state.authorized_hard_constraints == ()
        assert for_constraints(state).constraints == ()
        assert for_constraints(state).excluded_hypothesis_count >= 1
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["soft_hypothesis_count"] == 1
        assert diagnostics["hard_constraint_count"] == 0
        assert diagnostics["constraint_kinds"] == []


# =========================================================================== #
# 15. UNKNOWN stays UNKNOWN
# =========================================================================== #


def test_unknown_constraint_stays_unknown_not_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A free-form (UNVERIFIABLE) constraint yields UNKNOWN verdicts, never exclusions."""
    with _harness(tmp_path, monkeypatch, mode="apply_constraints", name="unknown") as (
        harness,
        db,
    ):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        _seed_events(
            db,
            [
                constraint_event(
                    event_id="m5evt-seed-unknown",
                    user_key=user_key,
                    task_id=task_id,
                    session_id=task_id,
                    kind="free_form_constraint",
                    value="extra large",
                )
            ],
        )
        response = _shopping(
            harness, session_id, NEUTRAL_MESSAGE, k=4, sources=["history"]
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["recommendations"] == []
        assert body["terminal"]["status"] == "waiting_for_user"
        assert body["clarification"]["kind"] == "unverified"
        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert eligibility["ineligible_count"] == 0
        assert eligibility["unresolved_count"] == len(eligibility["assessments"]) == 4
        for assessment in eligibility["assessments"]:
            assert assessment["eligibility"] == "unresolved"
            assert all(report["verdict"] == "unknown" for report in assessment["reports"])
        # The enforced state constraint is visible with its true provenance.
        assert len(body["intent"]["hard"]) == 1
        assert body["intent"]["hard"][0]["kind"] == "free_form_constraint"
        assert body["intent"]["hard"][0]["origin"] == "decision_state"


# =========================================================================== #
# 16. trusted candidate / ranking snapshot references
# =========================================================================== #


def test_presented_response_records_trusted_snapshot_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opaque candidate/ranking refs are recorded after presentation; no identity/score."""
    with _harness(tmp_path, monkeypatch, mode="shadow", name="snapshot") as (harness, db):
        session_id = _create_session(harness)
        user_key, task_id = _identity(harness, session_id)
        response = _shopping(
            harness, session_id, NEUTRAL_MESSAGE, k=3, sources=["history"]
        )
        assert response.status_code == 200, response.text
        body = response.json()
        events, state = _read_state(db, user_key, task_id)

        pool_event = _last_event(events, "candidate_pool_referenced")
        ranking_event = _last_event(events, "ranking_snapshot_referenced")
        for event in (pool_event, ranking_event):
            assert event.actor.value == "trusted_component"
            assert event.scope is None
            assert _SHA256.match(event.payload.digest)
            assert event.payload.snapshot_schema_version == SNAPSHOT_SCHEMA_VERSION == 1
            assert event.payload.candidate_count >= 2
            assert "cand-" not in event.event_id
            assert "cand-" not in event.payload.ref
            assert "cand-" not in event.payload.digest
        assert pool_event.source == "shopping_candidate_pool"
        assert ranking_event.source == "shopping_presented_order"
        assert pool_event.payload.source == "shopping_candidate_pool"
        assert ranking_event.payload.source == "shopping_presented_order"
        assert _CANDIDATE_REF.match(pool_event.payload.ref)
        assert _RANKING_REF.match(ranking_event.payload.ref)
        assert ranking_event.payload.method == "presented_order"
        assert pool_event.payload.candidate_count == len(body["grounded"]["candidates"])
        assert ranking_event.payload.candidate_count == len(body["recommendations"])
        # The state's read-only refs are exactly the recorded references and project forward.
        assert state.candidate_pool_ref.ref == pool_event.payload.ref
        assert state.ranking_snapshot_ref.ref == ranking_event.payload.ref
        assert for_agent(state).candidate_pool_ref == pool_event.payload.ref
        assert for_agent(state).ranking_snapshot_ref == ranking_event.payload.ref
        assert for_ranker(state).candidate_pool_ref == pool_event.payload.ref
        assert for_ranker(state).ranking_snapshot_ref == ranking_event.payload.ref

        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        snapshot = diagnostics["snapshot"]
        assert snapshot == {
            "candidate_pool_ref": pool_event.payload.ref,
            "candidate_count": pool_event.payload.candidate_count,
            "candidate_digest_present": True,
            "ranking_ref": ranking_event.payload.ref,
            "ranking_count": ranking_event.payload.candidate_count,
            "ranking_method": "presented_order",
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "state_digest": state.state_digest(),
            "event_count": state.interaction.event_count,
            "snapshot_error": None,
        }
        assert diagnostics["state_digest"] == state.state_digest()
        assert diagnostics["event_count"] == state.interaction.event_count == len(events)


def test_snapshot_input_validation_and_mode_policy(tmp_path: Path) -> None:
    """A malformed snapshot input is a bounded SHADOW diagnostic and a fail-closed APPLY error."""
    database = tmp_path / "snapshot-validation.sqlite3"
    identity = dict(
        user_key="user-snapshot",
        session_id="task-snapshot",
        task_id="task-snapshot",
        turn_key="turn-snapshot",
    )
    shadow = ShoppingStateIntegration(mode=StateMode.SHADOW, database=database)
    try:
        # No empty snapshot is fabricated.
        assert (
            shadow.record_candidate_snapshot(
                **identity,
                candidate_identities=(),
                ranking_identities=(),
                ranking_method="presented_order",
            )
            is None
        )
        assert shadow.diagnostics("task-snapshot") is None

        shadow.integrate_turn(
            **identity,
            message=NEUTRAL_MESSAGE,
            intent_label="recommend",
            preferences=(),
            constraints=(),
            trusted_constraints=(),
        )
        # A repeated identity in the presented order is malformed trusted provenance.
        outcome = shadow.record_candidate_snapshot(
            **identity,
            candidate_identities=("cand-red", "cand-blue"),
            ranking_identities=("cand-red", "cand-red"),
            ranking_method="presented_order",
        )
        assert outcome is None  # SHADOW never raises
        record = shadow.diagnostics("task-snapshot")
        assert record is not None
        assert record["snapshot"]["snapshot_error"] == "state_event_invalid"
        assert record["event_count"] == 1  # only the task-context event: nothing appended

        # A blank identity is malformed too.
        assert (
            shadow.record_candidate_snapshot(
                **identity,
                candidate_identities=("cand-red", "  "),
                ranking_identities=("cand-red",),
                ranking_method="presented_order",
            )
            is None
        )
        # And a blank method label.
        assert (
            shadow.record_candidate_snapshot(
                **identity,
                candidate_identities=("cand-red",),
                ranking_identities=("cand-red",),
                ranking_method="   ",
            )
            is None
        )
        assert shadow.diagnostics("task-snapshot")["snapshot"]["snapshot_error"] == (
            "state_event_invalid"
        )
    finally:
        shadow.close()

    apply_database = tmp_path / "snapshot-validation-apply.sqlite3"
    applying = ShoppingStateIntegration(
        mode=StateMode.APPLY_CONSTRAINTS, database=apply_database
    )
    try:
        applying.integrate_turn(
            **identity,
            message=NEUTRAL_MESSAGE,
            intent_label="recommend",
            preferences=(),
            constraints=(),
            trusted_constraints=(),
        )
        with pytest.raises(StateIntegrationError) as raised:
            applying.record_candidate_snapshot(
                **identity,
                candidate_identities=("cand-red",),
                ranking_identities=("cand-red", "cand-red"),
                ranking_method="presented_order",
            )
        assert raised.value.code == "state_event_invalid"
        record = applying.diagnostics("task-snapshot")
        assert record is not None
        assert record["snapshot"]["snapshot_error"] == "state_event_invalid"
    finally:
        applying.close()


# =========================================================================== #
# 17. invalid event is not appended (SHADOW vs APPLY)
# =========================================================================== #


def test_invalid_event_is_not_appended_and_policy_follows_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fact M5 rejects is a SHADOW diagnostic (answer unchanged) and an APPLY 502."""
    with _harness(tmp_path, monkeypatch, mode="off", name="invalid-off") as (off, _off_db):
        off_session = _create_session(off)
        off.runtime.memory_service._extractor = _InvalidExtractor()
        off_body = _normalised(_shopping(off, off_session, NEUTRAL_MESSAGE, sources=["history"]).json())

    with _harness(
        tmp_path, monkeypatch, mode="shadow", name="invalid-shadow"
    ) as (shadow, shadow_db):
        shadow_session = _create_session(shadow)
        shadow_user, shadow_task = _identity(shadow, shadow_session)
        shadow.runtime.memory_service._extractor = _InvalidExtractor()
        response = _shopping(shadow, shadow_session, NEUTRAL_MESSAGE, sources=["history"])
        assert response.status_code == 200, response.text
        assert _normalised(response.json()) == off_body
        diagnostics = _service(shadow).state_diagnostics(shadow_task)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_event_invalid"
        assert diagnostics["events"] == 0
        assert diagnostics["event_count"] == 0
        assert diagnostics["snapshot"] is None
        # The rejected event was never appended and no snapshot was attached to the failed turn.
        assert _stored_rows(shadow_db, shadow_user) == 0

    with _harness(
        tmp_path, monkeypatch, mode="apply_constraints", name="invalid-apply"
    ) as (applying, applying_db):
        applying_session = _create_session(applying)
        applying_user, applying_task = _identity(applying, applying_session)
        applying.runtime.memory_service._extractor = _InvalidExtractor()
        response = _shopping(applying, applying_session, NEUTRAL_MESSAGE, sources=["history"])
        assert response.status_code == 502, response.text
        assert response.json()["error"] in {"agent_failed", "state_integration_failed"}
        diagnostics = _service(applying).state_diagnostics(applying_task)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_event_invalid"
        assert _stored_rows(applying_db, applying_user) == 0


# =========================================================================== #
# 18. store failure modes
# =========================================================================== #


def test_store_failure_is_shadow_diagnostic_and_apply_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unopenable state store cannot change SHADOW's answer; APPLY fails closed."""
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    blocked_db = blocker / "state.sqlite3"

    with _harness(tmp_path, monkeypatch, mode="shadow", name="store-shadow") as (
        shadow,
        _shadow_db,
    ):
        # Point the composed integration at the unopenable path: the harness env is set from
        # the helper, so replace it before the store is opened.
        shadow.runtime.agent_service()._state_integration._database = blocked_db
        session_id = _create_session(shadow)
        _, task_id = _identity(shadow, session_id)
        response = _shopping(shadow, session_id, NEUTRAL_MESSAGE, sources=["history"])
        assert response.status_code == 200, response.text
        diagnostics = _service(shadow).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_store_unavailable"
        assert diagnostics["state_digest"] is None
        assert diagnostics["events"] == 0
        assert diagnostics["snapshot"] is None

    with _harness(
        tmp_path, monkeypatch, mode="apply_constraints", name="store-apply"
    ) as (applying, _apply_db):
        applying.runtime.agent_service()._state_integration._database = blocked_db
        session_id = _create_session(applying)
        _, task_id = _identity(applying, session_id)
        response = _shopping(applying, session_id, NEUTRAL_MESSAGE, sources=["history"])
        assert response.status_code == 502, response.text
        assert response.json()["error"] in {"agent_failed", "state_integration_failed"}
        diagnostics = _service(applying).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_store_unavailable"
