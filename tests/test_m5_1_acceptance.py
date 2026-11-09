"""M5.1 independent black-box acceptance suite (requirements A-E).

The suite drives the demo service over HTTP with the real offline harness
(:func:`tests.demo_fixture.build_harness`), flips ``AGENTRECX_STATE_MODE`` through the
environment, and inspects the decision-state database and the documented diagnostics
surface directly.  It never imports or patches the integration's internals, so a passing
run is evidence about the service contract rather than about a helper function.

Requirement map
---------------
A. :func:`test_off_mode_matches_explicit_off_and_writes_no_state` — OFF is the default and
   explicit ``off`` is identical; no decision-state database is opened or written.
B. :func:`test_shadow_mode_is_user_visible_parity_and_records_state` — SHADOW records
   validated events/projections and a state digest, is user-visible identical to OFF, and
   :func:`test_stateless_agent_route_under_shadow_writes_no_state_and_matches_off` — the
   stateless agent route has no stable task identity, so SHADOW writes nothing there.
C. :func:`test_apply_mode_uses_seeded_explicit_constraint_to_narrow_eligibility`,
   :func:`test_apply_mode_inferred_hypothesis_never_filters_candidates`,
   :func:`test_apply_mode_unverifiable_constraint_stays_unknown_not_ineligible` —
   experimental APPLY_CONSTRAINTS behaviour: authorized constraints narrow eligibility
   through the existing evaluator, inference never hard-filters, and ``UNKNOWN`` stays
   ``UNKNOWN`` rather than becoming an ineligibility.
D. :func:`test_shadow_duplicate_turn_is_idempotent` — a duplicate logical turn re-emits
   the same event ids, so the store no-ops and replay stays valid.
E. :func:`test_state_replays_after_restart_from_the_diagnostics_digest`,
   :func:`test_state_is_isolated_between_sessions`,
   :func:`test_shadow_store_unavailable_is_diagnostics_only`,
   :func:`test_apply_store_unavailable_fails_closed`,
   :func:`test_invalid_state_mode_fails_closed_at_composition`,
   :func:`test_legacy_memory_disagreement_is_diagnostics_only_and_schema_unchanged` —
   regression/robustness: replay after a restart, user isolation, store-unavailable policy
   per mode, invalid mode, and a bounded mismatch report when the accepted M9 memory
   disagrees with the M5 state (with no M9 schema change).

Normalisation for parity comparisons
------------------------------------
Two independently created demo sessions cannot have byte-identical responses: the
existing response contract carries controller-generated opaque identities (a per-run
``terminal.run_id`` / ``observation.run_ref``) and server-owned turn ids derived from the
session's random UUID, plus wall-clock ``timing``.  Those are identities and clocks, not
behaviour, so :func:`normalised_response` drops ``timing`` and masks exactly those
identity values; every other field must match exactly.  The one known consequence of
content-addressed turn identity (two identical logical turns collapse) is asserted
explicitly in the idempotency test instead of being hidden by the normaliser.
"""

from __future__ import annotations

import contextlib
import copy
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.state.event_store import SQLiteEventStore  # noqa: E402
from recommendation.state.schemas import PreferenceScope  # noqa: E402
from recommendation.state.state_manager import StateManager  # noqa: E402
from recommendation.state.state_projector import (  # noqa: E402
    for_agent,
    for_constraints,
    for_ranker,
)
from tests.agent_reranking_fixture import CANDIDATE_PRICES, INITIAL_ORDER  # noqa: E402
from tests.demo_fixture import PROFILE_A, PROFILE_B, build_harness, close_harness  # noqa: E402
from tests.state_fixtures import constraint_event, hypothesis_event  # noqa: E402

#: Environment switches under test (documented M5.1 configuration surface).
STATE_MODE_ENV = "AGENTRECX_STATE_MODE"
STATE_DB_ENV = "AGENTRECX_STATE_DB"

#: The fixed shopping request used across modes.  It yields exactly one explicit
#: preference (the accepted M9 extractor: colour "black") and no hard constraint, so a
#: state event count is stable and a body constraint is the only hard signal.
FIXED_MESSAGE = "Recommend some hiking gear. I prefer black."
FIXED_K = 4
FIXED_SOURCES = ("history", "catalog_search")

#: A trusted hard constraint exercised in the parity and idempotency paths: only
#: ``cand-black`` ($15.00) satisfies it.
FIXED_CONSTRAINTS = ({"kind": "price_max", "value": "20"},)

#: Server-owned turn ids look like ``demo-session:<uuid>:<hex>``; they are session
#: identities, not behaviour, so the parity normaliser masks them.
_SESSION_TURN_ID = re.compile(r"^demo-session:[0-9a-f-]{36}:[0-9a-f]{1,32}$")

#: Keys holding controller-generated run identities (random per run by design).
_MASKED_KEYS = frozenset({"timing", "run_id", "run_ref"})

#: Diagnostics keys the M5.1 contract promises.  The record is bounded and non-sensitive:
#: counts, versions, kind labels and one full state digest.
DIAGNOSTIC_KEYS = frozenset(
    {
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
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# harness / request helpers
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _harness(tmp_path: Path, monkeypatch, mode: str | None, label: str):
    """Build one demo harness with the given state mode; yield ``(harness, state_db)``.

    ``mode=None`` means the environment variable is *unset* (the default-OFF path).
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


def _create_session(harness, profile: str = PROFILE_A) -> str:
    """Create a demo session over HTTP and return its server-owned id."""
    response = harness.client.post("/v1/demo/sessions", json={"profile_id": profile})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _session(harness, session_id: str):
    """Return the live demo session object (server-owned identities)."""
    return harness.runtime.sessions.get(session_id)


def _service(harness):
    """Return the composed agent service (the M5.1 diagnostics surface)."""
    return harness.runtime.agent_service()


def _shopping_turn(
    harness,
    session_id: str,
    *,
    message: str = FIXED_MESSAGE,
    k: int = FIXED_K,
    sources=FIXED_SOURCES,
    constraints=FIXED_CONSTRAINTS,
    idempotency_key: str | None = None,
):
    """POST one shopping turn and return the raw response.

    ``idempotency_key`` is the M5.2 client-supplied retry identity; when it is omitted the
    server-owned turn id identifies the turn (so two identical messages are distinct).
    """
    body = {"message": message, "k": k, "sources": list(sources)}
    if constraints is not None:
        body["constraints"] = list(constraints)
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    return harness.client.post(
        f"/v1/demo/sessions/{session_id}/agent/shopping", json=body
    )


def _stateless_turn(harness, *, message: str = FIXED_MESSAGE, k: int = FIXED_K):
    """POST one stateless agent turn (no session, no stable task identity)."""
    return harness.client.post(
        "/v1/demo/agent/recommend",
        json={
            "message": message,
            "history": list(INITIAL_ORDER),
            "k": k,
            "sources": list(FIXED_SOURCES),
        },
    )


# --------------------------------------------------------------------------- #
# response normalisation and comparison
# --------------------------------------------------------------------------- #


def _normalise(value):
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


def normalised_response(body: dict) -> dict:
    """Return the response with only non-behavioural fields normalised.

    Dropped: top-level ``timing`` (wall clock).  Masked: ``run_id`` / ``run_ref`` (random
    per-run controller identities) and session-derived turn ids.  Nothing else is
    normalised: a behaviour change must still fail the comparison.
    """
    data = copy.deepcopy(body)
    data.pop("timing", None)
    return _normalise(data)


def _diff(left, right, path: str = "$") -> list[tuple[str, object, object]]:
    """Return the differing paths between two JSON-like values."""
    if type(left) is not type(right):
        return [(path, left, right)]
    if isinstance(left, dict):
        differences: list[tuple[str, object, object]] = []
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


def assert_response_parity(expected: dict, actual: dict, *, context: str) -> None:
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


def _seed_state(db_path: Path, user_key: str, task_id: str, events) -> None:
    """Append fixture events for one task through a real ``StateManager``."""
    with _open_state(db_path) as (_store, manager):
        manager.load_or_create(user_key, session_id=task_id, task_id=task_id)
        for event in events:
            manager.apply_event(event)


def _read_state(db_path: Path, user_key: str, task_id: str) -> dict:
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


def _event_type_counts(events) -> dict[str, int]:
    """Count stored events by type."""
    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type.value] = counts.get(event.event_type.value, 0) + 1
    return counts


def _assert_diagnostics(diagnostics: dict, *, mode: str) -> None:
    """Assert the documented diagnostics shape and its non-sensitivity bounds."""
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
    # Bounded and non-sensitive: no raw message, no user key, no unbounded blob.
    serialised = json.dumps(diagnostics, sort_keys=True)
    assert FIXED_MESSAGE not in serialised
    assert "demo-session:" not in serialised
    assert len(serialised) < 4096, "diagnostics record is not bounded"
    if diagnostics["state_digest"] is not None:
        assert _HEX64.match(diagnostics["state_digest"]), "state digest is a SHA-256 hex digest"
    assert re.fullmatch(r"[0-9a-f]{12}", diagnostics["turn_key_digest"])


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


# --------------------------------------------------------------------------- #
# A. OFF-mode parity and no state writes
# --------------------------------------------------------------------------- #


def test_off_mode_matches_explicit_off_and_writes_no_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement A: default OFF == explicit ``off``; no state DB is opened or written."""
    with _harness(tmp_path, monkeypatch, None, "off-default") as (default, default_db):
        default_session = _create_session(default)
        default_response = _shopping_turn(default, default_session)
        assert default_response.status_code == 200, default_response.text
        assert _service(default).state_mode == "off"
        assert _service(default).state_diagnostics(default_session) is None
        assert not default_db.exists(), "OFF opened/created the decision-state database"

    with _harness(tmp_path, monkeypatch, "off", "off-explicit") as (explicit, explicit_db):
        explicit_session = _create_session(explicit)
        explicit_response = _shopping_turn(explicit, explicit_session)
        assert explicit_response.status_code == 200, explicit_response.text
        assert _service(explicit).state_mode == "off"
        assert not explicit_db.exists(), "explicit off opened/created the state database"

    default_body = default_response.json()
    explicit_body = explicit_response.json()
    assert_response_parity(default_body, explicit_body, context="OFF vs explicit off")

    # The named semantic blocks the requirement lists, asserted explicitly.
    default_recommendations = [
        (item["parent_asin"], item["rank"], item["score"], item["score_kind"])
        for item in default_body["recommendations"]
    ]
    assert default_recommendations == [
        ("cand-black", 1, 0.7, "sasrec_logit"),
    ]
    assert default_body["message"] == explicit_body["message"]
    assert default_body["grounded"] == explicit_body["grounded"]
    assert default_body["eligibility"] == explicit_body["eligibility"]
    assert normalised_response(default_body["terminal"]) == normalised_response(
        explicit_body["terminal"]
    )
    assert default_body["terminal"]["status"] == "finished"


# --------------------------------------------------------------------------- #
# B. SHADOW-mode parity + recorded state/projections
# --------------------------------------------------------------------------- #


def test_shadow_mode_is_user_visible_parity_and_records_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement B: SHADOW records events/projections/digest, response identical to OFF."""
    with _harness(tmp_path, monkeypatch, "off", "shadow-off-reference") as (
        reference,
        _reference_db,
    ):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(reference, reference_session)
        assert reference_response.status_code == 200, reference_response.text

    with _harness(tmp_path, monkeypatch, "shadow", "shadow-on") as (shadow, state_db):
        session_id = _create_session(shadow)
        service = _service(shadow)
        assert service.state_mode == "shadow"

        response = _shopping_turn(shadow, session_id)
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="SHADOW vs OFF"
        )

        assert state_db.exists(), "SHADOW must persist the decision state"
        user_key = _session(shadow, session_id).user_key
        recorded = _read_state(state_db, user_key, session_id)
        events = recorded["events"]
        counts = _event_type_counts(events)
        # M5.2 superset: the same explicit facts plus the two trusted snapshot references a
        # **presented** response records (candidate pool + presented ranking).  The fact
        # events are unchanged and remain asserted below.
        assert counts == {
            "task_context_updated": 1,
            "user_preference_stated": 1,
            "user_constraint_stated": 1,
            "candidate_pool_referenced": 1,
            "ranking_snapshot_referenced": 1,
        }
        assert {
            event.event_type.value for event in events
        } <= {
            "task_context_updated",
            "user_preference_stated",
            "user_constraint_stated",
            "candidate_pool_referenced",
            "ranking_snapshot_referenced",
        }
        for event in events:
            assert event.user_key == user_key
            assert event.task_id == session_id
            # All M5.1 events are TASK-scoped; a shopping turn never writes durable state.
            assert event.scope is None or event.scope is PreferenceScope.TASK
            assert event.applies_long_term is False

        state = recorded["state"]
        assert state.task_context.intent in {"unknown", "recommend"}
        assert state.task_context.request_text == FIXED_MESSAGE
        assert [(item.kind.value, item.value) for item in state.active_preference_beliefs] == [
            ("color", "black")
        ]
        assert [(item.kind.value, item.value) for item in state.authorized_hard_constraints] == [
            ("price_max", "20")
        ]

        agent_view = recorded["agent"]
        ranker_view = recorded["ranker"]
        constraint_view = recorded["constraints"]
        assert agent_view.projection_version == 1
        assert ranker_view.ranker_signal_version == 1
        assert constraint_view.constraints_version == 1
        assert [item.value for item in agent_view.preferences] == ["black"]
        assert [item.value for item in constraint_view.constraints] == ["20"]
        assert len(ranker_view.features) == 1
        assert ranker_view.features[0].strength_rank == 3
        assert ranker_view.features[0].is_soft is False

        diagnostics = service.state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        assert diagnostics["state_version"] == state.state_version
        assert diagnostics["state_digest"] == recorded["digest"]
        assert diagnostics["state_digest"] == state.state_digest()
        assert diagnostics["event_count"] == len(events) == state.interaction.event_count
        assert diagnostics["events"] == len(events)
        assert diagnostics["preference_count"] == len(agent_view.preferences)
        assert diagnostics["soft_hypothesis_count"] == 0
        assert diagnostics["hard_constraint_count"] == len(constraint_view.constraints)
        assert diagnostics["constraint_kinds"] == ["price_max"]
        assert diagnostics["agent_projection_version"] == agent_view.projection_version
        assert diagnostics["ranker_signal_version"] == ranker_view.ranker_signal_version
        assert diagnostics["constraints_version"] == constraint_view.constraints_version
        assert diagnostics["excluded_hypothesis_count"] == 0
        assert diagnostics["trusted_constraint_count"] == 1
        assert diagnostics["constraint_match"] == {
            "matched": 1,
            "missing_from_state": [],
            "extra_in_state": [],
        }

        # No M5.1 field is added to the wire contract.
        assert "state" not in response.json()
        assert "diagnostics" not in response.json()


def test_stateless_agent_route_under_shadow_writes_no_state_and_matches_off(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement B: the stateless route has no task identity, so SHADOW is inert there."""
    with _harness(tmp_path, monkeypatch, "off", "stateless-off") as (reference, _ref_db):
        reference_response = _stateless_turn(reference)
        assert reference_response.status_code == 200, reference_response.text

    with _harness(tmp_path, monkeypatch, "shadow", "stateless-shadow") as (shadow, state_db):
        assert _service(shadow).state_mode == "shadow"
        response = _stateless_turn(shadow)
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="shadow stateless vs OFF"
        )
        assert not state_db.exists(), "a stateless turn must not create decision-state rows"


# --------------------------------------------------------------------------- #
# C. Experimental constraint behaviour (APPLY_CONSTRAINTS)
# --------------------------------------------------------------------------- #


def test_apply_mode_uses_seeded_explicit_constraint_to_narrow_eligibility(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement C: an authorized explicit constraint narrows the eligible candidates."""
    with _harness(tmp_path, monkeypatch, "off", "apply-off-reference") as (
        reference,
        _reference_db,
    ):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(reference, reference_session, constraints=())
        assert reference_response.status_code == 200, reference_response.text
        baseline_identities = [
            item["parent_asin"] for item in reference_response.json()["recommendations"]
        ]
        assert baseline_identities == ["cand-red", "cand-blue", "cand-black", "cand-green"]

    with _harness(tmp_path, monkeypatch, "apply_constraints", "apply-on") as (
        applying,
        state_db,
    ):
        session_id = _create_session(applying)
        session = _session(applying, session_id)
        # Seed an explicit user-authorized constraint into the task state *before* the turn.
        _seed_state(
            state_db,
            session.user_key,
            session_id,
            [
                constraint_event(
                    user_key=session.user_key,
                    task_id=session_id,
                    session_id=session_id,
                    kind="price_max",
                    value="30",
                )
            ],
        )

        response = _shopping_turn(applying, session_id, constraints=())
        assert response.status_code == 200, response.text
        body = response.json()

        identities = [item["parent_asin"] for item in body["recommendations"]]
        assert identities == ["cand-red", "cand-black"], (
            "APPLY must use the state's authorized price_max to narrow eligibility"
        )
        for identity in identities:
            assert float(CANDIDATE_PRICES[identity]) <= 30.0

        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert eligibility["requirements"] == [{"kind": "price_max", "expected": "30"}]
        assert eligibility["verified_eligible_count"] == 2
        assert eligibility["ineligible_count"] == 2
        assert eligibility["unresolved_count"] == 0
        assessments = eligibility["assessments"]
        assert len(assessments) == 4
        assert (
            eligibility["verified_eligible_count"]
            + eligibility["ineligible_count"]
            + eligibility["unresolved_count"]
            == len(assessments)
        )
        ineligible = {
            entry["parent_asin"]
            for entry in assessments
            if entry["eligibility"] == "ineligible"
        }
        assert ineligible == {"cand-blue", "cand-green"}
        for entry in assessments:
            if entry["eligibility"] == "ineligible":
                assert any(
                    report["verdict"] == "violated" for report in entry["reports"]
                ), "an ineligible candidate must carry a proved violation"

        # The state itself, replayed independently, holds exactly the seeded constraint.
        user_key = session.user_key
        recorded = _read_state(state_db, user_key, session_id)
        assert [
            (item.kind.value, item.value) for item in recorded["constraints"].constraints
        ] == [("price_max", "30")]
        diagnostics = _service(applying).state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] is None
        assert diagnostics["hard_constraint_count"] == 1
        assert diagnostics["constraint_kinds"] == ["price_max"]
        assert diagnostics["constraint_match"]["extra_in_state"] == ["price_max"]
        assert diagnostics["state_digest"] == recorded["digest"]


def test_apply_mode_inferred_hypothesis_never_filters_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement C: an inferred hypothesis never becomes a hard filter, even in APPLY."""
    with _harness(tmp_path, monkeypatch, "off", "hypothesis-off-reference") as (
        reference,
        _reference_db,
    ):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(reference, reference_session, constraints=())
        assert reference_response.status_code == 200, reference_response.text

    with _harness(tmp_path, monkeypatch, "apply_constraints", "hypothesis-apply") as (
        applying,
        state_db,
    ):
        session_id = _create_session(applying)
        session = _session(applying, session_id)
        # A hypothesis whose *value* would exclude every candidate if it were hard:
        # cand prices are 15..60, so price_max=1 cannot be satisfied by any of them.
        _seed_state(
            state_db,
            session.user_key,
            session_id,
            [
                hypothesis_event(
                    user_key=session.user_key,
                    task_id=session_id,
                    session_id=session_id,
                    kind="price_max",
                    value="1",
                )
            ],
        )

        response = _shopping_turn(applying, session_id, constraints=())
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="APPLY hypothesis vs OFF"
        )
        body = response.json()
        assert [item["parent_asin"] for item in body["recommendations"]] == list(INITIAL_ORDER)
        assert body["eligibility"]["evaluated"] is False

        user_key = session.user_key
        recorded = _read_state(state_db, user_key, session_id)
        state = recorded["state"]
        assert len(state.inferred_hypotheses) == 1
        assert [
            (item.kind.value, item.value) for item in state.inferred_hypotheses
        ] == [("price_max", "1")]
        assert state.authorized_hard_constraints == ()
        assert recorded["constraints"].constraints == ()
        assert recorded["constraints"].excluded_hypothesis_count >= 1
        diagnostics = _service(applying).state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] is None
        assert diagnostics["hard_constraint_count"] == 0
        assert diagnostics["constraint_kinds"] == []
        assert diagnostics["soft_hypothesis_count"] == 1
        assert diagnostics["excluded_hypothesis_count"] >= 1


def test_apply_mode_unverifiable_constraint_stays_unknown_not_ineligible(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement C: an unevaluable constraint leaves candidates UNKNOWN, not ineligible.

    ``free_form_constraint`` is the accepted dimension the deterministic evaluator cannot
    ground, so it is the documented unevaluable case: every candidate must be reported
    ``unresolved``/``unknown`` and *no* candidate may be reported ``ineligible`` from an
    UNKNOWN verdict.  A constraint the catalogue *can* decide (for example a colour) is a
    proved violation instead, which this test also pins so the two are never conflated.
    """
    with _harness(tmp_path, monkeypatch, "apply_constraints", "unknown-apply") as (
        applying,
        state_db,
    ):
        session_id = _create_session(applying)
        session = _session(applying, session_id)
        _seed_state(
            state_db,
            session.user_key,
            session_id,
            [
                constraint_event(
                    user_key=session.user_key,
                    task_id=session_id,
                    session_id=session_id,
                    kind="free_form_constraint",
                    value="extra-large",
                )
            ],
        )

        response = _shopping_turn(applying, session_id, constraints=())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["recommendations"] == []
        assert body["terminal"]["status"] == "waiting_for_user"
        assert body["terminal"]["termination_reason"] == "awaiting_user"
        assert body["clarification"] is not None
        assert body["clarification"]["kind"] == "unverified"

        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert eligibility["ineligible_count"] == 0, (
            "no candidate may be reported ineligible solely from UNKNOWN"
        )
        assessments = eligibility["assessments"]
        assert eligibility["unresolved_count"] == len(assessments) == 4
        assert (
            eligibility["verified_eligible_count"]
            + eligibility["ineligible_count"]
            + eligibility["unresolved_count"]
            == len(assessments)
        )
        for entry in assessments:
            assert entry["eligibility"] == "unresolved"
            assert entry["reports"]
            assert all(report["verdict"] == "unknown" for report in entry["reports"])

    # A catalogue-decidable constraint is a proved violation, not UNKNOWN: the evaluator
    # must not use "no match" as a synonym for "cannot decide".
    with _harness(tmp_path, monkeypatch, "apply_constraints", "decidable-apply") as (
        applying,
        state_db,
    ):
        session_id = _create_session(applying)
        session = _session(applying, session_id)
        _seed_state(
            state_db,
            session.user_key,
            session_id,
            [
                constraint_event(
                    user_key=session.user_key,
                    task_id=session_id,
                    session_id=session_id,
                    kind="color",
                    value="puce",
                )
            ],
        )
        response = _shopping_turn(applying, session_id, constraints=())
        assert response.status_code == 200, response.text
        body = response.json()
        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert eligibility["unresolved_count"] == 0
        assert eligibility["ineligible_count"] == 4
        for entry in eligibility["assessments"]:
            assert entry["eligibility"] == "ineligible"
            assert all(report["verdict"] == "violated" for report in entry["reports"])


# --------------------------------------------------------------------------- #
# D. Idempotency of a duplicate logical turn
# --------------------------------------------------------------------------- #


def test_shadow_duplicate_turn_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Requirement D: a retried logical turn adds no events and keeps the digest.

    M5.2 replaced the content-addressed M5.1 turn key with a server-owned identity, so a
    transport retry is deduplicated by the client ``idempotency_key`` (the documented retry
    identity).  The assertions about ids/count/digest are unchanged; the duplicate turn
    simply carries that key.  A final unkeyed request proves the complementary M5.2 rule:
    two distinct turns with identical content never collapse.
    """
    with _harness(tmp_path, monkeypatch, "shadow", "duplicate") as (harness, state_db):
        session_id = _create_session(harness)
        service = _service(harness)
        user_key = _session(harness, session_id).user_key
        retry_key = "m5-1-acceptance-retry"

        first_response = _shopping_turn(harness, session_id, idempotency_key=retry_key)
        assert first_response.status_code == 200, first_response.text
        first = _read_state(state_db, user_key, session_id)
        first_diagnostics = service.state_diagnostics(session_id)
        # context + explicit preference + explicit constraint + the two M5.2 trusted
        # snapshot references (M5.1 superset).
        assert first_diagnostics["event_count"] == len(first["events"]) == 5
        assert first_diagnostics["turn_count"] == 1
        unique_ids = {event.event_id for event in first["events"]}
        assert len(unique_ids) == len(first["events"])

        second_response = _shopping_turn(harness, session_id, idempotency_key=retry_key)
        assert second_response.status_code == 200, second_response.text
        second = _read_state(state_db, user_key, session_id)
        second_diagnostics = service.state_diagnostics(session_id)
        assert second_diagnostics["turn_count"] == 2
        assert second_diagnostics["events"] == 5

        # Same client idempotency key: the retry rebuilt the same logical turn identity and
        # the same event ids, so the append-only store treated it as a retry.
        assert len(second["events"]) == len(first["events"]) == 5
        assert {event.event_id for event in second["events"]} == unique_ids
        assert second["count"] == first["count"] == 5
        assert second_diagnostics["event_count"] == 5
        assert second_diagnostics["state_digest"] == first_diagnostics["state_digest"]
        assert second["digest"] == first["digest"]

        # Replay is still valid after the duplicate submission.
        replay = _read_state(state_db, user_key, session_id)
        assert replay["state"].interaction.event_count == 5
        assert replay["digest"] == first_diagnostics["state_digest"]
        assert second_diagnostics["error"] is None

        # The complementary M5.2 rule: an unkeyed identical request is a NEW server turn,
        # not a retry, so it appends its own (distinct) events.
        third_response = _shopping_turn(harness, session_id)
        assert third_response.status_code == 200, third_response.text
        third = _read_state(state_db, user_key, session_id)
        third_diagnostics = service.state_diagnostics(session_id)
        new_events = [
            event for event in third["events"] if event.event_id not in unique_ids
        ]
        assert new_events, "a distinct server turn with identical content must append events"
        assert {event.event_id for event in new_events}.isdisjoint(unique_ids)
        assert len(third["events"]) == 2 * len(first["events"])
        assert third_diagnostics["event_count"] == len(third["events"])
        assert third_diagnostics["state_digest"] == third["digest"]


# --------------------------------------------------------------------------- #
# E. Regression / robustness
# --------------------------------------------------------------------------- #


def test_state_replays_after_restart_from_the_diagnostics_digest(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement E: a fresh StateManager over the same DB reproduces the diagnostics digest."""
    state_db: Path
    with _harness(tmp_path, monkeypatch, "shadow", "restart") as (harness, db):
        state_db = db
        session_id = _create_session(harness)
        response = _shopping_turn(harness, session_id)
        assert response.status_code == 200, response.text
        user_key = _session(harness, session_id).user_key
        diagnostics = _service(harness).state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        expected_digest = diagnostics["state_digest"]
        expected_count = diagnostics["event_count"]

    # The app is closed (its store and integration are shut down); replay happens in a
    # completely fresh store/manager over the same file, i.e. after a "restart".
    replayed = _read_state(state_db, user_key, session_id)
    assert replayed["digest"] == expected_digest
    assert replayed["state"].interaction.event_count == expected_count
    assert replayed["agent"].projection_version == 1
    assert replayed["ranker"].ranker_signal_version == 1
    assert replayed["constraints"].constraints_version == 1


def test_state_is_isolated_between_sessions(tmp_path: Path, monkeypatch) -> None:
    """Requirement E: two sessions are separate user/task state; nothing crosses over."""
    with _harness(tmp_path, monkeypatch, "shadow", "isolation") as (harness, state_db):
        first_session = _create_session(harness, PROFILE_A)
        second_session = _create_session(harness, PROFILE_B)
        assert first_session != second_session
        first_key = _session(harness, first_session).user_key
        second_key = _session(harness, second_session).user_key
        assert first_key != second_key

        first_response = _shopping_turn(
            harness, first_session, message="Recommend some hiking gear. I prefer black."
        )
        second_response = _shopping_turn(
            harness, second_session, message="Recommend some hiking gear. I prefer red."
        )
        assert first_response.status_code == 200, first_response.text
        assert second_response.status_code == 200, second_response.text

        first = _read_state(state_db, first_key, first_session)
        second = _read_state(state_db, second_key, second_session)

        first_values = {
            (item.kind.value, item.value) for item in first["state"].active_preference_beliefs
        }
        second_values = {
            (item.kind.value, item.value) for item in second["state"].active_preference_beliefs
        }
        assert ("color", "black") in first_values
        assert ("color", "red") not in first_values
        assert ("color", "red") in second_values
        assert ("color", "black") not in second_values

        # Every stored event belongs to its owning user and task, and cross-reads are empty.
        for event in first["events"]:
            assert event.user_key == first_key
            assert event.task_id == first_session
        for event in second["events"]:
            assert event.user_key == second_key
            assert event.task_id == second_session
        with _open_state(state_db) as (store, _manager):
            assert store.get_events(second_session, first_key) == ()
            assert store.get_events(first_session, second_key) == ()
            assert store.count(first_key) == len(first["events"])
            assert store.count(second_key) == len(second["events"])

        assert first["digest"] != second["digest"]
        assert _service(harness).state_diagnostics(first_session)["state_digest"] == first["digest"]
        assert (
            _service(harness).state_diagnostics(second_session)["state_digest"]
            == second["digest"]
        )


def test_shadow_store_unavailable_is_diagnostics_only(tmp_path: Path, monkeypatch) -> None:
    """Requirement E: a SHADOW state-store failure is surfaced but cannot change the answer."""
    with _harness(tmp_path, monkeypatch, "off", "unavailable-off-reference") as (
        reference,
        _reference_db,
    ):
        reference_session = _create_session(reference)
        reference_response = _shopping_turn(reference, reference_session)
        assert reference_response.status_code == 200, reference_response.text

    # A path whose parent is a regular file: the store cannot be opened, which is a
    # realistic misconfiguration and is not an exception the request loop may crash on.
    blocked_root = tmp_path / "shadow-unavailable"
    blocked_root.mkdir(parents=True, exist_ok=True)
    (blocked_root / "blocked").write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(STATE_DB_ENV, str(blocked_root / "blocked" / "state.sqlite3"))

    harness = build_harness(blocked_root)
    try:
        session_id = _create_session(harness)
        response = _shopping_turn(harness, session_id)
        assert response.status_code == 200, response.text
    finally:
        close_harness(harness)

    assert_response_parity(
        reference_response.json(), response.json(), context="SHADOW store unavailable vs OFF"
    )


def test_shadow_store_unavailable_records_a_bounded_error(tmp_path: Path, monkeypatch) -> None:
    """Requirement E (diagnostics side): the SHADOW failure is surfaced, not swallowed."""
    blocked_root = tmp_path / "shadow-diagnostics"
    blocked_root.mkdir(parents=True, exist_ok=True)
    (blocked_root / "blocked").write_text("not a directory", encoding="utf-8")
    state_db = blocked_root / "blocked" / "state.sqlite3"
    monkeypatch.setenv(STATE_MODE_ENV, "shadow")
    monkeypatch.setenv(STATE_DB_ENV, str(state_db))

    harness = build_harness(blocked_root)
    try:
        session_id = _create_session(harness)
        response = _shopping_turn(harness, session_id)
        assert response.status_code == 200, response.text
        diagnostics = _service(harness).state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] == "state_store_unavailable"
        assert diagnostics["state_digest"] is None
        assert diagnostics["event_count"] == 0
        assert diagnostics["events"] == 0
    finally:
        close_harness(harness)


def test_apply_store_unavailable_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Requirement E: APPLY_CONSTRAINTS fails closed (502) instead of falling back."""
    blocked_root = tmp_path / "apply-unavailable"
    blocked_root.mkdir(parents=True, exist_ok=True)
    (blocked_root / "blocked").write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(STATE_MODE_ENV, "apply_constraints")
    monkeypatch.setenv(STATE_DB_ENV, str(blocked_root / "blocked" / "state.sqlite3"))

    harness = build_harness(blocked_root)
    try:
        session_id = _create_session(harness)
        response = _shopping_turn(harness, session_id)
        assert response.status_code == 502, response.text
        body = response.json()
        assert body["error"] in {"agent_failed", "state_integration_failed"}
        assert "recommendations" not in body
        diagnostics = _service(harness).state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="apply_constraints")
        assert diagnostics["error"] == "state_store_unavailable"
    finally:
        close_harness(harness)


def test_invalid_state_mode_fails_closed_at_composition(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement E: an unknown mode value is refused (503 demo_unavailable), never OFF."""
    with _harness(tmp_path, monkeypatch, "banana", "invalid-mode") as (harness, state_db):
        session_id = _create_session(harness)
        response = _shopping_turn(harness, session_id)
        assert response.status_code == 503, response.text
        assert response.json()["error"] == "demo_unavailable"
        assert "banana" not in response.text, "the invalid value must not be echoed"
        assert not state_db.exists(), "an invalid mode must never open a state store"
        # The stateless route composes the same service and must refuse identically.
        stateless = _stateless_turn(harness)
        assert stateless.status_code == 503
        assert stateless.json()["error"] == "demo_unavailable"


def test_legacy_memory_disagreement_is_diagnostics_only_and_schema_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    """Requirement E: when M9 memory and M5 state disagree, SHADOW stays observational.

    The accepted M9 preference memory (the legacy long-term interpretation) is seeded with
    a preference; the M5 task state is seeded with an *extra* explicit hard constraint the
    trusted request does not carry.  SHADOW must keep the user-visible answer identical to
    OFF, report the bounded kind-level mismatch in diagnostics, and leave the M9 schema
    untouched.
    """
    with _harness(tmp_path, monkeypatch, "off", "legacy-off") as (reference, reference_db):
        reference_session = _create_session(reference)
        reference_user = _session(reference, reference_session).user_key
        reference.runtime.memory_service.process_turn(
            user_key=reference_user,
            user_message="I prefer red.",
            turn_id="legacy-memory:seed",
        )
        reference_response = _shopping_turn(reference, reference_session, constraints=())
        assert reference_response.status_code == 200, reference_response.text
        reference_off = {
            (entry.kind.value, entry.polarity.value, entry.value)
            for entry in reference.runtime.memory_service.get_active_preferences(
                reference_user
            ).active_entries
        }
        reference_schema = _sqlite_schema(_memory_db(reference_db.parent))

    with _harness(tmp_path, monkeypatch, "shadow", "legacy-shadow") as (shadow, state_db):
        session_id = _create_session(shadow)
        session = _session(shadow, session_id)
        shadow.runtime.memory_service.process_turn(
            user_key=session.user_key,
            user_message="I prefer red.",
            # The same seed turn id in both harnesses: the accepted M9 memory records it
            # in `source_turn_id`, which is part of the response's intent view.  That is
            # *test input* identity, not a mode difference, so it must be held constant.
            turn_id="legacy-memory:seed",
        )
        # M5 state disagrees with the trusted interpretation for this request: an explicit
        # hard constraint the legacy request does not carry.
        _seed_state(
            state_db,
            session.user_key,
            session_id,
            [
                constraint_event(
                    user_key=session.user_key,
                    task_id=session_id,
                    session_id=session_id,
                    kind="price_max",
                    value="30",
                )
            ],
        )
        response = _shopping_turn(shadow, session_id, constraints=())
        assert response.status_code == 200, response.text
        assert_response_parity(
            reference_response.json(), response.json(), context="legacy disagreement vs OFF"
        )
        # SHADOW did not apply the M5-only constraint: the same four candidates are shown.
        assert [item["parent_asin"] for item in response.json()["recommendations"]] == list(
            INITIAL_ORDER
        )

        shadow_off = {
            (entry.kind.value, entry.polarity.value, entry.value)
            for entry in shadow.runtime.memory_service.get_active_preferences(
                session.user_key
            ).active_entries
        }
        assert shadow_off == reference_off

        diagnostics = _service(shadow).state_diagnostics(session_id)
        _assert_diagnostics(diagnostics, mode="shadow")
        assert diagnostics["error"] is None
        match = diagnostics["constraint_match"]
        assert match["matched"] == 0
        assert match["extra_in_state"] == ["price_max"]
        assert match["missing_from_state"] == []
        assert isinstance(diagnostics["hard_constraint_count"], int)
        assert isinstance(diagnostics["trusted_constraint_count"], int)

        memory_db = _memory_db(state_db.parent)
        assert memory_db.exists()
        assert state_db != memory_db, "decision state and M9 memory must be separate files"
        shadow_schema = _sqlite_schema(memory_db)
        assert shadow_schema == reference_schema, "the M9 store schema must not change"
        # The decision-state engine never adds tables to the accepted memory database.
        assert "events" not in shadow_schema
        state_schema = _sqlite_schema(state_db)
        assert "events" in state_schema and "meta" in state_schema
