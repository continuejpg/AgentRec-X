"""Focused tests for M5.1: unified decision state integration in the shopping runtime.

Scope of these tests is the **integration**, not the accepted components it composes:
the M5 state engine, the M9 memory service, the shopping intent and the eligibility
evaluator are already covered by their own suites.  Here we pin:

* A. OFF is the default and changes nothing (no store, no events, byte-identical answer
  after normalising the response fields the accepted control plane randomises per run);
* B. SHADOW records validated facts, produces all three projections, keeps the
  user-visible answer identical to OFF and never lets a failure change it;
* C. APPLY_CONSTRAINTS merges only user-authorized state constraints into the existing
  trusted inputs, keeps UNKNOWN three-state (never a violation), and fails closed;
* D. identity/idempotency/replay: content-addressed turn keys, user+task isolation,
  deterministic replay after restart, stateless route untouched;
* E. failure surfaces: unavailable store, invalid event, legacy-memory disagreement,
  invalid mode -- each with the documented mode-specific policy.

The fixture catalogue is the established synthetic one: prices 25.00 / 40.00 / 15.00 /
60.00 for cand-red / cand-blue / cand-black / cand-green (see
``tests/agent_reranking_fixture.py``).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.state_integration import (  # noqa: E402
    DEFAULT_DEMO_STATE_DB,
    ENV_STATE_MODE,
    MAX_DIAGNOSTIC_TASKS,
    STATE_MODE_VALUES,
    ShoppingStateIntegration,
    StateIntegrationError,
    StateMode,
    StateTurnOutcome,
    resolve_state_mode,
    state_database_path,
)
from recommendation.memory import (  # noqa: E402
    PreferenceExtraction,
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
)
from recommendation.state import (  # noqa: E402
    SQLiteEventStore,
    StateManager,
    for_constraints,
    new_event,
)
from recommendation.state.shopping_adapter import (  # noqa: E402
    ExplicitConstraint,
    ExplicitPreference,
)
from recommendation.state.state_projector import (  # noqa: E402
    AGENT_PROJECTION_VERSION,
    CONSTRAINT_PROJECTION_VERSION,
    RANKER_SIGNAL_VERSION,
)
from tests.agent_reranking_fixture import INITIAL_ORDER  # noqa: E402
from tests.demo_fixture import (  # noqa: E402
    PROFILE_A,
    PROFILE_B,
    build_harness,
    close_harness,
)

BASE = "/v1/demo"
CREATE = BASE + "/sessions"
SHOPPING = BASE + "/sessions/{session_id}/agent/shopping"
RECOMMEND = BASE + "/agent/recommend"

ENV_STATE_DB = "AGENTRECX_STATE_DB"

#: A message the accepted rule-based extractor resolves to a soft explicit preference
#: (colour "black"), so a shadow turn normally emits a preference event as well.
PREFERENCE_MESSAGE = "I prefer black shoes."
#: A plain recommendation request that states nothing: only the task-context event.
NEUTRAL_MESSAGE = "Recommend some hiking gear."

#: Fields the **accepted** control plane randomises per request (run ids are uuid-based)
#: or that embed the concrete server-owned session id.  They are not M5.1 output and must
#: not make two genuinely identical answers compare unequal.
_RANDOM_KEYS = frozenset(
    {
        "timing",
        "run_id",
        "run_ref",
        # The intent's provenance ids embed the concrete session id.
        "source_turn_id",
        "turn_ids",
    }
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _normalised(payload: Any) -> Any:
    """Response body with the accepted per-request random fields stripped."""
    if isinstance(payload, dict):
        return {
            key: _normalised(value)
            for key, value in payload.items()
            if key not in _RANDOM_KEYS
        }
    if isinstance(payload, (list, tuple)):
        return [_normalised(value) for value in payload]
    return payload


def _set_state_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str | None,
    database: str | Path | None,
) -> None:
    """Set (or clear) the two M5.1 environment variables for one test."""
    if mode is None:
        monkeypatch.delenv(ENV_STATE_MODE, raising=False)
    else:
        monkeypatch.setenv(ENV_STATE_MODE, mode)
    if database is None:
        monkeypatch.delenv(ENV_STATE_DB, raising=False)
    else:
        monkeypatch.setenv(ENV_STATE_DB, str(database))


@contextmanager
def _harness(tmp_path: Path, name: str) -> Iterator[Any]:
    """Build one independent offline demo harness under its own directory."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    harness = build_harness(root, memory_db=root / "memory.sqlite3")
    try:
        yield harness
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


def _state_path(tmp_path: Path, name: str = "state.sqlite3") -> Path:
    return tmp_path / name


def _seed_event(path: Path, **event_kwargs: Any) -> None:
    """Append one validated event to a standalone decision-state database."""
    store = SQLiteEventStore(path)
    try:
        StateManager(store).apply_event(new_event(**event_kwargs))
    finally:
        store.close()


def _seed_constraint(
    path: Path,
    *,
    user_key: str,
    task_id: str,
    kind: str = "price_max",
    value: str = "30",
    event_id: str = "m5evt-seed-constraint",
) -> None:
    _seed_event(
        path,
        event_type="user_constraint_stated",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=task_id,
        actor="user",
        source="test_seed",
        timestamp=1_700_000_000.0,
        kind=kind,
        value=value,
        polarity="prefer",
        scope="task",
        source_text=f"seeded {kind} {value}",
    )


def _seed_hypothesis(
    path: Path,
    *,
    user_key: str,
    task_id: str,
    kind: str = "feature",
    value: str = "lightweight",
    event_id: str = "m5evt-seed-hypothesis",
) -> None:
    _seed_event(
        path,
        event_type="preference_hypothesis_proposed",
        event_id=event_id,
        user_key=user_key,
        task_id=task_id,
        session_id=task_id,
        actor="agent",
        source="test_seed",
        timestamp=1_700_000_000.0,
        kind=kind,
        value=value,
        polarity="prefer",
        scope="task",
        strength="weak",
        basis="browsing_history",
        reason="seeded hypothesis",
        evidence_refs=("seed-evidence",),
    )


def _replay(path: Path, user_key: str, task_id: str) -> Any:
    """Replay one task from the store on disk with a fresh manager."""
    store = SQLiteEventStore(path)
    try:
        return StateManager(store).replay_state(user_key, task_id)
    finally:
        store.close()


def _stored_events(path: Path, user_key: str, task_id: str) -> list[Any]:
    """Read one task's stored event stream from disk (no state replay)."""
    store = SQLiteEventStore(path)
    try:
        return list(store.get_events(task_id, user_key))
    finally:
        store.close()


def _stored_rows(path: Path, user_key: str, task_id: str) -> int:
    """Number of durable event rows for one user+task (direct SQL, no model access)."""
    connection = sqlite3.connect(str(path))
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM events WHERE user_key = ? AND task_id = ?",
            (user_key, task_id),
        ).fetchone()
        return int(row[0])
    finally:
        connection.close()


def _diagnostics_blob(record: dict[str, Any]) -> str:
    """Serialise a diagnostics record to prove it is JSON-safe and bounded."""
    return json.dumps(record, sort_keys=True)


class _InvalidExtractor:
    """An extractor whose *turn-preview* output the runtime tolerates but M5 rejects.

    The accepted runtime calls the extractor twice per shopping turn: once for the intent
    preview in ``run_shopping_turn`` and once when the loop commits memory after the run.
    Only the first (preview) call returns the malformed candidate -- a 200-character
    preference value, structurally impossible to store under the accepted 120-character
    M9 bound -- so the accepted memory write is unaffected and OFF still serves a normal
    answer.  The M5 adapter, which consumes the preview, must reject it as an invalid
    event; that is exactly the failure the SHADOW/APPLY policies are tested on.
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


# =========================================================================== #
# A. OFF is the default and changes nothing
# =========================================================================== #


def test_mode_resolution_and_inert_off_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modes resolve strictly; OFF opens nothing and returns ``None``."""
    monkeypatch.delenv(ENV_STATE_MODE, raising=False)
    monkeypatch.delenv(ENV_STATE_DB, raising=False)
    assert STATE_MODE_VALUES == ("off", "shadow", "apply_constraints")
    assert resolve_state_mode() is StateMode.OFF
    assert resolve_state_mode("shadow") is StateMode.SHADOW
    assert resolve_state_mode("apply_constraints") is StateMode.APPLY_CONSTRAINTS
    assert ShoppingStateIntegration(mode=StateMode.OFF).enabled is False

    monkeypatch.setenv(ENV_STATE_MODE, "shadow")
    assert resolve_state_mode() is StateMode.SHADOW
    assert resolve_state_mode("off") is StateMode.OFF

    monkeypatch.setenv(ENV_STATE_MODE, "bogus")
    with pytest.raises(StateIntegrationError) as raised:
        resolve_state_mode()
    assert raised.value.code == "invalid_state_mode"
    # The string form is a fixed category message: it never echoes the raw value.
    assert "bogus" not in str(raised.value)
    with pytest.raises(StateIntegrationError):
        ShoppingStateIntegration.from_env()
    with pytest.raises(StateIntegrationError):
        ShoppingStateIntegration(mode="bogus")  # type: ignore[arg-type]

    monkeypatch.delenv(ENV_STATE_MODE, raising=False)
    assert state_database_path() == DEFAULT_DEMO_STATE_DB
    configured = tmp_path / "configured.sqlite3"
    monkeypatch.setenv(ENV_STATE_DB, str(configured))
    assert state_database_path() == configured

    # OFF is inert: it must not open the (deliberately unopenable) store.
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory", encoding="utf-8")
    integration = ShoppingStateIntegration(
        mode=StateMode.OFF, database=blocked / "state.sqlite3"
    )
    try:
        assert integration.integrate_turn(
            user_key="user-1",
            session_id="task-1",
            task_id="task-1",
            turn_key="turn-1",
            message=NEUTRAL_MESSAGE,
            intent_label="recommend",
            preferences=(),
            constraints=(),
            trusted_constraints=(),
        ) is None
        assert integration.diagnostics("task-1") is None
    finally:
        integration.close()
    integration.close()  # idempotent


def test_off_default_and_explicit_off_are_identical_and_touch_no_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default (unset) and explicit ``off`` are the same behaviour and open nothing."""
    request = dict(
        message=PREFERENCE_MESSAGE,
        # A decidable budget every fixture candidate satisfies: it exercises the
        # constraint event path without making the run suspend on an unverified candidate.
        constraints=[{"kind": "price_max", "value": "120"}],
    )
    default_db = _state_path(tmp_path, "state_default.sqlite3")
    _set_state_env(monkeypatch, mode=None, database=default_db)
    with _harness(tmp_path, "default") as harness:
        session = _create_session(harness)
        response = _shopping(harness, session, **request)
        assert response.status_code == 200, response.text
        default_body = _normalised(response.json())
        service = _service(harness)
        assert service.state_mode == "off"
        assert service.state_diagnostics(_identity(harness, session)[1]) is None
    assert not default_db.exists(), "OFF opened the decision-state store"

    explicit_db = _state_path(tmp_path, "state_explicit.sqlite3")
    _set_state_env(monkeypatch, mode="off", database=explicit_db)
    with _harness(tmp_path, "explicit") as harness:
        session = _create_session(harness)
        response = _shopping(harness, session, **request)
        assert response.status_code == 200, response.text
        explicit_body = _normalised(response.json())
        assert _service(harness).state_mode == "off"
    assert not explicit_db.exists(), "explicit OFF opened the decision-state store"
    # Recommendations, scores, grounding, eligibility, message and terminal are identical.
    assert default_body == explicit_body


# =========================================================================== #
# B. SHADOW records, projects, and never changes the answer
# =========================================================================== #


def test_shadow_is_user_visible_parity_and_records_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SHADOW answer == OFF answer; the state, projections and diagnostics exist."""
    request = dict(
        message=PREFERENCE_MESSAGE,
        constraints=[{"kind": "price_max", "value": "120"}],
    )
    off_db = _state_path(tmp_path, "off.sqlite3")
    _set_state_env(monkeypatch, mode=None, database=off_db)
    with _harness(tmp_path, "off") as harness:
        session = _create_session(harness)
        off_response = _shopping(harness, session, **request)
        assert off_response.status_code == 200, off_response.text
        off_body = _normalised(off_response.json())

    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        response = _shopping(harness, session, **request)
        assert response.status_code == 200, response.text
        assert _normalised(response.json()) == off_body

        service = _service(harness)
        assert service.state_mode == "shadow"
        diagnostics = service.state_diagnostics(task_id)
        assert diagnostics is not None
        state = _replay(shadow_db, user_key, task_id)

        # The state exists on disk and is exactly the state the diagnostics describe.
        assert diagnostics["state_digest"] == state.state_digest()
        assert diagnostics["event_count"] == state.interaction.event_count
        assert diagnostics["event_count"] >= 3  # context + preference + constraint
        assert _stored_rows(shadow_db, user_key, task_id) == diagnostics["event_count"]
        # The three projections were produced for this state.
        assert diagnostics["agent_projection_version"] == AGENT_PROJECTION_VERSION
        assert diagnostics["ranker_signal_version"] == RANKER_SIGNAL_VERSION
        assert diagnostics["constraints_version"] == CONSTRAINT_PROJECTION_VERSION
        assert diagnostics["ranker_feature_count"] >= 1
        assert diagnostics["preference_count"] >= 1
        assert diagnostics["hard_constraint_count"] == 1
        assert diagnostics["constraint_kinds"] == ["price_max"]
        assert diagnostics["constraint_match"] == {
            "matched": 1,
            "missing_from_state": [],
            "extra_in_state": [],
        }
        assert diagnostics["trusted_constraint_count"] == 1
        assert diagnostics["mode"] == "shadow"
        assert diagnostics["task_id"] == task_id
        assert len(diagnostics["turn_key_digest"]) == 12
        assert diagnostics["state_version"] == 1
        assert diagnostics["error"] is None
        assert diagnostics["turn_count"] == 1
        # Bounded and non-sensitive: nothing from the raw turn is present.
        blob = _diagnostics_blob(diagnostics)
        assert "120" not in blob
        assert PREFERENCE_MESSAGE not in blob
        assert user_key not in blob


# =========================================================================== #
# C. APPLY_CONSTRAINTS: authorized state constraints narrow eligibility
# =========================================================================== #


def test_apply_constraints_narrows_candidates_from_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seeded authorized constraint narrows the presented set through the existing path."""
    request = dict(message=NEUTRAL_MESSAGE)
    _set_state_env(monkeypatch, mode=None, database=_state_path(tmp_path, "off.sqlite3"))
    with _harness(tmp_path, "off") as harness:
        session = _create_session(harness)
        off_response = _shopping(harness, session, **request)
        assert off_response.status_code == 200, off_response.text
        off_ids = [item["parent_asin"] for item in off_response.json()["recommendations"]]
        assert "cand-blue" in off_ids  # 40.00: excluded once the budget is applied

    apply_db = _state_path(tmp_path, "apply.sqlite3")
    _set_state_env(monkeypatch, mode="apply_constraints", database=apply_db)
    with _harness(tmp_path, "apply") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        _seed_constraint(apply_db, user_key=user_key, task_id=task_id, value="30")

        response = _shopping(harness, session, **request)
        assert response.status_code == 200, response.text
        body = response.json()
        eligibility = body["eligibility"]
        assert eligibility["evaluated"] is True
        assert {"kind": "price_max", "expected": "30"} in eligibility["requirements"]
        presented = [item["parent_asin"] for item in body["recommendations"]]
        assert "cand-blue" not in presented  # 40.00
        assert "cand-green" not in presented  # 60.00
        assert "cand-red" in presented  # 25.00
        assert "cand-black" in presented  # 15.00
        assert body["terminal"]["status"] == "finished"

        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["constraint_kinds"] == ["price_max"]
        # The seeded state holds the only authorized constraint; it was not in the request.
        assert diagnostics["trusted_constraint_count"] == 0
        assert diagnostics["constraint_match"]["extra_in_state"] == ["price_max"]


def test_inferred_hypothesis_never_becomes_a_hard_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A soft hypothesis is counted and excluded from constraints; it filters nothing."""
    _set_state_env(monkeypatch, mode=None, database=_state_path(tmp_path, "off.sqlite3"))
    with _harness(tmp_path, "off") as harness:
        session = _create_session(harness)
        off_response = _shopping(harness, session, message=NEUTRAL_MESSAGE)
        assert off_response.status_code == 200, off_response.text
        off_ids = [item["parent_asin"] for item in off_response.json()["recommendations"]]

    apply_db = _state_path(tmp_path, "apply.sqlite3")
    _set_state_env(monkeypatch, mode="apply_constraints", database=apply_db)
    with _harness(tmp_path, "apply") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        _seed_hypothesis(apply_db, user_key=user_key, task_id=task_id)

        response = _shopping(harness, session, message=NEUTRAL_MESSAGE)
        assert response.status_code == 200, response.text
        body = response.json()
        present_ids = [item["parent_asin"] for item in body["recommendations"]]
        # No filtering at all: same set and order as OFF, because nothing is constrained.
        assert present_ids == off_ids
        assert body["eligibility"]["evaluated"] is False

        state = _replay(apply_db, user_key, task_id)
        projection = for_constraints(state)
        assert projection.constraints == ()
        assert projection.excluded_hypothesis_count == 1
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["excluded_hypothesis_count"] >= 1
        assert diagnostics["soft_hypothesis_count"] >= 1
        assert diagnostics["hard_constraint_count"] == 0
        assert diagnostics["constraint_kinds"] == []


def test_unknown_constraint_is_not_reported_as_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constraint the catalogue cannot decide stays UNKNOWN: nobody is ineligible from it."""
    apply_db = _state_path(tmp_path, "apply.sqlite3")
    _set_state_env(monkeypatch, mode="apply_constraints", database=apply_db)
    with _harness(tmp_path, "apply") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        # ``free_form_constraint`` maps to the evaluator's UNVERIFIABLE dimension: the
        # catalogue has no attribute that could decide it, so every verdict must be
        # UNKNOWN -- never VIOLATED.
        _seed_constraint(
            apply_db,
            user_key=user_key,
            task_id=task_id,
            kind="free_form_constraint",
            value="extra large",
        )

        response = _shopping(harness, session, message=NEUTRAL_MESSAGE)
        assert response.status_code == 200, response.text
        eligibility = response.json()["eligibility"]
        assert eligibility["evaluated"] is True
        assert eligibility["requirements"][0]["kind"] == "unverifiable"
        assert eligibility["ineligible_count"] == 0
        assert eligibility["assessments"], "each grounded candidate is assessed"
        for assessment in eligibility["assessments"]:
            assert assessment["eligibility"] == "unresolved"
        # Three-state bookkeeping is intact: unresolved + verified + ineligible == all.
        assert (
            eligibility["verified_eligible_count"]
            + eligibility["ineligible_count"]
            + eligibility["unresolved_count"]
            == len(eligibility["assessments"])
        )


# =========================================================================== #
# D. Identity, idempotency, replay, isolation
# =========================================================================== #


def test_shadow_turn_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same logical turn (same client idempotency key) appends nothing twice.

    M5.2 identifies a logical turn by ``idempotency_key`` > server turn id > content
    digest.  Two HTTP requests are therefore distinct server turns unless they carry the
    same client key, which is the documented retry identity; the assertions about ids,
    counts and digests are unchanged.
    """
    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        request = dict(
            message=PREFERENCE_MESSAGE,
            constraints=[{"kind": "price_max", "value": "120"}],
            idempotency_key="m5-1-integration-retry",
        )
        first = _shopping(harness, session, **request)
        assert first.status_code == 200, first.text
        first_state = _replay(shadow_db, user_key, task_id)
        first_diagnostics = _service(harness).state_diagnostics(task_id)
        assert first_diagnostics is not None
        first_rows = _stored_rows(shadow_db, user_key, task_id)

        second = _shopping(harness, session, **request)
        assert second.status_code == 200, second.text
        second_state = _replay(shadow_db, user_key, task_id)
        second_diagnostics = _service(harness).state_diagnostics(task_id)
        assert second_diagnostics is not None

        # Same client-keyed logical turn: the retry re-emits the same event ids, so the
        # store's duplicate rule makes it a no-op.
        assert second_diagnostics["turn_key_digest"] == first_diagnostics["turn_key_digest"]
        assert second_diagnostics["events"] == first_diagnostics["events"]
        assert second_diagnostics["event_count"] == first_diagnostics["event_count"]
        assert second_diagnostics["state_digest"] == first_diagnostics["state_digest"]
        assert first_state.state_digest() == second_state.state_digest()
        assert _stored_rows(shadow_db, user_key, task_id) == first_rows
        assert first_rows == first_diagnostics["event_count"]
        # The diagnostics counter still counts the attempt, which is what makes a retry
        # visible even though no event was appended.
        assert second_diagnostics["turn_count"] == first_diagnostics["turn_count"] + 1

        # The complementary M5.2 rule: without a client key an identical request is a new
        # server turn and appends its own, distinct event ids (no content collapse).
        first_ids = {
            event.event_id
            for event in _stored_events(shadow_db, user_key, task_id)
        }
        third = _shopping(
            harness,
            session,
            message=PREFERENCE_MESSAGE,
            constraints=[{"kind": "price_max", "value": "120"}],
        )
        assert third.status_code == 200, third.text
        third_events = _stored_events(shadow_db, user_key, task_id)
        new_events = [event for event in third_events if event.event_id not in first_ids]
        assert new_events, "a distinct server turn must append distinct events"
        assert {event.event_id for event in new_events}.isdisjoint(first_ids)
        third_diagnostics = _service(harness).state_diagnostics(task_id)
        assert third_diagnostics is not None
        assert third_diagnostics["event_count"] == len(third_events)
        assert third_diagnostics["state_digest"] == _replay(
            shadow_db, user_key, task_id
        ).state_digest()


def test_shadow_state_replays_identically_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing the live integration and replaying from disk reproduces digest and count."""
    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        response = _shopping(
            harness,
            session,
            message=PREFERENCE_MESSAGE,
            constraints=[{"kind": "price_max", "value": "120"}],
        )
        assert response.status_code == 200, response.text
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None

        integration = _service(harness)._state_integration
        integration.close()  # release the live connection before replaying from disk

        first = _replay(shadow_db, user_key, task_id)
        second = _replay(shadow_db, user_key, task_id)
        assert first.state_digest() == diagnostics["state_digest"]
        assert second.state_digest() == diagnostics["state_digest"]
        assert first.state_digest() == second.state_digest()
        assert first.interaction.event_count == diagnostics["event_count"]
        assert second.interaction.event_count == diagnostics["event_count"]


def test_shadow_user_and_task_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sessions each own their task: diagnostics and state never cross sessions."""
    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        session_a = _create_session(harness, PROFILE_A)
        session_b = _create_session(harness, PROFILE_B)
        user_a, task_a = _identity(harness, session_a)
        user_b, task_b = _identity(harness, session_b)
        assert user_a != user_b and task_a != task_b

        response_a = _shopping(harness, session_a, message=PREFERENCE_MESSAGE)
        assert response_a.status_code == 200, response_a.text
        service = _service(harness)
        assert service.state_diagnostics(task_a) is not None
        assert service.state_diagnostics(task_b) is None
        assert _replay(shadow_db, user_b, task_b).interaction.event_count == 0

        response_b = _shopping(harness, session_b, message=PREFERENCE_MESSAGE)
        assert response_b.status_code == 200, response_b.text
        diagnostics_a = service.state_diagnostics(task_a)
        diagnostics_b = service.state_diagnostics(task_b)
        assert diagnostics_a is not None and diagnostics_b is not None
        assert diagnostics_a["task_id"] == task_a
        assert diagnostics_b["task_id"] == task_b
        state_a = _replay(shadow_db, user_a, task_a)
        state_b = _replay(shadow_db, user_b, task_b)
        assert state_a.task_id == task_a and state_b.task_id == task_b
        assert state_a.user_key == user_a and state_b.user_key == user_b
        assert state_a.state_digest() == diagnostics_a["state_digest"]
        assert state_b.state_digest() == diagnostics_b["state_digest"]
        assert state_a.state_digest() != state_b.state_digest()


def test_direct_integration_isolates_tasks_for_one_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two task ids on one user key are separate states with separate constraints."""
    monkeypatch.delenv(ENV_STATE_DB, raising=False)
    database = _state_path(tmp_path, "direct.sqlite3")
    integration = ShoppingStateIntegration(mode=StateMode.SHADOW, database=database)
    try:
        def turn(task_id: str, turn_key: str, value: str) -> StateTurnOutcome:
            outcome = integration.integrate_turn(
                user_key="shared-user",
                session_id=task_id,
                task_id=task_id,
                turn_key=turn_key,
                message=f"budget {value}",
                intent_label="recommend",
                preferences=(
                    ExplicitPreference(
                        kind=PreferenceKind.BRAND,
                        value=f"brand-{task_id}",
                        polarity=PreferencePolarity.PREFER,
                        source_text="I prefer this brand",
                    ),
                ),
                constraints=(
                    ExplicitConstraint(
                        kind=PreferenceKind.PRICE_MAX,
                        value=value,
                        polarity=PreferencePolarity.PREFER,
                        source_text=f"under {value}",
                    ),
                ),
                trusted_constraints=(("price_max", value),),
            )
            assert isinstance(outcome, StateTurnOutcome)
            return outcome

        first = turn("task-a", "turn-a", "30")
        second = turn("task-b", "turn-b", "50")
        assert first.state.task_id == "task-a"
        assert second.state.task_id == "task-b"
        assert first.state.user_key == second.state.user_key == "shared-user"
        assert first.authorized_constraints == (("price_max", "30"),)
        assert second.authorized_constraints == (("price_max", "50"),)
        assert first.state.state_digest() != second.state.state_digest()
        assert first.diagnostics["task_id"] == "task-a"
        assert second.diagnostics["task_id"] == "task-b"
        assert integration.diagnostics("task-c") is None
        # Both tasks are durable for the one user; each replay is its own stream.
        replay_a = _replay(database, "shared-user", "task-a")
        replay_b = _replay(database, "shared-user", "task-b")
        assert replay_a.state_digest() == first.state.state_digest()
        assert replay_b.state_digest() == second.state.state_digest()
    finally:
        integration.close()


def test_stateless_agent_route_writes_no_state_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stateless route has no stable task identity: SHADOW leaves it (and the store) alone."""
    body = {
        "message": NEUTRAL_MESSAGE,
        "history": list(INITIAL_ORDER),
        "k": 3,
    }
    _set_state_env(monkeypatch, mode=None, database=_state_path(tmp_path, "off.sqlite3"))
    with _harness(tmp_path, "off") as harness:
        off_response = harness.client.post(RECOMMEND, json=body)
        assert off_response.status_code == 200, off_response.text
        off_body = _normalised(off_response.json())

    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        shadow_response = harness.client.post(RECOMMEND, json=body)
        assert shadow_response.status_code == 200, shadow_response.text
        assert _normalised(shadow_response.json()) == off_body
        assert _service(harness).state_mode == "shadow"
    # No integration ran, so the store was never even opened.
    if shadow_db.exists():
        connection = sqlite3.connect(str(shadow_db))
        try:
            rows = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            connection.close()
        assert int(rows) == 0


# =========================================================================== #
# E. Failure surfaces and their mode-specific policy
# =========================================================================== #


def test_store_unavailable_is_shadow_diagnostic_only_and_apply_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unopenable store cannot change SHADOW's answer; APPLY fails closed."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    blocked_db = blocker / "state.sqlite3"
    request = dict(message=NEUTRAL_MESSAGE)

    _set_state_env(monkeypatch, mode=None, database=_state_path(tmp_path, "off.sqlite3"))
    with _harness(tmp_path, "off") as harness:
        session = _create_session(harness)
        off_response = _shopping(harness, session, **request)
        assert off_response.status_code == 200, off_response.text
        off_body = _normalised(off_response.json())

    _set_state_env(monkeypatch, mode="shadow", database=blocked_db)
    with _harness(tmp_path, "shadow") as harness:
        session = _create_session(harness)
        _, task_id = _identity(harness, session)
        response = _shopping(harness, session, **request)
        # SHADOW is observational: identical answer, failure surfaced in diagnostics.
        assert response.status_code == 200, response.text
        assert _normalised(response.json()) == off_body
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_store_unavailable"
        assert diagnostics["events"] == 0
        assert diagnostics["event_count"] == 0
        assert diagnostics["state_digest"] is None

    _set_state_env(monkeypatch, mode="apply_constraints", database=blocked_db)
    with _harness(tmp_path, "apply") as harness:
        session = _create_session(harness)
        _, task_id = _identity(harness, session)
        response = _shopping(harness, session, **request)
        # APPLY is fail-closed: a state failure is a refused turn, never a silent fallback.
        assert response.status_code == 502, response.text
        assert response.json()["error"] == "agent_failed"
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_store_unavailable"


def test_unknown_state_mode_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid mode refuses composition (503) instead of silently running OFF."""
    _set_state_env(monkeypatch, mode="bogus", database=_state_path(tmp_path, "state.sqlite3"))
    with _harness(tmp_path, "bogus") as harness:
        session = _create_session(harness)
        response = _shopping(harness, session)
        assert response.status_code == 503, response.text
        assert response.json()["error"] == "demo_unavailable"


def test_invalid_extraction_event_is_shadow_diagnostic_and_apply_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fact that cannot be stored is an event-validation failure, not an answer change."""
    request = dict(message=NEUTRAL_MESSAGE)

    _set_state_env(monkeypatch, mode=None, database=_state_path(tmp_path, "off.sqlite3"))
    with _harness(tmp_path, "off") as harness:
        session = _create_session(harness)
        # The parity comparison must use the same extractor behaviour on both sides; only
        # the M5.1 mode differs.
        monkeypatch.setattr(harness.runtime.memory_service, "_extractor", _InvalidExtractor())
        off_body = _normalised(_shopping(harness, session, **request).json())

    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        session = _create_session(harness)
        _, task_id = _identity(harness, session)
        monkeypatch.setattr(harness.runtime.memory_service, "_extractor", _InvalidExtractor())
        response = _shopping(harness, session, **request)
        assert response.status_code == 200, response.text
        assert _normalised(response.json()) == off_body
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_event_invalid"
        assert diagnostics["event_count"] == 0

    _set_state_env(monkeypatch, mode="apply_constraints", database=shadow_db)
    with _harness(tmp_path, "apply") as harness:
        session = _create_session(harness)
        _, task_id = _identity(harness, session)
        monkeypatch.setattr(harness.runtime.memory_service, "_extractor", _InvalidExtractor())
        response = _shopping(harness, session, **request)
        assert response.status_code == 502, response.text
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["error"] == "state_event_invalid"


def test_legacy_memory_disagreement_is_reported_at_kind_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State and legacy memory can disagree; SHADOW reports only the bounded kind diff."""
    request = dict(message=NEUTRAL_MESSAGE)

    def seed_legacy(harness: Any, user_key: str) -> None:
        # M9 wording with "under" is classified hard by the accepted intent rule, so the
        # legacy interpretation becomes a trusted price_max constraint for the turn.
        harness.runtime.memory_service.process_turn(
            user_key=user_key,
            user_message="I need it under $20",
            turn_id="test-seed-legacy",
        )

    _set_state_env(monkeypatch, mode=None, database=_state_path(tmp_path, "off.sqlite3"))
    with _harness(tmp_path, "off") as harness:
        session = _create_session(harness)
        user_key, _ = _identity(harness, session)
        seed_legacy(harness, user_key)
        off_response = _shopping(harness, session, **request)
        assert off_response.status_code == 200, off_response.text
        off_body = _normalised(off_response.json())

    shadow_db = _state_path(tmp_path, "shadow.sqlite3")
    _set_state_env(monkeypatch, mode="shadow", database=shadow_db)
    with _harness(tmp_path, "shadow") as harness:
        session = _create_session(harness)
        user_key, task_id = _identity(harness, session)
        seed_legacy(harness, user_key)
        # The state already holds a different budget value for this task (as if an earlier
        # M5 turn recorded it).  M5.1 keeps both active records; diagnostics must expose the
        # disagreement at kind level without leaking either value.
        _seed_constraint(shadow_db, user_key=user_key, task_id=task_id, value="30")

        response = _shopping(harness, session, **request)
        assert response.status_code == 200, response.text
        # SHADOW never acts on the state: the answer equals OFF.
        assert _normalised(response.json()) == off_body
        diagnostics = _service(harness).state_diagnostics(task_id)
        assert diagnostics is not None
        assert diagnostics["trusted_constraint_count"] >= 1
        match = diagnostics["constraint_match"]
        assert match["matched"] >= 1
        # A bounded kind-level difference is reported -- kinds only, no values.
        assert match["extra_in_state"] == ["price_max"]
        assert match["missing_from_state"] == []
        blob = _diagnostics_blob(diagnostics)
        assert "under $20" not in blob
        assert user_key not in blob


def test_diagnostics_are_bounded_and_json_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostics keep at most the last 16 tasks and are always JSON-serialisable."""
    monkeypatch.delenv(ENV_STATE_MODE, raising=False)
    database = _state_path(tmp_path, "direct.sqlite3")
    integration = ShoppingStateIntegration(mode=StateMode.SHADOW, database=database)
    try:
        for index in range(MAX_DIAGNOSTIC_TASKS + 1):
            task_id = f"task-{index:02d}"
            outcome = integration.integrate_turn(
                user_key="bounded-user",
                session_id=task_id,
                task_id=task_id,
                turn_key=f"turn-{index}",
                message=NEUTRAL_MESSAGE,
                intent_label="recommend",
                preferences=(),
                constraints=(),
                trusted_constraints=(),
            )
            assert outcome is not None
            _diagnostics_blob(outcome.diagnostics)  # JSON-safe on every successful turn
        # Oldest evicted, newest kept.
        assert integration.diagnostics("task-00") is None
        newest = integration.diagnostics(f"task-{MAX_DIAGNOSTIC_TASKS:02d}")
        assert newest is not None
        assert newest["turn_count"] == 1
    finally:
        integration.close()
