"""M8 — decision-level evaluation tests (offline lab only).

Covers: trajectory record/replay, metric correctness and NOT_EXECUTED semantics, the
controlled A0/A1/A2 comparison on fixed tasks, service recording + final-choice identity
validation, and the default-off / no-target-leakage boundaries.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.agent_service import DemoAgentError  # noqa: E402
from recommendation.evaluation.decision import (  # noqa: E402
    ARM_A0,
    ARM_A1,
    ARM_A2,
    DecisionCandidate,
    DecisionTaskLabels,
    DecisionTaskView,
    DecisionTrajectoryRecorder,
    Feedback,
    MetricResult,
    PreferenceChange,
    RankingRefreshRecord,
    compute_decision_metrics,
    decision_trajectory_enabled,
    run_controlled_comparison,
)
from tests.agent_reranking_fixture import CANDIDATE_ROWS, build_index  # noqa: E402
from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402

SHOPPING = "/v1/demo/sessions/{session}/agent/shopping"
CREATE = "/v1/demo/sessions"
ENV_M8 = "AGENTRECX_M8_TRAJECTORY"
ENV_M7 = "AGENTRECX_M7_GROUNDED_ACTIONS"
ENV_M7_BUDGET = "AGENTRECX_M7_BUDGET"
ENV_STATE_MODE = "AGENTRECX_STATE_MODE"
ENV_STATE_DB = "AGENTRECX_STATE_DB"


def _records() -> dict[str, Any]:
    return dict(build_index().records)


def _trajectory(arm: str = ARM_A2, *, real_choice: str | None = None) -> Any:
    recorder = DecisionTrajectoryRecorder(
        trajectory_id=f"traj-{arm}",
        task_id="task-1",
        user_key="user-1",
        arm=arm,
        seed=0,
        budget=2,
        k=3,
    )
    recorder.begin_action(
        action="ASK",
        displayed=(
            DecisionCandidate("cand-red", 1, {"price": 25.0}),
            DecisionCandidate("cand-blue", 2, {"price": 40.0}),
            DecisionCandidate("cand-black", 3, {"price": 15.0}),
        ),
        evidence={"attribute": "price"},
        budget_remaining=1,
    )
    recorder.record_feedback(
        option_id="pref:price:lower",
        label="Lower price",
        action="lower",
        attribute="price",
        accepted=True,
        preference_changes=(PreferenceChange("price", "lower", None, "Lower price"),),
    )
    recorder.record_refresh(
        RankingRefreshRecord(
            ran=True,
            changed=True,
            order_before=("cand-red", "cand-blue", "cand-black"),
            order_after=("cand-black", "cand-red", "cand-blue"),
        )
    )
    recorder.finalize(("cand-black", "cand-red", "cand-blue"))
    if real_choice is not None:
        recorder.record_choice(real_choice, source="user")
    return recorder.trajectory()


def test_trajectory_round_trip_is_deterministic() -> None:
    trajectory = _trajectory(real_choice="cand-black")
    payload = trajectory.as_dict()
    restored = type(trajectory).from_dict(payload)
    assert restored.as_dict() == payload
    assert restored.digest() == trajectory.digest()
    encoded = json.dumps(payload, sort_keys=True)
    assert "target" not in encoded
    assert "cand-black" in restored.final_top_k


def test_metrics_compute_only_with_real_choice_and_labels() -> None:
    labels = {
        "task-1": DecisionTaskLabels(
            task_id="task-1", target_ids=("cand-black",), label_source="synthetic_contract"
        )
    }
    real = _trajectory(real_choice="cand-black")
    metrics = compute_decision_metrics((real,), labels, k=3)
    assert metrics.recall_at_k.status == "MEASURED"
    assert metrics.recall_at_k.value == 1.0
    assert metrics.team_accuracy.status == "MEASURED"
    assert metrics.team_accuracy.value == 1.0
    assert metrics.conditional_selection_accuracy.status == "MEASURED"
    assert metrics.conditional_selection_accuracy.value == 1.0
    assert metrics.interaction_cost.status == "MEASURED"
    assert metrics.evidence_coverage.status == "MEASURED"
    assert metrics.ranking_change_rate.status == "MEASURED"
    assert metrics.ranking_change_rate.value == 1.0

    scripted = _trajectory()
    scripted_metrics = compute_decision_metrics((scripted,), labels, k=3)
    assert scripted_metrics.recall_at_k.status == "MEASURED"
    assert scripted_metrics.team_accuracy.status == "NOT_EXECUTED"
    assert scripted_metrics.conditional_selection_accuracy.status == "NOT_EXECUTED"
    assert scripted_metrics.team_accuracy.value is None
    assert "real user final choice" in scripted_metrics.team_accuracy.reason

    no_labels = compute_decision_metrics((real,), {}, k=3)
    assert no_labels.recall_at_k.status == "NOT_EXECUTED"
    assert no_labels.team_accuracy.status == "NOT_EXECUTED"


def test_controlled_arms_are_fixed_and_reproducible() -> None:
    records = _records()
    tasks = (
        DecisionTaskView(
            task_id="t1",
            user_key="u1",
            candidate_ids=("cand-red", "cand-blue", "cand-black"),
            records=records,
            message="recommend hiking gear",
            scripted_feedback={"weight": "pref:weight:lower"},
            budget=1,
            k=3,
        ),
        DecisionTaskView(
            task_id="t2",
            user_key="u2",
            candidate_ids=("cand-red", "cand-blue", "cand-black"),
            records=records,
            message="recommend gear under a tight budget",
            scripted_feedback={"price": "pref:price:lower"},
            budget=1,
            k=3,
        ),
    )
    labels = {
        "t1": DecisionTaskLabels("t1", ("cand-black",), label_source="synthetic_contract"),
        "t2": DecisionTaskLabels("t2", ("cand-black",), label_source="synthetic_contract"),
    }
    first = run_controlled_comparison(tasks, labels, seed=7, k=3, budget=1)
    second = run_controlled_comparison(tasks, labels, seed=7, k=3, budget=1)
    assert first.report_digest == second.report_digest
    assert [report.arm for report in first.arms] == [ARM_A0, ARM_A1, ARM_A2]
    assert first.label_source == "synthetic_contract"
    assert first.quality_status == "SYNTHETIC_CONTRACT_ONLY_NOT_QUALITY"
    for arm in first.arms:
        assert arm.task_count == 2
        assert arm.metrics.team_accuracy.status == "NOT_EXECUTED"
        for trajectory in arm.trajectories:
            assert "target" not in json.dumps(trajectory.as_dict(), sort_keys=True)
    a0 = next(arm for arm in first.arms if arm.arm == ARM_A0)
    a1 = next(arm for arm in first.arms if arm.arm == ARM_A1)
    a2 = next(arm for arm in first.arms if arm.arm == ARM_A2)
    assert all(not trajectory.steps for trajectory in a0.trajectories)
    assert all(
        step.action == "ASK"
        for trajectory in a1.trajectories
        for step in trajectory.steps
    )
    assert any(
        step.action in {"COMPARE", "EXPLAIN"}
        for trajectory in a2.trajectories
        for step in trajectory.steps
    )
    assert a0.metrics.ranking_change_rate.status == "NOT_EXECUTED"


@contextmanager
def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    m8: str | None = "on",
) -> Iterator[Any]:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    state_db = root / "state.sqlite3"
    monkeypatch.setenv(ENV_M7, "on")
    monkeypatch.setenv(ENV_M7_BUDGET, "1")
    monkeypatch.setenv(ENV_STATE_MODE, "shadow")
    monkeypatch.setenv(ENV_STATE_DB, str(state_db))
    if m8 is None:
        monkeypatch.delenv(ENV_M8, raising=False)
    else:
        monkeypatch.setenv(ENV_M8, m8)
    harness = build_harness(root, memory_db=root / "memory.sqlite3")
    try:
        yield harness
    finally:
        close_harness(harness)


def _identity(harness: Any, session_id: str) -> tuple[str, str]:
    session = harness.runtime.sessions.get(session_id)
    return session.user_key, session.session_id


def _create(harness: Any) -> str:
    response = harness.client.post(CREATE, json={"profile_id": PROFILE_A})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _shop(harness: Any, session_id: str, message: str, **overrides: Any) -> Any:
    body = {
        "message": message,
        "k": 3,
        "sources": ["history"],
        "preference_discovery": True,
    }
    body.update(overrides)
    return harness.client.post(SHOPPING.format(session=session_id), json=body)


def test_service_records_trajectory_and_choice_identity(tmp_path: Path, monkeypatch) -> None:
    with _harness(tmp_path, monkeypatch, name="service") as harness:
        session_id = _create(harness)
        user_key, task_id = _identity(harness, session_id)
        first = _shop(harness, session_id, "recommend hiking gear").json()
        option = next(
            item for item in first["preference"]["options"] if item["action"] == "lower"
        )
        final = _shop(
            harness,
            session_id,
            "choose",
            clarification_id=first["clarification"]["clarification_id"],
            clarification_option=option["option_id"],
        ).json()
        assert final["terminal"]["status"] == "finished"
        service = harness.runtime.agent_service()
        trajectory = service.decision_trajectory(task_id)
        assert trajectory is not None
        assert trajectory["task_id"] == task_id
        assert trajectory["final_top_k"] == [
            item["parent_asin"] for item in final["recommendations"]
        ]
        assert trajectory["steps"][0]["action"] in {"ASK", "COMPARE", "EXPLAIN"}
        assert trajectory["steps"][0]["displayed_candidates"]
        assert trajectory["steps"][0]["feedback"]["option_id"] == option["option_id"]
        assert trajectory["steps"][0]["refresh"] is not None
        # Identity validation: a wrong user/task or an item outside final top-K is refused.
        with pytest.raises(DemoAgentError):
            service.record_decision_choice(
                task_id=task_id,
                user_key="wrong-user",
                session_id=task_id,
                selected_parent_asin=trajectory["final_top_k"][0],
            )
        with pytest.raises(DemoAgentError):
            service.record_decision_choice(
                task_id=task_id,
                user_key=user_key,
                session_id=task_id,
                selected_parent_asin="not-a-candidate",
            )
        choice = service.record_decision_choice(
            task_id=task_id,
            user_key=user_key,
            session_id=task_id,
            selected_parent_asin=trajectory["final_top_k"][0],
        )
        assert choice["choice"] == {
            "selected_candidate_id": trajectory["final_top_k"][0],
            "source": "user",
            "no_suitable_item": False,
        }
        encoded = json.dumps(choice, sort_keys=True)
        assert "target" not in encoded


def test_default_off_records_nothing(tmp_path: Path, monkeypatch) -> None:
    with _harness(tmp_path, monkeypatch, name="off", m8=None) as harness:
        session_id = _create(harness)
        user_key, task_id = _identity(harness, session_id)
        first = _shop(harness, session_id, "recommend hiking gear").json()
        assert decision_trajectory_enabled(None) is False
        assert harness.runtime.agent_service().decision_trajectory(task_id) is None
        with pytest.raises(DemoAgentError):
            harness.runtime.agent_service().record_decision_choice(
                task_id=task_id,
                user_key=user_key,
                session_id=task_id,
                selected_parent_asin="cand-red",
            )
