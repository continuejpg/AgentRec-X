"""Phase 6E: safe degradation when product research fails.

These tests exercise the actual ``product_research_failure`` branch and the completion guard.
They use a controlled failing research seam, so no external service or model is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    ConstraintOrigin,
    GroundedReasoner,
    LoopLimits,
    PolicyActionError,
    ReasoningExecutor,
    TaskConstraint,
    TaskIntent,
    TaskState,
)
from recommendation.demo.agent_trajectory import build_research_view  # noqa: E402
from recommendation.demo.agent_service import _task_state  # noqa: E402
from recommendation.demo.agent_schemas import AgentRecommendRequest  # noqa: E402
from recommendation.demo.research_policy import DynamicProductResearchPolicy  # noqa: E402
from recommendation.memory import PreferenceKind  # noqa: E402
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402
from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402


class _RecommendThenFinish:
    name = "recommend-then-finish"

    def choose(self, context: Any) -> ActionProposal:
        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)
        return ActionProposal(action=ActionKind.FINISH)


class _FailingResearch:
    name = "failing-product-research"

    def __init__(self) -> None:
        self.calls = 0

    def available_actions(self) -> tuple[ActionKind, ...]:
        return (ActionKind.INSPECT_PRODUCT, ActionKind.COMPARE_PRODUCTS)

    def execute(self, action: Any, *, authorized_candidates: Any = None) -> Any:
        self.calls += 1
        raise PolicyActionError("catalogue temporarily unavailable", code="research_failed")


def _run(
    *,
    intent: TaskIntent,
    task_state: TaskState | None = None,
    with_eligibility: bool = False,
) -> tuple[Any, _FailingResearch]:
    state = task_state or TaskState(intent=intent)
    harness = build_control_harness(
        policy=_RecommendThenFinish(),
        limits=LoopLimits(max_steps=8, max_tool_calls=5, max_retries=1),
        with_eligibility=with_eligibility,
        task_state=state,
    )
    failing = _FailingResearch()
    harness.controller._reasoning = failing  # noqa: SLF001 - controlled failure seam
    harness.controller._policy = DynamicProductResearchPolicy(  # noqa: SLF001
        harness.controller._policy
    )
    result = harness.controller.run(
        "compare the first two by price and color", HISTORY, task_state=state
    )
    return result, failing


def _research_failure(result: Any) -> dict[str, Any]:
    return next(
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "product_research_failure"
    )


def test_optional_failure_degrades_to_verified_candidates() -> None:
    result, failing = _run(intent=TaskIntent.RECOMMEND)
    assert result.succeeded is True
    assert result.control.termination_reason.value == "completed"
    assert failing.calls == 1
    assert result.trajectory.actions().count("compare_products") == 1
    # The already-verified candidates and their order survive the failed research call.
    identities = tuple(
        item.parent_asin for item in result.state["tool_result"].recommendations
    )
    assert identities == ("cand-red", "cand-blue", "cand-black")
    failure = _research_failure(result)
    assert failure["tool"] == "compare_products"
    assert failure["status"] == "failed"
    assert failure["code"] == "research_failed"
    # The final response is a normal list of those identities, not a fabricated comparison.
    assert "cand-red" in (result.final_response or "")
    assert "cand-blue" in (result.final_response or "")
    view = build_research_view(result)
    assert view is not None
    assert any(action.status == "failed" for action in view.actions)
    assert "did not complete" in view.note
    assert "price" not in view.note.lower() or "failed" in view.note.lower()


def test_explicit_comparison_failure_is_not_a_completion() -> None:
    result, failing = _run(intent=TaskIntent.COMPARE)
    assert failing.calls == 1
    assert result.trajectory.actions().count("compare_products") == 1
    assert result.control.termination_reason.value != "completed"
    assert result.succeeded is False
    # The prior candidate set is retained for audit, but it is not presented as the answer.
    stored = tuple(
        item.parent_asin for item in result.state["tool_result"].recommendations
    )
    assert stored == ("cand-red", "cand-blue", "cand-black")
    assert not result.final_response or "cand-red" not in result.final_response
    assert _research_failure(result)["status"] == "failed"


def test_failure_with_unresolved_hard_constraint_is_not_degraded() -> None:
    state = TaskState(intent=TaskIntent.RECOMMEND).with_constraint(
        TaskConstraint(
            kind=PreferenceKind.MATERIAL,
            value="leather",
            origin=ConstraintOrigin.CURRENT_TURN,
            source_text="must be leather",
        )
    )
    result, failing = _run(
        intent=TaskIntent.RECOMMEND, task_state=state, with_eligibility=True
    )
    assert failing.calls == 1
    assert result.control.termination_reason.value != "completed"
    assert result.succeeded is False
    assert result.trajectory.actions().count("compare_products") == 1
    assert _research_failure(result)["status"] == "failed"
    # Identity remains grounded but the unresolved constraint blocks presentation.
    assert result.state["tool_result"].recommendations
    assert not result.final_response or "cand-red" not in result.final_response


def test_satisfied_hard_constraint_allows_degradation() -> None:
    state = TaskState(intent=TaskIntent.RECOMMEND).with_constraint(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX,
            value="30",
            origin=ConstraintOrigin.CURRENT_TURN,
            source_text="under 30",
        )
    )
    result, failing = _run(
        intent=TaskIntent.RECOMMEND, task_state=state, with_eligibility=True
    )
    assert failing.calls == 1
    assert result.succeeded is True
    assert result.control.termination_reason.value == "completed"
    assert result.trajectory.actions().count("compare_products") == 1
    assert _research_failure(result)["status"] == "failed"
    final = result.final_response or ""
    assert "cand-red" in final and "cand-black" in final
    assert "cand-blue" not in final


def test_failure_does_not_retry_the_same_tool() -> None:
    result, failing = _run(intent=TaskIntent.RECOMMEND)
    assert failing.calls == 1
    assert result.trajectory.actions().count("compare_products") == 1
    assert result.trajectory.actions().count("inspect_product") == 0


def test_task_intent_distinguishes_required_from_optional_research() -> None:
    optional = _task_state(
        AgentRecommendRequest(
            message="recommend gear with good material",
            history=list(HISTORY),
            k=3,
            sources=("history",),
        )
    )
    required_details = _task_state(
        AgentRecommendRequest(
            message="is the material leather?",
            history=list(HISTORY),
            k=3,
            sources=("history",),
        )
    )
    required_compare = _task_state(
        AgentRecommendRequest(
            message="compare the first two by price and color",
            history=list(HISTORY),
            k=3,
            sources=("history",),
        )
    )
    assert optional.intent is TaskIntent.RECOMMEND
    assert required_details.intent is TaskIntent.DETAILS
    assert required_compare.intent is TaskIntent.COMPARE


def test_shopping_response_exposes_partial_failure_and_keeps_candidates(
    tmp_path: Path,
) -> None:
    harness = build_harness(tmp_path)
    try:
        service = harness.runtime.agent_service()
        failing = _FailingResearch()
        service._reasoning_executor = failing  # noqa: SLF001 - controlled failure seam
        session = harness.create_session(PROFILE_A)["session_id"]
        response = harness.client.post(
            f"/v1/demo/sessions/{session}/agent/shopping",
            json={
                "message": "recommend gear with good material",
                "k": 3,
                "sources": ["history"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red",
            "cand-blue",
            "cand-black",
        ]
        assert body["research"]["executed"] is True
        assert body["research"]["actions"][-1]["status"] == "failed"
        assert "did not complete" in body["research"]["note"]
        assert all(item["reason"] for item in body["recommendations"])
        assert failing.calls == 1
    finally:
        harness.runtime.close()


def test_required_comparison_failure_keeps_the_run_incomplete(
    tmp_path: Path,
) -> None:
    harness = build_harness(tmp_path)
    try:
        service = harness.runtime.agent_service()
        failing = _FailingResearch()
        service._reasoning_executor = failing  # noqa: SLF001 - controlled failure seam
        session = harness.create_session(PROFILE_A)["session_id"]
        response = harness.client.post(
            f"/v1/demo/sessions/{session}/agent/shopping",
            json={
                "message": "compare the first two by price and color",
                "k": 3,
                "sources": ["history"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] != "finished"
        assert body["terminal"]["succeeded"] is False
        assert body["recommendations"] == []
        assert "cand-red" not in body["message"]
        assert body["research"]["actions"][-1]["status"] == "failed"
        assert "did not complete" in body["research"]["note"]
        assert failing.calls == 1
    finally:
        harness.runtime.close()
