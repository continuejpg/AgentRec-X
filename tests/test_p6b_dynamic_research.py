"""Phase 6B: bounded, observation-driven product research in the shopping loop.

The scripted inner policy recommends once and then proposes FINISH.  The dynamic wrapper is
the only component that can add a research action, and every action it adds is derived from
the current request and the last Observation.  No external model is called.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    GroundedReasoner,
    LoopLimits,
    PolicyActionError,
    ReasoningExecutor,
)
from recommendation.demo.agent_schemas import AgentRecommendRequest  # noqa: E402
from recommendation.demo.research_policy import (  # noqa: E402
    DynamicProductResearchPolicy,
    requested_attributes,
    wants_comparison,
)
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402
from tests.demo_fixture import build_harness  # noqa: E402


class _RecommendThenFinish:
    """The fixed baseline: recommend once, then finish.  No research is planned."""

    name = "fixed-recommend-finish"

    def choose(self, context: Any) -> ActionProposal:
        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3)
        return ActionProposal(action=ActionKind.FINISH)


def _dynamic_harness(
    *,
    limits: LoopLimits | None = None,
    reasoning: Any | None = None,
) -> Any:
    harness = build_control_harness(
        policy=_RecommendThenFinish(),
        limits=limits or LoopLimits(max_steps=8, max_tool_calls=5, max_retries=1),
    )
    harness.controller._reasoning = reasoning or ReasoningExecutor(  # noqa: SLF001
        GroundedReasoner(harness.parts["enricher"].metadata)
    )
    harness.controller._policy = DynamicProductResearchPolicy(  # noqa: SLF001
        harness.controller._policy
    )
    return harness


def _run(harness: Any, message: str) -> Any:
    return harness.controller.run(message, HISTORY, run_id="p6b")


# --------------------------------------------------------------------------- #
# Dynamic decisions
# --------------------------------------------------------------------------- #


def test_no_research_when_candidates_are_sufficient() -> None:
    harness = _dynamic_harness()
    result = _run(harness, "recommend outdoor gear")
    assert result.trajectory.actions() == ("recommend_from_history", "finish")
    assert not any(
        step.observation and step.observation.get("kind", "").startswith("product_")
        for step in result.trajectory.steps
    )


def test_explicit_comparison_calls_compare_with_requested_attributes() -> None:
    harness = _dynamic_harness()
    result = _run(harness, "compare the first two by price and color")
    assert result.trajectory.actions() == (
        "recommend_from_history",
        "compare_products",
        "finish",
    )
    comparison = next(
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "product_comparison"
    )
    assert comparison["products"] == ("cand-red", "cand-blue")
    assert comparison["requested_attributes"] == ("price", "color")
    assert comparison["status"] == "ok"
    assert result.state["tool_result"].recommendations[0].parent_asin == "cand-red"


def test_missing_key_attribute_calls_inspect_then_clarifies() -> None:
    harness = _dynamic_harness()
    result = _run(harness, "is the material leather?")
    assert result.trajectory.actions() == (
        "recommend_from_history",
        "inspect_product",
        "ask_clarification",
    )
    inspection = next(
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "product_inspection"
    )
    assert inspection["tool"] == "inspect_product"
    assert inspection["parent_asin"] == "cand-red"
    assert "material" in inspection["missing_attributes"]
    clarification = next(
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "clarification"
    )
    assert clarification["question"].startswith(
        "The catalogue does not state material"
    )
    assert result.status.value == "waiting_for_user"


def test_comparison_missing_a_field_inspects_once_then_asks() -> None:
    harness = _dynamic_harness()
    result = _run(harness, "compare the first two by material")
    assert result.trajectory.actions() == (
        "recommend_from_history",
        "compare_products",
        "inspect_product",
        "ask_clarification",
    )
    inspections = [
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "product_inspection"
    ]
    assert len(inspections) == 1
    assert inspections[0]["parent_asin"] == "cand-red"
    # The same product was never inspected twice, even though the field is still missing.
    inspect_proposals = [
        step.action_proposal
        for step in result.trajectory.steps
        if step.action_proposal and step.action_proposal.get("action") == "inspect_product"
    ]
    assert len(inspect_proposals) == 1


def test_compare_only_uses_grounded_run_candidates_not_message_identities() -> None:
    harness = _dynamic_harness()
    result = _run(
        harness,
        "compare cand-green and cand-black by price",
    )
    proposals = [
        step.action_proposal
        for step in result.trajectory.steps
        if step.action_proposal
    ]
    compared = next(
        proposal["arguments"]["parent_asins"]
        for proposal in proposals
        if proposal["action"] == "compare_products"
    )
    assert compared == ["cand-red", "cand-blue"]
    assert "cand-green" not in compared
    grounded = {
        item.parent_asin for item in result.state["tool_result"].recommendations
    }
    assert set(compared) <= grounded


class _FailingResearch:
    """A reasoning seam that returns a real failure Observation for every research call."""

    name = "failing-product-research"

    def available_actions(self) -> tuple[ActionKind, ...]:
        return (ActionKind.INSPECT_PRODUCT, ActionKind.COMPARE_PRODUCTS)

    def execute(self, action: Any, *, authorized_candidates: Any = None) -> Any:
        raise PolicyActionError("the product catalogue is temporarily unavailable", code="research_failed")


def test_research_failure_is_observed_and_not_retried() -> None:
    harness = _dynamic_harness(reasoning=_FailingResearch())
    result = _run(harness, "compare the first two by price and color")
    actions = result.trajectory.actions()
    assert actions[:3] == (
        "recommend_from_history",
        "compare_products",
        "finish",
    )
    assert actions.count("compare_products") == 1
    assert actions.count("inspect_product") == 0
    failure = next(
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("kind") == "product_research_failure"
    )
    assert failure["status"] == "failed"
    assert failure["tool"] == "compare_products"
    assert failure["code"] == "research_failed"
    assert failure["products"] == ("cand-red", "cand-blue")


def test_tool_budget_exhaustion_stops_research_without_false_success() -> None:
    harness = _dynamic_harness(
        limits=LoopLimits(max_steps=5, max_tool_calls=1, max_retries=1)
    )
    result = _run(harness, "compare the first two by price and color")
    assert result.trajectory.actions() == ("recommend_from_history", "finish")
    assert not any(
        step.observation and step.observation.get("kind") == "product_comparison"
        for step in result.trajectory.steps
    )
    assert result.state["tool_result"].recommendations


def test_research_is_bounded_to_two_calls_per_run() -> None:
    harness = _dynamic_harness()
    result = _run(harness, "compare the first two by material")
    research_actions = [
        step.action_proposal["action"]
        for step in result.trajectory.steps
        if step.action_proposal
        and step.action_proposal["action"] in ("inspect_product", "compare_products")
    ]
    assert len(research_actions) <= 2
    assert research_actions.count("inspect_product") <= 1
    assert research_actions.count("compare_products") <= 1


def test_request_vocabulary_is_deterministic_and_not_free_text() -> None:
    assert wants_comparison("Compare these two by price")
    assert not wants_comparison("Recommend something for a hike")
    assert requested_attributes("compare price, colour and material") == (
        "price", "color", "material"
    )
    assert requested_attributes("recommend gear") == ()


# --------------------------------------------------------------------------- #
# Response visibility
# --------------------------------------------------------------------------- #


def test_shopping_response_exposes_research_and_keeps_candidate_order(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        service = harness.runtime.agent_service()
        response = service.run(
            AgentRecommendRequest(
                message="compare the first two by price and color",
                history=list(HISTORY),
                k=3,
            ),
            dynamic_research=True,
        )
        assert response.research is not None
        assert response.research.executed is True
        assert response.research.actions[0].tool == "compare_products"
        assert response.research.actions[0].products == ("cand-red", "cand-blue")
        assert response.research.actions[0].rows
        assert [item.parent_asin for item in response.recommendations] == [
            "cand-red", "cand-blue", "cand-black",
        ]
        assert "compare_products" in response.trajectory.actions
    finally:
        harness.runtime.close()


def test_shopping_response_shows_question_and_hides_products_when_fact_missing(
    tmp_path: Path,
) -> None:
    from tests.demo_fixture import PROFILE_A

    harness = build_harness(tmp_path)
    try:
        session = harness.create_session(PROFILE_A)["session_id"]
        body = harness.client.post(
            f"/v1/demo/sessions/{session}/agent/shopping",
            json={
                "message": "compare the first two by material",
                "k": 3,
                "sources": ["history", "catalog_search"],
            },
        )
        assert body.status_code == 200, body.text
        payload = body.json()
        assert payload["terminal"]["status"] == "waiting_for_user"
        assert payload["clarification"]["kind"] == "unverified"
        assert "material" in payload["clarification"]["question"]
        assert payload["recommendations"] == []
        assert payload["research"]["executed"] is True
        assert payload["research"]["actions"][0]["tool"] == "compare_products"
        assert "material" in payload["research"]["actions"][0]["missing_attributes"]
    finally:
        harness.runtime.close()
