"""E-01A-1: the LLM chooses tools, trusted code decides whether that choice may run.

The client here is duck-typed on purpose: the planner reads ``response.text`` and builds a real
``ModelRequest``, so these tests exercise the actual request construction, parsing and validation
without a provider or a network.  Nothing about an unavailable tool is faked - the allowed set is
simply what the caller offers, which is exactly how a not-ready TIGER becomes unreachable.
"""

from __future__ import annotations

import json
import types

import pytest

from recommendation.control.model_client import ModelCallError, ModelRequest
from recommendation.demo.tool_selection import (
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    ToolSelectionError,
    ToolSelectionPlanner,
    validate_selection,
    SourceSelection,
    source_plan_from_selection,
)

ALLOWED = {"history": "sequential purchase history", "catalog_search": "BM25 catalogue text"}
WITH_TIGER = {**ALLOWED, "tiger": "generative semantic-id retrieval"}


class ScriptedClient:
    """Returns prepared answers in order and records every request it was handed."""

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest):
        self.requests.append(request)
        text = self._answers.pop(0) if self._answers else self._answers_last
        self._answers_last = text
        return types.SimpleNamespace(text=text)


def answer(*sources: str, route: str = ROUTE_RECOMMEND, reason: str | None = "because") -> str:
    payload = {"route": route, "selected_sources": list(sources)}
    if reason is not None:
        payload["reason"] = reason
    return json.dumps(payload)


def planner(*answers: str) -> tuple[ToolSelectionPlanner, ScriptedClient]:
    client = ScriptedClient(*answers)
    return ToolSelectionPlanner(client), client


# ---------------------------------------------------------------- happy paths


def test_a_single_legal_tool_is_selected() -> None:
    plan, client = planner(answer("catalog_search"))
    selection = plan.plan(user_request="waterproof jacket under 100", allowed_sources=ALLOWED,
                          budget=3)
    assert selection.route == ROUTE_RECOMMEND
    assert selection.selected_sources == ("catalog_search",)
    assert source_plan_from_selection(selection) == ("catalog_search",)
    assert isinstance(client.requests[0], ModelRequest), "a real seam request must be built"
    offered = {tool["name"] for tool in client.requests[0].context_payload["available_tools"]}
    assert offered == set(ALLOWED), "only the offered tools are disclosed"
    assert client.requests[0].context_payload["tool_budget"] == 3


def test_several_tools_keep_the_model_order() -> None:
    plan, _ = planner(answer("catalog_search", "history"))
    selection = plan.plan(user_request="find me a tent", allowed_sources=WITH_TIGER, budget=4)
    assert selection.selected_sources == ("catalog_search", "history")


def test_the_same_tool_set_can_yield_different_plans() -> None:
    first, _ = planner(answer("history"))
    second, _ = planner(answer("catalog_search", "history"))
    assert first.plan(user_request="something like my last order", allowed_sources=ALLOWED,
                      budget=4).selected_sources == ("history",)
    assert second.plan(user_request="red waterproof jacket", allowed_sources=ALLOWED,
                       budget=4).selected_sources == ("catalog_search", "history")


def test_direct_answers_select_no_tool() -> None:
    plan, _ = planner(answer(route=ROUTE_DIRECT, reason=None))
    selection = plan.plan(user_request="thanks, that helps", allowed_sources=ALLOWED, budget=2)
    assert selection.selected_sources == () and source_plan_from_selection(selection) == ()


# ---------------------------------------------------------------- refusals


def test_an_unoffered_tool_is_refused() -> None:
    plan, _ = planner(answer("similar_item"))
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "unknown_source"


def test_tiger_is_refused_when_it_is_not_offered_this_turn() -> None:
    """The not-ready case: TIGER simply is not in the allowed set until its handshake completes."""
    plan, _ = planner(answer("tiger"))
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "unknown_source"
    offered, _ = planner(answer("tiger"))
    assert offered.plan(user_request="find me a tent", allowed_sources=WITH_TIGER,
                        budget=3).selected_sources == ("tiger",)


def test_exceeding_the_budget_is_refused() -> None:
    plan, _ = planner(answer("history", "catalog_search", "tiger"))
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="everything", allowed_sources=WITH_TIGER, budget=2)
    assert excinfo.value.code == "over_budget"


def test_a_repeated_tool_is_refused() -> None:
    plan, _ = planner(answer("history", "history"))
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "duplicate_source"


def test_a_recommendation_without_sources_is_refused() -> None:
    plan, _ = planner(answer())
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "no_sources"


def test_a_direct_answer_that_selects_tools_is_refused() -> None:
    plan, _ = planner(answer("history", route=ROUTE_DIRECT))
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="hello", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "direct_with_sources"


def test_malformed_output_fails_after_bounded_retries() -> None:
    plan, client = planner("{not json", "{still not json", "{nope")
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "invalid_plan"
    assert plan.calls == 2, "retries are bounded by max_attempts"
    assert client.requests[1].correction, "the retry carries the parse failure forward"


def test_a_recovered_second_attempt_is_accepted() -> None:
    plan, _ = planner("{}", answer("history"))
    selection = plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert selection.selected_sources == ("history",)


def test_a_product_id_or_score_cannot_be_smuggled_in() -> None:
    smuggled = json.dumps({"route": "recommend", "selected_sources": ["history"],
                           "item_id": 12345, "score": 0.9, "provenance": {"sources": ["tiger"]}})
    plan, _ = planner(smuggled, smuggled)
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code == "invalid_plan"


def test_input_contract_violations_fail_before_any_call() -> None:
    plan, client = planner(answer("history"))
    with pytest.raises(ToolSelectionError) as empty:
        plan.plan(user_request="   ", allowed_sources=ALLOWED, budget=3)
    assert empty.value.code == "empty_request"
    with pytest.raises(ToolSelectionError) as none:
        plan.plan(user_request="find me a tent", allowed_sources={}, budget=3)
    assert none.value.code == "no_tools_available"
    with pytest.raises(ToolSelectionError) as budget:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=0)
    assert budget.value.code == "invalid_budget"
    assert client.requests == [], "an invalid input contract must not reach the provider"


def test_a_provider_failure_is_reported_with_its_own_code() -> None:
    class Broken:
        def complete(self, request):
            try:
                raise ModelCallError("provider timeout", code="timeout")
            except TypeError:  # the exception's constructor may not take a code
                raise ModelCallError("provider timeout")

    plan = ToolSelectionPlanner(Broken())
    with pytest.raises(ToolSelectionError) as excinfo:
        plan.plan(user_request="find me a tent", allowed_sources=ALLOWED, budget=3)
    assert excinfo.value.code in ("timeout", "model_error")


def test_validate_selection_is_usable_without_a_model() -> None:
    allowed = ({"name": "history", "description": "h"},)
    from recommendation.demo.tool_selection import ToolOption

    options = (ToolOption(**allowed[0]),)
    ok = SourceSelection(route="recommend", selected_sources=("history",))
    assert validate_selection(ok, allowed=options, budget=1) is ok
    with pytest.raises(ToolSelectionError):
        validate_selection(SourceSelection(route="recommend", selected_sources=("tiger",)),
                           allowed=options, budget=1)
