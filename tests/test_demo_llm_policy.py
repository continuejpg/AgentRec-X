"""Tests for the optional LLM decision policy layer.

All provider calls are mocked with the repository's existing ``ScriptedModelClient``; the live
provider is never required.  The point of these tests is the authority boundary: the model
proposes a plan, trusted code validates and executes it, and nothing the model returns can
become a product id, a score, a piece of evidence, a rank or a mutation result.
"""

from __future__ import annotations

import pathlib

import pytest

from recommendation.control.model_client import ModelCallError, ScriptedModelClient
from recommendation.demo.llm_policy import (
    DECISION_MODES,
    LLM_EXTRACTOR_NAME,
    LLMPolicyError,
    LLMTurnPlan,
    LLMTurnPlanner,
    LLMPlanExtractor,
    RouteAction,
    plan_to_extraction,
    provider_availability,
)
from recommendation.memory import InMemoryPreferenceStore, PreferenceMemoryService
from recommendation.memory.extraction import RuleBasedPreferenceExtractor
from recommendation.memory.schemas import PreferenceExtractionError, PreferenceKind

FREE_TEXT = (
    "I'm going hiking in the rain. Drop the lightweight requirement, "
    "but I'd still prefer something durable."
)
PLAN = {
    "route": "recommend",
    "add": [{"value": "durable", "kind": "feature"}],
    "remove": ["lightweight"],
}


def _planner(*answers: object, max_attempts: int = 2) -> tuple[LLMTurnPlanner, ScriptedModelClient]:
    client = ScriptedModelClient(list(answers), repeat_last=True)
    return LLMTurnPlanner(client, max_attempts=max_attempts), client


def _service(planner: LLMTurnPlanner) -> PreferenceMemoryService:
    return PreferenceMemoryService(InMemoryPreferenceStore(), LLMPlanExtractor(planner))


def _active(service: PreferenceMemoryService, user_key: str = "alice") -> set[tuple[str, str]]:
    snapshot = service.get_active_preferences(user_key)
    return {(entry.kind.value, entry.value) for entry in snapshot.active_entries}


# --------------------------------------------------------------------------- #
# the provider abstraction is the existing one, and it is actually invoked
# --------------------------------------------------------------------------- #


def test_planner_invokes_the_existing_provider_seam() -> None:
    planner, client = _planner(PLAN)
    plan = planner.plan(FREE_TEXT)

    assert client.call_count == 1, "the provider abstraction was called exactly once"
    assert plan.route is RouteAction.RECOMMEND
    assert [item.value for item in plan.add] == ["durable"]
    assert plan.remove == ("lightweight",)
    # the model was given the user's own words, not product data
    sent = client.last_request
    assert sent is not None and sent.context_payload == {"user_request": FREE_TEXT}


def test_planner_uses_a_json_contract_and_no_chain_of_thought() -> None:
    planner, client = _planner(PLAN)
    planner.plan(FREE_TEXT)

    sent = client.last_request
    assert sent is not None
    assert "json" in sent.system_prompt.lower(), "the provider's json mode needs the word"
    offered = [entry["action"] for entry in sent.action_schema]
    assert offered == ["recommend", "direct", "add_preference", "remove_preference"]


def test_free_language_becomes_a_structured_plan() -> None:
    """The reason LLM mode exists: freer wording than the bounded parser accepts."""
    # The bounded parser extracts NOTHING from this turn: no leading "I prefer", and the
    # retraction names a value after "Drop the ... requirement", which its syntax does not
    # cover.  That gap is the reason the optional LLM layer exists.
    rule = RuleBasedPreferenceExtractor().extract(FREE_TEXT)
    assert not rule.preferences and not rule.removals, (
        "the bounded parser has no reading for this turn; LLM mode does"
    )

    planner, _client = _planner(PLAN)
    plan = planner.plan(FREE_TEXT)
    assert [item.value for item in plan.add] == ["durable"], "LLM mode reads the addition"
    assert plan.remove == ("lightweight",), "and the removal"


# --------------------------------------------------------------------------- #
# validated actions execute through the unchanged trusted path
# --------------------------------------------------------------------------- #


def test_validated_plan_persists_through_the_memory_service() -> None:
    planner, _client = _planner(PLAN)
    service = _service(planner)

    service.process_turn(user_message=FREE_TEXT, user_key="alice", turn_id="t1")

    snapshot = service.get_active_preferences("alice")
    values = {(entry.kind.value, entry.value) for entry in snapshot.active_entries}
    assert values == {("feature", "durable")}
    entry = snapshot.active_entries[0]
    assert entry.extractor == LLM_EXTRACTOR_NAME, "provenance records where it came from"


def test_removal_is_value_scoped_and_actually_removes() -> None:
    store = InMemoryPreferenceStore()
    service = PreferenceMemoryService(store, RuleBasedPreferenceExtractor())
    service.process_turn(
        user_message="I prefer lightweight and durable products.",
        user_key="alice",
        turn_id="t1",
    )
    assert _active(service) == {("feature", "lightweight"), ("feature", "durable")}

    planner, _client = _planner(PLAN)
    llm_service = PreferenceMemoryService(store, LLMPlanExtractor(planner))
    llm_service.process_turn(user_message=FREE_TEXT, user_key="alice", turn_id="t2")

    assert _active(llm_service) == {("feature", "durable")}, "only lightweight was withdrawn"


def test_plan_to_extraction_maps_route_and_preferences() -> None:
    plan = LLMTurnPlan.model_validate(PLAN)
    extraction = plan_to_extraction(plan, source_text=FREE_TEXT)

    assert [p.value for p in extraction.preferences] == ["durable"]
    assert extraction.preferences[0].kind is PreferenceKind.FEATURE
    assert [r.value for r in extraction.removals] == ["lightweight"]
    assert extraction.removals[0].kind is None, "value-scoped, matching the M9 contract"


# --------------------------------------------------------------------------- #
# the model cannot carry recommendation truth
# --------------------------------------------------------------------------- #


def test_plan_schema_has_no_field_for_truth() -> None:
    fields = set(LLMTurnPlan.model_fields)
    assert fields == {"route", "add", "remove"}
    for forbidden in ("parent_asin", "item_id", "score", "sasrec_score", "evidence",
                      "rank", "rerank", "match", "violation", "candidate"):
        assert forbidden not in fields


def test_model_cannot_inject_a_product_identity_or_score() -> None:
    injected = dict(PLAN)
    injected["parent_asin"] = "B000INJECTED"
    injected["sasrec_score"] = 99.0
    injected["final_rank"] = 1

    planner, _client = _planner(injected)
    with pytest.raises(LLMPolicyError) as excinfo:
        planner.plan(FREE_TEXT)

    assert excinfo.value.code == "invalid_plan", "an injected truth field is a rejection"


def test_model_cannot_inject_truth_inside_a_preference_entry() -> None:
    injected = {"route": "recommend", "add": [{"value": "durable", "item_id": 7}]}
    planner, _client = _planner(injected)
    with pytest.raises(LLMPolicyError):
        planner.plan(FREE_TEXT)


def test_unknown_preference_kind_is_rejected() -> None:
    planner, _client = _planner({"route": "recommend", "add": [{"value": "durable", "kind": "vibes"}]})
    with pytest.raises(LLMPolicyError) as excinfo:
        planner.plan(FREE_TEXT)
    assert excinfo.value.code == "invalid_plan"


# --------------------------------------------------------------------------- #
# failure behaviour
# --------------------------------------------------------------------------- #


def test_malformed_output_is_rejected_safely() -> None:
    planner, _client = _planner("I think you would like a nice tent!")
    with pytest.raises(LLMPolicyError) as excinfo:
        planner.plan(FREE_TEXT)
    assert excinfo.value.code == "invalid_plan"


def test_unsupported_action_is_rejected() -> None:
    planner, _client = _planner({"route": "select_source", "add": [], "remove": []})
    with pytest.raises(LLMPolicyError) as excinfo:
        planner.plan(FREE_TEXT)
    assert excinfo.value.code == "invalid_plan"


def test_blank_and_overlong_values_are_rejected() -> None:
    for bad in (
        {"route": "recommend", "add": [{"value": "   "}]},
        {"route": "recommend", "remove": ["  "]},
        {"route": "recommend", "remove": ["x" * 121]},
    ):
        planner, _client = _planner(bad)
        with pytest.raises(LLMPolicyError):
            planner.plan(FREE_TEXT)


def test_provider_failure_does_not_mutate_memory() -> None:
    client = ScriptedModelClient([ModelCallError("boom", code="timeout")])
    service = _service(LLMTurnPlanner(client, max_attempts=1))

    # Trusted code normalises a collaborator failure rather than trusting it; either way the
    # turn fails closed and nothing is written.
    with pytest.raises(PreferenceExtractionError):
        service.process_turn(user_message=FREE_TEXT, user_key="alice", turn_id="t1")

    assert _active(service) == set(), "a failed call must leave memory untouched"


def test_malformed_then_valid_retries_within_the_bound() -> None:
    planner, client = _planner("not json", PLAN, max_attempts=2)
    plan = planner.plan(FREE_TEXT)

    assert plan.route is RouteAction.RECOMMEND
    assert client.call_count == 2, "one retry on unparseable output, then success"


def test_persistent_malformed_output_fails_closed() -> None:
    planner, client = _planner("still not json", max_attempts=2)
    with pytest.raises(LLMPolicyError) as excinfo:
        planner.plan(FREE_TEXT)
    assert excinfo.value.code == "invalid_plan"
    assert client.call_count == 2, "retries are bounded, not unbounded"


def test_empty_turn_is_refused_before_any_provider_call() -> None:
    planner, client = _planner(PLAN)
    with pytest.raises(LLMPolicyError) as excinfo:
        planner.plan("   ")
    assert excinfo.value.code == "empty_message"
    assert client.call_count == 0


# --------------------------------------------------------------------------- #
# mode availability (no hardcoded labels)
# --------------------------------------------------------------------------- #


def test_availability_reports_a_reason_instead_of_claiming_llm_mode() -> None:
    state = provider_availability()
    assert set(state) == {"available", "reason", "provider", "model"}
    if not state["available"]:
        assert state["reason"] in {"not_configured", "missing_api_key"}


def test_decision_modes_are_the_two_documented_ones() -> None:
    assert DECISION_MODES == ("deterministic", "llm")


# --------------------------------------------------------------------------- #
# the deterministic path is untouched by this layer
# --------------------------------------------------------------------------- #


def test_deterministic_path_does_not_depend_on_the_llm_layer() -> None:
    """Additive by construction: nothing on the serving path imports this module."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for rel in (
        "recommendation/demo/decision.py",
        "recommendation/agent/pipeline.py",
        "recommendation/agent/graph.py",
        "recommendation/demo/serialization.py",
    ):
        source = (root / rel).read_text(encoding="utf-8")
        assert "llm_policy" not in source, f"{rel} must not depend on the LLM layer"


def test_rule_based_extraction_is_unchanged_by_this_layer() -> None:
    rule = RuleBasedPreferenceExtractor().extract("I prefer lightweight and durable products.")
    assert sorted((p.kind.value, p.value) for p in rule.preferences) == [
        ("feature", "durable"), ("feature", "lightweight")
    ]
