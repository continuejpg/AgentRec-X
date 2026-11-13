"""Tests for the minimal evidence-aware corrective retrieval path.

The positive case uses real catalogue products: an initial broad keyboard query misses
products with both `mechanical` and `wireless` evidence; a focused corrective query finds
them.  No Qwen/network call is made here.
"""

from __future__ import annotations

import pytest

from experiments.p14c0.electronics_executor import ElectronicsPlannerExecutor, prepare_hard
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.control.corrective_rag import (
    EVIDENCE_GAP,
    NO_GAP,
    REQUIREMENT_AMBIGUITY,
    RETRIEVAL_GAP,
    build_corrective_query,
    classify_evidence_gap,
)
from recommendation.demo.planner_contract import ActionProposal, PlannerAction, PlannerObservation

CURRENT = "I need a mechanical wireless keyboard for my desk."
HARD = [
    {"kind": "attribute", "value": "mechanical", "source_text": "It needs to be mechanical."},
    {"kind": "attribute", "value": "wireless", "source_text": "It needs to be wireless."},
]


@pytest.fixture(scope="module")
def catalogue() -> ElectronicsCatalogue:
    return ElectronicsCatalogue.load()


def test_natural_language_hard_constraints_are_traceable() -> None:
    hard = prepare_hard("I need a wireless mechanical keyboard for my desk.")
    assert {item["value"] for item in hard} == {"wireless", "mechanical"}
    for item in hard:
        start, end = item["source_span"]
        assert "I need a wireless mechanical keyboard for my desk."[start:end].lower() == item["source_text"].lower()
    with_phrase = prepare_hard("I need a keyboard with mechanical switches and wireless connectivity.")
    assert {item["value"] for item in with_phrase} == {"mechanical", "wireless"}
    assert prepare_hard("I need headphones for commuting.") == []
    assert prepare_hard("I do not want wireless headphones.") == []


def test_classify_retrieval_gap_and_avoid_duplicate_query() -> None:
    check = classify_evidence_gap(
        gate_action="NEED_MORE_PRODUCT_EVIDENCE", hard_requirements=HARD, query=CURRENT,
        product_type="keyboard", search_history=("keyboard for desk",),
    )
    assert check.kind == RETRIEVAL_GAP
    assert check.recoverable is True
    assert check.suggested_query
    assert "mechanical" in check.suggested_query and "wireless" in check.suggested_query
    assert check.suggested_query != "keyboard for desk"


def test_unverifiable_facts_do_not_trigger_a_search() -> None:
    check = classify_evidence_gap(
        gate_action="NEED_MORE_PRODUCT_EVIDENCE",
        hard_requirements=[{"kind": "unverifiable", "value": "price"},
                           {"kind": "unverifiable", "value": "warranty"}],
        query="I need a laptop under $500 with a three-year warranty.",
        product_type="laptop",
    )
    assert check.kind == EVIDENCE_GAP
    assert check.recoverable is False
    assert check.suggested_query is None


def test_clarification_is_not_sent_to_query_rewriter() -> None:
    check = classify_evidence_gap(
        gate_action="NEED_USER_CLARIFICATION", hard_requirements=[], query="I need something.",
        product_type=None,
    )
    assert check.kind == REQUIREMENT_AMBIGUITY
    assert check.recoverable is False
    assert check.suggested_query is None


def test_normal_finalize_does_not_trigger_corrective_retrieval(catalogue) -> None:
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query="I need wireless headphones for commuting.",
        preferences=[], hard=[{"kind": "attribute", "value": "wireless", "source_text": "x"}],
        corrective_rag=True,
    )
    observation = PlannerObservation(step=0, query=executor.current_query)
    observation, _, terminal = executor.execute(
        ActionProposal(PlannerAction.RETRIEVE, {"query": "wireless headphones"}, "test"), observation
    )
    assert terminal is False
    observation, tool, terminal = executor.execute(
        ActionProposal(PlannerAction.FINALIZE, {}, "test"), observation
    )
    assert terminal is True
    assert executor.terminal_status == "finalized"
    assert tool["status"] == "finalized"
    assert executor.corrective_attempts == 0
    assert executor.corrective_trace == []


def test_real_catalog_corrective_retrieval_recovers_verified_candidates(catalogue) -> None:
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=CURRENT, preferences=[], hard=HARD,
        corrective_rag=True,
    )
    observation = PlannerObservation(step=0, query=CURRENT)
    observation, _, terminal = executor.execute(
        ActionProposal(PlannerAction.RETRIEVE, {"query": "keyboard for desk"}, "test"), observation
    )
    assert terminal is False
    before = set(observation.candidate_ids)
    observation, first_tool, terminal = executor.execute(
        ActionProposal(PlannerAction.FINALIZE, {}, "test"), observation
    )
    assert terminal is True
    assert first_tool["status"] == "corrective_retrieval_revalidated"
    trace = first_tool["corrective"]
    assert trace["trigger_gate"] == "NEED_MORE_PRODUCT_EVIDENCE"
    assert trace["corrected_query"] != trace["original_query"]
    assert trace["new_candidate_count"] > 0
    assert len(executor.corrective_trace) == 1
    assert executor.corrective_attempts == 1
    assert set(before) <= set(trace["candidate_ids_before"])
    assert executor.terminal_status == "finalized"
    assert executor.selected_candidate_id is not None
    selected = catalogue.get(executor.selected_candidate_id)
    assert selected is not None
    assert selected.attributes.get("mechanical", {}).get("status") == "present"
    assert selected.attributes.get("wireless", {}).get("status") == "present"
    assert first_tool["revalidation"]["gate"]["action"] == "ALLOW_FINALIZE"
    assert first_tool["revalidation"]["gate"]["hard_constraint_status"] == {"hard_constraints": "verified"}


def test_no_gap_for_allow_finalize() -> None:
    check = classify_evidence_gap(
        gate_action="ALLOW_FINALIZE", hard_requirements=HARD, query=CURRENT, product_type="keyboard",
    )
    assert check.kind == NO_GAP
    assert check.recoverable is False
