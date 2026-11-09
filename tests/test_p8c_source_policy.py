"""P8-C source-policy contract tests (CPU; no model/GPU)."""

from __future__ import annotations

import pytest

from recommendation.control.candidate_ledger import (
    CandidateLedger,
    normalize_source_policy,
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
)
from recommendation.control.arguments import CandidateSource
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerContractError,
    validate_action_arguments,
)


def test_balanced_is_exactly_the_accepted_unweighted_rrf() -> None:
    rankings = {
        "history": ["h1", "h2", "h3"],
        "catalog_search": ["c2", "h2", "c3"],
    }
    weighted = weighted_reciprocal_rank_fusion(rankings, source_policy="balanced")
    accepted = reciprocal_rank_fusion(
        [rankings["history"], rankings["catalog_search"]]
    )
    assert weighted == accepted


def test_policy_weights_change_fusion_and_deduplicate_identity() -> None:
    rankings = {
        "history": ["h1", "shared", "h3"],
        "catalog_search": ["c1", "shared", "c3"],
    }
    balanced = weighted_reciprocal_rank_fusion(rankings, source_policy="balanced")
    intent = weighted_reciprocal_rank_fusion(rankings, source_policy="intent_focus")
    history = weighted_reciprocal_rank_fusion(rankings, source_policy="history_focus")
    assert len({identity for identity, _ in intent}) == 5
    assert intent[0][0] == "shared"
    assert history[0][0] == "shared"
    assert intent != history
    assert normalize_source_policy(None) == "balanced"
    assert normalize_source_policy("") == "balanced"
    with pytest.raises(ValueError):
        normalize_source_policy("unknown_policy")


def test_ledger_uses_policy_in_its_ranked_order() -> None:
    history = [("h1", 1, 1.0), ("shared", 2, 2.0)]
    catalog = [("c1", 1, 1.0), ("shared", 2, 2.0)]
    ledger = CandidateLedger(source_policy="history_focus")
    ledger.record_from_source(
        source=CandidateSource.HISTORY,
        candidates=history,
        grounded=frozenset({"h1", "shared"}),
    )
    ledger.record_from_source(
        source=CandidateSource.CATALOG_SEARCH,
        candidates=catalog,
        grounded=frozenset({"c1", "shared"}),
    )
    order = ledger.ranked_parent_asins()
    assert set(order) == {"h1", "shared", "c1"}
    assert ledger.source_policy == "history_focus"


def test_planner_contract_accepts_policy_and_rejects_unknown() -> None:
    validate_action_arguments(
        ActionProposal(
            action=PlannerAction.RETRIEVE,
            arguments={"query": "water filter", "source_policy": "intent_focus"},
        )
    )
    validate_action_arguments(
        ActionProposal(action=PlannerAction.RETRIEVE, arguments={"query": "water filter"})
    )
    with pytest.raises(PlannerContractError) as excinfo:
        validate_action_arguments(
            ActionProposal(
                action=PlannerAction.RETRIEVE,
                arguments={"query": "water filter", "source_policy": "made_up"},
            )
        )
    assert excinfo.value.code == "invalid_source_policy"