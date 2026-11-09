"""P14-B1 tests: Decision Gate rules, runtime hook, isolation and regressions."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from recommendation.preference.decision_gate import (
    ALLOW_FINALIZE,
    NEED_MORE_CANDIDATES,
    NEED_MORE_PRODUCT_EVIDENCE,
    NEED_USER_CLARIFICATION,
    NO_VERIFIED_CANDIDATE_IN_SCOPE,
    DecisionGate,
    GateBudget,
    GateContext,
)
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[1]
NS = "amazon_electronics_c4"


def _row(item_id: str, rank: int, hard_tier: int = 0, matched=None, conflicting=None, unknown=None):
    return {
        "product_ref": {"namespace": NS, "item_id": item_id},
        "query_rank": rank,
        "hard_tier": hard_tier,
        "matched_preferences": matched or [],
        "conflicting_preferences": conflicting or [],
        "unknown_preferences": unknown or [],
        "evidence_refs": [{"attribute": "wireless", "catalogue_evidence": "wireless"}],
    }


def _context(**overrides):
    data = {
        "query": "wireless headphones for commuting",
        "candidates": [_row("B1", 1)],
        "active_preferences": [{"value": "over-ear", "polarity": "positive"}],
        "hard_requirements": [{"kind": "attribute", "value": "wireless"}],
        "preference_conflicts": [],
        "namespace": NS,
        "query_specificity": True,
        "budget": GateBudget(),
    }
    data.update(overrides)
    return GateContext(**data)


def test_gate_allows_finalize_when_evidence_sufficient():
    decision = DecisionGate().check(_context())
    assert decision.action == ALLOW_FINALIZE and decision.allow_finalize
    assert decision.selected_candidate_id == "B1"


def test_gate_blocks_hard_constraint_violation():
    decision = DecisionGate().check(_context(candidates=[_row("B1", 1, hard_tier=2)]))
    assert not decision.allow_finalize
    assert decision.action in {NO_VERIFIED_CANDIDATE_IN_SCOPE, NEED_MORE_CANDIDATES, NEED_MORE_PRODUCT_EVIDENCE}


def test_gate_never_treats_unknown_hard_constraint_as_satisfied():
    decision = DecisionGate().check(_context(candidates=[_row("B1", 1, hard_tier=1)]))
    assert not decision.allow_finalize
    assert decision.action in {NEED_MORE_PRODUCT_EVIDENCE, NEED_MORE_CANDIDATES, NO_VERIFIED_CANDIDATE_IN_SCOPE}


def test_gate_allows_soft_unknown_but_reports_uncertainty():
    decision = DecisionGate().check(_context(candidates=[_row("B1", 1, unknown=[{"value": "foldable"}])]))
    assert decision.allow_finalize
    assert "foldable" in decision.unknown_fields


def test_gate_asks_when_preferences_conflict():
    decision = DecisionGate().check(_context(
        preference_conflicts=[{"left": "wireless", "right": "wired"}],
    ))
    assert decision.action == NEED_USER_CLARIFICATION
    assert decision.ask_question
    assert not decision.allow_finalize


def test_gate_asks_when_no_decision_criteria():
    decision = DecisionGate().check(_context(
        query="I need some new headphones", active_preferences=[], hard_requirements=[],
        query_specificity=False,
    ))
    assert decision.action == NEED_USER_CLARIFICATION
    assert "matters most" in (decision.ask_question or "")


def test_gate_requests_product_evidence_before_searching_more():
    decision = DecisionGate().check(_context(candidates=[_row("B1", 1, hard_tier=1)]))
    assert decision.action == NEED_MORE_PRODUCT_EVIDENCE


def test_gate_requests_more_candidates_after_evidence_round():
    decision = DecisionGate().check(_context(
        candidates=[_row("B1", 1, hard_tier=1)],
        budget=GateBudget(max_retrievals=1, max_product_evidence_rounds=1, product_evidence_rounds_used=1),
    ))
    assert decision.action == NEED_MORE_CANDIDATES


def test_gate_bounded_scope_after_budget_consumed():
    decision = DecisionGate().check(_context(
        candidates=[_row("B1", 1, hard_tier=1)],
        budget=GateBudget(max_retrievals=1, max_product_evidence_rounds=1,
                          retrievals_used=1, product_evidence_rounds_used=1),
    ))
    assert decision.action == NO_VERIFIED_CANDIDATE_IN_SCOPE
    assert decision.bounded_scope is True
    assert "whole catalogue" in decision.reason


def test_gate_blocks_pending_ask_and_does_not_fabricate_answer():
    decision = DecisionGate().check(_context(ask_pending=True))
    assert decision.action == NEED_USER_CLARIFICATION
    assert not decision.allow_finalize


def test_gate_rejects_foreign_namespace():
    decision = DecisionGate().check(_context(candidates=[_row("B1", 1) | {"product_ref": {"namespace": "amazon_sports_5core", "item_id": "1"}}]))
    assert not decision.allow_finalize
    assert decision.reason_code == "namespace_mismatch"


def test_gate_does_not_read_labels_and_is_deterministic():
    first = DecisionGate().check(_context()).as_dict()
    second = DecisionGate().check(_context()).as_dict()
    assert first == second
    serialized = json.dumps(first)
    for marker in ("expected_decision", "verified_candidate", "target_id", "ground_truth"):
        assert marker not in serialized


def test_p14e_labels_and_b0_artifacts_unchanged():
    manifest = json.loads((REPO / "data/p14e/benchmark/manifest.json").read_text(encoding="utf-8"))
    for name in ("agent_tasks.jsonl", "evaluator_only.jsonl"):
        digest = hashlib.sha256((REPO / "data/p14e/benchmark" / name).read_bytes()).hexdigest()
        assert digest == manifest["files"][name]["sha256"]
    assert (REPO / "runs/p14b0").is_dir(), "P14-B0 artifacts must be preserved"


def test_scorer_and_gate_are_separate_components():
    assert PreferenceAwareScorer().version == "preference_scorer_v0.1"
    assert DecisionGate().version == "decision_gate_v0.1"
