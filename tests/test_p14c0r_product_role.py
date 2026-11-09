"""P14-C0-R tests: product type/role contract, Gate integration, bounded retry terminal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.decision_gate import DecisionGate, GateBudget, GateContext
from recommendation.preference.product_role import (
    CandidateRole, RoleStatus, assess_candidate_role, infer_product_target,
)

REPO = Path(__file__).resolve().parents[1]
NS = "amazon_electronics_c4"


def _row(item_id: str, rank: int = 1, *, role: str = "MATCH", product_type: str | None = "keyboard",
         hard_tier: int = 0):
    return {
        "product_ref": {"namespace": NS, "item_id": item_id},
        "query_rank": rank,
        "hard_tier": hard_tier,
        "role_status": role,
        "product_type": product_type,
        "candidate_role": "main",
        "matched_preferences": [],
        "conflicting_preferences": [],
        "unknown_preferences": [],
        "evidence_refs": [],
    }


def _gate_context(candidates, **overrides):
    data = {
        "query": "I need a wireless keyboard for my office desk.",
        "candidates": candidates,
        "active_preferences": [{"value": "wireless", "polarity": "positive"}],
        "hard_requirements": [{"kind": "attribute", "value": "wireless"}],
        "preference_conflicts": [],
        "namespace": NS,
        "query_specificity": True,
        "expected_product_type": "keyboard",
        "expected_product_role": "main",
        "budget": GateBudget(max_retrievals=1, max_product_evidence_rounds=1),
    }
    data.update(overrides)
    return GateContext(**data)


def test_query_target_inference_main_accessory_and_none():
    assert infer_product_target("I need a wireless keyboard.").product_type == "keyboard"
    assert infer_product_target("I need a wireless keyboard.").role is CandidateRole.MAIN
    accessory = infer_product_target("I need a keyboard stand for my desk.")
    assert accessory.product_type == "keyboard" and accessory.role is CandidateRole.ACCESSORY
    assert infer_product_target("Please recommend something nice.").product_type is None


def test_role_assessment_main_mismatch_accessory_unknown_not_required():
    keyboard = infer_product_target("I need a keyboard for my office desk.")
    match = assess_candidate_role(product_type="keyboard", title="Mechanical Keyboard RGB", target=keyboard)
    assert match.status is RoleStatus.MATCH
    wrong_type = assess_candidate_role(product_type="headphones", title="Wireless Headphones", target=keyboard)
    assert wrong_type.status is RoleStatus.MISMATCH
    accessory = assess_candidate_role(product_type="keyboard", title="Keyboard Stand for Desk", target=keyboard)
    assert accessory.status is RoleStatus.MISMATCH and accessory.candidate_role is CandidateRole.ACCESSORY
    unknown = assess_candidate_role(product_type=None, title="Mind Reader Lap Desk", target=keyboard)
    assert unknown.status is RoleStatus.UNKNOWN
    no_target = assess_candidate_role(product_type="headphones", title="Headphones", target=infer_product_target("something nice"))
    assert no_target.status is RoleStatus.NOT_REQUIRED
    accessory_target = infer_product_target("I need a keyboard stand")
    accessory_match = assess_candidate_role(product_type="keyboard", title="Keyboard Stand for Desk", target=accessory_target)
    assert accessory_match.status is RoleStatus.MATCH
    main_vs_accessory = assess_candidate_role(product_type="keyboard", title="Mechanical Keyboard", target=accessory_target)
    assert main_vs_accessory.status is RoleStatus.MISMATCH


def test_gate_blocks_role_mismatch_and_allows_correct_type():
    gate = DecisionGate()
    blocked = gate.check(_gate_context([_row("B1", role="MISMATCH", product_type="earbuds")]))
    assert blocked.allow_finalize is False
    assert blocked.action in {"NEED_MORE_CANDIDATES", "NO_VERIFIED_CANDIDATE_IN_SCOPE"}
    allowed = gate.check(_gate_context([_row("B1", role="MATCH", product_type="keyboard")]))
    assert allowed.allow_finalize is True
    assert allowed.product_type == "keyboard"


def test_gate_does_not_fabricate_constraint_when_type_not_requested():
    gate = DecisionGate()
    context = _gate_context([_row("B1", role="NOT_REQUIRED", product_type=None)],
                            expected_product_type=None, expected_product_role=None)
    decision = gate.check(context)
    assert decision.allow_finalize is True


def test_gate_unknown_type_not_treated_as_match():
    gate = DecisionGate()
    decision = gate.check(_gate_context([_row("B1", role="UNKNOWN", product_type=None)]))
    assert decision.allow_finalize is False
    assert decision.action in {"NEED_MORE_PRODUCT_EVIDENCE", "NEED_MORE_CANDIDATES",
                               "NO_VERIFIED_CANDIDATE_IN_SCOPE"}


def test_gate_bounded_retry_terminal_status():
    gate = DecisionGate()
    exhausted = gate.check(_gate_context(
        [_row("B1", role="MISMATCH", product_type="earbuds")],
        budget=GateBudget(max_retrievals=1, max_product_evidence_rounds=1,
                          retrievals_used=1, product_evidence_rounds_used=1),
    ))
    assert exhausted.allow_finalize is False
    assert exhausted.action == "NO_VERIFIED_CANDIDATE_IN_SCOPE"
    assert exhausted.bounded_scope is True


def test_fixed_runs_use_same_role_rules_and_never_finalize_type_mismatch():
    comparison = REPO / "runs/p14c0r/comparison.json"
    assert comparison.is_file(), "run experiments.p14c0r.run_fixed first"
    data = json.loads(comparison.read_text(encoding="utf-8"))
    assert data["after_fixed"]["rule"]["tasks"] == 20
    assert data["after_fixed"]["qwen"]["tasks"] == 20
    assert data["after_fixed"]["qwen"]["total_model_calls"] > 0
    assert data["type_error_counts"]["rule"] == 0
    assert data["type_error_counts"]["qwen"] == 0
    for planner in ("rule", "qwen"):
        root = REPO / f"runs/p14c0r/{planner}_dev/trajectories"
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]
        assert len(records) == 20
        for record in records:
            assert record["terminal"]["status"] not in {
                "need_more_candidates", "need_more_product_evidence", "not_started", "in_progress"}
            # type check is covered by comparison.type_error_counts above (all 20 tasks)


def test_correct_type_tasks_not_over_rejected_and_old_results_preserved():
    data = json.loads((REPO / "runs/p14c0r/comparison.json").read_text(encoding="utf-8"))
    before = data["before_frozen_p14c0"]
    after = data["after_fixed"]
    assert before["rule"]["correct_top1"] == 6 and before["qwen"]["correct_top1"] == 8
    assert after["rule"]["correct_top1"] >= before["rule"]["correct_top1"] + 5
    assert after["qwen"]["correct_top1"] >= before["qwen"]["correct_top1"] + 4
    assert after["rule"]["false_ask"] == 0 and after["rule"]["false_refusal"] == 0
    assert after["qwen"]["false_ask"] <= 1 and after["qwen"]["false_refusal"] <= 1
    assert after["qwen"]["finalized_recommend"] >= 13
    assert data["type_error_counts"] == {"rule": 0, "qwen": 0}
    assert (REPO / "runs/p14c0/comparison.json").is_file(), "P14-C0 result must be preserved"


def test_reserved_test_not_touched_by_p14c0r():
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in (REPO / "runs/p14c0r").rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8") for task_id in reserved), path
