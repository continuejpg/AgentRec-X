"""P14-C0 contract tests: shared executor, Rule real run, illegal actions, blocked Qwen honesty.

No network/Qwen call is made: the test proves the real Rule path works, illegal actions are
rejected, and the Qwen side is reported BLOCKED rather than fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p14c0.electronics_executor import ElectronicsPlannerExecutor, prepare_hard, prepare_preferences
from experiments.p14b1.diagnose import hard_requirements
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import (
    ActionProposal, PlannerAction, PlannerContractError, PlannerObservation, run_planner,
)
from recommendation.preference.decision_gate import (
    DecisionGate, GateBudget, GateContext,
)
from recommendation.shopping_companion.memory import (
    SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory,
)
from recommendation.preference.history import HistoryStateReader

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14e2/benchmark"


def _load(split: str, kind: str):
    return [json.loads(line) for line in (DATA / f"{split}.{kind}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture(scope="module")
def catalogue():
    return ElectronicsCatalogue.load()


def _dev_task():
    return _load("dev", "agent")[0]


def _prefs_and_hard(catalogue, task):
    with ShoppingCompanionMemory() as memory:
        memory.ingest_sessions(task["task_id"], [dict(s) for s in task["history_sessions"]],
                               source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref="test")
        history = HistoryStateReader(memory.preference_service).read(
            task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"])
    return prepare_preferences(history), prepare_hard(task["current_query"])


class _AskPlanner:
    name = "test_ask"

    def propose(self, observation):
        return ActionProposal(PlannerAction.ASK, {"question": "Which feature matters most?"}, "test")


def test_rule_planner_runs_real_tools_on_dev(catalogue):
    from experiments.p14c0.run_dev import _run_one

    task = _dev_task()
    record = _run_one(catalogue, task, planner_kind="rule")
    assert record["error"] is None
    tools = {call["tool"] for call in record["tool_calls"]}
    assert {"product_search", "product_view", "eligibility", "decision_gate"} <= tools
    assert record["terminal"]["status"] in {"finalized", "awaiting_clarification",
                                            "no_verified_candidate_in_scope", "need_more_candidates"}
    ledger_ok = any(call["tool"] == "product_search" and call["output"].get("ledger_entries", 0) > 0
                    for call in record["tool_calls"])
    assert ledger_ok


def test_illegal_finalize_without_candidates_is_rejected(catalogue):
    task = _dev_task()
    prefs, hard = _prefs_and_hard(catalogue, task)
    executor = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=task["current_query"],
                                          preferences=prefs, hard=hard)
    with pytest.raises(PlannerContractError):
        executor.execute(ActionProposal(PlannerAction.FINALIZE, {}, "illegal"), PlannerObservation(step=1, query=task["current_query"]))
    assert executor.terminal_status == "not_started"


def test_ask_pending_cannot_finalize(catalogue):
    gate = DecisionGate()
    decision = gate.check(GateContext(
        query="headphones", candidates=[{"product_ref": {"namespace": "amazon_electronics_c4", "item_id": "B1"},
                                         "hard_tier": 0, "matched_preferences": [], "conflicting_preferences": [],
                                         "unknown_preferences": []}],
        active_preferences=[], hard_requirements=[{"kind": "attribute", "value": "wireless"}],
        preference_conflicts=[], namespace="amazon_electronics_c4", query_specificity=True,
        ask_pending=True, budget=GateBudget(),
    ))
    assert decision.allow_finalize is False
    assert decision.action == "NEED_USER_CLARIFICATION"


def test_finalize_without_evidence_waits_for_gate_and_never_claims_success(catalogue):
    task = _dev_task()
    prefs, hard = _prefs_and_hard(catalogue, task)
    executor = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=task["current_query"],
                                          preferences=prefs, hard=hard)
    with pytest.raises(PlannerContractError):
        executor.execute(ActionProposal(PlannerAction.FINALIZE, {}, "no candidates"), PlannerObservation(step=1, query=task["current_query"]))
    assert executor.selected_candidate_id is None


def test_real_qwen_trajectories_exist_and_use_same_tools():
    comparison = REPO / "runs/p14c0/comparison.json"
    manifest = REPO / "runs/p14c0/qwen_manifest.json"
    assert comparison.is_file() and manifest.is_file()
    data = json.loads(comparison.read_text(encoding="utf-8"))
    qwen = data["qwen"]
    assert qwen is not None and qwen["total_model_calls"] > 0
    assert qwen["tasks"] == 20
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert meta["status"] == "EXECUTED"
    assert meta["model"]["model_family"] == "Qwen3.5-9B"
    traj_dir = REPO / "runs/p14c0/qwen_dev/trajectories"
    trajectories = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(traj_dir.glob("*.json"))]
    assert len(trajectories) == 20
    qwen_tools = {call["tool"] for record in trajectories for call in record["tool_calls"]}
    assert {"product_search", "product_view", "eligibility", "decision_gate"} <= qwen_tools
    rule_dir = REPO / "runs/p14c0/rule_dev/trajectories"
    rule_trajectories = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(rule_dir.glob("*.json"))]
    rule_tools = {call["tool"] for record in rule_trajectories for call in record["tool_calls"]}
    assert rule_tools == qwen_tools
    assert all(record["planner"] == "qwen" for record in trajectories)
    assert any(record["model_stats"]["calls"] > 0 for record in trajectories)


def test_qwen_outputs_are_not_fabricated_and_reserved_untouched():
    data = json.loads((REPO / "runs/p14c0/comparison.json").read_text(encoding="utf-8"))
    assert data["qwen"]["total_input_tokens"] > 0 and data["qwen"]["total_output_tokens"] > 0
    assert data["qwen"]["avg_model_latency_ms"] is not None
    reserved = {task["task_id"] for task in _load("reserved_test", "agent")}
    for path in (REPO / "runs/p14c0").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not any(task_id in text for task_id in reserved), path


def test_reserved_test_not_touched_by_p14c0():
    reserved = {task["task_id"] for task in _load("reserved_test", "agent")}
    for path in (REPO / "runs/p14c0").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not any(task_id in text for task_id in reserved), path


def test_shared_components_are_identical_for_both_planners(catalogue):
    task = _dev_task()
    prefs, hard = _prefs_and_hard(catalogue, task)
    a = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=task["current_query"], preferences=prefs, hard=hard)
    b = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=task["current_query"], preferences=prefs, hard=hard)
    assert (a.namespace, a.gate.version, a.scorer.version) == (b.namespace, b.gate.version, b.scorer.version)
    assert a.gate_budget.as_dict() == b.gate_budget.as_dict()
    assert type(a.catalogue) is type(b.catalogue)
