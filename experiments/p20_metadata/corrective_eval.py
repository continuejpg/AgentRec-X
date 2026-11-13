"""Small real-catalog baseline/corrective evaluation for the Evidence-Aware RAG MVP.

This uses the existing Qwen planner/executor and, when requested, the real Qwen3.5-9B
HTTP service.  It is deliberately task-level: 10 existing/controlled tasks, no generated
synthetic corpus, no P18 Reserved task, no training.

Examples:
  AGENTRECX_QWEN_PLANNER=1 AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029 \
    .venv/bin/python experiments/p20_metadata/corrective_eval.py --planner qwen --policy baseline
  ... --policy corrective
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor, prepare_hard
from experiments.p14c0.run_dev import _evaluate, _load, _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
DATA_B3 = REPO / "data/p14b3_eval100"
OUT = REPO / "runs/p20_metadata"


class TracePlanner:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def propose(self, observation):
        proposal = self.inner.propose(observation)
        response = getattr(self.inner, "last_response", {}) or {}
        self.calls.append({
            "action": proposal.action.value,
            "arguments": dict(proposal.arguments),
            "rationale": proposal.rationale,
            "latency_ms": response.get("latency_ms"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "parser_errors": response.get("errors") or [],
        })
        return proposal


def _controlled(tasks: list[dict[str, Any]]) -> None:
    tasks.extend([
        {
            "task_id": "controlled_mechanical_wireless_keyboard",
            "current_query": "I need a keyboard for my desk.",
            "hard": [
                {"kind": "attribute", "value": "mechanical", "source_text": "controlled task constraint"},
                {"kind": "attribute", "value": "wireless", "source_text": "controlled task constraint"},
            ],
            "preferences": [],
            "expected": "recommend",
            "label_source": "manual_controlled_real_catalog",
            "expected_note": "a keyboard whose catalogue text has direct mechanical and wireless evidence",
        },
        {
            "task_id": "controlled_unverifiable_price_warranty",
            "current_query": "I need headphones with a two-year warranty under $50.",
            "hard": [
                {"kind": "unverifiable", "value": "price", "source_text": "controlled task constraint"},
                {"kind": "unverifiable", "value": "warranty", "source_text": "controlled task constraint"},
            ],
            "preferences": [],
            "expected": "abstain_or_ask",
            "label_source": "manual_controlled_real_catalog",
            "expected_note": "price/warranty are not verifiable in the fixed catalogue; do not search indefinitely",
        },
    ])


def _tasks(catalogue: ElectronicsCatalogue) -> list[dict[str, Any]]:
    agents = {r["task_id"]: r for r in _load("dev", "agent")}
    evaluators = {r["task_id"]: r for r in _load("dev", "evaluator")}
    tasks = []
    for task_id in sorted(agents):
        task = dict(agents[task_id])
        task["hard"] = prepare_hard(task["current_query"])
        task["scored_evaluator"] = evaluators[task_id]
        task["expected"] = evaluators[task_id]["expected_decision"]
        task["label_source"] = "p14e2_dev_automated_checked"
        task["split"] = "dev"
        tasks.append(task)
    b3_agent_rows = [json.loads(line) for line in
                     (DATA_B3 / "eval100.agent.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    b3_eval_rows = [json.loads(line) for line in
                    (DATA_B3 / "eval100.evaluator.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    b3_agents = {row["task_id"]: row for row in b3_agent_rows}
    b3_evaluators = {row["task_id"]: row for row in b3_eval_rows}
    for task_id in ["p14b3_eval100_0001", "p14b3_eval100_0065",
                    "p14b3_eval100_0089", "p14b3_eval100_0093",
                    "p14b3_eval100_0109"]:
        task = dict(b3_agents[task_id])
        task["hard"] = prepare_hard(task["current_query"])
        task["scored_evaluator"] = b3_evaluators[task_id]
        task["expected"] = b3_evaluators[task_id]["expected_decision"]
        task["label_source"] = "p14b3_eval100_pending_human_review"
        task["split"] = "eval100"
        tasks.append(task)
    _controlled(tasks)
    return tasks


def _memory_for(catalogue: ElectronicsCatalogue, task: dict[str, Any]) -> list[dict[str, Any]]:
    if "scored_evaluator" not in task:
        return list(task.get("preferences") or [])
    preferences, _history = _memory_read(catalogue, task)
    return preferences


def _is_success(task: dict[str, Any], status: str, selected: str | None,
                catalogue: ElectronicsCatalogue, evaluator: dict[str, Any] | None) -> bool:
    expected = task["expected"]
    if expected == "recommend":
        if status != "finalized" or not selected:
            return False
        if evaluator is not None:
            return bool(_evaluate({"terminal": {"status": status,
                                                "selected_candidate_id": selected}},
                                  evaluator, catalogue).get("correct_top1"))
        product = catalogue.get(selected)
        attrs = set((product.attributes or {}) if product else ())
        return {"mechanical", "wireless"}.issubset(attrs)
    if expected == "clarification_required" or expected == "abstain_or_ask":
        return status in {"awaiting_clarification", "no_verified_candidate_in_scope"}
    if expected in {"no_verified_match", "insufficient_evidence"}:
        return status == "no_verified_candidate_in_scope"
    return False


def _run_one(catalogue: ElectronicsCatalogue, task: dict[str, Any], *, planner_kind: str,
             corrective_rag: bool) -> dict[str, Any]:
    preferences = _memory_for(catalogue, task)
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=task["current_query"], preferences=preferences,
        hard=list(task["hard"]), corrective_rag=corrective_rag,
    )
    observation = PlannerObservation(step=0, query=task["current_query"],
                                     m5_state={"phase": PHASE_PRODUCT_RETRIEVE,
                                               "next_query": task["current_query"],
                                               "preferences": preferences})
    if planner_kind == "qwen":
        proposer = HttpPlannerProposer.from_env()
        planner: Any = TracePlanner(proposer)
    else:
        proposer = None
        planner = P14ARulePlanner()
    started = time.perf_counter()
    error = None
    try:
        run = run_planner(planner=planner, executor=executor, initial_observation=observation,
                          max_steps=MAX_STEPS)
        status = executor.terminal_status
        if status in {"not_started", "in_progress", "need_more_candidates",
                      "need_more_product_evidence"}:
            status = "budget_exhausted"
        selected = executor.selected_candidate_id
        steps = run.as_dict()["steps"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        status, selected, steps = "error", None, []
    latency = round(time.perf_counter() - started, 3)
    return {
        "task_id": task["task_id"], "current_query": task["current_query"],
        "expected": task["expected"], "label_source": task["label_source"],
        "status": status, "selected_product_id": selected,
        "success": _is_success(task, status, selected, catalogue, task.get("scored_evaluator")),
        "actions": [step.get("action") for step in steps],
        "steps": steps,
        "retrievals": executor.retrievals,
        "corrective_retrieval": {
            "enabled": bool(corrective_rag),
            "attempts": executor.corrective_attempts,
            "trace": list(executor.corrective_trace),
        },
        "gate": executor.last_decision,
        "planner": {
            "calls": len(planner.calls) if isinstance(planner, TracePlanner) else 0,
            "latency_ms": [c.get("latency_ms") for c in planner.calls] if isinstance(planner, TracePlanner) else [],
            "input_tokens": sum(int(c.get("input_tokens") or 0) for c in planner.calls) if isinstance(planner, TracePlanner) else 0,
            "output_tokens": sum(int(c.get("output_tokens") or 0) for c in planner.calls) if isinstance(planner, TracePlanner) else 0,
            "rationales": [c.get("rationale") for c in planner.calls] if isinstance(planner, TracePlanner) else [],
        },
        "latency_seconds": latency,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("rule", "qwen"), default="qwen")
    parser.add_argument("--policy", choices=("baseline", "corrective"), default="baseline")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    search = configure_catalogue(catalogue)
    tasks = _tasks(catalogue)
    records = []
    started = time.perf_counter()
    for task in tasks:
        record = _run_one(catalogue, task, planner_kind=args.planner,
                          corrective_rag=args.policy == "corrective")
        records.append(record)
        print(json.dumps({
            "task": record["task_id"], "expected": record["expected"],
            "status": record["status"], "selected": record["selected_product_id"],
            "success": record["success"], "actions": record["actions"],
            "corrective_attempts": record["corrective_retrieval"]["attempts"],
            "latency_s": record["latency_seconds"],
        }, ensure_ascii=False), flush=True)
    summary = {
        "schema": "agentrecx.p20.corrective_eval.v1",
        "planner": args.planner,
        "policy": args.policy,
        "metadata_mode": "enriched",
        "search": search,
        "tasks": records,
        "metrics": {
            "tasks": len(records),
            "success": sum(1 for r in records if r["success"]),
            "recommend_tasks": sum(1 for r in records if r["expected"] == "recommend"),
            "recommend_success": sum(1 for r in records if r["expected"] == "recommend" and r["success"]),
            "correct_ask_or_abstain": sum(1 for r in records if r["expected"] in
                                          {"clarification_required", "abstain_or_ask",
                                           "no_verified_match", "insufficient_evidence"}
                                          and r["success"]),
            "nonrecommend_tasks": sum(1 for r in records if r["expected"] != "recommend"),
            "constraint_violations": sum(1 for r in records if (r.get("gate") or {}).get("hard_constraint_status", {}).get("hard_constraints") == "violated"),
            "corrective_attempts": sum(r["corrective_retrieval"]["attempts"] for r in records),
            "corrective_recovered": sum(
                1 for r in records
                if r["corrective_retrieval"]["attempts"] and r["status"] == "finalized"
                and r["selected_product_id"]
            ),
            "corrective_no_improvement": sum(
                1 for r in records
                if r["corrective_retrieval"]["attempts"] and not r["selected_product_id"]
            ),
            "corrective_trigger_precision": (
                round(sum(1 for r in records if r["corrective_retrieval"]["trace"]
                          and any((t.get("new_hard_verified_count") or 0) > 0
                                  for t in r["corrective_retrieval"]["trace"]))
                      / max(1, sum(r["corrective_retrieval"]["attempts"] for r in records)), 3)
            ),
            "search_calls": sum(r["retrievals"] for r in records),
            "tool_calls": sum(len(r["steps"]) for r in records),
            "planner_calls": sum(r["planner"]["calls"] for r in records),
            "planner_input_tokens": sum(r["planner"]["input_tokens"] for r in records),
            "planner_output_tokens": sum(r["planner"]["output_tokens"] for r in records),
            "errors": sum(1 for r in records if r["error"]),
            "latency_seconds": round(time.perf_counter() - started, 3),
        },
    }
    out = args.out or OUT / f"corrective_eval_{args.planner}_{args.policy}.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "metrics": summary["metrics"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
