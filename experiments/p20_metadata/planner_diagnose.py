"""Small Qwen Planner diagnostic over P14-E2 dev tasks (not P18 Reserved).

Runs the real existing executor with the metadata-enriched catalogue and the existing
Qwen HTTP planner.  It records actions, rationales, candidate deltas, Gate state and
latency; it does not modify the catalogue or any existing benchmark result.

Usage:
  AGENTRECX_QWEN_PLANNER=1 \
  AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029 \
  .venv/bin/python experiments/p20_metadata/planner_diagnose.py --out runs/p20_metadata/planner_before.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor, prepare_hard
from experiments.p14c0.run_dev import _evaluate, _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import PlannerObservation, PlannerAction, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p20_metadata"
# One task per requested coverage family, from the existing dev split.  No P18 reserved task.
TASK_IDS = [
    "p14e2_lite_0041",  # wireless headphones, explicit hard requirement, recommend
    "p14e2_lite_0042",  # wireless earbuds, discover -> recommend
    "p14e2_lite_0043",  # wireless keyboard, explicit hard requirement
    "p14e2_lite_0045",  # long-term preference + current query, no hard requirement
    "p14e2_lite_0048",  # history preference speaker, evidence may be sparse
    "p14e2_lite_0049",  # water resistance preference, no explicit hard requirement
    "p14e2_lite_0055",  # wireless + noise preference
    "p14e2_lite_0056",  # current task overrides history wireless
    "p14e2_lite_0057",  # ambiguous, should ASK
    "p14e2_lite_0060",  # multiple constraints, no verified match / abstain
]


class TracePlanner:
    """Capture the model-authored action/rationale without changing the action."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.actions: list[dict[str, Any]] = []

    def propose(self, observation):
        started = time.perf_counter()
        proposal = self.inner.propose(observation)
        response = getattr(self.inner, "last_response", {}) or {}
        self.actions.append({
            "action": proposal.action.value,
            "arguments": dict(proposal.arguments),
            "rationale": proposal.rationale,
            "latency_ms": response.get("latency_ms"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "parser_errors": response.get("errors") or [],
            "observation": {
                "step": observation.step,
                "candidate_count": len(observation.candidate_ids),
                "candidate_ids": list(observation.candidate_ids[:10]),
                "allowed_actions": list(observation.allowed_actions),
                "needs_user_input": observation.needs_user_input,
                "remaining_steps": observation.remaining_steps,
                "retrieval_feedback": dict(observation.retrieval_feedback or {}),
                "last_tool_observation": dict(observation.last_tool_observation or {}),
                "m5_state": dict(observation.m5_state or {}),
            },
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 2),
        })
        return proposal


def _load(split: str, kind: str) -> list[dict[str, Any]]:
    path = DATA / f"{split}.{kind}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_one(catalogue: ElectronicsCatalogue, task: dict[str, Any], evaluator: dict[str, Any],
             *, corrective_rag: bool = False) -> dict[str, Any]:
    preferences, _history = _memory_read(catalogue, task)
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=task["current_query"], preferences=preferences,
        hard=prepare_hard(task["current_query"]),
        corrective_rag=corrective_rag,
    )
    proposer = HttpPlannerProposer.from_env()
    planner = TracePlanner(proposer)
    observation = PlannerObservation(
        step=0, query=task["current_query"],
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                  "preferences": preferences},
    )
    started = time.perf_counter()
    error = None
    try:
        run = run_planner(planner=planner, executor=executor, initial_observation=observation,
                          max_steps=MAX_STEPS)
        steps = run.as_dict()
        status = executor.terminal_status
        if status in {"not_started", "in_progress", "need_more_candidates",
                      "need_more_product_evidence"}:
            status = ("budget_exhausted"
                      if (executor.last_decision or {}).get("action") in
                      {"NEED_MORE_PRODUCT_EVIDENCE", "NEED_MORE_CANDIDATES"}
                      else "budget_exhausted")
            executor.terminal_status = status
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        steps = {"steps": [], "final_ids": [], "termination": "error"}
        status = "error"
    model_stats = {
        "calls": len(planner.actions),
        "latency_ms": [a.get("latency_ms") for a in planner.actions],
        "input_tokens": sum(int(a.get("input_tokens") or 0) for a in planner.actions),
        "output_tokens": sum(int(a.get("output_tokens") or 0) for a in planner.actions),
        "parser_errors": [a.get("parser_errors") for a in planner.actions if a.get("parser_errors")],
    }
    record = {
        "task_id": task["task_id"], "split": task["split"], "planner": "qwen",
        "task_type": evaluator.get("task_type"),
        "expected_decision": evaluator.get("expected_decision"),
        "current_query": task["current_query"],
        "preferences": preferences,
        "terminal": {"status": status, "selected_candidate_id": executor.selected_candidate_id,
                     "ask_question": executor.ask_question},
        "corrective_retrieval": {
            "enabled": bool(corrective_rag),
            "attempts": executor.corrective_attempts,
            "trace": list(executor.corrective_trace),
        },
        "planner_actions": planner.actions,
        "planner_run": steps,
        "tool_calls": executor.tool_calls,
        "gate": executor.last_decision,
        "last_eligibility": executor.last_eligibility,
        "model_stats": model_stats,
        "error": error,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }
    record["outcome"] = _evaluate(record, evaluator, catalogue)
    return record


def _redundant_retrieves(record: dict[str, Any]) -> int:
    actions = record.get("planner_actions") or []
    steps = (record.get("planner_run") or {}).get("steps") or []
    redundant = 0
    previous_ids: set[str] = set()
    for action, step in zip(actions, steps):
        if action.get("action") != "RETRIEVE":
            continue
        after = step.get("observation_after") or {}
        ids = set(after.get("candidate_ids") or [])
        if ids and ids.issubset(previous_ids):
            redundant += 1
        previous_ids.update(ids)
    return redundant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT / "planner_before.json")
    parser.add_argument("--max-tasks", type=int, default=len(TASK_IDS))
    parser.add_argument("--corrective-rag", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    agents = {r["task_id"]: r for r in _load("dev", "agent")}
    evaluators = {r["task_id"]: r for r in _load("dev", "evaluator")}
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    search_status = configure_catalogue(catalogue)
    print(json.dumps({"search": search_status}, ensure_ascii=False), flush=True)
    records = []
    started = time.perf_counter()
    for task_id in TASK_IDS[:args.max_tasks]:
        record = _run_one(catalogue, agents[task_id], evaluators[task_id],
                          corrective_rag=args.corrective_rag)
        record["redundant_retrieve"] = _redundant_retrieves(record)
        records.append(record)
        print(json.dumps({
            "task_id": record["task_id"], "expected": record["expected_decision"],
            "status": record["terminal"]["status"], "actions": [
                a["action"] for a in record["planner_actions"]],
            "redundant": record["redundant_retrieve"],
            "selected": record["terminal"]["selected_candidate_id"],
            "gate": (record.get("gate") or {}).get("action"),
            "latency_s": record["latency_seconds"],
        }, ensure_ascii=False), flush=True)
    summary = {
        "schema": "agentrecx.p20.planner_diagnose.v2",
        "tasks": len(records),
        "planner": "qwen",
        "corrective_rag": bool(args.corrective_rag),
        "qwen_url": os.environ.get("AGENTRECX_QWEN_PLANNER_URL"),
        "model": os.environ.get("AGENTRECX_QWEN_PLANNER_MODEL"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "records": records,
        "metrics": {
            "recommend_tasks": sum(1 for r in records if r["expected_decision"] == "recommend"),
            "finalized_recommend": sum(1 for r in records if r["expected_decision"] == "recommend"
                                       and r["outcome"].get("finalized")),
            "correct_recommend": sum(1 for r in records if r["expected_decision"] == "recommend"
                                     and r["outcome"].get("correct_top1")),
            "false_ask": sum(1 for r in records if r["outcome"].get("false_ask")),
            "false_refusal": sum(1 for r in records if r["outcome"].get("false_refusal")),
            "budget_exhausted": sum(1 for r in records if r["terminal"]["status"] == "budget_exhausted"),
            "redundant_retrieve": sum(r["redundant_retrieve"] for r in records),
            "total_tool_calls": sum(len(r["tool_calls"]) for r in records),
            "total_model_calls": sum(r["model_stats"]["calls"] for r in records),
            "non_recommend_correct": sum(1 for r in records if r["expected_decision"] != "recommend"
                                         and (r["outcome"].get("correct_clarification")
                                              or r["outcome"].get("correct_unknown_termination"))),
            "non_recommend_tasks": sum(1 for r in records if r["expected_decision"] != "recommend"),
            "errors": sum(1 for r in records if r.get("error")),
        },
    }
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=1, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
