"""Task-level Baseline vs Corrective evaluation over the 30 natural-language tasks.

This reuses the existing direct Qwen Planner--Executor harness (`corrective_eval._run_one`),
so both arms see the same catalogue, memories, Planner model and Gate.  It does not start
one MCP process per task.  The Complete Shadow arm is run separately through MCP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.p14c0.run_dev import _evaluate
from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p20_metadata.corrective_eval import _run_one

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p20_agent_eval"


def _violated(record: dict[str, Any]) -> bool:
    gate = record.get("gate") or {}
    hard = gate.get("hard_constraint_status") or {}
    return hard.get("hard_constraints") == "violated"


def _metrics(records: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {task["task_id"]: task for task in tasks}
    recommend = [r for r in records if by_id[r["task_id"]]["expected"] == "recommend"]
    nonrecommend = [r for r in records if by_id[r["task_id"]]["expected"] != "recommend"]
    memory = [r for r in records if by_id[r["task_id"]]["category"] == "memory_conflict_ask"]
    return {
        "tasks": len(records),
        "success": sum(1 for r in records if r["success"]),
        "success_rate": round(sum(1 for r in records if r["success"]) / max(1, len(records)), 3),
        "recommend_tasks": len(recommend),
        "recommend_success": sum(1 for r in recommend if r["success"]),
        "nonrecommend_tasks": len(nonrecommend),
        "nonrecommend_success": sum(1 for r in nonrecommend if r["success"]),
        "memory_conflict_tasks": len(memory),
        "memory_conflict_success": sum(1 for r in memory if r["success"]),
        "hard_constraint_violations": sum(1 for r in records if _violated(r)),
        "corrective_attempts": sum((r.get("corrective_retrieval") or {}).get("attempts", 0) for r in records),
        "corrective_recovered": sum(
            1 for r in records if (r.get("corrective_retrieval") or {}).get("attempts")
            and r.get("status") == "finalized" and r.get("selected_product_id")),
        "corrective_no_improvement": sum(
            1 for r in records if (r.get("corrective_retrieval") or {}).get("attempts")
            and not r.get("selected_product_id")),
        "planner_calls": sum((r.get("planner") or {}).get("calls", 0) for r in records),
        "search_calls": sum(int(r.get("retrievals") or 0) for r in records),
        "tool_steps": sum(len(r.get("steps") or []) for r in records),
        "input_tokens": sum(int((r.get("planner") or {}).get("input_tokens") or 0) for r in records),
        "output_tokens": sum(int((r.get("planner") or {}).get("output_tokens") or 0) for r in records),
        "latency_seconds": round(sum(float(r.get("latency_seconds") or 0) for r in records), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", choices=("baseline", "corrective"),
                        default=["baseline", "corrective"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    suite = json.loads((OUT / "task_suite.json").read_text(encoding="utf-8"))
    tasks = suite["tasks"]
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    all_payload: dict[str, Any] = {
        "schema": "agentrecx.p20.agent_task_arms.v1",
        "task_set": "30 natural P14-E2/P14-B3 tasks",
        "label_status": suite["label_status"],
        "metadata_mode": "enriched",
        "planner": "qwen",
        "arms": {},
    }
    for policy in args.policies:
        records = []
        for task in tasks:
            record = _run_one(catalogue, task, planner_kind="qwen",
                              corrective_rag=policy == "corrective")
            records.append(record)
            print(json.dumps({
                "policy": policy, "task": record["task_id"], "expected": record["expected"],
                "status": record["status"], "selected": record["selected_product_id"],
                "success": record["success"], "actions": record["actions"],
                "corrective_attempts": record["corrective_retrieval"]["attempts"],
                "latency_s": record["latency_seconds"],
            }, ensure_ascii=False), flush=True)
        metrics = _metrics(records, tasks)
        all_payload["arms"][policy] = {
            "policy": policy, "corrective_rag": policy == "corrective",
            "verifier": "disabled", "tasks": records, "metrics": metrics,
        }
        (OUT / f"task_arm_{policy}.json").write_text(
            json.dumps(all_payload["arms"][policy], indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps({"policy": policy, "metrics": metrics}, ensure_ascii=False), flush=True)
    (OUT / "task_arms.json").write_text(
        json.dumps(all_payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
