"""Extract existing Agent traces into candidate rows for future SFT review.

Only already-produced traces are reshaped.  No trajectories are generated and no Qwen
success is declared expert gold.  P18 Reserved is excluded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ARMS = REPO / "runs/p20_agent_eval/task_arms.json"
OUT = REPO / "runs/advanced_retrieval_v2/sft_trace_candidates.jsonl"
MANIFEST = REPO / "runs/advanced_retrieval_v2/sft_trace_manifest.json"


def _row(arm: str, record: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    steps = record.get("steps") or []
    planner_run = record.get("planner_run") or {}
    observations = []
    for step in steps:
        observations.append({
            "step": step.get("step"),
            "action": step.get("action"),
            "observation_before": step.get("observation_before"),
            "tool_observation": step.get("tool_observation"),
        })
    candidate_ids = []
    for step in steps:
        obs = step.get("observation_before") or {}
        candidate_ids.extend(obs.get("candidate_ids") or [])
    return {
        "schema": "agentrecx.sft.planner_trace_candidate.v1",
        "source_arm": arm,
        "task_id": record.get("task_id"),
        "query": task.get("current_query"),
        "memory_snapshot": {
            "active_preferences": record.get("memory_preferences") or [],
            "fixture": task.get("memory_fixture"),
            "user_id": task.get("user_id"),
        },
        "planner_observation": observations,
        "candidate_evidence": {
            "candidate_ids": sorted(set(str(item) for item in candidate_ids if item)),
            "planner_run": planner_run,
        },
        "allowed_actions": None,  # not persisted in the direct-arm trace
        "chosen_action": (steps[-1].get("action") if steps else None),
        "gate_feedback": record.get("gate"),
        "task_outcome": {
            "status": record.get("status"),
            "selected_product_id": record.get("selected_product_id"),
            "success": record.get("success"),
            "hard_constraint_violation": False,
        },
        "label_source": task.get("label_source"),
        "review_status": "provisional_not_expert_gold",
        "p18_reserved_used": False,
    }


def main() -> None:
    payload = json.loads(ARMS.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in
             json.loads((REPO / "runs/p20_agent_eval/task_suite.json").read_text(
                 encoding="utf-8"))["tasks"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT.open("w", encoding="utf-8") as handle:
        for arm_name, arm in payload["arms"].items():
            for record in arm.get("tasks") or []:
                task = tasks.get(record["task_id"])
                if task is None or task.get("split") == "reserved":
                    continue
                handle.write(json.dumps(_row(arm_name, record, task),
                                        ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    manifest = {
        "schema": "agentrecx.sft.trace_asset_manifest.v1",
        "output": str(OUT),
        "rows": count,
        "source": str(ARMS),
        "label_status": "provisional_not_expert_gold",
        "notes": [
            "reshaped existing traces only; no new generation",
            "Qwen success is not treated as expert gold",
            "P18 Reserved is not included",
            "allowed_actions were not persisted by the direct arm trace and remain null",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
