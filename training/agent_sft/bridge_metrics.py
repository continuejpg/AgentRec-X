from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.agent.evaluator import evaluate_runs
from benchmarks.agent.evaluator_v2 import evaluate_run_v2, load_v2_labels
from benchmarks.agent.schemas import PolicyId, TaskRun, TurnTrace
from benchmarks.agent.tasks import load_tasks
from benchmarks.agent.runner import _build_runtime

ARMS = {"C": Path("runs/p2bridge_c"), "D0": Path("runs/p2bridge_d0"), "D1": Path("runs/p2bridge_d1"), "D1v2": Path("runs/p2bridge_d1v2")}
FIELDS = ["runnable", "status", "final_ids", "active_preference_attributes", "retrieval_actions", "unavailable_reason", "error", "wall_ms", "input_tokens", "output_tokens", "total_tokens"]


def load_runs(path: Path) -> list[TaskRun]:
    result = []
    for line in (path / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        turns = []
        for turn in data.get("turns", []):
            subset = {k: turn.get(k) for k in TurnTrace.__dataclass_fields__}
            for key in ("clarification_options", "candidate_pool_ids", "final_ids", "retrieval_actions", "tool_calls", "llm_proposals"):
                subset[key] = tuple(subset.get(key) or ())
            subset["memory_events"] = tuple(subset.get("memory_events") or ())
            turns.append(TurnTrace(**subset))
        subset = {k: data.get(k) for k in TaskRun.__dataclass_fields__ if k != "turns"}
        subset["policy"] = PolicyId(data["policy"])
        subset["turns"] = turns
        for key in ("final_ids", "active_preference_attributes", "retrieval_actions"):
            subset[key] = tuple(subset.get(key) or ())
        result.append(TaskRun(**subset))
    return result


def main() -> int:
    tasks = {t.task_id: t for t in load_tasks("dev")}
    runtime = _build_runtime(Path("runs/p2bridge_c"))
    records = runtime.metadata.records
    labels = load_v2_labels()
    rows = []
    summary = {}
    for arm, path in ARMS.items():
        runs = load_runs(path)
        v2 = {e.task_id: e for e in (evaluate_run_v2(run, tasks[run.task_id], records, labels.get(run.task_id, {})) for run in runs)}
        legacy = evaluate_runs(runs, [tasks[run.task_id] for run in runs], records)
        wall = []
        for run in runs:
            valid = accepted = rejected = 0
            for turn in run.turns:
                for event in turn.memory_events:
                    bridge = event.get("proposal_bridge") if isinstance(event, dict) else None
                    if not isinstance(bridge, dict):
                        continue
                    valid += int(bool(bridge.get("valid")))
                    accepted += len(bridge.get("accepted_operations") or ())
                    rejected += len(bridge.get("rejected_operations") or ())
            evaluation = v2[run.task_id]
            row = {
                "task_id": run.task_id,
                "policy": arm,
                "status": run.status,
                "v2_task_success": evaluation.task_success_v2,
                "v2_task_state": evaluation.task_state_correct,
                "v2_preference_lifecycle": evaluation.preference_lifecycle_correct,
                "v2_hard_constraint": evaluation.hard_constraint_satisfaction,
                "v2_product_role": evaluation.product_role_satisfaction,
                "v2_retrieval": evaluation.retrieval_adequacy,
                "v2_tool_action": evaluation.tool_action_validity,
                "v2_exact_identity": evaluation.exact_identity_hit,
                "final_ids": "|".join(run.final_ids),
                "turns": len(run.turns),
                "proposal_turns_valid": valid,
                "accepted_operations": accepted,
                "rejected_operations": rejected,
                "retrieve_calls": sum(sum(1 for action in turn.retrieval_actions if action == "RETRIEVE") for turn in run.turns),
                "rerank_calls": sum(sum(1 for action in turn.retrieval_actions if action == "RERANK") for turn in run.turns),
                "tool_calls": sum(len(turn.tool_calls) for turn in run.turns),
                "wall_ms": run.wall_ms,
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
                "total_tokens": run.total_tokens,
            }
            rows.append(row)
            wall.append(run.wall_ms)
        passed = sum(1 for task_id, e in v2.items() if e.task_success_v2 == "PASS")
        failed = sum(1 for task_id, e in v2.items() if e.task_success_v2 == "FAIL")
        unknown = sum(1 for task_id, e in v2.items() if e.task_success_v2 == "UNKNOWN")
        summary[arm] = {
            "tasks": len(runs), "v2_pass": passed, "v2_fail": failed, "v2_unknown": unknown,
            "legacy_pass": sum(1 for e in legacy if e.passed),
            "p50_wall_ms": statistics.median(wall) if wall else None,
            "p95_wall_ms": sorted(wall)[max(0, int(0.95 * (len(wall) - 1)))] if wall else None,
            "input_tokens": sum(r.input_tokens for r in runs),
            "output_tokens": sum(r.output_tokens for r in runs),
            "total_tokens": sum(r.total_tokens for r in runs),
        }
    out = Path("training/agent_sft/eval/P2_BRIDGE_AGENT_COMPARISON.csv")
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path("training/agent_sft/eval/P2_BRIDGE_AGENT_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
