"""P14-C1 Rule/Base/SFT comparison on Dev20 under the same BGE backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/p14c1"
DATA = REPO / "data/p14e2/benchmark"
PLANNERS = ("rule", "qwen_base", "qwen_sft")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    evaluator = {row["task_id"]: row for row in load_jsonl(DATA / "dev.evaluator.jsonl")}
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for planner in PLANNERS:
        root = RUNS / planner / "trajectories"
        records[planner] = {row["task_id"]: row for row in
                            (json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json")))}
    per_task = []
    for task_id, ev in evaluator.items():
        item = {"task_id": task_id, "task_type": ev["task_type"], "expected_decision": ev["expected_decision"],
                "planners": {}}
        for planner in PLANNERS:
            record = records[planner][task_id]
            outcome = record["outcome"]
            item["planners"][planner] = {
                "status": record["terminal"]["status"],
                "selected": record["terminal"].get("selected_candidate_id"),
                "correct_top1": outcome.get("correct_top1"),
                "false_ask": outcome.get("false_ask"),
                "false_refusal": outcome.get("false_refusal"),
                "false_finalize": outcome.get("false_finalize"),
                "error": record.get("error"),
                "tool_calls": len(record["tool_calls"]),
                "search_calls": sum(1 for call in record["tool_calls"] if call["tool"] == "product_search"),
                "model_calls": (record.get("model_stats") or {}).get("calls", 0),
                "latency_ms": (record.get("model_stats") or {}).get("latency_ms"),
                "input_tokens": (record.get("model_stats") or {}).get("input_tokens"),
                "output_tokens": (record.get("model_stats") or {}).get("output_tokens"),
            }
        per_task.append(item)
    summary = {}
    for planner in PLANNERS:
        summary[planner] = json.loads((RUNS / planner / "summary.json").read_text(encoding="utf-8"))
    comparison = {
        "schema": "agentrecx.p14c1.comparison.v1",
        "search_backend": summary["rule"]["search_backend"],
        "planners": {planner: {key: value for key, value in summary[planner].items()
                               if key not in {"tasks", "search_backend"}} for planner in PLANNERS},
        "deltas": {
            "base_minus_rule_correct_top1": summary["qwen_base"]["correct_top1"] - summary["rule"]["correct_top1"],
            "sft_minus_base_correct_top1": summary["qwen_sft"]["correct_top1"] - summary["qwen_base"]["correct_top1"],
            "sft_minus_rule_correct_top1": summary["qwen_sft"]["correct_top1"] - summary["rule"]["correct_top1"],
        },
        "per_task": per_task,
        "notes": ["all planners share the BGE dense search backend, V0 Scorer, Product Type/Role, Gate and budget",
                  "each task gets a fresh Memory state; Dev is exploratory and repeatedly used for development",
                  "Reserved Test was not run; labels remain provisional"],
    }
    (RUNS / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (RUNS / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_task:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"planners": comparison["planners"], "deltas": comparison["deltas"],
                      "sft_failures": [row["task_id"] for row in per_task
                                       if row["expected_decision"] == "recommend"
                                       and not row["planners"]["qwen_sft"]["correct_top1"]],
                      "base_failures": [row["task_id"] for row in per_task
                                        if row["expected_decision"] == "recommend"
                                        and not row["planners"]["qwen_base"]["correct_top1"]]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
