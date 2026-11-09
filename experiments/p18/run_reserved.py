"""P18 one-time Reserved Test run, after the frozen core manifest.

The reserved rows are used only after the Agent has executed; they are never passed into the
Agent, search, memory or skills. This script refuses to run twice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.p14c0.run_dev import _aggregate, _evaluate, _run_one
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p18/reserved"
MANIFEST = OUT / "reserved_run_manifest.json"


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit("P18 Reserved Test was already run once; refusing a second run")
    tasks = load(DATA / "reserved_test.agent.jsonl")
    evaluators = {row["task_id"]: row for row in load(DATA / "reserved_test.evaluator.jsonl")}
    catalogue = ElectronicsCatalogue.load()
    search_config = configure_catalogue(catalogue)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for task in tasks:
        try:
            record = _run_one(catalogue, task, planner_kind="rule")
        except Exception as exc:  # record honestly; do not retry after seeing a result
            record = {"task_id": task["task_id"], "terminal": {"status": "error"}, "error": f"{type(exc).__name__}: {exc}",
                      "tool_calls": [], "model_stats": {}, "planner_run": {"steps": [], "termination": "error"}}
        evaluator = evaluators[task["task_id"]]
        try:
            outcome = _evaluate(record, evaluator, catalogue)
        except Exception as exc:
            outcome = {"error": f"{type(exc).__name__}: {exc}"}
        record["expected_decision"] = evaluator.get("expected_decision")
        record["task_type"] = evaluator.get("task_type")
        record["outcome"] = outcome
        records.append(record)
        (OUT / f"{task['task_id']}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    summary = _aggregate(records, "rule")
    summary["search_backend"] = search_config
    summary["reserved_probability"] = {
        "n": len(records), "split": "reserved_test", "planner": "rule", "mode": "CPU_MCP",
        "one_time_run": True, "run_index": 1,
        "labels": "simulated rule labels; workflow generalization check only, not real-user accuracy",
    }
    (OUT / "summary.json").write_text(json.dumps({"schema": "agentrecx.p18.reserved_summary.v1",
                                                  "elapsed_seconds": round(time.perf_counter() - started, 2),
                                                  "summary": summary, "tasks": records},
                                                 indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    results = [{"task_id": row["task_id"], "task_type": row.get("task_type"),
                "expected_decision": row.get("expected_decision"),
                "terminal_status": (row.get("terminal") or {}).get("status"),
                "selected_candidate_id": (row.get("terminal") or {}).get("selected_candidate_id"),
                "outcome": row.get("outcome"), "error": row.get("error")}
               for row in records]
    (OUT / "reserved_results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
                                                encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "schema": "agentrecx.p18.reserved_run_manifest.v1",
        "run_index": 1, "one_time": True, "post_result_tuning": False,
        "completed_at_unix": time.time(), "elapsed_seconds": round(time.perf_counter() - started, 2),
        "freeze_manifest": "runs/p18/freeze_manifest.json",
        "search_backend": search_config, "planner": "rule", "mode": "CPU_MCP",
        "reserved_rows": len(records), "note": "Reserved Test run once after freeze; no rerun/tuning based on this result.",
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("recommend_tasks", "correct_top1", "false_ask",
                                                    "false_refusal", "hard_violations", "errors")
                      if key in summary}, indent=1))


if __name__ == "__main__":
    main()
