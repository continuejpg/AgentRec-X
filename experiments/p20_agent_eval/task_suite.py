"""Build the natural-language Agent task suite for the P20 task-level evaluation.

All tasks are existing P14-E2/P14-B3 constructed user scenarios with public natural-language
queries.  Evaluator decisions are reused; labels remain provisional because no independent
human review was performed.  No P18 Reserved tasks are used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import prepare_hard

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data/p14e2/benchmark"
B3 = REPO / "data/p14b3_eval100"
OUT = REPO / "runs/p20_agent_eval"

SELECTED_B3 = [
    "p14b3_eval100_0001",  # recommend, wireless
    "p14b3_eval100_0045",  # recommend, headphones
    "p14b3_eval100_0065",  # recommend, no explicit hard constraint
    "p14b3_eval100_0073",  # recommend, wireless
    "p14b3_eval100_0081",  # clarification
    "p14b3_eval100_0089",  # clarification
    "p14b3_eval100_0093",  # no verified match, over-ear + mechanical
    "p14b3_eval100_0098",  # no verified match
    "p14b3_eval100_0109",  # insufficient evidence, budget
    "p14b3_eval100_0110",  # insufficient evidence, warranty
]

DEV_MEMORY_CONFLICT = {"p14e2_lite_0056", "p14e2_lite_0057"}


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["task_id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def _category(task_id: str, expected: str) -> str:
    if task_id in DEV_MEMORY_CONFLICT:
        return "memory_conflict_ask"
    if expected == "clarification_required":
        return "ambiguous_ask"
    if expected == "no_verified_match":
        return "unsatisfiable_no_match"
    if expected == "insufficient_evidence":
        return "unverifiable_abstain"
    return "recommend"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev_agents = _rows(DEV / "dev.agent.jsonl")
    dev_evals = _rows(DEV / "dev.evaluator.jsonl")
    b3_agents = _rows(B3 / "eval100.agent.jsonl")
    b3_evals = _rows(B3 / "eval100.evaluator.jsonl")
    tasks: list[dict[str, Any]] = []
    for task_id in sorted(dev_agents):
        task = dict(dev_agents[task_id])
        task["hard"] = prepare_hard(task["current_query"])
        task["expected"] = dev_evals[task_id]["expected_decision"]
        task["scored_evaluator"] = dev_evals[task_id]
        task["label_source"] = "p14e2_dev_automated_checked_provisional"
        task["split"] = "dev"
        task["category"] = _category(task_id, task["expected"])
        task["memory_fixture"] = str(DEV / "dev.agent.jsonl")
        tasks.append(task)
    for task_id in SELECTED_B3:
        task = dict(b3_agents[task_id])
        task["hard"] = prepare_hard(task["current_query"])
        task["expected"] = b3_evals[task_id]["expected_decision"]
        task["scored_evaluator"] = b3_evals[task_id]
        task["label_source"] = "p14b3_eval100_pending_human_review_provisional"
        task["split"] = "eval100"
        task["category"] = _category(task_id, task["expected"])
        task["memory_fixture"] = str(B3 / "eval100.agent.jsonl")
        tasks.append(task)

    manifest = {
        "schema": "agentrecx.p20.agent_task_suite.v1",
        "task_count": len(tasks),
        "splits": {split: sum(1 for t in tasks if t["split"] == split)
                   for split in ("dev", "eval100")},
        "categories": {cat: sum(1 for t in tasks if t["category"] == cat)
                       for cat in sorted({t["category"] for t in tasks})},
        "expected": {expected: sum(1 for t in tasks if t["expected"] == expected)
                     for expected in sorted({t["expected"] for t in tasks})},
        "label_status": "provisional_automated_or_pending_human_not_human_gold",
        "no_p18_reserved": True,
        "tasks": tasks,
    }
    (OUT / "task_suite.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("task_count", "splits", "categories", "expected", "label_status")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
