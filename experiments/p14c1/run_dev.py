"""P14-C1 Dev20 comparison: Rule, Qwen Base, Qwen SFT under the same BGE backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.p14c0 import run_dev as rd
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_http_proposer import HttpPlannerProposer, planner_enabled

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p14c1"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("rule", "qwen_base", "qwen_sft"), required=True)
    parser.add_argument("--service-url", default=os.environ.get("AGENTRECX_QWEN_PLANNER_URL"))
    args = parser.parse_args()
    catalogue = ElectronicsCatalogue.load()
    search_config = configure_catalogue(catalogue)
    if search_config["status"] != "active":
        raise RuntimeError(f"BGE integration failed: {search_config}")
    evaluator = {row["task_id"]: row for row in load_jsonl(DATA / "dev.evaluator.jsonl")}
    tasks = load_jsonl(DATA / "dev.agent.jsonl")
    if args.planner.startswith("qwen"):
        if not planner_enabled() or not args.service_url:
            raise RuntimeError("Qwen planner service is not configured")
        os.environ["AGENTRECX_QWEN_PLANNER"] = "1"
        os.environ["AGENTRECX_QWEN_PLANNER_URL"] = args.service_url
        os.environ.setdefault("AGENTRECX_QWEN_PLANNER_MODEL", "Qwen3.5-9B-P14C1")
    planner_kind = "rule" if args.planner == "rule" else "qwen"
    records = []
    for task in tasks:
        record = rd._run_one(catalogue, task, planner_kind=planner_kind)
        record["expected_decision"] = evaluator[task["task_id"]]["expected_decision"]
        record["outcome"] = rd._evaluate(record, evaluator[task["task_id"]], catalogue)
        record["planner_label"] = args.planner
        records.append(record)
    root = OUT / args.planner
    (root / "trajectories").mkdir(parents=True, exist_ok=True)
    for record in records:
        (root / "trajectories" / f"{record['task_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = rd._aggregate(records, args.planner)
    summary["search_backend"] = search_config
    summary["planner"] = args.planner
    (root / "summary.json").write_text(json.dumps(
        {"schema": f"agentrecx.p14c1.{args.planner}.v1", **summary, "tasks": records},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "tasks"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
