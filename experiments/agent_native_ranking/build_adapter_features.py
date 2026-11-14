"""Build unified Memory Adapter feature rows for Phase 4.2."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.agent_native_ranking.memory_adapter import candidate_features

REPO = Path(__file__).resolve().parents[2]
BM = REPO / "data/memory_ranking_benchmark_v2"
P20 = REPO / "runs/preference_ranking_v1/ranking_dataset.jsonl"
CF = REPO / "runs/memory_contribution_phase1/counterfactual_cases.jsonl"
OUT = REPO / "runs/agent_native_ranking_phase42/memory_adapter_features.jsonl"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    scenarios = read_jsonl(BM / "scenarios.jsonl")
    metadata = {row["item_id"]: row for row in read_jsonl(BM / "candidate_metadata.jsonl")}
    for scenario in scenarios:
        memory = scenario["memory_after"]
        for item_id in scenario["candidate_pool_ids"]:
            item = metadata[item_id]
            features = candidate_features(item, memory, query_soft=[])
            features["requires_ask"] = float(bool(scenario.get("requires_clarification")))
            rows.append({
                "source": "benchmark_v2",
                "group_id": scenario["scenario_id"],
                "base_query_id": scenario["base_query_id"],
                "candidate_id": item_id,
                "scenario_type": scenario["scenario_type"],
                "split": scenario["split"],
                "memory_snapshot_id": scenario["scenario_id"],
                "features": features,
            })
    p20 = read_jsonl(P20)
    p20 = [row for row in p20 if row.get("expected_decision") == "recommend"]
    for task in p20:
        memory = task.get("historical_memory") or []
        for item in task["candidate_pool"]:
            features = candidate_features(item, memory, query_soft=task.get("query_soft_preferences") or [])
            features["requires_ask"] = 0.0
            rows.append({
                "source": "p20",
                "group_id": task["task_id"],
                "base_query_id": task["task_id"],
                "candidate_id": item["item_id"],
                "scenario_type": "p20_recommend",
                "split": "frozen_p20",
                "memory_snapshot_id": f"{task['task_id']}:active",
                "features": features,
            })
    cases = read_jsonl(CF)
    pool_by_id = {row["task_id"]: row for row in p20}
    for case in cases:
        pool = pool_by_id.get(case["pool_id"])
        if not pool:
            continue
        for state_name in ("baseline_memory_state", "mutated_memory_state"):
            memory = (case.get(state_name) or {}).get("active_preferences") or []
            for item in pool["candidate_pool"]:
                if item["item_id"] not in set(case.get("eligible_candidate_ids") or []):
                    continue
                features = candidate_features(item, memory, query_soft=[])
                features["requires_ask"] = float(bool(case.get("requires_clarification")))
                rows.append({
                    "source": "counterfactual",
                    "group_id": case["case_id"],
                    "base_query_id": case["pool_id"],
                    "candidate_id": item["item_id"],
                    "scenario_type": case["category"],
                    "split": "mechanistic",
                    "memory_snapshot_id": f"{case['case_id']}:{state_name}",
                    "features": features,
                })
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sources": {source: sum(1 for row in rows if row["source"] == source) for source in ("benchmark_v2", "p20", "counterfactual")}}, indent=1))


if __name__ == "__main__":
    main()
