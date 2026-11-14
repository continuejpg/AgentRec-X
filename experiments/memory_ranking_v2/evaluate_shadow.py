"""Evaluate ShopRanker / Existing Scorer shadows on Benchmark v2 pair tasks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

BENCH = Path("data/memory_ranking_benchmark_v2")
RUNS = Path("runs/memory_ranking_benchmark_v2")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    pairs = {row["scenario_id"]: row for row in read_jsonl(BENCH / "pair_tasks.jsonl")}
    shop = {row["scenario_id"]: row for row in json.loads((RUNS / "shopranker_shadow.json").read_text(encoding="utf-8"))["per_scenario"]}
    existing = {row["scenario_id"]: row for row in json.loads((RUNS / "existing_scorer_shadow.json").read_text(encoding="utf-8"))["per_scenario"]}

    def predictions(row: dict, arm: str) -> dict:
        if arm == "shop_query":
            ranking = row["query_only_ranking"]
            scores = row["query_only_scores"]
        elif arm == "shop_memory":
            ranking = row["memory_ranking"]
            scores = row["memory_scores"]
        else:
            ranking = row["ranking"]
            scores = row["scores"]
        return {"ranking": ranking, "scores": scores}

    stats = defaultdict(lambda: {"correct": 0, "n": 0, "rank_delta": []})
    by_type = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "n": 0}))
    rows = []
    for scenario_id, pair in pairs.items():
        shop_row = shop[scenario_id]
        existing_row = existing[scenario_id]
        label = pair["choice"]
        detail = {"scenario_id": scenario_id, "base_query_id": pair["base_query_id"],
                  "split": pair["split"], "scenario_type": pair["scenario_type"],
                  "label": label, "label_provenance": pair["label_provenance"]}
        if label in {"A", "B"}:
            for arm in ("existing_scorer", "shop_query", "shop_memory"):
                pred = predictions(existing_row if arm == "existing_scorer" else shop_row, arm)
                a, b = pair["candidate_a_id"], pair["candidate_b_id"]
                preferred = a if label == "A" else b
                other = b if label == "A" else a
                predicted = a if pred["scores"][a] > pred["scores"][b] else b
                correct = predicted == preferred
                gap = abs(pred["ranking"].index(preferred) - pred["ranking"].index(other))
                stats[arm]["correct"] += int(correct)
                stats[arm]["n"] += 1
                stats[arm]["rank_delta"].append(gap)
                by_type[pair["scenario_type"]][arm]["correct"] += int(correct)
                by_type[pair["scenario_type"]][arm]["n"] += 1
                detail[arm] = {"preferred": preferred, "predicted": predicted, "correct": correct,
                               "rank_gap": pred["ranking"].index(preferred) - pred["ranking"].index(other)}
        rows.append(detail)

    aggregate = {}
    for arm, values in stats.items():
        n = values["n"]
        aggregate[arm] = {
            "pairwise_accuracy_silver": round(values["correct"] / n, 6) if n else None,
            "n_silver_pairs": n,
            "mean_rank_gap": round(sum(values["rank_delta"]) / n, 4) if n else None,
        }

    top1_changed = sum(1 for sid, row in shop.items() if row["query_only_ranking"][0] != row["memory_ranking"][0])
    top3_changed = sum(1 for sid, row in shop.items() if set(row["query_only_ranking"][:3]) != set(row["memory_ranking"][:3]))
    full_changed = sum(1 for sid, row in shop.items() if row["query_only_ranking"] != row["memory_ranking"])
    existing_vs_shop_top1 = sum(1 for sid, row in shop.items() if existing[sid]["ranking"][0] != row["memory_ranking"][0])
    existing_vs_shop_top3 = sum(1 for sid, row in shop.items() if set(existing[sid]["ranking"][:3]) != set(row["memory_ranking"][:3]))
    asK = sum(1 for pair in pairs.values() if pair["requires_clarification"])

    results = {
        "schema": "agentrecx.memory_ranking_benchmark_v2.shadow_metrics.v1",
        "scenarios": len(pairs),
        "scenarios_requiring_ask": asK,
        "silver_pairs": sum(1 for pair in pairs.values() if pair["choice"] in {"A", "B"}),
        "human_gold_pairs": 0,
        "aggregate_silver": aggregate,
        "by_scenario_type_silver": {
            scenario_type: {
                arm: {
                    "correct": values["correct"],
                    "n": values["n"],
                    "accuracy": round(values["correct"] / values["n"], 6) if values["n"] else None,
                }
                for arm, values in arms.items()
            }
            for scenario_type, arms in by_type.items()
        },
        "ranking_changes": {
            "shop_query_to_shop_memory": {
                "top1_changed": top1_changed,
                "top3_set_changed": top3_changed,
                "full_ranking_changed": full_changed,
            },
            "existing_to_shop_memory": {
                "top1_changed": existing_vs_shop_top1,
                "top3_set_changed": existing_vs_shop_top3,
            },
        },
        "hard_requirements": 0,
        "hard_constraint_violations": 0,
        "latency_ms": {
            "shopranker_query_memory": json.loads((RUNS / "shopranker_shadow.json").read_text())["latency_ms"],
            "existing_scorer": json.loads((RUNS / "existing_scorer_shadow.json").read_text())["latency_ms"],
        },
        "gpu_peak_memory_mb": json.loads((RUNS / "shopranker_shadow.json").read_text())["gpu_peak_memory_mb"],
        "pair_details": rows,
    }
    (RUNS / "shadow_metrics.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: results[key] for key in ("scenarios", "silver_pairs", "aggregate_silver", "ranking_changes")}, indent=1))


if __name__ == "__main__":
    main()
