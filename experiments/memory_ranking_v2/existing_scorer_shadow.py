"""Existing PreferenceAwareScorer shadow on Memory Ranking Benchmark v2.

All 20 candidates are eligible for this synthetic benchmark: no hard requirements are
declared for the ESCI-derived queries, so hard-requirement violations are zero.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from recommendation.preference.scorer import PreferenceAwareScorer

BENCH = Path("data/memory_ranking_benchmark_v2")
OUT = Path("runs/memory_ranking_benchmark_v2")
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preferences_for(memory: dict) -> list[dict]:
    result = []
    for polarity in ("positive", "negative"):
        for entry in memory.get(polarity) or []:
            result.append({
                "value": entry.get("value"),
                "polarity": polarity,
                "kind": "synthetic_attribute",
                "source_turn_id": "benchmark_memory",
            })
    return result


def main():
    scenarios = read_jsonl(BENCH / "scenarios.jsonl")
    metadata = {row["item_id"]: row for row in read_jsonl(BENCH / "candidate_metadata.jsonl")}
    pairs = {row["scenario_id"]: row for row in read_jsonl(BENCH / "pair_tasks.jsonl")}
    scorer = PreferenceAwareScorer()
    rows = []
    latencies = []
    for scenario in scenarios:
        ids = list(scenario["candidate_pool_ids"])
        candidates = []
        for index, item_id in enumerate(ids, start=1):
            item = metadata[item_id]
            attributes = {
                attr: {"value": "yes", "confidence": 0.9, "evidence": "metadata"}
                for attr in item.get("attributes") or []
            }
            candidates.append({
                "product_ref": {"namespace": "amazon_electronics_c4", "item_id": item_id},
                "query_rank": index,
                "query_score": None,
                "attributes": attributes,
            })
        started = time.perf_counter()
        scored = scorer.score(
            preferences=preferences_for(scenario["memory_after"]),
            current_query=scenario["query"],
            candidates=candidates,
            hard_requirements=(),
            namespace="amazon_electronics_c4",
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        ranking = [row["product_ref"]["item_id"] for row in scored]
        scores = {row["product_ref"]["item_id"]: row["final_score"] for row in scored}
        pair = pairs.get(scenario["scenario_id"])
        pair_detail = None
        if pair:
            a, b = pair["candidate_a_id"], pair["candidate_b_id"]
            pair_detail = {
                "candidate_a": a, "candidate_b": b,
                "label": pair["choice"], "label_provenance": pair["label_provenance"],
                "a_rank": ranking.index(a) + 1, "b_rank": ranking.index(b) + 1,
                "a_score": scores[a], "b_score": scores[b],
                "prefers_a": scores[a] > scores[b],
            }
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "base_query_id": scenario["base_query_id"],
            "scenario_type": scenario["scenario_type"],
            "split": scenario["split"],
            "requires_clarification": scenario["requires_clarification"],
            "ranking": ranking,
            "scores": scores,
            "pair": pair_detail,
        })
    payload = {
        "schema": "agentrecx.memory_ranking_benchmark_v2.existing_scorer_shadow.v1",
        "scorer": scorer.version,
        "scenarios": len(rows),
        "pair_tasks": sum(1 for row in rows if row["pair"]),
        "hard_requirements": 0,
        "hard_violations": 0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "max": round(max(latencies), 3),
            "scope": "20 candidates, CPU Scorer, Python overhead",
        },
        "per_scenario": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "existing_scorer_shadow.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("schema", "scorer", "scenarios", "pair_tasks", "latency_ms")}, indent=1))


if __name__ == "__main__":
    main()
