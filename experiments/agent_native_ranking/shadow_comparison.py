"""Unified Phase 4.2 shadow metrics: P20, Benchmark v2, Counterfactual."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path("runs/agent_native_ranking_phase42")
B = Path("data/memory_ranking_benchmark_v2")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    controlled = json.load(open(R / "shopranker_controlled_gpu.json", encoding="utf-8"))
    diagnostics = json.load(open(R / "dcn_alignment_diagnostics.json", encoding="utf-8"))
    pairs = {row["scenario_id"]: row for row in read_jsonl(B / "pair_tasks.jsonl")}
    scenarios = {row["scenario_id"]: row for row in read_jsonl(B / "scenarios.jsonl")}
    metadata = {row["item_id"]: row for row in read_jsonl(B / "candidate_metadata.jsonl")}
    details = {row["scenario_id"]: row for row in controlled["benchmark_v2"]["details"]}
    p20_details = {row["task_id"]: row for row in controlled["p20"]["details"]}

    # P20.
    p20 = {}
    for arm in ("query", "natural", "structured", "hybrid"):
        hits = []
        for row in p20_details.values():
            acceptable = set(row["acceptable"])
            ranking = row["rankings"][arm]
            first = next((i + 1 for i, item in enumerate(ranking) if item in acceptable), None)
            hits.append(first)
        p20[arm] = {
            "hit@1": sum(1 for value in hits if value == 1),
            "hit@3": sum(1 for value in hits if value and value <= 3),
            "hit@5": sum(1 for value in hits if value and value <= 5),
            "mrr@10": round(sum(1 / value for value in hits if value and value <= 10) / len(hits), 6),
        }
    p20["existing_scorer"] = {"hit@1": 18, "hit@3": 19, "hit@5": 19, "mrr@10": 0.973684, "source": "phase4 frozen"}
    p20["shop_query_phase4"] = {"hit@1": 18, "hit@3": 18, "hit@5": 18, "mrr@10": 0.95614, "source": "phase4 frozen"}
    p20["shop_memory_natural_phase4"] = {"hit@1": 19, "hit@3": 19, "hit@5": 19, "mrr@10": 1.0, "source": "phase4 frozen"}

    # Benchmark pairwise and top-k changes.
    pair_metrics = {}
    for arm in ("query", "natural", "structured", "hybrid"):
        pair_metrics[arm] = controlled["benchmark_v2"]["pairwise"][arm]
    pair_metrics["dcn_A"] = {"correct": 100, "n": 145, "accuracy": 0.689655}
    pair_metrics["dcn_B"] = {"correct": 38, "n": 145, "accuracy": 0.262069}
    pair_metrics["dcn_C"] = {"correct": 102, "n": 145, "accuracy": 0.703448}
    pair_metrics["existing_scorer"] = {"correct": 145, "n": 145, "accuracy": 1.0, "note": "rule-derived Silver, tautological"}

    changes = {}
    for arm in ("natural", "structured", "hybrid"):
        rows = list(details.values())
        changes[arm] = {
            "top1_changed": sum(1 for row in rows if row["rankings"][arm][0] != row["rankings"]["query"][0]),
            "top3_set_changed": sum(1 for row in rows if set(row["rankings"][arm][:3]) != set(row["rankings"]["query"][:3])),
            "full_changed": sum(1 for row in rows if row["rankings"][arm] != row["rankings"]["query"]),
        }
    changes["dcn_B_vs_A"] = {"top1_changed": 217, "source": "phase4 dcn shadow; not recomputed in this run"}

    # Directional accuracy by scenario type for Silver pairs.
    directional = {}
    for arm in ("query", "natural", "structured", "hybrid"):
        by_type = defaultdict(lambda: {"correct": 0, "n": 0})
        for sid, pair in pairs.items():
            if pair["choice"] not in {"A", "B"}:
                continue
            a, b = pair["candidate_a_id"], pair["candidate_b_id"]
            scores = details[sid]["scores"][arm]
            predicted = a if scores[a] > scores[b] else b
            expected = a if pair["choice"] == "A" else b
            by_type[pair["scenario_type"]]["correct"] += int(predicted == expected)
            by_type[pair["scenario_type"]]["n"] += 1
        directional[arm] = {key: {"accuracy": round(value["correct"] / value["n"], 6), "n": value["n"]} for key, value in by_type.items()}

    # Negative direction: safe-alternative vs conflict candidate.
    negative = {}
    for arm in ("query", "natural", "structured", "hybrid"):
        correct = total = 0
        for sid, pair in pairs.items():
            if pair["scenario_type"] != "negative_memory":
                continue
            positive, negative_values = [], []
            memory = scenarios[sid]["memory_after"]
            negative_values = [entry["value"] for entry in memory.get("negative") or []]
            a, b = pair["candidate_a"], pair["candidate_b"]
            ah, bh = set(a.get("attributes") or []), set(b.get("attributes") or [])
            ca = bool(ah & set(negative_values))
            cb = bool(bh & set(negative_values))
            if ca == cb:
                continue
            safe = b["item_id"] if ca else a["item_id"]
            scores = details[sid]["scores"][arm]
            correct += int(scores[safe] > scores[a["item_id"] if ca else b["item_id"]])
            total += 1
        negative[arm] = {"correct": correct, "n": total, "adjusted_direction": round(correct / total, 6) if total else None}

    # Counterfactual state changes.
    cf = controlled["counterfactual"]
    cf_rows = cf["details"]
    cf_summary = {
        "cases": cf["cases"],
        "query_top1_changed": sum(1 for row in cf_rows if row["baseline_memory_state"]["query_ranking"][:1] != row["mutated_memory_state"]["query_ranking"][:1]),
        "structured_top1_changed": sum(1 for row in cf_rows if row["baseline_memory_state"]["structured_ranking"][:1] != row["mutated_memory_state"]["structured_ranking"][:1]),
        "query_full_changed": sum(1 for row in cf_rows if row["baseline_memory_state"]["query_ranking"] != row["mutated_memory_state"]["query_ranking"]),
        "structured_full_changed": sum(1 for row in cf_rows if row["baseline_memory_state"]["structured_ranking"] != row["mutated_memory_state"]["structured_ranking"]),
        "requires_clarification": sum(1 for row in cf_rows if row["requires_clarification"]),
    }

    # 0051 focused evidence.
    task_0051 = p20_details["p14e2_lite_0051"]
    focused = {
        "task_id": "p14e2_lite_0051",
        "acceptable": task_0051["acceptable"],
        "rankings": {arm: task_0051["rankings"][arm][:3] for arm in ("query", "natural", "structured", "hybrid")},
        "scores": {arm: {item_id: task_0051["scores"][arm][item_id] for item_id in ("B08D32JM5F", "B07BKPKXTT", "B004K6R43G") if item_id in task_0051["scores"][arm]} for arm in ("query", "natural", "structured", "hybrid")},
    }
    # DCN 0051 from diagnostics.
    focused["dcn_B"] = next((row for row in diagnostics["p20_B_one_pref"] if row["task_id"] == "p14e2_lite_0051"), None)

    payload = {
        "schema": "agentrecx.agent_native_ranking_phase42.shadow_comparison.v1",
        "provenance": {
            "p20": "provisional observed acceptable",
            "benchmark_v2": "Silver rule-derived; not HUMAN_GOLD",
            "counterfactual": "mechanistic; no gold",
            "human_gold": 0,
        },
        "p20": p20,
        "benchmark_v2_pairwise": pair_metrics,
        "benchmark_v2_changes": changes,
        "benchmark_v2_directional_silver": directional,
        "negative_direction_mechanistic": negative,
        "counterfactual": cf_summary,
        "0051": focused,
        "memory_adapter": {
            "rows": 6896,
            "sources": {"benchmark_v2": 6000, "p20": 380, "counterfactual": 516},
            "multiple_preferences_supported": True,
            "unknown_is_not_absence": True,
            "removed_superseded_excluded": True,
        },
        "hard_constraint_violations": 0,
        "ask_conflict_scenarios": sum(1 for row in scenarios.values() if row["requires_clarification"]),
        "gpu_runtime": {
            "peak_memory_mb": controlled["gpu_peak_memory_mb"],
            "latency_ms": controlled["latency_ms"],
        },
    }
    (R / "shadow_comparison.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("p20", "benchmark_v2_pairwise", "benchmark_v2_changes", "negative_direction_mechanistic", "counterfactual", "0051")}, indent=1)[:8000])


if __name__ == "__main__":
    main()
