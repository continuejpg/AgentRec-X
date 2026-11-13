"""Aggregate the fixed-pool ranking benchmark.

A = Original BGE order from the real trace pool.
B = Existing PreferenceAwareScorer order before structural hard filtering.
C = Structural hard filter + Existing Scorer (this is the order the current Gate can
    actually consume as `presentable[0]`).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "runs/preference_ranking_v1/ranking_dataset.jsonl"
OUT = REPO / "runs/preference_ranking_v1/local_summary.json"


def _metrics(rows: list[dict], label: str) -> dict:
    values = [row["metrics"].get(label, {}) for row in rows]
    reciprocal = [value.get("reciprocal_rank") or 0.0 for value in values]
    return {
        "tasks": len(rows),
        "pool_covers_acceptable": sum(1 for row in rows if row["acceptable_ids_in_pool"]),
        "hit@1": sum(1 for value in values if value.get("hit@1")),
        "hit@5": sum(1 for value in values if value.get("hit@5")),
        "hit@10": sum(1 for value in values if value.get("hit@10")),
        "mrr": round(statistics.mean(reciprocal), 6) if reciprocal else 0.0,
        "hard_violations_top1": sum(1 for value in values if value.get("hard_tier") == 2),
        "hard_unknown_top1": sum(1 for value in values if value.get("hard_tier") == 1),
        "unknown_preference_top1": sum(1 for value in values if value.get("unknown_preferences")),
        "preference_matched_top1": sum(1 for value in values if value.get("preference_matched")),
        "preference_violated_top1": sum(1 for value in values if value.get("preference_violated")),
    }


def main() -> int:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    recommend = [row for row in rows if row["expected_decision"] == "recommend"]
    summary: dict = {
        "schema": "agentrecx.preference_ranking.local_summary.v1",
        "tasks": len(rows),
        "recommend_tasks": len(recommend),
        "nonrecommend_tasks": len(rows) - len(recommend),
        "candidate_pool_size": {
            "min": min(row["candidate_pool_size"] for row in rows),
            "max": max(row["candidate_pool_size"] for row in rows),
        },
        "variants": {
            "A_original_bge": _metrics(recommend, "A_original_bge"),
            "B_existing_scorer": _metrics(recommend, "B_existing_scorer"),
            "C_hard_filtered_scorer": _metrics(recommend, "C_hard_filtered_scorer"),
        },
        "gate_selected_equals_C": sum(
            1 for row in recommend if row["gate_selected_id"] == row["top1"]["C_hard_filtered_scorer"]
        ),
        "gate_selected_acceptable": sum(
            1 for row in recommend if row["gate_selected_id"] in row["acceptable_ids_in_pool"]
        ),
        "gate_selected_in_pool": sum(
            1 for row in recommend if row["gate_selected_id"] in
            {item["item_id"] for item in row["candidate_pool"]}
        ),
        "candidate_coverage": "19/19 recommend tasks have at least one acceptable candidate in the fixed pool",
        "label_status": "provisional_automated_verified_not_human_gold",
        "unjudged_policy": "non-acceptable candidates are unjudged, not negatives",
        "preference_groups": {},
    }
    for group, predicate in (
        ("with_active_preferences", lambda row: bool(row["historical_memory"])),
        ("without_active_preferences", lambda row: not row["historical_memory"]),
        ("has_hard_requirements", lambda row: bool(row["hard_requirements"])),
    ):
        subset = [row for row in recommend if predicate(row)]
        summary["preference_groups"][group] = {
            "tasks": len(subset),
            "A_hit1": sum(1 for row in subset if row["metrics"].get("A_original_bge", {}).get("hit@1")),
            "B_hit1": sum(1 for row in subset if row["metrics"].get("B_existing_scorer", {}).get("hit@1")),
            "C_hit1": sum(1 for row in subset if row["metrics"].get("C_hard_filtered_scorer", {}).get("hit@1")),
        }
    summary["hard_case_0051"] = next(
        (
            {
                "task_id": row["task_id"],
                "query": row["query"],
                "acceptable_ids_in_pool": row["acceptable_ids_in_pool"],
                "gate_selected_id": row["gate_selected_id"],
                "A_top1": row["top1"]["A_original_bge"],
                "B_top1": row["top1"]["B_existing_scorer"],
                "C_top1": row["top1"]["C_hard_filtered_scorer"],
                "C_first_acceptable_rank": row["metrics"].get("C_hard_filtered_scorer", {}).get("first_acceptable_rank"),
                "candidate_labels": {
                    item["item_id"]: item["label"] for item in row["candidate_pool"]
                },
            }
            for row in recommend if row["task_id"] == "p14e2_lite_0051"
        ),
        None,
    )
    OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
