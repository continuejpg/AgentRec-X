"""Combine the two resumable P9 chunks into one 300-case summary (no model calls)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT = Path("runs/p9/candidate_depth")
FIRST = OUT / "per_case.first210.jsonl"
LAST = OUT / "per_case.last80.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


rows = load(FIRST) + load(LAST)
if len(rows) != 300 or len({row["case_id"] for row in rows}) != 300:
    raise SystemExit("expected exactly 300 unique cases")
(OUT / "per_case.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
)



count_fields = [
    "catalog_target", "history_target_50", "history_target_100",
    "sasrec_target_50", "sasrec_target_100", "sasrec_target_probe",
    "two_tower_target_50", "two_tower_target_100", "two_tower_target_probe",
]
reachability = {field: sum(1 for row in rows if row[field]) for field in count_fields}
newly = {
    "history_50_to_100": [
        {"case_id": row["case_id"], "target": row["target"]}
        for row in rows if not row["history_target_50"] and row["history_target_100"]
    ],
    "sasrec_50_to_100": [
        {"case_id": row["case_id"], "target": row["target"]}
        for row in rows if not row["sasrec_target_50"] and row["sasrec_target_100"]
    ],
    "two_tower_50_to_100": [
        {"case_id": row["case_id"], "target": row["target"]}
        for row in rows if not row["two_tower_target_50"] and row["two_tower_target_100"]
    ],
    "sasrec_100_to_probe": [
        {"case_id": row["case_id"], "target": row["target"]}
        for row in rows if not row["sasrec_target_100"] and row["sasrec_target_probe"]
    ],
    "two_tower_100_to_probe": [
        {"case_id": row["case_id"], "target": row["target"]}
        for row in rows if not row["two_tower_target_100"] and row["two_tower_target_probe"]
    ],
}
lost = {
    field: sum(1 for row in rows if row["component_lost_by_history"][field])
    for field in (
        "sasrec_100_not_history", "two_tower_100_not_history",
        "sasrec_50_not_history", "two_tower_50_not_history",
    )
}
final = {}
for label in (
    "history50:balanced", "history50:history_focus",
    "history100:balanced", "history100:history_focus",
):
    vals = [row["final_replay"][label] for row in rows]
    final[label] = {
        "recall@10": sum(v["recall@10"] for v in vals) / len(vals),
        "ndcg@10": sum(v["ndcg@10"] for v in vals) / len(vals),
        "mrr@10": sum(v["mrr@10"] for v in vals) / len(vals),
        "target_in_top10": sum(1 for v in vals if v["target_rank_top10"] is not None),
        "target_in_union": sum(1 for v in vals if v["target_rank_full"] is not None),
    }
summary = {
    "schema_version": "agentrecx.p9.candidate_depth_eval.v1",
    "created_utc": "2026-11-08",
    "cases": len(rows),
    "source": "P8-D 300 dev cases; validation target only; Holdout untouched",
    "config": {
        "history_k": 100,
        "component_probe_limit": 200,
        "serving_fusion": "serving_v1",
        "serving_sources": ["sequential", "two_tower", "catalog_lexical"],
        "serving_head_depth": 100,
        "serving_max_k": 100,
        "probe_label": "100 is legal serving MAX_K; 200 is an offline score-matrix headroom probe, not a Serving change",
        "catalog_budget": 50,
        "target": "validation_parent_asin",
    },
    "time": {
        "engine_load_seconds_split_runs": 173.24,
        "measured_per_case_seconds_sum": sum(
            row["timing_seconds"]["history100"]
            + row["timing_seconds"]["sasrec_scores_plus_200"]
            + row["timing_seconds"]["two_tower_scores_plus_200"]
            for row in rows
        ),
        "mean_history_source_seconds": (
            sum(row["timing_seconds"]["history100"] for row in rows) / len(rows)
        ),
        "mean_sasrec_seconds": sum(row["timing_seconds"]["sasrec_scores_plus_200"] for row in rows) / len(rows),
        "mean_two_tower_seconds": sum(row["timing_seconds"]["two_tower_scores_plus_200"] for row in rows) / len(rows),
    },
    "reachability": reachability,
    "newly_reachable": newly,
    "lost_by_fusion": lost,
    "final_top10": final,
    "notes": {
        "lost_by_fusion": "target appears in a component source ranking but not in the actual History source",
        "final_top10": "actual History source Top-50/Top-100 fused with cached Hybrid Catalog Top-50; balanced/history_focus; Top-10 only",
        "resume": "two resumable chunks (first 220, last 80) were combined without rerunning model scoring",
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
