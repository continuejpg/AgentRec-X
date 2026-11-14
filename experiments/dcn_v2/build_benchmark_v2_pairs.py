"""Build DCN-V2 pairwise training data from Benchmark v2 SILVER pairs.

Only SILVER A/B pairs are used. UNJUDGED/ASK scenarios are not turned into negatives.
Splits are inherited from base_query_id, so all memory variants of a query stay in the
same split. HUMAN_GOLD is zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data/memory_ranking_benchmark_v2"
OUT = BENCH / "dcn_v2_pairs"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_row(candidate: dict[str, Any], memory: dict[str, Any], query_mentions: set[str], winner: int, pair: dict[str, Any], side: str) -> dict[str, Any]:
    attrs = set(candidate.get("attributes") or [])
    positive = {item["value"] for item in memory.get("positive") or []}
    negative = {item["value"] for item in memory.get("negative") or []}
    pos_match = attrs & positive
    neg_conflict = attrs & negative
    known = len(attrs & set(ATTRS))
    unknown = [attr for attr in ATTRS if attr not in attrs]
    row = {
        "group_id": pair["pair_task_id"],
        "base_query_id": pair["base_query_id"],
        "scenario_id": pair["scenario_id"],
        "scenario_type": pair["scenario_type"],
        "split": pair["split"],
        "candidate_item_id": candidate["item_id"],
        "side": side,
        "winner": winner,
        "label_provenance": "SILVER",
        "query_len": float(len(pair["query"])),
        "query_token_count": float(len(pair["query"].split())),
        "query_mentions_count": float(len(query_mentions)),
        "hard_requirement_count": 0.0,
        "query_soft_preference_count": 0.0,
        "bge_rank": float(candidate.get("bge_rank") or 0),
        "bge_score": float(candidate.get("bge_score") or 0.0),
        "title_len": float(len(str(candidate.get("title") or ""))),
        "attribute_count": float(known),
        "attribute_coverage": float(known / len(ATTRS)),
        "unknown_count": float(len(unknown)),
        "has_metadata": 0.0 if candidate.get("metadata_missing") else 1.0,
        "active_positive_count": float(len(positive)),
        "active_negative_count": float(len(negative)),
        "active_memory_count": float(len(positive) + len(negative)),
        "positive_match_count": float(len(pos_match)),
        "negative_conflict_count": float(len(neg_conflict)),
        "positive_match_rate": float(len(pos_match) / known) if known else 0.0,
        "negative_conflict_rate": float(len(neg_conflict) / known) if known else 0.0,
        "memory_match_binary": 1.0 if pos_match else 0.0,
        "memory_conflict_binary": 1.0 if neg_conflict else 0.0,
    }
    for attr in ATTRS:
        row[f"candidate_attr_{attr}"] = 1.0 if attr in attrs else 0.0
        row[f"query_mentions_{attr}"] = 1.0 if attr in query_mentions else 0.0
        row[f"memory_positive_{attr}"] = 1.0 if attr in positive else 0.0
        row[f"memory_negative_{attr}"] = 1.0 if attr in negative else 0.0
    return row


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [row for row in read_jsonl(BENCH / "pair_tasks.jsonl") if row["label_provenance"] == "SILVER" and row["choice"] in {"A", "B"}]
    rows: list[dict[str, Any]] = []
    scenarios = {row["scenario_id"]: row for row in read_jsonl(BENCH / "scenarios.jsonl")}
    for pair in pairs:
        scenario = scenarios[pair["scenario_id"]]
        mentions = set(scenario.get("query_mentions") or [])
        for side, key in (("A", "candidate_a"), ("B", "candidate_b")):
            candidate = pair[key]
            winner = 1 if pair["choice"] == side else 0
            rows.append(candidate_row(candidate, pair["memory_after"], mentions, winner, pair, side))
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "pairs.parquet", index=False, compression="zstd")
    id_like = {"group_id", "base_query_id", "scenario_id", "scenario_type", "split", "candidate_item_id", "label_provenance", "side"}
    feature_names = [name for name in frame.columns if name not in id_like and name != "winner"]
    memory_features = [name for name in feature_names if name.startswith(("memory_", "active_", "positive_match_", "negative_conflict_"))]
    schema = {
        "schema": "agentrecx.dcn_v2_benchmark_pairs.feature_schema.v1",
        "numeric_features": feature_names,
        "memory_features": memory_features,
        "id_columns": sorted(id_like),
        "label": "winner",
        "label_provenance": "SILVER only; UNJUDGED/ASK excluded",
    }
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    split_counts = frame.groupby("split").size().to_dict()
    group_splits = frame.groupby("split")["group_id"].nunique().to_dict()
    manifest = {
        "schema": "agentrecx.dcn_v2_benchmark_pairs.manifest.v1",
        "dataset": "dcn_v2_pairs",
        "source": "memory_ranking_benchmark_v2 SILVER A/B pairs only",
        "groups": int(frame["group_id"].nunique()),
        "rows": int(len(frame)),
        "base_queries": int(frame["base_query_id"].nunique()),
        "label_provenance": {"SILVER": int(len(frame)), "HUMAN_GOLD": 0, "MECHANISTIC": 0, "UNJUDGED": 0},
        "split_rows": {str(key): int(value) for key, value in split_counts.items()},
        "split_groups": {str(key): int(value) for key, value in group_splits.items()},
        "features": len(feature_names),
        "memory_features": len(memory_features),
        "memory_off_features": len(feature_names) - len(memory_features),
        "split_policy": "base_query_family; all memory states of a query in one split",
        "warnings": [
            "SILVER is rule-derived, not independent human preference gold.",
            "Candidate A is not always the winner; labels are balanced by alternating pair order.",
            "UNJUDGED candidates are not treated as negatives.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
