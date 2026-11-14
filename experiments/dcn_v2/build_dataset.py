"""Build the DCN-V2 Memory-Conditioned Ranking dataset skeleton.

The core target is: Query + Event-Sourced Memory + Candidate Evidence -> ranking score.

This dataset is deliberately labelled honestly:

* P20 fixed-pool rows have only provisional observed-acceptable labels; non-acceptable
  candidates are UNJUDGED, not negative examples.
* Counterfactual rows are rule-derived state-consistency cases, not user preference gold.
* No unobserved item is fabricated as a negative.

Because only 19 P20 groups exist, this dataset is NOT sufficient for formal DCN-V2
training.  It is the feature/schema/data pipeline for the next round.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RANKING = REPO / "runs/preference_ranking_v1/ranking_dataset.jsonl"
PER_TASK = REPO / "runs/memory_contribution_phase1/per_task_results.jsonl"
CASES = REPO / "runs/memory_contribution_phase1/counterfactual_cases.jsonl"
OUT = REPO / "data/dcn_v2_ranking_v1"

ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _attr_set(values: list[dict[str, Any]], *, polarity: str | None = None) -> set[str]:
    result = set()
    for item in values:
        value = str(item.get("value") or "")
        if value not in ATTRS:
            continue
        item_polarity = str(item.get("polarity") or "positive").lower()
        if polarity is None or item_polarity == polarity:
            result.add(value)
    return result


def _split_for(group_id: str) -> str:
    bucket = int(hashlib.md5(group_id.encode()).hexdigest(), 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def _candidate_features(item: dict[str, Any], active_pos: set[str], active_neg: set[str]) -> dict[str, float]:
    attrs = set((item.get("attributes") or {}).keys())
    attrs &= set(ATTRS)
    scorer = item.get("scorer") or {}
    present = sorted(attrs)
    pos_match = attrs & active_pos
    neg_conflict = attrs & active_neg
    unknown = [attr for attr in ATTRS if attr not in attrs]
    known = len(attrs)
    return {
        "bge_rank": float(item.get("original_bge_rank") or 0),
        "bge_score": float(item.get("original_bge_score") or 0.0),
        "title_len": float(len(str(item.get("title") or ""))),
        "attribute_count": float(known),
        "attribute_coverage": float(known / len(ATTRS)),
        "unknown_count": float(len(unknown)),
        "positive_match_count": float(len(pos_match)),
        "negative_conflict_count": float(len(neg_conflict)),
        "positive_match_rate": float(len(pos_match) / known) if known else 0.0,
        "negative_conflict_rate": float(len(neg_conflict) / known) if known else 0.0,
        "memory_match_binary": 1.0 if pos_match else 0.0,
        "memory_conflict_binary": 1.0 if neg_conflict else 0.0,
        "hard_tier": float(scorer.get("hard_tier") or 0),
        "role_match": 1.0 if str(scorer.get("role_status") or "") in {"MATCH", "NOT_REQUIRED"} else 0.0,
        "query_relevance_score": float(scorer.get("query_relevance_score") or 0.0),
        "preference_match_score": float(scorer.get("preference_match_score") or 0.0),
        "preference_conflict_score": float(scorer.get("preference_conflict_score") or 0.0),
        **{f"candidate_attr_{attr}": 1.0 if attr in attrs else 0.0 for attr in ATTRS},
    }


def _query_features(query: str, active_pos: set[str], active_neg: set[str], hard: list[dict[str, Any]], soft: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "query_len": float(len(query)),
        "query_token_count": float(len(query.split())),
        "hard_requirement_count": float(len(hard)),
        "query_soft_preference_count": float(len(soft)),
        "active_positive_count": float(len(active_pos)),
        "active_negative_count": float(len(active_neg)),
        "active_memory_count": float(len(active_pos) + len(active_neg)),
        **{f"query_memory_positive_{attr}": 1.0 if attr in active_pos else 0.0 for attr in ATTRS},
        **{f"query_memory_negative_{attr}": 1.0 if attr in active_neg else 0.0 for attr in ATTRS},
    }


def _memory_state(memory: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    return _attr_set(memory, polarity="positive"), _attr_set(memory, polarity="negative")


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    ranking_rows = _read_jsonl(RANKING)
    ranking_rows = [row for row in ranking_rows if row["expected_decision"] == "recommend"]
    ranking_by_id = {row["task_id"]: row for row in ranking_rows}
    per_task = _read_jsonl(PER_TASK)
    per_task_by_id = {row["task_id"]: row for row in per_task}
    cases = _read_jsonl(CASES)

    rows: list[dict[str, Any]] = []
    group_meta: dict[str, dict[str, Any]] = {}

    for task in ranking_rows:
        group_id = f"p20:{task['task_id']}"
        active_pos, active_neg = _memory_state(task.get("historical_memory") or [])
        soft = task.get("query_soft_preferences") or []
        hard = task.get("hard_requirements") or []
        split = _split_for(group_id)
        acceptable = set(task.get("acceptable_ids_in_pool") or [])
        group_meta[group_id] = {
            "group_id": group_id,
            "source": "p20_fixed_pool",
            "query_id": task["task_id"],
            "query": task["query"],
            "memory_snapshot_id": f"{task['task_id']}:active_memory_on",
            "split": split,
            "candidate_pool_size": len(task["candidate_pool"]),
        }
        query_features = _query_features(task["query"], active_pos, active_neg, hard, soft)
        for item in task["candidate_pool"]:
            candidate = _candidate_features(item, active_pos, active_neg)
            rows.append({
                "group_id": group_id,
                "source": "p20_fixed_pool",
                "query_id": task["task_id"],
                "memory_snapshot_id": f"{task['task_id']}:active_memory_on",
                "candidate_item_id": item["item_id"],
                "split": split,
                "label": 1.0 if item["item_id"] in acceptable else None,
                "label_status": "provisional_observed_acceptable" if item["item_id"] in acceptable else "unjudged",
                "label_provenance": "p20_provisional_acceptable_label_not_human_gold",
                **query_features,
                **candidate,
            })

    # Counterfactual mechanistic groups: useful for feature/schema and consistency
    # evaluation, NOT for supervised training.
    for case in cases:
        pool_id = case["pool_id"]
        ranking = ranking_by_id.get(pool_id)
        if ranking is None:
            continue
        for state in ("baseline_memory_state", "mutated_memory_state"):
            memory = (case.get(state) or {}).get("active_preferences") or []
            active_pos, active_neg = _memory_state(memory)
            group_id = f"cf:{case['case_id']}:{state}"
            group_meta[group_id] = {
                "group_id": group_id,
                "source": "counterfactual_mechanistic",
                "query_id": case["case_id"],
                "query": case["query"],
                "memory_snapshot_id": group_id,
                "split": "mechanistic",
                "candidate_pool_size": len(ranking["candidate_pool"]),
            }
            query_features = _query_features(case["query"], active_pos, active_neg,
                                             ranking.get("hard_requirements") or [], [])
            for item in ranking["candidate_pool"]:
                if item["item_id"] not in set(case.get("eligible_candidate_ids") or []):
                    continue
                rows.append({
                    "group_id": group_id,
                    "source": "counterfactual_mechanistic",
                    "query_id": case["case_id"],
                    "memory_snapshot_id": group_id,
                    "candidate_item_id": item["item_id"],
                    "split": "mechanistic",
                    "label": None,
                    "label_status": "not_usable_for_supervision",
                    "label_provenance": "rule_derived_counterfactual_state_consistency_not_user_gold",
                    **query_features,
                    **_candidate_features(item, active_pos, active_neg),
                })

    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "groups.parquet", index=False, compression="zstd")
    query_feature_names = [name for name in _query_features("", set(), set(), [], [])]
    candidate_feature_names = list(_candidate_features({"attributes": {}}, set(), set()).keys())
    numeric_features = [name for name in frame.columns if name not in {
        "group_id", "source", "query_id", "memory_snapshot_id", "candidate_item_id",
        "split", "label", "label_status", "label_provenance",
    }]
    memory_feature_names = [
        name for name in numeric_features
        if name.startswith((
            "query_memory_", "active_positive_count", "active_negative_count",
            "active_memory_count", "positive_match_", "negative_conflict_",
            "memory_match_", "memory_conflict_", "preference_match_",
            "preference_conflict_",
        ))
    ]
    schema = {
        "schema": "agentrecx.dcn_v2.feature_schema.v1",
        "version": "dcn_v2_ranking_v1",
        "id_columns": ["group_id", "source", "query_id", "memory_snapshot_id", "candidate_item_id"],
        "label_columns": ["label", "label_status", "label_provenance"],
        "numeric_features": numeric_features,
        "query_features": query_feature_names,
        "candidate_features": candidate_feature_names,
        "memory_features": memory_feature_names,
        "leakage_controls": [
            "group_id/query_id/candidate_item_id are identifiers, not model inputs",
            "no unobserved candidate is labelled as a negative",
            "P20 labels are provisional observed-acceptable, not human pairwise preference",
            "counterfactual groups are mechanistic state-consistency cases, not user preference gold",
        ],
    }
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    by_source = frame.groupby("source").agg(rows=("group_id", "size"), groups=("group_id", "nunique")).reset_index()
    split_counts = frame.groupby(["source", "split"]).size().to_dict()
    group_source = {group: meta["source"] for group, meta in group_meta.items()}
    manifest = {
        "schema": "agentrecx.dcn_v2.dataset_manifest.v1",
        "dataset": "dcn_v2_ranking_v1",
        "rows": int(len(frame)),
        "groups": int(frame["group_id"].nunique()),
        "sources": by_source.to_dict("records"),
        "split_counts": {f"{key[0]}/{key[1]}": int(value) for key, value in split_counts.items()},
        "label_counts": {
            "acceptable": int((frame["label"] == 1.0).sum()),
            "unjudged": int(frame["label_status"].eq("unjudged").sum()),
            "mechanistic_unlabelled": int(frame["label_status"].eq("not_usable_for_supervision").sum()),
        },
        "group_meta": group_meta,
        "training_ready": False,
        "training_block_reason": (
            "Only 19 P20 recommend groups / 380 candidate rows with provisional acceptable "
            "labels; non-acceptable candidates are unjudged, not negatives. The 16 "
            "counterfactual cases are mechanistic state-consistency tests, not independent "
            "user preference labels. This is insufficient for formal DCN-V2 Memory ON/OFF training."
        ),
        "available_but_not_memory_labels": {
            "ESCI": "human query relevance E/S/C/I; no Event-Sourced Memory",
            "TREC 2025": "human relationship qrels; test-only; no Event-Sourced Memory",
            "Amazon Reviews": "observed ratings / implicit interactions; no explicit memory events",
        },
        "features": {
            "query": len(query_feature_names),
            "candidate": len(candidate_feature_names),
            "memory": len(memory_feature_names),
            "numeric_total": len(numeric_features),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
