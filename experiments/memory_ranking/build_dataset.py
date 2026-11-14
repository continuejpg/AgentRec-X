"""Build the real Amazon Reviews-derived Memory Ranking v1 dataset.

The only raw labels are observed Amazon Electronics ratings.  No user was asked to
rate a recommendation, no unobserved item is treated as negative, and no synthetic
preference is generated.

Run from the repository root with the local CPU Python (no torch required)::

    python -m experiments.memory_ranking.build_dataset
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from experiments.memory_ranking import features

OUT = features.OUT_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _history_length_bucket(length: int) -> str:
    if length == 0:
        return "0"
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 4:
        return "3-4"
    if length <= 9:
        return "5-9"
    return "10+"


def _length_distribution(frame: pd.DataFrame) -> dict[str, int]:
    buckets = Counter(_history_length_bucket(int(value)) for value in frame["history_length"])
    return {name: int(buckets.get(name, 0)) for name in ("0", "1", "2", "3-4", "5-9", "10+")}


def _rating_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target_rating"].value_counts().sort_index()
    return {str(int(rating)): int(count) for rating, count in counts.items()}


def _target_row(
    *,
    user_id: str,
    event: Mapping[str, Any],
    split: str,
    history: list[dict[str, Any]],
    item_feature_lookup: Mapping[str, Mapping[str, Any]],
    metadata_lookup: Mapping[str, Mapping[str, Any]],
    popularity: Mapping[str, Any],
    target_index: int,
    user_n_events: int,
    pair_group_id: str | None = None,
) -> dict[str, Any]:
    item_id = str(event["item_id"])
    rating = float(event["rating"])
    timestamp = int(event["timestamp"])
    meta = metadata_lookup.get(item_id) or features.empty_item_metadata()
    item_row = item_feature_lookup[item_id]
    target_attributes = list(meta.get("attributes") or ())
    target_main = meta.get("main_category")
    target_leaf = meta.get("leaf_category")
    row: dict[str, Any] = {
        "user_id": user_id,
        "target_item_id": item_id,
        "target_timestamp": timestamp,
        "split": split,
        "history_length": len(history),
        "history_item_ids": [str(item["item_id"]) for item in history],
        "history_ratings": [float(item["rating"]) for item in history],
        "history_timestamps": [int(item["timestamp"]) for item in history],
        "history_metadata_json": [
            json.dumps(
                {
                    "item_id": str(item["item_id"]),
                    "title": (metadata_lookup.get(item["item_id"]) or features.empty_item_metadata()).get("title"),
                    "main_category": (metadata_lookup.get(item["item_id"]) or features.empty_item_metadata()).get("main_category"),
                    "leaf_category": (metadata_lookup.get(item["item_id"]) or features.empty_item_metadata()).get("leaf_category"),
                    "attributes": list(
                        (metadata_lookup.get(item["item_id"]) or features.empty_item_metadata()).get("attributes") or ()
                    ),
                    "attribute_coverage": float(
                        (metadata_lookup.get(item["item_id"]) or features.empty_item_metadata()).get("attribute_coverage") or 0.0
                    ),
                    "price": (metadata_lookup.get(item["item_id"]) or features.empty_item_metadata()).get("price"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for item in history
        ],
        "target_title": meta.get("title"),
        "target_main_category": target_main,
        "target_leaf_category": target_leaf,
        "target_attributes": target_attributes,
        "target_attribute_count": int(meta.get("attribute_count") or 0),
        "target_attribute_coverage": float(meta.get("attribute_coverage") or 0.0),
        "target_price": meta.get("price"),
        "target_rating": rating,
        "target_high": 1.0 if rating >= 4 else (0.0 if rating <= 2 else None),
        "label_source": "amazon_reviews_2023_electronics_observed_rating",
        "data_source": "electronics_catalog_ratings.csv",
        "source_file_split": str(event.get("source_split") or ""),
        "target_index": int(target_index),
        "user_n_events": int(user_n_events),
    }
    if pair_group_id is not None:
        row["pair_group_id"] = pair_group_id
    for name in features.non_memory_feature_names():
        if name.startswith("pop_"):
            continue
        row[name] = item_row[name]
    row.update(
        features.popularity_feature_row(
            item_id=item_id,
            rating=rating,
            split=split,
            target_timestamp=timestamp,
            item_main_category=target_main,
            item_leaf_category=target_leaf,
            table=popularity,
            category_counts=popularity["category_counts"],
        )
    )
    row.update(
        features.memory_feature_row(
            history=history,
            target_attributes=target_attributes,
            target_main_category=target_main,
            target_leaf_category=target_leaf,
            target_timestamp=timestamp,
        )
    )
    return row


def _history_entry(
    event: Mapping[str, Any], metadata_lookup: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    item_id = str(event["item_id"])
    meta = metadata_lookup.get(item_id) or features.empty_item_metadata()
    return {
        "item_id": item_id,
        "rating": float(event["rating"]),
        "timestamp": int(event["timestamp"]),
        "attributes": tuple(meta.get("attributes") or ()),
        "attribute_coverage": float(meta.get("attribute_coverage") or 0.0),
        "main_category": meta.get("main_category"),
        "leaf_category": meta.get("leaf_category"),
        "price_log": meta.get("price_log", float("nan")),
    }


def build(ratings_path: Path = features.RATINGS_PATH) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ratings_path)
    raw_rows = len(raw)
    raw_users = int(raw["user_id"].nunique())
    raw_items = int(raw["parent_asin"].nunique())
    frame = features.load_ratings()
    rating_items = sorted(frame["item_id"].unique())
    metadata = features.load_item_metadata()
    item_feature_frame = features.build_item_feature_frame(rating_items, metadata)
    item_feature_lookup = {
        str(row["item_id"]): row for row in item_feature_frame.to_dict("records")
    }
    metadata_lookup = {
        item_id: (metadata.get(item_id) or features.empty_item_metadata())
        for item_id in rating_items
    }
    popularity = features.popularity_tables(frame, metadata_lookup)

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        events = group.sort_values("timestamp", kind="mergesort").to_dict("records")
        n = len(events)
        history: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            split = str(event["split"])
            row = _target_row(
                user_id=str(user_id),
                event=event,
                split=split,
                history=history,
                item_feature_lookup=item_feature_lookup,
                metadata_lookup=metadata_lookup,
                popularity=popularity,
                target_index=index,
                user_n_events=n,
            )
            rows.append(row)
            if split == "val" and n >= 4:
                pair_group_id = str(user_id)
                # Same history cutoff for both real rated candidates.  This is a
                # separate ranking-evaluation group, never used for training.
                for candidate in (event, events[index + 1]):
                    pair_rows.append(
                        _target_row(
                            user_id=str(user_id),
                            event=candidate,
                            split="ranking_pair_test",
                            history=history,
                            item_feature_lookup=item_feature_lookup,
                            metadata_lookup=metadata_lookup,
                            popularity=popularity,
                            target_index=index if candidate is event else index + 1,
                            user_n_events=n,
                            pair_group_id=pair_group_id,
                        )
                    )
            history.append(_history_entry(event, metadata_lookup))

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise RuntimeError("no training rows were built")
    pair_frame = pd.DataFrame(pair_rows)
    label_and_meta_columns = list(features.ID_COLUMNS)
    feature_columns = features.non_memory_feature_names() + features.memory_feature_names()
    ordered = (
        label_and_meta_columns
        + [name for name in feature_columns if name not in label_and_meta_columns]
        + [name for name in dataset.columns if name not in set(label_and_meta_columns) | set(feature_columns)]
    )
    dataset = dataset[[name for name in ordered if name in dataset.columns]]

    split_frames = {
        "train": dataset[dataset["split"] == "train"].reset_index(drop=True),
        "val": dataset[dataset["split"] == "val"].reset_index(drop=True),
        "test": dataset[dataset["split"] == "test"].reset_index(drop=True),
    }
    for name, frame_part in split_frames.items():
        frame_part.to_parquet(OUT / f"{name}.parquet", index=False, compression="zstd")
    item_feature_frame.to_parquet(OUT / "item_features.parquet", index=False, compression="zstd")
    if not pair_frame.empty:
        pair_frame = pair_frame[
            ["pair_group_id"]
            + [name for name in ordered if name in pair_frame.columns]
            + [name for name in pair_frame.columns if name not in ordered and name != "pair_group_id"]
        ]
        pair_frame.to_parquet(OUT / "ranking_pairs.parquet", index=False, compression="zstd")

    split_stats = {}
    for name, frame_part in split_frames.items():
        split_stats[name] = {
            "rows": len(frame_part),
            "users": int(frame_part["user_id"].nunique()),
            "items": int(frame_part["target_item_id"].nunique()),
            "history_length_mean": round(float(frame_part["history_length"].mean()), 6),
            "history_length_max": int(frame_part["history_length"].max()),
            "history_length_distribution": _length_distribution(frame_part),
            "rating_distribution": _rating_distribution(frame_part),
            "high_label_count": int((frame_part["target_high"] == 1).sum()),
            "low_label_count": int((frame_part["target_high"] == 0).sum()),
            "neutral_rating_count": int((frame_part["target_rating"] == 3).sum()),
        }

    users = frame["user_id"].nunique()
    item_count = int(frame["item_id"].nunique())
    metadata_missing = int(sum(1 for item_id in rating_items if item_id not in metadata))
    history_lengths = dataset["history_length"]
    group_counts = frame.groupby("user_id")["item_id"].size()
    ranking_groups = {
        "pointwise_target_protocol": "one target event per split per user; no multi-candidate group in train/val/test",
        "users_ge_4": int((group_counts >= 4).sum()),
        "users_ge_5": int((group_counts >= 5).sum()),
        "users_ge_6": int((group_counts >= 6).sum()),
        "heldout_pair_groups_size_2": int(len(pair_frame) // 2) if not pair_frame.empty else 0,
        "heldout_pair_users": int(pair_frame["pair_group_id"].nunique()) if not pair_frame.empty else 0,
        "note": "ranking_pairs.parquet groups the last two real-rated items under one history cutoff; group size is 2 and nDCG@5/@10 is not meaningful",
    }

    feature_schema = {
        "schema": "agentrecx.memory_ranking.feature_schema.v1",
        "dataset": "memory_ranking_v1",
        "label_task": "observed_rating_regression_and_high_low_classification",
        "id_and_metadata_columns": list(features.ID_COLUMNS),
        "label_columns": list(features.LABEL_COLUMNS),
        "non_memory_features": features.non_memory_feature_names(),
        "memory_features": features.memory_feature_names(),
        "all_features_memory_off": features.non_memory_feature_names(),
        "all_features_memory_on": features.non_memory_feature_names() + features.memory_feature_names(),
        "feature_sources": {
            "meta_*": "item metadata sidecar (title/features/details/categories/price); average_rating and rating_number excluded",
            "pop_*": "training-split item/category popularity; leave-one-out for training rows",
            "mem_*": "inferred_from_ratings: prefix user ratings and item attributes; not explicit_event_sourced_preference",
        },
        "forbidden_features": [
            "user_id", "target_item_id", "target_rating", "target_high",
            "history_ratings", "history_item_ids", "history_timestamps",
            "history_metadata_json", "source_file_split", "target_title",
        ],
        "leakage_controls": features.leakage_controls(),
    }
    _json_dump(OUT / "feature_schema.json", feature_schema)

    source_files = {
        "ratings": {
            "path": str(ratings_path.relative_to(features.REPO)).replace("\\", "/"),
            "sha256": _sha256(ratings_path),
            "bytes": ratings_path.stat().st_size,
            "rows_read": int(raw_rows),
            "local_raw_train_partition": False,
            "local_source_splits": ["Electronics.valid", "Electronics.test"],
        },
        "metadata_sidecar": {
            "path": str(features.SIDECAR_PATH.relative_to(features.REPO)).replace("\\", "/"),
            "sha256": _sha256(features.SIDECAR_PATH),
            "bytes": features.SIDECAR_PATH.stat().st_size,
        },
    }
    manifest = {
        "schema": "agentrecx.memory_ranking.v1.manifest",
        "dataset": "memory_ranking_v1",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "task": "Memory-Aware Ranking v1: real Amazon Electronics ratings, pointwise LightGBM",
        "raw_data": source_files,
        "rows": {
            "raw_csv_rows": int(raw_rows),
            "raw_csv_users": raw_users,
            "raw_csv_items": raw_items,
            "valid_rating_rows": int(raw["rating"].between(1, 5).sum()),
            "rows_after_user_filter": int(len(frame)),
            "rating_events_used": int(len(dataset)),
            "train": len(split_frames["train"]),
            "val": len(split_frames["val"]),
            "test": len(split_frames["test"]),
            "users": int(users),
            "items_used": int(item_count),
            "users_ge_3": int(users),
            "users_excluded_lt_3": int(raw_users - users),
            "users_ge_4": int((group_counts >= 4).sum()),
            "users_ge_5": int((group_counts >= 5).sum()),
        },
        "target": {
            "label_type": "observed_rating_1_5",
            "human_pairwise": False,
            "unobserved_policy": "UNOBSERVED/UNJUDGED, never a negative",
            "classification": ">=4 positive, <=2 negative, rating==3 retained for regression only",
            "label_source": "Amazon Reviews 2023 Electronics rating-only split files",
        },
        "split_rule": features.split_rule_text(),
        "split_stats": split_stats,
        "history": {
            "mean": round(float(history_lengths.mean()), 6),
            "median": float(history_lengths.median()),
            "max": int(history_lengths.max()),
            "zero_history_rows": int((history_lengths == 0).sum()),
            "nonzero_history_rows": int((history_lengths > 0).sum()),
            "distribution": {
                "all": _length_distribution(dataset),
                **{
                    f"split_{name}": _length_distribution(frame_part)
                    for name, frame_part in split_frames.items()
                },
            },
        },
        "metadata": {
            "items_used": int(item_count),
            "source_catalog_rating_items": raw_items,
            "sidecar_items": int(len(metadata)),
            "rating_items_missing_sidecar_in_used_set": metadata_missing,
            "coverage_in_used_set": round(1.0 - metadata_missing / item_count, 6),
            "features_excluded_for_leakage": ["average_rating", "rating_number", "store (unused)"],
        },
        "features": {
            "non_memory_count": len(features.non_memory_feature_names()),
            "memory_count": len(features.memory_feature_names()),
            "memory_off_count": len(features.non_memory_feature_names()),
            "memory_on_count": len(features.non_memory_feature_names()) + len(features.memory_feature_names()),
            "memory_feature_source": "inferred_from_ratings",
            "explicit_event_sourced_preference": "not present in this dataset",
        },
        "ranking_groups": ranking_groups,
        "leakage_controls": features.leakage_controls(),
        "files": {
            "train": {"rows": len(split_frames["train"]), "sha256": _sha256(OUT / "train.parquet")},
            "val": {"rows": len(split_frames["val"]), "sha256": _sha256(OUT / "val.parquet")},
            "test": {"rows": len(split_frames["test"]), "sha256": _sha256(OUT / "test.parquet")},
            "item_features": {"rows": len(item_feature_frame), "sha256": _sha256(OUT / "item_features.parquet")},
            "ranking_pairs": {
                "rows": len(pair_frame),
                "groups": int(pair_frame["pair_group_id"].nunique()) if not pair_frame.empty else 0,
                "sha256": _sha256(OUT / "ranking_pairs.parquet") if not pair_frame.empty else None,
            },
            "feature_schema": {"sha256": _sha256(OUT / "feature_schema.json")},
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "gpu_used": False,
        },
        "git_head": _git_head(),
        "protection": {
            "production_ranker_modified": False,
            "planner_modified": False,
            "scorer_modified": False,
            "gate_modified": False,
            "p18_modified": False,
        },
    }
    _json_dump(OUT / "manifest.json", manifest)
    return manifest


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=features.REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    manifest = build()
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
