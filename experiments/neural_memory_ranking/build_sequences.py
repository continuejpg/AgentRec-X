"""Build expanded Electronics sequence/ranking data from full observed interactions.

Target items are restricted to the current AgentRec-X Electronics catalog.  Histories
use **all** observed Electronics interactions (catalog and non-catalog).  Non-catalog
history items have no metadata in the available sidecar; they are encoded as an
UNKNOWN item with rating/time only, and the missing metadata is reported explicitly.

Remote paths are the AutoDL layout used in this milestone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.memory_ranking import features as p2

RAW = Path("/root/autodl-tmp/p13r/raw")
CATALOG_IDS = Path("/root/autodl-tmp/catalog_ids.txt")
OUT = Path("/root/autodl-tmp/neural_memory_v1")
SIDECAR = Path("/root/autodl-tmp/AgentRec-X-phase2/data/enrichment/electronics_metadata_sidecar.jsonl")

MS_PER_DAY = 86_400_000.0
MAX_HISTORY = 64
TRAIN_NEGATIVES = 32
EVAL_CANDIDATES = 100  # positive + 99 sampled unobserved catalog items
HALF_LIFE_DAYS = 180.0

ATTRS = p2.ATTRS
ITEM_FEATURE_NAMES = list(p2.item_feature_names()) + [
    "pop_seen_train", "pop_count_log", "pop_user_count_log", "pop_mean_rating",
    "pop_std_rating", "pop_high_share", "pop_low_share", "pop_popularity_pct",
]

MEMORY_NAMES = [
    "h_count", "h_count_log", "h_mean", "h_std", "h_min", "h_max",
    "h_high_share", "h_low_share", "h_neutral_share", "h_last_rating",
    "h_last3_mean", "h_last5_mean", "h_last_minus_prev", "h_trend",
    "h_days_since_last", "h_tenure_days", "h_avg_gap_days",
    "h_decay_mean", "h_decay_high_share", "h_decay_low_share",
    "h_catalog_count", "h_catalog_coverage", "h_catalog_attr_coverage_mean",
]
for kind in ("all", "high", "low"):
    MEMORY_NAMES.extend(f"h_{kind}_attr_{attr}" for attr in ATTRS)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def _load_catalog() -> list[str]:
    with CATALOG_IDS.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_ratings() -> pd.DataFrame:
    frames = []
    for name in ("Electronics.valid.csv", "Electronics.test.csv"):
        frame = pd.read_csv(RAW / name, usecols=["user_id", "parent_asin", "rating", "timestamp"])
        frame["source_split"] = name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[(frame["rating"] > 0) & (frame["rating"] <= 5)].copy()
    frame["parent_asin"] = frame["parent_asin"].astype(str)
    frame["user_id"] = frame["user_id"].astype(str)
    frame["rating"] = frame["rating"].astype(float)
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="first")
    frame = frame.sort_values(["user_id", "timestamp", "parent_asin"], kind="mergesort")
    return frame.reset_index(drop=True)


def _item_feature_matrix(catalog: list[str], metadata: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for item_id in catalog:
        meta = metadata.get(item_id) or p2.empty_item_metadata()
        row = p2.item_feature_row(item_id, meta)
        rows.append([float(row[name]) for name in p2.item_feature_names()])
    base = np.asarray(rows, dtype=np.float32)
    # NaN values in metadata price/log are filled with zero; the missing flag is
    # already an explicit feature (meta_missing / meta_has_price).
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    return base, np.ones(len(catalog), dtype=np.float32)


def _history_item_features(
    item_id: str,
    catalog_set: set[str],
    catalog_index: dict[str, int],
    base_item_features: np.ndarray,
) -> tuple[int, int]:
    if item_id in catalog_set:
        return catalog_index[item_id], 1
    return len(catalog_set), 0


def _memory_vector(
    *,
    history_length: int,
    history_ratings: list[float],
    history_timestamps: list[int],
    target_timestamp: int,
    catalog_history_ratings: list[float],
    catalog_history_attrs: list[tuple[str, ...]],
    all_attr_counts: dict[str, int],
    high_attr_counts: dict[str, int],
    low_attr_counts: dict[str, int],
) -> np.ndarray:
    out = np.zeros(len(MEMORY_NAMES), dtype=np.float32)
    n = history_length
    out[0] = n
    out[1] = math.log1p(n)
    if n == 0:
        return out
    ratings = np.asarray(history_ratings, dtype=np.float64)
    timestamps = np.asarray(history_timestamps, dtype=np.int64)
    out[2] = ratings.mean()
    out[3] = ratings.std(ddof=0)
    out[4] = ratings.min()
    out[5] = ratings.max()
    out[6] = (ratings >= 4).mean()
    out[7] = (ratings <= 2).mean()
    out[8] = (ratings == 3).mean()
    out[9] = ratings[-1]
    out[10] = ratings[-3:].mean()
    out[11] = ratings[-5:].mean()
    out[12] = ratings[-1] - ratings[:-1].mean() if n > 1 else 0.0
    x = np.arange(n, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float((x_centered * x_centered).sum())
    out[13] = (x_centered * (ratings - ratings.mean())).sum() / denom if denom > 0 else 0.0
    out[14] = (target_timestamp - timestamps[-1]) / MS_PER_DAY
    out[15] = (target_timestamp - timestamps[0]) / MS_PER_DAY
    out[16] = (timestamps[-1] - timestamps[0]) / MS_PER_DAY / (n - 1) if n > 1 else 0.0
    days = (target_timestamp - timestamps) / MS_PER_DAY
    decay = np.power(0.5, np.maximum(days, 0.0) / HALF_LIFE_DAYS)
    decay_sum = float(decay.sum())
    if decay_sum > 0:
        out[17] = float((decay * ratings).sum() / decay_sum)
        out[18] = float((decay * (ratings >= 4)).sum() / decay_sum)
        out[19] = float((decay * (ratings <= 2)).sum() / decay_sum)
    out[20] = len(catalog_history_ratings)
    out[21] = len(catalog_history_ratings) / n
    if catalog_history_ratings:
        out[22] = sum(len(attrs) for attrs in catalog_history_attrs) / len(ATTRS) / len(catalog_history_attrs)
    offset = 23
    for kind, counts in (("all", all_attr_counts), ("high", high_attr_counts), ("low", low_attr_counts)):
        for attr in ATTRS:
            out[offset] = counts[attr]
            offset += 1
    return out


def _candidate_ids(
    *,
    positive_index: int,
    excluded: set[int],
    n_items: int,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    negatives: list[int] = []
    while len(negatives) < k:
        value = int(rng.integers(0, n_items))
        if value == positive_index or value in excluded or value in negatives:
            continue
        negatives.append(value)
    candidates = np.asarray([positive_index] + negatives, dtype=np.int32)
    positive_position = int(rng.integers(0, k + 1))
    candidates[[0, positive_position]] = candidates[[positive_position, 0]]
    return candidates, np.asarray([positive_position], dtype=np.int32)


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()
    catalog_set = set(catalog)
    catalog_index = {item_id: index for index, item_id in enumerate(catalog)}
    metadata = p2.load_item_metadata(SIDECAR)
    base_item_features, _ = _item_feature_matrix(catalog, metadata)
    n_items = len(catalog)

    raw = _load_ratings()
    counts = raw.groupby("user_id").agg(n=("parent_asin", "size"), ncat=("parent_asin", lambda values: int(sum(v in catalog_set for v in values))))
    users = set(counts[counts["ncat"] >= 2].index)
    frame = raw[raw["user_id"].isin(users)].copy()
    print("users>=2 catalog", len(users), "full events", len(frame), flush=True)

    rows = {"train": [], "val": [], "test": []}
    memory_rows = {"train": [], "val": [], "test": []}
    sequence_rows = {"train": [], "val": [], "test": []}
    train_catalog_counts: dict[int, list[float]] = {}
    pairs: list[dict[str, Any]] = []

    for user_id, group in frame.groupby("user_id", sort=False):
        events = group.to_dict("records")
        catalog_positions = [i for i, event in enumerate(events) if event["parent_asin"] in catalog_set]
        ncat = len(catalog_positions)
        if ncat < 2:
            continue
        last_position = catalog_positions[-1]
        second_position = catalog_positions[-2]
        catalog_events = [events[pos] for pos in catalog_positions]
        history_ratings: list[float] = []
        history_timestamps: list[int] = []
        catalog_history_ratings: list[float] = []
        catalog_history_attrs: list[tuple[str, ...]] = []
        all_attr_counts = {attr: 0 for attr in ATTRS}
        high_attr_counts = {attr: 0 for attr in ATTRS}
        low_attr_counts = {attr: 0 for attr in ATTRS}
        catalog_history_ids: list[int] = []
        history_deque: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)

        for event_index, event in enumerate(events):
            item_id = str(event["parent_asin"])
            is_catalog = item_id in catalog_set
            if is_catalog:
                item_index = catalog_index[item_id]
                if event_index == last_position:
                    split = "test"
                elif ncat >= 3 and event_index == second_position:
                    split = "val"
                else:
                    split = "train"
                history_length = len(history_ratings)
                if split == "train" and history_length == 0:
                    # not a valid next-item target
                    pass
                else:
                    target_attrs = tuple((metadata.get(item_id) or p2.empty_item_metadata()).get("attributes") or ())
                    memory = _memory_vector(
                        history_length=history_length,
                        history_ratings=history_ratings,
                        history_timestamps=history_timestamps,
                        target_timestamp=int(event["timestamp"]),
                        catalog_history_ratings=catalog_history_ratings,
                        catalog_history_attrs=catalog_history_attrs,
                        all_attr_counts=all_attr_counts,
                        high_attr_counts=high_attr_counts,
                        low_attr_counts=low_attr_counts,
                    )
                    row = {
                        "user_id": str(user_id),
                        "target_item_index": int(item_index),
                        "target_item_id": item_id,
                        "target_timestamp": int(event["timestamp"]),
                        "target_rating": float(event["rating"]),
                        "split": split,
                        "history_length": int(history_length),
                        "catalog_history_count": len(catalog_history_ids),
                        "n_catalog_events": int(ncat),
                    }
                    rows[split].append(row)
                    memory_rows[split].append(memory)
                    seq = {
                        "history_item_index": [int(history_deque_entry["item_index"]) for history_deque_entry in history_deque],
                        "history_rating": [float(history_deque_entry["rating"]) for history_deque_entry in history_deque],
                        "history_timestamp": [int(history_deque_entry["timestamp"]) for history_deque_entry in history_deque],
                        "catalog_history_index": [int(value) for value in catalog_history_ids],
                    }
                    sequence_rows[split].append(seq)
                    if split == "train":
                        train_catalog_counts.setdefault(item_index, []).append(float(event["rating"]))
                    if split == "val":
                        val_index = len(rows["val"]) - 1
                        next_event = catalog_events[catalog_positions.index(event_index) + 1]
                        next_item = str(next_event["parent_asin"])
                        pair_group = str(user_id)
                        for candidate_event in (event, next_event):
                            pairs.append({
                                "pair_group_id": pair_group,
                                "val_row_index": int(val_index),
                                "candidate_item_index": int(catalog_index[str(candidate_event["parent_asin"])]),
                                "candidate_item_id": str(candidate_event["parent_asin"]),
                                "candidate_rating": float(candidate_event["rating"]),
                            })
            # update history with this event after target creation
            if is_catalog:
                item_meta = metadata.get(item_id) or p2.empty_item_metadata()
                attrs = tuple(item_meta.get("attributes") or ())
                catalog_history_ids.append(int(catalog_index[item_id]))
                catalog_history_ratings.append(float(event["rating"]))
                catalog_history_attrs.append(attrs)
                for attr in attrs:
                    all_attr_counts[attr] += 1
                    if float(event["rating"]) >= 4:
                        high_attr_counts[attr] += 1
                    elif float(event["rating"]) <= 2:
                        low_attr_counts[attr] += 1
                item_index = int(catalog_index[item_id])
            else:
                item_index = int(n_items)
            history_deque.append({
                "item_index": item_index,
                "rating": float(event["rating"]),
                "timestamp": int(event["timestamp"]),
            })
            history_ratings.append(float(event["rating"]))
            history_timestamps.append(int(event["timestamp"]))
        if len(history_ratings) > 1000:
            raise RuntimeError("unexpectedly long user history")

    # Popularity features from train target positives.
    global_mean = float(np.mean([r for values in train_catalog_counts.values() for r in values])) if train_catalog_counts else 4.0
    global_std = 1.0
    pop = np.zeros((n_items, len(ITEM_FEATURE_NAMES) - len(p2.item_feature_names())), dtype=np.float32)
    raw_counts = np.zeros(n_items, dtype=np.float64)
    raw_sums = np.zeros(n_items, dtype=np.float64)
    raw_sumsq = np.zeros(n_items, dtype=np.float64)
    raw_high = np.zeros(n_items, dtype=np.float64)
    raw_low = np.zeros(n_items, dtype=np.float64)
    for item_index, values in train_catalog_counts.items():
        arr = np.asarray(values, dtype=np.float64)
        raw_counts[item_index] = len(arr)
        raw_sums[item_index] = arr.sum()
        raw_sumsq[item_index] = (arr * arr).sum()
        raw_high[item_index] = (arr >= 4).sum()
        raw_low[item_index] = (arr <= 2).sum()
    mean = np.where(raw_counts > 0, raw_sums / np.maximum(raw_counts, 1), global_mean)
    var = np.where(raw_counts > 0, raw_sumsq / np.maximum(raw_counts, 1) - mean * mean, global_std ** 2)
    std = np.sqrt(np.maximum(var, 0))
    high_share = raw_high / np.maximum(raw_counts, 1)
    low_share = raw_low / np.maximum(raw_counts, 1)
    rank_pct = pd.Series(raw_counts).rank(pct=True, method="average").to_numpy()
    pop[:, 0] = (raw_counts > 0).astype(np.float32)
    pop[:, 1] = np.log1p(raw_counts)
    pop[:, 2] = np.log1p(raw_counts)  # user_count proxy (one target per user-item)
    pop[:, 3] = mean
    pop[:, 4] = std
    pop[:, 5] = high_share
    pop[:, 6] = low_share
    pop[:, 7] = rank_pct
    item_features = np.concatenate([base_item_features, pop], axis=1).astype(np.float32)
    item_features = np.vstack([item_features, np.zeros((1, item_features.shape[1]), dtype=np.float32)])
    item_feature_names = ITEM_FEATURE_NAMES

    # Persist item features / metadata mapping.
    np.save(OUT / "item_features.npy", item_features)
    (OUT / "item_ids.json").write_text(json.dumps(catalog), encoding="utf-8")
    (OUT / "item_feature_names.json").write_text(json.dumps(item_feature_names), encoding="utf-8")
    (OUT / "memory_feature_names.json").write_text(json.dumps(MEMORY_NAMES), encoding="utf-8")

    # Build split parquet / arrays / candidate sets.
    stats: dict[str, Any] = {"catalog_items": n_items, "users": int(len(users))}
    for split in ("train", "val", "test"):
        split_rows = rows[split]
        split_mem = np.asarray(memory_rows[split], dtype=np.float32)
        seqs = sequence_rows[split]
        data = pd.DataFrame(split_rows)
        data["history_item_index"] = [item["history_item_index"] for item in seqs]
        data["history_rating"] = [item["history_rating"] for item in seqs]
        data["history_timestamp"] = [item["history_timestamp"] for item in seqs]
        data["catalog_history_index"] = [item["catalog_history_index"] for item in seqs]
        data.to_parquet(OUT / f"{split}.parquet", index=False, compression="zstd")
        np.save(OUT / f"memory_{split}.npy", split_mem)
        stats[split] = {
            "rows": len(data),
            "users": int(data["user_id"].nunique()),
            "items": int(data["target_item_index"].nunique()),
            "history_mean": round(float(data["history_length"].mean()), 6),
            "history_max": int(data["history_length"].max()),
            "zero_history": int((data["history_length"] == 0).sum()),
            "rating_distribution": {str(int(k)): int(v) for k, v in data["target_rating"].value_counts().sort_index().items()},
        }

    # Candidate arrays.
    train = pd.read_parquet(OUT / "train.parquet")
    val = pd.read_parquet(OUT / "val.parquet")
    test = pd.read_parquet(OUT / "test.parquet")
    rng_arrays: dict[str, Any] = {}
    for split, data, k in (("train", train, TRAIN_NEGATIVES), ("val", val, EVAL_CANDIDATES - 1), ("test", test, EVAL_CANDIDATES - 1)):
        negatives = np.zeros((len(data), k), dtype=np.int32)
        positives = np.zeros((len(data), 1), dtype=np.int32)
        for position, row in enumerate(data.itertuples(index=False)):
            positive = int(row.target_item_index)
            excluded = set(int(value) for value in row.catalog_history_index)
            excluded.add(positive)
            candidates, positive_position = _candidate_ids(
                positive_index=positive,
                excluded=excluded,
                n_items=n_items,
                k=k,
                seed=_stable_seed(row.user_id, row.target_timestamp, split),
            )
            if split == "train":
                negatives[position] = [value for value in candidates if value != positive]
                positives[position, 0] = positive
            else:
                rng_arrays.setdefault(f"{split}_candidates", []).append(candidates)
                rng_arrays.setdefault(f"{split}_positive_position", []).append(int(positive_position[0]))
        if split == "train":
            np.save(OUT / "train_negatives.npy", negatives)
            np.save(OUT / "train_positive.npy", positives)
        else:
            rng_arrays[f"{split}_candidates"] = np.asarray(rng_arrays[f"{split}_candidates"], dtype=np.int32)
            rng_arrays[f"{split}_positive_position"] = np.asarray(rng_arrays[f"{split}_positive_position"], dtype=np.int32)
            np.save(OUT / f"{split}_candidates.npy", rng_arrays[f"{split}_candidates"])
            np.save(OUT / f"{split}_positive_position.npy", rng_arrays[f"{split}_positive_position"])
        stats[split]["candidate_rows"] = int(len(data) * (k + 1 if split != "train" else k + 1))

    common = np.where(test["n_catalog_events"].to_numpy() >= 3)[0].astype(np.int64)
    np.save(OUT / "test_common_indices.npy", common)
    stats["test_common"] = {
        "rows": int(len(common)),
        "users": int(test.iloc[common]["user_id"].nunique()),
    }

    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_parquet(OUT / "pairs.parquet", index=False, compression="zstd")
    stats["pairs"] = {
        "rows": int(len(pair_frame)),
        "groups": int(pair_frame["pair_group_id"].nunique()) if len(pair_frame) else 0,
        "group_size": 2,
    }
    stats["metadata"] = {
        "sidecar_items": int(len(metadata)),
        "catalog_items": n_items,
        "catalog_missing_sidecar": int(sum(1 for item in catalog if item not in metadata)),
        "non_catalog_history_metadata": "unavailable; encoded as UNKNOWN item with rating/time only",
    }
    stats["protocol"] = {
        "targets": "catalog items only",
        "history": "full Electronics interactions, last 64 retained for sequence encoder; aggregate memory uses full history",
        "train_negatives": TRAIN_NEGATIVES,
        "eval_candidates": EVAL_CANDIDATES,
        "unobserved_policy": "sampled unobserved catalog candidates for training/eval; not confirmed negatives",
        "split": "last catalog=test; second-last=val for users with >=3 catalog events; earlier catalog=train; train rows with empty history dropped",
    }
    stats["hashes"] = {
        "valid_sha256": _sha(RAW / "Electronics.valid.csv"),
        "test_sha256": _sha(RAW / "Electronics.test.csv"),
        "sidecar_sha256": _sha(SIDECAR),
        "catalog_ids_sha256": _sha(CATALOG_IDS),
    }
    (OUT / "manifest.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
