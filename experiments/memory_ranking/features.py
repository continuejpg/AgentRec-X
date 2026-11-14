"""Feature engineering for Memory-Aware Recommendation v1.

The feature contract is deliberately explicit:

* ``meta_*`` features are item metadata available at prediction time.  They never use
  the target rating, the target user's other ratings, or review-derived aggregates
  (``average_rating`` / ``rating_number`` from the metadata sidecar are deliberately
  excluded).
* ``pop_*`` features are item/category popularity statistics computed from the
  **training split only**.  For a training row the item's own rating and count are
  removed (leave-one-out) so the row cannot read its own label.  Validation and test
  rows only ever see training statistics.
* ``mem_*`` features are inferred from the user's *earlier* ratings.  They are
  ``inferred_from_ratings`` preferences, not Event-Sourced explicit preferences.
  The current AgentRec-X ``PreferenceAwareScorer`` uses explicit memory events; the
  adapter in ``experiments/memory_ranking/features.py`` documents the gap.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RATINGS_PATH = REPO / "data/gold/amazon_reviews_electronics_v1/electronics_catalog_ratings.csv"
SIDECAR_PATH = REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"
OUT_DIR = REPO / "data/memory_ranking_v1"

HALF_LIFE_DAYS = 180.0
MS_PER_DAY = 86_400_000.0

# Canonical attribute vocabulary.  The names intentionally mirror the AgentRec-X
# catalogue attribute vocabulary used by the preference scorer where possible.
ATTRS: tuple[str, ...] = (
    "over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth",
    "microphone", "rechargeable", "noise_cancelling", "waterproof", "foldable",
    "portable", "gaming", "mechanical", "silent", "ergonomic", "touch", "rgb",
    "led", "usb", "hdmi", "4k", "remote",
)

ATTR_PATTERNS: dict[str, re.Pattern[str]] = {
    "over_ear": re.compile(r"(?<![a-z])over[ -]?ear(?![a-z])", re.I),
    "on_ear": re.compile(r"(?<![a-z])on[ -]?ear(?![a-z])", re.I),
    "in_ear": re.compile(r"(?<![a-z])(?:in[ -]?ear|inear|earbuds?)(?![a-z])", re.I),
    "wireless": re.compile(r"(?<![a-z])(?:wireless|wi[ -]?fi)(?![a-z])", re.I),
    "wired": re.compile(r"(?<![a-z])(?:wired|corded|3\.5\s?mm)(?![a-z])", re.I),
    "bluetooth": re.compile(r"(?<![a-z])bluetooth(?![a-z])", re.I),
    "microphone": re.compile(r"(?<![a-z])(?:microphone|mic)(?![a-z])", re.I),
    "rechargeable": re.compile(r"(?<![a-z])(?:rechargeable|battery|batteries)(?![a-z])", re.I),
    "noise_cancelling": re.compile(r"noise[ -]?cancel", re.I),
    "waterproof": re.compile(r"waterproof|water[ -]?resist|(?<![a-z])ipx\d", re.I),
    "foldable": re.compile(r"foldable|folding", re.I),
    "portable": re.compile(r"(?<![a-z])(?:portable|compact)(?![a-z])", re.I),
    "gaming": re.compile(r"(?<![a-z])(?:gaming|gamer)(?![a-z])", re.I),
    "mechanical": re.compile(r"mechanical", re.I),
    "silent": re.compile(r"(?<![a-z])(?:silent|quiet)(?![a-z])", re.I),
    "ergonomic": re.compile(r"ergonomic", re.I),
    "touch": re.compile(r"(?<![a-z])touch(?![a-z])", re.I),
    "rgb": re.compile(r"(?<![a-z])rgb(?![a-z])", re.I),
    "led": re.compile(r"(?<![a-z])led(?![a-z])", re.I),
    "usb": re.compile(r"(?<![a-z])usb(?![a-z])", re.I),
    "hdmi": re.compile(r"(?<![a-z])hdmi(?![a-z])", re.I),
    "4k": re.compile(r"(?<![a-z])(?:4k|uhd)(?![a-z])", re.I),
    "remote": re.compile(r"(?<![a-z])remote(?![a-z])", re.I),
}

MAIN_CATEGORIES: tuple[str, ...] = (
    "Computers", "All Electronics", "Camera & Photo", "Cell Phones & Accessories",
    "Home Audio & Theater", "Industrial & Scientific", "Car Electronics",
    "Tools & Home Improvement", "Amazon Home", "Video Games", "Office Products",
)

LEAF_CATEGORIES: tuple[str, ...] = (
    "Computers & Accessories", "Camera & Photo", "Accessories",
    "Computer Accessories & Peripherals", "Bags, Cases & Sleeves",
    "Laptop Accessories", "Tablet Accessories", "Cases", "Television & Video",
    "Headphones, Earbuds & Accessories", "Computer Components",
    "Car & Vehicle Electronics", "Internal Components", "Home Audio",
    "Portable Audio & Video",
)

DETAIL_KEYS: tuple[str, ...] = (
    "Brand", "Manufacturer", "Color", "Connectivity Technology", "Form Factor",
    "Special Feature", "Material", "Item Weight", "Product Dimensions",
    "Compatible Devices", "Power Source", "Batteries", "Model Number",
    "Item model number", "Date First Available", "Included Components",
    "Cable Type", "Hardware Platform", "Operating System", "Screen Size",
)

LABEL_COLUMNS: tuple[str, ...] = ("target_rating", "target_high")
ID_COLUMNS: tuple[str, ...] = (
    "user_id", "target_item_id", "target_timestamp", "split", "history_length",
    "history_item_ids", "history_ratings", "history_timestamps",
    "target_title", "target_main_category", "target_leaf_category",
    "target_attributes", "target_attribute_count", "target_attribute_coverage",
    "target_price", "target_rating", "target_high", "label_source", "data_source",
)
HISTORY_COLUMNS: tuple[str, ...] = (
    "history_item_ids", "history_ratings", "history_timestamps",
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _text_for_metadata(item: Mapping[str, Any]) -> str:
    parts = [str(item.get("title") or "")]
    parts.extend(str(value) for value in (item.get("features") or ()))
    details = item.get("details") or {}
    parts.extend(f"{key} {value}" for key, value in details.items())
    parts.extend(str(value) for value in (item.get("categories") or ()))
    return " ".join(parts)


def extract_attributes(item: Mapping[str, Any]) -> tuple[str, ...]:
    text = _text_for_metadata(item)
    return tuple(name for name in ATTRS if ATTR_PATTERNS[name].search(text))


def empty_item_metadata() -> dict[str, Any]:
    return {
        "title": None,
        "main_category": None,
        "leaf_category": None,
        "categories": [],
        "price": None,
        "attributes": (),
        "attribute_count": 0,
        "attribute_coverage": 0.0,
        "feature_count": 0,
        "description_count": 0,
        "title_len": 0,
        "text_len_log": 0.0,
        "detail_count": 0,
        "has_price": 0.0,
        "price_log": float("nan"),
        "has_store": 0.0,
        "missing": 1.0,
    }


def normalise_metadata_item(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one metadata-sidecar record into the feature builder's item dict."""
    title = " ".join(str(raw.get("title") or "").split())
    features = [str(value) for value in (raw.get("features") or ())]
    descriptions = [str(value) for value in (raw.get("description") or ())]
    details = raw.get("details") or {}
    categories = [str(value) for value in (raw.get("categories") or ()) if value]
    main_category = raw.get("main_category")
    leaf_category = categories[-1] if categories else (main_category or None)
    price = _safe_float(raw.get("price"))
    text = _text_for_metadata(raw)
    attrs = extract_attributes(raw)
    return {
        "title": title or None,
        "main_category": str(main_category) if main_category else None,
        "leaf_category": str(leaf_category) if leaf_category else None,
        "categories": categories,
        "price": price,
        "attributes": attrs,
        "attribute_count": len(attrs),
        "attribute_coverage": round(len(attrs) / len(ATTRS), 6),
        "feature_count": len(features),
        "description_count": len(descriptions),
        "title_len": len(title),
        "text_len_log": round(math.log1p(len(text)), 6),
        "detail_count": len(details),
        "detail_keys": tuple(str(key) for key in details),
        "has_price": 1.0 if price is not None and price > 0 else 0.0,
        "price_log": round(math.log1p(price), 6)
        if price is not None and price > 0
        else float("nan"),
        "has_store": 1.0 if raw.get("store") else 0.0,
        "missing": 0.0,
    }


def load_item_metadata(path: Path = SIDECAR_PATH) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            item_id = str(raw.get("item_id") or "").strip()
            if not item_id:
                continue
            metadata[item_id] = normalise_metadata_item(raw)
    return metadata


def item_feature_names() -> list[str]:
    names = [
        "meta_missing", "meta_has_price", "meta_price_log", "meta_title_len",
        "meta_text_len_log", "meta_feature_count", "meta_desc_count",
        "meta_detail_count", "meta_has_store", "meta_attr_evidence_count",
        "meta_attr_coverage", "meta_main_category_missing", "meta_leaf_category_missing",
    ]
    names.extend(f"meta_main_cat_{_slug(category)}" for category in MAIN_CATEGORIES)
    names.extend(f"meta_leaf_cat_{_slug(category)}" for category in LEAF_CATEGORIES)
    names.extend(f"meta_detail_{_slug(key)}" for key in DETAIL_KEYS)
    names.extend(f"meta_attr_{attr}" for attr in ATTRS)
    return names


POP_FEATURES: tuple[str, ...] = (
    "pop_seen_train", "pop_count", "pop_count_log", "pop_user_count",
    "pop_user_count_log", "pop_mean_rating", "pop_std_rating", "pop_high_share",
    "pop_low_share", "pop_popularity_pct", "pop_age_days",
    "pop_main_cat_count_log", "pop_leaf_cat_count_log",
)


def memory_feature_names() -> list[str]:
    names = [
        "mem_hist_count", "mem_hist_count_log", "mem_hist_mean_rating",
        "mem_hist_std_rating", "mem_hist_min_rating", "mem_hist_max_rating",
        "mem_hist_high_count", "mem_hist_low_count", "mem_hist_neutral_count",
        "mem_hist_high_share", "mem_hist_low_share", "mem_hist_neutral_share",
        "mem_hist_rating_1_share", "mem_hist_rating_2_share", "mem_hist_rating_3_share",
        "mem_hist_rating_4_share", "mem_hist_rating_5_share", "mem_hist_last_rating",
        "mem_hist_last2_mean", "mem_hist_last3_mean", "mem_hist_last5_mean",
        "mem_hist_last_minus_prev_mean", "mem_hist_trend_slope",
        "mem_hist_days_since_last", "mem_hist_tenure_days", "mem_hist_avg_gap_days",
        "mem_hist_last_gap_days", "mem_hist_decay_weight_sum",
        "mem_hist_decay_mean_rating", "mem_hist_decay_high_share",
        "mem_hist_decay_low_share", "mem_hist_attr_coverage_mean",
        "mem_hist_high_attr_coverage_mean", "mem_hist_low_attr_coverage_mean",
        "mem_hist_price_mean_log", "mem_hist_high_price_mean_log",
        "mem_hist_low_price_mean_log", "mem_candidate_high_match_count",
        "mem_candidate_low_conflict_count", "mem_candidate_high_match_rate",
        "mem_candidate_low_conflict_rate", "mem_candidate_high_attr_weight",
        "mem_candidate_low_attr_weight", "mem_candidate_attr_seen_ratio",
        "mem_candidate_attr_unknown_count", "mem_candidate_high_main_cat_share",
        "mem_candidate_low_main_cat_share", "mem_candidate_high_leaf_cat_share",
        "mem_candidate_low_leaf_cat_share",
    ]
    names.extend(f"mem_all_attr_count_{attr}" for attr in ATTRS)
    names.extend(f"mem_high_attr_count_{attr}" for attr in ATTRS)
    names.extend(f"mem_low_attr_count_{attr}" for attr in ATTRS)
    return names


def non_memory_feature_names() -> list[str]:
    return item_feature_names() + list(POP_FEATURES)


def item_feature_row(
    item_id: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": item_id,
        "title": metadata.get("title"),
        "main_category": metadata.get("main_category"),
        "leaf_category": metadata.get("leaf_category"),
        "categories": list(metadata.get("categories") or ()),
        "price": metadata.get("price"),
        "attributes": list(metadata.get("attributes") or ()),
        "attribute_count": int(metadata.get("attribute_count") or 0),
        "attribute_coverage": float(metadata.get("attribute_coverage") or 0.0),
    }
    row.update(
        {
            "meta_missing": float(metadata.get("missing") or 0.0),
            "meta_has_price": float(metadata.get("has_price") or 0.0),
            "meta_price_log": metadata.get("price_log", float("nan")),
            "meta_title_len": float(metadata.get("title_len") or 0),
            "meta_text_len_log": float(metadata.get("text_len_log") or 0.0),
            "meta_feature_count": float(metadata.get("feature_count") or 0),
            "meta_desc_count": float(metadata.get("description_count") or 0),
            "meta_detail_count": float(metadata.get("detail_count") or 0),
            "meta_has_store": float(metadata.get("has_store") or 0.0),
            "meta_attr_evidence_count": float(metadata.get("attribute_count") or 0),
            "meta_attr_coverage": float(metadata.get("attribute_coverage") or 0.0),
            "meta_main_category_missing": 0.0
            if metadata.get("main_category")
            else 1.0,
            "meta_leaf_category_missing": 0.0
            if metadata.get("leaf_category")
            else 1.0,
        }
    )
    main = metadata.get("main_category")
    leaf = metadata.get("leaf_category")
    for category in MAIN_CATEGORIES:
        row[f"meta_main_cat_{_slug(category)}"] = 1.0 if main == category else 0.0
    for category in LEAF_CATEGORIES:
        row[f"meta_leaf_cat_{_slug(category)}"] = 1.0 if leaf == category else 0.0
    details = set(metadata.get("detail_keys") or ())
    for key in DETAIL_KEYS:
        row[f"meta_detail_{_slug(key)}"] = 1.0 if key in details else 0.0
    attrs = set(metadata.get("attributes") or ())
    for attr in ATTRS:
        row[f"meta_attr_{attr}"] = 1.0 if attr in attrs else 0.0
    return row


def load_ratings() -> pd.DataFrame:
    frame = pd.read_csv(
        RATINGS_PATH,
        usecols=["user_id", "parent_asin", "rating", "timestamp", "source_split"],
    )
    frame = frame.rename(columns={"parent_asin": "item_id"})
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame[
        frame["rating"].between(1, 5, inclusive="both")
        & frame["timestamp"].notna()
        & frame["user_id"].ne("")
        & frame["item_id"].ne("")
    ].copy()
    frame["rating"] = frame["rating"].astype(float)
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="first")
    frame = frame.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    counts = frame.groupby("user_id")["item_id"].transform("size")
    frame["user_n_events"] = counts.astype(int)
    frame = frame[frame["user_n_events"] >= 3].copy()
    frame["target_index"] = frame.groupby("user_id").cumcount().astype(int)
    frame["from_end"] = (frame["user_n_events"] - frame["target_index"]).astype(int)
    frame["split"] = np.where(
        frame["from_end"] == 1, "test",
        np.where(frame["from_end"] == 2, "val", "train"),
    )
    return frame.reset_index(drop=True)


def popularity_tables(
    frame: pd.DataFrame, item_metadata: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    train = frame[frame["split"] == "train"].copy()
    global_mean = float(train["rating"].mean()) if len(train) else 3.0
    global_std = float(train["rating"].std(ddof=0)) if len(train) > 1 else 1.0
    stats = (
        train.groupby("item_id")
        .agg(
            count=("rating", "size"),
            rating_sum=("rating", "sum"),
            rating_sumsq=("rating", lambda values: float(np.square(values).sum())),
            high=("rating", lambda values: int((values >= 4).sum())),
            low=("rating", lambda values: int((values <= 2).sum())),
            users=("user_id", "nunique"),
            first_timestamp=("timestamp", "min"),
        )
        .reset_index()
    )
    stats["popularity_pct"] = stats["count"].rank(pct=True, method="average")
    item_stats = {row.item_id: row for row in stats.itertuples(index=False)}
    category_counts: Counter[tuple[str, str | None]] = Counter()
    for item_id in train["item_id"]:
        meta = item_metadata.get(item_id) or empty_item_metadata()
        category_counts[("main", meta.get("main_category"))] += 1
        category_counts[("leaf", meta.get("leaf_category"))] += 1
    return {
        "item_stats": item_stats,
        "global_mean": global_mean,
        "global_std": global_std,
        "train_count": len(train),
        "category_counts": category_counts,
    }


def popularity_feature_row(
    *,
    item_id: str,
    rating: float,
    split: str,
    target_timestamp: int,
    item_main_category: str | None,
    item_leaf_category: str | None,
    table: Mapping[str, Any],
    category_counts: Mapping[str, Any],
) -> dict[str, float]:
    stats = table["item_stats"].get(item_id)
    global_mean = float(table["global_mean"])
    global_std = float(table["global_std"])
    if stats is None:
        base = {
            "pop_seen_train": 0.0,
            "pop_count": 0.0,
            "pop_count_log": 0.0,
            "pop_user_count": 0.0,
            "pop_user_count_log": 0.0,
            "pop_mean_rating": global_mean,
            "pop_std_rating": global_std,
            "pop_high_share": 0.0,
            "pop_low_share": 0.0,
            "pop_popularity_pct": 0.0,
            "pop_age_days": 0.0,
            "pop_main_cat_count_log": float(
                math.log1p(category_counts.get(("main", item_main_category), 0))
            ),
            "pop_leaf_cat_count_log": float(
                math.log1p(category_counts.get(("leaf", item_leaf_category), 0))
            ),
        }
        return base

    # Training rows must not read their own rating/count.  The target is present in
    # the training split, so subtract exactly one observation for this row.
    subtract = 1 if split == "train" else 0
    count = max(0, int(stats.count) - subtract)
    users = max(0, int(stats.users) - subtract)
    high = max(0, int(stats.high) - (1 if split == "train" and rating >= 4 else 0))
    low = max(0, int(stats.low) - (1 if split == "train" and rating <= 2 else 0))
    total = float(stats.rating_sum) - (rating if subtract else 0.0)
    sumsq = float(stats.rating_sumsq) - (rating * rating if subtract else 0.0)
    if count > 0:
        mean = total / count
        variance = max(0.0, sumsq / count - mean * mean)
        std = math.sqrt(variance)
        high_share = high / count
        low_share = low / count
        age_days = max(0.0, (target_timestamp - float(stats.first_timestamp)) / MS_PER_DAY)
    else:
        mean = global_mean
        std = global_std
        high_share = 0.0
        low_share = 0.0
        age_days = 0.0
    return {
        "pop_seen_train": 1.0 if count > 0 else 0.0,
        "pop_count": float(count),
        "pop_count_log": float(math.log1p(count)),
        "pop_user_count": float(users),
        "pop_user_count_log": float(math.log1p(users)),
        "pop_mean_rating": float(mean),
        "pop_std_rating": float(std),
        "pop_high_share": float(high_share),
        "pop_low_share": float(low_share),
        "pop_popularity_pct": float(stats.popularity_pct),
        "pop_age_days": float(age_days),
        "pop_main_cat_count_log": float(
            math.log1p(max(0, category_counts.get(("main", item_main_category), 0) - subtract))
        ),
        "pop_leaf_cat_count_log": float(
            math.log1p(max(0, category_counts.get(("leaf", item_leaf_category), 0) - subtract))
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def memory_feature_row(
    *,
    history: Sequence[Mapping[str, Any]],
    target_attributes: Iterable[str],
    target_main_category: str | None,
    target_leaf_category: str | None,
    target_timestamp: int,
) -> dict[str, float]:
    values: dict[str, float] = {name: 0.0 for name in memory_feature_names()}
    # Defaults for quantities that are undefined with no history.
    for name in (
        "mem_hist_mean_rating", "mem_hist_std_rating", "mem_hist_min_rating",
        "mem_hist_max_rating", "mem_hist_last_rating", "mem_hist_last2_mean",
        "mem_hist_last3_mean", "mem_hist_last5_mean", "mem_hist_last_minus_prev_mean",
        "mem_hist_trend_slope", "mem_hist_days_since_last", "mem_hist_tenure_days",
        "mem_hist_avg_gap_days", "mem_hist_last_gap_days", "mem_hist_decay_mean_rating",
        "mem_hist_attr_coverage_mean", "mem_hist_high_attr_coverage_mean",
        "mem_hist_low_attr_coverage_mean", "mem_hist_price_mean_log",
        "mem_hist_high_price_mean_log", "mem_hist_low_price_mean_log",
    ):
        values[name] = float("nan")

    n = len(history)
    values["mem_hist_count"] = float(n)
    values["mem_hist_count_log"] = float(math.log1p(n))
    candidate_attrs = [str(attr) for attr in target_attributes]
    values["mem_candidate_attr_unknown_count"] = float(len(candidate_attrs))
    if n == 0:
        return values

    ratings = np.asarray([float(item["rating"]) for item in history], dtype=float)
    timestamps = np.asarray([int(item["timestamp"]) for item in history], dtype=np.int64)
    high_mask = ratings >= 4
    low_mask = ratings <= 2
    neutral_mask = ratings == 3
    high_count = int(high_mask.sum())
    low_count = int(low_mask.sum())
    neutral_count = int(neutral_mask.sum())
    values["mem_hist_mean_rating"] = float(ratings.mean())
    values["mem_hist_std_rating"] = float(ratings.std(ddof=0))
    values["mem_hist_min_rating"] = float(ratings.min())
    values["mem_hist_max_rating"] = float(ratings.max())
    values["mem_hist_high_count"] = float(high_count)
    values["mem_hist_low_count"] = float(low_count)
    values["mem_hist_neutral_count"] = float(neutral_count)
    values["mem_hist_high_share"] = high_count / n
    values["mem_hist_low_share"] = low_count / n
    values["mem_hist_neutral_share"] = neutral_count / n
    for rating in (1, 2, 3, 4, 5):
        values[f"mem_hist_rating_{rating}_share"] = float((ratings == rating).mean())
    values["mem_hist_last_rating"] = float(ratings[-1])
    values["mem_hist_last2_mean"] = float(ratings[-2:].mean())
    values["mem_hist_last3_mean"] = float(ratings[-3:].mean())
    values["mem_hist_last5_mean"] = float(ratings[-5:].mean())
    if n >= 2:
        values["mem_hist_last_minus_prev_mean"] = float(
            ratings[-1] - ratings[:-1].mean()
        )
        values["mem_hist_last_gap_days"] = float(
            (timestamps[-1] - timestamps[-2]) / MS_PER_DAY
        )
        values["mem_hist_avg_gap_days"] = float(
            (timestamps[-1] - timestamps[0]) / MS_PER_DAY / (n - 1)
        )
    else:
        values["mem_hist_last_minus_prev_mean"] = 0.0
        values["mem_hist_last_gap_days"] = float("nan")
        values["mem_hist_avg_gap_days"] = float("nan")
    values["mem_hist_days_since_last"] = float(
        (target_timestamp - timestamps[-1]) / MS_PER_DAY
    )
    values["mem_hist_tenure_days"] = float(
        (target_timestamp - timestamps[0]) / MS_PER_DAY
    )
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    denominator = float((x_centered * x_centered).sum())
    if denominator > 0:
        values["mem_hist_trend_slope"] = float(
            (x_centered * (ratings - ratings.mean())).sum() / denominator
        )
    else:
        values["mem_hist_trend_slope"] = 0.0

    days = (target_timestamp - timestamps) / MS_PER_DAY
    decay = np.power(0.5, np.maximum(days, 0.0) / HALF_LIFE_DAYS)
    decay_sum = float(decay.sum())
    values["mem_hist_decay_weight_sum"] = decay_sum
    if decay_sum > 0:
        values["mem_hist_decay_mean_rating"] = float((decay * ratings).sum() / decay_sum)
        values["mem_hist_decay_high_share"] = float(
            (decay * high_mask.astype(float)).sum() / decay_sum
        )
        values["mem_hist_decay_low_share"] = float(
            (decay * low_mask.astype(float)).sum() / decay_sum
        )

    coverages = np.asarray(
        [float(item.get("attribute_coverage") or 0.0) for item in history], dtype=float
    )
    values["mem_hist_attr_coverage_mean"] = float(coverages.mean())
    if high_count:
        values["mem_hist_high_attr_coverage_mean"] = float(coverages[high_mask].mean())
    if low_count:
        values["mem_hist_low_attr_coverage_mean"] = float(coverages[low_mask].mean())

    history_prices = np.asarray(
        [float(item.get("price_log") or np.nan) for item in history], dtype=float
    )
    price_mask = np.isfinite(history_prices)
    if price_mask.any():
        values["mem_hist_price_mean_log"] = float(history_prices[price_mask].mean())
        high_price_mask = price_mask & high_mask
        low_price_mask = price_mask & low_mask
        if high_price_mask.any():
            values["mem_hist_high_price_mean_log"] = float(
                history_prices[high_price_mask].mean()
            )
        if low_price_mask.any():
            values["mem_hist_low_price_mean_log"] = float(
                history_prices[low_price_mask].mean()
            )

    attr_counts: dict[str, int] = {attr: 0 for attr in ATTRS}
    high_attr_counts: dict[str, int] = {attr: 0 for attr in ATTRS}
    low_attr_counts: dict[str, int] = {attr: 0 for attr in ATTRS}
    for item, rating in zip(history, ratings):
        item_attrs = item.get("attributes") or ()
        for attr in item_attrs:
            if attr not in attr_counts:
                continue
            attr_counts[attr] += 1
            if rating >= 4:
                high_attr_counts[attr] += 1
            elif rating <= 2:
                low_attr_counts[attr] += 1
    for attr in ATTRS:
        values[f"mem_all_attr_count_{attr}"] = float(attr_counts[attr])
        values[f"mem_high_attr_count_{attr}"] = float(high_attr_counts[attr])
        values[f"mem_low_attr_count_{attr}"] = float(low_attr_counts[attr])

    seen_attrs = [attr for attr in candidate_attrs if attr_counts.get(attr, 0) > 0]
    high_match = [attr for attr in candidate_attrs if high_attr_counts.get(attr, 0) > 0]
    low_match = [attr for attr in candidate_attrs if low_attr_counts.get(attr, 0) > 0]
    values["mem_candidate_high_match_count"] = float(len(high_match))
    values["mem_candidate_low_conflict_count"] = float(len(low_match))
    values["mem_candidate_attr_seen_ratio"] = float(len(seen_attrs) / len(candidate_attrs)) if candidate_attrs else 0.0
    values["mem_candidate_high_match_rate"] = float(len(high_match) / len(candidate_attrs)) if candidate_attrs else 0.0
    values["mem_candidate_low_conflict_rate"] = float(len(low_match) / len(candidate_attrs)) if candidate_attrs else 0.0
    values["mem_candidate_attr_unknown_count"] = float(
        len(candidate_attrs) - len(seen_attrs)
    )
    if high_count:
        values["mem_candidate_high_attr_weight"] = float(
            sum(high_attr_counts.get(attr, 0) for attr in candidate_attrs) / high_count
        )
    if low_count:
        values["mem_candidate_low_attr_weight"] = float(
            sum(low_attr_counts.get(attr, 0) for attr in candidate_attrs) / low_count
        )

    def _category_share(mask: np.ndarray, field: str, wanted: str | None) -> float:
        if not wanted or not mask.any():
            return 0.0
        matched = sum(
            1 for item, selected in zip(history, mask)
            if selected and item.get(field) == wanted
        )
        return float(matched / int(mask.sum()))

    values["mem_candidate_high_main_cat_share"] = _category_share(
        high_mask, "main_category", target_main_category
    )
    values["mem_candidate_low_main_cat_share"] = _category_share(
        low_mask, "main_category", target_main_category
    )
    values["mem_candidate_high_leaf_cat_share"] = _category_share(
        high_mask, "leaf_category", target_leaf_category
    )
    values["mem_candidate_low_leaf_cat_share"] = _category_share(
        low_mask, "leaf_category", target_leaf_category
    )
    return values


def add_memory_features(
    row: dict[str, Any], memory_values: Mapping[str, float]
) -> None:
    row.update({name: memory_values[name] for name in memory_feature_names()})


def build_item_feature_frame(
    rating_items: Iterable[str],
    metadata: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for item_id in sorted(set(rating_items)):
        meta = metadata.get(item_id) or empty_item_metadata()
        rows.append(item_feature_row(item_id, meta))
    return pd.DataFrame(rows)


def split_rule_text() -> str:
    return (
        "per-user chronological: the last catalog rating is test, the second-last is "
        "validation, and every earlier rating is a training target (the first target "
        "has an empty history); history contains only strictly earlier timestamps"
    )


def label_policy_text() -> str:
    return (
        "target_rating is the observed 1-5 Amazon rating; target_high is 1 for rating>=4 "
        "and 0 for rating<=2; rating==3 is kept for regression but excluded from binary "
        "classification.  No unobserved item is ever a negative."
    )


def leakage_controls() -> list[str]:
    return [
        "history is strictly earlier than the target timestamp; the target item is never in its own history",
        "train/val/test are assigned by per-user chronological order, one target event per split",
        "item/category popularity statistics are computed from the training split only",
        "for a training row, the target's own rating/count is removed from item popularity (leave-one-out)",
        "metadata average_rating and rating_number are excluded because they aggregate reviews that may include the target",
        "memory features are computed from ratings that occur strictly before the target timestamp",
        "no user_id, target rating, target item identity list, or history future event is used as a feature",
    ]
