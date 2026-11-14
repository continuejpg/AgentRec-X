"""Regression tests for the real Amazon Reviews Memory Ranking v1 dataset.

These tests use the generated local artifacts and the real ratings/metadata files.
They assert the temporal-validity invariants that matter most for this milestone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("lightgbm")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.memory_ranking import features  # noqa: E402

DATA = REPO_ROOT / "data/memory_ranking_v1"


@pytest.fixture(scope="module")
def splits():
    return {
        "train": pd.read_parquet(DATA / "train.parquet"),
        "val": pd.read_parquet(DATA / "val.parquet"),
        "test": pd.read_parquet(DATA / "test.parquet"),
    }


def test_history_is_strictly_before_target_and_never_contains_target(splits):
    for name, frame in splits.items():
        for row in frame[
            ["target_item_id", "target_timestamp", "history_item_ids", "history_timestamps", "history_length"]
        ].itertuples(index=False):
            ids = list(row.history_item_ids) if row.history_item_ids is not None else []
            times = list(row.history_timestamps) if row.history_timestamps is not None else []
            assert row.target_item_id not in ids, name
            assert len(ids) == int(row.history_length), name
            assert len(times) == int(row.history_length), name
            assert all(int(value) < int(row.target_timestamp) for value in times), name


def test_each_event_is_a_target_in_exactly_one_split(splits):
    all_targets = pd.concat(
        [
            frame[["user_id", "target_item_id", "target_timestamp"]].assign(split=name)
            for name, frame in splits.items()
        ]
    )
    assert len(all_targets) == 50741
    assert len(all_targets.drop_duplicates(["user_id", "target_item_id", "target_timestamp"])) == len(all_targets)
    assert splits["val"]["user_id"].is_unique
    assert splits["test"]["user_id"].is_unique
    assert set(splits["train"]["split"]) == {"train"}
    assert set(splits["val"]["split"]) == {"val"}
    assert set(splits["test"]["split"]) == {"test"}


def test_no_forbidden_column_is_a_model_feature(splits):
    forbidden = {
        "user_id", "target_item_id", "target_rating", "target_high",
        "history_item_ids", "history_ratings", "history_timestamps",
        "pair_group_id", "source_file_split",
    }
    non_memory = set(features.non_memory_feature_names())
    memory = set(features.memory_feature_names())
    assert not (non_memory & forbidden)
    assert not (memory & forbidden)
    assert non_memory.isdisjoint(memory)
    assert not any(name.startswith("mem_") for name in non_memory)
    assert all(name.startswith("mem_") for name in memory)
    assert set(features.non_memory_feature_names()) <= set(splits["train"].columns)
    assert set(features.memory_feature_names()) <= set(splits["train"].columns)


def test_labels_are_observed_ratings_and_unobserved_are_absent(splits):
    for name, frame in splits.items():
        assert frame["target_rating"].between(1, 5).all()
        assert frame["target_high"].dropna().isin([0.0, 1.0]).all()
        expected_high = np.where(
            frame["target_rating"] >= 4, 1.0,
            np.where(frame["target_rating"] <= 2, 0.0, np.nan),
        )
        assert np.allclose(
            frame["target_high"].to_numpy(dtype=float),
            expected_high,
            equal_nan=True,
        ), name
        assert (frame["label_source"] == "amazon_reviews_2023_electronics_observed_rating").all()


def test_ranking_pairs_share_one_history_cutoff():
    pairs = pd.read_parquet(DATA / "ranking_pairs.parquet")
    assert len(pairs) == 8436
    for group_id, group in pairs.groupby("pair_group_id"):
        assert len(group) == 2
        left_history = list(group.iloc[0]["history_item_ids"])
        right_history = list(group.iloc[1]["history_item_ids"])
        assert left_history == right_history
        history_ids = set(left_history)
        assert set(group["target_item_id"]).isdisjoint(history_ids)
        assert group["history_length"].nunique() == 1


def test_cold_start_rows_have_empty_history_and_missing_rating_memory(splits):
    train = splits["train"]
    cold = train[train["history_length"] == 0]
    assert len(cold) == 14157
    assert (cold["history_item_ids"].map(len) == 0).all()
    assert cold["mem_hist_count"].eq(0).all()
    assert cold["mem_hist_mean_rating"].isna().all()


def test_feature_schema_matches_data_columns():
    schema = json.loads((DATA / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["all_features_memory_on"] == (
        schema["non_memory_features"] + schema["memory_features"]
    )
    assert len(schema["non_memory_features"]) == 95
    assert len(schema["memory_features"]) == 118
    assert "average_rating" not in schema["all_features_memory_on"]
    assert "rating_number" not in schema["all_features_memory_on"]


def test_attribute_extraction_uses_real_metadata_fields():
    metadata = {
        "title": "Wireless Over-Ear Headphones with Microphone",
        "features": ["Rechargeable battery", "Noise cancelling"],
        "details": {"Form Factor": "Over Ear", "Color": "Black"},
        "categories": ["Electronics", "Headphones"],
    }
    attributes = set(features.extract_attributes(metadata))
    assert {"wireless", "over_ear", "microphone", "rechargeable", "noise_cancelling"} <= attributes
