"""Data-integrity tests for the expanded neural-memory dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data/neural_memory_v1"


@pytest.fixture(scope="module")
def data():
    result = {}
    for split in ("train", "val", "test"):
        result[split] = pd.read_parquet(DATA / f"{split}.parquet")
        result[f"memory_{split}"] = np.load(DATA / f"memory_{split}.npy")
    result["item_features"] = np.load(DATA / "item_features.npy")
    result["item_names"] = json.loads((DATA / "item_feature_names.json").read_text())
    result["memory_names"] = json.loads((DATA / "memory_feature_names.json").read_text())
    result["train_negatives"] = np.load(DATA / "train_negatives.npy")
    result["train_negatives_hard"] = np.load(DATA / "train_negatives_hard.npy")
    result["val_candidates"] = np.load(DATA / "val_candidates.npy")
    result["test_candidates"] = np.load(DATA / "test_candidates.npy")
    result["val_positive"] = np.load(DATA / "val_positive_position.npy")
    result["test_positive"] = np.load(DATA / "test_positive_position.npy")
    result["pairs"] = pd.read_parquet(DATA / "pairs.parquet")
    return result


def test_split_counts_and_no_history_leak(data):
    assert len(data["train"]) == 43435
    assert len(data["val"]) == 14157
    assert len(data["test"]) == 77029
    assert data["train"]["user_id"].nunique() == 36345
    assert data["val"]["user_id"].is_unique
    assert data["test"]["user_id"].is_unique
    for split in ("train", "val", "test"):
        frame = data[split]
        assert (frame["history_length"] > 0).all()
        assert frame["history_item_index"].map(len).max() <= 64
        assert (frame["history_item_index"].map(len) == frame["history_length"].clip(upper=64)).all()
        assert frame["target_item_index"].between(0, len(data["item_features"]) - 2).all()
        for row in frame[["target_item_index", "catalog_history_index", "history_item_index"]].itertuples(index=False):
            assert int(row.target_item_index) not in set(int(v) for v in row.catalog_history_index)
            assert int(row.target_item_index) not in set(int(v) for v in row.history_item_index)


def test_memory_and_item_feature_shapes(data):
    assert data["item_features"].shape[1] == len(data["item_names"])
    assert data["item_features"].shape[0] == 76919  # 76,918 catalog + unknown row
    for split in ("train", "val", "test"):
        assert data[f"memory_{split}"].shape[1] == len(data["memory_names"])
        assert len(data[f"memory_{split}"]) == len(data[split])
        assert np.isfinite(data[f"memory_{split}"]).all()


def test_candidate_arrays_have_real_positive_and_no_self(data):
    for split, candidates, positive in (
        ("val", data["val_candidates"], data["val_positive"]),
        ("test", data["test_candidates"], data["test_positive"]),
    ):
        frame = data[split].reset_index(drop=True)
        for index in range(min(2000, len(frame))):
            pos = int(positive[index])
            positive_item = int(candidates[index, pos])
            assert positive_item == int(frame.loc[index, "target_item_index"])
            history = set(int(v) for v in frame.loc[index, "catalog_history_index"])
            # negatives are sampled unobserved candidates excluding catalog history
            for candidate in candidates[index]:
                assert int(candidate) not in history
    for index in range(min(2000, len(data["train"]))):
        positive = int(data["train"].loc[index, "target_item_index"])
        assert positive not in set(int(v) for v in data["train_negatives"][index])
        assert positive not in set(int(v) for v in data["train_negatives_hard"][index])


def test_pair_groups_share_history_and_are_real_ratings(data):
    pairs = data["pairs"]
    assert len(pairs) == 28314
    assert pairs["pair_group_id"].nunique() == 14157
    assert pairs.groupby("pair_group_id").size().eq(2).all()
    assert pairs["val_row_index"].between(0, len(data["val"]) - 1).all()
    assert pairs["candidate_rating"].between(1, 5).all()


def test_manifest_has_protocol_and_labels():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["protocol"]["targets"] == "catalog items only"
    assert "unobserved" in manifest["protocol"]["unobserved_policy"]
    assert manifest["metadata"]["non_catalog_history_metadata"].startswith("unavailable")
    assert manifest["train"]["rows"] == 43435
    assert manifest["test"]["rows"] == 77029
