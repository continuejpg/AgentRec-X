from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/public_preference_ranking_v1"


def test_esci_manifest_counts_and_splits():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert counts["queries_used"] == 630
    assert counts["train_queries"] + counts["val_queries"] + counts["test_queries"] == 630
    assert counts["candidate_rows"] == 11352
    assert counts["train_rows"] + counts["val_rows"] + counts["test_rows"] == 11352
    assert manifest["preference_source"] == "query_derived_proxy"
    assert manifest["not_historical_memory"] is True


def test_esci_no_benchmark_v2_test_queries():
    frame = pd.read_parquet(DATA / "esci_preference_dataset.parquet")
    benchmark = [json.loads(line) for line in (REPO / "data/memory_ranking_benchmark_v2/base_queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    benchmark_ids = {str(row["query_id"]) for row in benchmark}
    test = frame[frame["split"] == "test"]
    assert not (set(test["query_id"].astype(str)) & benchmark_ids)


def test_feature_views_do_not_leak_text_or_preference():
    schema = json.loads((DATA / "dcn_feature_schema.json").read_text(encoding="utf-8"))
    assert "base_query" not in schema["views"]["A_base_query_only"]
    assert "base_query" not in schema["views"]["B_preference_on"]
    assert not any(name.startswith("pref_") or name.startswith("preference_") for name in schema["views"]["A_base_query_only"])
    assert not any(name.startswith("pref_") or name.startswith("preference_") for name in schema["views"]["C_full_query"])
    assert any(name.startswith("pref_") for name in schema["views"]["B_preference_on"])


def test_dcn_labels_are_official_esci_only():
    frame = pd.read_parquet(DATA / "dcn_features.parquet")
    assert set(frame["esci_label"]) <= {"E", "S", "C", "I"}
    assert frame["group_id"].nunique() == 630
    assert set(frame["split"]) == {"train", "val", "test"}
    # Query-family split: no query_id may appear in more than one split.
    assert (frame.groupby("query_id")["split"].nunique() == 1).all()
