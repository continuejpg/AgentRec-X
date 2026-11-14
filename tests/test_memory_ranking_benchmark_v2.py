from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data/memory_ranking_benchmark_v2"


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (BENCH / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_benchmark_v2_counts_and_labels():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert counts["base_queries"] == 100
    assert counts["scenarios"] == 300
    assert counts["pair_tasks"] == 300
    assert counts["human_gold"] == 0
    assert counts["silver"] + counts["mechanistic"] + counts["unjudged"] == 300
    assert counts["candidate_pool_rows"] == 2000


def test_benchmark_v2_annotation_hides_model_outputs():
    forbidden = ("bge_rank", "bge_score", "shopranker", "final_score", "label_provenance", "dcn")
    for row in _jsonl("annotation_tasks.jsonl"):
        text = json.dumps(row, ensure_ascii=False).lower()
        assert not any(token in text for token in forbidden), row["pair_task_id"]


def test_benchmark_v2_split_is_by_base_query():
    pairs = _jsonl("pair_tasks.jsonl")
    by_base = {}
    for row in pairs:
        by_base.setdefault(row["base_query_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_base.values())


def test_dcn_pairs_balanced_and_silver_only():
    frame = pd.read_parquet(BENCH / "dcn_v2_pairs" / "pairs.parquet")
    assert len(frame) == 290
    assert frame["group_id"].nunique() == 145
    assert set(frame["side"]) == {"A", "B"}
    assert (frame.groupby("group_id")["winner"].sum() == 1).all()
    assert set(frame["label_provenance"]) == {"SILVER"}
    manifest = json.loads((BENCH / "dcn_v2_pairs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_provenance"]["HUMAN_GOLD"] == 0
    assert manifest["split_groups"] == {"train": 97, "val": 34, "test": 14}
