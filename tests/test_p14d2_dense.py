"""P14-D2 tests: identity-aligned BGE index, retrieval comparison, fixed-pool history ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p14d2"
DATA = REPO / "data/p14d1"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_bge_index_identity_is_exactly_the_electronics_catalogue():
    catalogue = ElectronicsCatalogue.load()
    index = np.load(RUNS / "bge_index.npz", allow_pickle=False)
    ids = [str(value) for value in index["identities"]]
    vectors = index["vectors"]
    assert len(ids) == 76263
    assert len(set(ids)) == 76263
    assert set(ids) == {product.item_id for product in catalogue._products.values()}
    assert vectors.shape == (76263, 768)
    assert np.allclose(np.linalg.norm(vectors[:10], axis=1), 1.0, atol=1e-4)
    manifest = json.loads((RUNS / "dense_index_manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity_check"].startswith("identities exactly")
    assert manifest["old_index_difference"]["old_identity_intersection"] == 0
    assert manifest["old_index_difference"]["old_index_reused"] is False


def test_retrieval_comparison_and_target_filter_loss():
    result = json.loads((RUNS / "natural_retrieval.json").read_text(encoding="utf-8"))
    variants = result["variants"]
    assert variants["original"]["bm25"]["recall_at20"] == 10
    assert variants["original"]["bm25"]["recall_at100"] == 20
    assert variants["original"]["bge"]["recall_at20"] == 27
    assert variants["original"]["bge"]["recall_at100"] == 66
    assert variants["original"]["hybrid_rrf"]["recall_at20"] == 22
    assert variants["original"]["hybrid_rrf"]["recall_at100"] == 47
    assert variants["rewritten"]["bm25"]["recall_at20"] == 18
    assert variants["rewritten"]["bge"]["recall_at20"] == 20
    assert variants["rewritten"]["hybrid_rrf"]["recall_at20"] == 17
    assert result["selected_fixed_pool"]["method"] == "bge"
    assert result["selected_fixed_pool"]["query_variant"] == "original"
    assert result["target_filter_loss"]["target_not_in_catalogue"] == 1
    assert result["target_filter_loss"]["target_would_be_filtered_by_role"] == 6


def test_original_and_rewritten_queries_not_mixed():
    result = json.loads((RUNS / "natural_retrieval.json").read_text(encoding="utf-8"))
    assert set(result["variants"]) == {"original", "rewritten"}
    comparisons = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparisons["dense_retrieval"]["original"]["bge"]["recall_at20"] == 27
    assert comparisons["dense_retrieval"]["rewritten"]["bge"]["recall_at20"] == 20


def test_fixed_pool_is_shared_and_target_not_marked():
    pools = _jsonl(RUNS / "pools/improved_natural.jsonl")
    assert len(pools) == 135
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["history_pool_target_reach"] == 27
    for task in comparison["per_task"]:
        pool = next(row for row in pools if row["task_id"] == task["task_id"])
        candidates = {candidate["item_id"] for candidate in pool["candidates"]}
        for method in ("qwen3_query_only", "qwen3_history_recovered", "qwen3_history_shuffled"):
            assert task["methods"][method]["target_rank"] is None or task["methods"][method]["target_rank"] <= 20
        assert pool["retrieval_method"] == "bge"
        assert pool["retrieval_query"] == "original"
        assert candidates == {candidate["item_id"] for candidate in pool["candidates"]}


def test_history_ablation_and_decision():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    methods = comparison["history_ablation_on_bge_pool"]["methods"]
    assert methods["qwen3_query_only"]["all"]["hit_at_1"] == 13
    assert methods["qwen3_history_recovered"]["all"]["hit_at_1"] == 15
    assert methods["qwen3_history_shuffled"]["all"]["hit_at_1"] == 16
    assert methods["qwen3_history_recovered"]["all"]["hit_at_1"] <= methods["qwen3_history_shuffled"]["all"]["hit_at_1"]
    decision = json.loads((RUNS / "decision.json").read_text(encoding="utf-8"))
    assert decision["DENSE_RETRIEVAL_REPAIR"] == "PASS"
    assert decision["HYBRID_RETRIEVAL"] == "FAIL"
    assert decision["HISTORY_INCREMENT"] == "FAIL"
    assert decision["CONTINUE_REAL_HISTORY_RANKER"] is False


def test_no_training_and_old_results_preserved():
    assert (REPO / "runs/p14d1r/comparison.json").is_file()
    assert (REPO / "runs/p14d1/comparison.json").is_file()
    assert not list(RUNS.rglob("*.pt")) and not list(RUNS.rglob("*.bin"))
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8", errors="ignore") for task_id in reserved), path
