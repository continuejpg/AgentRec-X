"""P14-D1 tests: real-data alignment, temporal isolation, fixed pools, real history comparison."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14d1"
RUNS = REPO / "runs/p14d1"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_real_data_alignment_and_query_mapping():
    tasks = _jsonl(DATA / "dev_tasks.jsonl")
    targets = _jsonl(DATA / "dev_targets.jsonl")
    alignment = json.loads((DATA / "alignment_manifest.json").read_text(encoding="utf-8"))
    assert len(tasks) == 135 and len(targets) == 135
    assert alignment["aligned_tasks"] == 135
    assert alignment["counts"]["target_in_catalogue"] == 134
    assert alignment["counts"]["target_missing_catalogue"] == 1
    assert alignment["counts"]["rewritten_available"] == 135
    assert all(task["query_original"] for task in tasks)
    assert all(task["history_length"] >= 0 for task in tasks)
    assert all(target["target_in_catalogue"] in (True, False) for target in targets)


def test_temporal_isolation_and_target_exclusion():
    tasks = {row["task_id"]: row for row in _jsonl(DATA / "dev_tasks.jsonl")}
    targets = {row["task_id"]: row for row in _jsonl(DATA / "dev_targets.jsonl")}
    for task_id, task in tasks.items():
        timestamps = [item["timestamp_ms"] for item in task["history"]]
        assert timestamps == sorted(timestamps)
        assert all(item["item_id"] != targets[task_id]["target_parent_asin"] for item in task["history"])
        assert all(item["timestamp_ms"] < targets[task_id]["target_timestamp_ms"] for item in task["history"])
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["temporal_validation"]["target_in_history"] == 0
    assert comparison["temporal_validation"]["history_at_or_after_target"] == 0


def test_fixed_pools_and_target_not_marked_in_model_input():
    natural = _jsonl(RUNS / "pools/natural.jsonl")
    target5 = _jsonl(RUNS / "pools/target5.jsonl")
    assert len(natural) == 135 and len(target5) == 134
    for pool in natural + target5:
        assert "target_parent_asin" not in json.dumps({k: v for k, v in pool.items() if k != "history_summary"})
        assert all("item_id" in candidate and "title" in candidate and "text" in candidate
                   for candidate in pool["candidates"])
        assert pool["query_original"]
        assert pool["history_summary"] is not None
    manifest = json.loads((RUNS / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_in_catalogue"] == 134
    assert manifest["target_in_bm25_top20"] == 10
    assert manifest["target_in_bm25_top100"] == 20
    assert manifest["history_items_in_catalogue"] == 126
    assert manifest["target5_tasks"] == 134


def test_real_qwen_scores_cover_pools_and_history_variants():
    natural = {row["task_id"]: row for row in _jsonl(RUNS / "pools/natural.jsonl")}
    target5 = {row["task_id"]: row for row in _jsonl(RUNS / "pools/target5.jsonl")}
    ns = json.loads((RUNS / "scores_natural.json").read_text(encoding="utf-8"))
    ts = json.loads((RUNS / "scores_target5.json").read_text(encoding="utf-8"))
    assert ns["model"]["repo_id"] == "Qwen/Qwen3-Reranker-4B"
    assert ns["model"]["revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert set(ns["scores"]) == {"query_only", "history", "query_only_rewritten"}
    assert set(ts["scores"]) == {"query_only", "history", "query_only_rewritten"}
    for pool in natural.values():
        for variant in ns["scores"].values():
            assert set(variant[pool["task_id"]]) == {c["item_id"] for c in pool["candidates"]}
    for pool in target5.values():
        for variant in ts["scores"].values():
            assert set(variant[pool["task_id"]]) == {c["item_id"] for c in pool["candidates"]}
    assert ns["resources"]["pairs_scored"] == 3 * sum(len(p["candidates"]) for p in natural.values())
    assert ts["resources"]["pairs_scored"] == 3 * sum(len(p["candidates"]) for p in target5.values())


def test_primary_result_and_history_increment_are_honest():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    natural = comparison["protocol_a_natural"]["methods"]
    target = comparison["protocol_b_target5"]["methods"]
    # Natural retrieval coverage is the main limitation, and is not hidden.
    assert comparison["target_in_bm25_top20"] == 10
    assert comparison["target_in_bm25_top100"] == 20
    assert natural["bm25"]["natural_all_135"]["hit_at_1"] == 2
    assert natural["qwen3_query_only"]["natural_all_135"]["hit_at_1"] == 6
    assert natural["qwen3_history"]["natural_all_135"]["hit_at_1"] == 8
    # Target-included diagnostic: history did not improve the same-model ranking.
    assert target["qwen3_query_only"]["hit_at_1"] == 108
    assert target["qwen3_history"]["hit_at_1"] == 101
    assert target["qwen3_history"]["mrr_at_10"] < target["qwen3_query_only"]["mrr_at_10"]
    increments = comparison["history_increment"]["target5"]
    assert increments["history_hit1"] <= increments["query_only_hit1"]


def test_v0_history_compatibility_is_weak_and_reported():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["history_metadata_coverage"]["tasks_with_proxy_preferences"] == 35
    assert comparison["history_metadata_coverage"]["history_items_in_catalogue"] == 126
    target = comparison["protocol_b_target5"]["methods"]
    assert target["v0_proxy"]["hit_at_1"] == 7
    assert target["v0_proxy"]["hit_at_1"] < target["bm25"]["hit_at_1"] + 5
    assert "weak history attribute proxy" in comparison["model_manifest"]["v0_proxy"]["name"]


def test_blind_review_has_no_target_or_model_ranking_and_is_pending():
    review = json.loads((RUNS / "blind_review.json").read_text(encoding="utf-8"))
    assert 20 <= len(review["cases"]) <= 30
    assert review["human_verified"] is False
    target_ids = {row["target_parent_asin"] for row in _jsonl(DATA / "dev_targets.jsonl")}
    for case in review["cases"]:
        text = json.dumps(case, ensure_ascii=False)
        assert not any(target_id in text for target_id in target_ids)
        assert "target" not in case
        assert "rank" not in {key.lower() for candidate in case["candidates"] for key in candidate}
        assert "score" not in {key.lower() for candidate in case["candidates"] for key in candidate}
        assert case["review_status"] == "pending_human_review"


def test_old_benchmarks_and_reserved_are_untouched():
    assert (REPO / "runs/p14b3_eval100/comparison.json").is_file()
    assert (REPO / "runs/p14b2/comparison.json").is_file()
    assert (REPO / "runs/p14c0r/comparison.json").is_file()
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8") for task_id in reserved), path
