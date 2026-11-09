"""P14-D1-R tests: metadata recovery, retrieval diagnosis, real vs shuffled history ablation."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14d1"
DATA_R = REPO / "data/p14d1r"
RUNS = REPO / "runs/p14d1r"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_history_metadata_recovery_is_real_and_improved():
    manifest = json.loads((DATA_R / "history_metadata_manifest.json").read_text(encoding="utf-8"))
    assert manifest["unique_history_items"] == 14655
    assert manifest["unique_items_with_metadata"] == 5576
    assert manifest["coverage_ratio"] == 0.3805
    assert manifest["tasks_with_any_metadata"] == 135
    assert manifest["before_metadata_coverage"]["history_items_in_catalogue"] == 126
    assert manifest["before_metadata_coverage"]["tasks_with_any_history_metadata"] == 50
    assert manifest["after_metadata_coverage"]["history_items_with_metadata"] == 5783
    assert manifest["official_source"]["recovered_unique_items"] == 5155
    assert manifest["c4_sampled_source"]["recovered_unique_items"] == 803
    assert set(manifest["official_source"]["category_counts"]) == {
        "Grocery_and_Gourmet_Food", "Health_and_Household", "Pet_Supplies",
        "Beauty_and_Personal_Care", "Electronics"}


def test_target_and_temporal_boundaries_are_preserved():
    tasks = _jsonl(DATA / "dev_tasks.jsonl")
    targets = {row["task_id"]: row for row in _jsonl(DATA / "dev_targets.jsonl")}
    for task in tasks:
        target = targets[task["task_id"]]
        assert all(item["item_id"] != target["target_parent_asin"] for item in task["history"])
        assert all(item["timestamp_ms"] < target["target_timestamp_ms"] for item in task["history"])
        assert [item["timestamp_ms"] for item in task["history"]] == sorted(
            item["timestamp_ms"] for item in task["history"])


def test_candidate_pools_are_identical_to_d1_and_target_not_marked():
    d1_natural = {row["task_id"]: row for row in _jsonl(REPO / "runs/p14d1/pools/natural.jsonl")}
    d1_target5 = {row["task_id"]: row for row in _jsonl(REPO / "runs/p14d1/pools/target5.jsonl")}
    natural = _jsonl(RUNS / "pools/natural.jsonl")
    target5 = _jsonl(RUNS / "pools/target5.jsonl")
    for row in natural:
        assert {c["item_id"] for c in row["candidates"]} == {
            c["item_id"] for c in d1_natural[row["task_id"]]["candidates"]}
        assert row["history_summary_real"] and row["history_summary_shuffled"]
        assert row["history_summary_real"] != row["history_summary_shuffled"]
    for row in target5:
        assert {c["item_id"] for c in row["candidates"]} == {
            c["item_id"] for c in d1_target5[row["task_id"]]["candidates"]}


def test_retrieval_diagnosis_and_bge_alignment():
    diagnosis = json.loads((RUNS / "retrieval_diagnosis.json").read_text(encoding="utf-8"))
    variants = diagnosis["bm25_query_variants"]
    assert variants["original"]["target_in_top20"] == 10
    assert variants["rewritten"]["target_in_top20"] == 18
    assert variants["original"]["target_in_top100"] == 20
    assert variants["rewritten"]["target_in_top100"] == 30
    assert diagnosis["bge"]["aligned"] is False
    assert diagnosis["bge"]["intersection_with_p13r_item_ids"] == 0
    improved = json.loads((RUNS / "retrieval_improved_manifest.json").read_text(encoding="utf-8"))
    assert improved["target_in_pool"] == 18
    assert improved["query_for_candidates"].startswith("released")


def test_history_ablation_shows_no_gain_and_shuffled_control():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    target = comparison["protocol_b_target5"]["methods"]
    natural = comparison["protocol_a_natural"]["methods"]
    assert target["qwen3_query_only"]["hit_at_1"] == 108
    assert target["qwen3_history_recovered"]["hit_at_1"] == 100
    assert target["qwen3_history_shuffled"]["hit_at_1"] == 102
    assert target["qwen3_history_recovered"]["hit_at_1"] <= target["qwen3_history_shuffled"]["hit_at_1"]
    assert natural["qwen3_history_recovered"]["all"]["hit_at_1"] == 7
    assert natural["qwen3_history_shuffled"]["all"]["hit_at_1"] == 8
    improved = comparison["protocol_a_rewritten_retrieval"]["methods"]
    assert improved["qwen3_query_only"]["all"]["hit_at_1"] == 9
    assert improved["qwen3_history_recovered"]["all"]["hit_at_1"] == 8
    assert improved["qwen3_history_shuffled"]["all"]["hit_at_1"] == 9
    decision = json.loads((RUNS / "decision.json").read_text(encoding="utf-8"))
    assert decision["HISTORY_ABLATION"] == "FAIL"
    assert decision["P14_D1_R"] == "LIMITED"
    assert decision["learned_real_history_model_worth"] is False


def test_v0_real_history_proxy_is_limited():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    target = comparison["protocol_b_target5"]["methods"]
    assert target["v0_proxy_recovered"]["hit_at_1"] == 18
    assert target["v0_proxy_recovered"]["hit_at_1"] < target["qwen3_query_only"]["hit_at_1"]
    manifest = json.loads((DATA_R / "history_metadata_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tasks_with_proxy_preferences"] == 126


def test_blind_review_pending_and_old_results_preserved():
    review = json.loads((RUNS / "blind_review.json").read_text(encoding="utf-8"))
    assert 20 <= len(review["cases"]) <= 30
    assert review["human_verified"] is False
    target_ids = {row["target_parent_asin"] for row in _jsonl(DATA / "dev_targets.jsonl")}
    for case in review["cases"]:
        text = json.dumps(case, ensure_ascii=False)
        assert not any(target_id in text for target_id in target_ids)
        assert all(set(candidate) <= {"label", "title", "evidence"} for candidate in case["candidates"])
        assert case["review_status"] == "pending_human_review"
    assert (REPO / "runs/p14d1/comparison.json").is_file()
    assert (REPO / "runs/p14b3_eval100/comparison.json").is_file()
    assert (REPO / "runs/p14c0r/comparison.json").is_file()
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8") for task_id in reserved), path
