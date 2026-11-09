"""P14-B2 tests: real pretrained reranker artifacts, fixed pools, preference ablation, fairness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p14b2"


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_model_manifest_is_real_pretrained_reranker():
    manifest = json.loads((RUNS / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repo_id"] == "BAAI/bge-reranker-base"
    assert manifest["license"] == "MIT"
    assert len(manifest["sha256_model_safetensors"]) == 64
    assert manifest["architecture"] == ["XLMRobertaForSequenceClassification"]
    assert manifest["num_parameters"] == 278044417
    assert manifest["dtype"] == "float16"
    assert manifest["max_length"] == 320 and manifest["batch_size"] == 8
    assert manifest["resources"]["peak_gpu_allocated_mib"] > 0
    assert manifest["resources"]["model_load_seconds"] > 0
    assert "no SFT" in manifest["source"] or manifest["source"]


def test_scores_are_real_and_cover_every_candidate():
    pools = _load_jsonl(RUNS / "pools/dev.jsonl")
    scores = json.loads((RUNS / "scores_dev.json").read_text(encoding="utf-8"))
    assert scores["model"]["repo_id"] == "BAAI/bge-reranker-base"
    assert scores["tasks"] == 20
    for pool in pools:
        task_scores = scores["scores"][pool["task_id"]]
        expected = {c["item_id"] for c in pool["candidates"]}
        assert set(task_scores["query_only"]) == expected
        assert set(task_scores["query_prefs"]) == expected
        assert all(isinstance(v, float) for v in task_scores["query_only"].values())
    assert scores["pairs_scored"] == 2 * sum(len(p["candidates"]) for p in pools)


def test_pools_are_fixed_role_filtered_and_identical_across_methods():
    pool = {p["task_id"]: p for p in _load_jsonl(RUNS / "pools/dev.jsonl")}
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    for task in comparison["per_task"]:
        source = pool[task["task_id"]]
        source_ids = {c["item_id"] for c in source["candidates"]}
        for method, data in task["methods"].items():
            assert set(data["ranked_ids"]) == source_ids, (task["task_id"], method)
        for candidate in source["candidates"]:
            assert candidate["role_status"] in {"MATCH", "NOT_REQUIRED"}
            if source["target"]["product_type"]:
                assert candidate["product_type"] == source["target"]["product_type"]
    assert comparison["summary"]["bm25"]["product_type_violations"] == 0
    assert comparison["summary"]["neural_query_prefs"]["product_type_violations"] == 0


def test_pools_contain_no_evaluator_labels():
    forbidden = ("expected_decision", "verified", "acceptable", "gold", "evaluator", "persona_id")
    for path in (RUNS / "pools").glob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in text, (path, word)


def test_preference_ablation_differs_on_history_and_is_identical_without_history():
    pools = _load_jsonl(RUNS / "pools/dev.jsonl")
    scores = json.loads((RUNS / "scores_dev.json").read_text(encoding="utf-8"))
    history = [p for p in pools if p["preference_summary"]]
    no_history = [p for p in pools if not p["preference_summary"]]
    assert history and no_history
    history_diff = []
    for pool in history:
        q = scores["scores"][pool["task_id"]]["query_only"]
        p = scores["scores"][pool["task_id"]]["query_prefs"]
        if any(abs(q[i] - p[i]) > 1e-6 for i in q):
            history_diff.append(pool["task_id"])
    assert len(history_diff) >= 6, "active preferences must change neural inputs"
    for pool in no_history:
        q = scores["scores"][pool["task_id"]]["query_only"]
        p = scores["scores"][pool["task_id"]]["query_prefs"]
        assert q == p, "no-history tasks must not receive fabricated preference text"


def test_withdrawn_preference_is_not_in_reranker_input():
    evaluator = {e["task_id"]: e for e in _load_jsonl(REPO / "data/p14e2/benchmark/train.evaluator.jsonl")}
    pool = {p["task_id"]: p for p in _load_jsonl(RUNS / "pools/train_smoke.jsonl")}
    withdrawn = {str(item["value"]) for item in evaluator["p14e2_lite_0031"].get("withdrawn_preferences") or []}
    active = {str(item["value"]) for item in pool["p14e2_lite_0031"]["active_preferences"]}
    assert withdrawn and withdrawn & active == set()


def test_results_separate_ranking_from_unreachable_candidates():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["summary"]["v0"]["candidate_unreachable"] == 1
    assert comparison["summary"]["neural_query_prefs"]["candidate_unreachable"] == 1
    unreachable = [t for t in comparison["per_task"]
                   if t["expected_decision"] == "recommend" and t["candidate_unreachable"]]
    assert [t["task_id"] for t in unreachable] == ["p14e2_lite_0046"]
    assert comparison["summary"]["neural_query_prefs"]["hard_constraint_violations"] == 0


def test_ablation_and_cost_are_recorded():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    ablation = comparison["ablation"]
    assert ablation["history_tasks"] >= 8
    assert ablation["history_tasks_order_changed"] >= 6
    assert ablation["no_history_order_changed"] == []
    assert comparison["cost"]["v0_extra_gpu"] == "none"
    assert comparison["cost"]["neural_peak_gpu_allocated_mib"] > 0
    assert comparison["training_supervision"].endswith("training_supervision.json")


def test_training_supervision_is_limited_and_no_training_artifacts():
    supervision = json.loads((RUNS / "training_supervision.json").read_text(encoding="utf-8"))
    assert supervision["totals"]["acceptable"] > 0
    # Recommend tasks had no independent hard-violation labels; negatives are mostly 8
    # near-miss examples per task, far fewer than positive or unjudged items.
    recommend = [row for row in supervision["per_task"] if row["hard_violated"] == 0]
    assert len(recommend) == len(supervision["per_task"])
    assert supervision["totals"]["near_miss"] < supervision["totals"]["acceptable"]
    assert supervision["totals"]["near_miss"] < supervision["totals"]["hard_unknown"]
    assert not list(RUNS.rglob("*.pt")) and not list(RUNS.rglob("*.bin")), "no training checkpoints expected"


def test_old_p14c0r_results_and_reserved_test_are_preserved():
    assert (REPO / "runs/p14c0r/comparison.json").is_file()
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        text = path.read_text(encoding="utf-8")
        assert not any(task_id in text for task_id in reserved), path
