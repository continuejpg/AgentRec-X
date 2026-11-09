"""P14-B3 tests: real Qwen3-Reranker-4B variants, fixed pools, Setwise resource block."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs/p14b3"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_qwen3_reranker_manifest_is_real_and_hashed():
    manifest = json.loads((RUNS / "model_manifest.json").read_text(encoding="utf-8"))
    model = manifest["qwen3_reranker"]
    assert model["repo_id"] == "Qwen/Qwen3-Reranker-4B"
    assert model["license"] == "Apache-2.0"
    assert model["revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert len(model["sha256_weights"]) == 2
    assert all(len(value) == 64 for value in model["sha256_weights"].values())
    assert model["resources"]["pairs_scored"] == 333
    assert model["resources"]["device_actual"] == "cuda:0"
    assert model["resources"]["peak_gpu_allocated_mib"] > 0


def test_qwen3_scores_cover_three_variants_and_all_candidates():
    pools = _jsonl(RUNS / "pools/dev.jsonl")
    scores = json.loads((RUNS / "scores_qwen3reranker_dev.json").read_text(encoding="utf-8"))
    assert set(scores["scores"]) == {"query_only", "query_prefs", "prefs_instruction"}
    for pool in pools:
        expected = {candidate["item_id"] for candidate in pool["candidates"]}
        for variant in scores["scores"].values():
            assert set(variant[pool["task_id"]]) == expected
    assert scores["resources"]["pairs_scored"] == 3 * sum(len(p["candidates"]) for p in pools)


def test_same_fixed_pool_and_no_type_bypass():
    pools = _jsonl(RUNS / "pools/dev.jsonl")
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    for task in comparison["per_task"]:
        pool = next(pool for pool in pools if pool["task_id"] == task["task_id"])
        expected = {candidate["item_id"] for candidate in pool["candidates"]}
        for method, data in task["methods"].items():
            assert set(data["ranked_ids"]) == expected, (task["task_id"], method)
        for candidate in pool["candidates"]:
            assert candidate["role_status"] in {"MATCH", "NOT_REQUIRED"}
            if pool["target"]["product_type"]:
                assert candidate["product_type"] == pool["target"]["product_type"]
    for method, summary in comparison["summary"].items():
        assert summary["product_type_violations"] == 0
        assert summary["hard_constraint_violations"] == 0


def test_preference_inputs_and_ablation_are_real():
    pools = _jsonl(RUNS / "pools/dev.jsonl")
    scores = json.loads((RUNS / "scores_qwen3reranker_dev.json").read_text(encoding="utf-8"))
    no_history = [pool for pool in pools if not pool["preference_summary"]]
    history = [pool for pool in pools if pool["preference_summary"]]
    assert history and no_history
    for pool in no_history:
        q = scores["scores"]["query_only"][pool["task_id"]]
        p = scores["scores"]["query_prefs"][pool["task_id"]]
        assert q == p, "no-history tasks must not get artificial preference text"
    changed = 0
    for pool in history:
        q = scores["scores"]["query_only"][pool["task_id"]]
        p = scores["scores"]["query_prefs"][pool["task_id"]]
        i = scores["scores"]["prefs_instruction"][pool["task_id"]]
        if q != p:
            changed += 1
        assert p != i, "preference instruction must affect the reranker input"
    assert changed >= 8


def test_preference_instruction_gain_over_query_prefs():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    summary = comparison["summary"]
    assert summary["qwen3_prefs_instruction"]["top1_valid"] >= 12
    assert summary["qwen3_prefs_instruction"]["top1_valid"] > summary["qwen3_query_only"]["top1_valid"]
    gain = comparison["qwen3_ablations"]["qwen3_query_prefs->qwen3_prefs_instruction"]
    assert gain["top1_changed"] >= 3
    assert not gain["correct_lost"]


def test_withdrawn_preference_not_in_input():
    evaluator = {row["task_id"]: row for row in
                 _jsonl(REPO / "data/p14e2/benchmark/train.evaluator.jsonl")}
    pool = {row["task_id"]: row for row in _jsonl(RUNS / "pools/train_smoke.jsonl")}
    withdrawn = {str(item["value"]) for item in evaluator["p14e2_lite_0031"].get("withdrawn_preferences") or []}
    active = {str(item["value"]) for item in pool["p14e2_lite_0031"]["active_preferences"]}
    assert withdrawn and not (withdrawn & active)


def test_setwise_is_blocked_not_fabricated():
    status = json.loads((RUNS / "setwise_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "BLOCKED_RESOURCE"
    assert status["no_fabricated_result"] is True
    assert status["setwise_metrics_reported"] is None
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert "setwise" not in comparison["summary"]
    for task in comparison["per_task"]:
        assert "setwise" not in task["methods"]
    assert not (RUNS / "setwise_dev.json").is_file()


def test_candidate_unreachable_is_reported_separately():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["summary"]["v0"]["candidate_unreachable"] == 1
    unreachable = [task for task in comparison["per_task"]
                   if task["expected_decision"] == "recommend" and task["candidate_unreachable"]]
    assert [task["task_id"] for task in unreachable] == ["p14e2_lite_0046"]


def test_old_results_and_reserved_test_are_preserved():
    assert (REPO / "runs/p14b2/comparison.json").is_file()
    assert (REPO / "runs/p14c0r/comparison.json").is_file()
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        text = path.read_text(encoding="utf-8")
        assert not any(task_id in text for task_id in reserved), path
