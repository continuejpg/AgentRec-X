"""P14-B3-Eval100 tests: new task isolation, fixed pools, real Qwen3 scores, V0 decision."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14b3_eval100"
RUNS = REPO / "runs/p14b3_eval100"


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_new_tasks_are_isolated_and_validated():
    agent = _jsonl(DATA / "eval100.agent.jsonl")
    evaluator = _jsonl(DATA / "eval100.evaluator.jsonl")
    validation = json.loads((RUNS / "validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True and validation["tasks"] == 100
    assert len(agent) == len(evaluator) == 100
    assert validation["expected_decisions"]["recommend"] == 80
    assert validation["overlap_with_prior_tasks"] == 0
    assert validation["overlap_with_prior_personas"] == 0
    assert validation["overlap_with_prior_queries"] == 0
    assert validation["human_verified"] is False
    assert all(row["split"] == "eval100" for row in agent)
    assert all(row["construction_metadata"]["review_status"] == "pending_human_review" for row in agent)
    assert not ({"expected_decision", "labels", "effective_preferences", "hard_constraints"} &
                set(agent[0]))


def test_pools_are_fixed_role_filtered_and_shared():
    pools = _jsonl(RUNS / "pools/eval100.jsonl")
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert len(pools) == 100
    pool_by_id = {row["task_id"]: row for row in pools}
    assert comparison["pool"]["recommend_tasks"] == 80
    assert comparison["pool"]["non_recommend_tasks"] == 20
    for task in comparison["per_task"]:
        source = pool_by_id[task["task_id"]]
        expected = {candidate["item_id"] for candidate in source["candidates"]}
        for method, data in task["methods"].items():
            assert set(data["ranked_ids"]) == expected
        for candidate in source["candidates"]:
            assert candidate["role_status"] in {"MATCH", "NOT_REQUIRED"}
            if source["target"]["product_type"]:
                assert candidate["product_type"] == source["target"]["product_type"]
    for method in comparison["summary"]:
        assert comparison["summary"][method]["product_type_violations"] == 0


def test_qwen3_scores_cover_all_candidates_and_variants():
    pools = _jsonl(RUNS / "pools/eval100.jsonl")
    scores = json.loads((RUNS / "scores_eval100.json").read_text(encoding="utf-8"))
    assert len(pools) == 100
    assert set(scores["scores"]) == {"query_only", "query_prefs", "prefs_instruction"}
    for pool in pools:
        expected = {candidate["item_id"] for candidate in pool["candidates"]}
        for variant in scores["scores"].values():
            assert set(variant[pool["task_id"]]) == expected
    assert scores["resources"]["pairs_scored"] == 3 * sum(len(pool["candidates"]) for pool in pools)
    assert scores["model"]["repo_id"] == "Qwen/Qwen3-Reranker-4B"
    assert scores["model"]["revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


def test_reachable_metrics_show_v0_better_than_qwen3():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["reachable_tasks"] == 73
    assert comparison["summary_all_80"]["v0"]["top1_valid"] == 67
    v0 = comparison["summary"]["v0"]
    qwen = comparison["summary"]["qwen3_prefs_instruction"]
    assert v0["top1_valid"] == 67 and qwen["top1_valid"] == 58
    assert v0["mrr_at_10"] > qwen["mrr_at_10"]
    assert len(comparison["qwen3_lost_vs_v0"]) > len(comparison["qwen3_gained_vs_v0"])
    assert v0["preference_violated_top1"] <= qwen["preference_violated_top1"]
    assert comparison["candidate_unreachable_tasks"] == [
        "p14b3_eval100_0037", "p14b3_eval100_0045", "p14b3_eval100_0049",
        "p14b3_eval100_0050", "p14b3_eval100_0061", "p14b3_eval100_0062",
        "p14b3_eval100_0069",
    ]


def test_candidate_unreachable_separate_and_no_history_tied():
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    by_type = comparison["category_breakdown"]
    assert by_type["no_history_recommend"]["v0_top1"] == by_type["no_history_recommend"]["qwen3_top1"] == 20
    assert comparison["summary"]["v0"]["candidate_unreachable"] if "candidate_unreachable" in comparison["summary"]["v0"] else True
    unreachable = set(comparison["candidate_unreachable_tasks"])
    reached = [task for task in comparison["per_task"] if task["task_id"] not in unreachable]
    assert len(reached) == 73


def test_nonrecommend_gate_is_evaluated_separately():
    gate = json.loads((RUNS / "nonrecommend_gate.json").read_text(encoding="utf-8"))
    assert gate["total"] == 20 and gate["correct"] == 19
    assert all(row["task_type"] in {"insufficient_information", "preference_conflict",
                                    "no_verified_match", "unverifiable_constraint"} for row in gate["tasks"])
    wrong = [row for row in gate["tasks"] if not row["correct"]]
    assert [row["task_id"] for row in wrong] == ["p14b3_eval100_0087"]
    comparison = json.loads((RUNS / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["nonrecommend_gate"]["total"] == 20


def test_preference_ablation_and_withdrawn_preference():
    pools = _jsonl(RUNS / "pools/eval100.jsonl")
    scores = json.loads((RUNS / "scores_eval100.json").read_text(encoding="utf-8"))
    history = [pool for pool in pools if pool["preference_summary"]]
    no_history = [pool for pool in pools if not pool["preference_summary"]]
    assert history and no_history
    changed = 0
    for pool in history:
        if not pool["candidates"]:
            continue
        q = scores["scores"]["query_only"][pool["task_id"]]
        p = scores["scores"]["query_prefs"][pool["task_id"]]
        i = scores["scores"]["prefs_instruction"][pool["task_id"]]
        if q != p:
            changed += 1
        assert p != i
    assert changed > 10
    for pool in no_history:
        assert scores["scores"]["query_only"][pool["task_id"]] == scores["scores"]["query_prefs"][pool["task_id"]]
    # Withdrawn preference must not be in active preference summaries.
    withdrawn = {str(item["value"]) for item in
                 _jsonl(DATA / "eval100.evaluator.jsonl")[0].get("withdrawn_preferences") or []}
    _ = withdrawn  # per-task check below is data-driven
    for evaluator in _jsonl(DATA / "eval100.evaluator.jsonl"):
        if not evaluator["withdrawn_preferences"]:
            continue
        pool = next(pool for pool in pools if pool["task_id"] == evaluator["task_id"])
        active = {str(item["value"]) for item in pool["active_preferences"]}
        assert not ({str(item["value"]) for item in evaluator["withdrawn_preferences"]} & active)


def test_review_queue_pending_and_old_artifacts_preserved():
    review = json.loads((RUNS / "review_checklist_divergence.json").read_text(encoding="utf-8"))
    assert 20 <= len(review["cases"]) <= 30
    assert review["human_verified"] is False
    assert all(case["review_status"] == "pending_human_review" for case in review["cases"])
    assert (REPO / "runs/p14b3/comparison.json").is_file()
    assert (REPO / "runs/p14b2/comparison.json").is_file()
    assert (REPO / "runs/p14c0r/comparison.json").is_file()
    reserved = {json.loads(line)["task_id"] for line in
                (REPO / "data/p14e2/benchmark/reserved_test.agent.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()}
    for path in RUNS.rglob("*.json*"):
        assert not any(task_id in path.read_text(encoding="utf-8") for task_id in reserved), path
