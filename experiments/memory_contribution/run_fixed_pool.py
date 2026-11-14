"""Run the fixed-pool A / B / C Memory Contribution benchmark (local CPU).

Arms
----
A -- Relevance Only: Original-BGE rank inside the shared eligible set.
B -- Existing Scorer / Memory OFF: historical memory channel disabled, current-query
     conditions (hard requirements and any explicit query soft preference) retained.
C -- Existing Scorer / Memory ON: the legal active history preferences are added.

Every arm uses the same 20-candidate pool, the same catalogue metadata, the same
``prepare_hard`` requirements, the same structural eligibility filter, and the same
DecisionGate.  The only intended difference is the historical preference input.

Run from the repository root with the local CPU environment::

    .venv/bin/python -m experiments.memory_contribution.run_fixed_pool
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
from typing import Any, Mapping, Sequence

from experiments.memory_contribution import common

OUT = common.OUT_DIR


def _mean(values: Sequence[float]) -> float:
    return round(statistics.mean(values), 6) if values else 0.0


def _arm_metrics(
    tasks: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    values = [task["metrics"][arm] for task in tasks]
    return {
        "tasks": len(tasks),
        "hit@1": sum(1 for value in values if value["hit@1"]),
        "hit@3": sum(1 for value in values if value["hit@3"]),
        "hit@5": sum(1 for value in values if value["hit@5"]),
        "mrr@10": _mean([value["reciprocal_rank@10"] for value in values]),
        "tasks_with_acceptable_in_ranking": sum(
            1 for value in values if value["acceptable_in_ranking"]
        ),
        "mean_first_acceptable_rank": _mean(
            [
                value["first_acceptable_rank"]
                for value in values
                if value["first_acceptable_rank"] is not None
            ]
        ),
    }


def _gate_summary(tasks: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    actions: dict[str, int] = {}
    for task in tasks:
        action = str(task["gate"][arm]["action"])
        actions[action] = actions.get(action, 0) + 1
    selected = [task["gate"][arm].get("selected_candidate_id") for task in tasks]
    finalized_acceptable = sum(
        1
        for task in tasks
        if task["gate"][arm].get("action") == "ALLOW_FINALIZE"
        and task["gate"][arm].get("selected_candidate_id")
        in set(task["acceptable_ids_in_pool"])
    )
    return {
        "actions": dict(sorted(actions.items())),
        "selected_equals_rank1": sum(
            1
            for task in tasks
            if task["gate"][arm].get("selected_candidate_id") == task["top1"][arm]
        ),
        "finalized_acceptable": finalized_acceptable,
        "tasks_with_selection": sum(1 for value in selected if value),
    }


def _memory_effect(task: Mapping[str, Any]) -> dict[str, Any]:
    b_rank = {item["item_id"]: item["rank"] for item in task["candidates"]["B_scorer_memory_off"]}
    c_rank = {item["item_id"]: item["rank"] for item in task["candidates"]["C_scorer_memory_on"]}
    b_scores = {item["item_id"]: item["final_score"] for item in task["candidates"]["B_scorer_memory_off"]}
    c_scores = {item["item_id"]: item["final_score"] for item in task["candidates"]["C_scorer_memory_on"]}
    changed_scores = {
        item_id: {
            "score_B": b_scores[item_id],
            "score_C": c_scores[item_id],
            "delta": round(float(c_scores[item_id]) - float(b_scores[item_id]), 6),
            "rank_B": b_rank[item_id],
            "rank_C": c_rank[item_id],
        }
        for item_id in b_scores
        if abs(float(c_scores[item_id]) - float(b_scores[item_id])) > 1e-9
    }
    acceptable_ranks = {
        item_id: {
            "rank_B": b_rank.get(item_id),
            "rank_C": c_rank.get(item_id),
            "delta_B_minus_C": (
                b_rank[item_id] - c_rank[item_id] if item_id in b_rank and item_id in c_rank else None
            ),
        }
        for item_id in task["acceptable_ids_in_pool"]
    }
    best_b = task["metrics"]["B_scorer_memory_off"]["first_acceptable_rank"]
    best_c = task["metrics"]["C_scorer_memory_on"]["first_acceptable_rank"]
    if best_b is None or best_c is None:
        direction = "not_in_ranking"
    elif best_c < best_b:
        direction = "improved"
    elif best_c > best_b:
        direction = "regressed"
    else:
        direction = "unchanged"
    return {
        "scores_changed": bool(changed_scores),
        "ranking_changed": task["rankings"]["B_scorer_memory_off"]
        != task["rankings"]["C_scorer_memory_on"],
        "top1_changed": task["top1"]["B_scorer_memory_off"] != task["top1"]["C_scorer_memory_on"],
        "top3_set_changed": set(task["top3"]["B_scorer_memory_off"])
        != set(task["top3"]["C_scorer_memory_on"]),
        "changed_scores": changed_scores,
        "acceptable_ranks": acceptable_ranks,
        "first_acceptable_direction": direction,
        "candidate_changes": common.change_reasons(
            b_rows=task["candidates"]["B_scorer_memory_off"],
            c_rows=task["candidates"]["C_scorer_memory_on"],
        ),
    }


def _unchanged_reason(task: Mapping[str, Any]) -> str:
    if not task["active_memory_summary"]:
        return "no_active_historical_memory"
    effect = task["memory_effect"]
    if effect["scores_changed"] and not effect["ranking_changed"]:
        return "scores_changed_but_order_unchanged"
    matched = any(
        item["matched_preferences"] or item["conflicting_preferences"]
        for item in task["candidates"]["C_scorer_memory_on"]
    )
    if not matched:
        return "all_active_memory_unknown_in_pool"
    return "ranking_unchanged_despite_known_preference_effect"


def _build_task(row: Mapping[str, Any]) -> dict[str, Any]:
    query_prefs = common.query_soft_preferences(row["query"])
    history = common.history_preferences(row)
    off_prefs = common.combined_preferences(
        query_preferences=query_prefs, history_preferences=history, memory_enabled=False
    )
    on_prefs = common.combined_preferences(
        query_preferences=query_prefs, history_preferences=history, memory_enabled=True
    )
    eligible, excluded = common.eligibility(row)
    acceptable = list(row.get("acceptable_ids_in_pool") or [])
    acceptable_in_eligible = [item_id for item_id in acceptable if item_id in set(eligible)]
    pool_ids = [item["item_id"] for item in row["candidate_pool"]]

    scored_off = common.score_pool(row, off_prefs)
    scored_on = common.score_pool(row, on_prefs)
    rankings = {
        "A_relevance_only": [
            item_id
            for item_id in sorted(
                eligible,
                key=lambda value: next(
                    item["original_bge_rank"]
                    for item in row["candidate_pool"]
                    if item["item_id"] == value
                ),
            )
        ],
        "B_scorer_memory_off": common.ordered_ids(scored_off, eligible),
        "C_scorer_memory_on": common.ordered_ids(scored_on, eligible),
    }
    gated_rows = {"A_relevance_only": scored_off, "B_scorer_memory_off": scored_off,
                  "C_scorer_memory_on": scored_on}
    gate_prefs = {"A_relevance_only": off_prefs, "B_scorer_memory_off": off_prefs,
                  "C_scorer_memory_on": on_prefs}

    metrics: dict[str, Any] = {}
    hard: dict[str, Any] = {}
    scored_by_arm = {
        "A_relevance_only": scored_off,
        "B_scorer_memory_off": scored_off,
        "C_scorer_memory_on": scored_on,
    }
    for arm in common.ARMS:
        metrics[arm] = common.hit_metrics(rankings[arm], acceptable)
        by_id = {item["product_ref"]["item_id"]: item for item in scored_by_arm[arm]}
        hard[arm] = common.hard_violations(rankings[arm], by_id)

    candidates: dict[str, list[dict[str, Any]]] = {}
    for arm in common.ARMS:
        candidates[arm] = common.arm_candidate_rows(
            scored_rows=scored_by_arm[arm], order=rankings[arm]
        )
    candidates_by_id: list[dict[str, Any]] = []
    a_rank = {item["item_id"]: item["rank"] for item in candidates["A_relevance_only"]}
    b_rank = {item["item_id"]: item["rank"] for item in candidates["B_scorer_memory_off"]}
    c_rank = {item["item_id"]: item["rank"] for item in candidates["C_scorer_memory_on"]}
    b_by_id = {item["item_id"]: item for item in candidates["B_scorer_memory_off"]}
    c_by_id = {item["item_id"]: item for item in candidates["C_scorer_memory_on"]}
    for item in row["candidate_pool"]:
        item_id = item["item_id"]
        if item_id not in c_rank:
            continue
        b_row = b_by_id[item_id]
        c_row = c_by_id[item_id]
        candidates_by_id.append(
            {
                "item_id": item_id,
                "title": item.get("title"),
                "label": item.get("label", "unjudged"),
                "workflow_status": "eligible",
                "attributes": dict(item.get("attributes") or {}),
                "original_bge_rank": item["original_bge_rank"],
                "rank_A": a_rank.get(item_id),
                "rank_B": b_rank.get(item_id),
                "rank_C": c_rank.get(item_id),
                "hard_tier": c_row["hard_tier"],
                "hard_status": (item.get("scorer") or {}).get("hard_status"),
                "role_status": c_row["role_status"],
                "score_B": b_row["final_score"],
                "score_C": c_row["final_score"],
                "query_relevance_score": c_row["query_relevance_score"],
                "preference_match_score": c_row["preference_match_score"],
                "preference_conflict_score": c_row["preference_conflict_score"],
                "matched_preferences": c_row["matched_preferences"],
                "conflicting_preferences": c_row["conflicting_preferences"],
                "unknown_preferences": c_row["unknown_preferences"],
                "presentable": c_row["presentable"],
                "ranking_changed_B_vs_C": b_rank.get(item_id) != c_rank.get(item_id),
            }
        )

    task = {
        "schema": common.SCHEMA_TASK,
        "task_id": row["task_id"],
        "split": row.get("split"),
        "category": row.get("category"),
        "query": row["query"],
        "expected_decision": row.get("expected_decision"),
        "label_source": row.get("label_source"),
        "label_status": common.LABEL_STATUS,
        "unjudged_policy": "candidates not in the provisional acceptable set are unjudged, not negative",
        "memory_fixture": row.get("memory_fixture"),
        "candidate_pool_size": len(pool_ids),
        "candidate_ids": pool_ids,
        "eligible_candidate_ids": eligible,
        "excluded_candidates": excluded,
        "hard_requirements": row.get("hard_requirements") or [],
        "active_memory_summary": common.preference_summary(history),
        "query_soft_preferences": common.preference_summary(query_prefs),
        "memory_off_preferences": common.preference_summary(off_prefs),
        "memory_on_preferences": common.preference_summary(on_prefs),
        "acceptable_ids_in_pool": acceptable,
        "acceptable_ids_in_eligible": acceptable_in_eligible,
        "rankings": rankings,
        "top1": {arm: (rankings[arm][0] if rankings[arm] else None) for arm in common.ARMS},
        "top3": {arm: rankings[arm][:3] for arm in common.ARMS},
        "top5": {arm: rankings[arm][:5] for arm in common.ARMS},
        "metrics": metrics,
        "hard_constraint": {
            arm: {
                **hard[arm],
                "violations_from_eligible_ranking": True,
            }
            for arm in common.ARMS
        },
        "gate": {},
        "candidates": candidates,
        "candidate_table": candidates_by_id,
    }
    for arm in common.ARMS:
        task["gate"][arm] = common.gate_for_arm(
            row=row,
            scored_rows=gated_rows[arm],
            preferences=gate_prefs[arm],
            order=rankings[arm],
        )
    task["memory_effect"] = _memory_effect(task)
    task["memory_effect"]["unchanged_reason"] = (
        _unchanged_reason(task)
        if not task["memory_effect"]["ranking_changed"]
        else None
    )
    return task


def _aggregate(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top1_changed = [
        task["task_id"]
        for task in tasks
        if task["top1"]["B_scorer_memory_off"] != task["top1"]["C_scorer_memory_on"]
    ]
    top3_changed = [
        task["task_id"]
        for task in tasks
        if set(task["top3"]["B_scorer_memory_off"])
        != set(task["top3"]["C_scorer_memory_on"])
    ]
    ranking_changed = [
        task["task_id"]
        for task in tasks
        if task["rankings"]["B_scorer_memory_off"] != task["rankings"]["C_scorer_memory_on"]
    ]
    score_changed = [
        task["task_id"] for task in tasks if task["memory_effect"]["scores_changed"]
    ]
    active = [task for task in tasks if task["active_memory_summary"]]
    active_affected = [
        task for task in active
        if task["memory_effect"]["scores_changed"] or task["memory_effect"]["ranking_changed"]
    ]
    improved = [
        task["task_id"]
        for task in tasks
        if task["memory_effect"]["first_acceptable_direction"] == "improved"
    ]
    regressed = [
        task["task_id"]
        for task in tasks
        if task["memory_effect"]["first_acceptable_direction"] == "regressed"
    ]
    unchanged = [
        task["task_id"]
        for task in tasks
        if task["memory_effect"]["first_acceptable_direction"] == "unchanged"
    ]
    benefit = [
        task["task_id"]
        for task in tasks
        if task["metrics"]["C_scorer_memory_on"]["hit@1"]
        and not task["metrics"]["B_scorer_memory_off"]["hit@1"]
    ]
    regression = [
        task["task_id"]
        for task in tasks
        if task["metrics"]["B_scorer_memory_off"]["hit@1"]
        and not task["metrics"]["C_scorer_memory_on"]["hit@1"]
    ]
    active_benefit = [
        task for task in active if task["metrics"]["C_scorer_memory_on"]["hit@1"]
        and not task["metrics"]["B_scorer_memory_off"]["hit@1"]
    ]
    active_regression = [
        task for task in active if task["metrics"]["B_scorer_memory_off"]["hit@1"]
        and not task["metrics"]["C_scorer_memory_on"]["hit@1"]
    ]
    return {
        "arm_metrics": {arm: _arm_metrics(tasks, arm) for arm in common.ARMS},
        "b_vs_c": {
            "top1_changed_tasks": len(top1_changed),
            "top1_changed_task_ids": top1_changed,
            "top3_set_changed_tasks": len(top3_changed),
            "top3_set_changed_task_ids": top3_changed,
            "full_ranking_changed_tasks": len(ranking_changed),
            "full_ranking_changed_task_ids": ranking_changed,
            "any_score_changed_tasks": len(score_changed),
            "any_score_changed_task_ids": score_changed,
            "active_memory_tasks": len(active),
            "active_memory_affected_tasks": len(active_affected),
            "active_memory_affected_task_ids": [task["task_id"] for task in active_affected],
            "active_memory_top1_changed_tasks": len(
                [
                    task for task in active
                    if task["top1"]["B_scorer_memory_off"] != task["top1"]["C_scorer_memory_on"]
                ]
            ),
            "acceptable_first_rank_improved_tasks": len(improved),
            "acceptable_first_rank_improved_task_ids": improved,
            "acceptable_first_rank_regressed_tasks": len(regressed),
            "acceptable_first_rank_regressed_task_ids": regressed,
            "acceptable_first_rank_unchanged_tasks": len(unchanged),
            "acceptable_first_rank_unchanged_task_ids": unchanged,
            "hit1_benefit_tasks": benefit,
            "hit1_regression_tasks": regression,
            "active_memory_hit1_benefit_tasks": [task["task_id"] for task in active_benefit],
            "active_memory_hit1_regression_tasks": [task["task_id"] for task in active_regression],
            "unchanged_tasks": {
                task["task_id"]: task["memory_effect"]["unchanged_reason"]
                for task in tasks
                if not task["memory_effect"]["ranking_changed"]
            },
        },
        "a_vs_b": {
            "top1_changed_tasks": sum(
                1
                for task in tasks
                if task["top1"]["A_relevance_only"]
                != task["top1"]["B_scorer_memory_off"]
            ),
            "full_ranking_changed_tasks": sum(
                1
                for task in tasks
                if task["rankings"]["A_relevance_only"]
                != task["rankings"]["B_scorer_memory_off"]
            ),
        },
        "subgroups": {
            "with_active_memory": {
                "tasks": len(active),
                "arm_metrics": {arm: _arm_metrics(active, arm) for arm in common.ARMS},
            },
            "without_active_memory": {
                "tasks": len(tasks) - len(active),
                "arm_metrics": {
                    arm: _arm_metrics(
                        [task for task in tasks if not task["active_memory_summary"]], arm
                    )
                    for arm in common.ARMS
                },
            },
        },
        "gate": {
            arm: _gate_summary(tasks, arm) for arm in common.ARMS
        },
        "hard_constraint": {
            "note": "rankings are restricted to the shared eligible set, so tier-2 hard violations are 0 by construction",
            "prefilter": {
                "hard_tier_1_candidates": sum(
                    1
                    for task in tasks
                    for reason in task["excluded_candidates"].values()
                    if reason == "hard_tier_1"
                ),
                "hard_tier_2_candidates": sum(
                    1
                    for task in tasks
                    for reason in task["excluded_candidates"].values()
                    if reason == "hard_tier_2"
                ),
                "role_excluded_candidates": sum(
                    1
                    for task in tasks
                    for reason in task["excluded_candidates"].values()
                    if reason.startswith("role_")
                ),
            },
        },
    }


def _diagnose_0051(task: Mapping[str, Any] | None) -> dict[str, Any]:
    if task is None:
        return {"available": False, "reason": "p14e2_lite_0051 not found"}
    focus = {"B07BKPKXTT": "gate_selected", "B08D32JM5F": "observed_acceptable"}
    candidates = {item["item_id"]: item for item in task["candidate_table"]}
    breakdown = {}
    for item_id, role in focus.items():
        item = candidates.get(item_id) or {}
        breakdown[item_id] = {
            "role": role,
            "title": item.get("title"),
            "attributes": sorted(candidates.get(item_id, {}).get("attributes", {}).keys())
            if candidates.get(item_id, {}).get("attributes")
            else None,
            "original_bge_rank": item.get("original_bge_rank"),
            "rank_A": item.get("rank_A"),
            "rank_B": item.get("rank_B"),
            "rank_C": item.get("rank_C"),
            "score_B": item.get("score_B"),
            "score_C": item.get("score_C"),
            "query_relevance_score": item.get("query_relevance_score"),
            "query_component": round(0.5 * float(item.get("query_relevance_score") or 0.0), 6),
            "memory_match_component": item.get("preference_match_score"),
            "memory_conflict_component": item.get("preference_conflict_score"),
            "matched_preferences": item.get("matched_preferences"),
            "conflicting_preferences": item.get("conflicting_preferences"),
            "unknown_preferences": item.get("unknown_preferences"),
            "label": item.get("label"),
        }
    b07 = breakdown.get("B07BKPKXTT", {})
    b08 = breakdown.get("B08D32JM5F", {})
    score_gap_b = (
        round(float(b07.get("score_B") or 0) - float(b08.get("score_B") or 0), 6)
        if b07 and b08
        else None
    )
    score_gap_c = (
        round(float(b07.get("score_C") or 0) - float(b08.get("score_C") or 0), 6)
        if b07 and b08
        else None
    )
    return {
        "available": True,
        "schema": "agentrecx.memory_contribution.case_0051.v1",
        "task_id": task["task_id"],
        "query": task["query"],
        "hard_requirements": task["hard_requirements"],
        "active_memory": task["active_memory_summary"],
        "acceptable_ids_in_pool": task["acceptable_ids_in_pool"],
        "gate_C": task["gate"]["C_scorer_memory_on"],
        "top1": task["top1"],
        "rankings": task["rankings"],
        "candidates": breakdown,
        "score_gap_B07_minus_B08": {"B": score_gap_b, "C": score_gap_c},
        "gap_narrowed_by_memory": (
            round(float(score_gap_b) - float(score_gap_c), 6)
            if score_gap_b is not None and score_gap_c is not None
            else None
        ),
        "why_B07_still_top": (
            "B07 has a query-relevance component of 0.315789 and no memory match/conflict; "
            "B08 has 0.105263 query plus the 0.2 microphone match = 0.305263. "
            "The 0.2 positive bonus is smaller than the 0.210526 query-score gap, and "
            "B07's missing microphone/rechargeable facts are UNKNOWN, not a conflict, so "
            "the fixed-weight scorer still ranks B07 first."
        ),
        "unknown_is_not_absence": (
            "B07 has no catalogue evidence for microphone or rechargeable; those preferences "
            "are recorded as unknown and earn zero bonus. They are not treated as 'no microphone'."
        ),
    }


def main() -> int:
    rows = common.load_ranking_rows()
    tasks = [_build_task(row) for row in rows]
    by_id = {task["task_id"]: task for task in tasks}
    aggregate = _aggregate(tasks)
    aggregate.update(
        {
            "schema": common.SCHEMA_COMPARISON,
            "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": str(common.RANKING_DATASET.relative_to(common.REPO)).replace("\\", "/"),
            "dataset_sha256": common.sha256_file(common.RANKING_DATASET),
            "tasks": len(tasks),
            "recommend_tasks": len(tasks),
            "label_status": common.LABEL_STATUS,
            "label_policy": "provisional observed-acceptable labels joined after ranking; all other candidates unjudged",
            "scorer_version": common.SCORER_VERSION_REF,
            "scorer_config": {
                "query_weight": 0.5,
                "positive_weight": 0.2,
                "negative_weight": 0.35,
                "max_positive_terms": 3,
                "hard_unknown_penalty": 0.1,
            },
            "memory_ablation_design": {
                "A_relevance_only": "Original-BGE order within the shared eligible set; no preference input.",
                "B_scorer_memory_off": "PreferenceAwareScorer with current-query soft preferences retained and the historical memory channel disabled.",
                "C_scorer_memory_on": "PreferenceAwareScorer with current-query soft preferences plus the legal active history preferences.",
                "shared": "same 20-candidate fixed pool, same catalogue metadata, same prepare_hard requirements, same structural eligibility filter, same DecisionGate budget",
                "query_soft_preference_channel": "the current pipeline extracts current-query conditions as hard requirements; no query-soft-preference extractor exists, so B retains an empty query-soft list but never drops a current-query hard requirement",
            },
            "hard_constraint_summary": {
                "violations@1_by_arm": {
                    arm: sum(task["hard_constraint"][arm]["violations@1"] for task in tasks)
                    for arm in common.ARMS
                },
                "violations@3_by_arm": {
                    arm: sum(task["hard_constraint"][arm]["violations@3"] for task in tasks)
                    for arm in common.ARMS
                },
                "violations@5_by_arm": {
                    arm: sum(task["hard_constraint"][arm]["violations@5"] for task in tasks)
                    for arm in common.ARMS
                },
            },
            "cross_check_existing_ranking_v1": {},
            "case_0051": _diagnose_0051(by_id.get("p14e2_lite_0051")),
        }
    )

    # Cross-check the new C order against the recorded Ranking v1 C order.
    mismatch = []
    for row in rows:
        task = by_id[row["task_id"]]
        old = list((row.get("rankings") or {}).get("C_hard_filtered_scorer") or [])
        new = list(task["rankings"]["C_scorer_memory_on"])
        if old != new:
            mismatch.append({"task_id": row["task_id"], "old": old, "new": new})
    aggregate["cross_check_existing_ranking_v1"] = {
        "old_C_order_equals_new_C_count": len(rows) - len(mismatch),
        "mismatch_tasks": mismatch,
        "old_C_hit1_count": sum(
            1
            for row in rows
            if (row.get("metrics") or {}).get("C_hard_filtered_scorer", {}).get("hit@1")
        ),
        "new_C_hit1_count": aggregate["arm_metrics"]["C_scorer_memory_on"]["hit@1"],
        "old_gate_selected_acceptable_count": sum(
            1
            for row in rows
            if row.get("gate_selected_id") in set(row.get("acceptable_ids_in_pool") or [])
        ),
    }

    # Q1/Q2 headline facts kept in the machine-readable comparison.
    aggregate["headline"] = {
        "Q1_memory_changes_ranking": aggregate["b_vs_c"]["full_ranking_changed_tasks"] > 0,
        "Q1_changed_task_count": aggregate["b_vs_c"]["full_ranking_changed_tasks"],
        "Q2_C_hit1": aggregate["arm_metrics"]["C_scorer_memory_on"]["hit@1"],
        "Q2_B_hit1": aggregate["arm_metrics"]["B_scorer_memory_off"]["hit@1"],
        "Q2_C_hit3": aggregate["arm_metrics"]["C_scorer_memory_on"]["hit@3"],
        "Q2_B_hit3": aggregate["arm_metrics"]["B_scorer_memory_off"]["hit@3"],
        "Q2_C_minus_B_hit1": (
            aggregate["arm_metrics"]["C_scorer_memory_on"]["hit@1"]
            - aggregate["arm_metrics"]["B_scorer_memory_off"]["hit@1"]
        ),
        "Q2_improved_tasks": aggregate["b_vs_c"]["acceptable_first_rank_improved_tasks"],
        "Q2_regressed_tasks": aggregate["b_vs_c"]["acceptable_first_rank_regressed_tasks"],
        "real_user_satisfaction": "not established",
    }

    common.write_json(OUT / "fixed_pool_comparison.json", aggregate)
    common.write_jsonl(OUT / "per_task_results.jsonl", tasks)
    common.write_json(OUT / "0051_diagnostic.json", aggregate["case_0051"])
    _write_supporting_files(aggregate, tasks)
    print(json_summary(aggregate))
    return 0


def _write_supporting_files(aggregate: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> None:
    """Write the small manifest plus a flat arm table for quick inspection."""
    flat = []
    for task in tasks:
        for arm in common.ARMS:
            flat.append(
                {
                    "task_id": task["task_id"],
                    "arm": arm,
                    "top1": task["top1"][arm],
                    "hit@1": task["metrics"][arm]["hit@1"],
                    "hit@3": task["metrics"][arm]["hit@3"],
                    "hit@5": task["metrics"][arm]["hit@5"],
                    "first_acceptable_rank": task["metrics"][arm]["first_acceptable_rank"],
                    "gate_action": task["gate"][arm]["action"],
                    "gate_selected": task["gate"][arm].get("selected_candidate_id"),
                }
            )
    common.write_jsonl(OUT / "arm_results.jsonl", flat)


def json_summary(aggregate: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "arm_metrics": aggregate["arm_metrics"],
            "b_vs_c": {
                key: aggregate["b_vs_c"][key]
                for key in (
                    "top1_changed_tasks",
                    "top3_set_changed_tasks",
                    "full_ranking_changed_tasks",
                    "any_score_changed_tasks",
                    "active_memory_tasks",
                    "active_memory_affected_tasks",
                    "acceptable_first_rank_improved_tasks",
                    "acceptable_first_rank_regressed_tasks",
                    "hit1_benefit_tasks",
                    "hit1_regression_tasks",
                    "unchanged_tasks",
                )
            },
            "case_0051": {
                "rank_B": aggregate["case_0051"]["candidates"]["B08D32JM5F"]["rank_B"],
                "rank_C": aggregate["case_0051"]["candidates"]["B08D32JM5F"]["rank_C"],
                "gap_B": aggregate["case_0051"]["score_gap_B07_minus_B08"]["B"],
                "gap_C": aggregate["case_0051"]["score_gap_B07_minus_B08"]["C"],
            },
            "cross_check": aggregate["cross_check_existing_ranking_v1"],
        },
        indent=1,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
