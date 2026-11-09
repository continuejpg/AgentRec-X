"""Post-run evaluator for P14-A trajectories.

This module is the **only** place allowed to read the evaluator-only ground truth
(``data/p14a/dataset/train_eval.jsonl``).  It runs *after* a trajectory has been
completed and never returns anything into the Agent runtime, so the agent cannot see
reference products, oracle dialogues or preferences.

It reports honest, limited-catalogue-aware diagnostics rather than a single
"success rate": target reachability in the real search tool, target presence in the
final recommendation, whether hard constraints were evaluated, and (for oracle
diagnostics only) whether the recovered preference text overlaps the released
reference preferences.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

__all__ = [
    "EVAL_SCHEMA",
    "evaluate_trajectory",
    "ground_truth_target_ids",
    "preference_recovery_overlap",
]

EVAL_SCHEMA = "agentrecx.p14a.trajectory_eval.v1"

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: object) -> set[str]:
    return set(_TOKEN.findall(str(text or "").casefold()))


def _wanted_features(entry: Mapping[str, Any]) -> list[str]:
    features = entry.get("wanted_features") or ()
    if not isinstance(features, (list, tuple)):
        return []
    return [str(item).strip() for item in features if str(item).strip()]


def ground_truth_target_ids(eval_row: Mapping[str, Any]) -> list[str]:
    """Reference product ids from the released train ground truth (evaluator only)."""
    ground_truth = eval_row.get("ground_truth")
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return []
    if not isinstance(ground_truth, Mapping):
        return []
    if "preferences" in ground_truth and isinstance(ground_truth["preferences"], Sequence):
        ids = []
        for entry in ground_truth["preferences"]:
            if isinstance(entry, Mapping) and entry.get("product_id"):
                ids.append(str(entry["product_id"]))
        return ids
    if ground_truth.get("product_id"):
        return [str(ground_truth["product_id"])]
    return []


def _single_preference_entry(ground_truth: Mapping[str, Any]) -> Mapping[str, Any]:
    if "preferences" in ground_truth and isinstance(ground_truth["preferences"], Sequence) and ground_truth["preferences"]:
        first = ground_truth["preferences"][0]
        if isinstance(first, Mapping):
            return first
    return ground_truth


def preference_recovery_overlap(
    trajectory: Mapping[str, Any],
    eval_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Diagnostic only: token overlap between recovered preference text and reference features.

    Only meaningful for explicit ``oracle_diagnostic`` trajectories (normal runs have no
    history and therefore no recovered preferences); the result records that distinction.
    """
    ground_truth = eval_row.get("ground_truth")
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            ground_truth = {}
    if not isinstance(ground_truth, Mapping):
        return {"applicable": False, "reason": "ground truth is not an object"}
    entry = _single_preference_entry(ground_truth)
    wanted = _wanted_features(entry)
    reference_tokens: set[str] = set()
    for feature in wanted:
        reference_tokens |= _tokens(feature)
    memory = trajectory.get("memory") or {}
    preferences = memory.get("preferences") or ()
    recovered_tokens: set[str] = set()
    for preference in preferences:
        if not isinstance(preference, Mapping):
            continue
        value = preference.get("value")
        if isinstance(value, (list, tuple)):
            for item in value:
                recovered_tokens |= _tokens(item)
        else:
            recovered_tokens |= _tokens(value)
    mode = str(trajectory.get("mode") or "")
    return {
        "applicable": mode == "oracle_diagnostic",
        "mode": mode,
        "recovered_preference_count": len(preferences),
        "reference_feature_count": len(wanted),
        "token_overlap": len(recovered_tokens & reference_tokens),
        "recovered_token_count": len(recovered_tokens),
        "reference_token_count": len(reference_tokens),
        "note": (
            "diagnostic only: oracle_diagnostic runs use the released oracle memory block; "
            "normal runs have no released history and cannot recover preferences"
            if mode != "oracle_diagnostic"
            else "diagnostic only; not a normal Agent success"
        ),
    }


def _search_product_ids(trajectory: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for call in trajectory.get("tool_calls") or ():
        if not isinstance(call, Mapping) or call.get("tool") != "product_search":
            continue
        output = call.get("output")
        if not isinstance(output, Mapping):
            continue
        for hit in output.get("hits") or ():
            if isinstance(hit, Mapping) and hit.get("product_id"):
                ids.append(str(hit["product_id"]))
    return ids


def _trajectory_products(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for item in trajectory.get("recommendation") or ():
        if isinstance(item, Mapping) and item.get("product_id"):
            products.append(dict(item))
    return products


def evaluate_trajectory(
    trajectory: Mapping[str, Any],
    eval_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one completed trajectory against the evaluator-only train ground truth."""
    targets = ground_truth_target_ids(eval_row)
    searched = _search_product_ids(trajectory)
    recommended = _trajectory_products(trajectory)
    recommended_ids = [str(item["product_id"]) for item in recommended]
    target_set = set(targets)
    catalogue = trajectory.get("catalogue") or {}
    terminal = trajectory.get("terminal") or {}
    constraints = trajectory.get("constraints") or ()
    stage_2 = (trajectory.get("stages") or {}).get("stage_2") or {}
    stage_2_result = stage_2.get("result") or {}
    eligibility = stage_2_result.get("eligibility") or {}

    question_type = (trajectory.get("task") or {}).get("question_type")
    presented_count = len(recommended_ids)
    task_completed = bool((trajectory.get("terminal") or {}).get("succeeded"))
    single_product_presentation_ok = (
        presented_count == 1 if question_type == "single_product" and task_completed else None
    )
    return {
        "schema": EVAL_SCHEMA,
        "mode": trajectory.get("mode"),
        "task": trajectory.get("task"),
        "catalogue": {
            "catalogue_size": catalogue.get("catalogue_size"),
            "limited_catalogue": catalogue.get("limited_catalogue"),
        },
        "terminal": dict(terminal),
        "target_ids": targets,
        "target_in_search_hits": bool(target_set & set(searched)),
        "target_in_recommendation": bool(target_set & set(recommended_ids)),
        "task_completed": task_completed,
        "awaiting_clarification": bool(trajectory.get("awaiting_clarification")),
        "stage_2_executed": bool(trajectory.get("stage_2_executed")),
        "stage_2_diagnostic_only": bool(trajectory.get("stage_2_diagnostic_only")),
        "presented_recommendation_count": presented_count,
        "single_product_presentation_ok": single_product_presentation_ok,
        "target_rank_in_search": next(
            (index + 1 for index, product_id in enumerate(searched) if product_id in target_set),
            None,
        ),
        "recommendation_top1": recommended_ids[0] if recommended_ids else None,
        "recommendation_ids": recommended_ids,
        "search_hit_count": len(searched),
        "ledger_entry_count": len((trajectory.get("ledger") or {}).get("entries") or ()),
        "constraints": list(constraints),
        "eligibility_evaluated": bool(eligibility.get("requirements") is not None),
        "verified_eligible_count": eligibility.get("verified_eligible_count"),
        "ineligible_count": eligibility.get("ineligible_count"),
        "unresolved_count": eligibility.get("unresolved_count"),
        "preference_recovery": preference_recovery_overlap(trajectory, eval_row),
        "notes": (
            "limited-catalogue evaluation: target reachability depends on whether the reference "
            "product happened to be in the real catalogue sample; it is not a full-catalogue metric"
            if catalogue.get("limited_catalogue")
            else "full-catalogue mode was not achieved by P14-A"
        ),
    }
