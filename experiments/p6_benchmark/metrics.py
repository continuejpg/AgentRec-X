"""P6 Benchmark v0 metric definitions and minimal CPU calculators."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def rank_of_target(ranked_ids: Sequence[str], target: str) -> int | None:
    """1-based rank of the single positive target, or None when absent."""
    for position, identity in enumerate(ranked_ids, start=1):
        if str(identity) == str(target):
            return position
    return None


def classical_case_metrics(ranked_ids: Sequence[str], target: str) -> dict[str, float]:
    rank = rank_of_target(ranked_ids, target)
    if rank is None:
        return {"Recall@10": 0.0, "Recall@50": 0.0, "NDCG@10": 0.0, "MRR@10": 0.0}
    return {
        "Recall@10": float(rank <= 10),
        "Recall@50": float(rank <= 50),
        "NDCG@10": (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
        "MRR@10": (1.0 / rank) if rank <= 10 else 0.0,
    }


def aggregate_classical(
    cases: Iterable[tuple[Sequence[str], str]],
    *,
    catalogue_size: int,
) -> dict[str, Any]:
    rows = list(cases)
    totals = Counter()
    recommended: set[str] = set()
    for ranked, target in rows:
        metrics = classical_case_metrics(ranked, target)
        for key, value in metrics.items():
            totals[key] += value
        recommended.update(str(item) for item in ranked[:10])
    count = max(1, len(rows))
    return {
        "cases": len(rows),
        **{key: totals[key] / count for key in ("Recall@10", "Recall@50", "NDCG@10", "MRR@10")},
        "CatalogueCoverage@10": len(recommended) / max(1, int(catalogue_size)),
    }


def dynamic_metrics(results: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Metrics from a controlled result schema.

    Required per-task result fields:
    verified_success, hard_constraint_violations, role_correct, role_total,
    preference_update_consistent, eligible_candidate_present, tool_calls, unknown.
    """
    rows = list(results)
    count = max(1, len(rows))
    violations = sum(1 for row in rows if int(row.get("hard_constraint_violations", 0)) > 0)
    role_correct = sum(int(row.get("role_correct", 0)) for row in rows)
    role_total = sum(int(row.get("role_total", 0)) for row in rows)
    return {
        "tasks": len(rows),
        "VerifiedTaskSuccess": sum(bool(row.get("verified_success")) for row in rows) / count,
        "HardConstraintViolationRate": violations / count,
        "ProductRoleCorrectness": role_correct / max(1, role_total),
        "PreferenceUpdateConsistency": sum(bool(row.get("preference_update_consistent")) for row in rows) / count,
        "EligibleCandidateCoverage": sum(bool(row.get("eligible_candidate_present")) for row in rows) / count,
        "ToolCallsPerTask": sum(int(row.get("tool_calls", 0)) for row in rows) / count,
        "UnknownRate": sum(bool(row.get("unknown")) for row in rows) / count,
    }
