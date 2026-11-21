"""Deterministic metrics for the Web Agent Golden Dataset.

The metric layer is pure: it receives normalized case results from the real runner and never
calls an Agent, a model, or a catalogue.  Every rate reports its numerator and denominator;
a missing denominator is represented as ``None``, not as 0.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Iterable, Mapping

from agent_eval.dataset import GoldenCase, GoldenDataset

__all__ = [
    "actual_kind_for",
    "case_requirements_checks",
    "case_success",
    "compute_metrics",
    "evaluate_case",
    "percentile",
]


def percentile(values: Iterable[float], fraction: float) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if len(data) == 1:
        return round(data[0], 2)
    position = (len(data) - 1) * max(0.0, min(1.0, fraction))
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    weight = position - lower
    return round(data[lower] * (1.0 - weight) + data[upper] * weight, 2)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _requirements_actual(result: Mapping[str, Any]) -> dict | None:
    value = result.get("requirements_actual")
    return dict(value) if isinstance(value, Mapping) else None


def _actual_budget(requirements: Mapping[str, Any] | None) -> dict | None:
    if not isinstance(requirements, Mapping):
        return None
    budget = requirements.get("budget")
    return dict(budget) if isinstance(budget, Mapping) else None


def case_requirements_checks(
    case: GoldenCase, requirements_actual: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Return the three explicit requirement checks for one case.

    The three fields are ``budget`` (presence + max/min/currency), ``category`` and
    ``use_cases``.  This fixed denominator is intentional and is reported.
    """
    expected = case.expected_requirements
    actual = dict(requirements_actual) if isinstance(requirements_actual, Mapping) else {}
    actual_budget = _actual_budget(actual)
    expected_budget = expected.as_dict()["budget"]

    budget_ok: bool
    if expected_budget is None:
        budget_ok = actual_budget is None
    elif actual_budget is None:
        budget_ok = False
    else:
        budget_ok = all(
            actual_budget.get(key) == expected_budget.get(key)
            for key in ("max_amount", "min_amount", "currency")
        )

    actual_category = actual.get("category")
    category_ok = actual_category == expected.category
    actual_use_cases = actual.get("use_cases")
    actual_cases = sorted(str(item) for item in actual_use_cases) if isinstance(
        actual_use_cases, list
    ) else []
    expected_cases = sorted(expected.use_cases)
    use_cases_ok = actual_cases == expected_cases

    return [
        {
            "field": "budget",
            "expected": expected_budget,
            "actual": actual_budget,
            "passed": budget_ok,
        },
        {
            "field": "category",
            "expected": expected.category,
            "actual": actual_category,
            "passed": category_ok,
        },
        {
            "field": "use_cases",
            "expected": expected_cases,
            "actual": actual_cases,
            "passed": use_cases_ok,
        },
    ]


def _selected_card(result: Mapping[str, Any]) -> dict | None:
    value = result.get("selected_product")
    return dict(value) if isinstance(value, Mapping) else None


def _selected_price(result: Mapping[str, Any]) -> dict | None:
    card = _selected_card(result)
    if card is None:
        return None
    price = card.get("price")
    return dict(price) if isinstance(price, Mapping) else None


def _hard_checks(case: GoldenCase, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    card = _selected_card(result)
    price = _selected_price(result)
    constraints = case.hard_constraints

    if case.acceptable_product_ids:
        selected_id = result.get("selected_product_id")
        passed = selected_id in case.acceptable_product_ids
        checks.append({
            "name": "acceptable_product_id",
            "expected": list(case.acceptable_product_ids),
            "actual": selected_id,
            "passed": passed,
        })

    if constraints.max_price is not None:
        amount = _as_float((price or {}).get("value"))
        passed = amount is not None and amount <= float(constraints.max_price)
        checks.append({
            "name": "max_price",
            "expected": constraints.max_price,
            "actual": amount,
            "passed": passed,
        })
    if constraints.currency is not None:
        currency = (price or {}).get("currency")
        checks.append({
            "name": "currency",
            "expected": constraints.currency,
            "actual": currency,
            "passed": currency == constraints.currency,
        })
    if constraints.price_kind is not None:
        kind = (price or {}).get("price_kind")
        checks.append({
            "name": "price_kind",
            "expected": constraints.price_kind,
            "actual": kind,
            "passed": kind == constraints.price_kind,
        })
    if constraints.product_role is not None:
        role = (card or {}).get("product_role")
        checks.append({
            "name": "product_role",
            "expected": constraints.product_role,
            "actual": role,
            "passed": role == constraints.product_role,
        })
    return checks


def actual_kind_for(result: Mapping[str, Any]) -> str:
    """Classify one completed case without conflating HTTP success with task success."""
    http_status = result.get("http_status")
    if http_status is not None and int(http_status) != 200:
        return "execution_error"
    kind = str(result.get("actual_outcome_kind") or "empty")
    if kind in {"recommendation", "refusal", "clarification"}:
        return kind
    if kind in {"empty", "execution_error"}:
        return kind
    return "empty"


def _budget_violation(case: GoldenCase, result: Mapping[str, Any]) -> bool:
    expected_budget = case.expected_requirements.budget
    if expected_budget is None:
        return False
    if actual_kind_for(result) != "recommendation":
        return False
    price = _selected_price(result)
    if not isinstance(price, Mapping):
        return True
    amount = _as_float(price.get("value"))
    currency = price.get("currency")
    if amount is None or currency != expected_budget.currency:
        return True
    if expected_budget.max_amount is not None and amount > float(expected_budget.max_amount):
        return True
    if expected_budget.min_amount is not None and amount < float(expected_budget.min_amount):
        return True
    budget_verification = result.get("budget_verification")
    if isinstance(budget_verification, Mapping):
        selected_status = budget_verification.get("selected_status")
        if selected_status not in (None, "within"):
            return True
    return False


def case_success(case: GoldenCase, result: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return task success and failure types for one case."""
    failures: list[str] = []
    kind = actual_kind_for(result)
    expected_kind = case.expected_outcome.kind

    if result.get("error"):
        failures.append("execution_error")
    if result.get("mode_failure"):
        failures.append(str(result["mode_failure"]))
    if kind == "execution_error":
        failures.append("http_error")
    if kind != expected_kind:
        failures.append(f"outcome_{kind}_not_{expected_kind}")
        if kind == "clarification" and expected_kind == "recommendation":
            failures.append("unexpected_clarification")

    if expected_kind == "refusal":
        refusal_code = result.get("actual_refusal_code")
        if kind == "refusal" and refusal_code not in case.expected_outcome.refusal_codes:
            failures.append("wrong_refusal_code")
        if kind == "recommendation":
            failures.append("unexpected_recommendation")
        if kind == "empty":
            failures.append("missing_refusal_payload")
    elif expected_kind == "recommendation":
        if kind != "recommendation":
            failures.append("missing_recommendation")
        else:
            checks = _hard_checks(case, result)
            failed = [check["name"] for check in checks if not check["passed"]]
            if failed:
                failures.append("hard_constraint_failed")
                failures.extend(f"hard_constraint:{name}" for name in failed)
        if _budget_violation(case, result):
            failures.append("budget_violation")

    if case.expected_requirements and _requirements_actual(result) is not None:
        if not all(check["passed"] for check in case_requirements_checks(
            case, _requirements_actual(result)
        )):
            failures.append("requirement_mismatch")

    replay = result.get("replay")
    if isinstance(replay, Mapping) and replay.get("performed"):
        if not replay.get("same_response"):
            failures.append("replay_mismatch")
        if int(replay.get("mcp_calls") or 0) != 0:
            failures.append("replay_called_mcp")
        if not replay.get("completed"):
            failures.append("replay_not_completed")

    cold = result.get("cold_resume")
    if isinstance(cold, Mapping) and cold.get("performed"):
        if not cold.get("process_changed"):
            failures.append("cold_resume_process_not_changed")
        if not cold.get("new_handle_ok"):
            failures.append("cold_resume_handle_not_restored")

    return not failures, failures


def evaluate_case(case: GoldenCase, result: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the per-case scoring record stored in the JSON report."""
    success, failures = case_success(case, result)
    requirements = _requirements_actual(result)
    requirement_checks = (
        case_requirements_checks(case, requirements) if requirements is not None else []
    )
    hard_checks = _hard_checks(case, result) if actual_kind_for(result) == "recommendation" else []
    budget_violation = _budget_violation(case, result)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "execution_mode": case.execution_mode,
        "expected_outcome": case.expected_outcome.kind,
        "actual_outcome": actual_kind_for(result),
        "task_success": success,
        "failure_types": sorted(set(failures)),
        "requirement_checks": requirement_checks,
        "requirements_exact": bool(requirement_checks) and all(
            check["passed"] for check in requirement_checks
        ),
        "hard_checks": hard_checks,
        "budget_checked": case.expected_requirements.budget is not None,
        "budget_violation": budget_violation,
        "selected_product_id": result.get("selected_product_id"),
        "recommendation_id": result.get("recommendation_id"),
        "actual_refusal_code": result.get("actual_refusal_code"),
        "http_status": result.get("http_status"),
        "latency_ms": dict(result.get("latency_ms") or {}),
    }


def _judged_precision(
    cases: list[GoldenCase], results: list[Mapping[str, Any]]
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    numerator = 0
    denominator = 0
    by_id = {case.case_id: case for case in cases}
    for result in results:
        case = by_id.get(str(result.get("case_id")))
        if case is None or case.relevance is None:
            continue
        top5 = result.get("retrieval_top5")
        if not isinstance(top5, list) or not top5:
            excluded.append({"case_id": case.case_id, "reason": "no retrieval top-5"})
            continue
        judged = case.relevance.judged_ids
        unjudged = [str(item) for item in top5 if str(item) not in judged]
        if unjudged:
            excluded.append({
                "case_id": case.case_id,
                "reason": f"unjudged retrieved ids: {', '.join(unjudged)}",
            })
            continue
        relevant = sum(1 for item in top5[:5] if judged.get(str(item), 0) > 0)
        numerator += relevant
        denominator += 5
        per_case.append({
            "case_id": case.case_id,
            "top5": [str(item) for item in top5[:5]],
            "relevant": relevant,
            "denominator": 5,
            "precision": round(relevant / 5.0, 4),
        })
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 4) if denominator else None,
        "judged_cases": len(per_case),
        "cases": per_case,
        "excluded_cases": excluded,
        "definition": "mean over judged cases of relevant@5 / 5; only fully judged top-5 pools count",
    }


def compute_metrics(
    dataset: GoldenDataset,
    results: list[Mapping[str, Any]],
    case_evaluations: list[Mapping[str, Any]],
    *,
    total_cases: int | None = None,
) -> dict[str, Any]:
    """Aggregate the fixed M3-B metric set with explicit denominators."""
    total = int(total_cases) if total_cases is not None else len(dataset.cases)
    correct = sum(1 for row in case_evaluations if row.get("task_success"))
    kinds_correct = sum(
        1 for row in case_evaluations
        if row.get("actual_outcome") == row.get("expected_outcome")
    )

    requirement_rows = [
        row for row in case_evaluations if row.get("requirement_checks")
    ]
    requirement_passed = sum(
        1 for row in requirement_rows for check in row["requirement_checks"] if check["passed"]
    )
    requirement_total = sum(len(row["requirement_checks"]) for row in requirement_rows)
    requirement_by_field: dict[str, dict[str, int]] = {}
    for row in requirement_rows:
        for check in row["requirement_checks"]:
            bucket = requirement_by_field.setdefault(
                check["field"], {"passed": 0, "total": 0}
            )
            bucket["total"] += 1
            if check["passed"]:
                bucket["passed"] += 1

    budget_rows = [row for row in case_evaluations if row.get("budget_checked")]
    budget_violations = [row for row in budget_rows if row.get("budget_violation")]

    latencies = [
        float(row["latency_ms"]["end_to_end_ms"])
        for row in results
        if isinstance(row.get("latency_ms"), Mapping)
        and isinstance(row["latency_ms"].get("end_to_end_ms"), (int, float))
    ]

    failure_counts: Counter[str] = Counter()
    failure_cases: dict[str, list[str]] = {}
    for row in case_evaluations:
        for failure in row.get("failure_types") or []:
            failure_counts[failure] += 1
            failure_cases.setdefault(failure, []).append(row["case_id"])

    refusal_rows = [
        row for row in case_evaluations if row.get("expected_outcome") == "refusal"
    ]
    correct_refusals = [
        row for row in refusal_rows
        if row.get("actual_outcome") == "refusal" and row.get("task_success")
    ]
    execution_failures = [
        row for row in case_evaluations if row.get("actual_outcome") == "execution_error"
    ]

    return {
        "n_cases": total,
        "n_executed": sum(
            1 for row in results if row.get("http_status") is not None or row.get("error")
        ),
        "task_outcome_accuracy": {
            "numerator": correct,
            "denominator": total,
            "value": round(correct / total, 4) if total else None,
            "outcome_kind_numerator": kinds_correct,
            "outcome_kind_accuracy": round(kinds_correct / total, 4) if total else None,
            "definition": "task success / all Golden cases; success requires expected business outcome plus hard constraints",
        },
        "requirement_accuracy": {
            "numerator": requirement_passed,
            "denominator": requirement_total,
            "value": round(requirement_passed / requirement_total, 4)
            if requirement_total else None,
            "observed_cases": len(requirement_rows),
            "excluded_cases": [
                row["case_id"] for row in case_evaluations
                if not row.get("requirement_checks")
            ],
            "by_field": requirement_by_field,
            "definition": "passed (budget, category, use_cases) checks / 3 per case with observed requirements",
        },
        "budget_violations": {
            "count": len(budget_violations),
            "denominator": len(budget_rows),
            "rate": round(len(budget_violations) / len(budget_rows), 4)
            if budget_rows else None,
            "cases": [row["case_id"] for row in budget_violations],
            "definition": "recommendations with unknown/over/foreign-currency price among budget-constrained cases; refusals are not violations",
        },
        "retrieval_precision_at_5": _judged_precision(list(dataset.cases), results),
        "latency_ms": {
            "count": len(latencies),
            "mean": round(mean(latencies), 2) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 2) if latencies else None,
            "definition": "measured wall time per case from session/ASK start to final reply or direct product response; excludes app/MCP startup",
        },
        "failure_types": {
            "counts": dict(sorted(failure_counts.items())),
            "cases": {key: sorted(value) for key, value in sorted(failure_cases.items())},
        },
        "correct_refusals": {
            "numerator": len(correct_refusals),
            "denominator": len(refusal_rows),
            "value": round(len(correct_refusals) / len(refusal_rows), 4)
            if refusal_rows else None,
            "definition": "HTTP 200 + expected refusal code + no product / expected refusal cases",
        },
        "execution_failures": {
            "count": len(execution_failures),
            "cases": [row["case_id"] for row in execution_failures],
            "definition": "HTTP non-200 or runner error; a correct refusal is not an execution failure",
        },
        "case_results": case_evaluations,
    }
