"""Unit tests for Golden task metrics (no Agent startup)."""

from __future__ import annotations

import copy

from agent_eval.dataset import GoldenCase, load_dataset
from agent_eval.metrics import case_success, compute_metrics, evaluate_case, percentile


def _result_for(case: GoldenCase) -> dict:
    requirements = case.expected_requirements.as_dict()
    result = {
        "case_id": case.case_id,
        "http_status": 200,
        "actual_outcome_kind": "empty",
        "actual_refusal_code": None,
        "requirements_actual": requirements,
        "selected_product": None,
        "selected_product_id": None,
        "recommendation_id": None,
        "budget_verification": None,
        "retrieval_top5": (
            list(case.relevance.judged_ids) if case.relevance is not None else []
        ),
        "latency_ms": {"end_to_end_ms": 100.0},
        "error": None,
        "replay": {"performed": False, "completed": False, "same_response": False, "mcp_calls": 0},
        "cold_resume": {"performed": False, "process_changed": False, "new_handle_ok": False},
    }
    if case.expected_outcome.kind == "recommendation":
        selected_id = case.acceptable_product_ids[0] if case.acceptable_product_ids else "SYNTH"
        constraints = case.hard_constraints
        max_price = constraints.max_price
        amount = (max_price - 1.0) if max_price is not None else 100.0
        card = {
            "product_id": selected_id,
            "product_role": constraints.product_role or "main",
            "price": {
                "value": amount,
                "currency": constraints.currency or "CNY",
                "price_kind": constraints.price_kind or "synthetic_price",
            },
        }
        result.update({
            "actual_outcome_kind": "recommendation",
            "selected_product": card,
            "selected_product_id": selected_id,
            "recommendation_id": f"rec-{case.case_id}",
            "budget_verification": {"selected_status": "within"},
        })
    elif case.expected_outcome.kind == "refusal":
        result.update({
            "actual_outcome_kind": "refusal",
            "actual_refusal_code": case.expected_outcome.refusal_codes[0],
        })
    return result


def _evaluation_set():
    dataset = load_dataset()
    results = [_result_for(case) for case in dataset.cases]
    evaluations = [evaluate_case(case, result) for case, result in zip(
        dataset.cases, results, strict=True
    )]
    return dataset, results, evaluations


def test_all_good_synthetic_run_has_explicit_denominators() -> None:
    dataset, results, evaluations = _evaluation_set()
    metrics = compute_metrics(dataset, results, evaluations, total_cases=len(results))
    assert metrics["task_outcome_accuracy"]["numerator"] == 15
    assert metrics["task_outcome_accuracy"]["denominator"] == 15
    assert metrics["task_outcome_accuracy"]["value"] == 1.0
    assert metrics["requirement_accuracy"] == {
        "numerator": 45,
        "denominator": 45,
        "value": 1.0,
        "observed_cases": 15,
        "excluded_cases": [],
        "by_field": {
            "budget": {"passed": 15, "total": 15},
            "category": {"passed": 15, "total": 15},
            "use_cases": {"passed": 15, "total": 15},
        },
        "definition": (
            "passed (budget, category, use_cases) checks / 3 per case with observed requirements"
        ),
    }
    assert metrics["budget_violations"]["count"] == 0
    assert metrics["budget_violations"]["denominator"] == 12
    assert metrics["retrieval_precision_at_5"]["numerator"] == 1
    assert metrics["retrieval_precision_at_5"]["denominator"] == 15
    assert metrics["retrieval_precision_at_5"]["value"] == 0.0667
    assert metrics["latency_ms"]["count"] == 15
    assert metrics["correct_refusals"]["numerator"] == 2


def test_budget_violation_and_hard_constraint_failure_are_counted() -> None:
    dataset, results, _ = _evaluation_set()
    target = next(i for i, result in enumerate(results) if result["case_id"] == "laptop_update_warm")
    bad = copy.deepcopy(results[target])
    bad["selected_product"]["price"]["value"] = 9000.0
    results[target] = bad
    evaluations = [evaluate_case(case, result) for case, result in zip(
        dataset.cases, results, strict=True
    )]
    metrics = compute_metrics(dataset, results, evaluations, total_cases=len(results))
    assert metrics["budget_violations"]["count"] == 1
    assert metrics["budget_violations"]["cases"] == ["laptop_update_warm"]
    assert metrics["task_outcome_accuracy"]["numerator"] == 14
    assert "budget_violation" in evaluations[target]["failure_types"]
    assert "hard_constraint_failed" in evaluations[target]["failure_types"]


def test_wrong_refusal_code_is_not_a_correct_refusal() -> None:
    dataset, results, _ = _evaluation_set()
    target = next(i for i, result in enumerate(results) if result["case_id"] == "laptop_budget_exceeded_1000")
    bad = copy.deepcopy(results[target])
    bad["actual_refusal_code"] = "budget_unverifiable"
    results[target] = bad
    evaluations = [evaluate_case(case, result) for case, result in zip(
        dataset.cases, results, strict=True
    )]
    metrics = compute_metrics(dataset, results, evaluations, total_cases=len(results))
    assert evaluations[target]["task_success"] is False
    assert "wrong_refusal_code" in evaluations[target]["failure_types"]
    assert metrics["correct_refusals"]["numerator"] == 1
    assert metrics["correct_refusals"]["denominator"] == 2


def test_unjudged_retrieval_is_excluded_not_assumed_non_relevant() -> None:
    dataset, results, _ = _evaluation_set()
    target = next(i for i, result in enumerate(results) if result["case_id"] == "hp_noise_warm")
    results[target]["retrieval_top5"] = ["UNJUDGED_ID", *results[target]["retrieval_top5"][:4]]
    evaluations = [evaluate_case(case, result) for case, result in zip(
        dataset.cases, results, strict=True
    )]
    metrics = compute_metrics(dataset, results, evaluations, total_cases=len(results))
    precision = metrics["retrieval_precision_at_5"]
    assert precision["judged_cases"] == 2
    assert precision["denominator"] == 10
    assert precision["value"] == 0.0
    assert any(row["case_id"] == "hp_noise_warm" for row in precision["excluded_cases"])


def test_missing_refusal_payload_is_distinct_from_a_correct_refusal() -> None:
    dataset, results, _ = _evaluation_set()
    target = next(i for i, result in enumerate(results) if result["case_id"] == "hp_budget_unverifiable_500")
    bad = copy.deepcopy(results[target])
    bad["actual_outcome_kind"] = "empty"
    bad["actual_refusal_code"] = None
    results[target] = bad
    success, failures = case_success(dataset.cases[target], bad)
    assert success is False
    assert "missing_refusal_payload" in failures


def test_percentile_is_deterministic() -> None:
    assert percentile([], 0.95) is None
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([10.0, 20.0], 0.5) == 15.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 3.85
