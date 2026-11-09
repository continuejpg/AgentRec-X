from __future__ import annotations

import json
from pathlib import Path

from experiments.p6_benchmark.metrics import (
    aggregate_classical,
    classical_case_metrics,
    dynamic_metrics,
)

ROOT = Path("data/p6_benchmark_v0")


def _rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_classical_metrics_known_values() -> None:
    assert classical_case_metrics(["x", "target", "y"], "target") == {
        "Recall@10": 1.0,
        "Recall@50": 1.0,
        "NDCG@10": 1.0 / 1.584962500721156,
        "MRR@10": 0.5,
    }
    assert classical_case_metrics(["x", "y"], "missing")["Recall@10"] == 0.0
    summary = aggregate_classical([(["a", "b", "target"], "target"), (["a"], "target")], catalogue_size=100)
    assert summary["cases"] == 2
    assert 0.0 < summary["Recall@10"] <= 1.0
    assert 0.0 < summary["CatalogueCoverage@10"] <= 1.0


def test_classical_split_has_no_future_history() -> None:
    dev = _rows("classical_dev.jsonl")
    holdout = _rows("classical_holdout.jsonl")
    assert len(dev) == 160
    assert len(holdout) == 40
    assert not ({row["user_id"] for row in dev} & {row["user_id"] for row in holdout})
    for row in dev + holdout:
        assert row["validation_time_ms"] < row["test_time_ms"]
        assert len(row["history_parent_asins"]) == row["history_len"]
        assert len(row["history_item_ids"]) == len(row["history_parent_asins"])
        assert row["history_len"] >= 1
        assert 0 not in row["history_item_ids"]
    report = json.loads((ROOT / "leakage_report.json").read_text(encoding="utf-8"))
    assert report["user_overlap"] == 0
    assert report["history_target_overlap_cases"] >= 0


def test_dynamic_tasks_are_split_by_template_and_have_checks() -> None:
    dev = _rows("dynamic_dev.jsonl")
    holdout = _rows("dynamic_holdout.jsonl")
    assert len(dev) == 6
    assert len(holdout) == 4
    assert not ({row["template_id"] for row in dev} & {row["template_id"] for row in holdout})
    for row in dev + holdout:
        assert row["synthetic_user_request"] is True
        assert row["source"]["not_frozen"] is True
        assert row["turns"]
        assert "must_preserve_hard_constraints" in row["checks"]
        assert row["max_steps"] > 0


def test_dynamic_metric_definitions() -> None:
    rows = [
        {"verified_success": True, "hard_constraint_violations": 0, "role_correct": 2, "role_total": 2,
         "preference_update_consistent": True, "eligible_candidate_present": True, "tool_calls": 3, "unknown": False},
        {"verified_success": False, "hard_constraint_violations": 1, "role_correct": 1, "role_total": 2,
         "preference_update_consistent": False, "eligible_candidate_present": False, "tool_calls": 1, "unknown": True},
    ]
    metrics = dynamic_metrics(rows)
    assert metrics["VerifiedTaskSuccess"] == 0.5
    assert metrics["HardConstraintViolationRate"] == 0.5
    assert metrics["ProductRoleCorrectness"] == 0.75
    assert metrics["ToolCallsPerTask"] == 2.0
    assert metrics["UnknownRate"] == 0.5
