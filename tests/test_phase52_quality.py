from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/agentrec_x_phase52"
P100 = REPO / "runs/p100_extended_evaluation"


def _json(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


pytestmark = pytest.mark.skipif(not (RUN / "p100_retrieval_judgment_waterfall.json").exists(), reason="Phase 5.2 artifacts not present")


def test_retrieval_waterfall_is_monotone_and_conditional_recall_is_labelled():
    data = _json("p100_retrieval_judgment_waterfall.json")
    totals = data["totals"]
    assert totals["judged_products"] == 1971
    assert totals["positive_products"] == 1597
    assert totals["catalog_judged"] == 61
    assert totals["index_positive"] == 52
    assert totals["top20_positive"] == 14
    assert totals["topk_positive"] == 4
    assert 0 < totals["conditional_recall_at20_indexed_positive"] < 1
    assert "conditional" in data["denominator_note"].lower()


def test_evidence_failures_are_data_coverage_not_presenter_bug():
    rows = [json.loads(line) for line in (RUN / "p100_evidence_unsupported_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 33
    assert {row["reason"] for row in rows} == {"no_verified_attributes_or_evidence"}


def test_memory_anomalies_are_classified_without_bypass():
    data = _json("p100_memory_writer_anomalies.json")
    anomalies = [row for row in data["cases"] if row.get("anomaly")]
    assert len(anomalies) == 9
    assert sum(1 for row in anomalies if row["scenario_type"] == "unknown_evidence") == 7
    assert sum(1 for row in anomalies if row["scenario_type"] == "multiple_preferences") == 2


def test_ask_analysis_uses_existing_trajectories():
    data = _json("p100_ask_analysis.json")["summary"]
    assert data["status_counts"]["awaiting_clarification"] == 140
    assert data["status_counts"]["finalized"] == 154
    assert data["status_counts"]["error"] == 3
    assert data["status_counts"]["no_verified_candidate_in_scope"] == 3
    assert len(data["no_verified_candidates"]) == 3
    assert len(data["planner_errors"]) == 3


def test_bge_judged_baseline_and_paired_difference():
    metrics = _json("p100_judged_bge_metrics.json")
    assert metrics["valid_queries"] == 100
    assert metrics["ndcg@10"] > 0.8
    vs = _json("p100_judged_bge_vs_shopranker.json")
    assert vs["mean_ndcg10_diff_bge_minus_shop"] > 0
    assert vs["paired_bootstrap_ci95"][0] > 0
    assert vs["bge_wins"] > vs["shopranker_wins"]
