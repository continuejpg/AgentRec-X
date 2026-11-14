from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "runs/agent_ranker_phase45"


def test_topk_count_and_metrics():
    data = json.loads((R / "topk_e2e.json").read_text(encoding="utf-8"))
    assert len(data["records"]) == 57
    natural = [row for row in data["records"] if row["arm"] == "natural"]
    existing = [row for row in data["records"] if row["arm"] == "existing"]
    hybrid = [row for row in data["records"] if row["arm"] == "hybrid"]
    assert sum(row["hit@1"] for row in natural) == 19
    assert sum(row["hit@3"] for row in natural) == 19
    assert sum(row["hit@1"] for row in existing) == 17
    assert sum(row["hit@1"] for row in hybrid) == 18
    assert all(row["hard_violations"] == 0 for row in data["records"])
    assert all(row["ask"] is False for row in data["records"])


def test_qwen_explanations_are_shadow_only():
    data = json.loads((R / "topk_explanations.json").read_text(encoding="utf-8"))
    assert data["tasks"] == 19
    assert all("deterministic" in row for row in data["records"])
    assert all(row["error"] is None for row in data["records"])


def test_cf010_is_truncation_not_empty_action():
    data = json.loads((R / "cf010_parse_diagnosis.json").read_text(encoding="utf-8"))
    assert data["invalid_payloads"] == 6
    assert data["raw_lengths"] == [1128] * 6
    assert "truncated" in data["cause"].lower() or "incomplete" in data["cause"].lower()


def test_human_review_is_zero_not_fabricated():
    data = json.loads((R / "human_preference_eval.json").read_text(encoding="utf-8"))
    assert data["tasks_total"] == 40
    assert data["human_review_count"] == 0
    assert data["review_provenance"] == "UNJUDGED"
