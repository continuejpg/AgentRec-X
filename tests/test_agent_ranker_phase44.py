from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "runs/agent_ranker_phase44"


def _json(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def test_hybrid_budget_fix():
    data = _json("hybrid_budget_fix.json")
    assert data["before_status"]["budget_exhausted"] == 16
    assert data["after_status"]["budget_exhausted"] == 0
    assert data["after_status"]["finalized"] == 28
    assert data["formula_unchanged"].startswith("pool_z")


def test_complete_cf_matrices():
    rule = _json("cf_matrix_rule.json")
    qwen = _json("cf_matrix_qwen.json")
    assert rule["trajectories"] == 96 and len(rule["records"]) == 96
    assert qwen["trajectories"] == 96 and len(qwen["records"]) == 96
    for data in (rule, qwen):
        hybrid = [row for row in data["records"] if row["arm"] == "hybrid"]
        assert not any(row["status"] == "budget_exhausted" for row in hybrid)
        assert all(row["hard_constraint_violation"] is False for row in data["records"])
    assert sum(1 for row in qwen["records"] if row["status"] == "error") == 3


def test_ranking_metrics_and_blind_tool():
    ranking = _json("cf_ranking_metrics.json")
    assert set(ranking["by_arm"]) == {"existing", "natural", "hybrid"}
    tool = (R / "blind_review_tool.html").read_text(encoding="utf-8")
    assert "AgentRec-X Blind Preference Review" in tool
    assert "AgentRec-X" in tool
    tasks = (Path(__file__).resolve().parents[1] / "runs/agent_ranker_phase43/blind_review_tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([line for line in tasks if line.strip()]) == 40
