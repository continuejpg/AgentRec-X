from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "runs/agent_ranker_phase43"


def _json(name: str):
    return json.loads((R / name).read_text(encoding="utf-8"))


def test_p20_e2e_outcomes_and_constraints():
    summary = _json("e2e_summary.json")
    arms = summary["p20_e2e"]["arms"]
    assert set(arms) == {"existing", "natural", "hybrid"}
    assert arms["natural"]["success"] == 19
    assert arms["existing"]["success"] == 18
    assert arms["hybrid"]["success"] == 18
    for arm in arms.values():
        assert arm["finalized"] == 19
        assert arm["hard_constraint_violations"] == 0
        assert arm["final_answer_evidence_supported"] == 19
        assert arm["errors"] == 0


def test_counterfactual_e2e_records():
    data = _json("e2e_counterfactual.json")
    assert len(data["records"]) == 64
    assert data["records"][0]["hard_constraint_violation"] is False
    assert any(row["status"] == "budget_exhausted" for row in data["records"])


def test_memory_constraints_all_pass():
    data = _json("memory_constraint_tests.json")
    assert data["all_pass"] is True
    assert data["no_memory_invariance"]["query_vs_hybrid_ranking_changed"] == 0
    assert data["conflict_ask"]["scenarios"] == 16


def test_reproduction_and_blind_review():
    data = _json("reproduction_differences.json")
    assert data["pair_changes"]["pair_prediction_changed"] == 13
    assert data["format_control_no_memory"]["top1_changed"] == 43
    blind = (R / "blind_review_tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([line for line in blind if line.strip()]) == 40
