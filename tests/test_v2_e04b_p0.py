"""E-04B P0: the B2 behaviour baseline's own consistency checks (no weights, no network, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.e04b_p0_rule_baseline import (
    ARM,
    DATA_KIND,
    MEASUREMENT_KIND,
    NOT_CLAIMED,
    PROTOCOL_NAME,
    compute_behaviour_metrics,
)

RESULT = Path(__file__).resolve().parents[1] / "runs" / "e04b_p0" / "p0_baseline.json"


def _outcome(case_id: str, **metrics: object) -> SimpleNamespace:
    return SimpleNamespace(case=SimpleNamespace(case_id=case_id), metrics=SimpleNamespace(**metrics))


def test_every_rate_carries_its_own_numerator_and_denominator() -> None:
    outcomes = [
        _outcome("a", tool_calls=1, tool_selection_correct=True, terminal_outcome="COMPLETED",
                 required_actions_met=True),
        _outcome("b", tool_calls=0, tool_selection_correct=False,
                 terminal_outcome="ASKED_CLARIFICATION"),
    ]
    metrics = compute_behaviour_metrics(outcomes)
    for key in ("declared_action_contract_compliance", "budget_compliance",
                "declared_expectation_match"):
        assert set(metrics[key]) == {"numerator", "denominator", "value"}, key
        assert metrics[key]["denominator"] > 0, key
    assert metrics["declared_action_contract_compliance"]["numerator"] == 1
    # An absent counter must never be reported as a zero: the harness exposes no failure/empty
    # counters, so those metrics must say so instead of showing 0/33.
    for key in ("tool_failures", "tool_empty_results", "executed_sources",
                "grounded_candidates_total"):
        assert metrics[key]["status"] == "not_exposed_by_harness", key
    assert metrics["tool_calls_total"] == 1
    assert metrics["terminal_distribution"]["COMPLETED"] == 1


def test_the_baseline_never_claims_a_quality_metric_it_cannot_support() -> None:
    for forbidden in ("task_success_rate", "hr_at_k", "ndcg_at_k", "agent_beats_fixed_policy"):
        assert forbidden in NOT_CLAIMED
    metrics = compute_behaviour_metrics([])
    assert metrics["cases"] == 0
    assert metrics["declared_action_contract_compliance"]["value"] is None   # no denominator, no claim
    for forbidden in NOT_CLAIMED:
        assert forbidden not in {k for k in metrics if not k.startswith("not_claimed")}


def test_the_frozen_protocol_marks_this_arm_executable_offline() -> None:
    from experiments.e04b_protocol_v2 import ARMS, arm_status

    assert arm_status(ARM) == "EXECUTABLE_OFFLINE"
    assert ARMS[ARM]["budget"]["llm_calls"] == 0
    assert PROTOCOL_NAME == "agentrecx.eval_protocol.v1"


def test_the_recorded_baseline_is_a_new_fixture_measurement() -> None:
    if not RESULT.exists():
        import pytest

        pytest.skip("run experiments.e04b_p0_rule_baseline first")
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["arm"] == ARM
    assert payload["llm_calls"] == 0
    assert payload["data_kind"] == DATA_KIND == "synthetic_fixture"
    assert payload["measurement_kind"] == MEASUREMENT_KIND == "new_execution"
    assert payload["case_count"] == len(payload["case_ids"]) == len(payload["cases"])
    assert len(set(payload["case_ids"])) == payload["case_count"], "case ids must be unique"
    assert payload["metrics"]["budget_compliance"]["value"] == 1.0
    # The frozen original uses the old key; the corrected run uses the audited name.  Either way the
    # compliance figure must be 17/18 and no phantom zero may be presented as a measurement.
    compliance = payload["metrics"].get("declared_action_contract_compliance") or \
        payload["metrics"].get("legal_tool_selection")
    assert compliance["numerator"] == 17 and compliance["denominator"] == 18
    for key in ("tool_failures", "tool_empty_results"):
        value = payload["metrics"][key]
        if isinstance(value, dict) and value.get("status") == "not_exposed_by_harness":
            # Corrected shape: the counter is declared unmeasurable, so it claims no number.
            assert value.get("value") is None
        else:
            # Legacy frozen shape: a rate with its own numerator and denominator, never a bare 0.
            assert set(value) == {"numerator", "denominator", "value"}
    assert payload["metrics"]["hallucinated_candidates"] == 0
    assert payload["metrics"]["constraint_violations"] == 0
    # The one known, pre-existing gap is recorded rather than hidden.
    failing = {row["case_id"] for row in payload["cases"] if not row["passed"]}
    assert failing <= {"premature-finish-recovery"}, failing
