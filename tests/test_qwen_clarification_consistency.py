"""Regression tests for current-task constraint priority after a clarification reply."""

from __future__ import annotations

from recommendation.preference.current_task_constraints import (
    CurrentTaskConstraints,
    candidate_constraint_status,
    effective_preferences,
)
from recommendation.preference.decision_gate import (
    ALLOW_FINALIZE,
    DecisionGate,
    GateBudget,
    GateContext,
)
from experiments.p14b1.diagnose import conflicts


NS = "amazon_electronics_c4"


def _reply_constraints() -> CurrentTaskConstraints:
    return CurrentTaskConstraints.from_payload({
        "preferences": [
            {
                "kind": "feature",
                "value": "wireless headphones for commuting",
                "polarity": "prefer",
                "source_text": "For this purchase, I prefer wireless headphones for commuting.",
                "mode": "add",
            },
            {
                "kind": "feature",
                "value": "wired headphones",
                "polarity": "avoid",
                "source_text": "Please avoid wired headphones.",
                "mode": "add",
            },
        ],
        "removals": [],
    }, source_turn_id="reply-1")


def test_current_avoid_overrides_history_only_in_effective_task_view():
    constraints = _reply_constraints()
    assert [item.attribute for item in constraints.hard_exclusions] == ["wired"]
    assert [item.attribute for item in constraints.soft_preferences] == ["wireless"]

    historical = [
        {"value": "wired", "polarity": "positive", "source_turn_id": "old"},
        {"value": "wireless", "polarity": "positive", "source_turn_id": "old"},
    ]
    effective = effective_preferences(historical, constraints)

    # The historical entries are untouched; only the task effective view changes.
    assert historical[0]["polarity"] == "positive"
    assert not any(
        item["value"] == "wired" and item["polarity"] == "positive"
        for item in effective
    )
    assert any(
        item["value"] == "wired" and item["polarity"] == "negative"
        for item in effective
    )
    assert any(
        item["value"] == "wireless" and item["polarity"] == "positive"
        for item in effective
    )
    assert conflicts(effective) == []


def test_current_avoid_does_not_turn_into_permanent_retraction():
    constraints = _reply_constraints()
    assert constraints.hard_exclusions[0].polarity == "exclude"
    assert constraints.hard_exclusions[0].source_turn_id == "reply-1"
    payload = constraints.as_dict()
    assert "removed" not in payload


def test_product_evidence_matching_distinguishes_wired_wireless_unknown_and_negation():
    constraints = _reply_constraints()
    wired = candidate_constraint_status(
        attributes={"wired": {"evidence": "Earphones Wired Headphones"}},
        form_factor="in_ear", constraints=constraints, product_type="headphones",
    )
    assert wired["status"] == "violated"
    assert wired["violations"][0]["attribute"] == "wired"

    wireless = candidate_constraint_status(
        attributes={"wireless": {"evidence": "Wireless Bluetooth Headphones"}},
        form_factor="in_ear", constraints=constraints, product_type="headphones",
    )
    assert wireless["status"] == "unknown"  # absence of wired evidence is not proof
    assert wireless["violations"] == []
    assert wireless["matched"][0]["attribute"] == "wireless"

    unknown = candidate_constraint_status(
        attributes={}, form_factor=None, constraints=constraints, product_type="headphones",
    )
    assert unknown["status"] == "unknown"
    assert unknown["violations"] == []

    negated = candidate_constraint_status(
        attributes={"wired": {"evidence": "wireless headphones, not wired"}},
        form_factor="in_ear", constraints=constraints, product_type="headphones",
    )
    assert negated["violations"] == []


def test_brand_exclusion_requires_trusted_brand_evidence():
    constraints = CurrentTaskConstraints.from_payload({
        "preferences": [{
            "kind": "brand", "value": "Acme", "polarity": "avoid",
            "source_text": "Please avoid Acme.", "mode": "add",
        }],
        "removals": [],
    }, source_turn_id="reply-brand")
    assert constraints.hard_exclusions[0].attribute == "brand"

    trusted = candidate_constraint_status(
        attributes={"brand": {"evidence": "Acme Bluetooth Headphones"}},
        form_factor=None, constraints=constraints, product_type="headphones",
    )
    assert trusted["status"] == "violated"

    unknown = candidate_constraint_status(
        attributes={}, form_factor=None, constraints=constraints, product_type="headphones",
    )
    assert unknown["status"] == "unknown"
    assert unknown["violations"] == []


def test_explicit_remove_is_not_a_current_task_hard_exclusion():
    constraints = CurrentTaskConstraints.from_payload({
        "preferences": [],
        "removals": [{
            "kind": "feature", "value": "wired",
            "source_text": "Forget my wired preference.", "mode": "remove",
        }],
    }, source_turn_id="reply-retract")
    assert constraints.is_empty


def _row(item_id: str, rank: int, *, task_conflicts=None):
    return {
        "product_ref": {"namespace": NS, "item_id": item_id},
        "query_rank": rank,
        "hard_tier": 0,
        "role_status": "MATCH",
        "product_type": "headphones",
        "candidate_role": "main",
        "matched_preferences": [],
        "conflicting_preferences": [],
        "unknown_preferences": [],
        "task_constraint_conflicts": task_conflicts or [],
        "task_constraint_status": {"status": "violated" if task_conflicts else "unknown"},
        "evidence_refs": [],
    }


def test_gate_filters_current_task_violation_and_never_finalizes_wired_candidate():
    gate = DecisionGate()
    constraint = _reply_constraints().hard_exclusions[0].as_dict()
    context = GateContext(
        query="I need headphones.",
        candidates=[_row("WIRED", 1, task_conflicts=[constraint]), _row("UNKNOWN", 2)],
        active_preferences=[{"value": "wireless", "polarity": "positive"}],
        hard_requirements=[],
        preference_conflicts=[],
        namespace=NS,
        query_specificity=True,
        expected_product_type="headphones",
        expected_product_role="main",
        budget=GateBudget(),
        task_constraints=[constraint],
    )
    decision = gate.check(context)
    assert decision.allow_finalize is True
    assert decision.action == ALLOW_FINALIZE
    assert decision.selected_candidate_id == "UNKNOWN"
    assert decision.task_constraint_status == "unknown"

    blocked = gate.check(GateContext(
        query="I need headphones.",
        candidates=[_row("WIRED", 1, task_conflicts=[constraint])],
        active_preferences=[{"value": "wireless", "polarity": "positive"}],
        hard_requirements=[],
        preference_conflicts=[],
        namespace=NS,
        query_specificity=True,
        expected_product_type="headphones",
        expected_product_role="main",
        budget=GateBudget(max_retrievals=1),
        task_constraints=[constraint],
    ))
    assert blocked.allow_finalize is False
    assert blocked.task_constraint_status == "violated"
    assert blocked.reason_code == "current_task_constraint_violation"
