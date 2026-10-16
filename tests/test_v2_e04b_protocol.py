"""E-04B protocol checks: the frozen dual-layer rules, asserted (no weights, no network).

These tests are cheap on purpose: they check the *protocol configuration* and the invariants a later
Gate B / Agent evaluation must satisfy, not the metrics themselves (Gate A's protocol test already
executes the real split, mask and identity checks on the real artifacts).
"""

from __future__ import annotations

import pytest

from experiments.e04b_protocol_v2 import (
    ANNOTATION_MINIMUM,
    ARMS,
    LEAK_RULES,
    METRICS,
    PROTOCOL_FREEZE_SEED,
    PROTOCOL_NAME,
    QUERY_INPUT_FIELDS,
    SCENARIOS,
    SPLIT_NAME,
    ProtocolViolation,
    arm_status,
    assert_query_is_label_free,
    assert_same_budget,
    assert_same_case_ids,
    metric_is_computable,
)


def test_the_protocol_identities_are_frozen() -> None:
    assert PROTOCOL_NAME == "agentrecx.eval_protocol.v1"
    assert SPLIT_NAME == "temporal_leave_two_out"
    assert isinstance(PROTOCOL_FREEZE_SEED, int)


def test_every_scenario_declares_inputs_and_label_status() -> None:
    assert {"A.history_only", "B.text_only", "C.history_plus_text", "D.multiturn_feedback"} <= set(
        SCENARIOS
    )
    for name, spec in SCENARIOS.items():
        assert spec["inputs"], name
        assert isinstance(spec["relevance_labels"], bool), name
        assert spec["labels"], name
    # A scenario without real relevance labels must say so rather than imply a quality metric.
    for name in ("B.text_only", "C.history_plus_text", "D.multiturn_feedback"):
        assert SCENARIOS[name]["relevance_labels"] is False, name
        assert "MISSING" in SCENARIOS[name]["labels"] or "PARTIAL" in SCENARIOS[name]["labels"]
    assert SCENARIOS["A.history_only"]["relevance_labels"] is True


def test_every_arm_declares_capabilities_budget_entry_and_status() -> None:
    assert {"B0.single_source", "B1.fixed_multi_source_rrf", "B2.rule_tool_selection",
            "B3.llm_tool_selection", "B4.full_agent"} == set(ARMS)
    for name, spec in ARMS.items():
        assert spec["capabilities"] and spec["entry"] and spec["budget"], name
        assert spec["selection_rule"], name
        assert arm_status(name) in {
            "EXECUTABLE_OFFLINE", "BLOCKED_NO_PROVIDER_AUTHORIZATION", "PARTIAL_REPLAY_ONLY",
            "NOT_IMPLEMENTED_FOR_EVAL",
        }, name
    # No arm may quietly claim it can run today when the provider budget is exhausted.
    assert arm_status("B3.llm_tool_selection") == "BLOCKED_NO_PROVIDER_AUTHORIZATION"
    assert arm_status("B4.full_agent") in {"PARTIAL_REPLAY_ONLY", "BLOCKED_NO_PROVIDER_AUTHORIZATION"}


def test_a_best_of_per_case_source_rule_is_not_allowed() -> None:
    for name in ("B0.single_source", "B1.fixed_multi_source_rrf", "B2.rule_tool_selection"):
        rule = ARMS[name]["selection_rule"].lower()
        assert "frozen" in rule, name


def test_same_case_ids_is_enforced() -> None:
    ids = ["u1:1", "u2:2", "u3:3"]
    assert assert_same_case_ids({"B0": ids, "B1": list(ids), "B2": list(ids)}) == tuple(ids)
    with pytest.raises(ProtocolViolation):
        assert_same_case_ids({"B0": ids, "B1": ["u9:9"]})


def test_a_comparison_may_only_be_called_same_budget_when_it_is() -> None:
    same = {"B0": {"tool_calls": 1, "llm_calls": 0}, "B1": {"tool_calls": 1, "llm_calls": 0}}
    assert_same_budget(same, label="B0 vs B1")           # no raise
    mixed = {"B0": {"tool_calls": 1, "llm_calls": 0}, "B4": {"tool_calls": 2, "llm_calls": 3}}
    with pytest.raises(ProtocolViolation):
        assert_same_budget(mixed, label="B0 vs B4")


def test_metrics_need_their_labels_before_they_can_be_computed() -> None:
    assert metric_is_computable("hr_at_k", ["test_target"]) is True
    assert metric_is_computable("hr_at_k", []) is False
    assert metric_is_computable("tool_call_count", []) is True
    # The preference-following rate has no real labels today, so it must not be computable.
    assert metric_is_computable("preference_follow_rate", ["test_target"]) is False
    assert metric_is_computable("task_success_rate", []) is False
    with pytest.raises(ProtocolViolation):
        metric_is_computable("not_a_metric", [])


def test_every_metric_names_its_layer_source_and_code_path() -> None:
    for name, spec in METRICS.items():
        assert spec["layer"] in {"A.recommendation_quality", "B.agentic_recommendation_quality"}, name
        assert spec["source"] in {"log", "human"}, name
        assert spec["definition"], name
        assert spec["code"], name


def test_queries_may_never_be_derived_from_a_hidden_target() -> None:
    assert set(QUERY_INPUT_FIELDS) == {"test_history", "request_text"}
    assert_query_is_label_free(QUERY_INPUT_FIELDS)                       # no raise
    for forbidden in ("test_target", "validation_target", "required_frontier"):
        with pytest.raises(ProtocolViolation):
            assert_query_is_label_free((*QUERY_INPUT_FIELDS, forbidden))


def test_the_leak_rules_cover_the_four_named_risks() -> None:
    joined = " ".join(LEAK_RULES)
    for phrase in ("test_target", "hidden target", "memory", "labels are missing",
                   "past evidence", "tool budgets"):
        assert phrase in joined, phrase


def test_the_annotation_minimum_is_specified_not_assumed() -> None:
    assert ANNOTATION_MINIMUM["unit"] and ANNOTATION_MINIMUM["judges"] >= 2
    assert ANNOTATION_MINIMUM["size_for_exploratory"] > 0
    assert "without seeing the held-out target" in ANNOTATION_MINIMUM["provenance"]
    assert ANNOTATION_MINIMUM["forbidden"]


def test_the_existing_case_registry_drives_the_agent_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The controlled agent scenarios come from the repository's own case registry."""
    from recommendation.evaluation.agent import cases as agent_cases

    loaded = agent_cases.load_cases()
    assert loaded, "the agent case registry must be non-empty"
    ids = [case.case_id for case in loaded]
    assert len(ids) == len(set(ids)), "case ids must be unique so arms can be paired"
    # The same registry is what every B2-style arm scores; a paired comparison is therefore possible.
    assert assert_same_case_ids({arm: ids for arm in ("B2.rule_tool_selection",)}) == tuple(ids)
