from __future__ import annotations

from types import SimpleNamespace

from benchmarks.agent.capabilities import POLICY_CAPABILITIES, fairness_issues
from benchmarks.agent.evaluator import evaluate_run
from benchmarks.agent.evaluator_v2 import (
    EVALUATION_V2_VERSION,
    PASS,
    UNKNOWN,
    evaluate_run_v2,
    load_v2_labels,
)
from benchmarks.agent.schemas import BenchTask, BenchTurn, ExpectedOutcome, PolicyId, TaskRun, TurnTrace


def _record(title: str, category: str, price: float | None = None):
    return SimpleNamespace(title=title, categories=("Sports & Outdoors", category), price=price, details=())


def _run(final_ids, *, pool=(), actions=(), active=(), waiting=False):
    turns = [
        TurnTrace(
            turn_index=0,
            user_input="request",
            effective_input="request",
            response_status="waiting_for_user" if waiting else "finished",
            candidate_pool_ids=tuple(pool),
            final_ids=tuple(final_ids),
            retrieval_actions=tuple(actions),
        )
    ]
    return TaskRun(
        task_id="t",
        policy=PolicyId.C,
        runnable=True,
        status="finished",
        turns=turns,
        final_ids=tuple(final_ids),
        active_preference_attributes=tuple(active),
        retrieval_actions=tuple(actions),
    )


def _task(final_any, actions_any=(), must_exclude_title_terms=()):
    return BenchTask(
        task_id="t",
        split="dev",
        scenario="filter_bottle_role",
        history=("h",),
        turns=(BenchTurn("request"),),
        expected=ExpectedOutcome(
            final_any=tuple(final_any),
            actions_any=tuple(actions_any),
            must_exclude_title_terms=tuple(must_exclude_title_terms),
        ),
    )


def _label(**overrides):
    label = {
        "target_type": "water_filter",
        "target_role": "main",
        "accept_category_terms": ["water filters"],
        "accept_title_terms_any": ["water filter", "water filtration", "water purifier"],
        "reject_category_terms": ["sports water bottles"],
        "reject_title_terms": ["replacement", "cartridge", "mouthpiece"],
        "active_preference_attributes": [],
        "inactive_preference_attributes": [],
        "hard_constraints": [],
        "exact_identity_required": False,
    }
    label.update(overrides)
    return label


def test_v2_label_version_and_development_labels_are_present() -> None:
    labels = load_v2_labels()
    assert EVALUATION_V2_VERSION
    assert len(labels) == 20
    assert labels["dev_02_accessory_then_restore"]["reject_title_terms"] == [
        "replacement",
        "cartridge",
        "mouthpiece",
        "adapter",
        "tablet",
        "tablets",
    ]


def test_forbidden_bottle_title_term_does_not_make_a_complete_filter_wrong() -> None:
    # B09CJN-style product: main Water Filters category, title mentions included bottles.
    records = {"B09": _record("Water Filter Pump Set with 2 Collapsible Water Bottles", "Water Filters")}
    run = _run(["B09"], pool=["B09"], actions=["RETRIEVE"])
    task = _task(["B09"], actions_any=["RETRIEVE"], must_exclude_title_terms=["bottle"])
    ev = evaluate_run_v2(run, task, records, _label())
    assert ev.product_role_satisfaction == PASS
    assert ev.task_success_v2 == PASS
    assert ev.exact_identity_hit is True
    # Legacy remains unchanged and still sees the blanket title term.
    legacy = evaluate_run(run, task, records)
    assert legacy.passed is False


def test_role_correct_but_non_expected_identity_is_diagnostic_not_role_failure() -> None:
    records = {"STOVE": _record("Portable Camping Stove", "Camping Stoves")}
    run = _run(["STOVE"], pool=["STOVE"], actions=["RETRIEVE"])
    task = _task(["B01985FDA4"], actions_any=["RETRIEVE"])
    label = _label(
        target_type="stove",
        accept_category_terms=["camping stoves"],
        accept_title_terms_any=["stove"],
        reject_category_terms=["water filters"],
        reject_title_terms=["water filter"],
    )
    ev = evaluate_run_v2(run, task, records, label)
    assert ev.product_role_satisfaction == PASS
    assert ev.exact_identity_hit is False
    assert ev.task_success_v2 == PASS
    assert ev.evidence["exact_identity_required"] is False


def test_m7_answer_without_r1_action_is_not_required_when_snapshot_is_valid() -> None:
    records = {"FILTER": _record("Portable Water Filtration System", "Water Filters")}
    run = _run(["FILTER"], pool=["FILTER"], actions=[])
    task = _task(["FILTER"], actions_any=["RETRIEVE", "RERANK"])
    ev = evaluate_run_v2(run, task, records, _label())
    assert ev.tool_action_validity == "NOT_REQUIRED"
    assert ev.task_success_v2 == PASS
    legacy = evaluate_run(run, task, records)
    assert legacy.passed is False


def test_missing_retrieval_when_candidate_pool_cannot_satisfy_goal_is_a_failure() -> None:
    records = {"COOK": _record("Camping Cookware Set", "Open Fire Cookware")}
    run = _run([], pool=["COOK"], actions=[])
    task = _task(["FILTER"])
    ev = evaluate_run_v2(run, task, records, _label())
    assert ev.tool_action_validity == "REQUIRED_MISSING"
    assert ev.task_success_v2 != PASS
    assert ev.failure_class == "product_semantics" or ev.retrieval_adequacy in {
        "CANDIDATE_POOL_INSUFFICIENT",
        "SELECTION_OMITTED",
    }


def test_unknown_evidence_is_not_promoted_to_success() -> None:
    records = {"MYSTERY": _record("Mystery Outdoor Item", "Miscellaneous")}
    run = _run(["MYSTERY"], pool=["MYSTERY"], actions=["RETRIEVE"])
    task = _task(["MYSTERY"])
    ev = evaluate_run_v2(run, task, records, _label())
    assert ev.product_role_satisfaction == UNKNOWN
    assert ev.task_success_v2 == UNKNOWN


def test_policy_capability_matrix_declares_execution_variables() -> None:
    assert {item.policy for item in POLICY_CAPABILITIES} == {"A", "B", "C", "D"}
    for item in POLICY_CAPABILITIES:
        assert item.same_candidate_service is True
        assert item.same_catalogue is True
        assert item.same_model_sources is True
    issues = fairness_issues()
    assert any("role eligibility" in issue for issue in issues)
    assert any("soft rerank" in issue for issue in issues)
