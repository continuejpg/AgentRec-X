"""Regression tests for bounded Planner retry policy and observation feedback.

These tests use the existing shared Electronics executor and real Rule tools; they do not
call Qwen and do not touch P18 Reserved data.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.p14c0.electronics_executor import ElectronicsPlannerExecutor, prepare_hard, prepare_preferences
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerObservation,
    allowed_planner_actions,
    validate_action,
)
from recommendation.preference.history import HistoryStateReader
from recommendation.shopping_companion.memory import SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14e2/benchmark"


def _load(split: str, kind: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (DATA / f"{split}.{kind}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task(task_id: str) -> dict:
    return next(row for row in _load("dev", "agent") if row["task_id"] == task_id)


def _executor(catalogue: ElectronicsCatalogue, task: dict) -> ElectronicsPlannerExecutor:
    with ShoppingCompanionMemory() as memory:
        memory.ingest_sessions(
            task["task_id"], [dict(session) for session in task["history_sessions"]],
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref=f"test:{task['task_id']}",
        )
        history = HistoryStateReader(memory.preference_service).read(
            task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"]
        )
    return ElectronicsPlannerExecutor(
        catalogue=catalogue,
        current_query=task["current_query"],
        preferences=prepare_preferences(history),
        hard=prepare_hard(task["current_query"]),
    )


@pytest.fixture(scope="module")
def catalogue() -> ElectronicsCatalogue:
    return ElectronicsCatalogue.load()


def test_advertised_actions_are_accepted_by_the_validator() -> None:
    scenarios = [
        PlannerObservation(step=1, query="q"),
        PlannerObservation(step=1, query="q", candidate_ids=("a",), needs_user_input=False),
        PlannerObservation(step=1, query="q", candidate_ids=("a",), needs_user_input=True),
        PlannerObservation(step=1, query="q", candidate_ids=("a",), reranked=True),
        PlannerObservation(step=1, query="q", candidate_ids=("a",), reranked=True,
                           needs_user_input=True),
        PlannerObservation(step=1, query="q", candidate_ids=("a",),
                           retrieval_feedback={"retrievals_remaining": 0}),
    ]
    argument_for = {
        "ASK": {"question": "Which feature matters?"},
        "RETRIEVE": {"query": "q"},
        "RERANK": {},
        "FINALIZE": {},
    }
    for observation in scenarios:
        for action in allowed_planner_actions(observation):
            validate_action(
                ActionProposal(PlannerAction(action), argument_for[action], "test"),
                observation, step=1, max_steps=8,
            )


def test_retrieval_exposes_candidate_delta_and_bounded_evidence(catalogue) -> None:
    task = _task("p14e2_lite_0041")
    executor = _executor(catalogue, task)
    observation = PlannerObservation(step=0, query=task["current_query"])
    after, tool, terminal = executor.execute(
        ActionProposal(PlannerAction.RETRIEVE, {"query": "wireless headphones desk"}, "test"), observation
    )
    assert terminal is False
    assert after.last_retrieve_query == "wireless headphones desk"
    assert after.retrieval_feedback["new_candidate_count"] == len(after.candidate_ids)
    assert after.retrieval_feedback["candidate_set_changed"] is True
    assert after.retrieval_feedback["retrievals_remaining"] == 1
    assert 0 < len(after.candidate_evidence) <= 5
    assert after.candidate_evidence[0]["item_id"]
    assert "present_attributes" in after.candidate_evidence[0]
    assert tool["new_candidate_count"] == len(after.candidate_ids)
    assert "RETRIEVE" in allowed_planner_actions(after)
    assert "RERANK" in allowed_planner_actions(after)
    # candidates exist and the Gate has not signalled clarification: ASK is not legal
    # (validate_action would reject it), so it must not be advertised.
    assert "ASK" not in allowed_planner_actions(after)
    assert "ASK" in allowed_planner_actions(replace(after, needs_user_input=True))

    after_rerank, rerank_tool, rerank_terminal = executor.execute(
        ActionProposal(PlannerAction.RERANK, {}, "test"), after
    )
    assert rerank_terminal is False
    assert after_rerank.eligibility == rerank_tool["eligibility"]
    assert after_rerank.eligibility["scope"] == "category_eligibility_only"
    assert "hard_constraint_verified" in after_rerank.eligibility
    assert after_rerank.candidate_evidence
    assert after_rerank.last_retrieve_query == "wireless headphones desk"
    assert after_rerank.retrieval_feedback == after.retrieval_feedback
    # The failed-Gate RERANK hint must be consumed; a stale hint would tell the planner
    # to RERANK while allowed_actions only permits FINALIZE.
    assert after_rerank.m5_state.get("gate_next_step") == "FINALIZE"
    # A completed rerank must not advertise a second RERANK or an illegal RETRIEVE.
    assert "RERANK" not in allowed_planner_actions(after_rerank)
    assert "RETRIEVE" not in allowed_planner_actions(after_rerank)
    assert "FINALIZE" in allowed_planner_actions(after_rerank)
    assert "ASK" not in allowed_planner_actions(after_rerank)
    exhausted = replace(after, retrieval_feedback={"retrievals_remaining": 0})
    assert "RETRIEVE" not in allowed_planner_actions(exhausted)
    assert "FINALIZE" in allowed_planner_actions(exhausted)


def test_evidence_budget_exhaustion_is_bounded_not_budget_exhausted(catalogue) -> None:
    task = _task("p14e2_lite_0060")
    executor = _executor(catalogue, task)
    observation = PlannerObservation(step=0, query=task["current_query"])
    query = "over-ear headphones mechanical switches water resistance"

    observation, first_tool, first_terminal = executor.execute(
        ActionProposal(PlannerAction.RETRIEVE, {"query": query}, "test"), observation
    )
    assert first_terminal is False
    observation, gate_tool, gate_terminal = executor.execute(
        ActionProposal(PlannerAction.FINALIZE, {}, "test"), observation
    )
    assert gate_terminal is False
    assert gate_tool["status"] == "gate_more_evidence"
    assert observation.m5_state["gate_next_step"] == "RERANK"
    assert "RERANK" in allowed_planner_actions(observation)
    assert executor.last_eligibility["scope"] == "category_eligibility_only"
    assert executor.last_eligibility["hard_constraint_verified"] == 0
    assert executor.last_eligibility["hard_constraint_unknown"] > 0

    # A repeated query is still legal while the evidence round remains, but it adds no
    # candidates.  The Gate's evidence allowance is then exhausted.
    observation, second_tool, second_terminal = executor.execute(
        ActionProposal(PlannerAction.RETRIEVE, {"query": query}, "test"), observation
    )
    assert second_terminal is False
    assert second_tool["new_candidate_count"] == 0
    assert second_tool["candidate_set_changed"] is False
    assert observation.retrieval_feedback["same_query"] is True
    assert observation.retrieval_feedback["retrievals_remaining"] == 0

    observation, final_tool, final_terminal = executor.execute(
        ActionProposal(PlannerAction.FINALIZE, {}, "test"), observation
    )
    assert final_terminal is True
    assert final_tool["status"] == "no_verified_candidate_in_scope"
    assert executor.terminal_status == "no_verified_candidate_in_scope"
    assert executor.selected_candidate_id is None
    # No third search: the bounded evidence budget ends the task instead of consuming the
    # remaining search slot and ending as budget_exhausted.
    assert executor.retrievals == 2
