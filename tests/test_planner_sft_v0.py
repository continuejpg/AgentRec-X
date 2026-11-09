from __future__ import annotations

import json
from pathlib import Path

import pytest

from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerObservation,
    allowed_planner_actions,
    validate_action,
)
from recommendation.demo.planner_http_proposer import parse_action_payload

DATA = Path("data/planner_sft_v0")
FORBIDDEN = {"final_ids", "future_observation", "gold", "next_observation", "eval_label", "target_asin"}


def _rows(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_planner_sft_v0_contract_and_coverage() -> None:
    rows = _rows("train") + _rows("val")
    assert len(rows) == 320
    assert {row["split"] for row in rows} == {"train", "val"}
    assert len({row["sample_id"] for row in rows}) == len(rows)
    train_groups = {row["group_id"] for row in rows if row["split"] == "train"}
    val_groups = {row["group_id"] for row in rows if row["split"] == "val"}
    assert not (train_groups & val_groups)
    assert {row["target"]["action"] for row in rows if row["split"] == "val"} == {action.value for action in PlannerAction}
    actions = {row["target"]["action"] for row in rows}
    assert actions == {action.value for action in PlannerAction}
    scenarios = {row["source"]["task_family"] for row in rows}
    assert len(scenarios) == 8
    for row in rows:
        request = row["observation"]
        observation = PlannerObservation(
            step=request["step"],
            query=request["query"],
            candidate_ids=tuple(request["candidate_ids"]),
            state_version=request["state_version"],
            needs_user_input=request["needs_user_input"],
            reranked=request["reranked"],
            m5_state=request["m5_state"],
            allowed_actions=tuple(request["allowed_actions"]),
            last_tool_observation=request["last_tool_observation"],
            remaining_steps=request["remaining_steps"],
            candidate_evidence=tuple(request["candidate_evidence"]),
        )
        assert list(observation.allowed_actions) == list(allowed_planner_actions(observation))
        proposal = ActionProposal(
            PlannerAction(row["target"]["action"]),
            row["target"]["arguments"],
            row["target"]["rationale"],
        )
        validate_action(proposal, observation, step=1, max_steps=4)
        assert row["source"]["qwen_output_used"] is False
        assert not (set(_walk_keys(row)) & FORBIDDEN)
        if row["target"]["action"] in {"RERANK", "FINALIZE"}:
            assert observation.candidate_ids
            assert observation.candidate_evidence
        if row["target"]["action"] == "ASK":
            assert observation.needs_user_input
            assert not observation.candidate_ids
        if row["source"]["task_family"] == "rerank_soft_weight":
            assert any(item.get("weight") for item in observation.candidate_evidence)


def test_chat_conversion_is_planner_action_not_preference_operation() -> None:
    rows = [json.loads(line) for line in (DATA / "chat_val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
        payload = json.loads(row["messages"][-1]["content"])
        proposal = parse_action_payload(payload)
        assert proposal.action.value == row["target_action"]
        assert "operations" not in row["messages"][-1]["content"]


def test_dataset_has_no_gold_identifiers_in_target() -> None:
    for row in _rows("train"):
        assert set(row["target"]) <= {"action", "arguments", "rationale"}
        assert "parent_asin" not in json.dumps(row["target"])
        assert "final_ids" not in json.dumps(row["target"])
