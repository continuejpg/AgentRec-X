from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerContractError,
    PlannerObservation,
    StubPlanner,
    run_planner,
    validate_action,
)
from recommendation.demo.planner_http_proposer import (
    HttpPlannerError,
    HttpPlannerProposer,
    parse_action_payload,
)
from training.agent_sft.planner_prompt import planner_messages


class _FakeExecutor:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def execute(self, proposal: ActionProposal, observation: PlannerObservation):
        self.actions.append(proposal.action.value)
        if proposal.action is PlannerAction.RETRIEVE:
            return PlannerObservation(step=1, query=observation.query, candidate_ids=("A", "B"), state_version="v1"), {"tool": "fake_retrieve"}, False
        if proposal.action is PlannerAction.RERANK:
            return PlannerObservation(step=1, query=observation.query, candidate_ids=("B", "A"), state_version="v1", reranked=True), {"tool": "fake_rerank"}, False
        return PlannerObservation(step=1, query=observation.query, candidate_ids=("B", "A"), state_version="v1", reranked=True, terminal=True), {"tool": "fake_finalize"}, True


def test_stub_planner_executes_retrieve_rerank_finalize() -> None:
    executor = _FakeExecutor()
    run = run_planner(
        planner=StubPlanner(),
        executor=executor,
        initial_observation=PlannerObservation(step=0, query="water filter"),
    )
    assert executor.actions == ["RETRIEVE", "RERANK", "FINALIZE"]
    assert run.termination == "FINALIZE"
    assert run.final_ids == ("B", "A")
    assert len(run.steps) == 3
    assert all(step.state_version == "v1" for step in run.steps)


def test_finalize_without_candidates_is_rejected() -> None:
    with pytest.raises(PlannerContractError) as exc:
        validate_action(
            ActionProposal(PlannerAction.FINALIZE),
            PlannerObservation(step=0, query="q"),
            step=1,
            max_steps=4,
        )
    assert exc.value.code == "finalize_without_candidates"


def test_ask_requires_need_signal() -> None:
    with pytest.raises(PlannerContractError) as exc:
        validate_action(
            ActionProposal(PlannerAction.ASK, {"question": "Which category?"}),
            PlannerObservation(step=0, query="q", candidate_ids=("A",)),
            step=1,
            max_steps=4,
        )
    assert exc.value.code == "ask_without_need"


def test_max_steps_terminates() -> None:
    class _LoopExecutor:
        def execute(self, proposal, observation):
            return PlannerObservation(step=1, query=observation.query, candidate_ids=(), reranked=True), {}, False

    run = run_planner(
        planner=StubPlanner(),
        executor=_LoopExecutor(),
        initial_observation=PlannerObservation(step=0, query="q"),
        max_steps=2,
    )
    assert run.termination == "max_steps"
    assert len(run.steps) == 2


def test_http_planner_parser_accepts_action_contract() -> None:
    proposal = parse_action_payload({"action": "RETRIEVE", "arguments": {"query": "water filter"}, "rationale": "need candidates"})
    assert proposal.action is PlannerAction.RETRIEVE
    assert proposal.arguments["query"] == "water filter"


def test_http_planner_rejects_preference_operation_schema() -> None:
    with pytest.raises(HttpPlannerError) as exc:
        parse_action_payload({"operations": [{"op": "noop"}], "schema_version": "agentrecx.preference_operation.v1"})
    assert exc.value.code == "planner_used_preference_schema"


def test_http_planner_rejects_unknown_action() -> None:
    with pytest.raises(HttpPlannerError) as exc:
        parse_action_payload({"action": "FLY", "arguments": {}})
    assert exc.value.code == "planner_invalid_action"


def test_http_planner_requires_retrieve_query(monkeypatch) -> None:
    from recommendation.demo import planner_contract

    with pytest.raises(PlannerContractError) as exc:
        planner_contract.validate_action(
            ActionProposal(PlannerAction.RETRIEVE, {}),
            PlannerObservation(step=0, query="q"),
            step=1,
            max_steps=4,
        )
    assert exc.value.code == "retrieve_missing_query"


def test_http_planner_unavailable_raises_without_fallback() -> None:
    with pytest.raises(HttpPlannerError) as exc:
        HttpPlannerProposer("http://127.0.0.1:9", timeout=2).propose(PlannerObservation(step=0, query="q"))
    assert exc.value.code == "planner_unavailable"


def test_planner_prompt_contains_observation_and_contract_fields() -> None:
    messages = planner_messages({
        "contract_version": "agentrecx.planner_action.v1",
        "step": 2,
        "query": "water filter under $30",
        "candidate_ids": ["B1"],
        "candidate_count": 1,
        "state_version": "state-1",
        "m5_state": {"hard_constraint_count": 1, "constraint_kinds": ["price_max"]},
        "allowed_actions": ["ASK", "RETRIEVE", "RERANK", "FINALIZE"],
        "last_tool_observation": {"tool": "run_shopping_turn"},
        "remaining_steps": 2,
        "needs_user_input": False,
        "reranked": False,
        "last_retrieve_query": "water filter under $30",
        "retrieval_feedback": {"role_match_count": 2, "hard_constraint_eligible_count": 3},
        "role_target": {"type": "water_filter", "role": "main"},
        "eligibility": {"evaluated": True, "eligible": 3},
    })
    text = "\n".join(item["content"] for item in messages)
    for token in ("m5_state", "allowed_actions", "remaining_steps", "RERANK", "last_retrieve_query", "retrieval_feedback", "role_target", "eligibility"):
        assert token in text
    assert "operations" in text  # only the prohibition sentence, not a schema


class _PlannerStub(BaseHTTPRequestHandler):
    response = {"action": "RETRIEVE", "arguments": {"query": "water filter"}, "rationale": "stub"}

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = json.dumps(self.response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def test_http_planner_round_trip_protocol() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PlannerStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        proposal = HttpPlannerProposer(f"http://127.0.0.1:{server.server_address[1]}").propose(
            PlannerObservation(step=0, query="water filter", allowed_actions=("RETRIEVE",))
        )
        assert proposal.action is PlannerAction.RETRIEVE
        assert proposal.arguments == {"query": "water filter"}
    finally:
        server.shutdown()
