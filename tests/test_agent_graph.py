"""Minimal LangGraph agent-graph tests (Milestone 7B).

What these tests are for
------------------------
They prove the *orchestration contract* around the already accepted Milestone 7A
Recommendation Tool, not recommendation quality and not an LLM.  There is no
checkpoint, no catalog, no GPU and no network anywhere in this file: the Tool is
constructed with a duck-typed engine double and the decision model is injected, so
the LangGraph -> RecommendationTool -> engine chain is exercised for real while
staying offline and deterministic.

The single most important property is that the decision model can never supply or
alter the interaction history.  Several tests below attack that boundary directly.
"""

from __future__ import annotations

import ast
import json
import socket
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("langgraph", reason="Milestone 7B requires LangGraph")

from tests.agent_fakes import (  # noqa: E402
    HISTORY,
    HISTORY_WITH_DUPLICATE,
    RecordingEngine,
    ScriptedDecisionModel,
)
from recommendation.agent import (  # noqa: E402
    AGENT_DECISION_VERSION,
    AGENT_GRAPH_VERSION,
    NODE_DECIDE,
    NODE_FINALIZE,
    NODE_RECOMMEND,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentAction,
    AgentConfigurationError,
    AgentDecision,
    AgentGraph,
    AgentGraphError,
    AgentInput,
    MalformedDecision,
    history_digest,
    parse_agent_decision,
)
from recommendation.inference import (  # noqa: E402
    InferenceError,
    UnknownItemError,
)
from recommendation.tools import (  # noqa: E402
    MissingUserHistory,
    RecommendationTool,
    RecommendationToolError,
    RecommendationUnavailable,
)

RECOMMEND_PAYLOAD = {"action": "recommend", "k": 3}
DIRECT_PAYLOAD = {"action": "direct_response", "direct_response": "Hello! How can I help?"}


def make_graph(
    payload: object, engine: RecordingEngine | None = None
) -> tuple[AgentGraph, ScriptedDecisionModel, RecordingEngine]:
    """Build a graph over doubles and return everything needed for assertions."""
    engine = engine or RecordingEngine()
    decision_model = ScriptedDecisionModel(payload)
    graph = AgentGraph(decision_model, RecommendationTool(engine))
    return graph, decision_model, engine


# --------------------------------------------------------------------------- #
# A. Graph structure
# --------------------------------------------------------------------------- #


def test_graph_declares_exactly_the_milestone_7b_nodes() -> None:
    """The node set is the contract: decide, recommend, finalize - nothing else."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    declared = set(graph.node_names()) - {"__start__", "__end__"}
    assert declared == {NODE_DECIDE, NODE_RECOMMEND, NODE_FINALIZE}


def test_graph_wiring_matches_the_documented_topology() -> None:
    """START -> decide, decide -> {finalize, recommend}, recommend -> finalize, finalize -> END."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    mermaid = graph.mermaid()
    assert "__start__ --> decide" in mermaid
    assert "decide -.-> recommend" in mermaid
    assert "recommend --> finalize" in mermaid
    assert "finalize --> __end__" in mermaid
    # The direct route must reach finalize without passing through the tool node.
    # Mermaid renders conditional edges as `decide -.-> finalize` (labels are
    # HTML-escaped), so assert on the endpoint rather than the literal line.
    assert "decide" in mermaid and "finalize" in mermaid
    direct_edges = [
        line for line in mermaid.splitlines() if "decide" in line and "finalize" in line
    ]
    assert direct_edges, "decide must have a direct conditional edge to finalize"
    assert "recommend" not in direct_edges[0]


def test_graph_has_no_cycle() -> None:
    """No planner/retry/reflection loop exists in Milestone 7B."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    mermaid = graph.mermaid()
    assert "finalize --> decide" not in mermaid
    assert "recommend --> decide" not in mermaid
    assert "finalize --> recommend" not in mermaid


def test_graph_version_is_exposed() -> None:
    """The orchestration contract is versioned like the Tool contract."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    assert graph.version == AGENT_GRAPH_VERSION == 1
    assert AGENT_DECISION_VERSION == 1


def test_graph_injects_rather_than_constructs_collaborators() -> None:
    """Both collaborators are injected, which is what keeps the milestone offline."""
    graph, decision_model, engine = make_graph(RECOMMEND_PAYLOAD)
    assert graph.decision_model is decision_model
    assert graph.tool.engine is engine


@pytest.mark.parametrize(
    "decision_model",
    [None, object(), type("NoDecide", (), {})()],
)
def test_unusable_decision_model_is_rejected(decision_model: object) -> None:
    """A collaborator without decide(messages) is a configuration error."""
    with pytest.raises(AgentConfigurationError):
        AgentGraph(decision_model, RecommendationTool(RecordingEngine()))  # type: ignore[arg-type]


def test_a_non_tool_object_is_rejected() -> None:
    """The graph refuses anything that is not the accepted Tool."""
    with pytest.raises(AgentConfigurationError):
        AgentGraph(ScriptedDecisionModel(RECOMMEND_PAYLOAD), object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# B / C. The two routes
# --------------------------------------------------------------------------- #


def test_recommend_route_calls_the_tool_once_and_returns_candidates() -> None:
    """'recommend' produces tool-shaped output and does not answer from the model."""
    graph, decision_model, engine = make_graph(RECOMMEND_PAYLOAD)
    state = graph.run("suggest a tent", HISTORY)

    assert state["route"] == ROUTE_RECOMMEND
    assert decision_model.call_count == 1
    assert engine.call_count == 1

    result = state["tool_result"]
    assert result.requested_k == 3
    assert result.returned_k == 3
    assert result.history_length == len(HISTORY)

    text = state["final_response"]
    for item in result.recommendations:
        assert item.parent_asin in text
        assert f"{item.rank}." in text


def test_direct_route_never_calls_the_tool() -> None:
    """A greeting must not touch the recommender at all."""
    graph, decision_model, engine = make_graph(DIRECT_PAYLOAD)
    state = graph.run("hello", HISTORY)

    assert state["route"] == ROUTE_DIRECT
    assert state["final_response"] == DIRECT_PAYLOAD["direct_response"]
    assert decision_model.call_count == 1
    assert engine.call_count == 0
    assert "tool_result" not in state


def test_direct_route_produces_no_candidates() -> None:
    """There is no fabricated candidate list on the direct route."""
    graph, _, _ = make_graph(DIRECT_PAYLOAD)
    state = graph.run("thanks!", HISTORY)
    assert state.get("tool_result") is None
    assert "score" not in state["final_response"].lower()


def test_requested_k_reaches_the_engine_unchanged() -> None:
    """The decision's k is the only model-chosen tool argument, and it is honoured."""
    graph, _, engine = make_graph({"action": "recommend", "k": 7})
    state = graph.run("recommend seven things", HISTORY)
    assert engine.last_k == 7
    assert state["tool_result"].requested_k == 7


def test_absent_k_falls_back_to_the_tool_default() -> None:
    """Omitting k uses the Tool's documented default rather than inventing a number."""
    graph, _, engine = make_graph({"action": "recommend"})
    state = graph.run("recommend something", HISTORY)
    assert engine.last_k == 10
    assert state["tool_result"].requested_k == 10


def test_recommend_route_explains_raw_model_scores() -> None:
    """Model scores must not be presented as product evidence or confidence."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    text = graph.run("suggest a tent", HISTORY)["final_response"].lower()
    assert "score" in text
    assert "not probabilities" in text
    assert "ranking scores" in text


def test_recommend_route_avoids_product_attribute_claims() -> None:
    """The candidate-only path must not assert brand/price/quality/availability facts."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    text = graph.run("suggest a tent", HISTORY)["final_response"].lower()
    for forbidden in ("brand", "price", "$", "in stock", "rating of", "reviewers"):
        # The disclaimer names what the scores are *not*; it must never state a
        # product attribute as a fact.
        assert f"{forbidden}:" not in text
        assert f"is {forbidden}" not in text
        assert f"the {forbidden}" not in text
    assert "not evidence about a product's attributes" in text


def test_candidate_exhaustion_is_reported_not_fabricated() -> None:
    """Zero eligible candidates is a valid result, not an error and not padding."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD, RecordingEngine(available=0))
    state = graph.run("suggest a tent", HISTORY)

    assert engine.call_count == 1
    assert state["route"] == ROUTE_RECOMMEND
    assert state["tool_result"].returned_k == 0
    assert state["tool_result"].recommendations == []
    assert "no unseen product" in state["final_response"].lower()


def test_fewer_candidates_than_requested_is_a_normal_result() -> None:
    """Partial availability is surfaced, not padded up to k."""
    graph, _, _ = make_graph({"action": "recommend", "k": 5}, RecordingEngine(available=2))
    state = graph.run("suggest a tent", HISTORY)

    result = state["tool_result"]
    assert result.requested_k == 5
    assert result.returned_k == 2
    assert len(result.recommendations) == 2
    assert "only 2 of 5" in state["final_response"].lower()


# --------------------------------------------------------------------------- #
# D. The decision model never sees trusted history
# --------------------------------------------------------------------------- #


def test_decision_model_never_receives_trusted_history() -> None:
    """No history identifier may appear in the decision prompt, on either route."""
    for payload in (RECOMMEND_PAYLOAD, DIRECT_PAYLOAD):
        graph, decision_model, engine = make_graph(payload)
        graph.run("suggest a tent", HISTORY)

        prompt = decision_model.last_prompt_text
        for item in HISTORY:
            assert item not in prompt
        assert history_digest(HISTORY) not in prompt
        # ... while the Tool really did receive that history.
        if payload is RECOMMEND_PAYLOAD:
            assert engine.last_history == list(HISTORY)


def test_decision_model_sees_only_system_prompt_and_user_message() -> None:
    """The prompt is exactly two messages: policy plus the raw user text."""
    graph, decision_model, _ = make_graph(RECOMMEND_PAYLOAD)
    graph.run("suggest a tent", HISTORY)

    messages = decision_model.last_messages
    assert messages is not None
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "suggest a tent"


def test_decision_prompt_carries_no_internal_identifiers() -> None:
    """No item ids, tensor/checkpoint/mapping internals or encoded history leak in."""
    graph, decision_model, _ = make_graph(RECOMMEND_PAYLOAD)
    graph.run("suggest a tent", HISTORY)
    prompt = decision_model.last_prompt_text.lower()
    for forbidden in (
        "item_id",
        "tensor",
        "checkpoint",
        "run.json",
        "best.pt",
        "parent_asin",
        "user_history",
        "max_seq_len",
        "logits",
        "embedding",
    ):
        assert forbidden not in prompt


def test_decision_payload_cannot_supply_history() -> None:
    """A decision that tries to set the history is rejected, not merged."""
    graph, _, engine = make_graph(
        {"action": "recommend", "k": 3, "trusted_user_history": ["B000000009"]}
    )
    with pytest.raises(MalformedDecision):
        graph.run("suggest a tent", HISTORY)
    assert engine.call_count == 0


def test_decision_payload_cannot_supply_a_history_alias() -> None:
    """Any extra field at all is rejected, so there is no smuggling route."""
    graph, _, engine = make_graph(
        {"action": "recommend", "k": 3, "history": ["B000000009"]}
    )
    with pytest.raises(MalformedDecision):
        graph.run("suggest a tent", HISTORY)
    assert engine.call_count == 0


def test_history_is_unchanged_after_a_recommend_run() -> None:
    """The trusted history the run ends with is byte-identical to the input."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    source = list(HISTORY)
    state = graph.run("suggest a tent", HISTORY)

    assert state["trusted_user_history"] == HISTORY
    assert source == list(HISTORY)


def test_engine_receives_the_applications_history_verbatim() -> None:
    """Order and duplicates reach the engine untouched."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD)
    graph.run("suggest a tent", HISTORY_WITH_DUPLICATE)
    assert engine.last_history == list(HISTORY_WITH_DUPLICATE)


def test_engine_cannot_mutate_the_graph_history() -> None:
    """Even an engine that mutates its argument cannot corrupt the run's history."""

    class DestructiveEngine(RecordingEngine):
        def recommend(self, history_parent_asins, k=10):  # noqa: ANN001, ANN201
            history_parent_asins.append("B000000999")
            history_parent_asins.clear()
            return super().recommend(history_parent_asins, k=k)

    graph, _, _ = make_graph(RECOMMEND_PAYLOAD, DestructiveEngine())
    state = graph.run("suggest a tent", HISTORY)
    assert state["trusted_user_history"] == HISTORY


def test_two_runs_in_one_graph_do_not_leak_history() -> None:
    """Sequential runs on a reused graph keep their own trusted histories."""
    graph, decision_model, engine = make_graph(RECOMMEND_PAYLOAD)
    first = graph.run("first", HISTORY)
    second = graph.run("second", HISTORY_WITH_DUPLICATE)

    assert first["trusted_user_history"] == HISTORY
    assert second["trusted_user_history"] == HISTORY_WITH_DUPLICATE
    assert engine.calls[0]["history"] == list(HISTORY)
    assert engine.calls[1]["history"] == list(HISTORY_WITH_DUPLICATE)
    assert decision_model.calls[0][1].content == "first"
    assert decision_model.calls[1][1].content == "second"


# --------------------------------------------------------------------------- #
# E. Explicit failure on malformed decisions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": "browse"},
        {"action": "recommend", "k": "3"},
        {"action": "recommend", "k": True},
        {"action": "recommend", "k": 0},
        {"action": "recommend", "k": 101},
        {"action": "recommend", "k": 2.5},
        {"action": "recommend", "k": 3, "direct_response": "also this"},
        {"action": "direct_response"},
        {"action": "direct_response", "direct_response": "   "},
        {"action": "direct_response", "direct_response": "hi", "k": 3},
        {"action": "direct_response", "direct_response": 7},
        {"action": "recommend", "version": "1"},
        "not json at all",
        "",
        [1, 2, 3],
        b'{"action": "recommend"}',
    ],
)
def test_malformed_decisions_raise_instead_of_defaulting(payload: object) -> None:
    """Every unusable decision fails loudly and no Tool call is made."""
    graph, _, engine = make_graph(payload)
    with pytest.raises(MalformedDecision):
        graph.run("suggest a tent", HISTORY)
    assert engine.call_count == 0


def test_a_failing_decision_model_is_normalised_to_a_contract_error() -> None:
    """A collaborator exception must not surface as an arbitrary internal error."""
    graph, _, engine = make_graph(RuntimeError("provider exploded"))
    with pytest.raises(MalformedDecision):
        graph.run("suggest a tent", HISTORY)
    assert engine.call_count == 0


def test_malformed_decision_error_does_not_leak_the_payload() -> None:
    """The contract error reports the violation, not arbitrary raw model output."""
    graph, _, _ = make_graph({"action": "recommend", "k": 999, "secret": "B000000009"})
    with pytest.raises(MalformedDecision) as excinfo:
        graph.run("suggest a tent", HISTORY)
    assert "B000000009" not in str(excinfo.value)


def test_decisions_are_validated_at_the_boundary() -> None:
    """The parser accepts contract-shaped payloads and rejects everything else."""
    assert parse_agent_decision({"action": "recommend", "k": 3}).k == 3
    assert parse_agent_decision('{"action": "recommend"}').needs_recommendation is True
    decision = AgentDecision(action=AgentAction.DIRECT_RESPONSE, direct_response="hi")
    assert parse_agent_decision(decision) is decision
    with pytest.raises(MalformedDecision):
        parse_agent_decision(json.dumps({"action": "recommend", "k": "3"}))


def test_recommend_decision_requires_a_tool_result_to_finalize() -> None:
    """The finalize node refuses to invent output for a route that produced none."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    with pytest.raises(AgentGraphError):
        graph._finalize_node(  # noqa: SLF001
            {
                "decision": AgentDecision(action=AgentAction.RECOMMEND, k=3),
                "trusted_user_history": HISTORY,
            }
        )


def test_router_rejects_a_state_without_a_decision() -> None:
    """The conditional edge has no default branch."""
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    with pytest.raises(MalformedDecision):
        graph._route({})  # noqa: SLF001


def test_decide_node_rejects_a_blank_user_message() -> None:
    """A run cannot start from an empty turn even if it bypasses AgentInput."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD)
    with pytest.raises(MalformedDecision):
        graph._decide_node({"user_message": "   ", "trusted_user_history": list(HISTORY)})  # noqa: SLF001
    assert engine.call_count == 0


# --------------------------------------------------------------------------- #
# F. Missing / unknown history behaviour
# --------------------------------------------------------------------------- #


def test_missing_trusted_history_fails_before_any_recommendation() -> None:
    """No history means no recommendation: the Tool's strict error propagates."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD)
    with pytest.raises(MissingUserHistory):
        graph._recommend_node({"decision": AgentDecision(action=AgentAction.RECOMMEND, k=3)})  # noqa: SLF001
    assert engine.call_count == 0


def test_unknown_history_item_fails_strictly() -> None:
    """An unknown parent_asin is never dropped or silently replaced."""
    engine = RecordingEngine(error=UnknownItemError("unknown parent_asin in history"))
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD, engine)
    with pytest.raises(RecommendationToolError) as excinfo:
        graph.run("suggest a tent", HISTORY)
    assert excinfo.value.code == "unknown_history_item"


def test_engine_failure_maps_to_recommendation_unavailable() -> None:
    """An engine-level failure surfaces as the Tool's stable unavailable error."""
    engine = RecordingEngine(error=InferenceError("engine could not score"))
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD, engine)
    with pytest.raises(RecommendationUnavailable):
        graph.run("suggest a tent", HISTORY)


def test_tool_errors_are_not_rewritten_by_the_graph() -> None:
    """The graph adds no error taxonomy of its own on the recommender path."""
    engine = RecordingEngine(error=InferenceError("boom"))
    graph, _, _ = make_graph(RECOMMEND_PAYLOAD, engine)
    with pytest.raises(RecommendationToolError) as excinfo:
        graph.run("suggest a tent", HISTORY)
    assert excinfo.value.as_dict()["error"] == "recommendation_unavailable"


def test_history_is_validated_before_the_engine_is_called() -> None:
    """A malformed history never reaches the engine."""
    from pydantic import ValidationError

    graph, _, engine = make_graph(RECOMMEND_PAYLOAD, RecordingEngine())
    with pytest.raises(ValidationError):
        graph.run("suggest a tent", ("B000000001", "   "))
    assert engine.call_count == 0


# --------------------------------------------------------------------------- #
# G / H. Determinism, isolation and no side effects
# --------------------------------------------------------------------------- #


def test_repeated_runs_are_deterministic() -> None:
    """Identical inputs give identical routes, candidates and responses."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD)
    first = graph.run("suggest a tent", HISTORY)
    for _ in range(4):
        repeat = graph.run("suggest a tent", HISTORY)
        assert repeat["route"] == first["route"]
        assert repeat["final_response"] == first["final_response"]
        assert repeat["tool_result"] == first["tool_result"]
    assert engine.call_count == 5


def test_graph_opens_no_network_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole chain is in process: no sockets, no DNS, no HTTP client."""

    def _forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("the agent graph must not open a network connection")

    monkeypatch.setattr(socket, "socket", _forbid)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid)
    monkeypatch.setattr(socket, "gethostbyname", _forbid)

    graph, _, _ = make_graph(RECOMMEND_PAYLOAD)
    state = graph.run("suggest a tent", HISTORY)
    assert state["route"] == ROUTE_RECOMMEND


def test_importing_the_agent_package_does_not_import_a_provider_sdk() -> None:
    """No OpenAI/Anthropic/Google client is pulled in by Milestone 7B."""
    import importlib

    for module in ("recommendation.agent", "recommendation.agent.graph",
                   "recommendation.agent.decision", "recommendation.agent.state"):
        importlib.import_module(module)

    for forbidden in ("openai", "anthropic", "google.generativeai", "vertexai", "cohere"):
        assert forbidden not in sys.modules, f"{forbidden} must not be imported"


def test_graph_does_not_import_fastapi_or_an_http_client() -> None:
    """The agent talks to the engine in process, never to the Milestone 6 service."""
    from recommendation.agent import graph as graph_module

    source = Path(graph_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("fastapi", "requests", "httpx", "urllib.request"):
        assert forbidden not in source, f"graph.py must not reference {forbidden}"


def test_m7b_smoke_is_offline_only_and_has_no_real_chain_path() -> None:
    """Milestone 7B's smoke must not be able to load a real model.

    Regression guard for the M7B/M7C boundary: the checkpoint-based chain
    (real ``best.pt`` -> ``SASRecInferenceEngine`` -> Tool -> LangGraph) belongs to
    Milestone 7C, so no flag, import or artifact path for it may exist here.
    """
    smoke = REPO_ROOT / "experiments" / "agent_graph_smoke.py"
    source = smoke.read_text(encoding="utf-8")

    # Strip comments and the module docstring: documentation is *allowed* to name the
    # milestone that the chain belongs to; executable code is not.
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        first, last = tree.body[0].lineno, tree.body[0].end_lineno or tree.body[0].lineno
        docstring_lines = set(range(first, last + 1))

    code = "\n".join(
        line
        for number, line in enumerate(source.splitlines(), start=1)
        if number not in docstring_lines
        and line.strip()
        and not line.lstrip().startswith("#")
    )
    for forbidden in (
        "SASRecInferenceEngine",
        "InferenceConfig",
        "best.pt",
        "run.json",
        "ACCEPTED_SHA256",
        "load_real_history",
        "recommendation.inference",
        "torch",
        "runs/",
        "data/processed",
        "--real",
        "http",
    ):
        assert forbidden not in code, f"the M7B smoke must not reference {forbidden}"
    # And the smoke still proves both routes and the malformed-decision refusal.
    for required in (
        "direct route did not call the tool",
        "recommend route called the tool exactly once",
        "malformed decision refused rather than defaulted",
        "decision prompt contains no history identifier",
        "repeated recommendation run is deterministic",
    ):
        assert required in source, f"the M7B smoke must still check: {required}"


def test_agent_input_is_required_at_invoke() -> None:
    """invoke() accepts a validated AgentInput rather than a loose mapping."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD)
    with pytest.raises(AgentConfigurationError):
        graph.invoke({"user_message": "hi", "trusted_user_history": HISTORY})  # type: ignore[arg-type]
    assert engine.call_count == 0


def test_invoke_accepts_an_agent_input() -> None:
    """The validated entry point runs the graph normally."""
    graph, _, engine = make_graph(RECOMMEND_PAYLOAD)
    state = graph.invoke(AgentInput(user_message="suggest a tent", trusted_user_history=HISTORY))
    assert state["route"] == ROUTE_RECOMMEND
    assert engine.call_count == 1
