"""Milestone 7C: Agent <-> Recommendation Tool real-chain end-to-end integration.

What this file is for
---------------------
It proves the M6/M7A/M7B abstractions **compose** on the real artifacts:

    trusted application history
        -> AgentGraph (accepted M7B topology, unchanged)
        -> decision = RECOMMEND(k)  (injected deterministic DecisionModel)
        -> RecommendationTool       (accepted M7A contract, unchanged)
        -> SASRecInferenceEngine    (accepted M6 engine, unchanged)
        -> accepted M5 best.pt
        -> full-catalog ranking with seen-item masking
        -> structured result back into the graph's final state
        -> honest final response

It does **not** compute or claim recommendation quality, and it is not a new
benchmark: no metric is recomputed, no split changes, no retraining, no checkpoint
selection, and `runs/` is only ever read.

Runtime
-------
This is the one consciously expensive suite in the repository: it loads the accepted
348 MB checkpoint and the full 156,746-item catalog, so the runtime is built **once**
per session (module/class-scoped) and reused by every test.  All other tests stay
lightweight and checkpoint-free.

Run the M7C gate with::

    .venv/bin/python -m pytest -q tests/test_agent_tool_e2e.py

The suite skips cleanly (never fails) when the accepted git-ignored artifacts are
absent, matching the repository's existing `artifacts_available` convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch", reason="the M7C real chain requires PyTorch")

from tests.agent_tool_e2e_runtime import (  # noqa: E402
    ACCEPTED_CHECKPOINT_SHA256,
    ACCEPTED_MANIFEST_SHA256,
    AgentToolRuntime,
    HistorySelection,
    build_runtime,
    select_history,
)
from recommendation.agent import (  # noqa: E402
    NODE_DECIDE,
    NODE_FINALIZE,
    NODE_RECOMMEND,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentDecision,
    AgentGraph,
    AgentInput,
    MalformedDecision,
)
from recommendation.inference import (  # noqa: E402
    SASRecInferenceEngine,
    UnknownItemError,
)
from recommendation.tools import (  # noqa: E402
    MissingUserHistory,
    RecommendationContext,
    RecommendationTool,
    RecommendationToolRequest,
    UnknownHistoryItem,
)

#: Unknown, deliberately non-existent public identifier used for the unknown-item
#: error path.  This is a synthetic *test input*, never a candidate or a history
#: derived from data.
UNKNOWN_PARENT_ASIN = "B9999999999"

#: Requested candidate count for the formal path.
K = 5

#: How many identical invocations the determinism check performs.
REPEATS = 3


def _artifacts_available() -> bool:
    """True when the accepted checkpoint, manifest and mappings all exist."""
    from recommendation.api.app import (
        DEFAULT_CHECKPOINT,
        DEFAULT_MANIFEST,
        DEFAULT_MAPPINGS,
    )

    return (
        Path(DEFAULT_CHECKPOINT).exists()
        and Path(DEFAULT_MANIFEST).exists()
        and Path(DEFAULT_MAPPINGS).exists()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_available(),
    reason="accepted Milestone 5/6 artifacts not present (they are git-ignored)",
)


# --------------------------------------------------------------------------- #
# Session-scoped real chain: built once and reused
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def runtime() -> AgentToolRuntime:
    """The real engine -> Tool -> graph chain, constructed once for the module."""
    return build_runtime(device="cpu", k=K)


@pytest.fixture(scope="module")
def history() -> HistorySelection:
    """The deterministically selected trusted history (see `select_history`)."""
    return select_history()


@pytest.fixture()
def restored_decision(runtime: AgentToolRuntime):
    """Restore the formal RECOMMEND decision and counters after each test."""
    default = AgentDecision(action="recommend", k=K)
    runtime.decision_model.set_decision(default)
    runtime.engine.reset_counters()
    yield
    runtime.decision_model.set_decision(default)
    runtime.engine.reset_counters()


def run_recommend(runtime: AgentToolRuntime, history: HistorySelection, k: int = K):
    """Invoke the graph on the recommend route with the trusted history."""
    runtime.decision_model.set_decision(AgentDecision(action="recommend", k=k))
    return runtime.graph.run("please recommend something for me", history.parent_asins)


# --------------------------------------------------------------------------- #
# C. Real engine, real checkpoint identity
# --------------------------------------------------------------------------- #


class TestRealDependencyChain:
    """The chain is the accepted one, and it stays constructed once."""

    def test_engine_is_the_real_accepted_class(self, runtime: AgentToolRuntime) -> None:
        assert isinstance(runtime.engine, SASRecInferenceEngine)

    def test_tool_is_the_real_accepted_class(self, runtime: AgentToolRuntime) -> None:
        assert isinstance(runtime.tool, RecommendationTool)

    def test_graph_is_the_accepted_m7b_class(self, runtime: AgentToolRuntime) -> None:
        assert isinstance(runtime.graph, AgentGraph)

    def test_chain_is_wired_graph_to_tool_to_engine(self, runtime: AgentToolRuntime) -> None:
        """graph -> tool -> engine by object identity; no intermediate hop."""
        assert runtime.graph.tool is runtime.tool
        assert runtime.tool.engine is runtime.engine
        assert runtime.graph.decision_model is runtime.decision_model

    def test_accepted_checkpoint_sha256_matches(self, runtime: AgentToolRuntime) -> None:
        assert runtime.checkpoint_sha256 == ACCEPTED_CHECKPOINT_SHA256
        assert runtime.engine.checkpoint_sha256 == ACCEPTED_CHECKPOINT_SHA256

    def test_accepted_run_manifest_is_unchanged(self, runtime: AgentToolRuntime) -> None:
        assert runtime.manifest_sha256 == ACCEPTED_MANIFEST_SHA256

    def test_full_real_catalog_is_loaded(self, runtime: AgentToolRuntime) -> None:
        assert runtime.engine.num_items == 156_746

    def test_model_is_in_eval_mode_and_frozen(self, runtime: AgentToolRuntime) -> None:
        """No dropout/BN randomness and no gradient state on the serving path."""
        assert runtime.engine.model.training is False
        assert all(not p.requires_grad for p in runtime.engine.model.parameters())

    def test_no_fake_scorer_participates_in_the_formal_path(
        self, runtime: AgentToolRuntime
    ) -> None:
        """The formal chain contains only accepted production classes."""
        from recommendation.agent import AgentGraph as GraphClass

        assert type(runtime.graph) is GraphClass
        assert type(runtime.tool) is RecommendationTool
        assert isinstance(runtime.engine, SASRecInferenceEngine)
        # The engine really owns a loaded SASRec model with the accepted size.
        assert runtime.engine.model.config.num_items == runtime.engine.num_items


# --------------------------------------------------------------------------- #
# A / B / D. Real recommendation route
# --------------------------------------------------------------------------- #


class TestRealRecommendRoute:
    """The graph reaches the real Tool and returns real, valid candidates."""

    def test_graph_selects_the_recommend_route_and_reaches_the_real_tool(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        state = run_recommend(runtime, history)

        assert state["route"] == ROUTE_RECOMMEND
        assert state["decision"].needs_recommendation is True
        # Exactly one real engine invocation for exactly one graph run.
        assert runtime.engine.recommend_calls == 1
        assert state["tool_result"] is not None

    def test_requested_k_reaches_the_real_tool_unchanged(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        state = run_recommend(runtime, history, k=K)
        assert state["tool_result"].requested_k == K

    def test_trusted_history_reaches_the_real_engine_unchanged(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """The graph's trusted history is what the engine was actually given."""
        state = run_recommend(runtime, history)

        assert state["trusted_user_history"] == history.parent_asins
        assert runtime.engine.history_digests[-1] == history.digest
        assert runtime.engine.recommend_calls == 1
        # Length and effective length reflect the real supplied history.
        assert state["tool_result"].history_length == history.length
        assert state["tool_result"].effective_history_length == min(
            history.length, runtime.engine.max_seq_len
        )

    def test_callers_history_container_is_not_mutated(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        supplied = list(history.parent_asins)
        snapshot = list(supplied)
        run_recommend(runtime, history)
        assert supplied == snapshot

    def test_decision_model_never_receives_trusted_history(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """The real path keeps the M7B trust boundary: trusted history is not prompt input."""
        run_recommend(runtime, history)
        prompt = runtime.decision_model.last_prompt_text
        for asin in history.parent_asins:
            assert asin not in prompt
        assert history.digest not in prompt

    def test_no_fastapi_or_http_hop_is_involved(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision, monkeypatch
    ) -> None:
        """The real chain is in process: no sockets, DNS or HTTP client are used."""
        import socket

        def _forbid(*args: object, **kwargs: object) -> None:
            raise AssertionError("the M7C chain must not open a network connection")

        monkeypatch.setattr(socket, "socket", _forbid)
        monkeypatch.setattr(socket, "create_connection", _forbid)
        monkeypatch.setattr(socket, "getaddrinfo", _forbid)
        monkeypatch.setattr(socket, "gethostbyname", _forbid)

        state = run_recommend(runtime, history)
        assert state["route"] == ROUTE_RECOMMEND
        assert state["tool_result"].returned_k > 0

    # -- candidate validity ------------------------------------------------ #

    def test_returned_candidates_are_valid_real_items(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        state = run_recommend(runtime, history)
        result = state["tool_result"]
        engine = runtime.engine

        assert 0 < result.returned_k <= K
        seen = set(history.parent_asins)

        for item in result.recommendations:
            # exists in the accepted catalog and resolves through the real mapping
            assert engine.has_parent_asin(item.parent_asin), item.parent_asin
            assert engine.parent_asin_to_item_id(item.parent_asin) == item.item_id
            assert engine.item_id_to_parent_asin(item.item_id) == item.parent_asin
            # PAD is never returned
            assert item.item_id != 0
            # already-seen items are masked out under accepted M6/M7A semantics
            assert item.parent_asin not in seen
            # scores are finite raw model scores
            assert item.score == item.score
            assert abs(item.score) != float("inf")

    def test_candidate_ranks_are_contiguous_and_match_tool_order(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        result = run_recommend(runtime, history)["tool_result"]
        assert [item.rank for item in result.recommendations] == list(
            range(1, result.returned_k + 1)
        )

    def test_returned_count_never_exceeds_requested_k(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        result = run_recommend(runtime, history)["tool_result"]
        assert result.returned_k <= result.requested_k

    def test_eligible_candidate_count_matches_catalog_minus_seen(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        result = run_recommend(runtime, history)["tool_result"]
        expected = runtime.engine.num_items - history.distinct_length
        assert result.eligible_candidates == expected

    def test_real_ranking_is_score_ordered(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """Serving order is the accepted rule: higher score first."""
        result = run_recommend(runtime, history)["tool_result"]
        scores = [item.score for item in result.recommendations]
        assert scores == sorted(scores, reverse=True)

    # -- G. honest final response ------------------------------------------ #

    def test_final_response_is_honest_about_candidate_identity(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """No product metadata is fabricated: only identifiers, ranks and scores."""
        state = run_recommend(runtime, history)
        text = state["final_response"]
        result = state["tool_result"]

        for item in result.recommendations:
            assert item.parent_asin in text

        lowered = text.lower()
        # There is no Product RAG yet, so none of these may appear as claims.
        for forbidden in (
            "brand",
            "category",
            "material",
            "color",
            "colour",
            "price",
            "$",
            "in stock",
            "rating of",
            "reviewers",
            "description",
            "title:",
            "best seller",
        ):
            assert f"{forbidden}:" not in lowered
            assert f"the {forbidden}" not in lowered
        assert "not evidence about a product" in lowered

    def test_final_response_does_not_claim_quality(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """M7C proves composition, so no quality/personalisation claim is made."""
        text = run_recommend(runtime, history)["final_response"].lower()
        for claim in (
            "best for you",
            "perfect for you",
            "we recommend because",
            "you will love",
            "highly relevant",
            "outperforms",
            "better than",
        ):
            assert claim not in text


# --------------------------------------------------------------------------- #
# E / F. Determinism and dependency reuse
# --------------------------------------------------------------------------- #


class TestDeterminismAndReuse:
    """Repeated equivalent requests are semantically identical on one runtime."""

    def test_repeated_invocations_are_semantically_deterministic(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        baseline = run_recommend(runtime, history)
        baseline_semantics = [
            (i.rank, i.parent_asin, i.item_id, i.score)
            for i in baseline["tool_result"].recommendations
        ]
        baseline_text = baseline["final_response"]

        for _ in range(REPEATS - 1):
            repeat = run_recommend(runtime, history)
            assert repeat["route"] == baseline["route"]
            assert repeat["final_response"] == baseline_text
            assert [
                (i.rank, i.parent_asin, i.item_id, i.score)
                for i in repeat["tool_result"].recommendations
            ] == baseline_semantics

        # One real invocation per graph run: REPEATS total.
        assert runtime.engine.recommend_calls == REPEATS

    def test_engine_and_tool_are_not_reconstructed_between_invocations(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        engine_before = runtime.engine
        tool_before = runtime.tool
        loaded_at_before = runtime.engine.loaded_at

        run_recommend(runtime, history)
        run_recommend(runtime, history)

        assert runtime.engine is engine_before
        assert runtime.tool is tool_before
        assert runtime.graph.tool.engine is engine_before
        assert runtime.engine.loaded_at == loaded_at_before
        # Two graph runs, two engine invocations -- no per-node or per-call reload.
        assert runtime.engine.recommend_calls == 2

    def test_model_parameters_are_not_mutated_by_serving(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        import torch

        weights = runtime.engine.model.item_embedding.weight.detach().clone()
        before = runtime.engine.loaded_at
        run_recommend(runtime, history)
        assert torch.equal(runtime.engine.model.item_embedding.weight.detach(), weights)
        assert runtime.engine.loaded_at == before


# --------------------------------------------------------------------------- #
# H. Direct route stays isolated
# --------------------------------------------------------------------------- #


class TestDirectRouteIsolation:
    """The direct route must not invoke the recommender at all."""

    def test_direct_route_does_not_invoke_engine_or_tool(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        runtime.engine.reset_counters()
        runtime.decision_model.set_decision(
            AgentDecision(
                action="direct_response",
                direct_response="Happy to help with a product question.",
            )
        )
        try:
            state = runtime.graph.run("hello there", history.parent_asins)
        finally:
            runtime.decision_model.set_decision(AgentDecision(action="recommend", k=K))

        assert state["route"] == ROUTE_DIRECT
        assert state["final_response"] == "Happy to help with a product question."
        # No inference happened, and no Tool result was produced.
        assert runtime.engine.recommend_calls == 0
        assert "tool_result" not in state

    def test_direct_route_still_preserves_trusted_history(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        runtime.decision_model.set_decision(
            AgentDecision(action="direct_response", direct_response="Hello.")
        )
        try:
            state = runtime.graph.run("hello", history.parent_asins)
        finally:
            runtime.decision_model.set_decision(AgentDecision(action="recommend", k=K))

        assert state["trusted_user_history"] == history.parent_asins
        assert runtime.engine.recommend_calls == 0


# --------------------------------------------------------------------------- #
# 8. Error paths on the real runtime
# --------------------------------------------------------------------------- #


class TestRealRuntimeErrorPaths:
    """Failure modes stay explicit on the real chain; nothing is silently repaired."""

    def test_empty_trusted_history_fails_explicitly(
        self, runtime: AgentToolRuntime, restored_decision
    ) -> None:
        """No history means no recommendation: explicit failure, no fallback."""
        from recommendation.agent import AgentGraphState

        state = AgentGraphState(
            user_message="recommend something",
            trusted_user_history=(),
            decision=AgentDecision(action="recommend", k=K),
        )
        with pytest.raises(MissingUserHistory):
            runtime.graph._recommend_node(state)  # noqa: SLF001 - node-level contract check
        assert runtime.engine.recommend_calls == 0

    def test_empty_history_through_the_typed_input_is_rejected(
        self, runtime: AgentToolRuntime
    ) -> None:
        """The application boundary validates before any inference can happen."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AgentInput(user_message="recommend something", trusted_user_history=())
        assert runtime.engine.recommend_calls == 0

    def test_unknown_parent_asin_in_trusted_history_fails_explicitly(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """An unknown item is never dropped, coerced to PAD or partially used."""
        poisoned = history.parent_asins + (UNKNOWN_PARENT_ASIN,)
        runtime.decision_model.set_decision(AgentDecision(action="recommend", k=K))

        with pytest.raises(UnknownHistoryItem):
            runtime.graph.run("recommend something", poisoned)

        # The engine was reached once and rejected the history itself.
        assert runtime.engine.recommend_calls == 1

    def test_unknown_parent_asin_is_rejected_by_the_real_engine_directly(
        self, runtime: AgentToolRuntime, restored_decision
    ) -> None:
        """Underlying engine semantics are unchanged and strict."""
        with pytest.raises(UnknownItemError):
            runtime.engine.recommend([UNKNOWN_PARENT_ASIN], k=K)

    def test_invalid_k_is_rejected_by_the_accepted_typed_contracts(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """M7A/M7B typed contracts already reject bad k; M7C adds no new validation."""
        from pydantic import ValidationError

        for bad_k in (0, 101, -1):
            # The AgentDecision contract refuses it outright...
            with pytest.raises(ValidationError):
                AgentDecision(action="recommend", k=bad_k)
            # ... and so does the Tool request contract.
            with pytest.raises(ValidationError):
                RecommendationToolRequest(k=bad_k)
        assert runtime.engine.recommend_calls == 0

    def test_malformed_decision_is_refused_on_the_real_runtime(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """A broken decision never silently becomes a recommendation."""
        runtime.decision_model.decision = {"action": "browse"}  # type: ignore[assignment]
        try:
            with pytest.raises(MalformedDecision):
                runtime.graph.run("recommend something", history.parent_asins)
        finally:
            runtime.decision_model.set_decision(AgentDecision(action="recommend", k=K))
        assert runtime.engine.recommend_calls == 0

    def test_partial_availability_returns_fewer_than_max_k(
        self, runtime: AgentToolRuntime, restored_decision
    ) -> None:
        """A history longer than the maximum k cannot return k candidates.

        Selected deterministically by `select_history(require_partial_availability=True)`
        (first user whose supplied history already exceeds k=100).  With only 156,746
        catalog items and the longest real history covering ~700 items, genuine
        *exhaustion* (`returned_k == 0`) is not reachable on the accepted artifacts;
        the authoritative exhaustion assertion remains the lower-level M7A/M7B
        regression.  This test covers the reachable real-data partial case instead.
        """
        long_history = select_history(
            min_length=1, require_partial_availability=True, max_k=100
        )
        assert len(long_history.parent_asins) > 100

        runtime.decision_model.set_decision(AgentDecision(action="recommend", k=100))
        state = runtime.graph.run("recommend many things", long_history.parent_asins)
        result = state["tool_result"]

        assert runtime.engine.recommend_calls == 1
        assert result.requested_k == 100
        assert 0 < result.returned_k <= 100
        # History exceeds the 50-item model window, so truncation is reported.
        assert result.history_length == len(long_history.parent_asins)
        assert result.effective_history_length == runtime.engine.max_seq_len
        assert result.history_truncated is True
        # No already-seen item leaks back even with a long history.
        assert not (set(long_history.parent_asins) & {r.parent_asin for r in result.recommendations})


# --------------------------------------------------------------------------- #
# Legacy/contract compatibility of the real chain
# --------------------------------------------------------------------------- #


class TestRealChainMatchesDirectToolCall:
    """The graph adds orchestration only: semantics equal a direct Tool call."""

    def test_graph_result_equals_direct_tool_call(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        graph_state = run_recommend(runtime, history)
        graph_items = [
            (i.rank, i.parent_asin, i.item_id, i.score)
            for i in graph_state["tool_result"].recommendations
        ]

        direct = runtime.tool.run(
            request=RecommendationToolRequest(k=K),
            context=RecommendationContext(user_history=list(history.parent_asins)),
        )
        direct_items = [
            (i.rank, i.parent_asin, i.item_id, i.score) for i in direct.recommendations
        ]

        assert graph_items == direct_items
        assert graph_state["tool_result"].returned_k == direct.returned_k
        assert graph_state["tool_result"].eligible_candidates == direct.eligible_candidates

    def test_graph_does_not_bypass_the_tool(
        self, runtime: AgentToolRuntime, history: HistorySelection, restored_decision
    ) -> None:
        """Instrumented Tool subclass proves the graph calls the Tool, not the engine."""
        class CountingTool(RecommendationTool):
            def __init__(self, engine: SASRecInferenceEngine) -> None:
                super().__init__(engine)
                self.invocations: list[int] = []

            def run(self, request=None, context=None):  # noqa: ANN001, ANN201
                self.invocations.append(1)
                return super().run(request=request, context=context)

        counting_tool = CountingTool(runtime.engine)
        decision_model = runtime.decision_model
        decision_model.set_decision(AgentDecision(action="recommend", k=K))
        graph = AgentGraph(decision_model, counting_tool)
        graph.run("recommend something", history.parent_asins)

        assert counting_tool.invocations == [1]
        # The counting Tool wraps the *same* accepted engine instance.
        assert counting_tool.engine is runtime.engine
