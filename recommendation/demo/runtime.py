"""Milestone 11 demo runtime: build the accepted backend once, serve many turns.

Composition path::

    browser
      -> FastAPI demo endpoints            recommendation/api/demo_routes.py
      -> DemoSessionManager                session registry, turn ids, isolation
      -> AgentGraph                        accepted M7B/M8/M9/M10A/M10B/M10D route
      -> RecommendationTool -> SASRecInferenceEngine
      -> ProductEnricher    -> MetadataIndex
      -> PreferenceCandidateMatcher -> PreferenceReranker
      -> PreferenceMemoryService -> SQLitePreferenceStore

Everything expensive is constructed exactly once per server process:

==============================  ==================================================
``SASRecInferenceEngine``       ~349 MB checkpoint, loaded once
``RecommendationTool``          thin wrapper over that one engine
``MetadataIndex``               ~300 MB catalogue artifact, loaded once
``ProductEnricher``             thin wrapper over that one index
``PreferenceMemoryService``     one SQLite store for the whole demo
``PreferenceCandidateMatcher``  stateless, one instance
``PreferenceReranker``          stateless, one instance
``DemoSessionManager``          bounded live-session registry
==============================  ==================================================

Why the compiled graph is cached per ``(k, user_key)``
------------------------------------------------------
The accepted Milestone 9 graph contract binds a memory namespace at *construction*:
``AgentGraph(memory_service=..., user_key=...)`` loads preferences for that
``user_key``.  Two demo sessions must never share preference memory, so one graph
instance cannot serve two sessions.  Rather than change the frozen Milestone 7-10
components, the runtime caches one compiled :class:`AgentGraph` per
``(k, session user_key)``.

That cache is cheap and bounded: compiling a graph is pure Python over the **same**
process-scoped collaborators (engine, Tool, enricher, memory service, matcher,
reranker), and no model, index or service is ever rebuilt.  Entries are dropped when a
session is reset, and the cache has a hard size cap as a backstop.

The other per-request value is ``k``.  The accepted trust boundary routes ``k``
exclusively through ``AgentDecision``, so it cannot be passed to the graph any other
way; a per-``k`` decision model is why ``k`` is part of the cache key.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from recommendation import config as project_config
from recommendation.agent import AgentGraph, DecisionModel
from recommendation.catalog import MetadataIndex
from recommendation.memory import (
    MEMORY_DB_ENV_VAR,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
    SQLitePreferenceStore,
)
from recommendation.preference_matching import PreferenceCandidateMatcher
from recommendation.rag import ProductEnricher
from recommendation.reranking import PreferenceReranker
from recommendation.tools import RecommendationTool

from .control_plane import (
    CONTROL_PLANE_GRAPH,
    DEFAULT_CONTROL_PLANE,
    DEFAULT_LOOP_LIMITS,
    DemoLoopRunner,
    resolve_control_plane,
)
from .decision import DEFAULT_DEMO_K, DemoDecisionModel
from .profiles import DEFAULT_DEMO_PROFILE_COUNT, DemoProfile, build_demo_profiles
from .sessions import DEFAULT_MAX_SESSIONS, DemoSessionManager

__all__ = [
    "DEFAULT_DEMO_MEMORY_DB",
    "ENV_CATALOG_METADATA_PATH",
    "DemoRuntime",
    "DemoRuntimeError",
    "build_demo_runtime",
    "catalog_metadata_path",
    "memory_database_path",
]

#: Default preference-memory database for the demo.  One database file for the whole
#: runtime (never one per session); each session owns a distinct ``user_key`` namespace
#: inside it.  Git-ignored: it is runtime data, not an artifact.
DEFAULT_DEMO_MEMORY_DB = project_config.ARTIFACTS_DIR / "demo" / "preference_memory.sqlite3"

#: Hard cap on cached compiled graphs, independent of the session cap, so a long-lived
#: process with many short sessions cannot grow the cache without limit.
MAX_CACHED_GRAPHS = 512


class DemoRuntimeError(RuntimeError):
    """The demo runtime could not be composed from the accepted artifacts."""

    code = "demo_unavailable"


#: Environment variable that points the demo at a specific catalogue-metadata artifact.
#:
#: Why this exists: the one-command demo runs against the small synthetic catalogue in
#: ``recommendation/demo/artifacts/`` so a reviewer needs no 300 MB public dataset.  The
#: accepted conventional path (``AGENTRECX_DATA_DIR`` + the category slug) is still the
#: default, so a deployment with the real artifact is unaffected; this only lets the demo
#: name a specific file.  It is read here rather than in ``recommendation.config`` because it
#: configures the *demo*, not the preprocessing pipeline's output convention.
ENV_CATALOG_METADATA_PATH = "AGENTRECX_CATALOG_METADATA_PATH"


def catalog_metadata_path() -> Path:
    """Resolve the catalogue-metadata artifact path.

    Precedence: ``AGENTRECX_CATALOG_METADATA_PATH`` (an explicit file), then the accepted
    conventional path for the configured category.
    """
    configured = os.environ.get(ENV_CATALOG_METADATA_PATH, "").strip()
    if configured:
        return Path(configured)
    return project_config.default_catalog_metadata_path()


def memory_database_path() -> Path:
    """Resolve the demo preference-memory database path.

    Precedence: ``AGENTRECX_MEMORY_DB`` (the accepted Milestone 9 environment variable),
    then the demo default.
    """
    configured = os.environ.get(MEMORY_DB_ENV_VAR)
    if configured:
        return Path(configured)
    return DEFAULT_DEMO_MEMORY_DB


@dataclass
class DemoRuntime:
    """The composed demo stack plus its bounded compiled-graph cache."""

    settings: Any
    engine: Any
    tool: RecommendationTool
    metadata: MetadataIndex
    enricher: ProductEnricher
    memory_service: PreferenceMemoryService
    matcher: PreferenceCandidateMatcher
    reranker: PreferenceReranker
    profiles: Mapping[str, DemoProfile]
    sessions: DemoSessionManager
    decision_model_factory: Callable[[int], DecisionModel] = DemoDecisionModel
    metadata_artifact: Path | None = None
    memory_artifact: Path | None = None
    max_cached_graphs: int = MAX_CACHED_GRAPHS
    #: Which control plane serves a turn: the accepted DAG (default) or the 2.0-alpha loop.
    control_plane: str = DEFAULT_CONTROL_PLANE
    #: Whether the product-knowledge explanation capability (V2 E-02B) runs for chat turns that
    #: produce recommendations.  ``None`` means "ask the deployment switch", which is off unless
    #: the operator enabled it; an explicit value overrides it, which is how both states are tested.
    explanations: bool | None = None
    #: Whether an LLM-selected run may add one extra legal source inside the same run (E-03B).
    #: ``None`` means "ask the deployment switch", which is off unless the operator enabled it.
    replanning: bool | None = None
    #: Injection point for the optional LLM decision mode's structured-model client.  ``None``
    #: (the default) builds the configured provider on first use, so a deployment that never
    #: uses the mode never constructs a provider client.  A test - or a deployment with its own
    #: transport - may supply one instead; see :meth:`use_llm_client`.
    llm_client_factory: Callable[[], Any] | None = None
    _graphs: dict[tuple[int, str], AgentGraph] = field(default_factory=dict, repr=False)
    _loops: dict[tuple[int, str], DemoLoopRunner] = field(default_factory=dict, repr=False)
    _agent_service: Any = field(default=None, repr=False)
    _llm_planner: Any = field(default=None, repr=False)
    #: The one structured-model client both policy-backed modes share.  Held separately from the
    #: planners built over it, because the tool-selection mode needs the client itself (provider
    #: facts, the planner's own seam) and rebuilding one per mode would mean two clients - and two
    #: sets of provider calls - per process.
    _llm_client: Any = field(default=None, repr=False)
    _tool_planner: Any = field(default=None, repr=False)
    _graph_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def _replanning(self) -> bool:
        """Whether bounded replanning runs this process: the explicit override, else the switch."""
        from .replan_policy import replanning_enabled

        if self.replanning is None:
            return replanning_enabled()
        return bool(self.replanning)

    @property
    def _explanations(self) -> bool:
        """Whether explanations run this process: the explicit override, else the switch."""
        from .explanation import explanations_enabled

        if self.explanations is None:
            return explanations_enabled()
        return bool(self.explanations)

    # -- readiness --------------------------------------------------------- #

    @property
    def ready(self) -> bool:
        """True when the full accepted chain is present and usable."""
        return bool(
            self.engine is not None
            and self.metadata is not None
            and self.memory_service is not None
            and self.matcher is not None
            and self.reranker is not None
            and self.profiles
        )

    @property
    def model_loaded(self) -> bool:
        """True when the injected engine reports itself ready."""
        checker = getattr(self.engine, "is_ready", None)
        return bool(checker()) if callable(checker) else self.engine is not None

    @property
    def metadata_loaded(self) -> bool:
        """True when a catalogue index is present."""
        return self.metadata is not None

    # -- graph cache ------------------------------------------------------- #

    def graph_for(self, k: int = DEFAULT_DEMO_K, *, user_key: str) -> AgentGraph:
        """Return the compiled graph for one ``(k, user_key)`` pair.

        The graph is reused across turns of the same session and across sessions that
        share a ``k``; only the memory namespace and the decision model differ, and both
        are immutable per cached instance.
        """
        cache_key = (k, user_key)
        with self._graph_lock:
            graph = self._graphs.get(cache_key)
            if graph is not None:
                return graph
        graph = AgentGraph(
            self.decision_model_factory(k),
            self.tool,
            product_enricher=self.enricher,
            memory_service=self.memory_service,
            user_key=user_key,
            preference_matcher=self.matcher,
            reranker=self.reranker,
        )
        with self._graph_lock:
            existing = self._graphs.get(cache_key)
            if existing is not None:
                return existing
            if len(self._graphs) >= self.max_cached_graphs:
                # FIFO eviction: dict preserves insertion order, so the oldest key goes.
                oldest = next(iter(self._graphs))
                self._graphs.pop(oldest, None)
            self._graphs[cache_key] = graph
            return graph

    def loop_for(self, k: int = DEFAULT_DEMO_K, *, user_key: str) -> DemoLoopRunner:
        """Return the 2.0-alpha loop runner for one ``(k, user_key)`` pair.

        Cached exactly like :meth:`graph_for`, and for the same reasons: the loop binds a
        memory namespace and a request-scoped ``k`` at construction, and compiling one is
        pure Python over the same process-scoped collaborators.  No model, index or service
        is ever rebuilt.
        """
        cache_key = (k, user_key)
        with self._graph_lock:
            existing = self._loops.get(cache_key)
            if existing is not None:
                return existing
        runner = DemoLoopRunner(
            decision_model=self.decision_model_factory(k),
            tool=self.tool,
            enricher=self.enricher,
            matcher=self.matcher,
            reranker=self.reranker,
            memory_service=self.memory_service,
            user_key=user_key,
            limits=DEFAULT_LOOP_LIMITS,
        )
        with self._graph_lock:
            duplicate = self._loops.get(cache_key)
            if duplicate is not None:
                return duplicate
            if len(self._loops) >= self.max_cached_graphs:
                oldest = next(iter(self._loops))
                self._loops.pop(oldest, None)
            self._loops[cache_key] = runner
            return runner

    def runner_for(self, k: int = DEFAULT_DEMO_K, *, user_key: str) -> Any:
        """Return the runnable for the configured control plane.

        Both runnables accept an :class:`~recommendation.agent.state.AgentInput` and return
        the accepted :class:`~recommendation.agent.state.AgentGraphState`, which is what lets
        the HTTP layer stay control-plane agnostic.
        """
        if self.control_plane == "loop":
            return self.loop_for(k, user_key=user_key)
        return self.graph_for(k, user_key=user_key)

    def turn(self, k: int, *, user_key: str, agent_input: Any) -> Any:
        """Run one turn on the configured control plane and return its state."""
        return self.runner_for(k, user_key=user_key).invoke(agent_input)

    # -- the optional LLM decision modes ------------------------------------ #

    def llm_client(self) -> Any:
        """Return the process-scoped structured-model client, building it on first use.

        One client per process for **both** policy-backed modes: the turn planner and the
        tool-selection planner are built over this same object, so selecting both modes in one
        deployment cannot double the provider footprint.  Built lazily and memoised, so a
        deployment that never selects a policy-backed mode never constructs a client.
        """
        with self._graph_lock:
            if self._llm_client is None:
                from .llm_runtime import build_llm_client

                self._llm_client = build_llm_client(self.llm_client_factory)
            return self._llm_client

    def llm_planner(self) -> Any:
        """Return the process-scoped LLM turn planner, building it on first use.

        Built lazily and memoised, so a deployment that never selects the LLM decision mode
        never constructs a provider client, and a deployment that does builds exactly one for
        the process.
        """
        with self._graph_lock:
            if self._llm_planner is not None:
                return self._llm_planner
        from .llm_policy import LLMTurnPlanner
        from .llm_runtime import LLM_MAX_ATTEMPTS

        planner = LLMTurnPlanner(self.llm_client(), max_attempts=LLM_MAX_ATTEMPTS)
        with self._graph_lock:
            if self._llm_planner is None:
                self._llm_planner = planner
            return self._llm_planner

    def tool_planner(self) -> Any:
        """Return the process-scoped validated tool-selection planner, building it on first use.

        The tool-selection mode's counterpart to :meth:`llm_planner`: same one-client rule, same
        lazy construction, so a deployment that serves only the deterministic or ``llm`` mode
        never pays for it.
        """
        with self._graph_lock:
            if self._tool_planner is not None:
                return self._tool_planner
        from .tool_selection import ToolSelectionPlanner

        planner = ToolSelectionPlanner(self.llm_client())
        with self._graph_lock:
            if self._tool_planner is None:
                self._tool_planner = planner
            return self._tool_planner

    def use_llm_client(self, client: Any, *, max_attempts: int = 2) -> None:
        """Install the structured-model client the policy-backed modes will use.

        A deployment (or a test) that owns its client supplies it here instead of letting the
        runtime build one from ``AGENTRECX_LLM_*``.  The client must satisfy the accepted
        :class:`~recommendation.control.model_client.StructuredModelClient` seam.  Both planners
        are rebuilt over it, so an injection made after either planner was used takes effect for
        both modes rather than for one.
        """
        from .llm_policy import LLMTurnPlanner

        with self._graph_lock:
            self._llm_client = client
            self._llm_planner = LLMTurnPlanner(client, max_attempts=max_attempts)
            self._tool_planner = None

    def llm_turn(self, k: int, *, user_key: str, agent_input: Any) -> tuple[Any, Any]:
        """Run one turn whose decisions come from the optional LLM policy layer.

        Returns ``(state, run)``: the accepted :class:`~recommendation.agent.AgentGraphState`
        the deterministic path also returns, plus the turn's
        :class:`~recommendation.demo.llm_runtime.LLMTurnRun` describing the validated plan - the
        only source from which the response's decision fields may be built.

        Composition, in order, and the order is the point:

        1. the plan is obtained **first**, so a provider failure happens before any trusted
           component has run and cannot leave a partial preference mutation behind;
        2. the accepted :class:`~recommendation.agent.AgentGraph` executes the turn, with the
           plan's route supplied through the accepted decision-model seam and the same plan
           supplied to the accepted Milestone 9 memory service as its extractor;
        3. everything else - retrieval, masking, scoring, matching, reranking, persistence - is
           the unchanged accepted component set, constructed from this runtime's own
           collaborators, so one process still holds one checkpoint, one catalogue and one store.

        The graph is compiled per LLM turn rather than cached per ``(k, user_key)``: the plan is
        turn-scoped and the graph binds its decision model at construction.  Compiling it is pure
        Python over the same process-scoped collaborators, so no model, index or service is
        rebuilt.
        """
        from .llm_runtime import LLMTurnRun

        run = LLMTurnRun(
            self.llm_planner(), k=k, user_message=agent_input.user_message
        )
        run.plan()
        graph = AgentGraph(
            run.decision_model(),
            self.tool,
            product_enricher=self.enricher,
            memory_service=run.memory_service(self.memory_service.store),
            user_key=user_key,
            preference_matcher=self.matcher,
            reranker=self.reranker,
        )
        return graph.invoke(agent_input), run

    def llm_agent_turn(
        self,
        k: int,
        *,
        user_key: str,
        agent_input: Any,
        session_id: str,
        turn_id: str,
        turn_number: int,
    ) -> tuple[Any, dict[str, Any]]:
        """Run one chat turn that the model plans and the bounded agent loop executes.

        Composition, in order, and the order is the point:

        1. the session's **own** ACTIVE preferences are read once, before anything else, and
           narrowed to the projection the planner may see - not the M9 snapshot, and never merged
           into the request text (see :func:`~recommendation.demo.llm_agent.planner_preference_context`);
        2. the *menu* is derived from the agent service's live ``available_sources`` and the budget
           from that service's own loop limits - nothing here decides which sources exist, so a
           not-ready TIGER is unreachable rather than refused late;
        3. the plan is obtained next, so a refused plan or a provider failure happens before any
           trusted component has run and before any preference is written;
        4. a ``recommend`` plan persists this turn's explicit preferences through the accepted M9
           write path - keyed by the session's namespace and this server-owned turn id - and then
           executes through :meth:`DemoAgentService.run_selection`, the bounded loop over the
           candidate plane, the grounding verifier and the ledger.  The write happens *before* the
           loop and the loop runs memory-free, so this turn's retrieval cannot be shaped by this
           turn's own statement (the accepted M9 ordering) and a write failure stops the turn before
           any tool runs;
        5. a ``direct`` plan starts **no** controller: that turn is answered by the accepted direct
           capability, which does its own load/extract/persist/report.  Nothing is written twice,
           because step 4 belongs to the recommendation branch only.

        Returns ``(ChatResponse, evidence)``.  The response carries everything the chat contract
        can hold truthfully - the products and their provenance, the executed route, the decision
        facts of the plan that authorised the run, and the **real** memory write summary - while
        ``evidence`` carries the rest of the run's execution record for a caller that logs or
        asserts on it.

        ``user_key`` is the session's own server-owned namespace.  It is used for memory reads and
        writes here; the *run* still receives no ``user_key``, because ranking with preferences is
        not part of this milestone and must not be faked by a side channel.
        """
        from .agent_chat_adapter import (
            agent_execution_digest,
            build_agent_chat_response,
            direct_agent_execution,
        )
        from .llm_agent import LLMAgentTurn, planner_preference_context
        from recommendation.memory.query_adapter import project_active_preferences
        from .schemas import DECISION_MODE_LLM_AGENT, MemoryUpdateView
        from .serialization import (
            TurnDecision,
            active_preference_views,
            build_chat_response,
            memory_update_view,
        )
        from .tool_selection import ROUTE_RECOMMEND

        # 1. The session's own ACTIVE preferences, once, before anything runs.  The projection is
        #    value-level (no user key, no source text, no withdrawn entry) and is the only part of
        #    memory the planner is told about.
        snapshot = self._active_preference_snapshot(user_key)
        projection = project_active_preferences(snapshot)

        service = self.agent_service()
        turn = LLMAgentTurn(
            self.tool_planner(),
            service,
            k=k,
            user_message=agent_input.user_message,
            history=tuple(getattr(agent_input, "trusted_user_history", ()) or ()),
            preference_context=planner_preference_context(projection),
        )
        selection = turn.plan()
        evidence: dict[str, Any] = {
            "decision_mode": DECISION_MODE_LLM_AGENT,
            "selected_route": selection.route,
            "selected_sources": list(selection.selected_sources),
            "offered_sources": [option.name for option in turn.menu],
            "budget": service.tool_budget,
            "provider": turn.provider,
            "model": turn.model,
            "disclosed_preferences": len(projection.entries),
        }

        if selection.route != ROUTE_RECOMMEND:
            # No tool to run.  The accepted direct capability answers the turn and reports its own
            # memory behaviour (load, extract, persist) through the state it returns; this branch
            # adds no controller, no plan and no second write.  The execution record says exactly
            # that: a direct turn, with the plan's (empty) source list and every controller-owned
            # block absent - which is what keeps it distinguishable from a recommendation that
            # could not be completed.
            evidence["executed"] = "direct_capability"
            evidence["digest"] = None
            evidence["memory_write_attempted"] = "in_graph"
            decision = TurnDecision(
                decision_mode=DECISION_MODE_LLM_AGENT,
                provider=turn.provider,
                model=turn.model,
                proposed_route=selection.route,
            )
            state = self.turn(k, user_key=user_key, agent_input=agent_input)
            # E-03C: a direct turn can still carry product feedback ("don't show me that one
            # again").  The graph owns *this* turn's ordinary preference write - the branch adds no
            # second one for preferences - so a resolved product statement is a separate M9 call
            # under the same server-owned turn id, reported in ``audit.feedback`` and merged into the
            # panel below, because a turn's memory update is the union of what it really wrote.
            from .feedback import feedback_candidate, feedback_marks, product_feedback_view

            resolution = self.resolve_product_feedback(session_id, agent_input.user_message)
            feedback = product_feedback_view(resolution)
            written_entry = None
            if resolution.resolved:
                feedback_result = self._process_turn_memory(
                    user_key=user_key,
                    user_message=agent_input.user_message,
                    turn_id=turn_id,
                    extra_candidates=(feedback_candidate(resolution),),
                    # The sentence is feedback about one product; its generic reading ("second one")
                    # must not also be stored as a feature preference.
                    suppress_text_spans=(
                        (resolution.span,) if resolution.span else ()
                    ),
                )
                if feedback_result is not None and resolution.card is not None:
                    written_entry = next(
                        (
                            entry
                            for entry in feedback_result.active.entries
                            if getattr(getattr(entry, "kind", None), "value", None) == "product"
                            and str(entry.value) == resolution.card.parent_asin
                        ),
                        None,
                    )
                    active = feedback_result.active
                    if written_entry is None and resolution.action == "retract":
                        # A retraction leaves no active entry; the store's own summary is the record.
                        active = feedback_result.active
                    # Report what the store's own summary proves: a retraction is "written" only
                    # when it really tombstoned something, and an addition only when the entry is
                    # active now.  "The user asked" and "the store changed" are different facts.
                    removed = bool(getattr(feedback_result.update, "removed", ()))
                    written = removed if resolution.action == "retract" else written_entry is not None
                    feedback = product_feedback_view(
                        resolution,
                        before=snapshot,
                        after=active,
                        written=written,
                        entry_status=(
                            "removed"
                            if removed
                            else None
                            if written_entry is None
                            else str(
                                getattr(written_entry.status, "value", written_entry.status)
                            )
                        ),
                    )
                    evidence["product_feedback"] = {
                        "status": feedback["status"],
                        "action": feedback["action"],
                        "parent_asin": feedback["parent_asin"],
                        "written": feedback["written"],
                        "changed": bool(feedback_result.update.changed),
                    }
            response = build_chat_response(
                state,
                session_id=session_id,
                turn_id=turn_id,
                turn_number=turn_number,
                decision=decision,
                agent=direct_agent_execution(selection.selected_sources),
                feedback=feedback,
                feedback_marks=feedback_marks(snapshot),
            )
            if written_entry is not None or (resolution.action == "retract" and feedback["written"]):
                # This turn wrote twice under one turn id: the graph's ordinary extraction and the
                # resolved product statement.  The panel reports the **union**, because a turn's
                # memory update is what the turn really wrote.
                response = _merge_memory_write(response, feedback_result)
            return (response, evidence)

        # 4. One write per turn, through the accepted M9 service, keyed by this session's namespace
        #    and this server-owned turn id.  A turn that states nothing relevant writes nothing and
        #    says so; a store failure raises here, before any tool has run, rather than being
        #    reported as a successful update.
        # 3b. Product-level feedback (E-03C).  The reference is bound **here**, in the trusted
        #     layer, to a card this session really presented; the model never sees or chooses an
        #     identity.  An unresolvable reference writes nothing and says so, and the sentence it
        #     came from is suppressed so the generic extractor cannot widen it into a preference
        #     about a brand, a colour or a category.
        from .feedback import feedback_marks, product_feedback_view

        resolution = self.resolve_product_feedback(session_id, agent_input.user_message)
        extra_candidates: tuple[Any, ...] = ()
        suppress: tuple[str, ...] = ()
        if resolution.resolved:
            from .feedback import feedback_candidate

            extra_candidates = (feedback_candidate(resolution),)
        elif resolution.action != "none" and resolution.span:
            # The sentence *names a product* ("the second one", an identity, "this one") but it
            # cannot be bound: it is product feedback that must not be widened into a brand, colour
            # or category preference, so its generic reading is dropped and nothing is written.
            # A statement that names no product at all ("I prefer black") keeps its ordinary M9
            # meaning - it is not product feedback, and suppressing it would silently lose it.
            suppress = (resolution.span,)

        result = self._process_turn_memory(
            user_key=user_key,
            user_message=agent_input.user_message,
            turn_id=turn_id,
            extra_candidates=extra_candidates,
            suppress_text_spans=suppress,
        )
        evidence["executed"] = "agent_loop"
        evidence["memory_write_attempted"] = "runtime" if result is not None else "none"
        # The run executes memory-free for *ranking with the query adapter* (use_memory=False), but
        # the same pre-write snapshot is handed to the trusted ranking step, which reorders this
        # run's own presented candidates with the accepted M10A/M10B components.  Memory still never
        # chooses a tool: the plan above was validated before this call.
        response = turn.run(
            preference_snapshot=snapshot,
            # E-03B: the same client the plan was obtained from, handed to the controller's policy
            # so that - only when the gate opens - it can ask for one additional legal source
            # inside this same run.  Disabled: nothing is passed and no extra call can happen.
            replan_client=self.llm_client if self._replanning else None,
        )
        evidence["digest"] = agent_execution_digest(response)
        # The ranking that actually reached the response, as execution evidence: the baseline
        # (fused) position each candidate held, the position memory gave it, and M10A's real
        # counts.  Read back from the response rather than from the ranking object, so the record
        # describes what was published and nothing else.
        evidence["ranked_with_preferences"] = len(projection.entries)
        evidence["reranked"] = sorted(
            (
                {
                    "parent_asin": candidate.parent_asin,
                    "rank": candidate.rank,
                    "reranked_rank": candidate.reranked_rank,
                    "match_count": candidate.match_count,
                    "violation_count": candidate.violation_count,
                    "unknown_count": candidate.unknown_count,
                }
                for candidate in (response.recommendations or ())
                if candidate.reranked_rank is not None
            ),
            key=lambda item: item["reranked_rank"],
        )
        # The panel and the write summary both come from the accepted M9 result: the store's own
        # post-write snapshot and its own update summary, never the model's claim about either.
        memory_update = (
            memory_update_view({"memory_update": result}) if result is not None
            else MemoryUpdateView()
        )
        # The written record is read back from the store's own snapshot, never from the model's
        # answer: ``written`` means an entry of this identity is stored, and the entry's status says
        # which lifecycle state it ended in.
        written_entry = None
        if resolution.resolved and result is not None and resolution.card is not None:
            written_entry = next(
                (
                    entry
                    for entry in (result.active.entries if hasattr(result, "active") else ())
                    if getattr(getattr(entry, "kind", None), "value", None) == "product"
                    and str(entry.value) == resolution.card.parent_asin
                ),
                None,
            )
        feedback = product_feedback_view(
            resolution,
            before=snapshot,
            after=(None if result is None else result.active),
            written=written_entry is not None,
            entry_status=(
                None
                if written_entry is None
                else str(getattr(written_entry.status, "value", written_entry.status))
            ),
        )
        preferences = active_preference_views(
            result.active if result is not None else snapshot
        )
        # What the ranking above was performed under: the pre-write read, because a turn reads
        # memory before it writes it.  The panel shows the post-write state; the audit names the
        # set the ordering actually used, and the two differ exactly when this turn stated
        # something new.
        ranked_with = active_preference_views(snapshot)
        # 5. Product knowledge retrieval and grounded explanation, on the *final* presented list -
        #    after fusion, after ranking, after the presentation limit.  It can only describe what
        #    was decided above: it holds no candidate source, and its output is validated against
        #    each product's own catalogue evidence before any of it reaches a card.
        explanations, rag = self.explain_products(
            response, user_message=agent_input.user_message, snapshot=snapshot
        )
        evidence["explanations"] = {
            "accepted": None if explanations is None else explanations.accepted,
            "rejected": None if explanations is None else explanations.rejected,
            "abstained": None if explanations is None else explanations.abstained,
            "failure": None if explanations is None else explanations.failure,
        }
        # 3c. Remember what this turn **presented**, from the response's own list: the identity map
        #     a later feedback statement resolves against is the record of what the user really saw,
        #     so a truncated, ungrounded or never-shown product cannot enter it.
        self.sessions.get(session_id).record_presentation(
            [
                {
                    "parent_asin": card.parent_asin,
                    "item_id": card.item_id,
                    "position": position,
                    "title": card.title,
                }
                for position, card in enumerate(response.recommendations, start=1)
            ],
            turn_id=turn_id,
        )
        return (
            build_agent_chat_response(
                response,
                session_id=session_id,
                turn_id=turn_id,
                turn=turn_number,
                active_preferences=preferences,
                memory_update=memory_update,
                ranked_with=ranked_with,
                explanations=explanations,
                rag=rag,
                provider=turn.provider,
                model=turn.model,
                proposed_route=selection.route,
                planned_sources=selection.selected_sources,
                feedback=feedback,
                feedback_marks=feedback_marks(snapshot),
            ),
            evidence,
        )

    def explain_products(
        self, response: Any, *, user_message: str, snapshot: Any = None
    ) -> tuple[Any, Any]:
        """Retrieve catalogue knowledge for the presented products and validate explanations.

        The capability itself, public because a caller may run it against any agent response.
        ``llm_agent_turn`` calls it exactly once per recommend turn, after the response exists and
        after its final order and limit are fixed, so what it describes is what was decided.

        Returns ``(outcome, rag_audit)``.  ``(None, None)`` means the capability is disabled, and
        the turn is then exactly the turn it was before this milestone.  Every other path returns a
        :class:`~recommendation.demo.schemas.RagAuditView` describing what happened - including
        failures.  **Nothing in here can fail the turn or change a recommendation**: retrieval,
        the model call and validation are all bounded, and any problem returns no explanations with
        an honest audit code.
        """
        from .explanation import ExplanationError, generate_explanations
        from .product_knowledge import retrieve_product_knowledge
        from .schemas import RagAuditView

        if not self._explanations:
            return None, None
        products = tuple(response.recommendations or ())
        audit = RagAuditView(enabled=True, products_examined=len(products))
        if not products:
            return None, audit
        if self.metadata is None:
            return None, audit.model_copy(update={"failure": "no_catalogue"})

        active = () if snapshot is None else tuple(getattr(snapshot, "active_entries", ()) or ())
        try:
            # Retrieval is identity-scoped to exactly these products, in exactly this order.  The
            # preference values the ranking used join the query so a product's own matching facts
            # (for example the colour that made it rank first) are the ones surfaced; the product
            # set is already fixed, so this cannot change what is recommended.
            pack = retrieve_product_knowledge(
                items=products,
                metadata=self.metadata,
                query=user_message,
                related_terms=[str(entry.value) for entry in active],
            )
            audit = audit.model_copy(
                update={
                    "products_with_evidence": pack.products_with_evidence,
                    "evidence_units": pack.unit_count,
                }
            )
            # The preference linkage a reason may mention: the accepted M10A matches of this run,
            # read from the response's own per-candidate evidence.  Not recomputed, and never
            # merged into the catalogue facts.
            matches = {
                str(candidate.parent_asin): tuple(
                    (str(record.metadata_field or ""), str(record.metadata_value or ""))
                    for record in (getattr(candidate, "evidence", ()) or ())
                    if getattr(record, "status", "") == "match"
                )
                for candidate in products
            }
            outcome = generate_explanations(
                pack=pack,
                user_request=user_message,
                client=self.llm_client(),
                active_preferences=active,
                preference_matches=matches,
            )
        except ExplanationError as exc:
            return None, audit.model_copy(update={"failure": exc.code})
        except Exception as exc:  # noqa: BLE001 - an explanation never fails a recommendation
            return None, audit.model_copy(
                update={"failure": f"unexpected:{type(exc).__name__}"}
            )
        audit = audit.model_copy(
            update={
                "model_called": True,
                "explanations_accepted": outcome.accepted,
                "explanations_rejected": outcome.rejected,
                "explanations_abstained": outcome.abstained,
                "failure": outcome.failure,
            }
        )
        if outcome.failure is not None:
            return None, audit
        return outcome, audit


    # -- preference memory, one turn at a time ----------------------------- #

    def _active_preference_snapshot(self, user_key: str) -> Any:
        """This session's ACTIVE preferences, or ``None`` when it has no namespace.

        A runtime without a memory service, or a caller with no ``user_key``, has no preferences to
        report - which is a fact to state, not a reason to invent an empty-looking snapshot.
        """
        if self.memory_service is None or not isinstance(user_key, str) or not user_key.strip():
            return None
        return self.memory_service.get_active_preferences(user_key)

    def resolve_product_feedback(self, session_id: str, message: str) -> Any:
        """Bind one message to a product this session really presented (E-03C).

        The record it resolves against is the session's own last recommendation turn, built from the
        response's own cards - so a truncated, ungrounded or never-shown product cannot be named, and
        another session's history is unreachable by construction.
        """
        from .feedback import PresentedCard, resolve_reference

        demo_session = self.sessions.get(session_id)
        presented = [
            PresentedCard(
                parent_asin=str(card.get("parent_asin", "")),
                item_id=int(card.get("item_id", 0) or 0),
                position=int(card.get("position", 0) or 0),
                title=card.get("title"),
            )
            for card in (demo_session.last_presentation or ())
        ]
        return resolve_reference(message, presented)

    def _process_turn_memory(
        self,
        *,
        user_key: str,
        user_message: str,
        turn_id: str,
        extra_candidates: Any = (),
        suppress_text_spans: Any = (),
    ) -> Any:
        """Persist one turn's explicit preferences through the accepted M9 write path.

        Returns the M9 :class:`~recommendation.memory.service.TurnMemoryResult` (its ``update`` is
        what the response reports and its ``active`` is what the panel shows), or ``None`` when
        this session has no memory namespace and therefore nothing to write.

        Raises
        ------
        Whatever the accepted service raises.  A failed write must surface as a failed turn: the
        chat contract has no way to say "a write was attempted and did not happen", and reporting
        ``changed=False`` for it would be a false statement about memory.
        """
        if self.memory_service is None or not isinstance(user_key, str) or not user_key.strip():
            return None
        return self.memory_service.process_turn(
            user_key=user_key,
            user_message=user_message,
            turn_id=turn_id,
            # E-03C: a statement the trusted layer resolved to a real presented product, and the
            # span whose generic reading it overrides - so "I don't like the black shoes you showed
            # me" stores feedback about that product and does not become a dislike of black shoes.
            extra_candidates=tuple(extra_candidates),
            suppress_text_spans=tuple(suppress_text_spans),
        )

    def agent_service(self) -> Any:
        """Return the single-turn agent service, composing it once on first use.

        Compensating an extra entry point must not mean a second copy of anything: the
        service is handed the **same** engine, tool, catalogue, matcher, reranker and memory
        service this runtime already owns, so one process still holds one checkpoint and one
        store.  It is built lazily because a deployment that only serves the session API
        should not pay for the full-catalogue search index.
        """
        with self._graph_lock:
            if self._agent_service is None:
                from .agent_service import DemoAgentService

                # ``from_env`` is what applies the two explicit opt-ins
                # (AGENTRECX_AGENT_SIMILAR_ITEMS, AGENTRECX_AGENT_POLICY).  Both default to
                # off, so the composed service is offline unless the deployment asked
                # otherwise.
                self._agent_service = DemoAgentService.from_env(
                    tool=self.tool,
                    enricher=self.enricher,
                    metadata=self.metadata,
                    matcher=self.matcher,
                    reranker=self.reranker,
                    engine=self.engine,
                    memory_service=self.memory_service,
                    # E-03B's opt-in travels with the runtime: the deployment switch (or the
                    # explicit construction argument) decides whether an LLM-selected run may add
                    # one extra source.  The service never learns about models itself.
                    replanning=self._replanning,
                )
            return self._agent_service

    def release_user_key(self, user_key: str) -> int:
        """Drop every cached graph and loop bound to a namespace; returns how many."""
        with self._graph_lock:
            stale = [key for key in self._graphs if key[1] == user_key]
            for key in stale:
                self._graphs.pop(key, None)
            stale_loops = [key for key in self._loops if key[1] == user_key]
            for key in stale_loops:
                self._loops.pop(key, None)
            return len(stale) + len(stale_loops)

    @property
    def compiled_graph_count(self) -> int:
        """How many ``(k, user_key)`` pairs currently have a compiled graph."""
        with self._graph_lock:
            return len(self._graphs)

    # -- diagnostics ------------------------------------------------------- #

    def build_report(self) -> dict[str, Any]:
        """JSON-serialisable identity of the composed runtime.

        Reports *counts* (each heavy object exists once) and never a filesystem path that
        a client could use to locate the host's data.
        """
        return {
            "engine_builds": 1,
            "tool_builds": 1,
            "metadata_loads": 1,
            "enricher_builds": 1,
            "memory_service_builds": 1,
            "matcher_builds": 1,
            "reranker_builds": 1,
            "profile_count": len(self.profiles),
            "max_sessions": self.sessions.max_sessions,
            "profile_ids": tuple(self.profiles),
            "metadata_records": getattr(self.metadata, "size", None),
            "compiled_graph_count": self.compiled_graph_count,
            "control_plane": self.control_plane,
            "agent_service_composed": self._agent_service is not None,
            "agent_service_sources": (
                list(self._agent_service.available_sources)
                if self._agent_service is not None
                else []
            ),
            "agent_policy": (
                self._agent_service.policy_name if self._agent_service is not None else None
            ),
        }

    def close(self) -> None:
        """Close the preference store, if it owns one, and release the agent cache."""
        if self._agent_service is not None:
            closer = getattr(self._agent_service, "close", None)
            if callable(closer):
                closer()
        closer = getattr(getattr(self.memory_service, "store", None), "close", None)
        if callable(closer):
            closer()

    # -- composition ------------------------------------------------------- #

    @classmethod
    def from_env(
        cls,
        *,
        engine: Any = None,
        metadata: MetadataIndex | None = None,
        memory_service: PreferenceMemoryService | None = None,
        store: Any = None,
        profiles: Mapping[str, DemoProfile] | None = None,
        profile_count: int = DEFAULT_DEMO_PROFILE_COUNT,
        decision_model_factory: Callable[[int], DecisionModel] = DemoDecisionModel,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
        memory_db: Path | None = None,
        control_plane: str | None = None,
    ) -> DemoRuntime:
        """Compose the demo runtime from ``AGENTRECX_*`` configuration.

        See :func:`build_demo_runtime`, which this delegates to.
        """
        return build_demo_runtime(
            engine=engine,
            metadata=metadata,
            memory_service=memory_service,
            store=store,
            profiles=profiles,
            profile_count=profile_count,
            decision_model_factory=decision_model_factory,
            max_sessions=max_sessions,
            ttl_seconds=ttl_seconds,
            clock=clock,
            memory_db=memory_db,
            control_plane=control_plane,
        )


def build_demo_runtime(
    *,
    settings: Any = None,
    engine: Any = None,
    metadata: MetadataIndex | None = None,
    memory_service: PreferenceMemoryService | None = None,
    store: Any = None,
    profiles: Mapping[str, DemoProfile] | None = None,
    profile_count: int = DEFAULT_DEMO_PROFILE_COUNT,
    decision_model_factory: Callable[[int], DecisionModel] = DemoDecisionModel,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    ttl_seconds: float | None = None,
    clock: Callable[[], float] = time.time,
    memory_db: Path | None = None,
    control_plane: str | None = None,
    explanations: bool | None = None,
    replanning: bool | None = None,
) -> DemoRuntime:
    """Compose the demo runtime, constructing heavy objects once.

    Every collaborator may be injected, which is how the offline tests and the formal
    smoke run against a small synthetic checkpoint and a synthetic catalogue instead of
    the accepted 349 MB / 300 MB artifacts.  Injection is explicit: there is no hidden
    default engine and no global.

    Raises
    ------
    DemoRuntimeError
        A required accepted artifact is missing or unusable.  Failing here is deliberate:
        :mod:`recommendation.api.app` calls this during startup, so the server refuses to
        start rather than failing on the first browser request.
    """
    from recommendation.api.app import ServiceSettings  # local import: avoids a cycle

    settings = settings if settings is not None else ServiceSettings.from_env()

    resolved_engine = engine
    if resolved_engine is None:
        try:
            # Reuse the accepted Milestone 6 construction path (including its
            # checkpoint-digest verification switch) instead of duplicating model loading.
            from recommendation.api.app import build_engine

            resolved_engine, error = build_engine(settings)
        except Exception as exc:  # noqa: BLE001 - normalize to a demo-level failure
            raise DemoRuntimeError(
                f"the accepted checkpoint could not be loaded: {type(exc).__name__}"
            ) from exc
        if resolved_engine is None:
            raise DemoRuntimeError(f"the accepted checkpoint could not be loaded: {error}")

    artifact: Path | None = None
    resolved_metadata = metadata
    if resolved_metadata is None:
        artifact = catalog_metadata_path()
        if not artifact.exists():
            raise DemoRuntimeError(
                "the catalogue metadata artifact is missing; build it with "
                "`.venv/bin/python -m experiments.prepare_product_metadata`"
            )
        resolved_metadata = MetadataIndex.load(artifact)

    resolved_memory = memory_service
    resolved_db: Path | None = memory_db
    if resolved_memory is None:
        if store is None:
            resolved_db = Path(memory_db) if memory_db is not None else memory_database_path()
            resolved_db.parent.mkdir(parents=True, exist_ok=True)
            store = SQLitePreferenceStore(resolved_db)
        resolved_memory = PreferenceMemoryService(store, RuleBasedPreferenceExtractor())

    resolved_profiles = profiles if profiles is not None else build_demo_profiles(count=profile_count)

    return DemoRuntime(
        settings=settings,
        engine=resolved_engine,
        tool=RecommendationTool(resolved_engine),
        metadata=resolved_metadata,
        enricher=ProductEnricher(resolved_metadata),
        memory_service=resolved_memory,
        matcher=PreferenceCandidateMatcher(),
        reranker=PreferenceReranker(),
        profiles=resolved_profiles,
        sessions=DemoSessionManager(
            resolved_profiles, max_sessions=max_sessions, ttl_seconds=ttl_seconds, clock=clock
        ),
        decision_model_factory=decision_model_factory,
        metadata_artifact=artifact,
        memory_artifact=resolved_db,
        control_plane=resolve_control_plane(control_plane),
        explanations=explanations,
        replanning=replanning,
    )


def _merge_memory_write(response: Any, result: Any) -> Any:
    """Fold one M9 write's summary into a response built from another write of the same turn.

    Used only by the direct branch of :meth:`DemoRuntime.llm_agent_turn`, where the accepted graph
    owns the ordinary preference write and the trusted product-feedback write is a second M9 call
    under the same server-owned turn id.  Both summaries come from the store; nothing is inferred.
    """
    from .schemas import ActivePreferenceView, MemoryUpdateView, PreferenceMutationView
    from .serialization import active_preference_views

    def mutation(entry: Any) -> Any:
        return PreferenceMutationView(
            kind=str(getattr(getattr(entry, "kind", None), "value", entry.kind)),
            polarity=str(getattr(getattr(entry, "polarity", None), "value", entry.polarity)),
            value=str(entry.value),
        )

    update = response.memory_update
    summary = result.update
    merged = MemoryUpdateView(
        changed=bool(update.changed or summary.changed),
        added=tuple(update.added) + tuple(mutation(entry) for entry in summary.added),
        superseded=tuple(update.superseded)
        + tuple(mutation(entry) for entry in summary.superseded),
        removed=tuple(update.removed) + tuple(mutation(entry) for entry in summary.removed),
        skipped_duplicates=int(update.skipped_duplicates) + int(summary.skipped_duplicates),
        already_processed=bool(update.already_processed),
        removal_directives=int(update.removal_directives) + int(summary.removal_directives),
    )
    # Keyed by the fact itself, not by any store handle: the response contract publishes no entry
    # identifier, and two views that say the same thing are the same fact.
    preferences = {
        (view.kind, view.polarity, view.value): view for view in response.active_preferences
    }
    for view in active_preference_views(result.active):
        checked = ActivePreferenceView.model_validate(view)
        preferences.setdefault((checked.kind, checked.polarity, checked.value), checked)
    return response.model_copy(
        update={
            "memory_update": merged,
            "active_preferences": tuple(preferences.values()),
        }
    )
