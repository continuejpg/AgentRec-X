"""Single-turn agent service for the packaging-phase ``/recommend`` endpoint.

What it composes, and what it reuses
------------------------------------
One request runs the **existing** bounded agent loop
(:class:`~recommendation.control.LoopController`) over the **existing** accepted pipeline.
Nothing here is a new agent: the loop, the multi-source candidate plane, the grounding
verifier, the ledger, the rank fusion, the constraint eligibility evaluator, the completion
guard, the renderer and the recommendation tool are all the accepted components, constructed
exactly as the accepted evaluation harness constructs them.

The one thing this module adds is *composition*: a production entry point that wires those
parts together for a request whose history and constraints arrive over HTTP instead of from a
test fixture.

Why the candidate plane is wired here
-------------------------------------
The demo's session runtime drives the history-capability path, which records no provenance
because a single history call has no multi-source ledger.  This service builds the
:class:`~recommendation.control.CandidatePlane` so a request can name more than one trusted
source and receive per-candidate provenance.  That is the capability Phase 4 built and
Phase 5 measured; exposing it is the point of this phase.  The sources consulted are the
ones the **request** named - see :mod:`recommendation.demo.agent_policy` for why this is
deliberately not an adaptive selector.

No session, no hidden state
---------------------------
Each request composes its **own** controller over the process-scoped collaborators - the engine,
the catalogue, the lexical search index, the matcher, the reranker and the memory store are built
once and shared, while the controller, its policy and its candidate plane (with the ledger that
records provenance) live for exactly one request.  Nothing that belongs to a run outlives it, so
two identical requests cannot influence each other and two concurrent requests share no mutable
run state.

That per-request composition is deliberate and costs nothing measurable: the expensive objects are
the shared ones, and the loop compiles its driver graph per run regardless.  Preference memory is
touched **only** when the request supplies a ``user_key``; without one the run reads and writes no
memory at all, which is what makes the offline mode reproducible.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Mapping, Sequence

from recommendation.agent import AgentInput
from recommendation.agent.rendering import build_final_list_response
from recommendation.control import (
    CandidateEligibilityEvaluator,
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    GroundedReasoner,
    GroundingVerifier,
    LoopController,
    LoopLimits,
    RecommendFromHistoryCapability,
)
from recommendation.memory import PreferenceKind
from recommendation.control.task_state import TaskConstraint, TaskState

from .agent_policy import DEFAULT_SOURCE_PLAN, DemoSourcePlanPolicy, normalise_source_plan
from .replan_policy import BoundedReplanPolicy
from .agent_schemas import (
    AgentRecommendRequest,
    AgentRecommendResponse,
    AgentTimingView,
)
from .agent_trajectory import (
    build_eligibility_view,
    build_grounded_view,
    build_recommendations_view,
    build_terminal_view,
    build_trajectory_summary,
)
from .control_plane import CONTROL_PLANE_LOOP

#: The process-wide TIGER engine accessor.  Imported at module scope because
#: ``DemoAgentService.__init__`` calls it: while this import lived only inside
#: ``_build_tiger_engine`` the constructor raised ``NameError`` on every Agent request and the
#: route reported a generic 502 (the T-02.7C failure).
from recommendation.inference.tiger_engine import shared_engine  # noqa: E402

__all__ = [
    "AGENT_LOOP_LIMITS",
    "ENV_MEMORY_RANKING",
    "ENV_SIMILAR_ITEMS",
    "ENV_TIGER",
    "ENV_TIGER_CHECKPOINT",
    "AGENT_POLICY_LLM",
    "DemoAgentError",
    "DemoAgentService",
    "ENV_AGENT_POLICY",
    "llm_policy_selected",
]

#: Opt-in switch for the real-provider agent policy.
#:
#: Off by default, and deliberately a *separate* switch from the provider configuration: a
#: deployment may have ``AGENTRECX_LLM_*`` set for other reasons, and a packaging phase must
#: not silently start making billed network calls because credentials happen to be present.
#: Set ``AGENTRECX_AGENT_POLICY=llm`` to use the accepted :class:`LLMAgentPolicy` for this
#: endpoint's source decisions.  Unset or ``deterministic`` keeps the request-driven
#: :class:`~recommendation.demo.agent_policy.DemoSourcePlanPolicy`, which needs no network.
ENV_AGENT_POLICY = "AGENTRECX_AGENT_POLICY"

#: The policy name that opts into the real model.
AGENT_POLICY_LLM = "llm"

#: Switch for memory-aware ranking of an LLM-selected run (E-02A).
#:
#: On by default, because it can only act when the caller supplies a non-empty read of the
#: session's memory: a deployment with no memory, no ACTIVE preference, or no presented candidates
#: behaves exactly as it did before this capability existed.  Set it to ``0``/``false``/``off`` to
#: pin the pre-E-02A baseline - the fused order then always stands, whatever memory holds.
ENV_MEMORY_RANKING = "AGENTRECX_AGENT_MEMORY_RANKING"

#: Opt-in switch for the Two-Tower candidate source (post-Phase-5 model expansion).
#:
#: Off by default, and separate from the model's presence on disk: a deployment that has
#: trained a Two-Tower checkpoint must still say that it wants this run's candidate plane to
#: consult it, so which sources a demo consults stays an explicit deployment decision.
ENV_TWO_TOWER = "AGENTRECX_TWO_TOWER_CHECKPOINT"

#: The id mapping that checkpoint was trained against.  Separate from ``AGENTRECX_MAPPINGS_PATH``
#: because a Two-Tower model and the served demo engine can legitimately use different
#: catalogues (the public benchmark catalogue versus the demo's small synthetic one), and
#: reusing one mapping for both would either fail the cardinality check or silently resolve
#: identities against the wrong catalogue.  Falls back to ``AGENTRECX_MAPPINGS_PATH`` when the
#: two agree, which is the normal single-catalogue deployment.
ENV_TWO_TOWER_MAPPINGS = "AGENTRECX_TWO_TOWER_MAPPINGS_PATH"

#: Opt-in switch for real item-item retrieval.
#:
#: Building the TF-IDF index over the full 156 746-product catalogue was measured on this
#: project at **6.21 GB peak RSS and 18.8 s**, with a further ~2.6 s per neighbour query.
#: The demo host has 7 GB, so it is off by default and the deployment reports only the
#: sources it actually runs.  Set ``AGENTRECX_AGENT_SIMILAR_ITEMS=1`` on a machine with the
#: headroom to offer ``find_similar``.
ENV_SIMILAR_ITEMS = "AGENTRECX_AGENT_SIMILAR_ITEMS"
#: Opt-in for the TIGER generative source.  Any non-empty value enables it; the worker is a
#: long-lived subprocess that owns its own virtual environment.
ENV_TIGER = "AGENTRECX_TIGER"
#: Optional override for the accepted checkpoint directory (defaults to the frozen path).
ENV_TIGER_CHECKPOINT = "AGENTRECX_TIGER_CHECKPOINT"

#: Loop budgets for one HTTP request.  Sized for the longest plan this endpoint accepts -
#: one step per named source plus the terminal finish - with the same hard bounds as the
#: accepted demo loop, so an agent cannot turn one request into unbounded work.
AGENT_LOOP_LIMITS = LoopLimits(max_steps=6, max_tool_calls=4, max_retries=1)

#: The constraint dimensions a request may state.  This is the accepted preference ontology
#: (:class:`~recommendation.memory.PreferenceKind`), not a second vocabulary: the control
#: plane narrows a preference kind onto a checkable constraint dimension itself, and an
#: unmappable dimension stays active as ``UNVERIFIABLE`` rather than being dropped.
CONSTRAINT_KINDS: frozenset[str] = frozenset(
    kind.value for kind in PreferenceKind if kind is not PreferenceKind.FREE_FORM_CONSTRAINT
)


class DemoAgentError(RuntimeError):
    """The agent service could not serve a request.  Carries a stable code."""

    def __init__(self, detail: str, *, code: str = "agent_unavailable") -> None:
        super().__init__(detail)
        self.code = code


class DemoAgentService:
    """Run one bounded agent turn per request over the accepted pipeline.

    Parameters
    ----------
    tool, enricher, metadata, matcher, reranker, engine, memory_service:
        The process-scoped accepted collaborators, owned by
        :class:`~recommendation.demo.DemoRuntime`.  Injected rather than re-built, so one
        server process holds exactly one checkpoint, one catalogue and one store.
    similar_item_tool:
        Optional item-item retrieval tool, registered only when the deployment opted in (see
        :data:`ENV_SIMILAR_ITEMS`).  Building the real TF-IDF index over the full
        156 746-product catalogue was measured at 6.21 GB peak RSS and 18.8 s on this project,
        which a 7 GB demo host cannot afford; when it is absent ``find_similar`` is simply not
        offered, and the deployment reports the sources it actually runs rather than
        advertising one it cannot serve.
    """

    control_plane = CONTROL_PLANE_LOOP

    def __init__(
        self,
        *,
        tool: Any,
        enricher: Any,
        metadata: Any,
        matcher: Any,
        reranker: Any,
        engine: Any,
        memory_service: Any = None,
        similar_item_tool: Any = None,
        two_tower_tool: Any = None,
        tiger_engine: Any = None,
        tiger_enabled: bool = False,
        policy_factory: Any = None,
        limits: LoopLimits = AGENT_LOOP_LIMITS,
        memory_ranking: bool | None = None,
        replanning: bool | None = None,
    ) -> None:
        self._tool = tool
        self._enricher = enricher
        self._metadata = metadata
        self._matcher = matcher
        self._reranker = reranker
        self._engine = engine
        self._memory_service = memory_service
        self._similar_item_tool = similar_item_tool
        self._two_tower_tool = two_tower_tool
        #: The TIGER source is opt-in and gated on readiness: an engine that has not finished its
        #: (minutes-long) cold start is neither advertised nor registered in the plane, so no
        #: action can be offered that would then block on a model load.
        self._tiger_engine: Any = None
        #: True only when *this* service created the engine.  A shared engine borrowed from the
        #: serving fusion must survive this service being closed.
        self._owns_tiger_engine: bool = False
        self._tiger_tool: Any = None
        self._tiger_error: str | None = None
        self._tiger_state: str = "disabled"
        _adopted = None
        if tiger_engine is None and tiger_enabled:
            _adopted = shared_engine()
            tiger_engine = _build_tiger_engine(engine)
        if tiger_engine is not None:
            from recommendation.control.tiger_source import TigerSourceTool

            self._owns_tiger_engine = tiger_engine is not _adopted
            self._tiger_engine = tiger_engine
            self._tiger_tool = TigerSourceTool(tiger_engine)
            self._tiger_state = "ready" if getattr(tiger_engine, "started", False) else "not_started"
        #: Builds the run's policy.  Defaults to the deterministic, request-driven source plan;
        #: a deployment may inject the accepted model-driven policy instead.  Injection rather
        #: than a branch inside the loop is what keeps this endpoint's policy interchangeable.
        self._policy_factory = policy_factory or (
            lambda plan, k: DemoSourcePlanPolicy(plan, default_k=k)
        )
        self._limits = limits
        #: Whether a caller-supplied preference snapshot may reorder the presented candidates
        #: (E-02A).  ``None`` means "read the deployment switch"; a deployment that never supplies
        #: memory is unaffected either way.
        self._memory_ranking_enabled = memory_ranking_enabled(memory_ranking)
        #: Whether an LLM-selected run may add **one** extra legal source inside the same run,
        #: chosen by the model, inside the run's own step and tool budgets (E-03B).  Off unless the
        #: deployment asks for it; the request-declared path never uses it.
        from .replan_policy import replanning_enabled

        self._replanning_enabled = replanning_enabled(replanning)
        #: Diagnostics only: how many controllers this service has composed.  There is no
        #: controller cache to bound (see :meth:`_compose_controller`).
        self._controllers_built = 0
        self._lock = threading.Lock()
        #: Built once, because the catalogue search index is a per-process cost, not a
        #: per-request one.
        self._catalog_search = CatalogSearchSource(metadata)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DemoAgentService":
        """Build a service from ``AGENTRECX_*`` configuration.

        Two opt-ins are honoured, both off by default: :data:`ENV_SIMILAR_ITEMS` registers real
        full-catalogue item-item retrieval, and :data:`ENV_AGENT_POLICY=llm` uses the accepted
        model-driven policy.  A deployment with provider credentials but no ``llm`` policy
        setting stays fully offline - the switch is explicit so no billed call can start by
        accident.
        """
        if kwargs.get("similar_item_tool") is None and _similar_items_enabled():
            metadata = kwargs.get("metadata")
            if metadata is not None:
                from recommendation.control import build_similar_item_index

                from recommendation.control.similar_item import SimilarItemSource

                kwargs["similar_item_tool"] = SimilarItemSource(
                    build_similar_item_index(metadata)
                )
        if kwargs.get("two_tower_tool") is None and _two_tower_enabled():
            kwargs["two_tower_tool"] = _build_two_tower_tool()
        if kwargs.get("policy_factory") is None and llm_policy_selected():
            kwargs["policy_factory"] = _build_llm_policy_factory()
        kwargs.setdefault("tiger_enabled", _tiger_enabled())
        return cls(**kwargs)

    # -- metadata ---------------------------------------------------------- #

    @property
    def control_plane_name(self) -> str:
        """The control plane this service serves (recorded in every response)."""
        return CONTROL_PLANE_LOOP

    @property
    def policy_name(self) -> str:
        """The policy this service builds for a run (``deterministic`` or the model's name).

        Reported so a reviewer can tell which decision source served a deployment without
        having to inspect the response's trajectory.
        """
        return "model" if llm_policy_selected() else "deterministic"

    @property
    def available_sources(self) -> tuple[str, ...]:
        """The trusted candidate sources this deployment can actually consult.

        This is the *reported* set, and it is reported honestly: a source appears only when its
        tool is registered.  It can differ from what a policy is **offered**, which is the
        controller's decision and is derived from the plane's registered tools - see
        :attr:`consultable_note` for the one case where those differ today.
        """
        sources = ["history", "catalog_search"]
        if self._similar_item_tool is not None:
            sources.append("similar_item")
        if self._two_tower_tool is not None:
            sources.append("two_tower")
        # Advertised only when the engine is warm: the deployment must never claim a source it
        # cannot serve without a multi-minute model load.
        if self._tiger_tool is not None and self._tiger_state_now() == "ready":
            sources.append("tiger")
        return tuple(sources)

    @property
    def tool_budget(self) -> int:
        """How many tool calls one run of this service may execute.

        The controller's own bound (``LoopLimits.max_tool_calls``), read from this instance rather
        than assumed from the module default, because the limits are injectable.  A caller that
        offers this many tools - the tool-selection planner is the one that does - offers exactly
        what the loop is allowed to run, so a plan can never be larger than the run it authorises.
        """
        return int(self._limits.max_tool_calls)

    @property
    def consultable_note(self) -> str:
        """A one-line statement of which actions the controller currently offers."""
        offered = ["history", "catalog_search"]
        if self._similar_item_tool is not None:
            offered.append("similar_item")
        if self._two_tower_tool is not None:
            offered.append("two_tower")
        if self._tiger_state_now() == "ready":
            offered.append("tiger")
        return (
            "the controller offers SELECT_SOURCE for every registered plane source: "
            + ", ".join(offered)
        )

    def close(self) -> None:
        """Release what this service owns; calling it twice is harmless.

        There is no controller cache to drop: each request composes and then releases its own
        controller, whose policy, plane and ledger die with the request.  What is left to release
        is the TIGER worker this service started, and only when it started it - a shared engine
        borrowed from the serving fusion outlives this service.
        """
        engine, self._tiger_engine = self._tiger_engine, None
        self._tiger_tool = None
        if engine is not None and self._owns_tiger_engine:
            # Only an engine this service built is ours to stop: the serving fusion may still be
            # using a shared worker.
            try:
                engine.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
            self._tiger_state = "not_started"
        self._owns_tiger_engine = False

    def _tiger_state_now(self) -> str:
        """TIGER's state, read from the **engine** when one is present.

        The engine is shared with the serving fusion (one worker per process), so readiness must
        be whatever that engine reports - not a private copy of the state that no one updates.
        """
        if self._tiger_engine is None:
            return "disabled"
        health = getattr(self._tiger_engine, "health", None)
        if callable(health):
            try:
                return str(health().get("state") or self._tiger_state)
            except Exception:  # noqa: BLE001 - a broken engine reports its own failure
                return "failed"
        return self._tiger_state

    @property
    def tiger_status(self) -> dict[str, Any]:
        """TIGER's lifecycle state, so health can distinguish the four outcomes.

        ``disabled`` (not configured), ``not_started`` (configured, worker not launched),
        ``loading`` (prewarm in flight), ``ready`` (worker handshake completed), ``failed``
        (the worker refused to start or died - ``error`` carries the reason).
        """
        health = {}
        probe = getattr(self._tiger_engine, "health", None)
        if callable(probe):
            try:
                health = dict(probe())
            except Exception:  # noqa: BLE001
                health = {}
        return {
            "state": self._tiger_state_now(),
            "enabled": self._tiger_engine is not None,
            "registered": self._tiger_tool is not None,
            "advertised": "tiger" in self.available_sources,
            "error": health.get("error", self._tiger_error),
            "device": getattr(getattr(self._tiger_engine, "config", None), "device", None),
            "checkpoint_sha256": health.get("checkpoint_sha256"),
            "sid_artifact_sha256": health.get("sid_artifact_sha256"),
            "retrieval_mode": health.get("retrieval_mode"),
        }

    def prewarm_tiger(self, *, background: bool = True) -> dict[str, Any]:
        """Start the TIGER worker (cold start is minutes on CPU) and return the new status.

        ``background=True`` returns immediately with ``state="loading"``, which is what a
        deployment should do at boot; ``background=False`` blocks until the worker is ready or
        has failed.  An unconfigured deployment reports ``disabled`` and does nothing.
        """
        if self._tiger_engine is None:
            return self.tiger_status
        state = self._tiger_state_now()
        if state in {"ready", "loading"}:
            return self.tiger_status
        engine_prewarm = getattr(self._tiger_engine, "prewarm", None)
        if callable(engine_prewarm):
            # The engine owns the worker; delegating keeps one state machine, not two.
            engine_prewarm(background=background)
            return self.tiger_status

        def _load() -> None:
            try:
                self._tiger_engine.start()
                self._tiger_state = "ready"
                self._tiger_error = None
            except Exception as exc:  # noqa: BLE001 - reported through tiger_status
                self._tiger_state = "failed"
                self._tiger_error = f"{type(exc).__name__}: {exc}"

        self._tiger_state = "loading"
        if background:
            threading.Thread(target=_load, name="tiger-prewarm", daemon=True).start()
        else:
            _load()
        return self.tiger_status

    @property
    def controllers_built(self) -> int:
        """How many controllers this service has composed (monotonic; diagnostics only).

        Monotonic rather than a cache size on purpose: a controller is composed for exactly one
        request (see :meth:`_compose_controller`), so "currently cached" is not a fact this service
        has to report, while "how many were built" answers the question a reviewer actually asks -
        including whether a direct turn composed none.
        """
        with self._lock:
            return self._controllers_built

    # -- one controller per request ---------------------------------------- #

    def _compose_controller(
        self,
        *,
        use_memory: bool = True,
        k: int,
        user_key: str | None,
        plan: Sequence[str],
        replan_client: Any = None,
        preference_snapshot: Any = None,
    ) -> LoopController:
        """Compose a controller for **one** request.

        Deliberately not cached.  A controller owns three things that are per-run state, not
        process state:

        * the policy instance, which remembers which planned sources it already proposed (see
          :meth:`~recommendation.demo.agent_policy.DemoSourcePlanPolicy.choose`);
        * the :class:`~recommendation.control.CandidatePlane`'s ``CandidateLedger``, whose entries
          are read back as a run's grounded universe - shared across requests, a second request
          reports candidates it never retrieved and can decide it has already served a source;
        * the plane itself, which the loop consults to decide what it may offer.

        Composing one per request makes all three request-scoped, which is what makes two identical
        requests independent and what makes concurrent requests safe with no extra locking: they no
        longer share anything mutable.  It is cheap - the previous cache saved only attribute
        assignments, because everything expensive (engine, catalogue, lexical index, matcher,
        reranker, memory store) is owned by this service and injected, and the loop compiles its
        driver graph per run either way (``LoopController._run_graph_driver``).
        """
        with self._lock:
            self._controllers_built += 1
        return self._build_controller(
            k=k,
            user_key=user_key,
            plan=plan,
            use_memory=use_memory,
            replan_client=replan_client,
            preference_snapshot=preference_snapshot,
        )

    def _build_controller(
        self,
        *,
        use_memory: bool = True,
        k: int,
        user_key: str | None,
        plan: Sequence[str],
        replan_client: Any = None,
        preference_snapshot: Any = None,
    ) -> LoopController:
        """Compose one controller: capability, candidate plane, eligibility, then the loop.

        The order matters.  The eligibility evaluator is built over the **same** catalogue
        the plane grounds against, and the plane's ledger is created here, with the controller,
        so provenance is scoped to the one request this controller serves - a ledger that
        outlived its request would let the next one report candidates it never retrieved.
        """
        capability = RecommendFromHistoryCapability(
            self._tool,
            product_enricher=self._enricher,
            preference_matcher=self._matcher,
            preference_reranker=self._reranker,
        )
        plane = CandidatePlane(
            ledger=CandidateLedger(),
            grounding=GroundingVerifier(self._engine, self._metadata),
            # The history recommender is registered as a *plane source* as well as behind the
            # Stage 1 capability.  Without it the plane cannot serve SELECT_SOURCE{history},
            # and a run that names the history source would be refused even though the
            # deployment plainly has it.  Registration does not change the capability path:
            # the accepted RECOMMEND_FROM_HISTORY action still executes through the capability,
            # exactly as before.
            history_tool=self._tool,
            catalog_search=self._catalog_search,
            similar_item_tool=self._similar_item_tool,
            two_tower_tool=self._two_tower_tool,
            # Registered only once the worker is warm: the menu is derived from the plane, so
            # this is what keeps SELECT_SOURCE{tiger} from being offered before it can run.
            tiger_tool=self._tiger_tool if self._tiger_state_now() == "ready" else None,
        )
        eligibility = CandidateEligibilityEvaluator(GroundedReasoner(self._metadata))
        # Phase 2B: the per-source budget is a separate number from the final ``k``.
        # Unconfigured it resolves to ``k``, which is exactly the accepted behaviour.
        from .source_limit import resolve_source_limit

        source_limit = resolve_source_limit(k)
        # An LLM-selected run pins the isolated policy directly: the configured factory belongs to
        # the legacy path and must keep its own behaviour.
        if use_memory:
            policy = self._policy_factory(plan, source_limit)
        elif self._replanning_enabled and replan_client is not None:
            # The LLM-selected path, with bounded replanning: the plan policy still serves the
            # authorised plan, and at most one extra legal source may be appended inside this same
            # run, on this run's budgets (E-03B).  The request-declared path keeps its own factory.
            from recommendation.memory.query_adapter import project_active_preferences

            policy = BoundedReplanPolicy(
                plan,
                default_k=source_limit,
                use_memory=False,
                client=replan_client,
                # The source choice may see the user's ACTIVE preferences - value level only, the
                # same projection a policy is entitled to - because the loop this path composes
                # deliberately binds no memory of its own.
                preferences=project_active_preferences(preference_snapshot),
            )
        else:
            policy = DemoSourcePlanPolicy(plan, default_k=source_limit, use_memory=False)
        return LoopController(
            policy,
            capability,
            memory_service=self._memory_service if user_key else None,
            user_key=user_key,
            limits=self._limits,
            candidate_plane=plane,
            eligibility_evaluator=eligibility,
        )

    # -- one request ------------------------------------------------------- #

    def run(self, payload: AgentRecommendRequest) -> AgentRecommendResponse:
        """Serve one request: validate the plan, run the loop, project the result.

        Raises
        ------
        DemoAgentError
            The request named a source this deployment cannot consult, or the loop could not
            start.  Mapping a plan the run cannot honour onto a default would make an
            unsupported request look like a successful one.
        """
        try:
            plan = normalise_source_plan(payload.sources)
        except ValueError as exc:
            raise DemoAgentError(str(exc), code="unsupported_source_plan") from exc
        return self._execute_plan(plan=plan, payload=payload, use_memory=True)

    def run_selection(
        self,
        selection: Any,
        payload: AgentRecommendRequest,
        *,
        preference_snapshot: Any = None,
        replan_client: Any = None,
    ) -> AgentRecommendResponse:
        """Execute a **validated LLM tool selection** through this service's own controller.

        The plan comes from the model's validated choice, never from the request body, so the
        caller's ``sources`` field cannot override it.  Sources are re-checked against
        :attr:`available_sources` at execution time (a source may have become unavailable between
        planning and running), and the memory-conditioned query adapter is switched **off** for
        this path: an LLM-selected run must not append unrelated historical preferences.

        ``preference_snapshot`` is the *caller's* trusted read of this session's M9 memory.  The chat
        runtime supplies it, keyed by the server-owned session namespace, so the model cannot name
        one; when it holds ACTIVE entries and memory-aware ranking is enabled, the presented
        candidates are reordered by the accepted M10A/M10B components (see
        :mod:`recommendation.demo.memory_ranking`) and when it is empty the fused order stands.  The
        model's tool *choice* is never overridden by memory, and the request-declared
        :meth:`run` path deliberately passes no snapshot, which is what keeps the legacy endpoint
        exactly as it was.
        """
        from recommendation.demo.tool_selection import source_plan_from_selection

        if getattr(selection, "route", None) != "recommend":
            raise DemoAgentError(
                "a direct selection does not start a recommendation run",
                code="not_a_recommendation",
            )
        plan = source_plan_from_selection(selection)
        try:
            plan = normalise_source_plan(plan)
        except ValueError as exc:
            raise DemoAgentError(str(exc), code="unsupported_source_plan") from exc
        unavailable = [name for name in plan if name not in self.available_sources]
        if unavailable:
            raise DemoAgentError(
                "the selected source(s) are not available on this deployment: "
                + ", ".join(unavailable),
                code="unsupported_source_plan",
            )
        return self._execute_plan(
            plan=plan,
            payload=payload,
            use_memory=False,
            preference_snapshot=preference_snapshot,
            # Bounded replanning is part of this path only: the caller passes the turn's model
            # client, and the service composes the policy with it when the switch is on.
            replan_client=replan_client,
        )

    def _execute_plan(
        self,
        *,
        plan: tuple[str, ...],
        payload: AgentRecommendRequest,
        use_memory: bool,
        preference_snapshot: Any = None,
        replan_client: Any = None,
    ) -> AgentRecommendResponse:
        """Run one plan and project it.  Shared by the request-declared and LLM-selected paths.

        ``preference_snapshot`` is honoured only when this deployment has memory-aware ranking
        enabled (:data:`ENV_MEMORY_RANKING`): the run's presented candidates are then reordered by
        the accepted M10A/M10B components, over trusted catalogue metadata, without touching their
        identity, score, score kind or provenance.
        """
        # A learned-ranker base ordering cannot enter the dynamic multi-source plane: the model was
        # fitted on the fixed three-source protocol and already returns a final ordering, so fusing
        # it again would break the protocol and misreport who ranked the result.  Refuse BEFORE
        # composing the controller or recalling anything (no model load, no predict, no recall).
        from ..inference.learned_ranker import LearnedRankerEngine
        _engine = getattr(self, "_engine", None) or getattr(self, "engine", None)
        if isinstance(_engine, LearnedRankerEngine):
            raise DemoAgentError(
                "dynamic candidate selection does not support the learned-ranker serving profile",
                code="unsupported_source_plan",
            )
        started = time.perf_counter()
        controller = self._compose_controller(
            k=payload.k,
            user_key=payload.user_key,
            plan=plan,
            use_memory=use_memory,
            replan_client=replan_client,
            # Read-only: the bounded replanning policy may describe the user's ACTIVE preferences
            # to the model, and nothing else in the run changes because of it.
            preference_snapshot=preference_snapshot,
        )
        task_state = _task_state(payload)
        agent_input = AgentInput(
            user_message=payload.message,
            trusted_user_history=tuple(payload.history),
        )

        agent_started = time.perf_counter()
        try:
            result = controller.invoke(agent_input, task_state=task_state)
        except Exception as exc:  # noqa: BLE001 - normalised to a stable, leak-free error
            raise DemoAgentError(
                f"the agent run failed: {type(exc).__name__}", code="agent_failed"
            ) from exc
        agent_ms = (time.perf_counter() - agent_started) * 1000.0

        serialize_started = time.perf_counter()
        ledger = _ledger_of(controller)
        # The run's presentable candidates: every source it legally executed, fused by the accepted
        # rank-fusion rule (see ``presentable_candidates``).  One definition serves the ranking, the
        # titles and the view, so nothing can be ranked that the response would refuse to show.
        from .agent_trajectory import presentable_candidates

        presentable = presentable_candidates(result.state)
        ranking = self._memory_ranking(result, ledger, payload, preference_snapshot)
        # Route is derived from the run's own trusted artifacts.  A run that produced no
        # candidate set is a direct turn *or* a failed one, and an empty recommendation list is
        # the honest answer in both cases - so a failure is never dressed up as a successful
        # direct reply.
        produced_candidates = result.state.get("tool_result") is not None
        recommendations = build_recommendations_view(
            result.state,
            ledger,
            # Titles for exactly what may be shown, which is the run's accumulated candidate
            # set - not only the most recent source's list.
            titles=_titles_of(
                self._metadata, [str(item.parent_asin) for item in presentable]
            ),
            ranking=ranking,
            # ``k`` bounds the answer *after* the fusion (and after any memory-aware ranking),
            # so it is the top of the run's own ordering rather than a slice of one source.
            limit=payload.k,
        )
        grounded_view = build_grounded_view(result, ledger)
        eligibility_view = build_eligibility_view(result)
        terminal_view = build_terminal_view(result)
        trajectory_view = build_trajectory_summary(result)
        # The run-level observation and the deterministic replan gate (E-03A).  Built here, after
        # every decision is final, from the frozen views above and nothing else: no tool runs, the
        # planner is not consulted again, no query changes and no controller state is written - so
        # the answer above cannot move because of it.
        from .observation import build_observation, decide_replan

        replan_execution = _replan_execution_view(controller, recommendations, terminal_view)
        observation = build_observation(
            terminal=terminal_view,
            grounded=grounded_view,
            eligibility=eligibility_view,
            trajectory=trajectory_view,
            plan=plan,
            available_sources=self.available_sources,
            requested_k=payload.k,
            presented_count=len(recommendations),
        )
        # A run the completion guard refused because no candidate could be *verified* compliant
        # ends with no text at all.  The reason is already the trusted eligibility view, so the
        # answer states it instead of returning an empty string: an empty constrained run is a
        # constraint outcome, not a transport failure.  Completed empty runs are rendered by the
        # finalizer itself; this covers the refused-completion path only.
        message = result.final_response
        if not message and not recommendations:
            projection = result.state.get("constraint_projection")
            if projection is not None and projection.is_empty:
                from ..agent.rendering import build_constraint_empty_response

                message = build_constraint_empty_response(projection.view)
        # The explanation must name exactly the answer's own products, in its own order.  The
        # final list is already built above (fusion, constraints, optional memory re-rank), so
        # this is the one correct input; a matching accepted text is kept unchanged.
        message = _consistent_message(message, recommendations, self._metadata)
        response = AgentRecommendResponse(
            route="recommend" if produced_candidates else "direct",
            message=message,
            control_plane=self.control_plane_name,
            recommendations=recommendations,
            grounded=grounded_view,
            eligibility=eligibility_view,
            terminal=terminal_view,
            trajectory=trajectory_view,
            observation=observation,
            replan=decide_replan(observation),
            replan_execution=replan_execution,
            timing=AgentTimingView(
                total_ms=(time.perf_counter() - started) * 1000.0,
                agent_ms=agent_ms,
                serialize_ms=(time.perf_counter() - serialize_started) * 1000.0,
            ),
        )
        return response


    def _memory_ranking(
        self,
        result: Any,
        ledger: Any,
        payload: AgentRecommendRequest,
        preference_snapshot: Any,
    ) -> Any:
        """Rank this run's presented candidates with the session's ACTIVE preferences, or nothing.

        Returns ``None`` whenever there is nothing to do, which is the common case and the reason
        a memory-free deployment is unaffected: no snapshot, no ranking switch, no ACTIVE
        preference, or no presented candidates all leave the fused order exactly as it was.

        The candidates come from :func:`presentable_candidates`, i.e. the same set the response is
        about to publish, so the ranking can neither introduce nor hide a product.  Its trusted
        catalogue metadata comes from this service's own M8 enricher over the shared index, and the
        matching and ordering are the accepted M10A and M10B components - not a second rule set.
        """
        if not self._memory_ranking_enabled or preference_snapshot is None:
            return None
        if self._enricher is None or self._matcher is None or self._reranker is None:
            # Without the trusted metadata path there is no honest way to decide whether a
            # candidate matches a preference, and guessing from a title is exactly what this
            # milestone forbids.  No components, no ranking.
            return None
        from .agent_trajectory import presentable_candidates
        from .memory_ranking import rank_with_memory

        candidates = presentable_candidates(result.state)
        if not candidates:
            return None
        return rank_with_memory(
            candidates=candidates,
            tool_result=result.state.get("tool_result"),
            preferences=preference_snapshot,
            enricher=self._enricher,
            matcher=self._matcher,
            reranker=self._reranker,
            # The current request selects which catalogue facts are retrieved; it is never merged
            # with a stored preference, and the preferences never enter this string.
            query=payload.message,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def memory_ranking_enabled(value: Any = None) -> bool:
    """Whether memory-aware ranking is enabled for this deployment.

    Default **on**; an explicit ``0``/``false``/``no``/``off`` turns it off, so the pre-E-02A
    behaviour can be pinned without a code change.  An unrecognised value keeps the default rather
    than guessing: the switch can only ever *disable* a capability that is inert without memory.
    """
    raw = os.environ.get(ENV_MEMORY_RANKING) if value is None else value
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def llm_policy_selected() -> bool:
    """True only when the deployment explicitly asked for the real-provider policy.

    An exact, case-insensitive match on :data:`AGENT_POLICY_LLM` and nothing else: a typo or a
    near-miss must not start billed provider calls, so there is no prefix matching and no
    truthiness shortcut.
    """
    value = (os.environ.get(ENV_AGENT_POLICY) or "").strip().lower()
    return value == AGENT_POLICY_LLM


def _build_llm_policy_factory() -> Any:
    """A factory returning the accepted model-driven policy, over the configured provider.

    The policy is the repository's existing :class:`~recommendation.control.LLMAgentPolicy` and
    the client is the existing provider adapter, so this adds no decision logic: it is the
    documented composition of two accepted components.  Constructed lazily, so a deployment
    that never sets the switch never imports the provider boundary.
    """
    from recommendation.control.model_policy import LLMAgentPolicy
    from recommendation.control.provider_adapter import build_provider_client

    client = build_provider_client()

    def factory(_plan: Sequence[str], _k: int) -> Any:
        return LLMAgentPolicy(client, max_attempts=2)

    return factory


def _two_tower_enabled() -> bool:
    """True when the deployment named a Two-Tower checkpoint to consult as a source."""
    return bool((os.environ.get(ENV_TWO_TOWER) or "").strip())


def _build_two_tower_tool() -> Any:
    """Compose the Two-Tower candidate source from the configured checkpoint.

    The engine is built lazily here rather than at import time, so a deployment that does not
    set the switch never loads the model or the id mapping.
    """
    from pathlib import Path as _Path

    from recommendation.control.two_tower_source import TwoTowerSourceTool
    from recommendation.inference import TwoTowerInferenceConfig, TwoTowerInferenceEngine

    checkpoint = _Path((os.environ.get(ENV_TWO_TOWER) or "").strip())
    mappings = os.environ.get(ENV_TWO_TOWER_MAPPINGS) or os.environ.get(
        "AGENTRECX_MAPPINGS_PATH"
    )
    return TwoTowerSourceTool(
        TwoTowerInferenceEngine(
            TwoTowerInferenceConfig(
                checkpoint_path=checkpoint,
                mappings_path=_Path(mappings) if mappings else None,
            )
        )
    )


def _similar_items_enabled() -> bool:
    """True when the deployment opted into real full-catalogue item-item retrieval."""
    value = (os.environ.get(ENV_SIMILAR_ITEMS) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _task_state(payload: AgentRecommendRequest) -> TaskState:
    """Build the run's task state from the request's stated constraints.

    Only ``EXPLICIT``, ``CURRENT_TURN`` constraints are constructed, because that is the
    accepted hardness rule: an inferred or softened signal must not be able to narrow the
    feasible set.  A request with no constraint yields an empty task state, so an
    unconstrained run behaves exactly as it did before constraints existed.
    """
    state = TaskState(request_text=payload.message)
    for constraint in payload.constraints:
        state = state.with_constraint(
            TaskConstraint(
                kind=constraint.kind,
                value=constraint.value,
                source_text=payload.message[:500],
            )
        )
    return state


def _recommended_identities(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Identities in the run's ranked recommendation list, in backend order."""
    tool_result = state.get("tool_result")
    return tuple(
        str(item.parent_asin) for item in getattr(tool_result, "recommendations", ()) or ()
    )


def _replan_execution_view(
    controller: LoopController, recommendations: Any, terminal: Any = None
) -> Any:
    """What bounded replanning did in this run, read from the policy the controller actually used.

    ``None`` unless that policy is the bounded replanning one (E-03B), so the block is absent -
    rather than defaulted - when the capability is disabled or the run used another policy.  The
    initial snapshot is the one the policy took when it decided; the final snapshot is this
    response's own counts.
    """
    policy = getattr(controller, "policy", None)
    record = getattr(policy, "record", None)
    if not callable(record):
        return None
    state = record(final_candidates=len(recommendations))
    from .agent_schemas import AgentReplanView, ReplanObservationView

    return AgentReplanView(
        status=str(getattr(state, "status", "not_triggered")),
        reason_code=str(getattr(state, "reason_code", "") or ""),
        model_called=bool(getattr(state, "model_called", False)),
        chosen_source=getattr(state, "chosen_source", None),
        eligible_sources=tuple(getattr(state, "eligible_sources", ()) or ()),
        decision_step=getattr(state, "decision_step", None),
        initial=ReplanObservationView(
            candidates=int(getattr(state, "initial_candidates", 0) or 0),
            # The decision-time record, frozen before the appended source ran.
            sources_consulted=tuple(
                getattr(state, "initial_consulted_sources", ()) or ()
            ),
            source_outcomes=dict(getattr(state, "initial_source_outcomes", {}) or {}),
            remaining_tool_calls=int(getattr(state, "remaining_tool_calls", 0) or 0),
            remaining_steps=int(getattr(state, "remaining_steps", 0) or 0),
        ),
        final=ReplanObservationView(
            candidates=len(recommendations),
            sources_consulted=tuple(getattr(state, "consulted_sources", ()) or ()),
            source_outcomes=dict(getattr(state, "source_outcomes", {}) or {}),
            # The run's own remaining budget when it stopped, not the decision's.
            remaining_tool_calls=max(
                0,
                int((getattr(terminal, "limits", {}) or {}).get("max_tool_calls", 0) or 0)
                - int(getattr(terminal, "tool_calls", 0) or 0),
            ),
            remaining_steps=max(
                0,
                int((getattr(terminal, "limits", {}) or {}).get("max_steps", 0) or 0)
                - int(getattr(terminal, "steps", 0) or 0),
            ),
        ),
    )


def _ledger_of(controller: LoopController) -> Any | None:
    """The controller's candidate ledger, or ``None`` when no plane was configured."""
    plane = getattr(controller, "candidate_plane", None)
    return getattr(plane, "ledger", None) if plane is not None else None


def _titles_of(metadata: Any, parent_asins: Sequence[str]) -> Mapping[str, str]:
    """A ``parent_asin -> title`` map for exactly the identities being returned.

    Looked up per returned identity rather than by materialising a title for all 156 746
    catalogue records on every request: the response is bounded by ``k``, and copying the
    whole catalogue's titles per response would be work proportional to the catalogue.  An
    absent record or an absent title simply does not appear - never a placeholder.
    """
    records = getattr(metadata, "records", None)
    if not isinstance(records, Mapping):
        return {}
    titles: dict[str, str] = {}
    for identity in parent_asins:
        record = records.get(identity)
        title = getattr(record, "title", None)
        if isinstance(title, str) and title.strip():
            titles[str(identity)] = title
    return titles


#: One Amazon ``parent_asin``: a leading ``B`` and nine alphanumerics.  Used only to read
#: which products a *rendered text* names, never to derive an identity for a candidate.
_ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")


def _product_ids(text: Any) -> list[str]:
    """The product ids a rendered text names, in order, duplicates preserved."""
    if not isinstance(text, str):
        return []
    return _ASIN_RE.findall(text)


def _facts_for(metadata: Any, parent_asins: Sequence[str]) -> Mapping[str, Mapping[str, str]]:
    """A ``parent_asin -> catalogue facts`` map for exactly the identities being rendered.

    Keyed by identity so a fact can only ever be printed beside the product it belongs to.
    Absent fields are omitted rather than defaulted: the explanation prints what the
    catalogue says, and nothing where it says nothing.
    """
    records = getattr(metadata, "records", None)
    if not isinstance(records, Mapping):
        return {}
    facts: dict[str, Mapping[str, str]] = {}
    for identity in parent_asins:
        record = records.get(identity)
        if record is None:
            continue
        entry: dict[str, str] = {}
        title = getattr(record, "title", None)
        if isinstance(title, str) and title.strip():
            entry["Title"] = title
        main_category = getattr(record, "main_category", None)
        if isinstance(main_category, str) and main_category.strip():
            entry["Main category"] = main_category
        categories = getattr(record, "categories", None) or ()
        if categories:
            entry["Category"] = str(categories[-1])
        store = getattr(record, "store", None)
        if isinstance(store, str) and store.strip():
            entry["Store"] = store
        if entry:
            facts[str(identity)] = entry
    return facts


def _consistent_message(
    message: str, recommendations: Sequence[Any], metadata: Any
) -> str:
    """Return text that names **exactly** the final recommendation list, in its order.

    The accepted finalizer renders the sequential source's candidates, which can differ
    from the fused, constraint-projected answer whenever another source contributed.  The
    final list is already decided when this is called, so the only correct input to the
    explanation is that list.  A text that already names exactly those identities in order
    is left untouched; otherwise the explanation is re-rendered from the final list and
    identity-keyed catalogue facts.  No candidate, rank or list is modified here.
    """
    if not recommendations:
        return message
    final_ids = [str(item.parent_asin) for item in recommendations]
    if _product_ids(message) == final_ids:
        return message
    return build_final_list_response(recommendations, _facts_for(metadata, final_ids))


def _tiger_enabled() -> bool:
    """True when the deployment opted into the TIGER generative source."""
    value = (os.environ.get(ENV_TIGER) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _shared_tiger_engine() -> Any:
    """The process-wide engine, imported lazily so this module never hard-depends on it."""
    from recommendation.inference.tiger_engine import shared_engine

    return shared_engine()


def _build_tiger_engine(identity: Any, **overrides: Any) -> Any:
    """Compose the real TIGER serving engine over the deployment's own trusted identity map.

    The identity map is the accepted one this service already uses, so TIGER can never resolve a
    product AgentRec-X does not have.  The worker runs in the backend's own virtual environment
    and is started lazily by :meth:`DemoAgentService.prewarm_tiger`.
    """
    from recommendation.inference.tiger_engine import (
        TigerServingEngine,
        TigerWorkerConfig,
        shared_engine,
    )

    existing = shared_engine()
    if existing is not None:
        # One worker per process: adopt the engine the serving fusion already built and prewarmed.
        return existing
    config_kwargs: dict[str, Any] = {}
    checkpoint = (os.environ.get(ENV_TIGER_CHECKPOINT) or "").strip()
    if checkpoint:
        from pathlib import Path

        config_kwargs["checkpoint_dir"] = Path(checkpoint)
    config = TigerWorkerConfig(**{**config_kwargs, **overrides})
    return TigerServingEngine(identity, config=config)
