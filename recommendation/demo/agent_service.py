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
The service keeps a bounded cache of compiled controllers keyed by
``(k, user_key, source plan)``, exactly as the demo runtime caches compiled graphs per
``(k, user_key)``: a controller binds a source plan and an optional memory namespace at
construction, and compiling one is pure Python over the same process-scoped collaborators.
No model, index or store is ever rebuilt.  Preference memory is touched **only** when the
request supplies a ``user_key``; without one the run reads and writes no memory at all,
which is what makes the offline mode reproducible.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Mapping, Sequence

from recommendation.agent import AgentInput
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

__all__ = [
    "AGENT_LOOP_LIMITS",
    "DEFAULT_MAX_CACHED_CONTROLLERS",
    "ENV_SIMILAR_ITEMS",
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

#: Loop budgets for one HTTP request.  Sized for the longest plan this endpoint accepts -
#: one step per named source plus the terminal finish - with the same hard bounds as the
#: accepted demo loop, so an agent cannot turn one request into unbounded work.
AGENT_LOOP_LIMITS = LoopLimits(max_steps=6, max_tool_calls=4, max_retries=1)

#: Hard cap on cached controllers, independent of the request rate, so a long-lived process
#: cannot grow the cache without limit.
DEFAULT_MAX_CACHED_CONTROLLERS = 128

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
        policy_factory: Any = None,
        limits: LoopLimits = AGENT_LOOP_LIMITS,
        max_cached_controllers: int = DEFAULT_MAX_CACHED_CONTROLLERS,
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
        #: Builds the run's policy.  Defaults to the deterministic, request-driven source plan;
        #: a deployment may inject the accepted model-driven policy instead.  Injection rather
        #: than a branch inside the loop is what keeps this endpoint's policy interchangeable.
        self._policy_factory = policy_factory or (
            lambda plan, k: DemoSourcePlanPolicy(plan, default_k=k)
        )
        self._limits = limits
        self._max_cached = max_cached_controllers
        self._controllers: dict[tuple[int, str | None, tuple[str, ...]], LoopController] = {}
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
        return tuple(sources)

    @property
    def consultable_note(self) -> str:
        """A one-line statement of which actions the controller currently offers."""
        return (
            "history and catalog_search are reachable; a source registered without a "
            "similar-item tool is reported as available but the controller does not yet offer "
            "SELECT_SOURCE for it"
        )

    def close(self) -> None:
        """Drop every cached controller.

        Controllers hold compiled policies and a candidate ledger, but no store or model of
        their own - those belong to the runtime.  Releasing the cache is therefore enough to
        make a shutdown deterministic, and calling this twice is harmless.
        """
        with self._lock:
            self._controllers.clear()

    @property
    def compiled_controller_count(self) -> int:
        """How many distinct ``(k, user_key, plan)`` controllers are currently cached."""
        with self._lock:
            return len(self._controllers)

    # -- controller cache -------------------------------------------------- #

    def _controller_for(
        self, *, k: int, user_key: str | None, plan: Sequence[str]
    ) -> LoopController:
        cache_key = (k, user_key, tuple(plan))
        with self._lock:
            existing = self._controllers.get(cache_key)
            if existing is not None:
                return existing

        controller = self._build_controller(k=k, user_key=user_key, plan=plan)

        with self._lock:
            duplicate = self._controllers.get(cache_key)
            if duplicate is not None:
                return duplicate
            if len(self._controllers) >= self._max_cached:
                # FIFO eviction: dict preserves insertion order, so the oldest key goes.
                oldest = next(iter(self._controllers))
                self._controllers.pop(oldest, None)
            self._controllers[cache_key] = controller
            return controller

    def _build_controller(
        self, *, k: int, user_key: str | None, plan: Sequence[str]
    ) -> LoopController:
        """Compose one controller: capability, candidate plane, eligibility, then the loop.

        The order matters.  The eligibility evaluator is built over the **same** catalogue
        the plane grounds against, and the plane's ledger is a fresh one per controller, so
        provenance is scoped to that controller's runs and cannot accumulate across requests.
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
        )
        eligibility = CandidateEligibilityEvaluator(GroundedReasoner(self._metadata))
        policy = self._policy_factory(plan, k)
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
        started = time.perf_counter()
        try:
            plan = normalise_source_plan(payload.sources)
        except ValueError as exc:
            raise DemoAgentError(str(exc), code="unsupported_source_plan") from exc
        unavailable = [name for name in plan if name not in self.available_sources]
        if unavailable:
            raise DemoAgentError(
                "this deployment does not configure the requested source(s): "
                + ", ".join(unavailable),
                code="unsupported_source_plan",
            )

        controller = self._controller_for(
            k=payload.k, user_key=payload.user_key, plan=plan
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
        # Route is derived from the run's own trusted artifacts.  A run that produced no
        # candidate set is a direct turn *or* a failed one, and an empty recommendation list is
        # the honest answer in both cases - so a failure is never dressed up as a successful
        # direct reply.
        produced_candidates = result.state.get("tool_result") is not None
        response = AgentRecommendResponse(
            route="recommend" if produced_candidates else "direct",
            message=result.final_response,
            control_plane=self.control_plane_name,
            recommendations=build_recommendations_view(
                result.state,
                ledger,
                titles=_titles_of(self._metadata, _recommended_identities(result.state)),
            ),
            grounded=build_grounded_view(result, ledger),
            eligibility=build_eligibility_view(result),
            terminal=build_terminal_view(result),
            trajectory=build_trajectory_summary(result),
            timing=AgentTimingView(
                total_ms=(time.perf_counter() - started) * 1000.0,
                agent_ms=agent_ms,
                serialize_ms=(time.perf_counter() - serialize_started) * 1000.0,
            ),
        )
        return response


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


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
