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

import hashlib
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
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
    ReasoningExecutor,
    RecommendFromHistoryCapability,
)
from recommendation.memory import (
    PreferenceKind,
    PreferenceMode,
    PreferencePolarity,
)
from recommendation.control.task_state import TaskConstraint, TaskState, TaskIntent

from .agent_policy import (
    ClarificationPolicy,
    DemoSourcePlanPolicy,
    IntentQueryPolicy,
    normalise_source_plan,
)
from .research_policy import (
    DynamicProductResearchPolicy,
    requested_attributes,
    wants_comparison,
)
from .replan_policy import BoundedReplanPolicy
from .agent_schemas import (
    AgentClarificationView,
    AgentConstraintItem,
    AgentIntentConstraintView,
    AgentIntentView,
    AgentPreferenceEvidenceView,
    AgentPreferenceOptionView,
    AgentPreferenceOutcomeView,
    AgentRecommendRequest,
    AgentRecommendResponse,
    AgentResearchClarificationView,
    AgentResearchView,
    AgentTimingView,
)
from .preference_construction import (
    PreferenceOption,
    PreferencePrompt,
    PreferenceState,
    TaskPreference,
    augment_intent,
    blocked_attributes_from_history,
    blocked_attributes_from_intent,
    choose_preference_action,
    preference_from_option,
    price_evidence_note,
    rank_with_task_preferences,
    resolve_preference_option,
)
from .grounded_actions import (
    GroundedAction,
    choose_grounded_action,
    grounded_actions_enabled,
    grounded_budget,
    is_uncertain_message,
    refresh_candidate_order,
)
from .itemcf_rerank import ItemCFRerankSignal
from .retrieval_plan import (
    dynamic_retrieval_enabled,
    plan_dynamic_retrieval,
)
from .product_role import ProductTarget, TargetRole, infer_target
from .proposal_bridge import map_proposal
from .soft_preference_ranking import (
    rank_with_m5_soft_preferences,
    soft_preferences_from_ranker_projection,
)
from .research_clarification import (
    ResearchResume,
    build_research_clarification,
    resolve_research_option,
)
from .shopping_intent import (
    IntentEntry,
    ShoppingIntent,
    Strength,
    build_reason,
    derive_intent,
    promote_explicit_type_matches,
    query_terms,
)
from recommendation.evaluation.decision import (
    DecisionCandidate,
    DecisionTrajectoryRecorder,
    PreferenceChange,
    RankingRefreshRecord,
    decision_trajectory_enabled,
)
from .decision_choice_store import (
    ChoiceCollectionDisabled,
    DecisionChoiceConflict,
    DecisionChoiceStore,
    DecisionChoiceStoreError,
    choice_collection_enabled,
    choice_database_path,
)
from .agent_trajectory import (
    build_eligibility_view,
    build_grounded_view,
    build_recommendations_view,
    build_research_view,
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

#: Pending payloads are versioned so a future migration can distinguish a bounded resume
#: context from the older full-response form.  Restore accepts both versions.
PENDING_PAYLOAD_VERSION = 2

#: Fields needed to resume a confirmed preference against the exact candidate window.  The
#: rendered message and candidate identities are required; the trace/observation/UI-only
#: blocks are not.  ``request.message`` is kept separately because intent derivation needs it.
_PERSISTED_BASE_RESPONSE_FIELDS = (
    "api_version",
    "route",
    "message",
    "control_plane",
    "recommendations",
    "grounded",
    "eligibility",
    "terminal",
    "trajectory",
    "research",
    "timing",
)


def _bounded_base_response(response: AgentRecommendResponse) -> dict[str, Any]:
    """Return only the base-response fields a preference resume actually reads.

    This is deliberately smaller than ``response.model_dump()``: observation, research
    execution details outside the small view, replan/UI fields, and prompt/evidence duplicates
    are not persisted.  The remaining ``message`` is part of the resume contract and is
    deleted with the pending row when the question is consumed or retired.
    """
    dumped = response.model_dump()
    return {name: dumped[name] for name in _PERSISTED_BASE_RESPONSE_FIELDS if name in dumped}

#: Opt-in switch for the real-provider agent policy.
#:
#: Off by default, and deliberately a *separate* switch from the provider configuration: a
#: deployment may have ``AGENTRECX_LLM_*`` set for other reasons, and a packaging phase must
#: not silently start making billed network calls because credentials happen to be present.
#: Set ``AGENTRECX_AGENT_POLICY=llm`` to use the accepted :class:`LLMAgentPolicy` for this
#: endpoint's source decisions.  Unset or ``deterministic`` keeps the request-driven
#: :class:`~recommendation.demo.agent_policy.DemoSourcePlanPolicy`, which needs no network.
ENV_AGENT_POLICY = "AGENTRECX_AGENT_POLICY"
ENV_DENSE_RETRIEVAL = "AGENTRECX_DENSE_RETRIEVAL"
ENV_DENSE_INDEX = "AGENTRECX_DENSE_INDEX"
ENV_CATALOG_SEARCH_MODE = "AGENTRECX_CATALOG_SEARCH_MODE"
ENV_BGE_MODEL_PATH = "AGENTRECX_BGE_MODEL_PATH"
ENV_MAPPINGS_PATH = "AGENTRECX_MAPPINGS_PATH"

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

#: Phase 6D task-level bounds.  They are deliberately separate from the per-run tool-call
#: budget: a resume creates a new run, but it must not reset the research budget.
MAX_RESEARCH_CLARIFICATIONS = 2
MAX_TASK_RESEARCH_CALLS = 4


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
        checkpoint: Any = None,
        state_integration: Any = None,
        soft_m5_rerank: bool | None = None,
    ) -> None:
        self._tool = tool
        self._enricher = enricher
        self._metadata = metadata
        # Phase 6A: read-only product research over the same trusted metadata the
        # eligibility evaluator uses.  No second catalogue is loaded.  The executor is
        # optional so a deployment without a catalogue cannot advertise facts it cannot read.
        self._reasoning_executor = (
            ReasoningExecutor(GroundedReasoner(metadata)) if metadata is not None else None
        )
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
        #: Pending shopping clarifications, keyed by clarification id.  When a checkpoint is
        #: supplied these are restored and written through; without one they are process-local
        #: and a restart drops them.  Not a second preference store: the payload also holds the
        #: bounded resume context needed to apply the answer to the same candidate window.
        self._pending_clarifications: dict[str, dict[str, Any]] = {}
        #: Sessions that already had one clarification.  A second undecidable run degrades
        #: through the accepted termination path instead of looping on questions.
        self._clarified_sessions: set[str] = set()
        #: Candidate-aware preference construction is a shopping-only capability.  One or two
        #: bounded questions per task, and the chosen soft preferences live only in task state.
        self._preference_rounds: dict[str, int] = {}
        self._preference_second_enabled: dict[str, bool] = {}
        self._task_preferences: dict[str, tuple[TaskPreference, ...]] = {}
        #: R1 experiment: one bounded memory-guided recall attempt per preference turn.
        self._r1_retrieval_attempts: set[str] = set()
        self._r1_diagnostics: dict[str, dict[str, Any]] = {}
        #: Last successfully presented response per shopping task, for normal-turn RERANK.
        self._r1_last_responses: dict[str, AgentRecommendResponse] = {}
        #: Last non-corrective query terms per task, used to remove a withdrawn condition.
        self._r1_task_anchor_terms: dict[str, tuple[str, ...]] = {}
        #: Last M5-authorized constraint set consumed by R1 for one task.  Used only in
        #: APPLY_CONSTRAINTS to detect a state transition between two turns; SHADOW never
        #: populates it.
        self._m5_authorized_constraints: dict[str, tuple[tuple[str, str], ...]] = {}
        #: Task-scoped grounded type/role target.  Follow-ups and pure withdrawals retain
        #: the last main target instead of being reclassified from their own wording.
        self._role_main_targets: dict[str, Any] = {}
        self._role_current_targets: dict[str, Any] = {}
        self._role_diagnostics: dict[str, dict[str, Any]] = {}
        self._proposal_diagnostics: dict[str, dict[str, Any]] = {}
        #: Bounded, read-only candidate-filter trace for P3C-2 diagnostics.
        self._candidate_filter_diagnostics: dict[str, dict[str, Any]] = {}
        self._constraint_trace: dict[str, dict[str, Any]] = {}
        #: P1B-3: explicit soft_off/soft_on arm.  Off by default, so the accepted path
        #: and every existing test are unchanged unless the benchmark opts in.
        self._soft_m5_rerank = (
            bool(soft_m5_rerank)
            if soft_m5_rerank is not None
            else os.environ.get("AGENTRECX_M5_SOFT_RERANK", "off").strip().casefold()
            in {"1", "on", "true", "yes"}
        )
        self._ranker_projections: dict[str, Any] = {}
        self._soft_rerank_diagnostics: dict[str, dict[str, Any]] = {}
        self._soft_last_counts: dict[str, int] = {}
        self._soft_last_consumed_counts: dict[str, int] = {}
        #: Latest explicit task-scoped price bounds.  A correction replaces the old value
        #: for subsequent turns; the event/history store is not rewritten.
        self._task_price_constraints: dict[str, tuple[dict[str, str], ...]] = {}
        #: M7 experiment diagnostics (never read by the accepted serving path).
        self._m7_refresh: dict[str, dict[str, Any]] = {}
        self._m7_retrieval_guard = False
        #: M8 decision trajectories (off by default; experiment only).
        self._m8_trajectories: dict[str, DecisionTrajectoryRecorder] = {}
        #: M8.1 real-choice store (lazy; only when collection is enabled).
        self._m8_choice_store: DecisionChoiceStore | None = None
        self._m8_choice_store_error: str | None = None
        self._checkpoint = checkpoint
        if checkpoint is not None:
            self._restore_checkpoint()
        #: M5.1 unified decision-state integration (OFF/SHADOW/APPLY_CONSTRAINTS), composed
        #: once by the runtime and shared.  ``None`` means no integration was composed, which
        #: is exactly the pre-M5.1 behaviour.  It never decides the answer on its own: only
        #: APPLY_CONSTRAINTS may add validated, user-authorized constraints to this turn's
        #: existing trusted constraint inputs.
        self._state_integration = state_integration
        #: Built once, because the catalogue search index is a per-process cost, not a
        #: per-request one.
        search_mode = _catalog_search_mode()
        dense_index = None
        if search_mode in {"bge", "hybrid"}:
            dense_path = os.environ.get(ENV_DENSE_INDEX, "").strip()
            if not dense_path:
                raise DemoAgentError(
                    "BGE/hybrid catalogue search requires AGENTRECX_DENSE_INDEX",
                    code="dense_retrieval_unavailable",
                )
            try:
                from pathlib import Path as _DensePath

                from recommendation.inference.dense_catalog import (
                    DenseCatalogIndex,
                    mapping_fingerprint,
                )

                dense_index = DenseCatalogIndex.load(
                    _DensePath(dense_path),
                    model_path=os.environ.get(ENV_BGE_MODEL_PATH, "").strip() or None,
                    catalogue_records=metadata.records,
                    mapping_sha=mapping_fingerprint(
                        os.environ.get(ENV_MAPPINGS_PATH, "").strip() or None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - explicit opt-in must fail loudly
                raise DemoAgentError(
                    "the BGE/hybrid catalogue index could not be loaded or does not "
                    "match the current catalogue",
                    code="dense_retrieval_unavailable",
                ) from exc
        self._catalog_search = CatalogSearchSource(
            metadata, mode=search_mode, dense_index=dense_index
        )
        # Optional experimental collaborative rerank.  OFF is the default and does not
        # load the artifact or change the accepted ordering path.
        try:
            self._itemcf_signal = ItemCFRerankSignal.from_env()
        except Exception as exc:  # noqa: BLE001 - explicit opt-in must fail loudly
            raise DemoAgentError(
                "the experimental ItemCF rerank artifact could not be loaded",
                code="itemcf_rerank_unavailable",
            ) from exc

    def _restore_checkpoint(self) -> None:
        """Restore pending questions and task preferences from the durable store."""
        purge = getattr(self._checkpoint, "purge_expired_pending", None)
        if callable(purge):
            purge(CLARIFICATION_TTL_SECONDS)
        for clarification_id, payload in self._checkpoint.load_pending():
            self._pending_clarifications[str(clarification_id)] = dict(payload)
        for task_id, state in self._checkpoint.load_task_states():
            task_key = str(task_id)
            preferences: list[TaskPreference] = []
            for item in state.get("preferences") or ():
                try:
                    preferences.append(TaskPreference(**dict(item)))
                except (TypeError, ValueError):
                    # A malformed row is not silently used as a preference.  The checkpoint
                    # remains for audit; the task simply starts without that entry.
                    preferences = []
                    break
            self._task_preferences[task_key] = tuple(preferences)
            self._preference_rounds[task_key] = int(state.get("rounds") or 0)
            self._preference_second_enabled[task_key] = bool(
                state.get("second_enabled", False)
            )
            anchor = state.get("r1_anchor_terms") or ()
            if anchor:
                self._r1_task_anchor_terms[task_key] = tuple(str(term) for term in anchor)
            if state.get("clarified"):
                self._clarified_sessions.add(task_key)

    def _purge_expired_pending(self) -> None:
        """Apply the clarification TTL to the durable store when it supports it."""
        if self._checkpoint is None:
            return
        purge = getattr(self._checkpoint, "purge_expired_pending", None)
        if callable(purge):
            purge(CLARIFICATION_TTL_SECONDS)

    def forget_task(self, task_key: str) -> None:
        """Forget one reset task's in-process pending/task state and durable rows.

        The durable cleanup is also done by ``DemoSessionManager.delete()``; this method makes
        the process-local maps idempotent with that authoritative state.
        """
        with self._lock:
            for key, value in list(self._pending_clarifications.items()):
                if value.get("session_id") == task_key:
                    self._pending_clarifications.pop(key, None)
            self._preference_rounds.pop(task_key, None)
            self._preference_second_enabled.pop(task_key, None)
            self._task_preferences.pop(task_key, None)
            self._r1_diagnostics.pop(task_key, None)
            self._r1_last_responses.pop(task_key, None)
            self._r1_task_anchor_terms.pop(task_key, None)
            self._m5_authorized_constraints.pop(task_key, None)
            self._role_main_targets.pop(task_key, None)
            self._role_current_targets.pop(task_key, None)
            self._role_diagnostics.pop(task_key, None)
            self._proposal_diagnostics.pop(task_key, None)
            self._task_price_constraints.pop(task_key, None)
            self._ranker_projections.pop(task_key, None)
            self._soft_rerank_diagnostics.pop(task_key, None)
            self._soft_last_counts.pop(task_key, None)
            self._soft_last_consumed_counts.pop(task_key, None)
            self._clarified_sessions.discard(task_key)
            self._r1_retrieval_attempts = {
                key for key in self._r1_retrieval_attempts if not key.startswith(f"{task_key}:")
            }
        if self._checkpoint is not None:
            self._checkpoint.clear_pending_for_task(task_key)
            self._checkpoint.clear_task_state(task_key)
        # M5.1: forget this task's process-local decision-state diagnostics as well, so a
        # retired session cannot keep reporting a state summary.
        forget_state = getattr(self._state_integration, "forget_diagnostics", None)
        if callable(forget_state):
            forget_state(task_key)

    def _persist_task(self, task_key: str) -> None:
        if self._checkpoint is None:
            return
        self._checkpoint.save_task_state(
            task_key,
            preferences=[
                asdict(preference) for preference in self._task_preferences.get(task_key, ())
            ],
            rounds=int(self._preference_rounds.get(task_key, 0)),
            second_enabled=bool(self._preference_second_enabled.get(task_key, False)),
            clarified=task_key in self._clarified_sessions,
            r1_anchor_terms=self._r1_task_anchor_terms.get(task_key, ()),
        )

    def _save_pending(
        self,
        clarification_id: str,
        task_key: str,
        payload: Mapping[str, Any],
    ) -> None:
        self._pending_clarifications[clarification_id] = dict(payload)
        if self._checkpoint is not None:
            self._checkpoint.save_pending(
                clarification_id,
                task_key,
                dict(payload),
                created_at=float(payload.get("created_at") or time.time()),
            )

    def _drop_pending(self, clarification_id: str | None) -> None:
        if not clarification_id:
            return
        self._pending_clarifications.pop(clarification_id, None)
        if self._checkpoint is not None:
            self._checkpoint.delete_pending(clarification_id)

    def _clear_pending_for_task(self, task_key: str) -> None:
        for key, value in list(self._pending_clarifications.items()):
            if value.get("session_id") == task_key:
                self._pending_clarifications.pop(key, None)
                if self._checkpoint is not None:
                    self._checkpoint.delete_pending(key)

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
        store, self._m8_choice_store = self._m8_choice_store, None
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

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
        clarification_question: str | None = None,
        clarify_unresolved: bool = False,
        intent_terms: Sequence[str] = (),
        dynamic_research: bool = False,
        research_resume: ResearchResume | None = None,
        research_budget_remaining: int = MAX_TASK_RESEARCH_CALLS,
        clarification_budget_remaining: int = MAX_RESEARCH_CLARIFICATIONS,
        memory_user_message: str | None = None,
        persist_memory: bool = True,
        source_policy: str = "balanced",
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
            clarification_question=clarification_question,
            clarify_unresolved=clarify_unresolved,
            intent_terms=intent_terms,
            dynamic_research=dynamic_research,
            research_resume=research_resume,
            research_budget_remaining=research_budget_remaining,
            clarification_budget_remaining=clarification_budget_remaining,
            memory_user_message=memory_user_message,
            persist_memory=persist_memory,
            source_policy=source_policy,
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
        clarification_question: str | None = None,
        clarify_unresolved: bool = False,
        intent_terms: Sequence[str] = (),
        dynamic_research: bool = False,
        research_resume: ResearchResume | None = None,
        research_budget_remaining: int = MAX_TASK_RESEARCH_CALLS,
        clarification_budget_remaining: int = MAX_RESEARCH_CLARIFICATIONS,
        memory_user_message: str | None = None,
        persist_memory: bool = True,
        source_policy: str = "balanced",
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
            ledger=CandidateLedger(source_policy=source_policy),
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
        # The shopping-intent wrappers are additive and opt-in: the legacy request path and the
        # LLM-selected path compose the same policy object they always did.
        if intent_terms:
            policy = IntentQueryPolicy(policy, terms=intent_terms)
        if clarification_question or clarify_unresolved:
            policy = ClarificationPolicy(
                policy,
                question=clarification_question,
                ask_on_unresolved=clarify_unresolved,
            )
        if dynamic_research and self._reasoning_executor is not None:
            # Outer wrapper: it gets the last chance to research after the source plan (and
            # the optional clarification wrapper) says it is done.  It only proposes
            # INSPECT_PRODUCT / COMPARE_PRODUCTS, and only while the tool budget allows.
            policy = DynamicProductResearchPolicy(
                policy,
                resume=research_resume,
                research_budget_remaining=research_budget_remaining,
                clarification_budget_remaining=clarification_budget_remaining,
            )
        return LoopController(
            policy,
            capability,
            memory_service=self._memory_service if user_key else None,
            user_key=user_key,
            limits=self._limits,
            candidate_plane=plane,
            eligibility_evaluator=eligibility,
            reasoning_executor=self._reasoning_executor,
            memory_user_message=memory_user_message,
            persist_memory=persist_memory,
            source_policy=source_policy,
        )

    # -- one request ------------------------------------------------------- #

    def run(
        self,
        payload: AgentRecommendRequest,
        *,
        clarification_question: str | None = None,
        clarify_unresolved: bool = False,
        intent_terms: Sequence[str] = (),
        dynamic_research: bool = False,
        research_resume: ResearchResume | None = None,
        research_budget_remaining: int = MAX_TASK_RESEARCH_CALLS,
        clarification_budget_remaining: int = MAX_RESEARCH_CLARIFICATIONS,
        task_preferences: tuple[TaskPreference, ...] = (),
        planner_retrieval: bool = False,
        apply_user_state: bool = True,
        state_message: str | None = None,
        source_policy: str = "balanced",
    ) -> AgentRecommendResponse:
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
        return self._execute_plan(
            plan=plan,
            payload=payload,
            use_memory=True,
            clarification_question=clarification_question,
            clarify_unresolved=clarify_unresolved,
            intent_terms=intent_terms,
            dynamic_research=dynamic_research,
            research_resume=research_resume,
            research_budget_remaining=research_budget_remaining,
            clarification_budget_remaining=clarification_budget_remaining,
            task_preferences=task_preferences,
            planner_retrieval=planner_retrieval,
            apply_user_state=apply_user_state,
            state_message=state_message,
            source_policy=source_policy,
        )

    def run_shopping_turn(
        self,
        payload: AgentRecommendRequest,
        *,
        session_id: str,
        task_id: str | None = None,
        turn_id: str | None = None,
        idempotency_key: str | None = None,
        proposal_operations: Mapping[str, Any] | None = None,
        planner_retrieval: bool = False,
        apply_user_state: bool = True,
        state_message: str | None = None,
        source_policy: str = "balanced",
    ) -> AgentRecommendResponse:
        """Serve one turn of a multi-turn shopping session.

        ``session_id`` is the preference-memory namespace.  ``task_id`` is the concrete demo
        session that owns pending questions and task preferences.  They are separate because
        several sessions may share one memory user key; task choices must not leak across
        those sessions.

        ``turn_id`` is the **server-owned** turn allocation from the demo session (the HTTP
        route always supplies it) and ``idempotency_key`` is an optional client-supplied
        stable key for this logical turn.  The M5.2 decision-state identity resolves once per
        turn -- client idempotency key first, then the server turn id, then a content digest
        fallback for direct service callers -- so a genuine retry reuses the same event ids
        while two distinct identical messages never collapse.  Both values only select the
        decision-state event identity; they never change the recommendation path.
        """
        if self._memory_service is None:
            raise DemoAgentError(
                "multi-turn shopping requires the preference-memory store",
                code="memory_unavailable",
            )
        self._purge_expired_pending()
        task_key = task_id or session_id
        if planner_retrieval:
            # A planner RETRIEVE query is a retrieval rewrite, not a new user utterance.
            # The session role target was primed from the real user turn by the controlled
            # runner; do not let the rewritten query redefine it.
            role_target_changed = False
            self._mirror_role_target_to_user_key(task_key, payload.user_key)
        else:
            role_target_changed = self._update_role_target(task_key, payload.message)
            # The HTTP shopping route passes the session's memory namespace as
            # ``session_id`` (payload.user_key) and the concrete session id as ``task_id``.
            # Task-scoped role state is written under ``task_key`` but the rule path reads it
            # through ``payload.user_key`` in ``_execute_plan``.  Mirror the current target onto
            # this session's namespace on **every** turn, not only the proposal branch, so a
            # follow-up turn without type words still sees the same target.  The namespace is
            # session-owned (never shared by two demo sessions), so this cannot leak one
            # session's role target into another.
            self._mirror_role_target_to_user_key(task_key, payload.user_key)
        proposal_actions = None
        if proposal_operations is not None:
            proposal_actions = map_proposal(proposal_operations, payload.message)
            self._proposal_diagnostics[task_key] = {
                "valid": proposal_actions.valid,
                "accepted_operations": list(proposal_actions.accepted),
                "rejected_operations": list(proposal_actions.rejected),
                "errors": list(proposal_actions.errors),
                "preference_count": len(proposal_actions.preferences),
                "constraint_count": len(proposal_actions.constraints),
                "retraction_count": len(proposal_actions.retractions),
                "noop_kinds": list(proposal_actions.noop_kinds),
                "source": "llm_proposal",
            }
            # A proposal is the sole M5 input for this turn.  It must not be combined
            # with the rule extractor's same-turn preferences/constraints; that would
            # double-write the same shopping fact.
            if proposal_actions.valid:
                desired_role = (
                    proposal_actions.role_target.role
                    if proposal_actions.role_target is not None
                    else TargetRole.MAIN
                )
                previous = self._role_current_targets.get(task_key)
                previous_type = previous.type if previous is not None else None
                if proposal_actions.role_target is not None:
                    new_target = ProductTarget(previous_type, desired_role)
                    self._role_current_targets[task_key] = new_target
                    if desired_role is not TargetRole.ACCESSORY:
                        self._role_main_targets[task_key] = new_target
                    role_target_changed = True
            # The API session_id (task key) and the M9 user_key are distinct, while
            # _execute_plan looks up the role target by payload.user_key.  Mirror the
            # task-scoped target onto the session namespace so the explicit role target
            # is actually consumed by the recommendation path.
            if payload.user_key and task_key in self._role_current_targets:
                self._role_current_targets[str(payload.user_key)] = self._role_current_targets[task_key]
                if task_key in self._role_main_targets:
                    self._role_main_targets[str(payload.user_key)] = self._role_main_targets[task_key]
        # A new explicit goal/withdrawal that did not carry a clarification id must still
        # invalidate an orphaned pending question for this task.  Otherwise the old
        # question can survive after the benchmark/user abandons it and keep influencing
        # later turns.
        fresh_turn_override = False
        if not payload.clarification_id and _pending_message_starts_fresh_turn(
            payload.message
        ):
            self._clear_pending_for_task(task_key)
            fresh_turn_override = True
        # M5.2: resolve the logical turn identity ONCE, near the top, so every M5 fact this
        # turn records shares it (planning branch and clarification-answer branch alike).
        turn_key, identity_source = self._resolve_turn_identity(
            payload,
            task_key=task_key,
            server_turn_id=turn_id,
            idempotency_key=idempotency_key,
        )
        if payload.clarification_id:
            pending_preference = None
            with self._lock:
                pending = self._pending_clarifications.get(payload.clarification_id)
                if pending is not None and pending.get("kind") == "preference":
                    pending_preference = pending
            if pending_preference is not None:
                if (
                    not payload.clarification_option
                    and _pending_message_starts_fresh_turn(payload.message)
                ):
                    # The user is not answering the displayed option list.  A withdrawal
                    # or a genuinely new goal must be processed as a normal turn instead
                    # of being replayed against the stale stored request forever.
                    self._drop_pending(payload.clarification_id)
                    payload = payload.model_copy(
                        update={"clarification_id": None, "clarification_option": None}
                    )
                    fresh_turn_override = True
                else:
                    return self._answer_preference(
                        payload,
                        pending_preference,
                        task_key,
                        user_key=session_id,
                        turn_key=turn_key,
                        turn_identity_source=identity_source,
                    )
        preference_choice = self._task_preferences.get(task_key, ())
        if payload.clarification_id:
            pending = None
            research_resume: ResearchResume | None = None
            previous_research: AgentResearchView | None = None
            with self._lock:
                pending = self._pending_clarifications.get(payload.clarification_id)
                if pending is None:
                    raise DemoAgentError(
                        "the clarification id is unknown or belongs to another session",
                        code="unknown_clarification",
                    )
                if pending.get("session_id") != task_key:
                    # Do not consume another session's question: a cross-session attempt must
                    # not be able to invalidate it.
                    raise DemoAgentError(
                        "the clarification id is unknown or belongs to another session",
                        code="unknown_clarification",
                    )
                if time.time() - float(pending.get("created_at", 0.0)) > CLARIFICATION_TTL_SECONDS:
                    self._drop_pending(payload.clarification_id)
                    raise DemoAgentError(
                        "the clarification has expired; start a new shopping turn",
                        code="expired_clarification",
                    )
                # Phase 6C: validate/select a structured option BEFORE consuming the pending
                # question.  An unknown option must not burn the clarification.
                raw_clarification = pending.get("research_clarification")
                if raw_clarification is not None:
                    clarification = AgentResearchClarificationView.model_validate(
                        raw_clarification
                    )
                    option, error = resolve_research_option(
                        clarification,
                        option_id=payload.clarification_option,
                        message=payload.message,
                    )
                    if error is not None:
                        raise DemoAgentError(error, code="invalid_clarification_option")
                    if option is None:
                        # A free-text answer that is not one of the offered selections falls
                        # back to the safe, non-research choices.  It never becomes a fact.
                        option = next(
                            (
                                item for item in clarification.options
                                if item.action == "show_known_facts"
                            ),
                            next(
                                (
                                    item for item in clarification.options
                                    if item.action == "stop_research"
                                ),
                                None,
                            ),
                        )
                    if option is not None:
                        research_resume = ResearchResume(
                            clarification_id=payload.clarification_id or "",
                            option_id=option.option_id,
                            action=option.action,
                            attributes=tuple(option.attributes),
                            products=tuple(option.products),
                            known_facts=dict(clarification.known_facts),
                            inspected_products=tuple(clarification.inspected_products),
                            compared=bool(clarification.compared),
                        )
                elif payload.clarification_option is not None:
                    raise DemoAgentError(
                        "this clarification has no structured options",
                        code="invalid_clarification_option",
                    )
                raw_research = pending.get("research")
                if raw_research is not None:
                    previous_research = AgentResearchView.model_validate(raw_research)
                pending_clarification_count = int(pending.get("clarification_count", 0) or 0)
                pending_research_calls = int(pending.get("research_calls_used", 0) or 0)
                self._drop_pending(payload.clarification_id)
        else:
            research_resume = None
            previous_research = None
            pending_clarification_count = 0
            pending_research_calls = 0
        effective_constraints: tuple[AgentConstraintItem, ...] = tuple(payload.constraints)
        if (
            payload.clarification_id
            and not effective_constraints
            and previous_research is not None
            and pending is not None
            and pending.get("constraints")
        ):
            # A research clarification suspends the same shopping task.  Its hard constraints
            # remain in force unless this answer explicitly supplies a new set.
            effective_constraints = tuple(
                AgentConstraintItem.model_validate(item)
                for item in pending.get("constraints") or ()
            )
        snapshot = self._memory_service.get_active_preferences(session_id)
        if apply_user_state:
            # The canonical user turn owns state extraction.  A Planner RETRIEVE query may
            # differ from that utterance, so it must never become a new preference source.
            extraction = self._memory_service.extractor.extract(
                state_message or payload.message
            )
            self._update_task_price_constraints(task_key, extraction)
            # An uncertainty statement is a request for help, not a preference.  In the M7
            # experiment the rule extractor must not turn "I don't know which type..." into
            # an avoid/feature fact before the agent has shown any grounded examples.
            if grounded_actions_enabled() and is_uncertain_message(
                state_message or payload.message
            ):
                from recommendation.memory.schemas import PreferenceExtraction

                extraction = PreferenceExtraction()
        else:
            # Subsequent RETRIEVE/RERANK steps of the same user turn: reuse the canonical
            # state already written on the first step, and write no new preferences.
            from recommendation.memory.schemas import PreferenceExtraction

            extraction = PreferenceExtraction()
        r1_previous = self._r1_last_responses.get(task_key)
        r1_goal_correction = bool(
            dynamic_retrieval_enabled()
            and r1_previous is not None
            and _normal_turn_goal_correction(extraction)
        )
        if r1_goal_correction:
            extraction = _mark_goal_correction(extraction)
        if apply_user_state and dynamic_retrieval_enabled():
            old_choice = self._task_preferences.get(task_key, ())
            new_choice = _task_preferences_after_correction(
                old_choice,
                extraction,
                clear_all=r1_goal_correction,
            )
            if new_choice != old_choice:
                self._task_preferences[task_key] = new_choice
                self._persist_task(task_key)
            preference_choice = self._task_preferences.get(task_key, ())
        explicit = tuple(
            {"kind": getattr(item.kind, "value", str(item.kind)), "value": str(item.value)}
            for item in effective_constraints
        )
        explicit = tuple(dict(item) for item in explicit) + tuple(
            dict(item) for item in self._task_price_constraints.get(task_key, ())
        )
        # The accepted M9 memory turn id is separate from the M5.2 logical turn identity:
        # it addresses a preference-memory write, never a decision-state event.
        memory_turn_id = f"{session_id}:{payload.clarification_id or uuid.uuid4().hex[:12]}"
        intent = derive_intent(
            snapshot,
            extraction,
            turn_id=memory_turn_id,
            explicit_constraints=explicit,
        )
        terms = query_terms(intent, payload.message)
        if task_key in self._task_price_constraints:
            terms = _sanitize_price_query_terms(terms)
        r1_run_terms = tuple(terms)
        if (
            apply_user_state
            and dynamic_retrieval_enabled()
            and terms
            and task_key not in self._r1_task_anchor_terms
        ):
            self._r1_task_anchor_terms[task_key] = tuple(terms)
            self._persist_task(task_key)
        for preference in preference_choice:
            intent = augment_intent(intent, preference)
        # M5.2: record this turn's validated facts in the unified decision state BEFORE any
        # early return, so a suspended / conflict / awaiting-user turn still records the
        # explicit facts it carries.  OFF (the default) and SHADOW return the constraint
        # inputs unchanged; only APPLY_CONSTRAINTS extends them with the state's authorized
        # constraints and rebuilds the user-visible effective intent view.
        trusted_constraints = tuple(
            AgentConstraintItem(kind=str(item["kind"]), value=str(item["value"]))
            for item in intent.hard_constraints()
        )
        if apply_user_state:
            constraints, effective_intent, state_projection, state_constraints_changed = (
                self._integrate_state_turn(
                    payload=payload,
                    user_key=session_id,
                    task_id=task_key,
                    extraction=extraction,
                    intent=intent,
                    constraints=trusted_constraints,
                    turn_key=turn_key,
                    turn_identity_source=identity_source,
                    legacy_preference_count=len(
                        getattr(snapshot, "active_entries", ()) or ()
                    ),
                    proposal_actions=proposal_actions,
                )
            )
        else:
            # Subsequent Planner retrieval steps reuse the state applied on the first
            # step; they must not append M5 events or re-apply the turn's state operations.
            constraints = trusted_constraints
            effective_intent = intent
            state_projection = None
            state_constraints_changed = False
        if task_key in self._constraint_trace:
            self._constraint_trace[task_key]["returned_constraints"] = [
                [str(item.kind), str(item.value)] for item in constraints
            ]
        if (
            proposal_actions is not None
            and proposal_actions.valid
            and proposal_actions.constraints
        ):
            # Defence in depth for the proposal path: the M5-authored constraint is the
            # current user authority, so it must reach the same hard-constraint input the
            # legacy intent path uses.  Deduplicate by (kind, value) rather than replacing
            # any other still-active constraint.
            # Deduplicate by kind: a current proposal constraint replaces the same
            # kind's stale value rather than appending two limits.
            proposed_kinds = {str(item.kind.value).casefold() for item in proposal_actions.constraints}
            constraints = tuple(
                item for item in constraints if str(item.kind).casefold() not in proposed_kinds
            )
            added = tuple(
                AgentConstraintItem(kind=str(item.kind.value), value=str(item.value))
                for item in proposal_actions.constraints
            )
            if added:
                constraints = constraints + added
        if task_key in self._constraint_trace:
            self._constraint_trace[task_key]["final_constraints"] = [
                [str(item.kind), str(item.value)] for item in constraints
            ]
        if state_projection is not None:
            # APPLY_CONSTRAINTS only.  Hard constraints have already been merged into
            # `constraints`; recompute the catalogue terms from the effective intent so
            # an M5-authorized remembered requirement can influence R1's query rather
            # than only the post-retrieval eligibility filter.  Soft state preferences
            # stay SOFT in the effective intent and are never promoted to constraints.
            effective_terms = query_terms(effective_intent, payload.message)
            if effective_terms:
                terms = effective_terms
                r1_run_terms = tuple(effective_terms)
        if effective_intent.conflicts:
            response = self.run(
                payload.model_copy(update={"user_key": session_id, "constraints": ()}),
                clarification_question=effective_intent.conflicts[0].question,
                dynamic_research=True,
                planner_retrieval=planner_retrieval,
                apply_user_state=apply_user_state,
                state_message=state_message,
                source_policy=source_policy,
            )
            if apply_user_state:
                self._commit_clarify_turn(
                    session_id, state_message or payload.message, memory_turn_id
                )
            suspended = self._attach_clarification(
                response, effective_intent, task_key, kind="conflict"
            )
            return self._register_candidate_snapshot(
                suspended, user_key=session_id, task_id=task_key, turn_key=turn_key
            )
        # R1 normal-turn plan: a free-text goal correction/withdrawal invalidates the old
        # candidate window and must retrieve again; an unchanged/repeated request can reuse
        # the last presented response.  A turn carrying any new extracted preference is
        # treated conservatively as a re-retrieval, never as an unrelated candidate fill.
        r1_plan_action = "RETRIEVE"
        r1_plan_reason = "no_previous_candidate_window"
        if r1_previous is not None:
            r1_plan_reason = "goal_change_or_new_preference"
            if (
                not r1_goal_correction
                and not getattr(extraction, "preferences", ())
                and not getattr(extraction, "removals", ())
            ):
                r1_plan_action = "RERANK"
                r1_plan_reason = "previous_candidates_still_satisfy_request"
        if state_constraints_changed:
            # The M5-authorized hard set differs from the legacy/current-turn set.
            # R1 must re-run retrieval rather than reuse a candidate window created under
            # the previous effective constraints.
            r1_plan_action = "RETRIEVE"
            r1_plan_reason = "m5_effective_constraints_changed"
        if role_target_changed:
            # The user's explicit product type/role changed.  Even when the extraction is
            # soft or the current utterance is a short follow-up, the old candidate
            # window cannot be assumed to cover the new target.
            r1_plan_action = "RETRIEVE"
            r1_plan_reason = "grounded_role_target_changed"
        if r1_plan_action == "RERANK" and r1_previous is not None:
            self._r1_diagnostics[task_key] = {
                "action": "RERANK",
                "reason": r1_plan_reason,
                "query_terms": list(r1_run_terms),
                "before": [item.parent_asin for item in r1_previous.recommendations],
                "after": [item.parent_asin for item in r1_previous.recommendations],
                "constraints": [
                    {"kind": str(item.kind), "value": str(item.value)}
                    for item in constraints
                ],
                "m5_soft_preferences": (
                    [
                        {
                            "kind": item.kind.value,
                            "value": item.value,
                            "scope": item.scope.value,
                            "is_soft": item.is_soft,
                        }
                        for item in getattr(state_projection, "preferences", ()) or ()
                    ]
                    if state_projection is not None
                    else []
                ),
                "mode": "normal_turn",
            }
            reranked = self._attach_intent_and_reasons(r1_previous, effective_intent)
            if self._soft_m5_rerank and reranked.recommendations:
                soft_ranking, _combined, _soft, _soft_diag = self._m5_soft_rerank(
                    task_key,
                    reranked.recommendations,
                    existing=(),
                )
                if soft_ranking is not None:
                    ordered = soft_ranking.by_identity()
                    candidates = sorted(
                        reranked.recommendations,
                        key=lambda item: ordered[str(item.parent_asin)].reranked_rank,
                    )
                    candidates = [
                        item.model_copy(update={"rank": position})
                        for position, item in enumerate(candidates, start=1)
                    ]
                    reranked = reranked.model_copy(
                        update={
                            "recommendations": tuple(candidates),
                            "message": _consistent_message(
                                reranked.message,
                                candidates,
                                getattr(self._metadata, "records", {}) or {},
                            ),
                        }
                    )
                self._r1_diagnostics[task_key]["action"] = "RERANK"
                self._r1_diagnostics[task_key]["after"] = [
                    item.parent_asin for item in reranked.recommendations
                ]
                self._r1_diagnostics[task_key]["soft_rerank"] = _soft_diag
            self._m8_finalize(
                session_id=session_id,
                candidates=[item.parent_asin for item in reranked.recommendations],
            )
            return self._register_candidate_snapshot(
                reranked, user_key=session_id, task_id=task_key, turn_key=turn_key
            )
        if (
            dynamic_retrieval_enabled()
            and r1_previous is not None
            and getattr(extraction, "removals", ())
            and not getattr(extraction, "preferences", ())
        ):
            # A pure withdrawal must not keep using the withdrawn terms from the
            # correction.  Revert to the last non-withdrawn task query when available.
            anchor = self._r1_task_anchor_terms.get(task_key)
            if anchor:
                if task_key in self._task_price_constraints:
                    anchor = _sanitize_price_query_terms(anchor)
                    self._r1_task_anchor_terms[task_key] = anchor
                r1_run_terms = anchor
        elif (
            dynamic_retrieval_enabled()
            and r1_previous is None
            and extraction.is_empty
            and task_key in self._r1_task_anchor_terms
            and _is_more_options_request(payload.message)
        ):
            # After a service restart the in-process last candidate window is gone, but
            # the current task's durable query anchor still represents the valid memory.
            anchor = self._r1_task_anchor_terms[task_key]
            if task_key in self._task_price_constraints:
                anchor = _sanitize_price_query_terms(anchor)
            r1_run_terms = anchor
        # A normal follow-up such as "show me the best ones" has no product-type words.
        # Re-retrieval (for example because M5 constraints were re-projected) must not
        # replace the task's still-effective type with that generic follow-up.  Preserve
        # only the current effective target type; old query terms are deliberately not
        # reused, so withdrawn preferences and old budget wording cannot come back.
        # An explicit goal/role switch sets role_target_changed and keeps the new wording.
        if (
            r1_plan_action == "RETRIEVE"
            and not role_target_changed
            and not r1_goal_correction
        ):
            r1_run_terms = _with_effective_target_type(
                r1_run_terms, self._role_target_for(payload.user_key)
            )
        retrieval_sources = (
            ("catalog_search",)
            if r1_goal_correction and "catalog_search" in self.available_sources
            else tuple(payload.sources)
        )
        if task_key in self._constraint_trace:
            self._constraint_trace[task_key]["r1_actual"] = {
                "turn_key": turn_key,
                "action": r1_plan_action,
                "reason": r1_plan_reason,
                "constraints": [[str(item.kind), str(item.value)] for item in constraints],
                "query_terms": list(r1_run_terms),
            }
        response = self.run(
            payload.model_copy(
                update={
                    "user_key": session_id,
                    "constraints": constraints,
                    "sources": retrieval_sources,
                }
            ),            clarify_unresolved=task_key not in self._clarified_sessions,
            intent_terms=r1_run_terms,
            dynamic_research=True,
            research_resume=research_resume,
            research_budget_remaining=max(
                0, MAX_TASK_RESEARCH_CALLS - pending_research_calls
            ),
            clarification_budget_remaining=max(
                0, MAX_RESEARCH_CLARIFICATIONS - pending_clarification_count
            ),
            task_preferences=preference_choice,
            planner_retrieval=planner_retrieval,
            apply_user_state=apply_user_state,
            state_message=state_message,
            source_policy=source_policy,
        )
        if previous_research is not None:
            response = _merge_previous_research(
                response, previous_research, research_resume
            )
        # A clear product-type request (for example "water filter") must not be
        # interleaved with unrelated history candidates.  The lexical-source consensus
        # and the user's own words gate this conservative stable promotion; broad
        # requests and candidates without trusted categories are left unchanged.
        promoted = promote_explicit_type_matches(
            (
                " ".join(r1_run_terms)
                if dynamic_retrieval_enabled() and tuple(r1_run_terms) != tuple(terms)
                else payload.message
            ),
            response.recommendations,
            getattr(self._metadata, "records", {}),
        )
        if promoted != response.recommendations:
            response = response.model_copy(update={"recommendations": promoted})
            response = _reconcile_final_identity(response, self._metadata)
        else:
            response = _reconcile_final_identity(response, self._metadata)
        if response.terminal.termination_reason == "awaiting_user":
            self._commit_clarify_turn(session_id, payload.message, memory_turn_id)
            suspended = self._attach_clarification(
                response,
                effective_intent,
                task_key,
                kind="unverified",
                constraints=constraints,
                clarification_count=pending_clarification_count + 1,
            )
            return self._register_candidate_snapshot(
                suspended, user_key=session_id, task_id=task_key, turn_key=turn_key
            )
        if fresh_turn_override and _PENDING_WITHDRAW.search(payload.message):
            # An explicit withdrawal is not an invitation for a new preference
            # question.  Finish the current turn instead of immediately asking about
            # an unrelated attribute.
            preference_response = None
        else:
            preference_response = self._maybe_attach_preference_question(
                response, effective_intent, task_key, payload, terms=terms, user_key=session_id, turn_key=turn_key
            )
        if preference_response is not None:
            # A suspended preference question presents no candidate list, so there is no
            # trusted snapshot to register for it.
            return preference_response
        if grounded_actions_enabled():
            note = price_evidence_note(
                payload.message,
                tuple(item.parent_asin for item in response.recommendations),
                getattr(self._metadata, "records", {}) or {},
            )
            if note:
                response = response.model_copy(
                    update={"message": f"{response.message}\n\n{note}"}
                )
        final = self._attach_intent_and_reasons(response, effective_intent)
        # Fresh retrieval is a valid M5-soft-rerank path too.  Previously only the
        # candidate-reuse branch and the clarification-answer branch called M6.1 soft
        # rerank, so a newly confirmed soft preference could never affect a fresh
        # candidate window.
        if self._soft_m5_rerank and final.recommendations:
            soft_ranking, _combined, _soft, _soft_diag = self._m5_soft_rerank(
                task_key,
                final.recommendations,
                existing=(),
            )
            if soft_ranking is not None:
                ordered = soft_ranking.by_identity()
                candidates = sorted(
                    final.recommendations,
                    key=lambda item: ordered[str(item.parent_asin)].reranked_rank,
                )
                candidates = [
                    item.model_copy(update={"rank": position})
                    for position, item in enumerate(candidates, start=1)
                ]
                final = final.model_copy(
                    update={
                        "recommendations": tuple(candidates),
                        "message": _consistent_message(
                            final.message,
                            candidates,
                            getattr(self._metadata, "records", {}) or {},
                        ),
                    }
                )
            self._r1_diagnostics.setdefault(task_key, {})["soft_rerank"] = _soft_diag
        if dynamic_retrieval_enabled():
            before_ids = (
                tuple(item.parent_asin for item in r1_previous.recommendations)
                if r1_previous is not None
                else ()
            )
            after_ids = tuple(item.parent_asin for item in final.recommendations)
            self._r1_diagnostics[task_key] = {
                "action": r1_plan_action,
                "reason": r1_plan_reason,
                "query_terms": list(r1_run_terms),
                "sources": list(retrieval_sources),
                "before": list(before_ids),
                "after": list(after_ids),
                "added": [item for item in after_ids if item not in before_ids],
                "removed": [item for item in before_ids if item not in after_ids],
                "constraints": [
                    {"kind": str(item.kind), "value": str(item.value)}
                    for item in constraints
                ],
                "m5_soft_preferences": (
                    [
                        {
                            "kind": item.kind.value,
                            "value": item.value,
                            "scope": item.scope.value,
                            "is_soft": item.is_soft,
                        }
                        for item in getattr(state_projection, "preferences", ()) or ()
                    ]
                    if state_projection is not None
                    else []
                ),
                "goal_correction": r1_goal_correction,
                "task_preferences": [
                    {
                        "attribute": preference.attribute,
                        "action": preference.action,
                        "value": preference.value,
                    }
                    for preference in preference_choice
                ],
                "mode": "normal_turn",
            }
        return self._register_candidate_snapshot(
            final, user_key=session_id, task_id=task_key, turn_key=turn_key
        )

    # -- M5.1/M5.2 unified decision state (additive, opt-in) ---------------- #
    #
    # Scope of this integration: the multi-turn shopping loop only -- the branch with a
    # stable server-owned task identity that executes a recommendation plan, plus the
    # trusted clarification-answer branch (``_answer_preference``) and the suspension
    # branches.  The stateless :meth:`run` path has no stable task identity and is
    # deliberately not integrated.  Nothing here is a second agent or a second
    # recommender: the state engine only records validated facts and (in
    # APPLY_CONSTRAINTS) supplies user-authorized constraints to the existing trusted
    # eligibility path.  In SHADOW the presented answer is byte-identical to OFF; the
    # recorded state, projections, snapshots and diagnostics are observational only.

    @property
    def state_mode(self) -> str | None:
        """The composed integration's mode value, or ``None`` when not composed."""
        integration = self._state_integration
        if integration is None:
            return None
        mode = getattr(integration, "mode", None)
        return getattr(mode, "value", mode)

    def state_diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Bounded decision-state diagnostics for one task, or ``None`` when untracked.

        Present only when a decision-state integration is composed; it never contains the
        raw message, a support span or the user key.
        """
        integration = self._state_integration
        probe = getattr(integration, "diagnostics", None)
        if integration is None or not callable(probe):
            return None
        return probe(task_id)

    def m7_diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Bounded M7 experiment diagnostics for one task.

        Records whether the M6.1 refresh ran, whether it changed the order, and - when it
        did not run - the explicit reason and the accepted fallback.  It never contains a
        message, a candidate identity or a model score.
        """
        with self._lock:
            record = self._m7_refresh.get(str(task_id))
            return dict(record) if record is not None else None

    def r1_diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Bounded R1 retrieval-plan diagnostics for one task (experiment only)."""
        with self._lock:
            record = self._r1_diagnostics.get(str(task_id))
            return dict(record) if record is not None else None

    # -- M8 decision trajectories (off by default; experiment only) ---------- #

    def _m8_recorder(
        self, task_id: str, user_key: str, *, k: int = 5, budget: int = 2
    ) -> DecisionTrajectoryRecorder | None:
        if not decision_trajectory_enabled():
            return None
        with self._lock:
            recorder = self._m8_trajectories.get(str(task_id))
            if recorder is None:
                recorder = DecisionTrajectoryRecorder.for_service(
                    task_id=str(task_id),
                    user_key=str(user_key),
                    arm="A2",
                    seed=0,
                    budget=max(1, int(budget)),
                    k=max(1, int(k)),
                )
                self._m8_trajectories[str(task_id)] = recorder
            return recorder

    def decision_trajectory(self, task_id: str) -> dict[str, Any] | None:
        """Return the recorded M8 trajectory for one task (experiment switch only)."""
        if not decision_trajectory_enabled():
            return None
        with self._lock:
            recorder = self._m8_trajectories.get(str(task_id))
            return recorder.trajectory().as_dict() if recorder is not None else None

    def _decision_choice_store(self) -> DecisionChoiceStore | None:
        """Lazily open the M8.1 SQLite store only when collection is enabled."""
        if not choice_collection_enabled():
            return None
        if self._m8_choice_store is not None:
            return self._m8_choice_store
        if self._m8_choice_store_error is not None:
            raise DemoAgentError(self._m8_choice_store_error, code="m8_store_failed")
        try:
            database = choice_database_path()
            self._m8_choice_store = DecisionChoiceStore(database)
        except DecisionChoiceStoreError as exc:
            self._m8_choice_store_error = str(exc)
            raise DemoAgentError(
                "the decision-choice store could not be opened",
                code="m8_store_failed",
            ) from exc
        return self._m8_choice_store

    def decision_choice_history(self, session_id: str) -> tuple[dict[str, Any], ...]:
        """Read persisted real choices for one session (survives process restart)."""
        store = self._decision_choice_store()
        if store is None:
            return ()
        return tuple(choice.as_dict() for choice in store.choices_for_session(session_id))

    def record_decision_choice(
        self,
        *,
        task_id: str,
        user_key: str,
        session_id: str,
        selected_parent_asin: str | None = None,
        no_suitable_item: bool = False,
        satisfaction: int | None = None,
        confidence: int | None = None,
        comment: str | None = None,
        idempotency_key: str | None = None,
        interaction_source: str = "internal",
    ) -> dict[str, Any]:
        """Record a real user's final selection / no-suitable-item decision.

        Only enabled under the M8 experiment switch.  It validates user/task/session
        identity and that the selected item was in the recorded final top-K.  When the
        M8.1 collection switch is on, only the demo HTTP route (``interaction_source ==
        "demo_http"``) persists a record; a direct internal call stays contract-only and
        cannot be re-read as a real user choice.  Satisfaction/confidence are optional
        feedback and are never interpreted as correctness.
        """
        if not decision_trajectory_enabled():
            raise DemoAgentError(
                "decision-level evaluation is disabled",
                code="m8_disabled",
            )
        if no_suitable_item and selected_parent_asin is not None:
            raise DemoAgentError(
                "no-suitable-item and a selected candidate are mutually exclusive",
                code="m8_invalid_choice",
            )
        if not no_suitable_item and selected_parent_asin is None:
            raise DemoAgentError(
                "a selected candidate or no_suitable_item=True is required",
                code="m8_invalid_choice",
            )
        for name, value in (("satisfaction", satisfaction), ("confidence", confidence)):
            if value is not None and (not isinstance(value, int) or not 1 <= value <= 5):
                raise DemoAgentError(
                    f"{name} must be an integer in 1..5 when supplied",
                    code="m8_invalid_choice",
                )
        key = str(task_id)
        with self._lock:
            recorder = self._m8_trajectories.get(key)
            if recorder is None:
                raise DemoAgentError(
                    "no decision trajectory for this task",
                    code="m8_unknown_task",
                )
            if recorder.user_key != str(user_key) or key != str(session_id):
                raise DemoAgentError(
                    "the choice identity does not match the recorded task",
                    code="m8_identity_mismatch",
                )
            try:
                if no_suitable_item:
                    recorder.record_no_suitable_choice(source="user")
                else:
                    recorder.record_choice(str(selected_parent_asin), source="user")
            except ValueError as exc:
                raise DemoAgentError(str(exc), code="m8_invalid_choice") from exc
            trajectory = recorder.trajectory()
            result = trajectory.as_dict()
            result["digest"] = trajectory.digest()
            store = self._decision_choice_store()
            if store is not None and interaction_source == "demo_http":
                try:
                    collected = store.record(
                        session_id=str(session_id),
                        task_id=str(task_id),
                        user_key=str(user_key),
                        trajectory=result,
                        selected_parent_asin=selected_parent_asin,
                        no_suitable_item=bool(no_suitable_item),
                        satisfaction=satisfaction,
                        confidence=confidence,
                        comment=comment,
                        idempotency_key=idempotency_key,
                        interaction_source=interaction_source,
                    )
                except DecisionChoiceConflict as exc:
                    raise DemoAgentError(
                        "a different choice was already recorded for this task",
                        code="m8_choice_conflict",
                    ) from exc
                except DecisionChoiceStoreError as exc:
                    raise DemoAgentError(
                        "the decision choice could not be persisted",
                        code="m8_store_failed",
                    ) from exc
                result["collected"] = collected.as_dict()
            return result

    def _resolve_turn_identity(
        self,
        payload: AgentRecommendRequest,
        *,
        task_key: str,
        server_turn_id: str | None,
        idempotency_key: str | None,
    ) -> tuple[str, str]:
        """Resolve one logical M5 turn identity; report which source produced it.

        Off (no integration composed, or mode ``off``) returns ``("", "unspecified")``
        without importing the state engine, so an OFF deployment pays nothing.  In every
        other mode the precedence is the adapter's: client ``idempotency_key``, then the
        server-owned turn id, then the content digest fallback.  The server id is what the
        HTTP route always supplies, so two distinct identical messages never collapse;
        a retry reusing the same client key reuses the same event ids.
        """
        integration = self._state_integration
        if integration is None or not bool(getattr(integration, "enabled", False)):
            return "", "unspecified"

        from recommendation.state import shopping_adapter

        fallback = shopping_adapter.request_turn_key(
            message=payload.message,
            k=payload.k,
            sources=payload.sources,
            constraints=[(str(item.kind), str(item.value)) for item in payload.constraints],
            clarification_id=payload.clarification_id,
        )
        return shopping_adapter.resolve_turn_identity(
            task_id=task_key,
            idempotency_key=idempotency_key,
            server_turn_id=server_turn_id,
            fallback_content_key=fallback,
        )

    def _update_task_price_constraints(self, task_key: str, extraction: Any) -> None:
        """Track the latest explicit task-scoped price bounds across follow-up turns.

        This is a task-effective projection only.  State events/history are untouched;
        old values remain auditable but cannot become the effective constraint again
        merely because a later turn contains no price expression.
        """
        current = {
            str(item.get("kind", "")): dict(item)
            for item in self._task_price_constraints.get(task_key, ())
        }
        for preference in getattr(extraction, "preferences", ()) or ():
            kind = str(getattr(getattr(preference, "kind", ""), "value", getattr(preference, "kind", "")))
            if kind not in {"price_max", "price_min"}:
                continue
            current[kind] = {
                "kind": kind,
                "value": str(getattr(preference, "value", "")),
                "source_text": str(getattr(preference, "source_text", "") or ""),
            }
        for removal in getattr(extraction, "removals", ()) or ():
            kind = str(getattr(getattr(removal, "kind", None), "value", getattr(removal, "kind", "")))
            if kind in current:
                current.pop(kind, None)
        if current:
            self._task_price_constraints[task_key] = tuple(current.values())
        else:
            self._task_price_constraints.pop(task_key, None)

    def _m5_soft_rerank(
        self,
        task_key: str,
        candidates: Sequence[Any],
        *,
        existing: Sequence[Any] = (),
    ) -> tuple[Any | None, tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
        """Run M6.1 over a fixed candidate set with current M5 soft preferences."""
        projection = self._ranker_projections.get(task_key) if self._soft_m5_rerank else None
        records = getattr(self._metadata, "records", {}) or {}
        active_features = [
            feature
            for feature in (getattr(projection, "features", ()) or ())
            if not bool(getattr(feature, "is_soft", True))
            and str(getattr(getattr(feature, "lifecycle", ""), "value", getattr(feature, "lifecycle", "")))
            .strip()
            .casefold()
            in {"explicit", "confirmed"}
        ] if projection is not None else []
        ranking, combined, soft = rank_with_m5_soft_preferences(
            candidates,
            records,
            projection,
            existing=existing,
        )
        previous_count = self._soft_last_counts.get(task_key, 0)
        previous_consumed = self._soft_last_consumed_counts.get(task_key, 0)
        active_count = len(active_features)
        consumed_count = len(soft)
        unknown_count = (
            sum(int(getattr(item, "unknown_count", 0) or 0) for item in ranking.candidates)
            if ranking is not None
            else 0
        )
        if consumed_count == 0:
            reason = (
                "no_active_m5_soft_preference"
                if active_count == 0
                else "active_m5_soft_preference_not_supported_by_m6_1"
            )
        elif ranking is None:
            reason = "m6_1_rerank_not_applicable"
        elif ranking.changed_order:
            reason = "m5_soft_rerank_changed_order"
        else:
            reason = "m5_soft_rerank_same_order"
        diagnostic = {
            "active_soft_preference_count": active_count,
            "consumed_soft_preference_count": consumed_count,
            "soft_preference_source": "m5_for_ranker" if consumed_count else "none",
            "rerank_applied": bool(ranking is not None),
            "ranking_changed": bool(ranking is not None and ranking.changed_order),
            "candidate_identity_changed": False,
            "unknown_evidence_count": unknown_count,
            "retraction_effective": bool(previous_consumed > 0 and consumed_count == 0),
            "ranking_reason": reason,
            "consumed_preferences": [
                {
                    "preference_id": item.preference_id,
                    "attribute": item.attribute,
                    "action": item.action,
                    "value": item.value,
                    "scope": item.scope,
                }
                for item in soft
            ],
            "candidate_count": len(candidates),
        }
        self._soft_rerank_diagnostics[task_key] = diagnostic
        self._soft_last_counts[task_key] = active_count
        self._soft_last_consumed_counts[task_key] = consumed_count
        return ranking, tuple(combined), tuple(soft), diagnostic

    def soft_m5_diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Read-only P1B-3 soft-rerank diagnostics for one task."""
        with self._lock:
            record = self._soft_rerank_diagnostics.get(str(task_id))
            return dict(record) if record is not None else None

    def _update_role_target(self, task_key: str, message: str) -> bool:
        """Maintain the task's explicit type/role target across follow-up turns.

        Returns True only when the effective target changes.  A pure withdrawal or
        "show more options" turn must not reclassify the target from its own wording.  A
        new main target replaces the main target; an accessory request is temporary and a
        withdrawal falls back to the most recent main target.
        """
        target = infer_target(message)
        previous = self._role_current_targets.get(task_key)
        previous_main = self._role_main_targets.get(task_key)
        before = (previous.type, previous.role) if previous is not None else None
        if _PENDING_WITHDRAW.search(message):
            fallback = previous_main or previous
            if fallback is not None:
                self._role_current_targets[task_key] = fallback
        elif target.type is not None:
            self._role_current_targets[task_key] = target
            if target.role is not TargetRole.ACCESSORY:
                self._role_main_targets[task_key] = target
        elif target.role is TargetRole.ACCESSORY and previous is not None:
            # An accessory request can omit the parent product type ("replacement
            # mouthpiece instead").  Preserve the current type and only change the role;
            # otherwise the old main-product candidate window is wrongly reused.
            self._role_current_targets[task_key] = ProductTarget(previous.type, TargetRole.ACCESSORY)
        after_target = self._role_current_targets.get(task_key)
        after = (after_target.type, after_target.role) if after_target is not None else None
        return before != after

    def _mirror_role_target_to_user_key(self, task_key: str, user_key: str | None) -> None:
        """Mirror one task's current role target onto its session-owned namespace.

        ``task_key`` is the concrete shopping session (state owner); ``user_key`` is the
        session's M9 memory namespace, which is what ``_execute_plan`` reads.  Both are
        distinct on the HTTP route, so the rule path needs this explicit bridge.  The
        namespace is created per demo session; no other session can receive this target.
        """
        if not user_key:
            return
        target = self._role_current_targets.get(task_key)
        if target is None:
            return
        self._role_current_targets[str(user_key)] = target
        main_target = self._role_main_targets.get(task_key)
        if main_target is not None:
            self._role_main_targets[str(user_key)] = main_target

    def prime_task_role_target(
        self, *, task_key: str, user_key: str | None, message: str
    ) -> bool:
        """Seed this user turn's explicit role target before a Planner episode.

        The controlled runner applies one user turn's state exactly once.  A Planner
        executor then performs SELECT/RETRIEVE/RERANK steps using its own query text;
        seeding the role target from the **real user message** before the episode keeps
        the role state independent of any rewritten retrieval query.  No TargetRole rule
        is changed: this calls the same accepted ``_update_role_target`` + session mirror.
        Returns whether the effective target changed.
        """
        changed = self._update_role_target(str(task_key), message)
        self._mirror_role_target_to_user_key(str(task_key), user_key)
        return changed

    def _role_target_for(self, user_key: str | None) -> Any | None:
        return self._role_current_targets.get(str(user_key or "")) if user_key else None

    def role_diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Read-only role-selection diagnostics for one task (benchmark/audit only)."""
        with self._lock:
            value = self._role_diagnostics.get(str(task_id))
            return dict(value) if value is not None else None

    def proposal_diagnostics(self, task_id: str) -> dict[str, Any] | None:
        """Read-only LLM proposal -> M5 diagnostics for one task."""
        with self._lock:
            value = self._proposal_diagnostics.get(str(task_id))
            return dict(value) if value is not None else None

    def candidate_filter_diagnostics(self, session_id: str) -> dict[str, Any] | None:
        """Read-only candidate-filter trace for one session (P3C-2 diagnostic)."""
        with self._lock:
            value = self._candidate_filter_diagnostics.get(str(session_id))
            return dict(value) if value is not None else None

    def constraint_trace(self, task_id: str) -> dict[str, Any] | None:
        """Read-only per-turn constraint provenance trace for one task."""
        with self._lock:
            value = self._constraint_trace.get(str(task_id))
            return dict(value) if value is not None else None


    def _integrate_state_turn(
        self,
        *,
        payload: AgentRecommendRequest,
        user_key: str,
        task_id: str,
        extraction: Any,
        intent: Any,
        constraints: tuple[AgentConstraintItem, ...],
        turn_key: str,
        turn_identity_source: str,
        legacy_preference_count: int | None,
        proposal_actions: Any | None = None,
    ) -> tuple[tuple[AgentConstraintItem, ...], Any, Any, bool]:
        """Integrate one validated planning turn and return the actual planner input.

        ``OFF`` (no integration composed, or its mode is off) and ``SHADOW`` return the
        unchanged constraint inputs, with no applied state projection.  ``APPLY_CONSTRAINTS``
        returns the merged hard constraints, an effective intent containing the M5 soft
        projection separately, the ``for_agent`` projection (for diagnostics), and whether
        the hard set changed relative to the legacy/current-turn input.
        """
        integration = self._state_integration
        if integration is None or not bool(getattr(integration, "enabled", False)):
            return constraints, intent, None, False

        from recommendation.state import shopping_adapter

        from .state_integration import StateIntegrationError

        trusted_pairs = tuple(
            (str(item.kind), str(item.value)) for item in constraints
        )
        # Imported here (not at module scope) to keep the agent service's startup free of
        # the state engine until a mode actually asks for it.
        preferences = []
        for item in getattr(extraction, "preferences", ()) or ():
            replaces_value, replace_same_kind = self._replacement_fields(item)
            preferences.append(
                shopping_adapter.ExplicitPreference(
                    # A malformed kind is passed through as-is; the adapter (inside the
                    # integration) refuses it explicitly, which is what the mode-specific
                    # failure policy needs to see.  Nothing is coerced to a fallback kind.
                    kind=_enum_text(getattr(item, "kind", ""), ""),
                    value=str(item.value),
                    polarity=_enum_text(getattr(item, "polarity", "prefer"), "prefer"),
                    source_text=str(getattr(item, "source_text", "") or ""),
                    replaces_value=replaces_value,
                    replace_same_kind=replace_same_kind,
                )
            )
        preferences = tuple(preferences)
        retractions = self._retractions_from_extraction(
            extraction, shopping_adapter, message=payload.message
        )
        explicit_constraints = tuple(
            shopping_adapter.ExplicitConstraint(
                kind=str(entry.kind),
                value=str(entry.value),
                polarity=str(entry.polarity),
                source_text=str(entry.source_text or ""),
            )
            for entry in (getattr(intent, "hard", ()) or ())
        )
        if proposal_actions is not None:
            # LLM proposal is the sole same-turn M5 input.  Disable the rule extractor's
            # duplicate state writes and the legacy trusted constraints for this turn.
            trusted_pairs = ()
            if proposal_actions.valid:
                preferences = tuple(proposal_actions.preferences)
                retractions = tuple(proposal_actions.retractions)
                explicit_constraints = tuple(proposal_actions.constraints)
            else:
                preferences = ()
                retractions = ()
                explicit_constraints = ()
        try:
            outcome = integration.integrate_turn(
                user_key=user_key,
                session_id=task_id,
                task_id=task_id,
                turn_key=turn_key,
                message=payload.message,
                intent_label=_task_state(payload).intent.value,
                preferences=preferences,
                constraints=explicit_constraints,
                retractions=retractions,
                trusted_constraints=trusted_pairs,
                feedback_kind=_feedback_kind(extraction),
                turn_identity_source=turn_identity_source,
                legacy_preference_count=legacy_preference_count,
            )
        except StateIntegrationError as exc:
            # Fail closed: no fallback to the legacy constraint behaviour.  The route maps
            # this bounded code onto its documented 502 agent failure.
            raise DemoAgentError(
                "the decision-state integration failed in experimental mode",
                code="state_integration_failed",
            ) from exc
        mode = getattr(integration, "mode", None)
        mode_value = getattr(mode, "value", mode)
        if outcome is None:
            if mode_value == "apply_constraints":
                # Defence in depth: APPLY_CONSTRAINTS must never silently fall back to the
                # legacy constraints.  The integration raises on failure, so a missing
                # outcome here would itself be a contract violation -- fail closed anyway.
                raise DemoAgentError(
                    "the decision-state integration failed in experimental mode",
                    code="state_integration_failed",
                )
            # SHADOW (and a SHADOW failure) never change the inputs or the intent view.
            if self._soft_m5_rerank:
                from recommendation.state.state_projector import for_ranker

                self._ranker_projections[task_id] = for_ranker(outcome.state)
                self._ranker_projections[str(user_key)] = self._ranker_projections[task_id]
            return constraints, intent, None, False
        if mode_value != "apply_constraints":
            if self._soft_m5_rerank:
                from recommendation.state.state_projector import for_ranker

                projection = for_ranker(outcome.state)
                self._ranker_projections[task_id] = projection
                self._ranker_projections[str(user_key)] = projection
            return constraints, intent, None, False
        if proposal_actions is not None:
            # A proposal is not a trusted rule-extracted constraint for this turn:
            # its constraints are represented by the M5 authorized projection below.
            # Suppressing by kind here would drop the only current constraint.
            current_turn_kinds: tuple[str, ...] = ()
        else:
            # Rule mode: only constraints that are actually in this turn's trusted
            # pair set may suppress an M5-authorized value.  Historical
            # _task_price_constraints must not be treated as current authority.
            current_turn_kinds = tuple(kind for kind, _value in trusted_pairs)
        merged = shopping_adapter.merge_constraint_pairs(
            trusted_pairs,
            outcome.authorized_constraints,
            current_turn_kinds=current_turn_kinds,
        )
        self._constraint_trace[task_id] = {
            "trusted_pairs": [list(item) for item in trusted_pairs],
            "proposal_constraints": [
                [str(item.kind.value), str(item.value)]
                for item in (proposal_actions.constraints if proposal_actions is not None else ())
            ],
            "authorized_constraints": [list(item) for item in outcome.authorized_constraints],
            "current_turn_kinds": list(current_turn_kinds),
            "merged": [list(item) for item in merged],
        }
        from recommendation.state.state_projector import for_agent, for_ranker

        state_projection = for_agent(outcome.state)
        if self._soft_m5_rerank:
            self._ranker_projections[task_id] = for_ranker(outcome.state)
            self._ranker_projections[str(user_key)] = self._ranker_projections[task_id]
        # Soft M5 beliefs are exposed in the bounded state projection for inspection,
        # but are deliberately not appended to the user-visible/planner intent here:
        # this phase wires only the user-explicit hard channel.  A soft preference must
        # not become a hard constraint or silently change the legacy intent view.
        effective_intent = _effective_intent(intent, merged)
        authorized = tuple(outcome.authorized_constraints)
        previous_authorized = self._m5_authorized_constraints.get(task_id)
        changed = (
            previous_authorized is not None
            and tuple(previous_authorized) != tuple(authorized)
        )
        self._m5_authorized_constraints[task_id] = authorized
        return (
            tuple(AgentConstraintItem(kind=kind, value=value) for kind, value in merged),
            effective_intent,
            state_projection,
            changed,
        )

    @staticmethod
    def _retractions_from_extraction(
        extraction: Any, shopping_adapter: Any, *, message: str = ""
    ) -> tuple[Any, ...]:
        """Map the accepted M9 ``REMOVE`` output onto adapter retraction selectors.

        A removal carries exactly one selector (kind or value) by the M9 schema; an object
        with neither is not a retraction and is skipped rather than converted into an
        invented one.  Value-only selectors are resolved by the integration against the
        current task state, never guessed here.
        """
        retractions = []
        for removal in getattr(extraction, "removals", ()) or ():
            kind = _enum_text(getattr(removal, "kind", None))
            value = getattr(removal, "value", None)
            if kind is None and not str(value or "").strip():
                continue
            if kind is None and str(value or "").strip().casefold() in {"it", "that", "this", "them"}:
                # The rule extractor uses a pronoun for "clear it."; use the explicit
                # budget vocabulary in the same user message to select the hard-price
                # retraction without guessing a value.
                lowered = " ".join(str(message).casefold().split())
                if any(term in lowered for term in ("budget", "price", "spend", "cost", "maximum", "ceiling")):
                    from recommendation.state.schemas import PreferenceKind

                    kind = PreferenceKind.PRICE_MAX
                    value = None
            retractions.append(
                shopping_adapter.ExplicitRetraction(
                    kind=kind,
                    value=(None if value is None else str(value)),
                    source_text=str(getattr(removal, "source_text", "") or ""),
                )
            )
        return tuple(retractions)

    def _register_candidate_snapshot(
        self,
        response: AgentRecommendResponse,
        *,
        user_key: str,
        task_id: str,
        turn_key: str,
    ) -> AgentRecommendResponse:
        """Register the trusted candidate/ranking snapshot for a presented response.

        Only a **presented** answer has a snapshot: a suspended run (question, conflict or
        awaiting-user) has an empty recommendation list and is a no-op here.  The snapshot
        is also skipped when the state integration failed for this turn: the state does not
        describe the turn, so attaching a trusted reference to it would create a
        presentation record the decision state cannot justify.  OFF never reaches the
        integration at all.

        Every identity passed here comes from the response's own trusted views
        (``grounded.candidates`` and the presented ``recommendations`` order); the
        integration stores only opaque refs/counts/digests.  ``APPLY_CONSTRAINTS`` fails
        closed on a snapshot failure; SHADOW never lets it change the answer.
        """
        integration = self._state_integration
        if integration is None or not bool(getattr(integration, "enabled", False)):
            return response
        if not response.recommendations:
            return response
        # A failed turn's diagnostics record proves the state engine did not record it:
        # never attach a trusted snapshot to a turn the state could not record.
        probe = getattr(integration, "diagnostics", None)
        if callable(probe):
            record = probe(task_id)
            if isinstance(record, dict) and record.get("error") is not None:
                return response
        candidate_identities = tuple(
            str(item.parent_asin) for item in response.grounded.candidates
        )
        ranking_identities = tuple(
            str(item.parent_asin) for item in response.recommendations
        )
        ranking_method = (
            "task_preference_rerank"
            if response.preference_outcome is not None
            else "presented_order"
        )
        from .state_integration import StateIntegrationError

        try:
            integration.record_candidate_snapshot(
                user_key=user_key,
                session_id=task_id,
                task_id=task_id,
                turn_key=turn_key,
                candidate_identities=candidate_identities,
                ranking_identities=ranking_identities,
                ranking_method=ranking_method,
            )
        except StateIntegrationError as exc:
            raise DemoAgentError(
                "the decision-state integration failed in experimental mode",
                code="state_integration_failed",
            ) from exc
        if dynamic_retrieval_enabled() and response.recommendations:
            self._r1_last_responses[task_id] = response
        return response

    @staticmethod
    def _replacement_fields(candidate: Any) -> tuple[str | None, bool]:
        """The typed replacement semantics of one explicit extraction candidate.

        Returns ``(replaces_value, replace_same_kind)``.  A named correction ("blue
        instead of black") supersedes exactly that value.  A general correction the
        accepted M9 extractor recognised ("... instead.") has no named target: it may only
        supersede same-kind / same-polarity ACTIVE beliefs, which the M5 event expresses
        through the typed ``replace_same_kind`` flag -- a target value is never invented.
        An ordinary statement (``ADD``/``REMOVE``) carries neither.
        """
        mode_value = _enum_text(getattr(candidate, "mode", None))
        replacing = mode_value == "replace"
        replaces = getattr(candidate, "replaces", None)
        if replacing and replaces is not None:
            return str(replaces), False
        return None, replacing

    def _commit_clarify_turn(self, session_id: str, message: str, turn_id: str) -> None:
        """Persist the statement that led to a question, exactly once.

        A suspended run does not commit the turn (the loop commits only on a completed
        recommendation).  Without this, the conflict or constraint that caused the question
        would be forgotten before the user answers.  The write is the accepted memory
        service's own; it stores the user's message, never the question.
        """
        self._memory_service.process_turn(
            user_key=session_id,
            user_message=message,
            turn_id=turn_id,
        )

    def _attach_intent_and_reasons(
        self, response: AgentRecommendResponse, intent: Any
    ) -> AgentRecommendResponse:
        """Attach one evidence-bound reason per final item, plus the intent that produced it."""
        records = getattr(self._metadata, "records", {}) or {}
        recommendations = tuple(
            item.model_copy(
                update={"reason": build_reason(records.get(item.parent_asin), intent)}
            )
            for item in response.recommendations
        )
        return response.model_copy(
            update={"recommendations": recommendations, "intent": _intent_view(intent)}
        )

    def _answer_preference(
        self,
        payload: AgentRecommendRequest,
        pending: Mapping[str, Any],
        session_id: str,
        *,
        user_key: str,
        turn_key: str = "",
        turn_identity_source: str = "unspecified",
    ) -> AgentRecommendResponse:
        """Apply a confirmed preference to the same candidate set that produced the question.

        Rerunning retrieval here would let a memory write from the first turn change the
        candidate universe under the counterfactual test.  The preference is therefore applied
        to the exact response stored with the pending question; this is both cheaper and the
        only way the before/expected/actual top-K chain is causally comparable.

        M5.2: a resolved, executable option is also a validated explicit feedback fact, so it
        is recorded in the decision state with ``feedback_kind="clarification_option"`` before
        the answer is applied.  A declined/ambiguous answer (``none``/``skip``) records only
        the task-context event: no preference belief is created from "no preference".
        """
        prompt = _preference_prompt_from_payload(pending.get("preference_prompt"))
        with self._lock:
            if pending.get("session_id") != session_id:
                raise DemoAgentError(
                    "the clarification id is unknown or belongs to another session",
                    code="unknown_clarification",
                )
            if time.time() - float(pending.get("created_at", 0.0)) > CLARIFICATION_TTL_SECONDS:
                self._drop_pending(payload.clarification_id)
                raise DemoAgentError(
                    "the clarification has expired; start a new shopping turn",
                    code="expired_clarification",
                )
        option, error = resolve_preference_option(
            prompt,
            option_id=payload.clarification_option,
            message=payload.message,
        )
        if error is not None:
            raise DemoAgentError(error, code="invalid_preference_option")
        assert option is not None
        # Validate the option (already done above) and record the trusted feedback fact
        # BEFORE the pending question is consumed: a fail-closed APPLY failure must leave
        # the question answerable instead of silently eating it.
        outcome = self._integrate_feedback_choice(
            payload=payload,
            option=option,
            session_id=session_id,
            user_key=user_key,
            turn_key=turn_key,
            turn_identity_source=turn_identity_source,
        )
        if pending.get("grounded_action") == GroundedAction.COMPARE.value:
            self._record_grounded_interaction_events(
                action=GroundedAction.COMPARE,
                prompt=prompt,
                session_id=session_id,
                user_key=user_key,
                turn_key=turn_key,
                payload=payload,
                option_label=option.label,
            )
        with self._lock:
            stored = pending.get("request") or {}
            base = AgentRecommendResponse.model_validate(pending.get("base_response"))
            self._drop_pending(payload.clarification_id)
            round_no = int(pending.get("round") or 1)
            self._preference_rounds[session_id] = max(
                self._preference_rounds.get(session_id, 0), round_no
            )
            current = self._task_preferences.get(session_id, ())
            selected_pref: TaskPreference | None = None
            if option.action in {"none", "skip"}:
                prefs = current
                self._preference_second_enabled[session_id] = False
            else:
                selected = preference_from_option(
                    prompt, option, turn_id=payload.clarification_id
                )
                selected_pref = selected
                prefs = (
                    current
                    if any(item.attribute == selected.attribute for item in current)
                    else (*current, selected)
                )
                self._task_preferences[session_id] = prefs
                self._preference_second_enabled[session_id] = round_no < 2
            self._persist_task(session_id)
        request_payload = AgentRecommendRequest(
            message=str(stored.get("message") or payload.message),
            history=list(payload.history),
            k=int(stored.get("k") or payload.k),
            sources=tuple(stored.get("sources") or payload.sources),
            constraints=tuple(
                AgentConstraintItem.model_validate(item)
                for item in (stored.get("constraints") or ())
            ),
            preference_discovery=bool(
                stored.get("preference_discovery", payload.preference_discovery)
            ),
        )
        snapshot = self._memory_service.get_active_preferences(user_key)
        extraction = self._memory_service.extractor.extract(request_payload.message)
        explicit = tuple(
            {
                "kind": getattr(item.kind, "value", str(item.kind)),
                "value": str(item.value),
            }
            for item in request_payload.constraints
        )
        intent = derive_intent(
            snapshot,
            extraction,
            turn_id=f"{session_id}:preference",
            explicit_constraints=explicit,
        )
        terms = query_terms(intent, request_payload.message)
        for preference in prefs:
            intent = augment_intent(intent, preference)
        self._m8_feedback(
            session_id=session_id,
            prompt=prompt,
            option=option,
            preferences=prefs,
        )
        r1_rerank_only = False
        if selected_pref is not None and dynamic_retrieval_enabled():
            retrieval_key = f"{session_id}:{turn_key or payload.clarification_id or 'preference'}"
            before_ids = tuple(item.parent_asin for item in base.recommendations)
            plan = plan_dynamic_retrieval(
                request_payload.message,
                before_ids,
                getattr(self._metadata, "records", {}) or {},
                active_preferences=prefs,
                changed_preferences=(selected_pref,),
                intent=intent,
                sources=request_payload.sources,
                available_sources=self.available_sources,
                already_retrieved=retrieval_key in self._r1_retrieval_attempts,
            )
            diagnostic: dict[str, Any] = {
                "action": plan.action,
                "reason": plan.reason,
                "sources": list(plan.sources),
                "query_terms": list(plan.query_terms),
                "before": list(before_ids),
                "matched_candidates": plan.matched_candidates,
                "changed_attributes": list(plan.changed_attributes),
            }
            if plan.should_retrieve:
                self._r1_retrieval_attempts.add(retrieval_key)
                try:
                    retry_payload = request_payload.model_copy(
                        update={
                            "sources": plan.sources,
                            "preference_discovery": False,
                        }
                    )
                    fresh = self.run(
                        retry_payload,
                        intent_terms=plan.query_terms,
                        task_preferences=tuple(prefs),
                    )
                except Exception as exc:  # noqa: BLE001 - R1 must never break the answer
                    diagnostic["error"] = type(exc).__name__
                    diagnostic["fallback"] = "accepted_rerank_path"
                    self._r1_diagnostics[session_id] = diagnostic
                else:
                    # Keep the retrieval result inside the same trusted type scope the
                    # final shopping answer uses; the fresh lexical query can otherwise
                    # pull in products that merely share the preference value (for
                    # example orange tent stakes for an orange-cookware request).
                    promoted = promote_explicit_type_matches(
                        request_payload.message,
                        fresh.recommendations,
                        getattr(self._metadata, "records", {}) or {},
                    )
                    if promoted != fresh.recommendations:
                        fresh = fresh.model_copy(update={"recommendations": promoted})
                    after_ids = tuple(item.parent_asin for item in fresh.recommendations)
                    diagnostic["after"] = list(after_ids)
                    diagnostic["fresh_sources"] = list(
                        getattr(getattr(fresh, "grounded", None), "sources_present", ()) or ()
                    )
                    diagnostic["added"] = [item for item in after_ids if item not in before_ids]
                    diagnostic["removed"] = [item for item in before_ids if item not in after_ids]
                    self._r1_diagnostics[session_id] = diagnostic
                    if fresh.recommendations:
                        self._m8_finalize(
                            session_id=session_id,
                            candidates=list(after_ids),
                        )
                        retrieved = self._attach_intent_and_reasons(fresh, intent)
                        return self._register_candidate_snapshot(
                            retrieved,
                            user_key=user_key,
                            task_id=session_id,
                            turn_key=turn_key,
                        )
                    diagnostic["fallback"] = "empty_retrieval"
                    self._r1_diagnostics[session_id] = diagnostic
            else:
                self._r1_diagnostics[session_id] = diagnostic
                # R1 owns the recall decision when enabled: a RERANK plan must not be
                # silently overridden by the legacy "candidate count < k" retry.  That
                # retry can pull unrelated history items into a type-scoped answer.
                r1_rerank_only = plan.action == "RERANK"

        if option.action in {"none", "skip"}:
            # P1B-3: a final clarification answer is still a final ranking path.  Reuse
            # the already-presented candidate snapshot (no tool call) and apply M6.1 once
            # with the current effective M5 soft preferences.  Decline creates no new
            # preference, but existing preferences must remain effective.
            if self._soft_m5_rerank and base.recommendations:
                soft_ranking, _combined, _soft, _diag = self._m5_soft_rerank(
                    session_id,
                    base.recommendations,
                    existing=tuple(prefs),
                )
                if soft_ranking is not None:
                    ordered = soft_ranking.by_identity()
                    candidates = sorted(
                        base.recommendations,
                        key=lambda item: ordered[str(item.parent_asin)].reranked_rank,
                    )
                    candidates = [
                        item.model_copy(update={"rank": position})
                        for position, item in enumerate(candidates, start=1)
                    ]
                    base = base.model_copy(
                        update={
                            "recommendations": tuple(candidates),
                            "message": _consistent_message(
                                base.message,
                                candidates,
                                getattr(self._metadata, "records", {}) or {},
                            ),
                        }
                    )
            self._m8_finalize(
                session_id=session_id,
                candidates=[item.parent_asin for item in base.recommendations],
            )
            stopped = self._attach_intent_and_reasons(base, intent)
            return self._register_candidate_snapshot(
                stopped,
                user_key=user_key,
                task_id=session_id,
                turn_key=turn_key,
            )

        candidates = list(base.recommendations)
        # M7 experiment (independent switch, default off): ask the M6.1 public ranking
        # foundation to refresh the order of exactly the candidates the user was just
        # shown.  Any failure keeps the accepted ordering path below unchanged.
        refreshed = None
        if grounded_actions_enabled() and outcome is not None and prefs:
            try:
                from recommendation.state.state_projector import for_ranker

                base_projection = for_ranker(outcome.state)
                refreshed = refresh_candidate_order(
                    recommendations=base.recommendations,
                    records=getattr(self._metadata, "records", {}) or {},
                    base_projection=base_projection,
                    preferences=prefs,
                    user_key=user_key,
                    task_id=session_id,
                    turn_key=turn_key,
                )
            except Exception as exc:  # noqa: BLE001 - experiment diagnostics only
                refreshed = None
                self._m7_refresh[session_id] = {
                    "ran": False,
                    "error": type(exc).__name__,
                    "reason": "m6_1_refresh_exception",
                    "fallback": "accepted_ordering",
                }
            else:
                self._m7_refresh[session_id] = {
                    "ran": refreshed.ran,
                    "changed": refreshed.changed,
                    "error": refreshed.error,
                    "reason": (
                        None
                        if refreshed.ran
                        else "m6_1_refresh_failed"
                    ),
                    "fallback": None if refreshed.ran else "accepted_ordering",
                    "for_ranker_features": len(base_projection.features),
                }
        elif grounded_actions_enabled() and prefs:
            # No replayed ranker projection is available (M5 state integration disabled or
            # SHADOW absorbed a failure): the accepted ordering path remains authoritative
            # and the reason is explicit instead of silent.
            self._m7_refresh[session_id] = {
                "ran": False,
                "error": "ranker_projection_unavailable",
                "reason": "m5_state_projection_unavailable",
                "fallback": "accepted_ordering",
            }
        if refreshed is not None and refreshed.ran and refreshed.order:
            order_index = {
                identity: position for position, identity in enumerate(refreshed.order)
            }
            candidates.sort(
                key=lambda item: order_index.get(
                    str(item.parent_asin), len(order_index)
                )
            )
            candidates = [
                item.model_copy(update={"rank": position})
                for position, item in enumerate(candidates, start=1)
            ]
        else:
            # The trusted comparison tool compares at most five products, so the
            # preference question deliberately scopes its counterfactual to the prompt's
            # candidate window.  Reorder exactly that window and leave any remaining
            # requested-k products untouched; otherwise expected (5-item window) and
            # actual (full k list) would use different Top-K definitions.
            scope = tuple(str(identity) for identity in prompt.products)
            scope_set = set(scope)
            scoped = [item for item in candidates if item.parent_asin in scope_set]
            remainder = [item for item in candidates if item.parent_asin not in scope_set]
            ranking = rank_with_task_preferences(
                scoped,
                getattr(self._metadata, "records", {}) or {},
                prefs,
            )
            if ranking is not None:
                ordered = ranking.by_identity()
                scoped.sort(key=lambda item: ordered[item.parent_asin].reranked_rank)
            candidates = scoped + remainder
            candidates = [
                item.model_copy(update={"rank": position})
                for position, item in enumerate(candidates, start=1)
            ]
        self._m8_refresh(
            session_id=session_id,
            order_before=tuple(item.parent_asin for item in base.recommendations),
            order_after=tuple(item.parent_asin for item in candidates),
        )
        if (
            grounded_actions_enabled()
            and len(candidates) < max(2, int(payload.k))
            and not self._m7_retrieval_guard
            and not r1_rerank_only
        ):
            # Only re-retrieve when the refreshed candidate set is genuinely too small to
            # present; the accepted ranking path is never replaced for any other reason.
            self._m7_retrieval_guard = True
            try:
                retry_payload = request_payload.model_copy(
                    update={"preference_discovery": False}
                )
                fresh = self.run(
                    retry_payload,
                    task_preferences=tuple(prefs),
                )
                promoted = promote_explicit_type_matches(
                    request_payload.message,
                    fresh.recommendations,
                    getattr(self._metadata, "records", {}) or {},
                )
                if promoted != fresh.recommendations:
                    fresh = fresh.model_copy(update={"recommendations": promoted})
                fresh = _reconcile_final_identity(fresh, self._metadata)
                self._m8_finalize(
                    session_id=session_id,
                    candidates=[item.parent_asin for item in fresh.recommendations],
                )
                return self._register_candidate_snapshot(
                    fresh,
                    user_key=user_key,
                    task_id=session_id,
                    turn_key=turn_key,
                )
            finally:
                self._m7_retrieval_guard = False
        message = _consistent_message(
            base.message, candidates, getattr(self._metadata, "records", {})
        )
        response = base.model_copy(
            update={
                "recommendations": tuple(candidates),
                "message": message,
                "intent": _intent_view(intent),
            }
        )
        window = len(prompt.baseline_top_k) or len(scope)
        actual = tuple(item.parent_asin for item in candidates[:window])
        expected = tuple(
            prompt.counterfactual.get(option.option_id, ())
        )
        verified = bool(expected) and expected == actual
        refresh_ran = bool((self._m7_refresh.get(session_id) or {}).get("ran"))
        comparison_status = (
            "not_comparable"
            if refresh_ran
            else ("verified" if verified else "not_verified")
        )
        response = response.model_copy(
            update={
                "preference_outcome": AgentPreferenceOutcomeView(
                    selected_option_id=option.option_id,
                    attribute=prompt.attribute,
                    before_top_k=prompt.baseline_top_k,
                    expected_top_k=expected,
                    actual_top_k=actual,
                    changed=actual != prompt.baseline_top_k,
                    verified=verified,
                    comparison_status=comparison_status,
                )
            }
        )
        preference_response = self._maybe_attach_preference_question(
            response,
            intent,
            session_id,
            request_payload,
            terms=terms,
            user_key=user_key,
            turn_key=turn_key,
        )
        if preference_response is not None:
            # A follow-up preference question presents no list, so no trusted snapshot is
            # registered for it; the next answered turn records its own.
            return preference_response
        reranked = self._attach_intent_and_reasons(response, intent)
        self._m8_finalize(
            session_id=session_id,
            candidates=[item.parent_asin for item in reranked.recommendations],
        )
        return self._register_candidate_snapshot(
            reranked,
            user_key=user_key,
            task_id=session_id,
            turn_key=turn_key,
        )

    def _integrate_feedback_choice(
        self,
        *,
        payload: AgentRecommendRequest,
        option: PreferenceOption,
        session_id: str,
        user_key: str,
        turn_key: str,
        turn_identity_source: str,
    ) -> Any:
        """Record one trusted clarification answer as an M5.2 feedback fact.

        ``lower``/``higher`` become a ``FEATURE`` preference with a trusted
        ``minimize``/``maximize`` direction; a concrete ``value`` maps to the accepted
        kind for that attribute (colour/material/brand) or ``FEATURE`` otherwise.
        ``none``/``skip`` produce no preference: declining a preference is not a fact.
        Only the already-resolved option is mapped, never its label text.

        Returns the M5 ``StateTurnOutcome`` (or ``None`` when M5 is disabled / SHADOW
        absorbed a failure) so the M7 experiment can use the replayed ``for_ranker``
        projection without reading the event store a second time.

        A state failure follows the mode policy: SHADOW has already absorbed it inside
        the integration (the answer is untouched); APPLY_CONSTRAINTS fails closed with
        the existing ``state_integration_failed`` code.
        """
        integration = self._state_integration
        if integration is None or not bool(getattr(integration, "enabled", False)):
            return None
        from recommendation.state import shopping_adapter

        from .state_integration import StateIntegrationError

        preferences = self._clarification_feedback_preferences(option, shopping_adapter)
        snapshot = self._memory_service.get_active_preferences(user_key)
        try:
            outcome = integration.integrate_turn(
                user_key=user_key,
                session_id=session_id,
                task_id=session_id,
                turn_key=turn_key,
                message=str(payload.message),
                intent_label=_task_state(payload).intent.value,
                preferences=preferences,
                constraints=(),
                retractions=(),
                trusted_constraints=(),
                feedback_kind="clarification_option",
                turn_identity_source=turn_identity_source,
                legacy_preference_count=len(
                    getattr(snapshot, "active_entries", ()) or ()
                ),
            )
        except StateIntegrationError as exc:
            raise DemoAgentError(
                "the decision-state integration failed in experimental mode",
                code="state_integration_failed",
            ) from exc
        if outcome is not None and self._soft_m5_rerank:
            from recommendation.state.state_projector import for_ranker

            self._ranker_projections[session_id] = for_ranker(outcome.state)
            self._ranker_projections[str(user_key)] = self._ranker_projections[session_id]
        return outcome

    @staticmethod
    def _clarification_feedback_preferences(
        option: PreferenceOption, shopping_adapter: Any
    ) -> tuple[Any, ...]:
        """Map a resolved structured preference option to explicit adapter preferences.

        Returns an empty tuple when the option creates no fact (``none``/``skip``, an
        unknown action, or a value option without a value).  The attribute vocabulary of
        the preference prompt (``price``/``weight``/``color``/``material``/``brand``) is
        mapped onto the accepted preference ontology here, once, by a total mapping.
        """
        from recommendation.state.schemas import PreferenceDirection

        action = str(getattr(option, "action", "") or "").strip().lower()
        attribute = " ".join(str(getattr(option, "attribute", "") or "").split())
        label = str(getattr(option, "label", "") or "")
        if not attribute or action in {"none", "skip"}:
            return ()
        if action in {"lower", "higher"}:
            direction = (
                PreferenceDirection.MINIMIZE
                if action == "lower"
                else PreferenceDirection.MAXIMIZE
            )
            return (
                shopping_adapter.ExplicitPreference(
                    kind=PreferenceKind.FEATURE,
                    value=attribute,
                    polarity=PreferencePolarity.PREFER,
                    source_text=label,
                    direction=direction,
                ),
            )
        if action == "value":
            raw = getattr(option, "value", None)
            if raw is None or not str(raw).strip():
                return ()
            return (
                shopping_adapter.ExplicitPreference(
                    kind=_clarification_value_kind(attribute),
                    value=str(raw),
                    polarity=PreferencePolarity.PREFER,
                    source_text=label,
                ),
            )
        return ()

    def _maybe_attach_preference_question(
        self,
        response: AgentRecommendResponse,
        intent: Any,
        session_id: str,
        payload: AgentRecommendRequest,
        *,
        terms: Sequence[str] = (),
        user_key: str | None = None,
        turn_key: str = "",
    ) -> AgentRecommendResponse | None:
        """Ask at most two evidence-backed preference questions per shopping task.

        The feature is shopping-only and only precedes a normal completed recommendation.  The
        second question is allowed only after a first substantive answer and only when the
        counterfactual effect of another dimension is still material.  Every ask is validated
        against real candidate values and top-K consequences; a SKIP is a valid decision.
        """
        if not response.terminal.succeeded or response.terminal.status != "finished":
            return None
        if not payload.preference_discovery:
            return None
        rounds = int(self._preference_rounds.get(session_id, 0))
        grounded = grounded_actions_enabled()
        if not grounded:
            if rounds >= 2:
                return None
            if rounds >= 1 and not self._preference_second_enabled.get(session_id, False):
                return None
            if _task_state(payload).intent in (TaskIntent.COMPARE, TaskIntent.DETAILS):
                return None
        if response.research is not None and any(
            action.status == "failed" for action in response.research.actions
        ):
            return None
        ids = tuple(item.parent_asin for item in response.recommendations)
        if len(ids) < 2:
            return None
        current = self._task_preferences.get(session_id, ())
        blocked = blocked_attributes_from_intent(intent)
        revoked: set[str] = set()
        try:
            history = self._memory_service.get_memory_history(user_key or session_id)
            revoked |= blocked_attributes_from_history(history.entries)
            blocked |= revoked
        except Exception:  # noqa: BLE001 - absence of history must not fabricate a block
            pass
        state = PreferenceState(
            explicit=frozenset(blocked_attributes_from_intent(intent)),
            confirmed=frozenset(preference.attribute for preference in current),
            revoked=frozenset(revoked),
        )
        if grounded:
            budget = grounded_budget()
            if rounds >= budget:
                return None
            decision = choose_grounded_action(
                ids,
                getattr(self._metadata, "records", {}) or {},
                intent=intent,
                current_preferences=current,
                blocked_attributes=tuple(blocked),
                message=payload.message,
                top_k=payload.k,
                round_no=rounds + 1,
                state=state,
                budget_remaining=budget - rounds,
            )
            if decision.action is GroundedAction.FINALIZE or decision.prompt is None:
                return None
            return self._attach_preference_question(
                response,
                decision.prompt,
                intent,
                session_id,
                payload,
                round_no=rounds + 1,
                terms=terms,
                action=decision.action,
                comparison_rows=decision.comparison_rows,
                explanation=decision.explanation,
                user_requested=decision.user_requested,
                show_examples=decision.uncertain,
                user_key=user_key,
                turn_key=turn_key,
            )
        decision = choose_preference_action(
            ids,
            getattr(self._metadata, "records", {}) or {},
            intent=intent,
            current_preferences=current,
            blocked_attributes=tuple(blocked),
            message=payload.message,
            top_k=payload.k,
            round_no=rounds + 1,
            state=state,
        )
        if decision.action == "SKIP" or decision.dimension is None:
            self._preference_second_enabled[session_id] = False
            self._persist_task(session_id)
            return None
        return self._attach_preference_question(
            response,
            decision.dimension,
            intent,
            session_id,
            payload,
            round_no=rounds + 1,
            terms=terms,
            user_key=user_key,
            turn_key=turn_key,
        )

    def _record_grounded_interaction_events(
        self,
        *,
        action: GroundedAction,
        prompt: PreferencePrompt,
        session_id: str,
        user_key: str | None,
        turn_key: str,
        payload: AgentRecommendRequest,
        user_requested: bool = False,
        option_label: str | None = None,
    ) -> tuple[str, ...] | None:
        """Record the real M7 interaction events, never a fabricated user action."""
        integration = self._state_integration
        if integration is None or not bool(getattr(integration, "enabled", False)):
            return None
        recorder = getattr(integration, "record_grounded_interaction", None)
        if not callable(recorder):  # older/foreign integration seam
            return None
        from .state_integration import StateIntegrationError

        try:
            if action is GroundedAction.COMPARE and option_label is not None:
                return recorder(
                    user_key=str(user_key or session_id),
                    session_id=session_id,
                    task_id=session_id,
                    turn_key=turn_key,
                    compared_refs=tuple(str(item) for item in prompt.products),
                    comparison_feedback=str(option_label),
                    timestamp=time.time(),
                )
            if action is GroundedAction.EXPLAIN:
                ref = (
                    "m7exp_"
                    + hashlib.sha256(
                        f"{turn_key}:{prompt.attribute}".encode("utf-8")
                    ).hexdigest()[:24]
                )
                return recorder(
                    user_key=str(user_key or session_id),
                    session_id=session_id,
                    task_id=session_id,
                    turn_key=turn_key,
                    explanation_requested=(
                        prompt.attribute if user_requested else None
                    ),
                    question_text=(
                        str(payload.message) if user_requested else ""
                    ),
                    explanation_shown=prompt.attribute,
                    explanation_ref=ref,
                    timestamp=time.time(),
                )
        except StateIntegrationError as exc:
            raise DemoAgentError(
                "the grounded interaction event could not be recorded",
                code="state_integration_failed",
            ) from exc
        return None

    def _m8_feedback(
        self,
        *,
        session_id: str,
        prompt: PreferencePrompt,
        option: PreferenceOption,
        preferences: Sequence[TaskPreference],
    ) -> None:
        if not decision_trajectory_enabled():
            return
        with self._lock:
            recorder = self._m8_trajectories.get(str(session_id))
        if recorder is None:
            return
        try:
            recorder.record_feedback(
                option_id=option.option_id,
                label=option.label,
                action=option.action,
                attribute=prompt.attribute,
                value=option.value,
                accepted=option.action not in {"none", "skip"},
                preference_changes=tuple(
                    PreferenceChange(
                        attribute=preference.attribute,
                        action=preference.action,
                        value=preference.value,
                        label=preference.label,
                    )
                    for preference in preferences
                ),
            )
        except ValueError:
            pass

    def _m8_refresh(
        self,
        *,
        session_id: str,
        order_before: Sequence[str],
        order_after: Sequence[str],
    ) -> None:
        if not decision_trajectory_enabled():
            return
        with self._lock:
            recorder = self._m8_trajectories.get(str(session_id))
            diagnostic = dict(self._m7_refresh.get(str(session_id)) or {})
        if recorder is None or not diagnostic:
            return
        try:
            recorder.record_refresh(
                RankingRefreshRecord(
                    ran=bool(diagnostic.get("ran", False)),
                    changed=bool(diagnostic.get("changed", False)),
                    fallback=diagnostic.get("fallback"),
                    error=diagnostic.get("error"),
                    order_before=tuple(str(item) for item in order_before),
                    order_after=tuple(str(item) for item in order_after),
                )
            )
        except ValueError:
            pass

    def _m8_finalize(
        self,
        *,
        session_id: str,
        candidates: Sequence[str],
        terminal_status: str = "finished",
    ) -> None:
        if not decision_trajectory_enabled():
            return
        with self._lock:
            recorder = self._m8_trajectories.get(str(session_id))
        if recorder is None:
            return
        try:
            recorder.finalize(tuple(str(item) for item in candidates), terminal_status=terminal_status)
        except ValueError:
            pass

    def _attach_preference_question(        self,
        response: AgentRecommendResponse,
        prompt: PreferencePrompt,
        intent: Any,
        session_id: str,
        payload: AgentRecommendRequest,
        *,
        round_no: int,
        terms: Sequence[str] = (),
        action: GroundedAction | None = None,
        comparison_rows: Sequence[Mapping[str, Any]] = (),
        explanation: str | None = None,
        user_requested: bool = False,
        show_examples: bool = False,
        user_key: str | None = None,
        turn_key: str = "",
    ) -> AgentRecommendResponse:
        """Replace a completed list with one evidence-backed preference suspension."""
        from dataclasses import asdict, replace

        if action is GroundedAction.COMPARE:
            prompt = replace(prompt, interaction="COMPARE")
        if explanation:
            prompt = replace(prompt, explanation=explanation)
        clarification_id = uuid.uuid4().hex
        if action is GroundedAction.COMPARE:
            compared = "; ".join(
                f"{row.get('identity')}: {row.get('value')}"
                for row in comparison_rows
                if row.get("identity") is not None
            )
            question = (
                f"Compare {prompt.label}: {compared}. {prompt.reason} "
                "Which do you prefer?"
            )
        elif prompt.interaction == "ASK_DIRECT":
            question = (
                f"Which {prompt.label} do you prefer? {prompt.reason}"
            )
        elif prompt.interaction == "EXPLAIN" and prompt.explanation:
            question = (
                f"{prompt.explanation} Which {prompt.label} do you prefer? {prompt.reason}"
            )
        else:
            question = (
                f"These candidates differ on {prompt.label}. Which do you prefer? "
                f"{prompt.reason}"
            )
        if show_examples:
            examples = _catalogue_example_text(self._metadata, prompt.products)
            if examples:
                question = f"{question} Catalogue examples: {examples}"
        evidence = _preference_evidence_view(prompt)
        if action is not None:
            evidence = evidence.model_copy(update={"grounded_action": action.value})
        terminal = response.terminal.model_copy(
            update={
                "status": "waiting_for_user",
                "termination_reason": "awaiting_user",
                "termination_detail": "awaiting the user's bounded preference answer",
                "succeeded": False,
            }
        )
        clarification = AgentClarificationView(
            clarification_id=clarification_id,
            question=question,
            kind="preference",
            blocks="preference",
            options=tuple(option.label for option in prompt.options),
            reason=f"preference_{prompt.interaction.lower()}",
            products=prompt.products,
            preference=evidence,
        )
        with self._lock:
            self._clear_pending_for_task(session_id)
            self._preference_rounds[session_id] = round_no
            self._preference_second_enabled[session_id] = False
            self._save_pending(
                clarification_id,
                session_id,
                {
                    "session_id": session_id,
                    "kind": "preference",
                    "payload_version": PENDING_PAYLOAD_VERSION,
                    "created_at": time.time(),
                    "round": round_no,
                    "grounded_action": action.value if action is not None else None,
                    "preference_prompt": asdict(prompt),
                    "preference": evidence.model_dump(),
                    "base_response": _bounded_base_response(response),
                    "request": {
                        "message": payload.message,
                        "k": payload.k,
                        "sources": list(payload.sources),
                        "constraints": [item.model_dump() for item in payload.constraints],
                        "preference_discovery": bool(payload.preference_discovery),
                    },
                },
            )
            self._persist_task(session_id)
        recorder = self._m8_recorder(
            session_id,
            user_key or session_id,
            k=int(payload.k),
            budget=grounded_budget(),
        )
        if recorder is not None:
            attribute = prompt.attribute
            displayed = tuple(
                DecisionCandidate(
                    candidate_id=str(identity),
                    rank=position,
                    evidence=(
                        {attribute: prompt.values[identity]}
                        if identity in prompt.values
                        else {}
                    ),
                )
                for position, identity in enumerate(prompt.products, start=1)
            )
            try:
                recorder.begin_action(
                    action=action.value if action is not None else "ASK",
                    displayed=displayed,
                    evidence={
                        "attribute": attribute,
                        "values": dict(prompt.values),
                    },
                    budget_remaining=max(0, grounded_budget() - round_no),
                    explanation=prompt.explanation,
                )
            except ValueError:
                pass
        if action is GroundedAction.EXPLAIN:
            self._record_grounded_interaction_events(
                action=action,
                prompt=prompt,
                session_id=session_id,
                user_key=user_key,
                turn_key=turn_key,
                payload=payload,
                user_requested=user_requested,
            )
        return response.model_copy(
            update={
                "message": question,
                "recommendations": (),
                "terminal": terminal,
                "clarification": clarification,
                "preference": evidence,
                "intent": _intent_view(intent),
                "observation": None,
                "replan": None,
                "replan_execution": None,
            }
        )

    def _attach_clarification(
        self,
        response: AgentRecommendResponse,
        intent: Any,
        session_id: str,
        *,
        kind: str,
        constraints: Sequence[AgentConstraintItem] = (),
        clarification_count: int = 1,
    ) -> AgentRecommendResponse:
        """Record and expose one pending question for ``session_id``."""
        if response.terminal.termination_reason != "awaiting_user":
            return self._attach_intent_and_reasons(response, intent)
        clarification_id = uuid.uuid4().hex
        research_view = getattr(response, "research", None)
        if (
            research_view is not None
            and clarification_count > MAX_RESEARCH_CLARIFICATIONS
        ):
            raise DemoAgentError(
                "the research clarification limit was reached for this shopping task",
                code="clarification_limit_reached",
            )
        research_calls_used = (
            len(research_view.actions) if research_view is not None else 0
        )
        research_clarification = build_research_clarification(
            research_view,
            allow_compare=(
                research_calls_used < MAX_TASK_RESEARCH_CALLS
                and clarification_count < MAX_RESEARCH_CLARIFICATIONS
            ),
        )
        with self._lock:
            self._clear_pending_for_task(session_id)
            self._save_pending(
                clarification_id,
                session_id,
                {
                    "session_id": session_id,
                    "kind": kind,
                    "payload_version": PENDING_PAYLOAD_VERSION,
                    "created_at": time.time(),
                    # Phase 6C: preserve the real observation, not a policy claim, so the next
                    # turn can offer/execute only options the catalogue actually supports.
                    "research": research_view.model_dump() if research_view is not None else None,
                    "research_clarification": (
                        research_clarification.model_dump()
                        if research_clarification is not None
                        else None
                    ),
                    "clarification_count": int(clarification_count),
                    "research_calls_used": int(research_calls_used),
                    # A research clarification is a resumption of the same shopping task, so the
                    # hard constraints already in force must survive the answer.  Ordinary
                    # conflict/unverified questions do not carry them: their answer intentionally
                    # redefines the requirement.
                    "constraints": (
                        [item.model_dump() for item in constraints]
                        if research_view is not None
                        else []
                    ),
                },
            )
            self._clarified_sessions.add(session_id)
            self._persist_task(session_id)
        view = AgentClarificationView(
            clarification_id=clarification_id,
            question=response.message,
            kind=kind,
            options=(
                tuple(option.label for option in research_clarification.options)
                if research_clarification is not None
                else ()
            ),
            reason=(
                research_clarification.reason
                if research_clarification is not None
                else None
            ),
            missing_attributes=(
                research_clarification.missing_attributes
                if research_clarification is not None
                else ()
            ),
            products=(
                research_clarification.products
                if research_clarification is not None
                else ()
            ),
            known_facts=(
                research_clarification.known_facts
                if research_clarification is not None
                else {}
            ),
            research_options=(
                research_clarification.options
                if research_clarification is not None
                else ()
            ),
        )
        return response.model_copy(
            update={"clarification": view, "intent": _intent_view(intent)}
        )

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
        clarification_question: str | None = None,
        clarify_unresolved: bool = False,
        intent_terms: Sequence[str] = (),
        dynamic_research: bool = False,
        research_resume: ResearchResume | None = None,
        research_budget_remaining: int = MAX_TASK_RESEARCH_CALLS,
        clarification_budget_remaining: int = MAX_RESEARCH_CLARIFICATIONS,
        task_preferences: tuple[TaskPreference, ...] = (),
        planner_retrieval: bool = False,
        apply_user_state: bool = True,
        state_message: str | None = None,
        source_policy: str = "balanced",
    ) -> AgentRecommendResponse:
        """Run one plan and project it.  Shared by the request-declared and LLM-selected paths.

        ``planner_retrieval`` marks a Planner RETRIEVE query rather than a user utterance.
        With ``apply_user_state=True`` the first such retrieval may apply the current user
        turn's canonical state using ``state_message`` (the real user message) without
        extracting preferences from the rewritten query; later retrievals pass
        ``apply_user_state=False`` so repeated queries do not write M5/M9 facts again.

        ``preference_snapshot`` is honoured only when this deployment has memory-aware ranking
        enabled (:data:`ENV_MEMORY_RANKING`): the run's presented candidates are then reordered by
        the accepted M10A/M10B components, over trusted catalogue metadata, without touching their
        identity, score, score kind or provenance.
        """
        # A learned-ranker base ordering cannot enter the dynamic multi-source plane: the model was
        # fitted on the fixed three-source protocol and already returns a final ordering, so fusing
        # it again would break the protocol and misreport who ranked the result.  Refuse BEFORE
        # composing the controller or recalling anything (no model load, no predict, no recall).
        from recommendation.control.candidate_ledger import normalize_source_policy

        try:
            source_policy = normalize_source_policy(source_policy)
        except ValueError as exc:
            raise DemoAgentError(str(exc), code="unsupported_source_policy") from exc
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
            clarification_question=clarification_question,
            clarify_unresolved=clarify_unresolved,
            intent_terms=intent_terms,
            dynamic_research=dynamic_research,
            research_resume=research_resume,
            research_budget_remaining=research_budget_remaining,
            clarification_budget_remaining=clarification_budget_remaining,
            # First planner RETRIEVE: the graph may persist M9 from the real user text,
            # never from the rewritten query. Later retrieval steps persist nothing.
            memory_user_message=(
                state_message if planner_retrieval and apply_user_state else None
            ),
            persist_memory=not (planner_retrieval and not apply_user_state),
            source_policy=source_policy,
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
        # P1B-3 soft-on arm: M5 state features are consumed only through M6.1 over the
        # already-hard-filtered presentable set.  The candidate identity set is never
        # changed here; soft_off keeps the accepted path byte-for-byte.
        task_ranking = None
        soft_diagnostic: dict[str, Any] | None = None
        if self._soft_m5_rerank:
            (
                soft_ranking,
                combined_task_preferences,
                consumed_soft,
                soft_diagnostic,
            ) = self._m5_soft_rerank(
                str(payload.user_key or ""),
                presentable,
                existing=tuple(task_preferences),
            )
            if consumed_soft:
                task_ranking = soft_ranking
        if task_ranking is None and task_preferences:
            # Accepted M7-confirmed task preference path, used by the existing tests and
            # by soft_off.  It is the same M6.1 foundation.
            task_ranking = rank_with_task_preferences(
                presentable,
                getattr(self._metadata, "records", {}) or {},
                task_preferences,
            )
        memory_ranking = self._memory_ranking(
            result, ledger, payload, preference_snapshot
        )
        ranking = task_ranking or memory_ranking
        # Experimental base rerank: only when no user preference or task preference
        # ranking is active.  It reorders exactly the presentable candidates and never
        # adds/removes one; preference-aware ranking always takes precedence.
        itemcf_order = None
        if ranking is None and self._itemcf_signal is not None:
            itemcf_order = self._itemcf_signal.order(
                presentable, tuple(payload.history), ledger.fusion_scores()
            )
        # Route is derived from the run's own trusted artifacts.  A run that produced no
        # candidate set is a direct turn *or* a failed one, and an empty recommendation list is
        # the honest answer in both cases - so a failure is never dressed up as a successful
        # direct reply.
        produced_candidates = result.state.get("tool_result") is not None
        role_selection_out: list[Any] = []
        role_target = self._role_target_for(payload.user_key)
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
            itemcf_order=itemcf_order,
            role_target=role_target,
            role_records=getattr(self._metadata, "records", {}) or {},
            role_selection_out=role_selection_out,
        )
        if role_selection_out:
            selection = role_selection_out[-1]
            self._role_diagnostics[str(payload.user_key or "")] = {
                "target": {
                    "type": selection.target.type,
                    "role": selection.target.role.value,
                },
                "matched_count": selection.matched_count,
                "mismatch_count": selection.mismatch_count,
                "unknown_count": selection.unknown_count,
                "assessments": [item.as_dict() for item in selection.assessments],
                "rejection_reasons": list(selection.rejection_reasons),
            }
        grounded_view = build_grounded_view(result, ledger)
        eligibility_view = build_eligibility_view(result)
        terminal_view = build_terminal_view(result)
        # Minimal P3C-2 observability: retain the counts and the actual role/constraint
        # verdicts at the point where the final candidate list is assembled.  This is a
        # diagnostic only; it never adds, removes or reorders a candidate.
        if role_selection_out:
            selection = role_selection_out[-1]
            self._candidate_filter_diagnostics[str(payload.user_key or "")] = {
                "raw_recall_count": len(ledger.entries) if ledger is not None else len(presentable),
                "rrf_candidate_count": len(presentable),
                "grounded_candidate_count": len(presentable),
                "role_target": {
                    "type": selection.target.type,
                    "role": selection.target.role.value,
                },
                "role_match_count": selection.matched_count,
                "role_mismatch_count": selection.mismatch_count,
                "role_unknown_count": selection.unknown_count,
                "role_rejection_reasons": list(selection.rejection_reasons),
                "hard_constraint_evaluated": eligibility_view.evaluated,
                "hard_constraint_eligible_count": eligibility_view.verified_eligible_count,
                "hard_constraint_ineligible_count": eligibility_view.ineligible_count,
                "hard_constraint_unresolved_count": eligibility_view.unresolved_count,
                "active_constraints": [[str(item.kind), str(item.value)] for item in payload.constraints],
                "active_price_max": next(
                    (str(item.value) for item in payload.constraints if str(item.kind) == "price_max"),
                    None,
                ),
                "final_candidate_count": len(recommendations),
                "final_ids": [str(item.parent_asin) for item in recommendations],
                "stage_note": "raw_recall_count is the grounded ledger identity count; source-level raw recall is not persisted",
            }
        else:
            # A turn with no explicit role target still needs a filter trace: the absence
            # of role selection is itself the reason the role stage did not remove items.
            self._candidate_filter_diagnostics[str(payload.user_key or "")] = {
                "raw_recall_count": len(ledger.entries) if ledger is not None else len(presentable),
                "rrf_candidate_count": len(presentable),
                "grounded_candidate_count": len(presentable),
                "role_target": None,
                "role_match_count": 0,
                "role_mismatch_count": 0,
                "role_unknown_count": 0,
                "role_rejection_reasons": [],
                "hard_constraint_evaluated": eligibility_view.evaluated,
                "hard_constraint_eligible_count": eligibility_view.verified_eligible_count,
                "hard_constraint_ineligible_count": eligibility_view.ineligible_count,
                "hard_constraint_unresolved_count": eligibility_view.unresolved_count,
                "active_constraints": [[str(item.kind), str(item.value)] for item in payload.constraints],
                "active_price_max": next(
                    (str(item.value) for item in payload.constraints if str(item.kind) == "price_max"),
                    None,
                ),
                "final_candidate_count": len(recommendations),
                "final_ids": [str(item.parent_asin) for item in recommendations],
                "stage_note": "no role target on this turn; role stage made no removal",
            }
        trajectory_view = build_trajectory_summary(result)
        awaiting_user = terminal_view.termination_reason == "awaiting_user"
        # A suspended run must not present candidates: the answer that would decide their
        # eligibility/fact position is still outstanding.  Keep the question text intact as
        # well; reconciling it against a product list would replace the question with the list
        # the question is explicitly withholding.  A failed run likewise retains its verified
        # candidates for audit, but never presents them as a successful answer.
        if awaiting_user or not terminal_view.succeeded:
            recommendations = ()
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
        # A completed run's explanation must name exactly the answer's own products.  A waiting
        # run's message is its question and is never reconciled against a product list.
        message = result.final_response
        if not message and not recommendations:
            projection = result.state.get("constraint_projection")
            if projection is not None and projection.is_empty:
                from ..agent.rendering import build_constraint_empty_response

                message = build_constraint_empty_response(projection.view)
        if not awaiting_user:
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
            research=build_research_view(result),
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


def _mark_goal_correction(extraction: Any) -> Any:
    """Promote a corrected goal statement to the existing M5 REPLACE mode.

    The extractor's default is conservative (ADD); a free-text correction that clearly
    changes category/feature must supersede the conflicting task-scoped goal without
    inventing a new event type.  The exact source span and extractor provenance are kept.
    """
    preferences = tuple(
        (
            preference.model_copy(update={"mode": PreferenceMode.REPLACE})
            if _enum_text(getattr(preference, "kind", None))
            in {"category", "feature", "free_form_constraint"}
            and _enum_text(getattr(preference, "polarity", None)) == "prefer"
            else preference
        )
        for preference in (getattr(extraction, "preferences", ()) or ())
    )
    return extraction.model_copy(update={"preferences": preferences})


def _goal_changing_extraction(extraction: Any) -> bool:
    """True when the free-text turn explicitly replaces the shopping goal.

    A category/feature request or an ``instead``-style replacement is a goal change; a
    numeric preference (price/weight) is not.  The extractor already applied its own
    conservative patterns before this function sees the facts.
    """
    for preference in getattr(extraction, "preferences", ()) or ():
        if _enum_text(getattr(preference, "mode", None)) == "replace":
            return True
        kind = _enum_text(getattr(preference, "kind", None))
        if kind in {"category", "feature", "free_form_constraint"}:
            return True
        if _enum_text(getattr(preference, "polarity", None)) == "avoid":
            return True
    return bool(getattr(extraction, "removals", ()) or ())


def _task_preferences_after_correction(
    preferences: Sequence[TaskPreference], extraction: Any, *, clear_all: bool
) -> tuple[TaskPreference, ...]:
    """Remove stale task preferences for this task after an explicit correction.

    This is task-local task-choice state, not M5 long-term memory.  A goal-changing
    correction clears competing task choices; an explicit retraction removes only the
    named slot/value.
    """
    if clear_all:
        return ()
    removals = tuple(getattr(extraction, "removals", ()) or ())
    if not removals:
        return tuple(preferences)
    remaining: list[TaskPreference] = []
    for preference in preferences:
        removed = False
        for removal in removals:
            kind = _enum_text(getattr(removal, "kind", None))
            value = getattr(removal, "value", None)
            if kind is not None and str(preference.attribute).casefold() != str(kind).casefold():
                continue
            if value is not None and str(value).casefold() not in str(preference.value).casefold():
                continue
            removed = True
            break
        if not removed:
            remaining.append(preference)
    return tuple(remaining)


def _with_effective_target_type(
    terms: Sequence[str], target: Any
) -> tuple[str, ...]:
    """Keep the still-effective product type when a follow-up omits it.

    Only the current ``role_target.type`` is added.  Previous query terms/preferences
    are never reused, so withdrawn preferences and old budget wording cannot return.
    """
    target_type = getattr(target, "type", None)
    if not target_type:
        return tuple(terms)
    type_words = str(target_type).strip().replace("_", " ")
    current_query = " ".join(str(term) for term in terms).casefold()
    if type_words and type_words.casefold() not in current_query:
        return (type_words, *terms)
    return tuple(terms)


def _normal_turn_goal_correction(extraction: Any) -> bool:
    """A correction that must re-plan retrieval rather than reuse the old candidates."""
    return _goal_changing_extraction(extraction)


def _is_more_options_request(message: str) -> bool:
    """True for a continuation that carries no new shopping constraint."""
    folded = " ".join(str(message or "").casefold().split())
    return folded in {"show more", "show me more", "more options", "show more options", "more"}


def _dense_retrieval_enabled() -> bool:
    """Legacy opt-in switch; maps to hybrid when no explicit mode was supplied."""
    value = (os.environ.get(ENV_DENSE_RETRIEVAL) or "").strip().lower()
    return value in {"1", "true", "on", "yes"}


def _catalog_search_mode() -> str:
    """Explicit bm25/bge/hybrid mode; default is always the accepted BM25 path."""
    from recommendation.control.catalog_search import CATALOG_SEARCH_MODES

    explicit = (os.environ.get(ENV_CATALOG_SEARCH_MODE) or "").strip().casefold()
    if explicit:
        if explicit not in CATALOG_SEARCH_MODES:
            raise DemoAgentError(
                f"AGENTRECX_CATALOG_SEARCH_MODE must be one of {CATALOG_SEARCH_MODES}",
                code="unsupported_catalog_search_mode",
            )
        return explicit
    # Preserve the old opt-in switch, but make its effective mode explicit.
    if _dense_retrieval_enabled():
        return "hybrid"
    return "bm25"


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


def _preference_evidence_view(prompt: PreferencePrompt) -> AgentPreferenceEvidenceView:
    return AgentPreferenceEvidenceView(
        attribute=prompt.attribute,
        label=prompt.label,
        reason=prompt.reason,
        products=prompt.products,
        values=dict(prompt.values),
        comparison_kind=prompt.comparison_kind,
        options=tuple(
            AgentPreferenceOptionView(
                option_id=option.option_id,
                label=option.label,
                action=option.action,
                attribute=option.attribute,
                value=option.value,
            )
            for option in prompt.options
        ),
        interaction=prompt.interaction,
        status="unconfirmed",
        impact=prompt.impact,
        baseline_top_k=prompt.baseline_top_k,
        counterfactual=dict(prompt.counterfactual),
        explanation=prompt.explanation,
    )


def _preference_prompt_from_payload(raw: Any) -> PreferencePrompt:
    if not isinstance(raw, Mapping):
        raise DemoAgentError(
            "the preference question is unavailable; start a new shopping turn",
            code="invalid_preference_option",
        )
    options = tuple(
        PreferenceOption(
            option_id=str(item["option_id"]),
            label=str(item["label"]),
            action=str(item["action"]),
            attribute=str(item["attribute"]),
            value=item.get("value"),
        )
        for item in raw.get("options") or ()
    )
    return PreferencePrompt(
        attribute=str(raw.get("attribute") or ""),
        label=str(raw.get("label") or ""),
        reason=str(raw.get("reason") or ""),
        products=tuple(str(item) for item in raw.get("products") or ()),
        values=dict(raw.get("values") or {}),
        comparison_kind=str(raw.get("comparison_kind") or ""),
        options=options,
        interaction=str(raw.get("interaction") or "SHOW_CONTRAST"),
        status=str(raw.get("status") or "unconfirmed"),
        coverage=float(raw.get("coverage") or 0.0),
        difference=float(raw.get("difference") or 0.0),
        relevance=float(raw.get("relevance") or 0.0),
        impact=float(raw.get("impact") or 0.0),
        utility=float(raw.get("utility") or 0.0),
        baseline_top_k=tuple(str(item) for item in raw.get("baseline_top_k") or ()),
        counterfactual={
            str(key): tuple(str(item) for item in value)
            for key, value in (raw.get("counterfactual") or {}).items()
        },
        explanation=raw.get("explanation"),
    )


def _merge_previous_research(
    response: AgentRecommendResponse,
    previous: AgentResearchView,
    resume: ResearchResume | None,
) -> AgentRecommendResponse:
    """Expose prior research facts after a recovery turn without re-running a tool."""
    action = "show_known_facts" if resume is None else resume.action
    if action == "stop_research":
        note = "research stopped; previously verified facts are shown without a new tool call"
    elif action == "compare_attributes":
        note = "follow-up comparison after the selected option; previously verified facts retained"
    else:
        note = "previously verified facts shown; no new research tool was called"
    current = response.research
    if current is None:
        merged = previous.model_copy(update={"note": note})
    else:
        merged = AgentResearchView(
            executed=True,
            actions=previous.actions + current.actions,
            note=note,
        )
    return response.model_copy(update={"research": merged})


#: Wording that makes a named catalogue attribute an explicit information request rather than a
#: soft recommendation hint.  Only an explicit question/detail request makes research *required*;
#: research added to a recommendation remains optional and may degrade to its candidates.
_DETAIL_REQUEST = re.compile(
    r"\b(is|are|does|do|what|which|tell me|show me|details?|compare)\b",
    re.IGNORECASE,
)


def _enum_text(value: Any, default: Any = None) -> Any:
    """Return a str-enum's ``.value``, a plain string as-is, else ``default``.

    Used by the M5.1/M5.2 hook to read accepted enum fields (and to stay tolerant of the
    duck-typed extraction objects tests inject) without inventing a fallback *value*:
    an unrecognised kind reaches the state adapter unchanged and is refused there.
    """
    if value is None:
        return default
    return getattr(value, "value", value)


def _feedback_kind(extraction: Any) -> str:
    """The bounded M5.2 feedback label for one accepted extraction preview.

    ``retraction`` when the turn withdrew something, ``correction`` when the extractor
    recognised an explicit replacement, else ``explicit_statement``.  It is a single
    bounded label for diagnostics, not a second interpretation of the turn.
    """
    if getattr(extraction, "removals", ()) or ():
        return "retraction"
    modes = {
        _enum_text(getattr(item, "mode", None))
        for item in (getattr(extraction, "preferences", ()) or ())
    }
    return "correction" if "replace" in modes else "explicit_statement"


def _constraint_signature(kind: object, value: object) -> tuple[str, str]:
    """Comparison key of an intent constraint: lower-cased kind + case-folded value."""
    return (
        str(kind).strip().lower(),
        " ".join(str(value).split()).casefold(),
    )


def _effective_intent(
    intent: Any,
    merged: Sequence[tuple[str, str]],
) -> ShoppingIntent:
    """Rebuild the user-visible intent around the constraints actually enforced.

    Only hard constraints are merged.  M5 soft beliefs are deliberately not appended to
    this legacy intent view in P1B-0; they are exposed in the bounded state diagnostic
    and remain soft.  This keeps the accepted SHADOW intent view unchanged.
    """
    existing: dict[tuple[str, str], IntentEntry] = {}
    for entry in getattr(intent, "hard", ()) or ():
        existing.setdefault(
            _constraint_signature(entry.kind, entry.value), entry
        )
    hard: list[IntentEntry] = []
    for kind, value in merged:
        entry = existing.get(_constraint_signature(kind, value))
        if entry is not None:
            hard.append(entry)
            continue
        hard.append(
            IntentEntry(
                kind=str(kind),
                value=str(value),
                polarity="prefer",
                strength=Strength.HARD,
                source_turn_id="decision_state",
                source_text="authorized decision-state constraint",
                origin="decision_state",
            )
        )
    return ShoppingIntent(
        hard=tuple(hard),
        soft=tuple(getattr(intent, "soft", ()) or ()),
        conflicts=tuple(getattr(intent, "conflicts", ()) or ()),
        turn_ids=tuple(getattr(intent, "turn_ids", ()) or ()),
        active_preference_count=int(
            getattr(intent, "active_preference_count", 0) or 0
        ),
    )


def _clarification_value_kind(attribute: str) -> PreferenceKind:
    """Map a preference-prompt attribute onto the accepted preference kind.

    This is the one, total mapping the trusted option vocabulary needs: colour, material
    and brand name a specific kind; price, weight and any future/unknown attribute are a
    ``feature`` (the option carries the value, never an inferred category).
    """
    mapping = {
        "color": PreferenceKind.COLOR,
        "colour": PreferenceKind.COLOR,
        "material": PreferenceKind.MATERIAL,
        "brand": PreferenceKind.BRAND,
    }
    return mapping.get(str(attribute).strip().casefold(), PreferenceKind.FEATURE)


def _task_state(payload: AgentRecommendRequest) -> TaskState:
    """Build the run's task state from the request's stated constraints.

    Only ``EXPLICIT``, ``CURRENT_TURN`` constraints are constructed, because that is the
    accepted hardness rule: an inferred or softened signal must not be able to narrow the
    feasible set.  A request with no constraint yields an empty task state, so an
    unconstrained run behaves exactly as it did before constraints existed.
    """
    state = TaskState(request_text=payload.message)
    if wants_comparison(payload.message):
        state = state.model_copy(update={"intent": TaskIntent.COMPARE})
    elif requested_attributes(payload.message):
        state = state.model_copy(
            update={
                "intent": (
                    TaskIntent.DETAILS
                    if _DETAIL_REQUEST.search(payload.message)
                    else TaskIntent.RECOMMEND
                )
            }
        )
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


def _intent_view(intent: Any) -> AgentIntentView:
    """Project a shopping intent into the wire view (values only, no raw text).

    ``origin`` is carried through so a remembered or decision-state-authorized constraint
    is never presented as something the user stated in this turn; the accepted entry kind
    for entries that predate the field is ``user_stated`` (their own default).
    """

    def entries(values: Any) -> tuple[AgentIntentConstraintView, ...]:
        return tuple(
            AgentIntentConstraintView(
                kind=entry.kind,
                value=entry.value,
                polarity=entry.polarity,
                strength=entry.strength.value,
                source_turn_id=entry.source_turn_id,
                origin=getattr(entry, "origin", "user_stated"),
            )
            for entry in values
        )

    return AgentIntentView(
        hard=entries(intent.hard),
        soft=entries(intent.soft),
        conflicts=tuple(conflict.code for conflict in intent.conflicts),
        turn_ids=intent.turn_ids,
        active_preference_count=intent.active_preference_count,
    )


def _catalogue_example_text(metadata: Any, parent_asins: Sequence[str]) -> str:
    """Render bounded, trusted title/category examples for an exploratory prompt."""
    records = getattr(metadata, "records", None)
    if not isinstance(records, Mapping):
        return ""
    examples: list[str] = []
    for identity in tuple(parent_asins)[:3]:
        record = records.get(str(identity))
        if record is None:
            continue
        title = str(getattr(record, "title", "") or "").strip()
        categories = tuple(str(value) for value in getattr(record, "categories", ()) or ())
        leaf = categories[-1] if categories else ""
        if title and leaf:
            examples.append(f"{title} ({leaf})")
        elif title:
            examples.append(title)
        elif leaf:
            examples.append(leaf)
    return " | ".join(examples)


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

#: How long a pending shopping clarification may be answered.  Pending records are written to
#: the durable checkpoint when one is configured and swept on restore / each shopping turn;
#: with no checkpoint they are process-local.  An expired record is removed and reported,
#: never silently reused.
CLARIFICATION_TTL_SECONDS = 30 * 60

#: A pending preference question must not swallow an explicit withdrawal or a new goal.
#: These are user-level intents, not an answer to the question's option list.
_PENDING_WITHDRAW = re.compile(
    r"\b(forget|remove|drop|ignore|never\s+mind|cancel|clear|reset|start\s+over)\b",
    re.IGNORECASE,
)
#: A remembered catalogue phrase can contain an old budget clause ("water filter under
#: 30 dollars").  When a newer explicit price bound exists, R1 must not put the old
#: condition back into a fresh query.  This is a generic price-clause sanitizer, not a
#: task/ASIN rule.
_PRICE_CLAUSE = re.compile(
    r"\b(?:under|below|less\s+than|no\s+more\s+than|at\s+most|up\s+to|budget(?:\s+of)?|"
    r"over|above|more\s+than|at\s+least)\s*\$?\d+(?:\.\d+)?(?:\s*(?:dollars?|usd|bucks?))?",
    re.IGNORECASE,
)


def _sanitize_price_query_terms(terms: Sequence[str]) -> tuple[str, ...]:
    """Remove superseded price clauses from eligible catalogue query terms."""
    cleaned: list[str] = []
    for term in terms:
        text = " ".join(_PRICE_CLAUSE.sub(" ", str(term)).split())
        if text:
            cleaned.append(text)
    return tuple(cleaned)
_PENDING_NEW_GOAL = re.compile(
    r"(\b(?:actually|instead)\b.*\b(?:need|want|looking\s+for|show\s+me|find\s+me)\b)"
    r"|(\b(?:i\s+need|i\s+want|show\s+me|find\s+me|looking\s+for)\b)",
    re.IGNORECASE,
)


def _pending_message_starts_fresh_turn(message: str) -> bool:
    """True for an explicit withdrawal or a new shopping goal, not an option answer."""
    text = " ".join(str(message or "").split())
    if not text:
        return False
    if _PENDING_WITHDRAW.search(text):
        return True
    # A bare "I prefer lighter" is an answer, not a new goal.  The new-goal pattern
    # requires an explicit new search verb or an actually/instead correction.
    return bool(_PENDING_NEW_GOAL.search(text))


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


def _reconcile_final_identity(response: Any, metadata: Any) -> Any:
    """Renumber and re-render the final list after any filtering/reordering."""
    recommendations = tuple(
        item.model_copy(update={"rank": position})
        for position, item in enumerate(response.recommendations, start=1)
    )
    message = _consistent_message(
        response.message, recommendations, getattr(metadata, "records", {}) or {}
    )
    return response.model_copy(
        update={"recommendations": recommendations, "message": message}
    )


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
