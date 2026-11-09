"""Planner Action contract -> real Shopping Companion tool execution (P14-A).

This module is the P14-0 GAP-6 bridge, and it is deliberately thin:

* the Planner **contract is reused unchanged** --
  :class:`recommendation.demo.planner_contract.PlannerAction` /
  ``PlannerObservation`` / ``ActionProposal`` / ``validate_action`` /
  ``run_planner`` define the action semantics and the step loop;
* :class:`P14ARulePlanner` is a **deterministic Rule policy** (the brief explicitly
  asks to run the Rule/deterministic path first).  It is not Qwen and it is never
  reported as model reasoning;
* :class:`P14AToolExecutor` implements the existing ``PlannerExecutor`` protocol and
  really executes the Memory / Product Search / Product Info / CandidateLedger /
  GroundedReasoner+Eligibility tools.

No second Agent runtime and no second planner loop exist here: every step is
produced by ``run_planner``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence

from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.constraint_eligibility import (
    CandidateEligibilityEvaluator,
    ConstraintRequirement,
)
from recommendation.control.grounded_reasoning import GroundedReasoner
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerContractError,
    PlannerObservation,
)

from .catalogue import ShoppingCompanionCatalogue
from .namespace import SHOPPING_COMPANION_NAMESPACE, raw_product_id
from .product_info import ShoppingCompanionProductInfoTool
from .search import ShoppingCompanionSearchTool

__all__ = [
    "BridgeConfig",
    "MemorySearchProtocol",
    "P14ARulePlanner",
    "P14AToolExecutor",
    "PHASE_FINALIZE",
    "PHASE_INSPECT",
    "PHASE_MEMORY_RETRIEVE",
    "PHASE_PRODUCT_RETRIEVE",
    "PHASE_STAGE1_FINALIZE",
    "StageResult",
]

PHASE_MEMORY_RETRIEVE = "memory_retrieve"
PHASE_STAGE1_FINALIZE = "stage1_finalize"
PHASE_PRODUCT_RETRIEVE = "product_retrieve"
PHASE_INSPECT = "inspect"
PHASE_FINALIZE = "finalize"

#: Score kind the catalogue search actually reports; recorded verbatim in the ledger.
SEARCH_SCORE_KIND = "bm25"


class MemorySearchProtocol(Protocol):
    """The memory capability the bridge needs; honest ``available=False`` is valid."""

    def search(self, query: str, *, scope: str = "long_term", k: int = 10) -> dict[str, Any]: ...

    def history_status(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BridgeConfig:
    """Static bridge configuration (one instance per task run)."""

    search_limit: int = 10
    inspect_top_k: int = 5
    memory_k: int = 10
    step_index_base: int = 1
    #: How many products FINALIZE may present for one task.  Shopping Companion's
    #: single-product task is "find ONE product", so P14-A presents exactly one.
    final_count: int | None = 1


@dataclass
class StageResult:
    """What one stage (stage_1 / stage_2) actually established."""

    stage: str
    status: str = "not_run"
    preferences: tuple[dict[str, Any], ...] = ()
    evidence_ids: tuple[str, ...] = ()
    search: dict[str, Any] | None = None
    inspected: tuple[dict[str, Any], ...] = ()
    eligibility: dict[str, Any] | None = None
    final_ids: tuple[str, ...] = ()
    recommendations: tuple[dict[str, Any], ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "preferences": [dict(item) for item in self.preferences],
            "evidence_ids": list(self.evidence_ids),
            "search": self.search,
            "inspected": [dict(item) for item in self.inspected],
            "eligibility": self.eligibility,
            "final_ids": list(self.final_ids),
            "recommendations": [dict(item) for item in self.recommendations],
            "reason": self.reason,
        }


class P14ARulePlanner:
    """Deterministic Rule planner over the unchanged Action contract.

    It is **not** Qwen and it is not trained.  It reads the phase the executor
    recorded on ``observation.m5_state`` and proposes the next legal action.  This
    keeps the Action contract (ASK / RETRIEVE / RERANK / FINALIZE) exactly as the
    accepted P3D contract defines it.
    """

    name = "p14a_rule_planner"

    def propose(self, observation: PlannerObservation) -> ActionProposal:
        if observation.terminal:
            raise PlannerContractError("rule_terminal", "rule planner received a terminal observation")
        state = dict(observation.m5_state or {})
        phase = str(state.get("phase") or "")
        if observation.needs_user_input:
            question = str(state.get("ask_question") or "Please clarify the request.")
            return ActionProposal(PlannerAction.ASK, {"question": question}, "clarification required")
        if phase in (PHASE_MEMORY_RETRIEVE, PHASE_PRODUCT_RETRIEVE):
            query = str(state.get("next_query") or observation.query or "").strip()
            return ActionProposal(PlannerAction.RETRIEVE, {"query": query}, f"{phase}: real tool retrieval")
        if phase == PHASE_INSPECT:
            return ActionProposal(
                PlannerAction.RERANK,
                {"candidates": list(observation.candidate_ids)},
                "inspect candidates and verify hard constraints",
            )
        if phase in (PHASE_STAGE1_FINALIZE, PHASE_FINALIZE):
            return ActionProposal(
                PlannerAction.FINALIZE,
                {"candidates": list(observation.candidate_ids)},
                "finalize the established result",
            )
        # Defensive: an unknown phase is treated as a retry with the current query.
        return ActionProposal(PlannerAction.RETRIEVE, {"query": observation.query or "products"}, "phase fallback")


class P14AToolExecutor:
    """The real executor behind the P14-A Rule planner.

    One instance is reused across the stage_1 and stage_2 planner runs of a single
    task (the ledger belongs to the task run).  ``reset_stage`` selects which stage
    the next actions belong to.
    """

    def __init__(
        self,
        *,
        catalogue: ShoppingCompanionCatalogue,
        search_tool: ShoppingCompanionSearchTool,
        info_tool: ShoppingCompanionProductInfoTool,
        memory: MemorySearchProtocol | None,
        config: BridgeConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
        finalize_gate: Callable[[dict[str, Any]], Any] | None = None,
        max_gate_retries: int = 1,
    ) -> None:
        self._catalogue = catalogue
        self._search_tool = search_tool
        self._info_tool = info_tool
        self._memory = memory
        self._config = config or BridgeConfig()
        self._clock = clock
        self._ledger = CandidateLedger()
        self._reasoner = GroundedReasoner(catalogue)
        self._eligibility = CandidateEligibilityEvaluator(self._reasoner)
        self.tool_calls: list[dict[str, Any]] = []
        self.stage_results: dict[str, StageResult] = {}
        self._stage: str = "stage_1"
        self._requirements: tuple[ConstraintRequirement, ...] = ()
        #: Optional P14-B1 decision-gate hook.  ``None`` preserves the exact legacy
        #: FINALIZE behaviour; when installed, a gate decision that does not allow
        #: FINALIZE turns the action into ASK / another RETRIEVE/RERANK round or a
        #: bounded terminal refusal instead of presenting a product.
        self._finalize_gate = finalize_gate
        self._max_gate_retries = max(0, int(max_gate_retries))
        self._gate_finalize_attempts = 0

    # -- run-scoped state -------------------------------------------------- #

    @property
    def ledger(self) -> CandidateLedger:
        return self._ledger

    @property
    def eligibility_evaluator(self) -> CandidateEligibilityEvaluator:
        return self._eligibility

    def reset_stage(
        self,
        stage: str,
        *,
        requirements: Sequence[ConstraintRequirement] = (),
    ) -> None:
        if stage not in {"stage_1", "stage_2"}:
            raise ValueError(f"unknown stage {stage!r}")
        self._stage = stage
        self._requirements = tuple(requirements)
        self.stage_results[stage] = StageResult(stage=stage)

    def constraint_labels(self) -> list[dict[str, Any]]:
        return [requirement.as_dict() for requirement in self._requirements]

    def set_finalize_gate(self, gate: Callable[[dict[str, Any]], Any] | None) -> None:
        """Install/remove the optional P14-B1 FINALIZE gate hook."""
        self._finalize_gate = gate
        self._gate_finalize_attempts = 0

    def _gate_blocked(
        self,
        observation: Any,
        result: StageResult,
        data: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], bool]:
        """Turn a non-allow gate decision into ASK, another evidence round, or refusal."""
        action = str(data.get("action") or "BUDGET_EXHAUSTED")
        question = str(data.get("ask_question") or "I need more evidence before finalizing.")
        self._gate_finalize_attempts += 1
        payload = {"status": f"gate_{action.lower()}", "gate": data}
        result.status = f"gate_{action.lower()}"
        result.reason = str(data.get("reason") or "decision gate blocked FINALIZE")
        self.stage_results[self._stage] = result
        state = dict(observation.m5_state or {})
        state.update({"phase": PHASE_FINALIZE, "gate_action": action, "gate": data})
        retryable = (
            action in {"NEED_MORE_PRODUCT_EVIDENCE", "NEED_MORE_CANDIDATES"}
            and self._gate_finalize_attempts <= self._max_gate_retries
        )
        if retryable:
            next_phase = PHASE_INSPECT if action == "NEED_MORE_PRODUCT_EVIDENCE" else PHASE_PRODUCT_RETRIEVE
            state["phase"] = next_phase
            state.pop("ask_question", None)
            return replace(observation, m5_state=state), payload, False
        if action == "NEED_USER_CLARIFICATION":
            state["ask_question"] = question
            return replace(observation, needs_user_input=True, m5_state=state), payload, False
        return replace(observation, terminal=True, m5_state=state), payload, True

    # -- logging ----------------------------------------------------------- #

    def _log(
        self,
        tool: str,
        inputs: dict[str, Any],
        output: dict[str, Any] | None = None,
        *,
        error: dict[str, Any] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        self.tool_calls.append(
            {
                "stage": self._stage,
                "tool": tool,
                "inputs": dict(inputs),
                "output": dict(output) if output is not None else None,
                "error": dict(error) if error is not None else None,
                "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
            }
        )

    def _timed(self, started: float) -> float:
        return (self._clock() - started) * 1000.0

    # -- executor protocol ------------------------------------------------- #

    def execute(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        try:
            if proposal.action is PlannerAction.ASK:
                return self._handle_ask(proposal, observation)
            if proposal.action is PlannerAction.RETRIEVE:
                phase = str((observation.m5_state or {}).get("phase") or "")
                if self._stage == "stage_1" and phase == PHASE_MEMORY_RETRIEVE:
                    return self._memory_retrieve(proposal, observation)
                return self._product_retrieve(proposal, observation)
            if proposal.action is PlannerAction.RERANK:
                return self._inspect(proposal, observation)
            if proposal.action is PlannerAction.FINALIZE:
                return self._finalize(proposal, observation)
        except Exception as exc:  # tool error is recorded explicitly, never swallowed
            error = {"type": type(exc).__name__, "message": str(exc), "stage": self._stage}
            self._log(
                f"error:{proposal.action.value}",
                {"arguments": dict(proposal.arguments)},
                error=error,
            )
            result = self.stage_results.get(self._stage) or StageResult(stage=self._stage)
            result.status = "tool_error"
            result.reason = f"{type(exc).__name__}: {exc}"
            self.stage_results[self._stage] = result
            return (
                replace(
                    observation,
                    terminal=True,
                    note=f"tool_error:{type(exc).__name__}: {exc}",
                    m5_state={**(observation.m5_state or {}), "phase": PHASE_FINALIZE},
                ),
                {"status": "tool_error", "error": error},
                True,
            )
        raise PlannerContractError("unsupported_action", f"unsupported action {proposal.action!r}")

    # -- stage 1: real (or honestly absent) memory ------------------------- #

    def _memory_retrieve(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        query = str(proposal.arguments.get("query") or observation.query or "").strip()
        started = self._clock()
        result = self.stage_results[self._stage]
        if self._memory is None:
            payload: dict[str, Any] = {
                "available": False,
                "status": "history_unavailable",
                "preferences": [],
                "evidence": [],
                "conflicts": [],
                "reason": (
                    "the released Shopping Companion benchmark does not include the conversation "
                    "history corpus required by the official memory service; no history is fabricated"
                ),
            }
            self._log("memory.search", {"query": query, "scope": "long_term"}, payload, latency_ms=self._timed(started))
        else:
            search_kwargs = {"scope": "long_term", "k": self._config.memory_k}
            conversation_id = (observation.m5_state or {}).get("conversation_id")
            if conversation_id:
                try:
                    payload = self._memory.search(query, conversation_id=str(conversation_id), **search_kwargs)
                except TypeError:  # memory capability without conversation scoping
                    payload = self._memory.search(query, **search_kwargs)
            else:
                payload = self._memory.search(query, **search_kwargs)
            self._log("memory.search", {"query": query, "scope": "long_term"}, payload, latency_ms=self._timed(started))

        preferences = tuple(dict(item) for item in (payload.get("preferences") or ()))
        evidence = tuple(dict(item) for item in (payload.get("evidence") or ()))
        self._validate_memory_evidence(evidence)
        result.status = "ok" if payload.get("available") and preferences else str(payload.get("status") or "history_unavailable")
        result.preferences = preferences
        result.evidence_ids = tuple(f"memory:{index}" for index in range(len(evidence)))
        result.reason = str(payload.get("reason") or "")
        if not payload.get("available") or not preferences:
            question = (
                "The conversation history for this task is unavailable in the released benchmark; "
                "I cannot recover the user's long-term preferences without fabricating them."
            )
            self.stage_results[self._stage] = result
            return (
                replace(
                    observation,
                    needs_user_input=True,
                    m5_state={
                        **(observation.m5_state or {}),
                        "phase": PHASE_STAGE1_FINALIZE,
                        "ask_question": question,
                        "memory_status": str(payload.get("status") or "history_unavailable"),
                    },
                ),
                {"status": result.status, "available": False, "reason": result.reason,
                 "preferences": [], "evidence": []},
                False,
            )
        self.stage_results[self._stage] = result
        return (
            replace(
                observation,
                candidate_ids=result.evidence_ids,
                reranked=True,
                m5_state={
                    **(observation.m5_state or {}),
                    "phase": PHASE_STAGE1_FINALIZE,
                    "memory_status": str(payload.get("status") or "available"),
                },
            ),
            {"status": "ok", "available": True, "preferences": [dict(p) for p in preferences],
             "evidence": [dict(e) for e in evidence], "reason": result.reason},
            False,
        )

    def _validate_memory_evidence(self, evidence: Sequence[dict[str, Any]]) -> None:
        """Refuse oracle evidence unless the run is explicitly an oracle diagnostic.

        The oracle flag is installed on the executor by the runtime via
        ``allow_oracle_diagnostic``; the default is strict.
        """
        if getattr(self, "_allow_oracle", False):
            return
        for item in evidence:
            if str(item.get("source_kind") or "") == "released_oracle_diagnostic":
                raise RuntimeError(
                    "oracle-diagnostic memory evidence leaked into a non-diagnostic run"
                )

    def allow_oracle_diagnostic(self, allowed: bool) -> None:
        self._allow_oracle = bool(allowed)

    # -- stage 2: real search -> real inspection -> real finalize ---------- #

    def _product_retrieve(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        query = str(proposal.arguments.get("query") or observation.query or "").strip()
        started = self._clock()
        result = self.stage_results[self._stage]
        search = self._search_tool.search(query, limit=self._config.search_limit)
        self._log(
            "product_search",
            {"query": query, "limit": self._config.search_limit},
            search.as_dict(),
            latency_ms=self._timed(started),
        )
        result.search = search.as_dict()
        if not search.hits:
            question = (
                f"No product in the loaded limited catalogue matched {query!r}; the catalogue is a "
                "limited real sample, so absence is not evidence that no product exists."
            )
            result.status = "no_candidates"
            result.reason = question
            self.stage_results[self._stage] = result
            return (
                replace(
                    observation,
                    needs_user_input=True,
                    m5_state={
                        **(observation.m5_state or {}),
                        "phase": PHASE_FINALIZE,
                        "ask_question": question,
                    },
                ),
                {"status": "no_candidates", "search": search.as_dict()},
                False,
            )
        candidates = [(hit.namespaced_id, hit.rank, hit.score) for hit in search.hits]
        grounded = frozenset(hit.namespaced_id for hit in search.hits)
        self._ledger.record_from_source(
            source=CandidateSource.CATALOG_SEARCH,
            candidates=candidates,
            score_kind=SEARCH_SCORE_KIND,
            source_query=query,
            step_index=observation.step + self._config.step_index_base,
            grounded=grounded,
        )
        ids = self._ledger.ranked_parent_asins()
        result.status = "candidates"
        result.reason = "real BM25 catalogue search recorded in CandidateLedger"
        self.stage_results[self._stage] = result
        return (
            replace(
                observation,
                candidate_ids=ids,
                m5_state={
                    **(observation.m5_state or {}),
                    "phase": PHASE_INSPECT,
                    "search": search.as_dict(),
                    "ledger_version": self._ledger.version,
                },
            ),
            {
                "status": "ok",
                "search": search.as_dict(),
                "candidate_ids": list(ids),
                "ledger_version": self._ledger.version,
            },
            False,
        )

    def _inspect(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        candidates = list(observation.candidate_ids)
        if not candidates:
            raise PlannerContractError("inspect_without_candidates", "RERANK requires candidates")
        top = candidates[: self._config.inspect_top_k]
        started = self._clock()
        inspected: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for identity in top:
            try:
                view = self._info_tool.inspect(raw_product_id(identity))
                inspected.append(view.as_dict())
            except Exception as exc:  # record per-product tool error, keep inspecting
                errors.append({"identity": identity, "error": f"{type(exc).__name__}: {exc}"})
        self._log(
            "product_view",
            {"product_ids": list(top)},
            {"inspected": inspected, "errors": errors},
            latency_ms=self._timed(started),
        )
        result = self.stage_results[self._stage]
        result.inspected = tuple(inspected)
        feasibility = self._eligibility.assess(candidates, requirements=self._requirements)
        eligibility_dict = {
            "requirements": self.constraint_labels(),
            "projection": feasibility.as_dict(),
            "verified_eligible_count": feasibility.verified_eligible_count,
            "ineligible_count": feasibility.ineligible_count,
            "unresolved_count": feasibility.unresolved_count,
            "excluded": [{"identity": identity, "reason": reason} for identity, reason in feasibility.exclusions()],
        }
        feasible = feasibility.narrow(candidates)
        result.eligibility = eligibility_dict
        if not feasible:
            question = (
                "Every retrieved candidate either violates a stated hard constraint or cannot be "
                "verified against it; refusing to present an unverified candidate."
            )
            result.status = "constraint_unresolved"
            result.reason = question
            self.stage_results[self._stage] = result
            return (
                replace(
                    observation,
                    candidate_ids=(),
                    reranked=True,
                    needs_user_input=True,
                    m5_state={
                        **(observation.m5_state or {}),
                        "phase": PHASE_FINALIZE,
                        "ask_question": question,
                        "eligibility": eligibility_dict,
                    },
                ),
                {"status": "constraint_unresolved", "eligibility": eligibility_dict},
                False,
            )
        result.status = "inspected"
        result.reason = "real product views + eligibility on the stated hard constraints"
        self.stage_results[self._stage] = result
        return (
            replace(
                observation,
                candidate_ids=feasible,
                reranked=True,
                m5_state={
                    **(observation.m5_state or {}),
                    "phase": PHASE_FINALIZE,
                    "eligibility": eligibility_dict,
                    "inspected_count": len(inspected),
                },
            ),
            {
                "status": "ok",
                "inspected_count": len(inspected),
                "inspected": inspected,
                "errors": errors,
                "eligibility": eligibility_dict,
                "feasible_ids": list(feasible),
            },
            False,
        )

    def _finalize(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        ids = tuple(observation.candidate_ids)
        if not ids:
            raise PlannerContractError("finalize_without_candidates", "cannot finalize an empty candidate set")
        result = self.stage_results[self._stage]
        if self._stage == "stage_1":
            result.status = "preferences_recovered"
            result.final_ids = ids
            result.reason = "stage_1 preferences recovered from real memory evidence"
            self.stage_results[self._stage] = result
            return (
                replace(observation, terminal=True, note="stage_1 finalized"),
                {"status": "stage_1_finalized", "evidence_ids": list(ids),
                 "preferences": [dict(p) for p in result.preferences]},
                True,
            )

        recommendations: list[dict[str, Any]] = []
        if self._finalize_gate is not None:
            gate_data = {
                "stage": self._stage,
                "query": observation.query,
                "candidate_ids": list(ids),
                "preferences": [dict(p) for p in result.preferences],
                "constraints": self.constraint_labels(),
                "eligibility": result.eligibility,
                "inspected": [dict(item) for item in result.inspected],
                "tool_calls": len(self.tool_calls),
                "gate_attempts": self._gate_finalize_attempts,
            }
            decision = self._finalize_gate(gate_data)
            decision_dict = decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)
            if not bool(decision_dict.get("allow_finalize")):
                return self._gate_blocked(observation, result, decision_dict)
        # The official single-product task is "recommend ONE product"; presenting the
        # whole feasible head would be a task-semantics error.  The ledger keeps every
        # candidate for audit, but FINALIZE presents at most ``final_count``.
        limit = self._config.final_count
        presented = list(ids[:limit] if limit is not None else ids)
        for identity in presented:
            entry = self._ledger.get(identity)
            product = self._catalogue.get(identity)
            if entry is None or product is None:
                continue
            recommendations.append(
                {
                    "namespaced_id": identity,
                    "product_id": product.product_id,
                    "title": product.title,
                    "price": product.price,
                    "seller_id": product.seller_id,
                    "seller_name": product.seller_name,
                    "ledger_sources": [source.value for source in entry.sources],
                    "ledger_ranks": [records.source_rank for records in entry.provenance],
                    "ledger_score_kinds": sorted({records.score_kind for records in entry.provenance}),
                    "namespace": SHOPPING_COMPANION_NAMESPACE,
                }
            )
        result.status = "finalized"
        result.final_ids = tuple(presented)
        result.recommendations = tuple(recommendations)
        result.reason = "finalized from the CandidateLedger after eligibility verification"
        self.stage_results[self._stage] = result
        return (
            replace(observation, candidate_ids=tuple(presented), terminal=True, note="stage_2 finalized"),
            {
                "status": "finalized",
                "recommendations": recommendations,
                "presented_count": len(presented),
                "candidate_count": len(ids),
                "ledger_version": self._ledger.version,
            },
            True,
        )

    # -- ASK --------------------------------------------------------------- #

    def _handle_ask(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        question = str(proposal.arguments.get("question") or "").strip()
        self._log("ask", {"question": question}, {"status": "clarification_pending"})
        result = self.stage_results.get(self._stage) or StageResult(stage=self._stage)
        if result.status in {"not_run", "candidates", "inspected"}:
            result.status = "clarification_pending"
        result.reason = question
        self.stage_results[self._stage] = result
        return (
            replace(observation, terminal=True, needs_user_input=True, note=question),
            {"status": "clarification_pending", "question": question},
            True,
        )
