"""P14-C0 Electronics planner executor: the one executor shared by Rule and Qwen.

It implements the existing ``PlannerExecutor`` protocol and the existing Planner actions.
The only difference between Rule and Qwen is the planner object passed to ``run_planner``;
all tools, Memory read, Scorer, CandidateLedger, Eligibility and Decision Gate are the
same objects and budgets.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from experiments.p14b1.diagnose import (
    canonical_prefs,
    candidate_rows,
    conflicts,
    hard_requirements,
    query_specificity,
)
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.constraint_eligibility import CandidateEligibilityEvaluator, ConstraintRequirement
from recommendation.control.grounded_reasoning import ConstraintKind, GroundedReasoner
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerContractError,
    PlannerObservation,
)
from recommendation.preference.product_role import assess_candidate_role, infer_product_target
from recommendation.preference.decision_gate import (
    ALLOW_FINALIZE,
    NEED_MORE_CANDIDATES,
    NEED_MORE_PRODUCT_EVIDENCE,
    NEED_USER_CLARIFICATION,
    DecisionGate,
    GateBudget,
    GateContext,
)
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import (
    PHASE_FINALIZE,
    PHASE_INSPECT,
    PHASE_PRODUCT_RETRIEVE,
)

TOPK = 20
INSPECT_TOP_K = 5
MAX_STEPS = 8
GATE_EVIDENCE_ROUNDS = 2
GATE_RETRIEVAL_ROUNDS = 1


class ElectronicsPlannerExecutor:
    """Real tool executor over the P13-R Electronics catalogue, shared by both planners."""

    def __init__(
        self,
        *,
        catalogue: ElectronicsCatalogue,
        current_query: str,
        preferences: list[dict[str, Any]],
        hard: list[dict[str, Any]],
        gate: DecisionGate | None = None,
        namespace: str = ELECTRONICS_NAMESPACE,
        explainer: Any | None = None,
        task_context: dict[str, Any] | None = None,
        memory_history: dict[str, Any] | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.current_query = current_query
        self.preferences = list(preferences)
        self.hard = list(hard)
        self.namespace = namespace
        self.gate = gate or DecisionGate()
        self.scorer = PreferenceAwareScorer()
        self.reasoner = GroundedReasoner(catalogue)
        self.eligibility = CandidateEligibilityEvaluator(self.reasoner)
        self.search_source = catalogue.search_source
        self.ledger = CandidateLedger()
        self.product_target = infer_product_target(current_query)
        self.tool_calls: list[dict[str, Any]] = []
        self.last_rows: list[dict[str, Any]] = []
        self.last_decision: dict[str, Any] | None = None
        self.last_search: dict[str, Any] | None = None
        self.last_inspection: list[dict[str, Any]] = []
        self.last_eligibility: dict[str, Any] | None = None
        self.gate_budget = GateBudget(
            max_retrievals=GATE_RETRIEVAL_ROUNDS,
            max_product_evidence_rounds=GATE_EVIDENCE_ROUNDS,
        )
        self.retrievals = 0
        self.evidence_rounds = 0
        self.selected_candidate_id: str | None = None
        self.terminal_status = "not_started"
        self.ask_question: str | None = None
        self._rows_by_rank: dict[str, dict[str, Any]] = {}
        # P15 optional RAG hook: called only after the existing Gate allows FINALIZE.
        # None preserves the pre-P15 behavior byte-for-byte.
        self.explainer = explainer
        self.task_context = task_context
        self.memory_history = memory_history
        self.last_explanation: dict[str, Any] | None = None
        self.explanation_error: str | None = None

    # -- helpers ------------------------------------------------------------ #
    def _log(self, tool: str, inputs: dict[str, Any], output: dict[str, Any]) -> None:
        self.tool_calls.append({"tool": tool, "inputs": inputs, "output": output})

    def _search(self, query: str) -> list[str]:
        if self.retrievals >= 1 + GATE_RETRIEVAL_ROUNDS:
            raise PlannerContractError("tool_budget_exhausted", "search budget exhausted")
        result = self.search_source.search(query, limit=TOPK)
        self.retrievals += 1
        ids = [hit.parent_asin.split(":", 1)[-1] for hit in result.hits]
        self.ledger.record_from_source(
            source=CandidateSource.CATALOG_SEARCH,
            candidates=[(hit.parent_asin, rank + 1, hit.score) for rank, hit in enumerate(result.hits)],
            score_kind=result.score_kind,
            source_query=query[:200],
            step_index=len(self.tool_calls) + 1,
            grounded=frozenset(hit.parent_asin for hit in result.hits),
        )
        self.last_search = {
            "query": query, "hit_ids": ids, "catalogue_size": result.catalogue_size,
            "score_kind": result.score_kind, "scores": {ids[i]: hit.score for i, hit in enumerate(result.hits)},
        }
        self._log("product_search", {"query": query, "limit": TOPK},
                  {"hit_ids": ids, "ledger_entries": len(self.ledger.ranked_parent_asins())})
        return ids

    def _score_and_inspect(self, ids: list[str]) -> list[dict[str, Any]]:
        scores = (self.last_search or {}).get("scores", {})
        pool = {"task_id": "p14c0", "current_query": self.current_query,
                "candidate_ids": ids, "search_scores": {item_id: scores.get(item_id) for item_id in ids}}
        rows = candidate_rows(pool, self.catalogue, self.scorer, self.preferences, self.hard)
        for row in rows:
            product = self.catalogue.get(row["product_ref"]["item_id"])
            if product is not None:
                assessment = assess_candidate_role(
                    product_type=product.product_type, title=product.title,
                    target=self.product_target,
                )
                row.update(assessment.as_dict())
            else:
                row.update({"role_status": "UNKNOWN", "role_evidence": ["product not in catalogue"],
                            "role_source": "catalog_meta:title", "product_type": None,
                            "candidate_role": "unknown"})
        compatible = [
            row for row in rows
            if row.get("role_status") in {"MATCH", "NOT_REQUIRED"}
            or not self.product_target.product_type
        ]
        self._role_compatible_rows = compatible
        top = [row["product_ref"]["item_id"] for row in rows[:INSPECT_TOP_K]]
        inspected = []
        for item_id in top:
            view = self.catalogue.inspect(item_id, requested=sorted({p["value"] for p in self.hard} | {"price", "brand"}))
            inspected.append({
                "item_id": view["product_id"], "status": view["status"],
                "unknown": view["unknown_fields"],
                "present": [name for name, fact in view["facts"].items() if fact.get("status") == "present"],
            })
        self.last_inspection = inspected
        # The planner/Finalize only ever sees candidates that pass the Product Type/Role
        # check.  Mismatched/unknown-role products remain in the ledger for audit but are
        # not presented as recommendations.
        self.last_rows = compatible
        self._rows_by_rank = {row["product_ref"]["item_id"]: row for row in rows}
        namespaced = [f"{self.namespace}:{item_id}" for item_id in top]
        requirements = [ConstraintRequirement(kind=ConstraintKind.CATEGORY, expected="Electronics",
                                              source_text="catalogue category")]
        feasibility = self.eligibility.assess(namespaced, requirements=requirements)
        self.last_eligibility = {
            "verified": feasibility.verified_eligible_count,
            "ineligible": feasibility.ineligible_count,
            "unresolved": feasibility.unresolved_count,
        }
        self._log("product_view", {"product_ids": top}, {"inspected": inspected})
        self._log("eligibility", {"candidates": len(namespaced)}, self.last_eligibility)
        return rows

    # -- PlannerExecutor protocol ------------------------------------------ #
    def invoke_skill(self, registry: Any, name: str, **arguments: Any) -> dict[str, Any]:
        """Invoke a registered P16 skill from this existing Agent executor.

        This is a thin delegation only: the skill still uses the same catalogue/tools/history
        and cannot bypass Eligibility or the Decision Gate.  Existing default execution is
        unchanged when no skill is invoked.
        """
        if registry is None:
            raise ValueError("a skill registry is required")
        if self.task_context is not None:
            arguments.setdefault("task", self.task_context)
        arguments.setdefault("catalogue", self.catalogue)
        if self.memory_history is not None:
            arguments.setdefault("history", self.memory_history)
        return dict(registry.invoke(name, **arguments))

    def execute(
        self,
        proposal: ActionProposal,
        observation: PlannerObservation,
    ) -> tuple[PlannerObservation, dict[str, Any], bool]:
        action = proposal.action
        if action is PlannerAction.ASK:
            question = str(proposal.arguments.get("question") or "").strip()
            if not question:
                raise PlannerContractError("ask_missing_question", "ASK requires a question")
            self.ask_question = question
            self.terminal_status = "awaiting_clarification"
            return (replace(observation, terminal=True, needs_user_input=True,
                            m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                      "ask_question": question, "status": self.terminal_status}),
                    {"status": self.terminal_status, "question": question}, True)

        if action is PlannerAction.RETRIEVE:
            query = str(proposal.arguments.get("query") or observation.query or self.current_query).strip()
            if self.retrievals >= 1 + GATE_RETRIEVAL_ROUNDS:
                self.terminal_status = "budget_exhausted"
                return (replace(observation, terminal=True, candidate_ids=(),
                                m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                          "status": self.terminal_status}),
                        {"status": self.terminal_status, "reason": "search budget exhausted"}, True)
            ids = self._search(query)
            return (replace(observation, query=query, candidate_ids=tuple(ids), reranked=False,
                            m5_state={**observation.m5_state, "phase": PHASE_INSPECT,
                                      "last_search_query": query}),
                    {"status": "retrieved", "candidate_count": len(ids), "query": query}, False)

        if action is PlannerAction.RERANK:
            ids = list(observation.candidate_ids)
            if not ids:
                if self.last_search:
                    ids = list(self.last_search["hit_ids"])
                else:
                    raise PlannerContractError("rerank_without_candidates", "RERANK requires candidates")
            rows = self._score_and_inspect(ids)
            compatible = list(self.last_rows)
            if self.product_target.product_type and not compatible:
                if self.retrievals < 1 + GATE_RETRIEVAL_ROUNDS:
                    return (replace(observation, candidate_ids=(), reranked=False,
                                    m5_state={**observation.m5_state, "phase": PHASE_PRODUCT_RETRIEVE,
                                              "role_filter": "retry_no_compatible_candidate"}),
                            {"status": "no_role_compatible_candidate", "role_filter": "retry",
                             "examined": len(rows)}, False)
                self.terminal_status = "no_verified_candidate_in_scope"
                return (replace(observation, terminal=True, candidate_ids=(),
                                m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                          "status": self.terminal_status,
                                          "role_filter": "bounded_no_compatible_candidate"}),
                        {"status": self.terminal_status, "bounded_scope": True,
                         "role_filter": "exhausted", "examined": len(rows)}, True)
            ranked_ids = [row["product_ref"]["item_id"] for row in compatible]
            return (replace(observation, candidate_ids=tuple(ranked_ids), reranked=True,
                            m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                      "eligibility": self.last_eligibility}),
                    {"status": "reranked", "candidate_count": len(ranked_ids),
                     "eligibility": self.last_eligibility}, False)

        if action is PlannerAction.FINALIZE:
            if not observation.candidate_ids:
                raise PlannerContractError("finalize_without_candidates", "cannot finalize an empty candidate set")
            ids = list(observation.candidate_ids)
            if not self.last_rows:
                self._score_and_inspect(ids)
                ids = [row["product_ref"]["item_id"] for row in self.last_rows]
            specific = query_specificity(self.current_query, self.hard, self.preferences)
            context = GateContext(
                query=self.current_query,
                candidates=self.last_rows,
                active_preferences=self.preferences,
                hard_requirements=self.hard,
                preference_conflicts=conflicts(self.preferences),
                namespace=self.namespace,
                query_specificity=specific,
                expected_product_type=self.product_target.product_type,
                expected_product_role=self.product_target.role.value,
                budget=self.gate_budget,
            )
            decision = self.gate.check(context)
            self.last_decision = decision.as_dict()
            self._log("decision_gate", {"candidate_count": len(self.last_rows),
                                        "budget": self.gate_budget.as_dict()}, self.last_decision)
            if decision.allow_finalize:
                selected = decision.selected_candidate_id
                row = self._rows_by_rank.get(selected or "", None)
                self.selected_candidate_id = selected
                self.terminal_status = "finalized"
                recommendation = {
                    "product_ref": {"namespace": self.namespace, "item_id": selected},
                    "rank": row["rank"] if row else None,
                    "evidence_refs": (row or {}).get("evidence_refs"),
                    "unknown_fields": list(decision.unknown_fields),
                }
                if self.explainer is not None and self.task_context is not None:
                    try:
                        self.last_explanation = self.explainer.explain(
                            task=self.task_context,
                            selected_item_id=selected,
                            catalogue=self.catalogue,
                            history=self.memory_history or {},
                            gate_decision=decision.as_dict(),
                            search_mode=self.search_source.mode,
                            ledger_source=f"CatalogSearchSource(mode={self.search_source.mode})",
                            role_status=(row or {}).get("role_status"),
                            product_type=(row or {}).get("product_type"),
                        )
                        recommendation["rag"] = {
                            "final_answer": self.last_explanation.get("final_answer"),
                            "evidence_bundle": self.last_explanation.get("bundle"),
                            "claim_validation": self.last_explanation.get("validation"),
                        }
                    except Exception as exc:  # optional hook must never bypass/block the Gate
                        self.explanation_error = f"{type(exc).__name__}: {exc}"
                        recommendation["rag_error"] = self.explanation_error
                return (replace(observation, terminal=True, candidate_ids=(selected,),
                                m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                          "gate_action": decision.action, "status": self.terminal_status}),
                        {"status": self.terminal_status, "recommendation": recommendation,
                         "gate": decision.as_dict()}, True)

            self.gate_budget = replace(
                self.gate_budget,
                product_evidence_rounds_used=self.gate_budget.product_evidence_rounds_used
                + (1 if decision.action == NEED_MORE_PRODUCT_EVIDENCE else 0),
                retrievals_used=self.gate_budget.retrievals_used
                + (1 if decision.action == NEED_MORE_CANDIDATES else 0),
            )
            if decision.action == NEED_USER_CLARIFICATION:
                self.ask_question = decision.ask_question
                self.terminal_status = "awaiting_clarification"
                return (replace(observation, needs_user_input=True,
                                m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                          "gate_action": decision.action, "ask_question": decision.ask_question,
                                          "status": self.terminal_status}),
                        {"status": self.terminal_status, "gate": decision.as_dict()}, False)
            retryable = decision.action in {NEED_MORE_PRODUCT_EVIDENCE, NEED_MORE_CANDIDATES} and (
                self.gate_budget.product_evidence_remaining > 0 or self.gate_budget.retrievals_remaining > 0
            )
            if retryable:
                next_phase = PHASE_INSPECT if decision.action == NEED_MORE_PRODUCT_EVIDENCE else PHASE_PRODUCT_RETRIEVE
                return (replace(observation, reranked=False, needs_user_input=False,
                                m5_state={**observation.m5_state, "phase": next_phase,
                                          "gate_action": decision.action, "status": "gate_more_evidence"}),
                        {"status": "gate_more_evidence", "gate": decision.as_dict()}, False)
            if decision.action in {NEED_MORE_PRODUCT_EVIDENCE, NEED_MORE_CANDIDATES}:
                self.terminal_status = (
                    "no_verified_candidate_in_scope"
                    if decision.action == NEED_MORE_CANDIDATES else "budget_exhausted"
                )
            else:
                self.terminal_status = decision.action.lower()
            return (replace(observation, terminal=True, candidate_ids=(),
                            m5_state={**observation.m5_state, "phase": PHASE_FINALIZE,
                                      "gate_action": decision.action, "status": self.terminal_status}),
                    {"status": self.terminal_status, "gate": decision.as_dict(),
                     "bounded_scope": decision.bounded_scope}, True)

        raise PlannerContractError("unsupported_action", f"unsupported action {action!r}")


def prepare_preferences(memory_read: dict[str, Any]) -> list[dict[str, Any]]:
    """Canonicalise a legal history-state read for the executor (no evaluator labels)."""
    return canonical_prefs(memory_read["active_preferences"])


def prepare_hard(query: str) -> list[dict[str, Any]]:
    return hard_requirements(query)
