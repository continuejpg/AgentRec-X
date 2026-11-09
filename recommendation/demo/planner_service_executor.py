"""CPU executor for the P3D planner contract.

The executor reuses DemoAgentService.run_shopping_turn for RETRIEVE, the service's
existing M5/M6.1 private path for RERANK, and returns the service's already validated
recommendations for FINALIZE.  It never creates candidates, bypasses role checks or
bypasses price constraints.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from recommendation.control.candidate_ledger import (
    normalize_source_policy,
    source_policy_weights,
)
from recommendation.demo.agent_schemas import AgentRecommendRequest
from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerObservation,
)


@dataclass
class ServicePlannerExecutor:
    service: Any
    payload: AgentRecommendRequest
    session_id: str
    task_id: str
    turn_prefix: str = "p3d"
    _candidates: tuple[Any, ...] = ()
    _turn: int = 0
    _last_query: str = ""
    _presented_count: int = 0
    #: Read-only wiring fields for the P7-A controlled runner.  They are populated from
    #: the canonical per-user-turn effective state so multi-step RETRIEVE/RERANK reads
    #: the same state, and from the real Agent response for actual tool-call accounting.
    effective_role_target: dict[str, Any] | None = None
    effective_state: dict[str, Any] = field(default_factory=dict)
    last_response: Any = field(default=None, repr=False)
    tool_calls_used: int = 0
    #: Real user-turn text whose canonical state is applied on the first RETRIEVE.  Planner
    #: queries only rewrite retrieval; they must not create/retract user preferences.
    state_message: str = ""
    _state_applied: bool = field(default=False, repr=False)
    source_policy: str = "balanced"
    _source_feedback: dict[str, Any] = field(default_factory=dict, repr=False)

    def _candidate_evidence(self, candidate_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        metadata = getattr(self.service, "_metadata", None)
        records = getattr(metadata, "records", {}) if metadata is not None else {}
        evidence = []
        for parent_asin in candidate_ids:
            record = records.get(parent_asin)
            if record is None:
                evidence.append({"parent_asin": parent_asin})
                continue
            details = record.details_dict() if hasattr(record, "details_dict") else {}
            weight = next((str(value) for key, value in details.items() if "weight" in key.casefold()), "")
            evidence.append({
                "parent_asin": parent_asin,
                "title": str(getattr(record, "title", "") or ""),
                "price": str(getattr(record, "price_text", "") or ""),
                "weight": weight,
            })
        return tuple(evidence)

    def _build_source_feedback(self, response: Any) -> dict[str, Any]:
        """Small, factual source feedback from the actual response provenance."""
        grounded = getattr(response, "grounded", None)
        candidates = list(getattr(grounded, "candidates", ()) or ())
        by_source: Counter[str] = Counter()
        overlap: list[str] = []
        contributions: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            provenance = tuple(getattr(candidate, "provenance", ()) or ())
            sources = {str(getattr(record, "source", "")).replace("CandidateSource.", "").lower()
                       for record in provenance}
            for source in sources:
                by_source[source] += 1
            if len(sources) > 1:
                overlap.append(str(candidate.parent_asin))
            contributions[str(candidate.parent_asin)] = [
                {
                    "source": str(getattr(record, "source", "")).replace("CandidateSource.", "").lower(),
                    "rank": int(getattr(record, "source_rank", 0) or 0),
                    "score_kind": str(getattr(record, "score_kind", "")),
                }
                for record in provenance
            ]
        diagnostics = self.service.state_diagnostics(self.task_id) or {}
        feedback = self._retrieval_feedback()
        source_calls = (
            dict(getattr(getattr(response, "observation", None), "source_calls", {}) or {})
        )
        unattributed = sum(1 for candidate in candidates if not tuple(getattr(candidate, "provenance", ()) or ()))
        # The accepted history capability predates the ledger provenance channel, so
        # history-only candidates can be grounded yet have no provenance record.  Count
        # them only when history was actually executed; label the accounting explicitly.
        history_count = int(by_source.get("history", 0))
        if source_calls.get("history"):
            history_count += unattributed
        return {
            "source_policy": self.source_policy,
            "source_policy_weights": source_policy_weights(self.source_policy),
            "history_candidate_count": history_count,
            "catalog_candidate_count": int(by_source.get("catalog_search", 0)),
            "history_count_accounting": "history + unattributed grounded candidates",
            "overlap_count": len(overlap),
            "overlap_ids": overlap,
            "grounded_candidate_count": len(candidates),
            "fused_ids": [str(item.parent_asin) for item in (getattr(response, "recommendations", ()) or ())],
            "contributions": contributions,
            "source_calls": source_calls,
            "source_outcomes": (
                dict(getattr(getattr(response, "observation", None), "source_outcomes", {}) or {})
            ),
            "role_target": feedback.get("role_target") or dict(self.effective_role_target or {}),
            "role_match_count": feedback.get("role_match_count"),
            "role_mismatch_count": feedback.get("role_mismatch_count"),
            "role_unknown_count": feedback.get("role_unknown_count"),
            "eligibility": {
                "evaluated": feedback.get("hard_constraint_evaluated"),
                "eligible": feedback.get("hard_constraint_eligible_count"),
                "ineligible": feedback.get("hard_constraint_ineligible_count"),
                "unresolved": feedback.get("hard_constraint_unresolved_count"),
            },
            "soft_preference_summary": {
                "m5_preference_count": diagnostics.get("preference_count"),
                "hard_constraint_count": diagnostics.get("hard_constraint_count"),
                "soft_preference_count": max(
                    0,
                    int(diagnostics.get("preference_count") or 0)
                    - int(diagnostics.get("hard_constraint_count") or 0),
                ),
                "soft_evidence_visible": (
                    int(diagnostics.get("preference_count") or 0)
                    > int(diagnostics.get("hard_constraint_count") or 0)
                ),
            },
        }

    def _retrieval_feedback(self) -> dict[str, Any]:
        candidate = self.service.candidate_filter_diagnostics(self.session_id) or {}
        role = self.service.role_diagnostics(str(self.payload.user_key or self.session_id)) or {}
        return {
            "raw_recall_count": candidate.get("raw_recall_count"),
            "rrf_candidate_count": candidate.get("rrf_candidate_count"),
            "grounded_candidate_count": candidate.get("grounded_candidate_count"),
            "role_target": role.get("target") or candidate.get("role_target") or {},
            "role_match_count": candidate.get("role_match_count"),
            "role_mismatch_count": candidate.get("role_mismatch_count"),
            "role_unknown_count": candidate.get("role_unknown_count"),
            "role_rejection_reasons": candidate.get("role_rejection_reasons", ()),
            "hard_constraint_evaluated": candidate.get("hard_constraint_evaluated"),
            "hard_constraint_eligible_count": candidate.get("hard_constraint_eligible_count"),
            "hard_constraint_ineligible_count": candidate.get("hard_constraint_ineligible_count"),
            "hard_constraint_unresolved_count": candidate.get("hard_constraint_unresolved_count"),
            # candidate_filter_diagnostics is captured before the final explicit-type
            # promotion in run_shopping_turn.  Name that stage explicitly; do not call
            # it the final presented count.
            "post_role_eligibility_candidate_count": candidate.get("final_candidate_count"),
            "presented_candidate_count": self._presented_count,
        }

    def _observation(self, *, candidate_ids: tuple[str, ...], note: str = "", reranked: bool = False, terminal: bool = False, needs_user_input: bool = False) -> PlannerObservation:
        diagnostics = self.service.state_diagnostics(self.task_id) or {}
        candidate = self.service.candidate_filter_diagnostics(self.session_id) or {}
        role = self.service.role_diagnostics(str(self.payload.user_key or self.session_id)) or {}
        return PlannerObservation(
            step=diagnostics.get("turn_count", self._turn),
            query=self.payload.message,
            candidate_ids=candidate_ids,
            state_version=str(diagnostics.get("state_digest") or diagnostics.get("state_version") or ""),
            needs_user_input=needs_user_input,
            reranked=reranked,
            terminal=terminal,
            note=note,
            m5_state={
                "mode": diagnostics.get("mode"),
                "event_count": diagnostics.get("event_count"),
                "preference_count": diagnostics.get("preference_count"),
                "hard_constraint_count": diagnostics.get("hard_constraint_count"),
                "constraint_kinds": diagnostics.get("constraint_kinds", ()),
                "ranker_feature_count": diagnostics.get("ranker_feature_count"),
                "state_digest": diagnostics.get("state_digest"),
                # P7-A: canonical per-user-turn state supplied by the controlled runner.
                # It is the same objective state every arm receives; no Gold identity.
                "effective_state": dict(self.effective_state),
            },
            candidate_evidence=self._candidate_evidence(candidate_ids),
            last_retrieve_query=self._last_query or self.payload.message,
            retrieval_feedback=self._retrieval_feedback(),
            role_target=dict(
                self.effective_role_target
                or role.get("target")
                or candidate.get("role_target")
                or {}
            ),
            eligibility={
                "evaluated": candidate.get("hard_constraint_evaluated"),
                "eligible": candidate.get("hard_constraint_eligible_count"),
                "ineligible": candidate.get("hard_constraint_ineligible_count"),
                "unresolved": candidate.get("hard_constraint_unresolved_count"),
                "active_constraints": candidate.get("active_constraints", ()),
                "active_price_max": candidate.get("active_price_max"),
            },
            source_policy=self.source_policy,
            source_feedback=dict(self._source_feedback),
        )

    def execute(self, proposal: ActionProposal, observation: PlannerObservation) -> tuple[PlannerObservation, dict[str, Any], bool]:
        if proposal.action is PlannerAction.ASK:
            return (
                self._observation(
                    candidate_ids=observation.candidate_ids,
                    needs_user_input=True,
                    terminal=True,
                    note="awaiting user",
                ),
                {"tool": "ask", "question": proposal.arguments.get("question", "")},
                True,
            )
        if proposal.action is PlannerAction.RETRIEVE:
            self._turn += 1
            query = str(proposal.arguments.get("query") or self.payload.message)
            policy_value = proposal.arguments.get("source_policy")
            self.source_policy = normalize_source_policy(
                policy_value if policy_value is not None else self.source_policy
            )
            self._last_query = query
            response = self.service.run_shopping_turn(
                self.payload.model_copy(update={"message": query}),
                session_id=self.session_id,
                task_id=self.task_id,
                turn_id=f"{self.turn_prefix}:{self._turn:03d}",
                planner_retrieval=True,
                apply_user_state=not self._state_applied,
                state_message=self.state_message or None,
                source_policy=self.source_policy,
            )
            self._state_applied = True
            self.last_response = response
            observation = getattr(response, "observation", None)
            if observation is not None and hasattr(observation, "tool_calls_used"):
                self.tool_calls_used += int(observation.tool_calls_used)
            self._candidates = tuple(response.recommendations)
            ids = tuple(str(item.parent_asin) for item in self._candidates)
            self._presented_count = len(ids)
            terminal = response.terminal.termination_reason in {"awaiting_user", "failed"}
            feedback = self._retrieval_feedback()
            self._source_feedback = self._build_source_feedback(response)
            return (
                self._observation(
                    candidate_ids=ids,
                    note=f"route={response.route}; status={response.terminal.status}",
                    terminal=terminal,
                    needs_user_input=response.terminal.termination_reason == "awaiting_user",
                ),
                {
                    "tool": "run_shopping_turn",
                    "query": query,
                    "route": response.route,
                    "candidate_count": len(ids),
                    "terminal": response.terminal.termination_reason,
                    "state_version": str((self.service.state_diagnostics(self.task_id) or {}).get("state_digest") or ""),
                    "retrieval_feedback": feedback,
                    "source_policy": self.source_policy,
                    "source_feedback": self._source_feedback,
                },
                terminal,
            )
        if proposal.action is PlannerAction.RERANK:
            ranking, _combined, _soft, diagnostics = self.service._m5_soft_rerank(
                self.session_id,
                self._candidates,
                existing=(),
            )
            if ranking is not None:
                ordered = ranking.by_identity()
                self._candidates = tuple(
                    sorted(self._candidates, key=lambda item: ordered[str(item.parent_asin)].reranked_rank)
                )
            return (
                self._observation(
                    candidate_ids=tuple(str(item.parent_asin) for item in self._candidates),
                    reranked=True,
                    note="existing M6.1 soft rerank",
                ),
                {"tool": "m6_1_soft_rerank", "applied": ranking is not None, "diagnostics": diagnostics},
                False,
            )
        if proposal.action is PlannerAction.FINALIZE:
            ids = tuple(str(item.parent_asin) for item in self._candidates)
            return (
                self._observation(candidate_ids=ids, terminal=True, reranked=True, note="finalized"),
                {"tool": "finalize", "candidate_count": len(ids)},
                True,
            )
        raise ValueError(f"unsupported planner action {proposal.action!r}")
