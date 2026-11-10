"""P14-B1 Recommendation Decision Gate (deterministic, evidence-only, no search).

The gate answers one question: **does the evidence currently available justify
FINALIZE?**  It is not a ranker, not a planner, and it never retrieves anything.  It
consumes the already-produced Scorer rows (or search-order candidates), the active
preferences recovered through the legal history reader, the stated hard requirements,
and the tool-budget counters.  It never reads evaluator-only labels, target ids,
expected decisions or ground truth.

Output actions map onto the existing Planner vocabulary:

* ``ALLOW_FINALIZE``                 -> FINALIZE
* ``NEED_USER_CLARIFICATION``        -> ASK (a real question the user can answer)
* ``NEED_MORE_PRODUCT_EVIDENCE``     -> RETRIEVE/RERANK (inspect product facts)
* ``NEED_MORE_CANDIDATES``           -> RETRIEVE (search again within budget)
* ``NO_VERIFIED_CANDIDATE_IN_SCOPE`` -> bounded refusal (never claims the whole
                                        catalogue has no matching product)
* ``BUDGET_EXHAUSTED``               -> bounded refusal after tool budget is spent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "GATE_VERSION",
    "GateAction",
    "GateBudget",
    "GateContext",
    "GateDecision",
    "DecisionGate",
]

GATE_VERSION = "decision_gate_v0.1"

ALLOW_FINALIZE = "ALLOW_FINALIZE"
NEED_USER_CLARIFICATION = "NEED_USER_CLARIFICATION"
NEED_MORE_PRODUCT_EVIDENCE = "NEED_MORE_PRODUCT_EVIDENCE"
NEED_MORE_CANDIDATES = "NEED_MORE_CANDIDATES"
NO_VERIFIED_CANDIDATE_IN_SCOPE = "NO_VERIFIED_CANDIDATE_IN_SCOPE"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class GateAction:
    ALLOW_FINALIZE = ALLOW_FINALIZE
    NEED_USER_CLARIFICATION = NEED_USER_CLARIFICATION
    NEED_MORE_PRODUCT_EVIDENCE = NEED_MORE_PRODUCT_EVIDENCE
    NEED_MORE_CANDIDATES = NEED_MORE_CANDIDATES
    NO_VERIFIED_CANDIDATE_IN_SCOPE = NO_VERIFIED_CANDIDATE_IN_SCOPE
    BUDGET_EXHAUSTED = BUDGET_EXHAUSTED


@dataclass(frozen=True)
class GateBudget:
    """Explicit, bounded tool budget for a single task."""

    max_retrievals: int = 1
    max_product_evidence_rounds: int = 1
    retrievals_used: int = 0
    product_evidence_rounds_used: int = 0

    def __post_init__(self) -> None:
        if self.max_retrievals < 0 or self.max_product_evidence_rounds < 0:
            raise ValueError("budget bounds must be >= 0")

    @property
    def retrievals_remaining(self) -> int:
        return max(0, self.max_retrievals - self.retrievals_used)

    @property
    def product_evidence_remaining(self) -> int:
        return max(0, self.max_product_evidence_rounds - self.product_evidence_rounds_used)

    def as_dict(self) -> dict[str, int]:
        return {
            "max_retrievals": self.max_retrievals,
            "max_product_evidence_rounds": self.max_product_evidence_rounds,
            "retrievals_used": self.retrievals_used,
            "product_evidence_rounds_used": self.product_evidence_rounds_used,
            "retrievals_remaining": self.retrievals_remaining,
            "product_evidence_remaining": self.product_evidence_remaining,
        }


@dataclass
class GateContext:
    """Everything the gate may look at; all fields are agent-visible at decision time."""

    query: str
    candidates: Sequence[Mapping[str, Any]]
    active_preferences: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    hard_requirements: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    preference_conflicts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    namespace: str = ""
    query_specificity: bool | None = None
    ask_pending: bool = False
    expected_product_type: str | None = None
    expected_product_role: str | None = None
    budget: GateBudget = field(default_factory=GateBudget)
    extra_reasons: Sequence[str] = field(default_factory=tuple)
    #: Explicit current-task exclusions (e.g. "for this purchase, avoid wired").  These
    #: are task-scoped, not M9 state; rows carry the evaluated task_constraint_conflicts.
    task_constraints: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass
class GateDecision:
    action: str
    allow_finalize: bool
    reason_code: str
    reason: str
    selected_candidate_id: str | None = None
    candidate_presentable_ids: tuple[str, ...] = ()
    hard_constraint_status: dict[str, str] = field(default_factory=dict)
    preference_conflict_status: str = "none"
    unknown_fields: tuple[str, ...] = ()
    product_type: str | None = None
    product_role: str | None = None
    evidence_refs: tuple[dict[str, Any], ...] = ()
    ask_question: str | None = None
    bounded_scope: bool = True
    budget_used: dict[str, int] = field(default_factory=dict)
    gate_version: str = GATE_VERSION
    reasons: tuple[str, ...] = ()
    task_constraint_status: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "allow_finalize": self.allow_finalize,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_presentable_ids": list(self.candidate_presentable_ids),
            "hard_constraint_status": dict(self.hard_constraint_status),
            "preference_conflict_status": self.preference_conflict_status,
            "unknown_fields": list(self.unknown_fields),
            "product_type": self.product_type,
            "product_role": self.product_role,
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "ask_question": self.ask_question,
            "bounded_scope": self.bounded_scope,
            "budget_used": dict(self.budget_used),
            "gate_version": self.gate_version,
            "reasons": list(self.reasons),
            "task_constraint_status": self.task_constraint_status,
        }


def _candidate_id(candidate: Mapping[str, Any]) -> str | None:
    ref = candidate.get("product_ref") or {}
    item_id = ref.get("item_id")
    return str(item_id) if item_id else None


def _namespace(candidate: Mapping[str, Any]) -> str:
    return str((candidate.get("product_ref") or {}).get("namespace") or "")


def _hard_tier(candidate: Mapping[str, Any]) -> int:
    return int(candidate.get("hard_tier") or 0)


def _unknown_fields(candidate: Mapping[str, Any], hard_requirements: Sequence[Mapping[str, Any]]) -> list[str]:
    unknown: list[str] = []
    for requirement in hard_requirements:
        if str(requirement.get("kind")) == "unverifiable":
            unknown.append(str(requirement.get("value")))
    for item in candidate.get("unknown_preferences") or ():
        value = str(item.get("value") or "")
        if value:
            unknown.append(value)
    return sorted(set(unknown))


class DecisionGate:
    """Deterministic evidence sufficiency gate for FINALIZE."""

    def __init__(self, *, allow_soft_unknown_finalize: bool = True) -> None:
        self._allow_soft_unknown_finalize = bool(allow_soft_unknown_finalize)

    @property
    def version(self) -> str:
        return GATE_VERSION

    def check(self, context: GateContext) -> GateDecision:
        candidates = list(context.candidates)
        budget = context.budget
        budget_used = budget.as_dict()

        # 0. Namespace safety: a candidate from another catalogue is never presented.
        namespace = str(context.namespace or "")
        if not namespace:
            raise ValueError("gate requires the expected catalogue namespace")
        mismatched = sorted({_namespace(c) for c in candidates if _namespace(c) != namespace})
        if mismatched:
            return GateDecision(
                action=NO_VERIFIED_CANDIDATE_IN_SCOPE,
                allow_finalize=False,
                reason_code="namespace_mismatch",
                reason=f"candidates from foreign catalogue namespace(s) {mismatched} cannot be presented",
                hard_constraint_status={"namespace": "violated"},
                budget_used=budget_used,
            )

        # 0b. Current-task hard exclusions.  These are stronger than historical
        #     preferences for this purchase but do not delete the M9 audit trail.
        task_checked = []
        task_violations: list[Mapping[str, Any]] = []
        for candidate in candidates:
            conflicts = list(candidate.get("task_constraint_conflicts") or ())
            if conflicts:
                task_violations.append(candidate)
            else:
                task_checked.append(candidate)
        if task_violations:
            candidates = task_checked
            if not candidates:
                if budget.retrievals_remaining > 0:
                    action = NEED_MORE_CANDIDATES
                    reason_code = "current_task_constraint_violation"
                else:
                    action = NO_VERIFIED_CANDIDATE_IN_SCOPE
                    reason_code = "current_task_constraint_violation"
                return GateDecision(
                    action=action,
                    allow_finalize=False,
                    reason_code=reason_code,
                    reason=("all examined candidates violate the current task's explicit "
                            "exclusion; the excluded rows remain in the ledger for audit"),
                    hard_constraint_status={"current_task_constraints": "violated"},
                    task_constraint_status="violated",
                    bounded_scope=True,
                    budget_used=budget_used,
                )

        # 1. An unresolved clarification blocks FINALIZE; no fabricated answer is allowed.
        if context.ask_pending:
            return GateDecision(
                action=NEED_USER_CLARIFICATION,
                allow_finalize=False,
                reason_code="clarification_pending",
                reason="the agent is waiting for a user answer; FINALIZE would bypass it",
                ask_question="Please answer the pending clarification before I finalize a recommendation.",
                budget_used=budget_used,
            )

        # 2. Unresolved contradiction between active preferences needs the user to decide.
        if context.preference_conflicts:
            conflicts = [dict(item) for item in context.preference_conflicts]
            values = sorted({str(item.get("left") or item.get("value") or "?") for item in conflicts} |
                            {str(item.get("right") or "?") for item in conflicts if item.get("right")})
            question = (
                "I found conflicting preferences in your history"
                + (f" ({', '.join(values)})" if values else "")
                + ". Which one should take priority for this purchase?"
            )
            return GateDecision(
                action=NEED_USER_CLARIFICATION,
                allow_finalize=False,
                reason_code="preference_conflict_unresolved",
                reason="active history preferences contradict each other and no current hard constraint resolves them",
                preference_conflict_status="unresolved",
                ask_question=question,
                evidence_refs=tuple(
                    {"memory_id": item.get("memory_id"), "value": item.get("value")} for item in conflicts
                ),
                budget_used=budget_used,
            )

        # 3. No decision criteria at all (no hard request, no recovered preference,
        #    uninformative query) means the system would be guessing.
        hard = list(context.hard_requirements)
        specificity = context.query_specificity
        if specificity is None:
            specificity = bool(hard)
        if not hard and not list(context.active_preferences) and not specificity:
            return GateDecision(
                action=NEED_USER_CLARIFICATION,
                allow_finalize=False,
                reason_code="insufficient_decision_criteria",
                reason="the request has no recoverable preference, no hard constraint and no specific requirement; choosing would be a guess",
                ask_question=(
                    "What matters most for this purchase - for example the use case, a budget, "
                    "or a feature you need?"
                ),
                budget_used=budget_used,
            )

        def role_ok(candidate: Mapping[str, Any]) -> bool:
            if not context.expected_product_type:
                return True
            return str(candidate.get("role_status") or "NOT_REQUIRED") in {"MATCH", "NOT_REQUIRED"}

        presentable = [c for c in candidates if _hard_tier(c) == 0 and role_ok(c)]
        if not presentable:
            role_mismatch = [
                c for c in candidates
                if _hard_tier(c) == 0 and str(c.get("role_status") or "") == "MISMATCH"
            ]
            role_unknown = [
                c for c in candidates
                if _hard_tier(c) == 0 and str(c.get("role_status") or "") == "UNKNOWN"
            ]
            if role_mismatch or role_unknown:
                if role_unknown and budget.product_evidence_remaining > 0:
                    return GateDecision(
                        action=NEED_MORE_PRODUCT_EVIDENCE,
                        allow_finalize=False,
                        reason_code="product_type_unknown",
                        reason=("the request names an explicit product type but the examined candidates "
                                "do not have trusted type evidence; UNKNOWN must not be treated as MATCH"),
                        candidate_presentable_ids=tuple(filter(None, (
                            _candidate_id(c) for c in (role_unknown or role_mismatch)[:5]))),
                        hard_constraint_status={"product_type": "unknown"},
                        unknown_fields=("product_type",),
                        budget_used=budget_used,
                    )
                if budget.retrievals_remaining > 0:
                    return GateDecision(
                        action=NEED_MORE_CANDIDATES,
                        allow_finalize=False,
                        reason_code="no_candidate_of_requested_type_in_pool",
                        reason=("the examined pool has no candidate of the requested product type/role; "
                                "search wider before concluding"),
                        candidate_presentable_ids=tuple(filter(None, (
                            _candidate_id(c) for c in (role_mismatch or role_unknown)[:5]))),
                        hard_constraint_status={"product_type": "role_mismatch"},
                        unknown_fields=("product_type",),
                        budget_used=budget_used,
                    )
                return GateDecision(
                    action=NO_VERIFIED_CANDIDATE_IN_SCOPE,
                    allow_finalize=False,
                    reason_code="no_verified_candidate_of_requested_type_in_examined_scope",
                    reason=("no candidate of the requested product type/role in the currently examined "
                            "search scope; this is NOT a claim about the whole catalogue"),
                    hard_constraint_status={"product_type": "role_mismatch"},
                    unknown_fields=("product_type",),
                    bounded_scope=True,
                    budget_used=budget_used,
                )
            unknown_candidates = [c for c in candidates if _hard_tier(c) == 1]
            # 4a. UNKNOWN is not proof; first try a legal product-evidence round.
            if unknown_candidates and budget.product_evidence_remaining > 0:
                return GateDecision(
                    action=NEED_MORE_PRODUCT_EVIDENCE,
                    allow_finalize=False,
                    reason_code="hard_constraint_unknown",
                    reason=(
                        "no candidate has verifiable evidence for every hard requirement; "
                        "UNKNOWN must not be treated as satisfied"
                    ),
                    candidate_presentable_ids=tuple(filter(None, (_candidate_id(c) for c in unknown_candidates[:5]))),
                    hard_constraint_status={"hard_constraints": "unknown"},
                    unknown_fields=tuple(_unknown_fields(unknown_candidates[0], hard)),
                    budget_used=budget_used,
                )
            # 4b. Product facts cannot settle it; a legal re-search is allowed within budget.
            if budget.retrievals_remaining > 0:
                return GateDecision(
                    action=NEED_MORE_CANDIDATES,
                    allow_finalize=False,
                    reason_code="no_verified_candidate_in_pool",
                    reason="the examined pool contains no candidate with verified hard constraints; search wider before concluding",
                    candidate_presentable_ids=tuple(filter(None, (_candidate_id(c) for c in unknown_candidates[:5]))),
                    hard_constraint_status={"hard_constraints": "unknown_or_violated"},
                    unknown_fields=tuple(_unknown_fields(unknown_candidates[0], hard)) if unknown_candidates else (),
                    budget_used=budget_used,
                )
            return GateDecision(
                action=NO_VERIFIED_CANDIDATE_IN_SCOPE,
                allow_finalize=False,
                reason_code="no_verified_candidate_in_examined_scope",
                reason=(
                    "no candidate in the currently examined search scope has verifiable evidence for the "
                    "stated hard constraints; this is NOT a claim about the whole catalogue"
                ),
                hard_constraint_status={"hard_constraints": "unknown_or_violated"},
                unknown_fields=tuple(_unknown_fields(unknown_candidates[0], hard)) if unknown_candidates else (),
                bounded_scope=True,
                budget_used=budget_used,
            )

        # 5. The highest-ranked candidate that clears every hard constraint.
        chosen = presentable[0]
        chosen_id = _candidate_id(chosen)
        conflicts = list(chosen.get("conflicting_preferences") or ())
        if conflicts:
            question = (
                "The best verified candidate conflicts with one of your stated preferences "
                f"({', '.join(str(item.get('value')) for item in conflicts)}). "
                "Should I avoid it and look further, or is it acceptable?"
            )
            return GateDecision(
                action=NEED_USER_CLARIFICATION,
                allow_finalize=False,
                reason_code="candidate_conflicts_with_stated_preference",
                reason="hard constraints are verified but the candidate conflicts with an explicit negative preference",
                selected_candidate_id=chosen_id,
                preference_conflict_status="candidate_conflict",
                ask_question=question,
                budget_used=budget_used,
            )

        role_status = str(chosen.get("role_status") or ("NOT_REQUIRED" if not context.expected_product_type else "UNKNOWN"))
        unknown = _unknown_fields(chosen, hard)
        matched = [dict(item) for item in (chosen.get("matched_preferences") or ())]
        evidence_refs = tuple(
            dict(item) for item in (chosen.get("evidence_refs") or ())[:8]
        )
        soft_unknown = [item.get("value") for item in (chosen.get("unknown_preferences") or ()) if item.get("value")]
        uncertainty = bool(soft_unknown or unknown)
        if uncertainty and not self._allow_soft_unknown_finalize:
            return GateDecision(
                action=NEED_MORE_PRODUCT_EVIDENCE,
                allow_finalize=False,
                reason_code="soft_evidence_unknown",
                reason="strict gate configuration forbids finalizing with unknown soft evidence",
                selected_candidate_id=chosen_id,
                unknown_fields=tuple(_unknown_fields(chosen, hard)),
                budget_used=budget_used,
            )

        reason = "hard constraints verified; no explicit preference conflict"
        if matched:
            reason += f"; {len(matched)} positive preference(s) supported by catalogue evidence"
        if uncertainty:
            reason += "; limited recommendation: some soft/unknown evidence remains undisclosed"
        task_statuses = [
            str((c.get("task_constraint_status") or {}).get("status") or "none")
            for c in presentable
        ]
        if "violated" in task_statuses:
            task_constraint_status = "violated"
        elif "unknown" in task_statuses:
            task_constraint_status = "unknown"
        elif context.task_constraints:
            task_constraint_status = "verified"
        else:
            task_constraint_status = "none"
        return GateDecision(
            action=ALLOW_FINALIZE,
            allow_finalize=True,
            reason_code="evidence_sufficient",
            reason=reason,
            selected_candidate_id=chosen_id,
            candidate_presentable_ids=tuple(filter(None, (_candidate_id(c) for c in presentable[:10]))),
            hard_constraint_status={"hard_constraints": "verified"},
            preference_conflict_status="none",
            unknown_fields=tuple(sorted(set(unknown + [str(value) for value in soft_unknown if value]))),
            evidence_refs=evidence_refs,
            product_type=(chosen.get("product_type") or None),
            product_role=str(chosen.get("candidate_role") or role_status),
            budget_used=budget_used,
            task_constraint_status=task_constraint_status,
        )
