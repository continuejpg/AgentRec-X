"""Task-scoped candidate eligibility (AgentRec-X 2.0-alpha Phase 2).

``CandidateLedger != FeasibleCandidateView``
--------------------------------------------
The ledger answers *historical* questions: did this product ever enter the candidate
universe, which trusted source produced it, what ranks did it have, was its identity
grounded?  Those answers never change because a user set a budget.

This module answers a different and *narrower* question: for the constraints active on
**this task, right now**, is this candidate feasible?  A candidate that fails that question
is not deleted, demoted or hidden from the audit trail - it is simply not eligible to be
presented as a compliant recommendation.

The two questions must stay apart.  Deleting a violating candidate from the ledger would
destroy the evidence that the candidate was ever retrieved, which is exactly what an
evaluator needs in order to detect that a source offered something infeasible.  So:

    provenance is preserved; eligibility is constrained.

Three-state semantics
---------------------
The per-constraint verdicts are the accepted three-state ones
(:class:`~recommendation.control.grounded_reasoning.ConstraintVerdict`), and they are kept
in full.  This module only *derives* a per-candidate eligibility from them:

======================  ====================================================
``VERIFIED_ELIGIBLE``   every active hard constraint was proved to hold
``INELIGIBLE``          at least one active hard constraint was proved to fail
``UNRESOLVED``          no constraint failed, but at least one could not be decided
======================  ====================================================

The precedence is deliberate: **a proved violation outranks an unresolved one.**  A
candidate that violates ``price_max`` *and* carries an undecidable ``material`` constraint
is ``INELIGIBLE``, not ``UNRESOLVED`` - there is already enough evidence to exclude it, and
calling it merely unresolved would leave a known-bad candidate in the feasible view.

``UNKNOWN`` is neither pass nor fail
------------------------------------
A constraint the trusted catalogue cannot ground is ``UNKNOWN``.  It must not silently
satisfy the constraint (that would fabricate compliance) and must not be treated as a
violation (that would fabricate a failure).  It is carried as ``UNRESOLVED``, which is why
an unresolved candidate is never presented as a **verified** compliant match.

Task scope, and why nothing is cached
-------------------------------------
Eligibility is a statement about the current active constraint set, not a permanent
property of a product: the same product is ineligible under ``price_max=120`` and eligible
under ``price_max=200``.  This module therefore holds **no cache at all**: every view is
recomputed from the current constraints and the current grounded facts.  That is the
simplest design that cannot produce stale eligibility, and it costs one fact lookup per
candidate per constraint - not a catalogue scan.  A view is valid only for the constraints
and facts it was computed from, and the loop recomputes it whenever either changes.

Enforcement is not the model's
------------------------------
Nothing here reads a policy, a model answer or a plan.  The verdicts come from the
deterministic :class:`~recommendation.control.grounded_reasoning.GroundedReasoner`, and the
eligibility rule above is fixed.  A model policy can observe an assessment and choose a
next action; it cannot change a verdict, and it cannot make an ineligible candidate
eligible.

What is deliberately absent
---------------------------
No ranking, no ordering.  This module decides *membership*, never order.  Ordering
authority belongs to the accepted reranker, which must not be able to resurrect a candidate
that is ineligible here - which is why the only projection offered
(:meth:`FeasibleCandidateView.narrow`) preserves the caller's order and only removes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from .grounded_reasoning import (
    ConstraintKind,
    ConstraintReport,
    ConstraintVerdict,
    GroundedReasoner,
)

__all__ = [
    "CONSTRAINT_ELIGIBILITY_VERSION",
    "CandidateConstraintAssessment",
    "CandidateEligibility",
    "CandidateEligibilityEvaluator",
    "ConstraintRequirement",
    "FeasibilityProjection",
    "FeasibleCandidateView",
    "candidate_eligibility",
    "candidate_feasibility_projection",
    "constraint_kind_for",
    "constraints_from_task_state",
]

#: Version of the eligibility contract.  Bumped when the assessment or view shape changes,
#: so a stored trajectory can be read against the semantics that produced it.
CONSTRAINT_ELIGIBILITY_VERSION = 1


class CandidateEligibility(str, Enum):
    """Whether a candidate may be presented as a **verified compliant** recommendation.

    The three states are exhaustive and mutually exclusive, and the distinction between the
    last two is the point of the phase: ``INELIGIBLE`` means trusted evidence proved the
    candidate does not qualify, while ``UNRESOLVED`` means the trusted catalogue could not
    decide.  Collapsing them would either fabricate a violation or fabricate compliance.
    """

    #: Every active hard constraint was proved to hold for this candidate.
    VERIFIED_ELIGIBLE = "verified_eligible"
    #: At least one active hard constraint was proved to fail.
    INELIGIBLE = "ineligible"
    #: No constraint failed, but at least one could not be decided from grounded facts.
    UNRESOLVED = "unresolved"

    @property
    def is_verified_eligible(self) -> bool:
        """True only for the state a compliant recommendation may be built from."""
        return self is CandidateEligibility.VERIFIED_ELIGIBLE

    @property
    def is_ineligible(self) -> bool:
        """True only for a **proved** violation, never for an unknown."""
        return self is CandidateEligibility.INELIGIBLE


class ConstraintRequirement:
    """One active hard constraint, in the reasoner's own vocabulary.

    ``kind`` is a :class:`ConstraintKind`, not a preference kind: by the time a constraint
    reaches evaluation it has been narrowed to a dimension the evaluation can address - and
    when no deterministic dimension exists, it is narrowed to
    :attr:`ConstraintKind.UNVERIFIABLE` rather than dropped.  ``expected`` is the user's
    stated value, kept for the audit trail and for the message a refusal carries back to the
    policy.
    """

    __slots__ = ("kind", "expected", "source_text")

    def __init__(self, *, kind: ConstraintKind, expected: str, source_text: str = "") -> None:
        if not isinstance(kind, ConstraintKind):
            raise TypeError("a constraint requirement needs a ConstraintKind")
        self.kind = kind
        self.expected = str(expected)
        #: The user's own words, when the constraint came from task state.  Untrusted text,
        #: carried for attribution only; never re-interpreted and never a product fact.
        self.source_text = str(source_text or "")

    @property
    def is_evaluable(self) -> bool:
        """True when a deterministic catalogue check exists for this dimension."""
        return self.kind is not ConstraintKind.UNVERIFIABLE

    @property
    def label(self) -> str:
        """A short, stable, non-identifying label, e.g. ``price_max<=120``."""
        return f"{self.kind.value}<={self.expected}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {"kind": self.kind.value, "expected": self.expected}

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ConstraintRequirement({self.kind.value!r}, {self.expected!r})"


class CandidateConstraintAssessment:
    """The full, inspectable constraint assessment for one candidate.

    This is the detailed record the aggregate eligibility is *derived from* - never a
    replacement for it.  Every per-constraint verdict stays readable, with the observed
    value, so "why was this product excluded" is answerable from the trajectory instead of
    being inferred from a boolean.
    """

    __slots__ = ("parent_asin", "reports")

    def __init__(self, *, parent_asin: str, reports: Sequence[ConstraintReport]) -> None:
        self.parent_asin = str(parent_asin)
        self.reports: tuple[ConstraintReport, ...] = tuple(reports)

    @property
    def violated(self) -> tuple[ConstraintReport, ...]:
        """The constraints proved to fail.  Never includes an unknown."""
        return tuple(r for r in self.reports if r.verdict is ConstraintVerdict.VIOLATED)

    @property
    def unresolved(self) -> tuple[ConstraintReport, ...]:
        """The constraints the trusted catalogue could not decide."""
        return tuple(r for r in self.reports if r.verdict is ConstraintVerdict.UNKNOWN)

    @property
    def satisfied(self) -> tuple[ConstraintReport, ...]:
        """The constraints proved to hold."""
        return tuple(r for r in self.reports if r.verdict is ConstraintVerdict.SATISFIED)

    @property
    def eligibility(self) -> CandidateEligibility:
        """Derive the task eligibility from the detailed verdicts.

        A proved violation wins over an unresolved constraint, because excluding a candidate
        already disproved needs no further evidence.  Otherwise an undecided constraint keeps
        the candidate out of the *verified* set, and only a fully satisfied set makes it
        verified eligible.
        """
        if self.violated:
            return CandidateEligibility.INELIGIBLE
        if self.unresolved:
            return CandidateEligibility.UNRESOLVED
        return CandidateEligibility.VERIFIED_ELIGIBLE

    @property
    def is_assessed(self) -> bool:
        """True when at least one constraint was actually evaluated for this candidate."""
        return bool(self.reports)

    def explain(self) -> str:
        """A short reason for a refusal message or a diagnostic, naming the constraint."""
        if self.violated:
            report = self.violated[0]
            observed = f" (observed {report.observed})" if report.observed else ""
            return f"{report.kind.value} {report.expected}{observed}"
        if self.unresolved:
            report = self.unresolved[0]
            return f"{report.kind.value} {report.expected} could not be verified"
        return "all active constraints are satisfied"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "parent_asin": self.parent_asin,
            "eligibility": self.eligibility.value,
            "reports": [report.as_dict() for report in self.reports],
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"CandidateConstraintAssessment({self.parent_asin!r}, {self.eligibility.value})"


class FeasibleCandidateView:
    """Which candidates are feasible for the **current task**, and why.

    ``assessments`` covers every candidate that was assessed, including the ineligible ones:
    the view is the projection used for membership, but it is also the record an evaluator
    reads to see what was excluded and on what evidence.

    Immutable, and valid only for the constraint set and grounded facts it was built from.
    Nothing in the loop caches it across a constraint change.
    """

    __slots__ = ("requirements", "_by_identity", "_ordered")

    def __init__(
        self,
        *,
        requirements: Sequence[ConstraintRequirement],
        assessments: Sequence[CandidateConstraintAssessment],
        ordered: Sequence[str] = (),
    ) -> None:
        self.requirements: tuple[ConstraintRequirement, ...] = tuple(requirements)
        self._by_identity = {a.parent_asin: a for a in assessments}
        #: The candidate order the view was computed over.  Stored so the feasible projection
        #: is the caller's own order, filtered - never a reordering.
        self._ordered: tuple[str, ...] = tuple(ordered)

    # -- membership -------------------------------------------------------- #

    @property
    def has_requirements(self) -> bool:
        """True when at least one hard constraint is active for this task."""
        return bool(self.requirements)

    @property
    def has_evaluable_requirements(self) -> bool:
        """True when at least one active constraint has a deterministic catalogue check."""
        return any(r.is_evaluable for r in self.requirements)

    @property
    def assessments(self) -> tuple[CandidateConstraintAssessment, ...]:
        """Every assessment, in the order the candidates were supplied."""
        return tuple(self._by_identity.values())

    @property
    def assessed_identities(self) -> tuple[str, ...]:
        """The identities this view assessed, in the order supplied."""
        return self._ordered

    @property
    def feasible_identities(self) -> tuple[str, ...]:
        """The verified-feasible projection: the input order, filtered.  Never reordered."""
        return self.narrow(self._ordered)

    def assessment_for(self, parent_asin: str) -> CandidateConstraintAssessment | None:
        """The assessment for one candidate, or ``None`` when it was never assessed."""
        return self._by_identity.get(parent_asin)

    def eligibility_of(self, parent_asin: str) -> CandidateEligibility | None:
        """The eligibility of one candidate, or ``None`` when it was never assessed."""
        assessment = self._by_identity.get(parent_asin)
        return None if assessment is None else assessment.eligibility

    def is_feasible(self, parent_asin: str) -> bool:
        """True only for a candidate proved to satisfy every active hard constraint.

        An assessed-but-unresolved candidate is *not* feasible, and neither is one that was
        never assessed: the feasible view admits only verified compliance, which is what
        makes it safe for a presenter to consume directly.
        """
        assessment = self._by_identity.get(parent_asin)
        return assessment is not None and assessment.eligibility.is_verified_eligible

    def narrow(self, parent_asins: Sequence[str]) -> tuple[str, ...]:
        """Return the members of ``parent_asins`` that are verified feasible, in order.

        This is the one function a presenter needs.  It **preserves the caller's order** and
        only removes identities, so it cannot be used to reorder candidates - ordering stays
        with the accepted reranker.  A duplicate in the input is preserved as-is, because a
        duplicate identity is a caller bug the verifier already refuses, and silently
        de-duplicating here would hide it.
        """
        return tuple(a for a in parent_asins if self.is_feasible(a))

    def exclusions(self) -> tuple[tuple[str, str], ...]:
        """``(identity, reason)`` for every assessed candidate that is not feasible.

        The audit-facing view of what the feasible projection removed.  Ordered like the
        assessment, so a report reads in candidate order.
        """
        return tuple(
            (assessment.parent_asin, assessment.explain())
            for assessment in self._by_identity.values()
            if not assessment.eligibility.is_verified_eligible
        )

    # -- counts, for a bounded policy projection --------------------------- #

    def count(self, eligibility: CandidateEligibility) -> int:
        """How many assessed candidates hold one eligibility state."""
        return sum(1 for a in self._by_identity.values() if a.eligibility is eligibility)

    @property
    def verified_eligible_count(self) -> int:
        """Candidates proved to satisfy every active hard constraint."""
        return self.count(CandidateEligibility.VERIFIED_ELIGIBLE)

    @property
    def ineligible_count(self) -> int:
        """Candidates proved to violate at least one active hard constraint."""
        return self.count(CandidateEligibility.INELIGIBLE)

    @property
    def unresolved_count(self) -> int:
        """Candidates with no violation but at least one undecided constraint."""
        return self.count(CandidateEligibility.UNRESOLVED)

    @property
    def label(self) -> str:
        """A compact identity for the view, so a policy can tell two views apart.

        Opaque and non-reversible: counts only, never an identity.  ``""`` when no constraint
        is active, which is how a consumer tells "nothing was filtered" from "everything
        passed".
        """
        if not self.requirements:
            return ""
        return (
            f"eligibility:{self.verified_eligible_count}"
            f"/{self.ineligible_count}"
            f"/{self.unresolved_count}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable, payload-light view for a trajectory record."""
        return {
            "version": CONSTRAINT_ELIGIBILITY_VERSION,
            "requirements": [r.as_dict() for r in self.requirements],
            "verified_eligible_count": self.verified_eligible_count,
            "ineligible_count": self.ineligible_count,
            "unresolved_count": self.unresolved_count,
            "assessments": [a.as_dict() for a in self.assessments],
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"FeasibleCandidateView(requirements={len(self.requirements)}, "
            f"verified={self.verified_eligible_count}, "
            f"ineligible={self.ineligible_count}, unresolved={self.unresolved_count})"
        )


def candidate_eligibility(
    parent_asins: Sequence[str],
    *,
    requirements: Sequence[ConstraintRequirement],
    reasoner: Any,
) -> FeasibleCandidateView:
    """Evaluate ``requirements`` against ``parent_asins`` and derive task eligibility.

    Pure and stateless: the same constraints and the same grounded facts always produce the
    same view, whatever policy is running.  No catalogue-wide work happens here - the cost is
    one fact lookup per candidate per constraint, over the run's own grounded set, which is
    what keeps a per-turn eligibility update cheap.

    A candidate the reasoner cannot ground at all is still assessed, and every constraint
    against it resolves to ``UNKNOWN`` (the reasoner's own rule: a missing fact is never a
    pass and never a violation).  It therefore lands in ``UNRESOLVED`` rather than being
    silently dropped - dropping it would turn "we could not check this" into "this is fine".
    """
    if not isinstance(reasoner, GroundedReasoner):
        raise TypeError("eligibility evaluation needs a GroundedReasoner over the catalogue")

    ordered = tuple(dict.fromkeys(str(a) for a in parent_asins if a))
    assessments = tuple(
        CandidateConstraintAssessment(
            parent_asin=parent_asin,
            reports=tuple(
                reasoner.check_constraint(
                    parent_asin, kind=requirement.kind, expected=requirement.expected
                )
                for requirement in requirements
            ),
        )
        for parent_asin in ordered
    )
    return FeasibleCandidateView(
        requirements=requirements, assessments=assessments, ordered=ordered
    )


#: Which reasoner constraint dimension evaluates which preference kind.  A preference kind
#: with no entry here (``free_form_constraint``) has no deterministic check at all, so it maps to
#: :attr:`ConstraintKind.UNVERIFIABLE` rather than being dropped: dropping it would present a
#: candidate as compliant with a constraint that was never tested.
_PREFERENCE_TO_CONSTRAINT: dict[str, ConstraintKind] = {
    "price_max": ConstraintKind.PRICE_MAX,
    "price_min": ConstraintKind.PRICE_MIN,
    "category": ConstraintKind.CATEGORY,
    "brand": ConstraintKind.BRAND,
    "color": ConstraintKind.COLOR,
    # Phase 4: material and feature are decidable from **acquired trusted evidence**.  Before the
    # evidence layer existed they mapped to UNVERIFIABLE, which is why a stated material
    # constraint was permanently UNKNOWN.  They belong here now because a dimension that *can* be
    # resolved must be requested as a requirement, not discarded as unsupported.
    "material": ConstraintKind.MATERIAL,
    "feature": ConstraintKind.FEATURE,
}


def constraint_kind_for(preference_kind: Any) -> ConstraintKind:
    """Map one preference kind onto a constraint dimension, or ``UNVERIFIABLE``.

    Returning the explicit unmappable dimension (rather than ``None``) is what keeps a
    stated-but-untestable hard constraint from vanishing: it stays active, every candidate
    resolves to ``UNKNOWN`` against it, and no candidate can be presented as verified
    compliant while it is in force.
    """
    value = getattr(preference_kind, "value", preference_kind)
    if value is None:
        return ConstraintKind.UNVERIFIABLE
    return _PREFERENCE_TO_CONSTRAINT.get(str(value).strip().casefold(), ConstraintKind.UNVERIFIABLE)


def constraints_from_task_state(task_state: Any) -> tuple[ConstraintRequirement, ...]:
    """Read the **active hard** constraints out of task state as checkable requirements.

    Only ``TaskState.hard_constraints()`` is consulted, so the existing hardness rule is what
    decides feasibility: an explicit constraint from this turn or this session qualifies, an
    inferred or memory-derived signal does not.  Soft constraints and behavioural signals are
    therefore structurally unable to narrow the feasible domain - they remain ordering
    inputs, which is the accepted preference/hard-constraint separation.

    An empty result means "no hard constraint is active", which the view distinguishes from
    "constraints are active but none could be evaluated".
    """
    if task_state is None:
        return ()
    return tuple(
        ConstraintRequirement(
            kind=constraint_kind_for(getattr(constraint, "kind", None)),
            expected=str(getattr(constraint, "value", "")),
            source_text=str(getattr(constraint, "source_text", "") or ""),
        )
        for constraint in getattr(task_state, "hard_constraints", lambda: ())() or ()
    )


class FeasibilityProjection:
    """The feasible narrowing of the artifacts a renderer consumes.

    Two things must be true at once, and they pull in opposite directions:

    * the **presented** recommendation set must contain only candidates proved to satisfy
      every active hard constraint;
    * the **audit** record of what was retrieved, enriched and reranked must survive intact.

    So this is a *projection*, not a mutation.  The trusted artifacts in the graph state are
    left exactly as the accepted stages produced them, and a narrowed copy is handed to the
    renderer.  A violated candidate is therefore still in the ledger, still in the tool
    result, still in the trajectory - and absent from the answer.

    ``tool_result``, ``enrichment`` and ``reranking`` are narrowed to the **same** identity
    set, because the renderers align them by identity and refuse a mismatch (a reranked
    candidate with no enriched record is a hard error, and rightly so).  Narrowing all three
    together is what keeps that check meaningful rather than something to bypass.
    """

    __slots__ = ("view", "tool_result", "enrichment", "reranking", "excluded")

    def __init__(
        self,
        *,
        view: FeasibleCandidateView,
        tool_result: Any = None,
        enrichment: Any = None,
        reranking: Any = None,
    ) -> None:
        self.view = view
        self.tool_result = tool_result
        self.enrichment = enrichment
        self.reranking = reranking
        #: ``(identity, reason)`` for everything the narrowing removed.  Kept on the
        #: projection so a response layer or an evaluator can explain an exclusion without
        #: re-deriving the verdicts.
        self.excluded: tuple[tuple[str, str], ...] = tuple(
            (identity, reason)
            for identity, reason in view.exclusions()
            if identity in view.assessed_identities
        )

    @property
    def is_empty(self) -> bool:
        """True when no candidate is verified feasible.

        The caller must not fall back to the unfiltered set: "nothing is verifiably
        compliant" is a real answer, and substituting the least-bad violator for it is the
        exact failure this phase exists to prevent.
        """
        return not self.view.feasible_identities

    @property
    def presented_identities(self) -> tuple[str, ...]:
        """The identities the narrowed artifacts carry, in the run's own order."""
        return self.view.feasible_identities

    @property
    def excluded_identities(self) -> tuple[str, ...]:
        """The identities removed from the presented set, in assessment order."""
        return tuple(identity for identity, _ in self.excluded)

    def exclusion_reason(self, parent_asin: str) -> str | None:
        """Why one identity was excluded, or ``None`` when it was not."""
        for identity, reason in self.excluded:
            if identity == parent_asin:
                return reason
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for a trajectory record."""
        return {
            "presented": list(self.presented_identities),
            "excluded": [{"parent_asin": a, "reason": r} for a, r in self.excluded],
            "verified_eligible_count": self.view.verified_eligible_count,
            "ineligible_count": self.view.ineligible_count,
            "unresolved_count": self.view.unresolved_count,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"FeasibilityProjection(presented={len(self.presented_identities)}, "
            f"excluded={len(self.excluded)})"
        )


def _narrow_tool_result(result: Any, feasible: tuple[str, ...]) -> Any:
    """Copy ``result`` with only the feasible recommendations, ranks left untouched.

    ``rank`` is preserved rather than renumbered: it is the source's own rank, and the
    presentation already prints it as the upstream rank.  Renumbering here would make the
    response claim a contiguity the constrained set does not have.
    """
    if result is None:
        return None
    kept = tuple(item for item in result.recommendations if item.parent_asin in feasible)
    if len(kept) == len(result.recommendations):
        return result
    return result.model_copy(update={"recommendations": list(kept), "returned_k": len(kept)})


def _narrow_enrichment(enrichment: Any, feasible: tuple[str, ...]) -> Any:
    """Copy an ``EnrichmentResult`` down to the feasible candidates."""
    if enrichment is None:
        return None
    kept = tuple(item for item in enrichment.items if item.parent_asin in feasible)
    if len(kept) == len(enrichment.items):
        return enrichment
    return enrichment.model_copy(
        update={"items": kept, "returned_k": len(kept), "requested_k": len(kept)}
    )


def _narrow_reranking(reranking: Any, feasible: tuple[str, ...]) -> Any:
    """Copy a ``RerankingReport`` down to the feasible candidates.

    ``candidate_count`` is updated to the narrowed length - it is a count, not a rank, so
    leaving it stale would misreport the list it sits beside.  ``reranked_rank`` and
    ``original_rank`` are left as the accepted reranker assigned them: they are provenance,
    and rewriting them would fabricate a different reranking than the one that ran.
    """
    if reranking is None:
        return None
    kept = tuple(c for c in reranking.candidates if c.parent_asin in feasible)
    if len(kept) == len(reranking.candidates):
        return reranking
    return reranking.model_copy(
        update={"candidates": kept, "candidate_count": len(kept)}
    )


def candidate_feasibility_projection(
    view: FeasibleCandidateView,
    tool_result: Any = None,
    enrichment: Any = None,
    reranking: Any = None,
) -> FeasibilityProjection:
    """Narrow the renderer-facing artifacts to the candidates proved feasible.

    Pure: the inputs are frozen accepted models and are never mutated.  When every presented
    candidate is feasible (the common case) each artifact is returned unchanged, so an
    unconstrained or fully compliant run takes no copying cost and renders through exactly
    the bytes it did before.
    """
    feasible = view.feasible_identities
    return FeasibilityProjection(
        view=view,
        tool_result=_narrow_tool_result(tool_result, feasible),
        enrichment=_narrow_enrichment(enrichment, feasible),
        reranking=_narrow_reranking(reranking, feasible),
    )


class CandidateEligibilityEvaluator:
    """The loop's entry point into deterministic constraint eligibility.

    Injected into the controller exactly like the reasoning executor and the candidate plane,
    so the loop depends on this contract rather than on how a verdict is computed.  Two
    properties matter and both come from being a thin wrapper over
    :func:`candidate_eligibility`:

    * it is **stateless**, so a verdict can never be stale - there is nothing to invalidate
      when the constraint set or the grounded facts change;
    * it has **no access to a policy**, so no policy implementation can influence a verdict.
      A model policy and the deterministic policy receive identical eligibility for identical
      grounded facts, which is what makes enforcement independent of policy intelligence.
    """

    def __init__(self, reasoner: Any) -> None:
        if not isinstance(reasoner, GroundedReasoner):
            raise TypeError(
                "the eligibility evaluator needs a GroundedReasoner over the catalogue"
            )
        self._reasoner = reasoner

    @property
    def reasoner(self) -> GroundedReasoner:
        """The read-only reasoner this evaluator derives verdicts from."""
        return self._reasoner

    def assess(
        self,
        parent_asins: Sequence[str],
        *,
        requirements: Sequence[ConstraintRequirement],
    ) -> FeasibleCandidateView:
        """Derive the task-scoped feasible view for ``parent_asins``.

        Returns a view with no requirements when none are active, which callers read as
        "nothing to enforce" - distinct from a view whose every candidate is unresolved.
        """
        return candidate_eligibility(
            parent_asins, requirements=requirements, reasoner=self._reasoner
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"CandidateEligibilityEvaluator({type(self._reasoner).__name__})"
