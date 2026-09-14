"""Reranking-policy evaluation schemas (Milestone 10C).

M10C is an **observational** layer over the accepted M10B reranker.  It answers *when
and why* the canonical policy moves candidates, and how much displacement and
adherence change it causes::

    PreferenceEvidenceReport
        -> accepted M10B reranker            (reused, never duplicated)
        -> RerankingReport
        -> M10C diagnostics                  (read-only)

There are no preference-conditioned relevance labels in this project, so **nothing here
is a recommendation-quality metric**.  The diagnostics measure policy behaviour and
agreement with explicit preference evidence only.  Names such as ``top_k_overlap``
describe stability/displacement, not accuracy, and ``matches``/``violations`` describe
explicit-constraint agreement, not relevance, satisfaction or conversion.

Every metric reports a numerator and denominator so a number can never be read as a bare
percentage.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from recommendation.preference_matching.schemas import PreferenceKind

__all__ = [
    "ATTRIBUTION_SOURCES",
    "DEFAULT_DIAGNOSTIC_K",
    "EVALUATION_SCHEMA_VERSION",
    "AdherenceAtK",
    "AttributionSource",
    "BaselineComparison",
    "MovementCause",
    "CandidateEvidenceCoverage",
    "DisplacementMetrics",
    "EvaluationInvariants",
    "MovementAttribution",
    "PolicyConsistency",
    "PolicyEvaluationRequest",
    "PositionReason",
    "PreferenceCoverage",
    "PreferenceTypeCoverage",
    "RerankingEvaluationReport",
    "TopKOverlap",
    "ViolationProtection",
    "evaluation_digest",
]

#: Schema version of the M10C evaluation payload.
EVALUATION_SCHEMA_VERSION = 1

#: Default K values for displacement and adherence diagnostics.  A K larger than the
#: candidate count is skipped rather than padded.
DEFAULT_DIAGNOSTIC_K: tuple[int, ...] = (1, 3, 5, 10)


class PositionReason(str, Enum):
    """Why a candidate occupies its position, relative to the one immediately below.

    This is an **ordering** statement, not a movement statement: it is defined for every
    candidate, including ones that never moved.  It describes the first canonical key
    that placed this candidate strictly ahead of its neighbour.
    """

    #: Key 1: fewer explicit violations than the candidate below.
    FEWER_VIOLATIONS = "fewer_violations"
    #: Key 2: at equal violations, more explicit matches than the candidate below.
    MORE_MATCHES = "more_matches"
    #: Key 3: equal evidence; the original SASRec rank placed it here.
    ORDINAL_RANK = "ordinal_rank"
    #: Key 4: every richer key tied and ``item_id`` decided.  Unreachable for valid
    #: input, because the M10B reranker rejects duplicate ``original_rank`` values.
    ITEM_ID_TIEBREAK = "item_id_tiebreak"
    #: Final position: no candidate below to be ordered against.
    LAST_POSITION = "last_position"


class MovementCause(str, Enum):
    """Why a candidate's rank changed.

    Defined **only** for candidates whose ``original_rank != reranked_rank``, so the
    counts sum exactly to ``moved_count``.  Each moved candidate is attributed by
    comparing its canonical key against the nearest candidate it crossed: the key that
    decided that specific pair decides the cause.
    """

    #: It rose above the crossed candidate on fewer explicit violations.
    FEWER_VIOLATIONS = "fewer_violations"
    #: It rose above the crossed candidate on more explicit matches at equal violations.
    MORE_MATCHES = "more_matches"
    #: Its evidence tied the crossed candidate; the original SASRec rank decided the pair.
    ORDINAL_FALLBACK = "ordinal_fallback"
    #: Defensive: a genuine ``item_id`` tie-break.  Unreachable for valid input.
    ITEM_ID_TIEBREAK = "item_id_tiebreak"


#: The order in which position reasons are checked, matching the policy key order.
ATTRIBUTION_SOURCES: tuple[PositionReason, ...] = (
    PositionReason.FEWER_VIOLATIONS,
    PositionReason.MORE_MATCHES,
    PositionReason.ORDINAL_RANK,
    PositionReason.ITEM_ID_TIEBREAK,
)


class DisplacementMetrics(BaseModel):
    """How far candidates moved.  Displacement is not quality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(default=0, ge=0)
    moved_count: int = Field(default=0, ge=0)
    promoted_count: int = Field(default=0, ge=0)
    demoted_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    total_abs_delta: int = Field(default=0, ge=0)
    mean_abs_delta: float = Field(
        default=0.0, ge=0.0, description="total_abs_delta / candidate_count (0 when empty)."
    )
    max_abs_delta: int = Field(default=0, ge=0)
    median_abs_delta: float = Field(default=0.0, ge=0.0)

    @property
    def moved_fraction(self) -> float:
        """``moved_count / candidate_count`` (0 when empty)."""
        if self.candidate_count == 0:
            return 0.0
        return self.moved_count / self.candidate_count

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, including the derived fraction."""
        payload = self.model_dump()
        payload["moved_fraction"] = round(self.moved_fraction, 6)
        return payload


class TopKOverlap(BaseModel):
    """Stability diagnostic for one prefix length.

    ``overlap`` is ``|original_top_k INTERSECT reranked_top_k| / k``: a *stability and
    displacement* measure, never recommendation accuracy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(..., ge=1)
    overlap_count: int = Field(..., ge=0, description="Size of the intersection.")
    k_effective: int = Field(..., ge=1, description="Denominator actually used.")
    overlap: float = Field(..., ge=0.0, le=1.0)
    original_top_k: tuple[str, ...] = ()
    reranked_top_k: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class AdherenceAtK(BaseModel):
    """Explicit-preference agreement counts for one prefix length.

    Counts only.  No scalar preference score is produced, and these are not relevance
    metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(..., ge=1)
    violations_before: int = Field(default=0, ge=0)
    violations_after: int = Field(default=0, ge=0)
    matches_before: int = Field(default=0, ge=0)
    matches_after: int = Field(default=0, ge=0)
    unknown_before: int = Field(default=0, ge=0)
    unknown_after: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class CandidateEvidenceCoverage(BaseModel):
    """Evidence coverage for one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str
    original_rank: int = Field(..., ge=1)
    match_count: int = Field(default=0, ge=0)
    violation_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    total_count: int = Field(default=0, ge=0)

    @property
    def known_count(self) -> int:
        """Number of non-UNKNOWN records (MATCH + VIOLATION)."""
        return self.match_count + self.violation_count

    @property
    def all_unknown(self) -> bool:
        """True when every evidence record is UNKNOWN (or there are none)."""
        return self.known_count == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["all_unknown"] = self.all_unknown
        return payload


class PreferenceCoverage(BaseModel):
    """Evidence coverage for one active preference, across the candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preference_id: str
    kind: PreferenceKind
    value: str
    candidate_count: int = Field(default=0, ge=0)
    known_count: int = Field(default=0, ge=0, description="Candidates with non-UNKNOWN evidence.")
    unknown_count: int = Field(default=0, ge=0)

    @property
    def known_fraction(self) -> float:
        """``known_count / candidate_count`` (0 when there are no candidates)."""
        if self.candidate_count == 0:
            return 0.0
        return self.known_count / self.candidate_count

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["known_fraction"] = round(self.known_fraction, 6)
        return payload


class PreferenceTypeCoverage(BaseModel):
    """Aggregate coverage for one preference kind, across the request's preferences."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PreferenceKind
    preference_count: int = Field(default=0, ge=0)
    observations: int = Field(default=0, ge=0, description="preferences x candidates.")
    known_observations: int = Field(default=0, ge=0)
    unknown_observations: int = Field(default=0, ge=0)
    variant_count: int = Field(
        default=0,
        ge=0,
        description="How many distinct values of this kind were active in the request.",
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["known_fraction"] = (
            round(self.known_observations / self.observations, 6) if self.observations else 0.0
        )
        return payload


class PreferenceTypeMovement(BaseModel):
    """How many promotions this preference kind can be credited with.

    Reported per kind so it is visible whether movement is driven mainly by
    partially-supported free-text kinds (feature/category) rather than by the
    structured ones.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PreferenceKind
    promotion_credit: int = Field(default=0, ge=0)
    violation_credit: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class PositionReasonBreakdown(BaseModel):
    """Ordering-reason distribution over **all** candidates.

    This is a position distribution, not a movement measure: its denominator is
    ``candidate_count`` and it includes candidates that never moved.  It must never be
    presented as movement attribution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fewer_violations: int = Field(default=0, ge=0)
    more_matches: int = Field(default=0, ge=0)
    ordinal_rank: int = Field(default=0, ge=0)
    item_id_tiebreak: int = Field(default=0, ge=0)
    last_position: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        """Sum of the distribution (equals ``candidate_count``)."""
        return (
            self.fewer_violations
            + self.more_matches
            + self.ordinal_rank
            + self.item_id_tiebreak
            + self.last_position
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["total"] = self.total
        return payload


class MovementAttribution(BaseModel):
    """Movement-cause distribution over **moved** candidates only.

    The denominator is ``moved_count``, and ``attributed_total`` must equal it: every
    moved candidate receives exactly one cause.  Unchanged candidates are excluded by
    construction, so this block is never a position distribution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fewer_violations: int = Field(default=0, ge=0)
    more_matches: int = Field(default=0, ge=0)
    ordinal_fallback: int = Field(default=0, ge=0)
    item_id_tiebreak: int = Field(default=0, ge=0)
    moved_count: int = Field(
        default=0, ge=0, description="Denominator: candidates whose rank changed."
    )

    @property
    def attributed_total(self) -> int:
        """Sum of all movement causes; must equal ``moved_count``."""
        return (
            self.fewer_violations
            + self.more_matches
            + self.ordinal_fallback
            + self.item_id_tiebreak
        )

    @property
    def accounting_consistent(self) -> bool:
        """True when every moved candidate received exactly one cause."""
        return self.attributed_total == self.moved_count

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["attributed_total"] = self.attributed_total
        payload["accounting_consistent"] = self.accounting_consistent
        return payload


class ViolationProtection(BaseModel):
    """Whether violation precedence is actually respected in the output.

    ``pairs_checked`` counts candidate pairs where one has at least one violation and the
    other has none.  ``inversions`` counts how many such pairs end up with the violating
    candidate ranked ahead -- which the canonical policy must make impossible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(default=0, ge=0, description="Prefix length evaluated; 0 means all.")
    pairs_checked: int = Field(default=0, ge=0)
    inversions: int = Field(default=0, ge=0)
    violating_candidates_above_clean: int = Field(default=0, ge=0)

    @property
    def holds(self) -> bool:
        """True when no inversion was found."""
        return self.inversions == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["holds"] = self.holds
        return payload


class PolicyConsistency(BaseModel):
    """Whether the emitted order actually satisfies the declared canonical key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs_checked: int = Field(default=0, ge=0)
    violations: int = Field(default=0, ge=0, description="Adjacent pairs whose keys decrease.")
    sort_key: str = Field(default="", description="The policy key the order was checked against.")

    @property
    def order_valid(self) -> bool:
        """True when every adjacent pair is non-decreasing under the canonical key."""
        return self.violations == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.model_dump()
        payload["order_valid"] = self.order_valid
        return payload


class EvaluationInvariants(BaseModel):
    """Invariants M10C asserts about every evaluated request.

    A false flag is a correctness failure, not a warning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count_unchanged: bool = True
    candidate_universe_unchanged: bool = True
    item_ids_unchanged: bool = True
    sasrec_scores_unchanged: bool = True
    evidence_unchanged: bool = True
    original_ranks_retained: bool = True
    reranked_ranks_contiguous: bool = True
    no_candidate_dropped: bool = True
    policy_order_valid: bool = True
    violation_protection_holds: bool = True
    input_not_mutated: bool = True

    @property
    def all_hold(self) -> bool:
        """True when every invariant holds."""
        return all(
            value for key, value in self.model_dump().items() if isinstance(value, bool)
        )

    def failures(self) -> tuple[str, ...]:
        """Names of the invariants that failed."""
        return tuple(
            key for key, value in self.model_dump().items() if isinstance(value, bool) and not value
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (including ``all_hold``)."""
        payload = self.model_dump()
        payload["all_hold"] = self.all_hold
        return payload


class BaselineComparison(BaseModel):
    """Diagnostic comparison of the canonical policy against a restricted baseline.

    Baselines exist only to characterise what each canonical key contributes.  They are
    **not** production alternatives and must never be wired into serving.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Baseline identifier, e.g. 'original_sasrec_order'.")
    description: str
    order: tuple[str, ...] = ()
    movement_vs_original: int = Field(default=0, ge=0)
    movement_vs_canonical: int = Field(default=0, ge=0)
    violations_at_k: int = Field(default=0, ge=0)
    matches_at_k: int = Field(default=0, ge=0)
    unknown_at_k: int = Field(default=0, ge=0)
    top_k_overlap_with_canonical: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class PolicyEvaluationRequest(BaseModel):
    """Per-request diagnostics: the unit of the evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(..., description="Scenario label or cohort member identifier.")
    candidate_count: int = Field(default=0, ge=0)
    active_preference_count: int = Field(default=0, ge=0)
    sort_key: str = ""
    diagnostics_k: tuple[int, ...] = ()

    displacement: DisplacementMetrics = Field(default_factory=DisplacementMetrics)
    top_k_overlap: tuple[TopKOverlap, ...] = ()
    adherence: tuple[AdherenceAtK, ...] = ()
    position_reasons: PositionReasonBreakdown = Field(default_factory=PositionReasonBreakdown)
    movement_attribution: MovementAttribution = Field(default_factory=MovementAttribution)
    violation_protection: ViolationProtection = Field(default_factory=ViolationProtection)
    policy_consistency: PolicyConsistency = Field(default_factory=PolicyConsistency)

    preference_coverage: tuple[PreferenceCoverage, ...] = ()
    candidate_coverage: tuple[CandidateEvidenceCoverage, ...] = ()
    type_coverage: tuple[PreferenceTypeCoverage, ...] = ()
    type_movement: tuple[PreferenceTypeMovement, ...] = ()
    baselines: tuple[BaselineComparison, ...] = ()
    invariants: EvaluationInvariants = Field(default_factory=EvaluationInvariants)

    @property
    def moved(self) -> bool:
        """True when at least one candidate changed position."""
        return self.displacement.moved_count > 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


def evaluation_digest(payload: dict[str, Any]) -> str:
    """Return a stable digest of a **deterministic** evaluation payload.

    Callers must pass only deterministic content: timing and machine metadata are
    deliberately excluded so repeated runs can be compared (the M8 artifact
    reproducibility lesson).
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RerankingEvaluationReport(BaseModel):
    """Full M10C output for one scenario suite or cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = EVALUATION_SCHEMA_VERSION
    policy: str = Field(..., description="The M10B policy identifier consumed by the evaluator.")
    starting_commit: str = ""
    cohort_rule: str = ""
    cohort_size: int = Field(default=0, ge=0)
    candidate_k: int = Field(default=0, ge=0)
    diagnostics_k: tuple[int, ...] = ()
    preference_fixture_policy: str = ""
    synthetic_preferences: bool = True
    disclaimer: str = ""
    requests: tuple[PolicyEvaluationRequest, ...] = ()
    aggregate: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    def deterministic_payload(self) -> dict[str, Any]:
        """Return the reproducible part of the report, excluding timing/machine data."""
        return json.loads(self.model_dump_json(exclude={"aggregate"})) | {
            "aggregate": {
                key: value
                for key, value in self.aggregate.items()
                if key != "latency_ms"
            }
        }

    def digest(self) -> str:
        """Stable digest over the deterministic payload."""
        return evaluation_digest(self.deterministic_payload())

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()
