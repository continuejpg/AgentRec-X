"""Reranking schemas (Milestone 10B).

M10B turns the accepted M10A evidence into a deterministic **order** without touching
anything else about a candidate:

* the same candidates come out that went in — never added, removed, replaced or
  filtered;
* ``sasrec_score`` is copied exactly and never normalised or combined;
* the M10A evidence is passed through untouched;
* the ordering is a lexicographic policy with no weights, coefficients or final score.

Ordering policy (rendered in order)::

    (violation_count, -match_count, original_rank, item_id)

i.e. **fewer explicit violations wins**, then **more explicit matches wins**, then the
original SASRec rank breaks remaining ties, and lower ``item_id`` is the final
deterministic fallback.

Why lexicographic rather than a weighted sum: raw SASRec scores are uncalibrated and
preference evidence has no validated numeric scale, so M10B prioritizes explicit
constraint adherence *without* numerically combining uncalibrated signals.  The policy
is human-auditable and each output candidate carries a machine-readable reason.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from recommendation.preference_matching.schemas import (
    EvidenceStatus,
    PreferenceEvidence,
    PreferenceEvidenceReport,
)

__all__ = [
    "RERANK_SORT_KEY_DOC",
    "CandidateEvaluation",
    "RerankReason",
    "RerankedCandidate",
    "RerankingDiagnostics",
    "RerankingError",
    "RerankingReport",
    "TopKAdherence",
]

#: The canonical sort key, documented once and reused by code, tests and docs.
RERANK_SORT_KEY_DOC = "violation_count ASC, match_count DESC, original_rank ASC, item_id ASC"


class RerankingError(Exception):
    """The reranker was given input it cannot rank without guessing."""


class RerankReason(str, Enum):
    """Machine-readable justification for a candidate's reranked position.

    Exactly one reason is assigned per candidate, chosen by comparing its evidence
    against the candidate placed directly ahead of it.
    """

    #: Ranked ahead of another candidate because it has fewer explicit violations.
    FEWER_VIOLATIONS = "fewer_violations"
    #: Ranked ahead of another candidate because, at equal violations, it has more matches.
    MORE_MATCHES = "more_matches"
    #: Nothing moved anywhere in the report, so every candidate kept its SASRec position.
    PRESERVED_ORIGINAL_ORDER = "preserved_original_order"
    #: Moved only because every richer key tied and ``item_id`` decided the position.
    DETERMINISTIC_TIE_BREAK = "deterministic_tie_break"
    #: Placed last, so there is no candidate below it to be ordered against.
    RANKED_LAST = "ranked_last"


class CandidateEvaluation(BaseModel):
    """Descriptive counts for one candidate, derived from its M10A evidence.

    These are counts only.  They are not a score: no weighting, no sign and no sum.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_count: int = Field(default=0, ge=0)
    violation_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """Return this candidate's evidence-only part of the policy key."""
        return (self.violation_count, -self.match_count)

    @classmethod
    def from_evidence(
        cls, evidence: tuple[PreferenceEvidence, ...] | list[PreferenceEvidence]
    ) -> CandidateEvaluation:
        """Count MATCH / VIOLATION / UNKNOWN records.

        UNKNOWN is counted separately and is **neutral**: it is neither a match nor a
        violation and therefore never affects the ordering key.
        """
        return cls(
            match_count=sum(1 for record in evidence if record.status is EvidenceStatus.MATCH),
            violation_count=sum(
                1 for record in evidence if record.status is EvidenceStatus.VIOLATION
            ),
            unknown_count=sum(
                1 for record in evidence if record.status is EvidenceStatus.UNKNOWN
            ),
        )


class RerankedCandidate(BaseModel):
    """One candidate after reranking.

    ``original_rank`` and ``sasrec_score`` are copied unchanged from the M10A report, and
    ``evidence`` is the same tuple of evidence records.  ``reranked_rank`` is the only
    field that describes the new order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_rank: int = Field(..., ge=1, description="SASRec rank, unchanged.")
    reranked_rank: int = Field(
        ..., ge=1, description="1-based position after reranking; contiguous and unique."
    )
    item_id: int = Field(..., description="Internal model item id, unchanged.")
    parent_asin: str = Field(..., description="External identity, unchanged.")
    sasrec_score: float = Field(
        ..., description="Raw SASRec score, copied exactly; never modified or combined."
    )

    match_count: int = Field(default=0, ge=0)
    violation_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)

    evidence: tuple[PreferenceEvidence, ...] = ()
    rerank_reason: RerankReason = RerankReason.PRESERVED_ORIGINAL_ORDER
    reason_detail: str | None = Field(
        default=None, description="Short human-readable note; never the primary contract."
    )

    @property
    def rank_delta(self) -> int:
        """``original_rank - reranked_rank``.

        Positive means the candidate was **promoted** (moved earlier), ``0`` means it kept
        its position, and negative means it was **demoted**.
        """
        return self.original_rank - self.reranked_rank

    @property
    def moved(self) -> bool:
        """True when this candidate's position changed."""
        return self.rank_delta != 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class TopKAdherence(BaseModel):
    """Diagnostic adherence counts for one prefix length.

    These measure agreement with **explicit preference evidence only**.  They are not
    relevance, satisfaction or quality metrics: M10B has no preference-conditioned
    relevance labels, so no NDCG/HR claim can be made from them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(..., ge=1)
    violations_before: int = Field(default=0, ge=0)
    violations_after: int = Field(default=0, ge=0)
    matches_before: int = Field(default=0, ge=0)
    matches_after: int = Field(default=0, ge=0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class RerankingDiagnostics(BaseModel):
    """Descriptive reranking diagnostics.  Not quality metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(default=0, ge=0)
    moved_count: int = Field(default=0, ge=0)
    promoted_count: int = Field(default=0, ge=0)
    demoted_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    active_preference_count: int = Field(default=0, ge=0)
    top_k: tuple[TopKAdherence, ...] = Field(
        default=(), description="Adherence counts for selected prefixes."
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class RerankingReport(BaseModel):
    """Full M10B output: the same candidates, in the policy order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[RerankedCandidate, ...] = ()
    candidate_count: int = Field(default=0, ge=0)
    moved_count: int = Field(default=0, ge=0)
    unchanged_count: int = Field(default=0, ge=0)
    sort_key: str = Field(
        default=RERANK_SORT_KEY_DOC,
        description="The exact ordering policy applied, for auditability.",
    )
    diagnostics: RerankingDiagnostics = Field(default_factory=RerankingDiagnostics)

    @property
    def parent_asins(self) -> tuple[str, ...]:
        """Candidate identities in reranked order."""
        return tuple(candidate.parent_asin for candidate in self.candidates)

    @property
    def item_ids(self) -> tuple[int, ...]:
        """Candidate item ids in reranked order."""
        return tuple(candidate.item_id for candidate in self.candidates)

    @property
    def original_ranks(self) -> tuple[int, ...]:
        """Original ranks in reranked order."""
        return tuple(candidate.original_rank for candidate in self.candidates)

    @property
    def reranked_ranks(self) -> tuple[int, ...]:
        """Reranked ranks, expected to be ``1..N``."""
        return tuple(candidate.reranked_rank for candidate in self.candidates)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


def evaluations_from_report(
    report: PreferenceEvidenceReport,
) -> dict[str, CandidateEvaluation]:
    """Derive per-candidate counts from an accepted M10A report, keyed by ``parent_asin``."""
    evaluations: dict[str, CandidateEvaluation] = {}
    for candidate in report.candidates:
        evaluations[candidate.parent_asin] = CandidateEvaluation.from_evidence(
            candidate.evidence
        )
    return evaluations
