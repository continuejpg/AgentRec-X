"""Deterministic preference-aware reranker (Milestone 10B).

Reorders the candidates of an accepted M10A evidence report, and nothing else::

    RecommendationTool / SASRec        (upstream, not called here)
        -> M10A PreferenceEvidenceReport
        -> M10B reranker               (this module)
        -> same candidates, new order

The reranker is a pure function over already-computed evidence.  It does not read a
memory store, parse preferences, retrieve metadata, re-run matching, call SASRec, the
RecommendationTool, FastAPI or an LLM, and it imports none of them.

Ordering policy
---------------
Lexicographic, with no weights and no combined score:

    (violation_count ASC, match_count DESC, original_rank ASC, item_id ASC)

* **fewer explicit violations wins first**, and one fewer violation beats any number of
  extra matches — explicit constraint adherence dominates;
* at equal violations, **more explicit matches wins**;
* at equal evidence, the **original SASRec rank** decides, minimising unnecessary
  movement;
* ``item_id`` ascending is the final deterministic fallback, guaranteeing a total order
  even for input that should never occur.

``UNKNOWN`` evidence is **neutral**: it is counted separately and is neither rewarded nor
penalised, so metadata sparsity cannot change ranking.

Invariants (all asserted by tests)
----------------------------------
* the output contains exactly the input candidates — same count, item ids and
  ``parent_asin`` values, multiplicity 1;
* no candidate is filtered, even when it violates a preference;
* ``sasrec_score`` is copied exactly and never normalised, shifted or combined;
* the M10A evidence records are passed through untouched;
* the M10A report is not mutated.

Complexity is ``O(n log n)`` in the candidate count for the sort, with a linear pass for
the reason attribution and diagnostics.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from recommendation.preference_matching.schemas import (
    CandidatePreferenceEvidence,
    PreferenceEvidenceReport,
)

from .schemas import (
    CandidateEvaluation,
    RERANK_SORT_KEY_DOC,
    RerankReason,
    RerankedCandidate,
    RerankingDiagnostics,
    RerankingError,
    RerankingReport,
    TopKAdherence,
)

__all__ = [
    "DEFAULT_DIAGNOSTIC_K",
    "PreferenceReranker",
    "rerank_candidates",
    "sort_key_for",
]

#: Prefix lengths reported in the adherence diagnostics.
DEFAULT_DIAGNOSTIC_K: tuple[int, ...] = (1, 3, 5, 10)


def _validate(report: PreferenceEvidenceReport) -> None:
    """Validate the ranking-relevant invariants of an M10A report.

    Malformed input fails clearly instead of being silently repaired, because a
    duplicate rank or identifier would make the ordering ambiguous.
    """
    seen_ranks: set[int] = set()
    seen_items: set[int] = set()
    seen_asins: set[str] = set()

    for candidate in report.candidates:
        if candidate.original_rank < 1:
            raise RerankingError(
                f"original_rank must be positive, got {candidate.original_rank}"
            )
        if candidate.original_rank in seen_ranks:
            raise RerankingError(
                f"duplicate original_rank {candidate.original_rank} in the evidence report"
            )
        seen_ranks.add(candidate.original_rank)

        if candidate.item_id in seen_items:
            raise RerankingError(
                f"duplicate item_id {candidate.item_id} in the evidence report"
            )
        seen_items.add(candidate.item_id)

        if candidate.parent_asin in seen_asins:
            raise RerankingError(
                f"duplicate parent_asin {candidate.parent_asin!r} in the evidence report"
            )
        seen_asins.add(candidate.parent_asin)


def sort_key_for(
    candidate: CandidatePreferenceEvidence, evaluation: CandidateEvaluation
) -> tuple[int, int, int, int]:
    """Return the canonical lexicographic key for one candidate.

    The evidence part is ``(violations ASC, -matches ASC)`` so that a plain ascending
    sort applies the policy; ``original_rank`` then ``item_id`` complete the key.
    """
    return (
        evaluation.violation_count,
        -evaluation.match_count,
        candidate.original_rank,
        candidate.item_id,
    )


def _reason_for(
    *,
    placement_index: int,
    ordered: Sequence[CandidatePreferenceEvidence],
    evaluations: dict[str, CandidateEvaluation],
) -> tuple[RerankReason, str]:
    """Attribute a machine-readable reason for a candidate's reranked position.

    The reason describes the policy key that placed a candidate strictly **ahead of the
    candidate immediately below it**: fewer violations, then more matches, then its
    original rank.  Comparing downwards rather than upwards is what makes the label
    honest -- a promoted candidate is explained by the comparison it won, rather than by
    the comparison it lost against whatever now sits above it.

    The final candidate has nothing below it to be compared against, so it is labelled
    ``RANKED_LAST`` instead of being credited with a comparison it did not win.
    """
    candidate = ordered[placement_index]
    evaluation = evaluations[candidate.parent_asin]

    if placement_index == len(ordered) - 1:
        return (
            RerankReason.RANKED_LAST,
            "last position; no candidate below to be ordered against",
        )

    below = ordered[placement_index + 1]
    below_evaluation = evaluations[below.parent_asin]

    if evaluation.violation_count < below_evaluation.violation_count:
        return (
            RerankReason.FEWER_VIOLATIONS,
            f"{evaluation.violation_count} explicit violation(s) ahead of "
            f"{below_evaluation.violation_count}",
        )
    if evaluation.violation_count == below_evaluation.violation_count and (
        evaluation.match_count > below_evaluation.match_count
    ):
        return (
            RerankReason.MORE_MATCHES,
            f"{evaluation.match_count} explicit match(es) ahead of "
            f"{below_evaluation.match_count} at equal violations",
        )
    if evaluation.violation_count == below_evaluation.violation_count and (
        evaluation.match_count == below_evaluation.match_count
    ):
        if candidate.original_rank < below.original_rank:
            return (
                RerankReason.PRESERVED_ORIGINAL_ORDER,
                "equal evidence; the original SASRec rank decided this position",
            )
        return (
            RerankReason.DETERMINISTIC_TIE_BREAK,
            "all evidence and original-rank keys tied; item_id decided this position",
        )

    # Unreachable for a sorted sequence: a candidate can never have both more violations
    # and fewer matches than the one below it.  Reported honestly rather than guessed.
    return (  # pragma: no cover - defensive
        RerankReason.PRESERVED_ORIGINAL_ORDER,
        "position follows from the evidence of the candidates placed above it",
    )


def _top_k_adherence(
    before: Sequence[CandidatePreferenceEvidence],
    after: Sequence[RerankedCandidate],
    evaluations: dict[str, CandidateEvaluation],
    ks: Sequence[int],
) -> tuple[TopKAdherence, ...]:
    """Compute diagnostic adherence counts for the leading ``k`` candidates."""
    rows: list[TopKAdherence] = []
    for k in ks:
        if k < 1 or k > len(before):
            continue
        head_before = before[:k]
        head_after = after[:k]
        rows.append(
            TopKAdherence(
                k=k,
                violations_before=sum(
                    evaluations[c.parent_asin].violation_count for c in head_before
                ),
                violations_after=sum(c.violation_count for c in head_after),
                matches_before=sum(
                    evaluations[c.parent_asin].match_count for c in head_before
                ),
                matches_after=sum(c.match_count for c in head_after),
            )
        )
    return tuple(rows)


def rerank_candidates(
    *,
    report: PreferenceEvidenceReport,
    diagnostic_k: Sequence[int] = DEFAULT_DIAGNOSTIC_K,
) -> RerankingReport:
    """Reorder an accepted M10A evidence report under the canonical policy.

    Parameters
    ----------
    report:
        The accepted M10A output.  Treated as read-only: neither the report nor its
        candidates nor its evidence records are mutated.
    diagnostic_k:
        Prefix lengths for the adherence diagnostics.  Diagnostics never affect order.

    Returns
    -------
    RerankingReport
        The same candidates, reordered, each with its ``reranked_rank``, policy counts,
        unchanged evidence and a machine-readable ``rerank_reason``.

    An empty report is valid and yields a valid empty reranking report: M10B does not
    require a recommendation to exist and never fabricates a candidate.

    Raises
    ------
    RerankingError
        The report violates a ranking-relevant invariant (duplicate or non-positive
        original rank, duplicate item id, duplicate ``parent_asin``).
    """
    _validate(report)

    evaluations = {
        candidate.parent_asin: CandidateEvaluation.from_evidence(candidate.evidence)
        for candidate in report.candidates
    }

    # Sort by the canonical key.  `original_rank` is part of the key, so the result does
    # not depend on the order the report happened to arrive in.
    ordered = sorted(
        report.candidates,
        key=lambda candidate: sort_key_for(candidate, evaluations[candidate.parent_asin]),
    )

    reranked: list[RerankedCandidate] = []
    promoted = demoted = 0
    for position, candidate in enumerate(ordered, start=1):
        evaluation = evaluations[candidate.parent_asin]
        reason, detail = _reason_for(
            placement_index=position - 1,
            ordered=ordered,
            evaluations=evaluations,
        )
        if reason is RerankReason.RANKED_LAST and position > 1:
            # The final candidate is genuinely last either because it lost the policy
            # comparison or because only item_id separated it from the one above.
            above = ordered[position - 2]
            above_evaluation = evaluations[above.parent_asin]
            if (
                evaluation.violation_count == above_evaluation.violation_count
                and evaluation.match_count == above_evaluation.match_count
                and candidate.original_rank > above.original_rank
            ):
                reason = RerankReason.DETERMINISTIC_TIE_BREAK
                detail = (
                    "all evidence and original-rank keys tied; item_id decided this "
                    "position"
                )
        record = RerankedCandidate(
            original_rank=candidate.original_rank,
            reranked_rank=position,
            item_id=candidate.item_id,
            parent_asin=candidate.parent_asin,
            sasrec_score=candidate.sasrec_score,
            match_count=evaluation.match_count,
            violation_count=evaluation.violation_count,
            unknown_count=evaluation.unknown_count,
            # Evidence is passed through untouched: M10B counts statuses, it never
            # rewrites them.
            evidence=candidate.evidence,
            rerank_reason=reason,
            reason_detail=detail,
        )
        if record.rank_delta > 0:
            promoted += 1
        elif record.rank_delta < 0:
            demoted += 1
        reranked.append(record)

    candidate_count = len(reranked)

    # Post-condition: the candidate universe is exactly preserved, multiplicity
    # included.  This can only fail if the sort above were ever changed to add, drop or
    # merge candidates; it is checked with multisets so a duplicate cannot slip through.
    if Counter(c.parent_asin for c in reranked) != Counter(
        c.parent_asin for c in report.candidates
    ):  # pragma: no cover - defensive
        raise RerankingError("reranking must preserve the candidate universe exactly")
    if Counter(c.item_id for c in reranked) != Counter(
        c.item_id for c in report.candidates
    ):  # pragma: no cover - defensive
        raise RerankingError("reranking must preserve item ids exactly")

    diagnostics = RerankingDiagnostics(
        candidate_count=candidate_count,
        moved_count=promoted + demoted,
        promoted_count=promoted,
        demoted_count=demoted,
        unchanged_count=candidate_count - promoted - demoted,
        active_preference_count=report.active_preference_count,
        top_k=_top_k_adherence(report.candidates, reranked, evaluations, diagnostic_k),
    )

    return RerankingReport(
        candidates=tuple(reranked),
        candidate_count=candidate_count,
        moved_count=promoted + demoted,
        unchanged_count=candidate_count - promoted - demoted,
        sort_key=RERANK_SORT_KEY_DOC,
        diagnostics=diagnostics,
    )


class PreferenceReranker:
    """Stateless reranker; the public seam for later milestones.

    Holds no model, store, retriever or configuration, so the same instance is safe to
    share and cannot accumulate state between runs.
    """

    def __init__(self, diagnostic_k: Sequence[int] = DEFAULT_DIAGNOSTIC_K) -> None:
        self._diagnostic_k = tuple(diagnostic_k)

    @property
    def diagnostic_k(self) -> tuple[int, ...]:
        """Prefix lengths used for the adherence diagnostics."""
        return self._diagnostic_k

    def rerank(self, report: PreferenceEvidenceReport) -> RerankingReport:
        """Reorder an accepted M10A report under the canonical policy."""
        return rerank_candidates(report=report, diagnostic_k=self._diagnostic_k)

    @staticmethod
    def sort_key(candidate: CandidatePreferenceEvidence) -> tuple[int, int, int, int]:
        """Expose the canonical key for one candidate (inspection/tests)."""
        return sort_key_for(candidate, CandidateEvaluation.from_evidence(candidate.evidence))
