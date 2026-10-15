"""Memory-aware ranking of one agent run's presented candidates (V2 E-02A).

Where this sits, and why
------------------------
An agent turn's candidates come from the trusted candidate plane and its ledger: several sources
retrieve, the grounding verifier drops what it cannot confirm, and rank fusion orders what is left.
That fused order is the run's **baseline** order, and this module never changes it - it produces a
second, distinguishable order by reusing the accepted preference components, in the accepted order::

    fused candidates (ledger order, real identities, real provenance)
      -> M8  ProductEnricher               trusted catalogue metadata per identity, or "missing"
      -> M10A PreferenceCandidateMatcher   one real evidence record per (candidate, ACTIVE preference)
      -> M10B PreferenceReranker           canonical key: violations ASC, matches DESC, then baseline
      -> MemoryRanking                     positions + evidence + counts, projected for the response

The four things this module is careful about
--------------------------------------------
* **No second preference rule set.**  Matching is M10A's and ordering is M10B's; this module adapts
  the candidate envelope and projects the result.  It has no matching logic of its own.
* **No fabricated model scores.**  Candidates are handed to M10A carrying the score the accepted
  provenance already holds for them: the real single-model score when the history source produced
  the candidate, and the plane's deliberate ``0.0`` "no model score" placeholder for a lexical or
  similarity candidate.  A fusion/BM25/rank-fusion value is never placed in that slot - the ledger
  owns it and the response publishes it under its own ``score_kind``.  M10A's own report labels that
  copied slot ``sasrec_score`` for historical reasons, which is exactly why this module never
  publishes the report: the response keeps the ledger's ``score`` + ``score_kind`` untouched.
* **No filtering, no re-scoring, no re-provenancing.**  Every candidate that goes in comes out, with
  identity, ``item_id``, baseline position and provenance copied through unchanged.  The candidate
  set is compared before and after, so a reranker that added or dropped one is an error rather than
  a silent loss.
* **Missing metadata stays missing.**  A candidate with no catalogue record is enriched as
  ``metadata_status="missing"`` with ``metadata=None``; M10A then reports ``unknown`` for the
  preferences it cannot decide and M10B neither rewards nor punishes it.  Nothing here invents an
  attribute to make an ordering come out a particular way.

Why the baseline order is also the tie-break
--------------------------------------------
M10B's key ends in the candidate's ``original_rank``, and this module sets that to the candidate's
**baseline (fused) position**.  Two consequences follow, and both are asserted in the tests: with no
ACTIVE preference every key is ``(0, 0, baseline_position, item_id)``, so an unranked run keeps the
fused order exactly; and candidates with equal evidence are ordered by fusion, never by a second
scoring rule of this module's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from recommendation.memory.schemas import PreferenceMemorySnapshot
from recommendation.tools.schemas import RecommendationToolResult, ToolRecommendation

__all__ = [
    "MemoryRanking",
    "MemoryRankingError",
    "RankedCandidate",
    "rank_with_memory",
]


class MemoryRankingError(RuntimeError):
    """The reranker did not return the candidate set it was given.

    Raised instead of dropping, adding or renumbering a candidate: a run's presented set is the
    ledger's, and a component that changes it has broken the contract rather than ranked it.
    """

    code = "memory_ranking_error"


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate's memory-aware position, evidence and unchanged provenance."""

    parent_asin: str
    item_id: int
    #: 1-based position in the run's baseline (fused) order - the order the ledger produced.
    rank: int
    #: 1-based position under the accepted M10B key; contiguous and unique.
    reranked_rank: int
    match_count: int
    violation_count: int
    unknown_count: int
    #: The M10A evidence records for this candidate, in the order preferences were evaluated.
    evidence: tuple[Any, ...] = ()
    #: M10B's own machine-readable reason for this candidate's placement, when it provides one.
    reason: str | None = None


@dataclass(frozen=True)
class MemoryRanking:
    """The outcome: the same candidates, in the accepted preference order."""

    candidates: tuple[RankedCandidate, ...]
    #: The run's fused order (unchanged, for comparison and for the audit block).
    baseline_order: tuple[str, ...]
    #: The same identities under the M10B key.
    ranked_order: tuple[str, ...]
    moved_count: int
    active_preference_count: int
    sort_key: str

    @property
    def changed_order(self) -> bool:
        """True when the preference order differs from the baseline order.

        Named deliberately unlike the audit block's ``reranking_applied``, which means "the ranking
        ran": M10B can run, evaluate every candidate and move none of them, and that is a different
        fact from *this* one.
        """
        return self.ranked_order != self.baseline_order

    def position(self, parent_asin: str) -> int | None:
        """The memory-aware position of one identity, or ``None`` when it is not a candidate."""
        for candidate in self.candidates:
            if candidate.parent_asin == parent_asin:
                return candidate.reranked_rank
        return None

    def by_identity(self) -> dict[str, RankedCandidate]:
        """This ranking indexed by identity."""
        return {candidate.parent_asin: candidate for candidate in self.candidates}


def _narrowed_result(
    tool_result: RecommendationToolResult | None,
    candidates: Sequence[Any],
) -> RecommendationToolResult:
    """The enrichment envelope for exactly the presented candidates, in their own order.

    The candidates' own scores are copied through unchanged.  They are *not* re-derived from the
    ledger: the accepted plane path already decided what belongs in that field (the real
    single-model score for a history candidate, ``0.0`` for a lexical or similarity one), and
    substituting a fusion score here is precisely the mislabelling this module refuses.
    """
    recommendations = [
        ToolRecommendation(
            rank=position,
            item_id=int(item.item_id),
            parent_asin=str(item.parent_asin),
            score=float(getattr(item, "score", 0.0) or 0.0),
            score_kind=getattr(item, "score_kind", None),
        )
        for position, item in enumerate(candidates, start=1)
    ]
    if tool_result is None:
        return RecommendationToolResult(
            recommendations=recommendations,
            requested_k=len(recommendations),
            returned_k=len(recommendations),
            history_length=0,
            effective_history_length=0,
            history_truncated=False,
            eligible_candidates=len(recommendations),
        )
    return tool_result.model_copy(update={
        "recommendations": recommendations,
        "requested_k": tool_result.requested_k,
        "returned_k": len(recommendations),
    })


def _active_preferences(preferences: Any) -> tuple[Any, ...]:
    """The ACTIVE entries of whatever the caller supplied (snapshot, sequence, or nothing)."""
    if preferences is None:
        return ()
    if isinstance(preferences, PreferenceMemorySnapshot):
        return tuple(preferences.active_entries)
    return tuple(entry for entry in preferences if getattr(entry, "is_active", False))


def rank_with_memory(
    *,
    candidates: Sequence[Any],
    preferences: Any,
    enricher: Any,
    matcher: Any,
    reranker: Any,
    tool_result: RecommendationToolResult | None = None,
    query: str = "",
) -> MemoryRanking | None:
    """Rank ``candidates`` with the accepted M10A/M10B semantics, or ``None`` when there is nothing
    to rank with.

    ``candidates`` are the run's *presented* candidates, in the ledger's (fused) order - the same
    objects the response is about to publish, so nothing downstream depends on this module
    re-deriving the set.  ``preferences`` is the turn's read-only M9 snapshot (or an explicit
    sequence of entries); when it holds no ACTIVE entry the baseline order stands untouched and
    ``None`` is returned, which is what makes a memory-free deployment behave exactly as before.

    ``query`` is the current user message, passed to the enricher only to select which catalogue
    facts are retrieved - it is never concatenated into a stored preference, and the preferences
    never enter it.

    Raises
    ------
    MemoryRankingError
        The reranker returned a different candidate set, or ranks that are not a contiguous
        permutation of it.
    """
    active = _active_preferences(preferences)
    if not active or not candidates:
        return None

    envelope = _narrowed_result(tool_result, candidates)
    enrichment = enricher.enrich(envelope, query)
    evidence = matcher.match(
        candidates=enrichment.items,
        preferences=preferences if isinstance(preferences, PreferenceMemorySnapshot) else active,
    )
    report = reranker.rerank(evidence)

    before = [str(item.parent_asin) for item in candidates]
    after = [str(candidate.parent_asin) for candidate in report.candidates]
    if sorted(before) != sorted(after):
        raise MemoryRankingError(
            "the reranker changed the candidate set: "
            f"{len(before)} in, {len(after)} out"
        )
    ranks = sorted(int(candidate.reranked_rank) for candidate in report.candidates)
    if ranks != list(range(1, len(after) + 1)):
        raise MemoryRankingError(f"reranked positions are not a contiguous permutation: {ranks}")

    baseline = tuple(
        str(candidate.parent_asin)
        for candidate in sorted(evidence.candidates, key=lambda item: item.original_rank)
    )
    ranked = tuple(
        RankedCandidate(
            parent_asin=str(candidate.parent_asin),
            item_id=int(candidate.item_id),
            rank=int(candidate.original_rank),
            reranked_rank=int(candidate.reranked_rank),
            match_count=int(candidate.match_count),
            violation_count=int(candidate.violation_count),
            unknown_count=int(candidate.unknown_count),
            evidence=tuple(candidate.evidence),
            reason=(
                candidate.rerank_reason.value
                if hasattr(candidate.rerank_reason, "value")
                else (str(candidate.rerank_reason) if candidate.rerank_reason else None)
            ),
        )
        for candidate in report.candidates
    )
    return MemoryRanking(
        candidates=ranked,
        baseline_order=baseline,
        ranked_order=tuple(candidate.parent_asin for candidate in ranked),
        moved_count=int(report.moved_count),
        active_preference_count=int(evidence.active_preference_count),
        sort_key=str(report.sort_key),
    )
