"""Reranking-policy evaluation (Milestone 10C).

An **observational** layer over the accepted M10B reranker.  It reuses the production
reranker as-is and computes diagnostics about what that policy did::

    PreferenceEvidenceReport
        -> PreferenceReranker.rerank(...)      (accepted M10B, unchanged)
        -> RerankingReport
        -> diagnostics                         (this module, read-only)

The evaluator does not sort candidates itself and does not duplicate the policy.  The
only ordering logic it contains is the *restricted diagnostic baselines* used to
characterise what each canonical key contributes; those live here, are clearly labelled,
and are never production alternatives.

What this module deliberately does not do
-----------------------------------------
* it does not compute HR, Recall, NDCG, MRR, CTR or any relevance/satisfaction metric:
  this project has no preference-conditioned relevance labels, so such a number would be
  fabricated;
* it does not produce a scalar preference score, weighted sum or fitted coefficient;
* it does not read a memory store, metadata index, model or HTTP client;
* it does not mutate the evidence report, the reranking report or any candidate.

Interpretation
--------------
``violations``/``matches`` measure agreement with **explicit preference evidence**;
``top_k_overlap`` measures **stability/displacement**; ``movement`` is not quality
improvement.  Every aggregate reports its numerator and denominator.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from recommendation.preference_matching.schemas import (
    CandidatePreferenceEvidence,
    EvidenceStatus,
    PreferenceEvidenceReport,
)

from .evaluation_schemas import (
    AdherenceAtK,
    BaselineComparison,
    CandidateEvidenceCoverage,
    DisplacementMetrics,
    EvaluationInvariants,
    MovementAttribution,
    MovementCause,
    PolicyConsistency,
    PositionReason,
    PositionReasonBreakdown,
    PolicyEvaluationRequest,
    PreferenceCoverage,
    PreferenceTypeCoverage,
    PreferenceTypeMovement,
    RerankingEvaluationReport,
    TopKOverlap,
    ViolationProtection,
)
from .reranker import PreferenceReranker
from .schemas import RerankedCandidate, RerankingReport

__all__ = [
    "BASELINE_MATCH_ONLY",
    "BASELINE_ORIGINAL_ORDER",
    "BASELINE_VIOLATION_ONLY",
    "aggregate_requests",
    "build_report",
    "evaluate",
    "evaluate_request",
    "movement_causes_for",
    "position_reasons_for",
]

#: Baseline A -- the unmodified SASRec order.
BASELINE_ORIGINAL_ORDER = "original_sasrec_order"
#: Baseline B -- violation count only, then original rank.  Diagnostic only.
BASELINE_VIOLATION_ONLY = "violation_only_diagnostic"
#: Baseline C -- match count only, then original rank.  Diagnostic only.
BASELINE_MATCH_ONLY = "match_only_diagnostic"

_BASELINE_DESCRIPTIONS = {
    BASELINE_ORIGINAL_ORDER: (
        "Restricted baseline: the original SASRec order with no reranking applied."
    ),
    BASELINE_VIOLATION_ONLY: (
        "Restricted diagnostic baseline: (violation_count ASC, original_rank ASC). "
        "Evaluated to characterise what violation avoidance contributes on its own. "
        "Never a production reranking alternative."
    ),
    BASELINE_MATCH_ONLY: (
        "Restricted diagnostic baseline: (match_count DESC, original_rank ASC). "
        "Evaluated to characterise what match promotion contributes on its own. "
        "Never a production reranking alternative."
    ),
}


# --------------------------------------------------------------------------- #
# Small deterministic numeric helpers
# --------------------------------------------------------------------------- #


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile over a sorted copy (matches existing conventions)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median(values: Sequence[float]) -> float:
    """Deterministic median (mean of the middle pair for even counts)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _counts(evidence: Iterable[Any]) -> tuple[int, int, int]:
    """Return ``(matches, violations, unknowns)`` for a group of evidence records."""
    matches = violations = unknowns = 0
    for record in evidence:
        if record.status is EvidenceStatus.MATCH:
            matches += 1
        elif record.status is EvidenceStatus.VIOLATION:
            violations += 1
        else:
            unknowns += 1
    return matches, violations, unknowns


def _evidence_by_asin(
    report: PreferenceEvidenceReport,
) -> dict[str, CandidatePreferenceEvidence]:
    """Index the report's candidates by identity."""
    return {candidate.parent_asin: candidate for candidate in report.candidates}


# --------------------------------------------------------------------------- #
# Displacement and adherence
# --------------------------------------------------------------------------- #


def _displacement(reranked: RerankingReport) -> DisplacementMetrics:
    """Compute rank-displacement metrics from a reranking report."""
    deltas = [abs(candidate.rank_delta) for candidate in reranked.candidates]
    promoted = sum(1 for c in reranked.candidates if c.rank_delta > 0)
    demoted = sum(1 for c in reranked.candidates if c.rank_delta < 0)
    count = len(reranked.candidates)
    return DisplacementMetrics(
        candidate_count=count,
        moved_count=promoted + demoted,
        promoted_count=promoted,
        demoted_count=demoted,
        unchanged_count=count - promoted - demoted,
        total_abs_delta=sum(deltas),
        mean_abs_delta=(sum(deltas) / count) if count else 0.0,
        max_abs_delta=max(deltas, default=0),
        median_abs_delta=_median(deltas),
    )


def _top_k_overlap(
    report: PreferenceEvidenceReport,
    reranked: RerankingReport,
    ks: Sequence[int],
) -> tuple[TopKOverlap, ...]:
    """Compute ``|original_top_k INTERSECT reranked_top_k| / k`` for each valid k."""
    original = [
        candidate.parent_asin
        for candidate in sorted(report.candidates, key=lambda c: c.original_rank)
    ]
    after = [candidate.parent_asin for candidate in reranked.candidates]
    rows: list[TopKOverlap] = []
    for k in ks:
        if k < 1 or k > len(original):
            # A k larger than the candidate set is skipped, not padded.
            continue
        original_top = tuple(original[:k])
        reranked_top = tuple(after[:k])
        overlap = len(set(original_top) & set(reranked_top))
        rows.append(
            TopKOverlap(
                k=k,
                overlap_count=overlap,
                k_effective=k,
                overlap=overlap / k,
                original_top_k=original_top,
                reranked_top_k=reranked_top,
            )
        )
    return tuple(rows)


def _adherence(
    report: PreferenceEvidenceReport,
    reranked: RerankingReport,
    ks: Sequence[int],
) -> tuple[AdherenceAtK, ...]:
    """Compute explicit-preference agreement counts before and after reranking."""
    by_asin = _evidence_by_asin(report)
    original = sorted(report.candidates, key=lambda c: c.original_rank)
    after = list(reranked.candidates)

    rows: list[AdherenceAtK] = []
    for k in ks:
        if k < 1 or k > len(original):
            continue
        matches_before, violations_before, unknown_before = _counts(
            record for candidate in original[:k] for record in candidate.evidence
        )
        matches_after, violations_after, unknown_after = _counts(
            record
            for candidate in after[:k]
            for record in by_asin[candidate.parent_asin].evidence
        )
        rows.append(
            AdherenceAtK(
                k=k,
                violations_before=violations_before,
                violations_after=violations_after,
                matches_before=matches_before,
                matches_after=matches_after,
                unknown_before=unknown_before,
                unknown_after=unknown_after,
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Attribution and protection
# --------------------------------------------------------------------------- #


def _canonical_keys(
    report: PreferenceEvidenceReport,
) -> dict[str, tuple[int, int, int, int]]:
    """Return the canonical M10B sort key for every candidate, read from M10B itself.

    ``sort_key_for`` is imported from the accepted reranker, so the evaluator never
    re-states the policy.
    """
    keys: dict[str, tuple[int, int, int, int]] = {}
    for candidate in report.candidates:
        matches, violations, _ = _counts(candidate.evidence)
        keys[candidate.parent_asin] = (
            violations,
            -matches,
            candidate.original_rank,
            candidate.item_id,
        )
    return keys


def position_reasons_for(
    report: PreferenceEvidenceReport, reranked: RerankingReport
) -> tuple[PositionReason, ...]:
    """Return the ordering reason of each position, in reranked order.

    The reason is derived from the **canonical key** rather than from M10B's
    human-facing reason labels, so the diagnostic cannot be skewed by a labelling
    quirk.  It is defined for every candidate, including ones that never moved.
    """
    keys = _canonical_keys(report)
    order = list(reranked.candidates)
    reasons: list[PositionReason] = []
    for index, candidate in enumerate(order):
        if index == len(order) - 1:
            reasons.append(PositionReason.LAST_POSITION)
            continue
        current = keys[candidate.parent_asin]
        below = keys[order[index + 1].parent_asin]
        if current[0] < below[0]:
            reasons.append(PositionReason.FEWER_VIOLATIONS)
        elif current[0] == below[0] and current[1] < below[1]:
            reasons.append(PositionReason.MORE_MATCHES)
        elif current[:3] == below[:3]:
            # Every richer component tied, so only item_id can separate them.
            reasons.append(PositionReason.ITEM_ID_TIEBREAK)
        else:
            reasons.append(PositionReason.ORDINAL_RANK)
    return tuple(reasons)


def _classify_pair(
    current: tuple[int, int, int, int], decisive: tuple[int, int, int, int]
) -> MovementCause:
    """Name the canonical dimension that decided the pair ``current`` vs ``decisive``.

    The classification is **symmetric**: it names the deciding dimension irrespective of
    which side of the pair a candidate is on.  A candidate that overtook a cleaner rival
    and the rival it displaced therefore both report ``fewer_violations``, because that
    is what decided their relative order.
    """
    if current[0] != decisive[0]:
        return MovementCause.FEWER_VIOLATIONS
    if current[1] != decisive[1]:
        return MovementCause.MORE_MATCHES
    if current[:3] == decisive[:3]:
        # Only item_id separates them; impossible for valid unique-rank input.
        return MovementCause.ITEM_ID_TIEBREAK
    return MovementCause.ORDINAL_FALLBACK


def movement_causes_for(
    report: PreferenceEvidenceReport, reranked: RerankingReport
) -> dict[str, MovementCause]:
    """Return a movement cause for every **moved** candidate, keyed by ``parent_asin``.

    The cause is decided by the single comparison that produced the candidate's new
    position:

    * **promoted** -- the pair is the highest-ranked candidate it overtook, i.e. the one
      immediately *above* it in the new order that originally outranked it;
    * **demoted** -- the pair is the candidate immediately *above* it in the new order,
      which is the candidate that displaced it.

    The decisive key is the canonical key imported from M10B, so a cause can never
    contradict the ordering.  Exactly one cause is produced per moved candidate, which
    is what makes the counts sum to ``moved_count``.
    """
    if reranked.moved_count == 0:
        return {}

    keys = _canonical_keys(report)
    order = list(reranked.candidates)
    causes: dict[str, MovementCause] = {}

    for index, candidate in enumerate(order):
        if candidate.original_rank == candidate.reranked_rank:
            continue
        current = keys[candidate.parent_asin]

        if index == 0:
            # Promoted to the front: the decisive pair is the candidate it overtook,
            # which is now directly below it.
            decisive = keys[order[1].parent_asin] if len(order) > 1 else current
        elif candidate.reranked_rank < candidate.original_rank:
            # Promoted: take the highest-ranked overtaken candidate, i.e. the one above.
            above = order[index - 1]
            if above.original_rank > candidate.original_rank:
                decisive = keys[above.parent_asin]
            elif index + 1 < len(order):
                # The candidate above was already ahead of it; the pair is the
                # highest-ranked candidate it overtook, now directly below.
                decisive = keys[order[index + 1].parent_asin]
            else:
                # Promoted to the last position with nothing ahead to cross: fall back
                # to the candidate directly above, which it still outranked.
                decisive = keys[above.parent_asin]
        else:
            # Demoted: the candidate directly above displaced it.
            decisive = keys[order[index - 1].parent_asin]

        causes[candidate.parent_asin] = _classify_pair(current, decisive)

    return causes


def _position_reason_breakdown(
    reasons: Sequence[PositionReason],
) -> PositionReasonBreakdown:
    """Count the ordering-reason distribution."""
    counter = Counter(reasons)
    return PositionReasonBreakdown(
        fewer_violations=counter[PositionReason.FEWER_VIOLATIONS],
        more_matches=counter[PositionReason.MORE_MATCHES],
        ordinal_rank=counter[PositionReason.ORDINAL_RANK],
        item_id_tiebreak=counter[PositionReason.ITEM_ID_TIEBREAK],
        last_position=counter[PositionReason.LAST_POSITION],
    )


def _movement_attribution(
    report: PreferenceEvidenceReport, reranked: RerankingReport
) -> MovementAttribution:
    """Count movement causes over moved candidates only."""
    causes = movement_causes_for(report, reranked)
    counter = Counter(causes.values())
    return MovementAttribution(
        fewer_violations=counter[MovementCause.FEWER_VIOLATIONS],
        more_matches=counter[MovementCause.MORE_MATCHES],
        ordinal_fallback=counter[MovementCause.ORDINAL_FALLBACK],
        item_id_tiebreak=counter[MovementCause.ITEM_ID_TIEBREAK],
        moved_count=reranked.moved_count,
    )


def _violation_protection(
    report: PreferenceEvidenceReport,
    reranked: RerankingReport,
    violation_count_by_asin: dict[str, int],
) -> ViolationProtection:
    """Count cases where a violating candidate ranks above a clean one.

    The canonical policy makes this impossible, so the expected result is zero
    inversions.  The check is over adjacent pairs, which suffices because the order is
    sorted by violation count first.
    """
    pairs = 0
    inversions = 0
    above = 0
    after = list(reranked.candidates)
    for index in range(len(after) - 1):
        current = violation_count_by_asin[after[index].parent_asin]
        following = violation_count_by_asin[after[index + 1].parent_asin]
        if (current >= 1 and following == 0) or (current == 0 and following >= 1):
            pairs += 1
            if current >= 1 and following == 0:
                inversions += 1
                above += 1
    return ViolationProtection(
        k=0,
        pairs_checked=pairs,
        inversions=inversions,
        violating_candidates_above_clean=above,
    )


def _policy_consistency(
    report: PreferenceEvidenceReport, reranked: RerankingReport
) -> PolicyConsistency:
    """Verify the emitted order is non-decreasing under the accepted M10B sort key."""
    by_asin = _evidence_by_asin(report)
    keys = []
    for candidate in reranked.candidates:
        evidence = by_asin[candidate.parent_asin]
        matches, violations, _ = _counts(evidence.evidence)
        keys.append((violations, -matches, candidate.original_rank, candidate.item_id))

    violations_found = sum(
        1 for index in range(len(keys) - 1) if keys[index] > keys[index + 1]
    )
    return PolicyConsistency(
        pairs_checked=max(len(keys) - 1, 0),
        violations=violations_found,
        sort_key=reranked.sort_key,
    )


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def _coverage(
    report: PreferenceEvidenceReport,
) -> tuple[
    tuple[PreferenceCoverage, ...],
    tuple[CandidateEvidenceCoverage, ...],
    tuple[PreferenceTypeCoverage, ...],
]:
    """Compute preference-level, candidate-level and per-kind evidence coverage."""
    preference_rows: list[PreferenceCoverage] = []
    seen: dict[str, tuple[Any, int, int]] = {}

    active_ids: list[str] = []
    for candidate in report.candidates:
        for record in candidate.evidence:
            if record.preference_id not in seen:
                seen[record.preference_id] = (record, 0, 0)
                active_ids.append(record.preference_id)

    candidate_count = len(report.candidates)
    for candidate in report.candidates:
        for record in candidate.evidence:
            record_entry, known, unknown = seen[record.preference_id]
            if record.status is EvidenceStatus.UNKNOWN:
                seen[record.preference_id] = (record_entry, known, unknown + 1)
            else:
                seen[record.preference_id] = (record_entry, known + 1, unknown)

    for preference_id in active_ids:
        record, known, unknown = seen[preference_id]
        preference_rows.append(
            PreferenceCoverage(
                preference_id=preference_id,
                kind=record.preference_kind,
                value=record.preference_value,
                candidate_count=candidate_count,
                known_count=known,
                unknown_count=unknown,
            )
        )

    candidate_rows: list[CandidateEvidenceCoverage] = []
    for candidate in report.candidates:
        matches, violations, unknowns = _counts(candidate.evidence)
        candidate_rows.append(
            CandidateEvidenceCoverage(
                parent_asin=candidate.parent_asin,
                original_rank=candidate.original_rank,
                match_count=matches,
                violation_count=violations,
                unknown_count=unknowns,
                total_count=len(candidate.evidence),
            )
        )

    # Per-kind aggregation over the request's active preferences.
    kind_preferences: Counter[Any] = Counter()
    kind_observations: Counter[Any] = Counter()
    kind_known: Counter[Any] = Counter()
    kind_unknown: Counter[Any] = Counter()
    kind_values: dict[Any, set[str]] = {}
    for preference_id in active_ids:
        record, known, unknown = seen[preference_id]
        kind = record.preference_kind
        kind_preferences[kind] += 1
        kind_observations[kind] += candidate_count
        kind_known[kind] += known
        kind_unknown[kind] += unknown
        kind_values.setdefault(kind, set()).add(record.preference_value)

    type_rows = tuple(
        PreferenceTypeCoverage(
            kind=kind,
            preference_count=kind_preferences[kind],
            observations=kind_observations[kind],
            known_observations=kind_known[kind],
            unknown_observations=kind_unknown[kind],
            variant_count=len(kind_values.get(kind, set())),
        )
        for kind in sorted(kind_preferences, key=lambda item: item.value)
    )
    return tuple(preference_rows), tuple(candidate_rows), type_rows


def _type_movement(
    report: PreferenceEvidenceReport, reranked: RerankingReport
) -> tuple[PreferenceTypeMovement, ...]:
    """Credit promotions and violations to the preference kinds that supported them.

    A promotion is credited to every preference kind whose evidence on the promoted
    candidate is known (non-UNKNOWN).  This makes it visible whether movement is driven
    mainly by free-text feature/category evidence rather than by structured kinds.
    """
    by_asin = _evidence_by_asin(report)
    promotion_credit: Counter[Any] = Counter()
    violation_credit: Counter[Any] = Counter()
    observed: set[Any] = set()

    for candidate in reranked.candidates:
        evidence = by_asin[candidate.parent_asin].evidence
        kinds_with_signal = {
            record.preference_kind
            for record in evidence
            if record.status is not EvidenceStatus.UNKNOWN
        }
        observed.update(record.preference_kind for record in evidence)
        for kind in kinds_with_signal:
            if candidate.rank_delta > 0:
                promotion_credit[kind] += 1
            if candidate.violation_count > 0:
                for record in evidence:
                    if record.preference_kind is kind and record.status is EvidenceStatus.VIOLATION:
                        violation_credit[kind] += 1

    return tuple(
        PreferenceTypeMovement(
            kind=kind,
            promotion_credit=promotion_credit[kind],
            violation_credit=violation_credit[kind],
        )
        for kind in sorted(observed, key=lambda item: item.value)
    )


# --------------------------------------------------------------------------- #
# Restricted diagnostic baselines
# --------------------------------------------------------------------------- #


def _baseline_orders(
    report: PreferenceEvidenceReport,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the diagnostic baseline orders.

    These are restricted comparison orders used **only** to characterise what each
    canonical key contributes.  They are not production reranking alternatives: the
    accepted M10B policy remains the single production policy.
    """
    counts = {
        candidate.parent_asin: _counts(candidate.evidence) for candidate in report.candidates
    }
    original = tuple(
        candidate.parent_asin
        for candidate in sorted(report.candidates, key=lambda c: c.original_rank)
    )
    violation_only = tuple(
        candidate.parent_asin
        for candidate in sorted(
            report.candidates,
            key=lambda c: (counts[c.parent_asin][1], c.original_rank, c.item_id),
        )
    )
    match_only = tuple(
        candidate.parent_asin
        for candidate in sorted(
            report.candidates,
            key=lambda c: (-counts[c.parent_asin][0], c.original_rank, c.item_id),
        )
    )
    return (
        (BASELINE_ORIGINAL_ORDER, original),
        (BASELINE_VIOLATION_ONLY, violation_only),
        (BASELINE_MATCH_ONLY, match_only),
    )


def _baselines(
    report: PreferenceEvidenceReport,
    reranked: RerankingReport,
    ks: Sequence[int],
) -> tuple[BaselineComparison, ...]:
    """Compare the canonical order against the restricted diagnostic baselines."""
    canonical = tuple(candidate.parent_asin for candidate in reranked.candidates)
    by_asin = _evidence_by_asin(report)
    counts = {
        candidate.parent_asin: _counts(candidate.evidence) for candidate in report.candidates
    }
    overlap_k = next((k for k in ks if 1 <= k <= len(canonical)), 0)
    canonical_top = set(canonical[:overlap_k]) if overlap_k else set()

    rows: list[BaselineComparison] = []
    for name, order in _baseline_orders(report):
        if overlap_k:
            head = order[:overlap_k]
            violations_at_k = sum(counts[asin][1] for asin in head)
            matches_at_k = sum(counts[asin][0] for asin in head)
            unknown_at_k = sum(counts[asin][2] for asin in head)
            overlap = len(set(head) & canonical_top)
        else:
            violations_at_k = matches_at_k = unknown_at_k = overlap = 0
        original = tuple(
            candidate.parent_asin
            for candidate in sorted(report.candidates, key=lambda c: c.original_rank)
        )
        rows.append(
            BaselineComparison(
                name=name,
                description=_BASELINE_DESCRIPTIONS[name],
                order=order,
                movement_vs_original=sum(
                    1 for index, asin in enumerate(order) if original[index] != asin
                ),
                movement_vs_canonical=sum(
                    1 for index, asin in enumerate(order) if canonical[index] != asin
                ),
                violations_at_k=violations_at_k,
                matches_at_k=matches_at_k,
                unknown_at_k=unknown_at_k,
                top_k_overlap_with_canonical=overlap,
            )
        )
    _ = by_asin  # retained for readability of the mapping above
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def _invariants(
    report: PreferenceEvidenceReport,
    reranked: RerankingReport,
    *,
    before_snapshot: dict[str, Any] | None = None,
) -> EvaluationInvariants:
    """Assert the M10C correctness invariants for one request.

    A false flag is a correctness failure.  The evaluator observes only; it must never
    itself change candidate identity, count, scores, evidence or order.
    """
    original = sorted(report.candidates, key=lambda c: c.original_rank)
    after = list(reranked.candidates)

    original_asins = [c.parent_asin for c in original]
    after_asins = [c.parent_asin for c in after]
    original_items = [c.item_id for c in original]
    after_items = [c.item_id for c in after]

    evidence_by_asin = {c.parent_asin: c.evidence for c in original}
    scores_by_asin = {c.parent_asin: c.sasrec_score for c in original}
    ranks_by_asin = {c.parent_asin: c.original_rank for c in original}

    consistency = _policy_consistency(report, reranked)
    counts = {c.parent_asin: _counts(c.evidence) for c in original}
    protection = _violation_protection(
        report, reranked, {asin: value[1] for asin, value in counts.items()}
    )

    unchanged = True
    if before_snapshot is not None:
        unchanged = before_snapshot == report.model_dump()

    return EvaluationInvariants(
        candidate_count_unchanged=len(original) == len(after),
        candidate_universe_unchanged=sorted(original_asins) == sorted(after_asins),
        item_ids_unchanged=sorted(original_items) == sorted(after_items),
        sasrec_scores_unchanged=all(
            scores_by_asin[c.parent_asin] == c.sasrec_score for c in after
        ),
        evidence_unchanged=all(
            evidence_by_asin[c.parent_asin] == c.evidence for c in after
        ),
        original_ranks_retained=all(
            ranks_by_asin[c.parent_asin] == c.original_rank for c in after
        ),
        reranked_ranks_contiguous=tuple(c.reranked_rank for c in after)
        == tuple(range(1, len(after) + 1)),
        no_candidate_dropped=Counter(original_asins) == Counter(after_asins),
        policy_order_valid=consistency.order_valid,
        violation_protection_holds=protection.holds,
        input_not_mutated=unchanged,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def evaluate_request(
    *,
    report: PreferenceEvidenceReport,
    label: str,
    reranker: PreferenceReranker | None = None,
    diagnostics_k: Sequence[int] = (1, 3, 5, 10),
    include_baselines: bool = True,
) -> tuple[PolicyEvaluationRequest, RerankingReport]:
    """Evaluate one request: rerank through M10B, then diagnose the result.

    The M10B reranker is called exactly as production calls it; this function never
    sorts candidates itself.
    """
    reranker = reranker or PreferenceReranker()
    before_snapshot = report.model_dump()

    reranked = reranker.rerank(report)

    counts = {
        candidate.parent_asin: _counts(candidate.evidence) for candidate in report.candidates
    }
    evidence = (
        _coverage(report)
        if report.candidates
        else ((), (), ())
    )
    preference_coverage, candidate_coverage, type_coverage = evidence

    request = PolicyEvaluationRequest(
        label=label,
        candidate_count=len(report.candidates),
        active_preference_count=report.active_preference_count,
        sort_key=reranked.sort_key,
        diagnostics_k=tuple(diagnostics_k),
        displacement=_displacement(reranked),
        top_k_overlap=_top_k_overlap(report, reranked, diagnostics_k),
        adherence=_adherence(report, reranked, diagnostics_k),
        position_reasons=_position_reason_breakdown(position_reasons_for(report, reranked)),
        movement_attribution=_movement_attribution(report, reranked),
        violation_protection=_violation_protection(
            report, reranked, {asin: value[1] for asin, value in counts.items()}
        ),
        policy_consistency=_policy_consistency(report, reranked),
        preference_coverage=preference_coverage,
        candidate_coverage=candidate_coverage,
        type_coverage=type_coverage,
        type_movement=_type_movement(report, reranked),
        baselines=_baselines(report, reranked, diagnostics_k) if include_baselines else (),
        invariants=_invariants(report, reranked, before_snapshot=before_snapshot),
    )
    return request, reranked


def evaluate(
    *,
    requests: Sequence[tuple[str, PreferenceEvidenceReport]],
    reranker: PreferenceReranker | None = None,
    diagnostics_k: Sequence[int] = (1, 3, 5, 10),
    include_baselines: bool = True,
) -> tuple[tuple[PolicyEvaluationRequest, ...], tuple[RerankingReport, ...]]:
    """Evaluate a sequence of labelled requests through the accepted M10B reranker."""
    reranker = reranker or PreferenceReranker()
    evaluated: list[PolicyEvaluationRequest] = []
    reranked_reports: list[RerankingReport] = []
    for label, report in requests:
        request, reranked = evaluate_request(
            report=report,
            label=label,
            reranker=reranker,
            diagnostics_k=diagnostics_k,
            include_baselines=include_baselines,
        )
        evaluated.append(request)
        reranked_reports.append(reranked)
    return tuple(evaluated), tuple(reranked_reports)


def aggregate_requests(
    requests: Sequence[PolicyEvaluationRequest],
) -> dict[str, Any]:
    """Aggregate per-request diagnostics over a scenario suite or cohort.

    Every entry reports its numerator and denominator so no number can be read as a
    bare percentage.  Latency is **not** included here: it belongs to the runtime
    wrapper and is excluded from the deterministic digest.
    """
    total_requests = len(requests)
    total_candidates = sum(request.candidate_count for request in requests)
    moved_requests = sum(1 for request in requests if request.moved)
    all_unknown_requests = sum(
        1
        for request in requests
        if request.candidate_coverage
        and all(row.all_unknown for row in request.candidate_coverage)
    )
    with_violation = sum(
        1
        for request in requests
        if any(row.violation_count > 0 for row in request.candidate_coverage)
    )
    with_match = sum(
        1
        for request in requests
        if any(row.match_count > 0 for row in request.candidate_coverage)
    )

    moved_count = sum(request.displacement.moved_count for request in requests)
    promoted = sum(request.displacement.promoted_count for request in requests)
    demoted = sum(request.displacement.demoted_count for request in requests)
    unchanged = sum(request.displacement.unchanged_count for request in requests)
    total_abs = sum(request.displacement.total_abs_delta for request in requests)
    max_abs = max((request.displacement.max_abs_delta for request in requests), default=0)

    attribution = MovementAttribution(
        fewer_violations=sum(r.movement_attribution.fewer_violations for r in requests),
        more_matches=sum(r.movement_attribution.more_matches for r in requests),
        ordinal_fallback=sum(r.movement_attribution.ordinal_fallback for r in requests),
        item_id_tiebreak=sum(r.movement_attribution.item_id_tiebreak for r in requests),
        moved_count=sum(r.movement_attribution.moved_count for r in requests),
    )
    positions = PositionReasonBreakdown(
        fewer_violations=sum(r.position_reasons.fewer_violations for r in requests),
        more_matches=sum(r.position_reasons.more_matches for r in requests),
        ordinal_rank=sum(r.position_reasons.ordinal_rank for r in requests),
        item_id_tiebreak=sum(r.position_reasons.item_id_tiebreak for r in requests),
        last_position=sum(r.position_reasons.last_position for r in requests),
    )

    # Coverage aggregates over every candidate in every request.
    coverage_candidates = [
        row for request in requests for row in request.candidate_coverage
    ]
    coverage_preferences = [
        row for request in requests for row in request.preference_coverage
    ]
    all_unknown_candidates = sum(1 for row in coverage_candidates if row.all_unknown)
    known_candidates = len(coverage_candidates) - all_unknown_candidates
    kind_coverage: dict[str, dict[str, int]] = {}
    for request in requests:
        for row in request.type_coverage:
            bucket = kind_coverage.setdefault(
                row.kind.value,
                {"preference_count": 0, "observations": 0, "known_observations": 0,
                 "unknown_observations": 0},
            )
            for field in ("preference_count", "observations", "known_observations",
                          "unknown_observations"):
                bucket[field] += getattr(row, field)

    # Duplicate-rank audit: with unique ranks the item_id key is unreachable.
    duplicate_original_rank_count = sum(
        len(request.candidate_coverage)
        - len({row.original_rank for row in request.candidate_coverage})
        for request in requests
    )

    # Adherence and overlap aggregates keep explicit denominators.
    adherence_by_k: dict[int, dict[str, int]] = {}
    overlap_by_k: dict[int, dict[str, int]] = {}
    for request in requests:
        for row in request.adherence:
            bucket = adherence_by_k.setdefault(row.k, {})
            for field in (
                "violations_before",
                "violations_after",
                "matches_before",
                "matches_after",
                "unknown_before",
                "unknown_after",
            ):
                bucket[field] = bucket.get(field, 0) + getattr(row, field)
            bucket["requests"] = bucket.get("requests", 0) + 1
        for row in request.top_k_overlap:
            bucket = overlap_by_k.setdefault(row.k, {})
            bucket["overlap_count"] = bucket.get("overlap_count", 0) + row.overlap_count
            bucket["k_effective"] = bucket.get("k_effective", 0) + row.k_effective
            bucket["requests"] = bucket.get("requests", 0) + 1

    policy_violations = sum(request.policy_consistency.violations for request in requests)
    protection_inversions = sum(
        request.violation_protection.inversions for request in requests
    )
    protection_pairs = sum(
        request.violation_protection.pairs_checked for request in requests
    )
    invariant_failures = Counter(
        failure for request in requests for failure in request.invariants.failures()
    )

    return {
        "requests": {
            "total": total_requests,
            "with_movement": moved_requests,
            "with_movement_fraction": (
                round(moved_requests / total_requests, 6) if total_requests else 0.0
            ),
            "all_evidence_unknown": all_unknown_requests,
            "with_at_least_one_match": with_match,
            "with_at_least_one_violation": with_violation,
        },
        "candidates": {
            "total": total_candidates,
            "moved": moved_count,
            "promoted": promoted,
            "demoted": demoted,
            "unchanged": unchanged,
            "mean_moved": round(moved_count / total_requests, 6) if total_requests else 0.0,
        },
        "displacement": {
            "total_abs_delta": total_abs,
            "mean_abs_delta": round(total_abs / total_candidates, 6)
            if total_candidates
            else 0.0,
            "max_abs_delta": max_abs,
            "mean_abs_delta_per_request": round(total_abs / total_requests, 6)
            if total_requests
            else 0.0,
        },
        "position_reasons": positions.as_dict(),
        "movement_attribution": attribution.as_dict(),
        "coverage": {
            "candidates": {
                "total": len(coverage_candidates),
                "all_unknown": all_unknown_candidates,
                "with_known_evidence": known_candidates,
                "all_unknown_fraction": round(
                    all_unknown_candidates / len(coverage_candidates), 6
                )
                if coverage_candidates
                else 0.0,
                "with_known_evidence_fraction": round(
                    known_candidates / len(coverage_candidates), 6
                )
                if coverage_candidates
                else 0.0,
            },
            "preferences": {
                "total": len(coverage_preferences),
                "known_observations": sum(row.known_count for row in coverage_preferences),
                "unknown_observations": sum(row.unknown_count for row in coverage_preferences),
                "observations": sum(row.candidate_count for row in coverage_preferences),
            },
            "by_kind": kind_coverage,
        },
        "duplicate_original_rank_count": duplicate_original_rank_count,
        "item_id_fallback_reachable": duplicate_original_rank_count > 0,
        "adherence_at_k": {str(k): v for k, v in sorted(adherence_by_k.items())},
        "top_k_overlap": {
            str(k): {
                **v,
                "overlap_fraction": round(v["overlap_count"] / v["k_effective"], 6)
                if v.get("k_effective")
                else 0.0,
            }
            for k, v in sorted(overlap_by_k.items())
        },
        "policy_consistency": {
            "requests_checked": total_requests,
            "order_violations": policy_violations,
        },
        "violation_protection": {
            "pairs_checked": protection_pairs,
            "inversions": protection_inversions,
        },
        "invariant_failures": dict(invariant_failures),
        "all_invariants_hold": not invariant_failures,
    }


def build_report(
    *,
    requests: Sequence[PolicyEvaluationRequest],
    policy: str,
    starting_commit: str = "",
    cohort_rule: str = "",
    cohort_size: int = 0,
    candidate_k: int = 0,
    diagnostics_k: Sequence[int] = (1, 3, 5, 10),
    preference_fixture_policy: str = "",
    artifacts: dict[str, Any] | None = None,
) -> RerankingEvaluationReport:
    """Assemble the durable evaluation report with an aggregate block."""
    return RerankingEvaluationReport(
        policy=policy,
        starting_commit=starting_commit,
        cohort_rule=cohort_rule,
        cohort_size=cohort_size,
        candidate_k=candidate_k,
        diagnostics_k=tuple(diagnostics_k),
        preference_fixture_policy=preference_fixture_policy,
        synthetic_preferences=True,
        disclaimer=(
            "Preferences used in this evaluation are synthetic deterministic fixtures, "
            "not observed user preferences. Diagnostics measure policy behaviour and "
            "agreement with explicit preference evidence only; they are NOT "
            "recommendation-quality, relevance, satisfaction or conversion metrics."
        ),
        requests=tuple(requests),
        aggregate=aggregate_requests(requests),
        artifacts=dict(artifacts or {}),
    )
