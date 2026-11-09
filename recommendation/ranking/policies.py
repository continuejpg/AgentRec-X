"""M6.1 preregistered ranking policies A0 / A1 / A2 (offline lab only).

The three arms rank **exactly the same** candidate pool: the candidates of one
:class:`~recommendation.ranking.schemas.CandidateFeatureSet` after one shared
hard-constraint eligibility projection.  No arm retrieves, filters, scores or adds a
candidate, and no arm reads a raw model/RRF score numerically.

Arms (fixed by ``docs/M6_1_RANKING_PROTOCOL.md`` section 4)
----------------------------------------------------------
* **A0** -- the trusted original order (``original_rank ASC, item_id ASC``);
* **A1** -- the accepted M10A evidence report reordered by the **real, unmodified**
  M10B :func:`recommendation.reranking.reranker.rerank_candidates` policy
  ``(violations ASC, matches DESC, original_rank ASC, item_id ASC)``;
* **A2** -- the A1 explicit-preference keys first, then a bounded *ordinal* inferred
  belief tie-break, then the A1 deterministic tail::

      (explicit_violations ASC,
       -explicit_matches ASC,
       inferred_ordinal_penalty ASC,
       -inferred_ordinal_support ASC,
       original_rank ASC,
       item_id ASC)

Shared eligibility projection
-----------------------------
An eligibility assessment is applied exactly once, before any arm:

* if no candidate was assessed (``NOT_ASSESSED`` everywhere), the full candidate set is
  the ranking pool and this is recorded in the projection;
* otherwise only ``VERIFIED_ELIGIBLE`` candidates are presentable.
  ``INELIGIBLE`` **and** ``UNRESOLVED`` candidates are excluded from all arms in this
  experiment: an unresolvable constraint is not evidence of compliance, so an unresolved
  candidate is not presentable.  This is the conservative reading of the preregistered
  rule ("only VERIFIED_ELIGIBLE candidates enter the ranking pool") and it is asserted by
  tests.

No score mixing
---------------
``rrf_score`` / source ranks / source scores / the snapshot ``score`` are **provenance
only**.  A2 uses ordinal evidence counts and deterministic tie-breaks; changing any raw
score field cannot change any arm's order.  This is asserted by tests.

Determinism
-----------
Every arm sorts by an explicit lexicographic key whose final component is ``item_id``
(unique per snapshot), so the ordering is total and does not depend on input order,
wall-clock time or hash iteration.  Feeding the same features twice yields byte-identical
arm rankings and equal candidate-set digests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from recommendation.preference_matching.schemas import (
    CandidateEvidenceCounts,
    CandidatePreferenceEvidence,
    EvidenceStatus,
    PreferenceEvidenceReport,
    ReasonCode,
)
from recommendation.reranking.reranker import rerank_candidates

from .errors import RankingPolicyError
from .schemas import (
    ArmRanking,
    CandidateFeatureSet,
    CandidateRankingFeatures,
    CandidateSnapshot,
    EligibilityState,
    RankedCandidate,
    RankingArm,
)

__all__ = [
    "POLICY_DESCRIPTIONS",
    "POLICY_DESCRIPTION_A0",
    "POLICY_DESCRIPTION_A1",
    "POLICY_DESCRIPTION_A2",
    "POLICY_VERSION",
    "PREREGISTERED_ARMS",
    "REASON_CATEGORY_NUMERIC_ORDER",
    "REASON_EXPLICIT_PREFERENCE_ORDER",
    "REASON_INFERRED_ORDINAL_TIEBREAK",
    "REASON_ORIGINAL_ORDER",
    "REASON_ORIGINAL_RANK_TIEBREAK",
    "EligibilityProjection",
    "rank_all_arms",
    "rank_arm",
    "shared_eligibility_projection",
]

#: Bumped when the category-aware explicit-numeric overlay changes ranking semantics.
POLICY_VERSION = 2

#: Machine-readable placement reasons.  ``RankedCandidate.reason`` carries exactly one.
REASON_ORIGINAL_ORDER = "original_order"
REASON_EXPLICIT_PREFERENCE_ORDER = "explicit_preference_order"
REASON_INFERRED_ORDINAL_TIEBREAK = "inferred_ordinal_tiebreak"
REASON_ORIGINAL_RANK_TIEBREAK = "original_rank_tiebreak"
REASON_CATEGORY_NUMERIC_ORDER = "category_numeric_order"

_NUMERIC_REASON_CODES = frozenset(
    {
        ReasonCode.NUMERIC_PREFERRED_DIRECTION,
        ReasonCode.NUMERIC_WORSE_DIRECTION,
        ReasonCode.NUMERIC_NOT_COMPARABLE,
    }
)


def _textual_explicit_evidence(feature: CandidateRankingFeatures) -> tuple[Any, ...]:
    """Explicit evidence allowed into the accepted global M10B key.

    Numeric directional evidence is excluded from global counts; it is applied only by
    the category-slot overlay below.  A global count would let category size move a
    category ahead of unrelated categories.
    """
    return tuple(
        evidence
        for evidence in feature.explicit_evidence
        if evidence.reason_code not in _NUMERIC_REASON_CODES
    )


def _textual_explicit_key(feature: CandidateRankingFeatures) -> tuple[int, int]:
    evidence = _textual_explicit_evidence(feature)
    return (
        sum(1 for item in evidence if item.status is EvidenceStatus.VIOLATION),
        -sum(1 for item in evidence if item.status is EvidenceStatus.MATCH),
    )


def _numeric_status_counts(feature: CandidateRankingFeatures) -> tuple[int, int]:
    numeric = [
        evidence
        for evidence in feature.explicit_evidence
        if evidence.reason_code in _NUMERIC_REASON_CODES
        and evidence.status is not EvidenceStatus.UNKNOWN
    ]
    return (
        sum(1 for item in numeric if item.status is EvidenceStatus.VIOLATION),
        sum(1 for item in numeric if item.status is EvidenceStatus.MATCH),
    )


def _category_path(
    snapshot: CandidateSnapshot, candidate_id: str
) -> tuple[str, ...] | None:
    """Trusted full category path for one candidate, or ``None`` when not recorded."""
    for candidate in snapshot.candidates:
        if candidate.candidate_id != candidate_id:
            continue
        record = candidate.metadata
        if record is None:
            return None
        path = tuple(
            str(item).strip() for item in record.categories if str(item).strip()
        )
        return path or None
    return None


def _category_numeric_overlay(
    ordered: Sequence[CandidateRankingFeatures], snapshot: CandidateSnapshot
) -> list[CandidateRankingFeatures]:
    """Sort numeric explicit evidence only inside each trusted category's own slots.

    Category positions are preserved exactly, so same-category candidates can reorder
    while no category crosses another.  Candidates with UNKNOWN numeric evidence keep
    their slots (no reward, no penalty).  The numeric key is the accepted M10B key
    restricted to the pairwise numeric records.
    """
    result = list(ordered)
    groups: dict[tuple[str, ...], list[int]] = {}
    for position, feature in enumerate(result):
        violations, matches = _numeric_status_counts(feature)
        if violations == 0 and matches == 0:
            continue
        path = _category_path(snapshot, feature.candidate_id)
        if path is None:
            continue
        groups.setdefault(path, []).append(position)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        members = [result[position] for position in positions]
        members.sort(
            key=lambda feature: (
                _numeric_status_counts(feature)[0],
                -_numeric_status_counts(feature)[1],
                feature.original_rank,
                feature.item_id,
            )
        )
        for position, feature in zip(positions, members):
            result[position] = feature
    return result

#: Human-auditable policy descriptions recorded on every arm ranking.
POLICY_DESCRIPTION_A0 = (
    "a0_original_order: trusted snapshot order (original_rank ASC, item_id ASC)"
)
POLICY_DESCRIPTION_A1 = (
    "a1_explicit_rerank: unmodified M10B "
    "(explicit_violations ASC, explicit_matches DESC, original_rank ASC, item_id ASC)"
)
POLICY_DESCRIPTION_A2 = (
    "a2_belief_tiebreak: M10B explicit key first "
    "(explicit_violations ASC, -explicit_matches ASC), then ordinal inferred tie-break "
    "(inferred_ordinal_penalty ASC, -inferred_ordinal_support ASC), then "
    "original_rank ASC, item_id ASC"
)

POLICY_DESCRIPTIONS: dict[RankingArm, str] = {
    RankingArm.A0_ORIGINAL: POLICY_DESCRIPTION_A0,
    RankingArm.A1_EXPLICIT_RERANK: POLICY_DESCRIPTION_A1,
    RankingArm.A2_BELIEF_TIEBREAK: POLICY_DESCRIPTION_A2,
}

#: The preregistered arm order.  ``rank_all_arms`` always returns exactly this tuple.
PREREGISTERED_ARMS: tuple[RankingArm, ...] = (
    RankingArm.A0_ORIGINAL,
    RankingArm.A1_EXPLICIT_RERANK,
    RankingArm.A2_BELIEF_TIEBREAK,
)


@dataclass(frozen=True)
class EligibilityProjection:
    """The single shared eligibility decision applied before every arm.

    ``assessed`` records whether a trusted assessment existed at all.  When it did not,
    ``eligible_ids`` is the full candidate set and that fact is preserved here (and in the
    ``reason`` string) instead of being silently assumed.
    """

    assessed: bool
    eligible_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    eligible_count: int
    excluded_count: int
    reason: str


def _as_arm(arm: RankingArm | str) -> RankingArm:
    """Coerce ``arm`` to a :class:`RankingArm`, refusing anything outside the contract."""
    if isinstance(arm, RankingArm):
        return arm
    try:
        return RankingArm(str(arm))
    except ValueError as exc:
        raise RankingPolicyError(f"unknown ranking arm: {arm!r}") from exc


def _validate_inputs(
    feature_set: CandidateFeatureSet, snapshot: CandidateSnapshot
) -> None:
    """Validate that one feature set describes exactly one frozen snapshot.

    A policy may reorder the frozen candidates; it may never rank a candidate the snapshot
    does not contain, and it may never be handed a feature set for a different task.  A
    mismatch is a policy error, not something to repair silently.
    """
    if not isinstance(feature_set, CandidateFeatureSet):
        raise RankingPolicyError("feature_set must be a CandidateFeatureSet")
    if not isinstance(snapshot, CandidateSnapshot):
        raise RankingPolicyError("snapshot must be a CandidateSnapshot")
    if feature_set.task_id != snapshot.task_id:
        raise RankingPolicyError("feature set and snapshot describe different tasks")
    if feature_set.user_key != snapshot.user_key:
        raise RankingPolicyError("feature set and snapshot describe different users")
    if feature_set.turn_key != snapshot.turn_key:
        raise RankingPolicyError("feature set and snapshot describe different turns")
    if feature_set.snapshot_id != snapshot.snapshot_id:
        raise RankingPolicyError("feature set and snapshot have different snapshot ids")
    if feature_set.snapshot_candidate_count != snapshot.candidate_count:
        raise RankingPolicyError(
            "feature set candidate count does not equal the snapshot candidate count"
        )
    snapshot_by_id = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    feature_ids = [feature.candidate_id for feature in feature_set.features]
    if len(set(feature_ids)) != len(feature_ids):
        raise RankingPolicyError("feature set contains duplicate candidate identities")
    if set(feature_ids) != set(snapshot_by_id):
        raise RankingPolicyError(
            "feature set candidate identities do not match the frozen snapshot"
        )
    for feature in feature_set.features:
        record = snapshot_by_id[feature.candidate_id]
        if feature.item_id != record.item_id:
            raise RankingPolicyError(
                f"feature item_id for {feature.candidate_id!r} does not match the snapshot"
            )
        if feature.original_rank != record.original_rank:
            raise RankingPolicyError(
                f"feature original_rank for {feature.candidate_id!r} does not match the "
                "snapshot"
            )
        if feature.eligibility is not record.eligibility:
            raise RankingPolicyError(
                f"feature eligibility for {feature.candidate_id!r} does not match the "
                "snapshot assessment"
            )


def shared_eligibility_projection(
    feature_set: CandidateFeatureSet,
) -> EligibilityProjection:
    """Return the one eligibility projection every arm ranks over.

    * no candidate assessed -> the full candidate set, recorded as unassessed;
    * any assessment present -> only ``VERIFIED_ELIGIBLE`` candidates.

    Candidates are returned in trusted original order (``original_rank``, then
    ``item_id``) so every arm starts from the same deterministic pool.
    """
    if not isinstance(feature_set, CandidateFeatureSet):
        raise RankingPolicyError("feature_set must be a CandidateFeatureSet")
    ordered = sorted(
        feature_set.features, key=lambda feature: (feature.original_rank, feature.item_id)
    )
    assessed = any(feature.eligibility.was_assessed for feature in ordered)
    if not assessed:
        eligible = tuple(ordered)
        reason = (
            "no eligibility assessment exists (NOT_ASSESSED everywhere); the full "
            "candidate set is the ranking pool"
        )
    else:
        eligible = tuple(
            feature
            for feature in ordered
            if feature.eligibility is EligibilityState.VERIFIED_ELIGIBLE
        )
        reason = (
            "hard-constraint eligibility applied once before every arm; only "
            "VERIFIED_ELIGIBLE candidates are presentable (INELIGIBLE and UNRESOLVED are "
            "excluded in this experiment)"
        )
    eligible_ids = tuple(feature.candidate_id for feature in eligible)
    eligible_id_set = set(eligible_ids)
    excluded_ids = tuple(
        feature.candidate_id
        for feature in ordered
        if feature.candidate_id not in eligible_id_set
    )
    return EligibilityProjection(
        assessed=assessed,
        eligible_ids=eligible_ids,
        excluded_ids=excluded_ids,
        eligible_count=len(eligible_ids),
        excluded_count=len(excluded_ids),
        reason=reason,
    )


def _candidate_set_digest(identities: Sequence[str]) -> str:
    """Order-insensitive SHA-256 digest of the ranked candidate identity set.

    Reuses the M5 candidate-pool digest algorithm so a report can be joined to a state
    candidate-pool reference without storing product rows.  The same candidate set always
    produces the same digest regardless of arm order.
    """
    from recommendation.state.shopping_adapter import candidate_pool_snapshot

    return candidate_pool_snapshot(identities).digest


def _ranked_candidate(
    *,
    feature: CandidateRankingFeatures,
    rank: int,
    reason: str,
) -> RankedCandidate:
    """Build one :class:`RankedCandidate` by copying, never rescoring, the features."""
    return RankedCandidate(
        candidate_id=feature.candidate_id,
        item_id=feature.item_id,
        rank=rank,
        original_rank=feature.original_rank,
        explicit_violations=feature.explicit_violations,
        explicit_matches=feature.explicit_matches,
        inferred_violations=feature.inferred_violations,
        inferred_matches=feature.inferred_matches,
        eligibility=feature.eligibility,
        reason=reason,
    )


def _a1_report(
    features: Sequence[CandidateRankingFeatures],
    snapshot: CandidateSnapshot,
) -> PreferenceEvidenceReport:
    """Build the accepted M10A evidence report for the eligible pool only.

    Evidence is passed through verbatim from the feature records (already the accepted
    M10A records); the raw ordering score is copied from the trusted snapshot record
    because the report shape carries it.  Nothing is invented and nothing is re-matched.
    """
    records = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    candidates: list[CandidatePreferenceEvidence] = []
    for feature in sorted(features, key=lambda item: (item.original_rank, item.item_id)):
        record = records.get(feature.candidate_id)
        if record is None:  # pragma: no cover - validated by _validate_inputs
            raise RankingPolicyError(
                f"candidate {feature.candidate_id!r} is not in the frozen snapshot"
            )
        candidates.append(
            CandidatePreferenceEvidence(
                original_rank=feature.original_rank,
                item_id=feature.item_id,
                parent_asin=feature.candidate_id,
                sasrec_score=record.score,
                evidence=_textual_explicit_evidence(feature),
            )
        )
    preference_ids = {
        evidence.preference_id
        for candidate in candidates
        for evidence in candidate.evidence
    }
    statuses = [
        evidence.status for candidate in candidates for evidence in candidate.evidence
    ]
    counts = CandidateEvidenceCounts(
        match_count=sum(1 for status in statuses if status is EvidenceStatus.MATCH),
        violation_count=sum(
            1 for status in statuses if status is EvidenceStatus.VIOLATION
        ),
        unknown_count=sum(1 for status in statuses if status is EvidenceStatus.UNKNOWN),
    )
    return PreferenceEvidenceReport(
        candidates=tuple(candidates),
        active_preference_count=len(preference_ids),
        counts=counts,
    )


def _reason_pair(
    ordered: Sequence[CandidateRankingFeatures],
    position: int,
    *,
    include_inferred: bool,
) -> str:
    """Placement reason from a downward comparison with the candidate immediately below.

    Mirrors the accepted M10B attribution convention: a candidate is explained by the key
    comparison it won against the candidate placed below it.  The final candidate has
    nothing below it, so it is labelled by its deterministic tail key.
    """
    feature = ordered[position]
    if position == len(ordered) - 1:
        return REASON_ORIGINAL_RANK_TIEBREAK
    below = ordered[position + 1]
    if feature.explicit_key != below.explicit_key:
        return REASON_EXPLICIT_PREFERENCE_ORDER
    if include_inferred and feature.inferred_key != below.inferred_key:
        return REASON_INFERRED_ORDINAL_TIEBREAK
    return REASON_ORIGINAL_RANK_TIEBREAK


def _build_arm_ranking(
    *,
    arm: RankingArm,
    task_id: str,
    ordered: Sequence[CandidateRankingFeatures],
    reasons: Sequence[str],
    eligible_ids: Sequence[str],
) -> ArmRanking:
    """Assemble one arm ranking, asserting the candidate set was preserved exactly."""
    identities = tuple(feature.candidate_id for feature in ordered)
    if len(set(identities)) != len(identities):
        raise RankingPolicyError(
            f"arm {arm.value} produced duplicate candidate identities"
        )
    if set(identities) != set(eligible_ids):
        raise RankingPolicyError(
            f"arm {arm.value} changed the eligible candidate set: it must rank exactly "
            f"{len(eligible_ids)} candidates, found {len(identities)}"
        )
    if len(reasons) != len(ordered):  # pragma: no cover - internal invariant
        raise RankingPolicyError(f"arm {arm.value} has an inconsistent reason list")
    return ArmRanking(
        arm=arm,
        task_id=task_id,
        candidate_ids=identities,
        ranked=tuple(
            _ranked_candidate(feature=feature, rank=position, reason=reasons[position - 1])
            for position, feature in enumerate(ordered, start=1)
        ),
        candidate_count=len(identities),
        candidate_set_digest=_candidate_set_digest(identities),
        policy_description=POLICY_DESCRIPTIONS[arm],
    )


def _rank_projected(
    arm: RankingArm,
    eligible: Sequence[CandidateRankingFeatures],
    snapshot: CandidateSnapshot,
    *,
    task_id: str,
) -> ArmRanking:
    """Rank the already-projected eligible pool with one arm."""
    eligible_ids = [feature.candidate_id for feature in eligible]

    if arm is RankingArm.A0_ORIGINAL:
        ordered = sorted(
            eligible, key=lambda feature: (feature.original_rank, feature.item_id)
        )
        return _build_arm_ranking(
            arm=arm,
            task_id=task_id,
            ordered=ordered,
            reasons=[REASON_ORIGINAL_ORDER] * len(ordered),
            eligible_ids=eligible_ids,
        )

    if arm is RankingArm.A1_EXPLICIT_RERANK:
        report = _a1_report(eligible, snapshot)
        reranked = rerank_candidates(report=report)
        features_by_id = {feature.candidate_id: feature for feature in eligible}
        ordered = [
            features_by_id[candidate.parent_asin] for candidate in reranked.candidates
        ]
        ordered = _category_numeric_overlay(ordered, snapshot)
        return _build_arm_ranking(
            arm=arm,
            task_id=task_id,
            ordered=ordered,
            reasons=[
                _reason_pair(ordered, position, include_inferred=False)
                for position in range(len(ordered))
            ],
            eligible_ids=[feature.candidate_id for feature in ordered],
        )

    # A2_BELIEF_TIEBREAK: exact preregistered key; numeric directional evidence is
    # category-scoped by the overlay below, never by the global lexicographic counts.
    ordered = sorted(
        eligible,
        key=lambda feature: (
            *_textual_explicit_key(feature),
            feature.inferred_ordinal_penalty,
            -feature.inferred_ordinal_support,
            feature.original_rank,
            feature.item_id,
        ),
    )
    ordered = _category_numeric_overlay(ordered, snapshot)
    return _build_arm_ranking(
        arm=arm,
        task_id=task_id,
        ordered=ordered,
        reasons=[
            _reason_pair(ordered, position, include_inferred=True)
            for position in range(len(ordered))
        ],
        eligible_ids=[feature.candidate_id for feature in ordered],
    )


def rank_arm(
    arm: RankingArm | str,
    feature_set: CandidateFeatureSet,
    snapshot: CandidateSnapshot,
) -> ArmRanking:
    """Rank one arm over the shared eligibility-projected candidate pool.

    The eligibility projection is computed here (once) for direct callers; callers ranking
    several arms of one task should use :func:`rank_all_arms`, which computes the
    projection exactly once for all arms.
    """
    resolved = _as_arm(arm)
    _validate_inputs(feature_set, snapshot)
    projection = shared_eligibility_projection(feature_set)
    features_by_id = {feature.candidate_id: feature for feature in feature_set.features}
    eligible = [features_by_id[candidate_id] for candidate_id in projection.eligible_ids]
    return _rank_projected(resolved, eligible, snapshot, task_id=feature_set.task_id)


def rank_all_arms(
    feature_set: CandidateFeatureSet,
    snapshot: CandidateSnapshot,
) -> tuple[ArmRanking, ArmRanking, ArmRanking]:
    """Rank A0, A1 and A2 over **one** shared eligibility projection.

    Returns ``(a0, a1, a2)`` in preregistered order.  The shared projection is computed
    once; all three arms therefore rank exactly the same candidate set, and the function
    asserts that per arm (a set change raises :class:`RankingPolicyError`).
    """
    _validate_inputs(feature_set, snapshot)
    projection = shared_eligibility_projection(feature_set)
    features_by_id = {feature.candidate_id: feature for feature in feature_set.features}
    eligible = [features_by_id[candidate_id] for candidate_id in projection.eligible_ids]
    rankings = tuple(
        _rank_projected(arm, eligible, snapshot, task_id=feature_set.task_id)
        for arm in PREREGISTERED_ARMS
    )
    reference = set(rankings[0].candidate_ids)
    for ranking in rankings[1:]:
        if set(ranking.candidate_ids) != reference:  # pragma: no cover - defensive
            raise RankingPolicyError("arms disagree on the candidate set")
    return rankings  # type: ignore[return-value]
