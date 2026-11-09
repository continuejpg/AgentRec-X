"""Tests for the M6.1 preregistered ranking policies A0 / A1 / A2.

These tests use the frozen ranking contracts directly (no synthetic fixture file needed):
a real ``ProductMetadata`` catalogue record is normalized, a frozen
``CandidateSnapshot`` is built, and ``CandidateFeatureSet`` values are constructed by
hand so every contract invariant is asserted on exact inputs.

What is pinned here (M6.1 required categories):
* A0 is exactly the trusted snapshot order;
* A1 is exactly a direct call to the **real, unmodified** M10B ``rerank_candidates``;
* A2 is deterministic, exact-key ordered, and can never place worse explicit evidence
  ahead of better (inferred evidence only breaks explicit ties);
* all arms rank exactly the shared eligibility-projected candidate set (identity and set
  equality), and an unresolved/ineligible candidate is never presentable;
* raw score fields are provenance only (changing them cannot change any order);
* reason labels are the documented vocabulary and candidate-set digests are
  order-insensitive.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.metadata import normalize_product_record  # noqa: E402
from recommendation.memory import PreferenceKind, PreferencePolarity  # noqa: E402
from recommendation.preference_matching.schemas import (  # noqa: E402
    MATCHING_SUPPORT,
    CandidateEvidenceCounts,
    CandidatePreferenceEvidence,
    EvidenceStatus,
    PreferenceEvidence,
    PreferenceEvidenceReport,
    ReasonCode,
)
from recommendation.ranking.errors import RankingPolicyError  # noqa: E402
from recommendation.ranking.policies import (  # noqa: E402
    POLICY_DESCRIPTION_A0,
    POLICY_DESCRIPTION_A1,
    POLICY_DESCRIPTION_A2,
    REASON_EXPLICIT_PREFERENCE_ORDER,
    REASON_INFERRED_ORDINAL_TIEBREAK,
    REASON_ORIGINAL_ORDER,
    REASON_ORIGINAL_RANK_TIEBREAK,
    rank_all_arms,
    rank_arm,
    shared_eligibility_projection,
)
from recommendation.ranking.schemas import (  # noqa: E402
    ArmRanking,
    CandidateFeatureSet,
    CandidateRankingFeatures,
    CandidateRecord,
    CandidateSnapshot,
    CatalogueCoverage,
    EligibilityState,
    RankedCandidate,
    RankingArm,
    SourceRank,
)
from recommendation.reranking.reranker import rerank_candidates  # noqa: E402
from recommendation.state.state_projector import RankerStateProjection  # noqa: E402

# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #

#: Synthetic catalogue used by these tests.  Every record is normalized by the accepted
#: catalogue normalizer, so these are real ``ProductMetadata`` values.
_CATALOGUE: tuple[tuple[str, int, dict[str, Any]], ...] = (
    ("cand-a", 101, {"color": "red", "material": "leather", "price": "89.99", "brand": "Acme", "weight": "0.8"}),
    ("cand-b", 102, {"color": "blue", "material": "mesh", "price": "59.00", "brand": "BlueCo", "weight": "1.1"}),
    ("cand-c", 103, {"color": "green", "material": "canvas", "price": "120.00", "brand": "Acme", "weight": "3.4"}),
    ("cand-d", 104, {"color": "red", "material": "leather", "price": "75.00", "brand": "Acme", "weight": "0.4"}),
    ("cand-e", 105, {"color": "black", "material": "canvas", "price": "45.00", "brand": "Trail", "weight": "2.2"}),
    ("cand-f", 106, {"color": "red", "material": "mesh", "price": "150.00", "brand": "Trail", "weight": "1.5"}),
)

_ALL_IDS: tuple[str, ...] = tuple(spec[0] for spec in _CATALOGUE)


def _metadata(spec: dict[str, Any], parent_asin: str) -> Any:
    """Normalize one synthetic catalogue row into real ``ProductMetadata``."""
    record: dict[str, Any] = {
        "parent_asin": parent_asin,
        "title": f"{spec.get('color', 'item')} {parent_asin}",
        "details": {},
    }
    if spec.get("color"):
        record["details"]["Color"] = spec["color"]
    if spec.get("material"):
        record["details"]["Material"] = spec["material"]
    if spec.get("weight"):
        record["details"]["Item Weight"] = spec["weight"]
    if spec.get("brand"):
        record["store"] = spec["brand"]
    if spec.get("price") is not None:
        record["price"] = spec["price"]
    return normalize_product_record(record)


def build_snapshot(
    *,
    assessments: dict[str, EligibilityState] | None = None,
    scores: dict[str, float] | None = None,
    user_key: str = "user-1",
    task_id: str = "task-1",
    turn_key: str = "turn-1",
) -> CandidateSnapshot:
    """Build a frozen six-candidate snapshot with optional eligibility/score overrides."""
    assessments = assessments or {}
    scores = scores or {}
    records = []
    for index, (candidate_id, item_id, spec) in enumerate(_CATALOGUE):
        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                item_id=item_id,
                original_rank=index + 1,
                score=scores.get(candidate_id, 1.0 - index * 0.05),
                score_kind="sasrec_logit",
                metadata_status="found",
                metadata=_metadata(spec, candidate_id),
                source_ranks=(
                    SourceRank(source="history", rank=index + 1, score=1.0, score_kind="sasrec_logit"),
                ),
                rrf_score=0.5 / (index + 1),
                eligibility=assessments.get(candidate_id, EligibilityState.NOT_ASSESSED),
            )
        )
    return CandidateSnapshot(
        snapshot_id=f"snap-{task_id}",
        user_key=user_key,
        task_id=task_id,
        turn_key=turn_key,
        candidates=tuple(records),
        candidate_pool_ref=f"cp-{task_id}",
        ranking_snapshot_ref=f"rs-{task_id}",
    )


def color_evidence(
    status: EvidenceStatus,
    *,
    preference_id: str = "pref-colour",
    value: str = "red",
    polarity: PreferencePolarity = PreferencePolarity.PREFER,
    sequence: int = 1,
) -> PreferenceEvidence:
    """Build one accepted M10A evidence record with the requested three-state status."""
    reason = {
        EvidenceStatus.MATCH: ReasonCode.EXACT_VALUE_MATCH,
        EvidenceStatus.VIOLATION: ReasonCode.EXPLICIT_VALUE_CONFLICT,
        EvidenceStatus.UNKNOWN: ReasonCode.INSUFFICIENT_METADATA,
    }[status]
    return PreferenceEvidence(
        preference_id=preference_id,
        preference_kind=PreferenceKind.COLOR,
        preference_polarity=polarity,
        preference_value=value,
        preference_source_text="I prefer red",
        preference_source_turn_id="turn-1",
        preference_logical_seq=sequence,
        status=status,
        reason_code=reason,
        support=MATCHING_SUPPORT[PreferenceKind.COLOR],
        metadata_field="details.Color",
        metadata_value="red" if status is EvidenceStatus.MATCH else None,
        metadata_present=True,
    )


def default_feature_spec() -> dict[str, dict[str, Any]]:
    """The standard explicit/inferred scenario used by the policy tests.

    * ``cand-b`` explicitly violates the colour preference but carries the best inferred
      ordinal support -- explicit evidence must still place it after the compliant
      candidates;
    * ``cand-a`` and ``cand-d`` are explicit ties; ``cand-d`` has better ordinal inferred
      support, so A2 may promote it above ``cand-a`` (and only there);
    * ``cand-f`` is an explicit tie with a *penalty*, so A2 demotes it within its tie;
    * ``cand-c`` is an explicit UNKNOWN: neither rewarded nor penalised.
    """
    return {
        "cand-a": {
            "explicit": (color_evidence(EvidenceStatus.MATCH),),
            "inferred_ordinal_support": 0,
            "inferred_ordinal_penalty": 0,
        },
        "cand-b": {
            "explicit": (color_evidence(EvidenceStatus.VIOLATION),),
            "inferred_ordinal_support": 99,
            "inferred_ordinal_penalty": 0,
        },
        "cand-c": {
            "explicit": (color_evidence(EvidenceStatus.UNKNOWN),),
            "inferred_unknown": 1,
        },
        "cand-d": {
            "explicit": (color_evidence(EvidenceStatus.MATCH),),
            "inferred_ordinal_support": 1,
            "inferred_ordinal_penalty": 0,
        },
        "cand-e": {
            "explicit": (),
            "inferred_unknown": 1,
        },
        "cand-f": {
            "explicit": (color_evidence(EvidenceStatus.MATCH),),
            "inferred_ordinal_support": 0,
            "inferred_ordinal_penalty": 2,
        },
    }


def make_feature_set(
    snapshot: CandidateSnapshot,
    spec: dict[str, dict[str, Any]] | None = None,
    *,
    feature_order: Sequence[str] | None = None,
) -> CandidateFeatureSet:
    """Build a valid feature set whose counts are derived from the evidence records."""
    spec = spec if spec is not None else {}
    records = {record.candidate_id: record for record in snapshot.candidates}
    ordered_ids = tuple(feature_order) if feature_order is not None else snapshot.candidate_ids
    features: list[CandidateRankingFeatures] = []
    for candidate_id in ordered_ids:
        record = records[candidate_id]
        options = dict(spec.get(candidate_id, {}))
        explicit = tuple(options.pop("explicit", ()))
        inferred = tuple(options.pop("inferred", ()))
        features.append(
            CandidateRankingFeatures(
                candidate_id=record.candidate_id,
                item_id=record.item_id,
                original_rank=record.original_rank,
                source_ranks=record.source_ranks,
                rrf_score=record.rrf_score,
                explicit_matches=sum(
                    1 for item in explicit if item.status is EvidenceStatus.MATCH
                ),
                explicit_violations=sum(
                    1 for item in explicit if item.status is EvidenceStatus.VIOLATION
                ),
                explicit_unknown=sum(
                    1 for item in explicit if item.status is EvidenceStatus.UNKNOWN
                ),
                inferred_evidence=inferred,
                eligibility=record.eligibility,
                explicit_evidence=explicit,
                **options,
            )
        )
    return CandidateFeatureSet(
        task_id=snapshot.task_id,
        user_key=snapshot.user_key,
        turn_key=snapshot.turn_key,
        snapshot_id=snapshot.snapshot_id,
        snapshot_candidate_count=snapshot.candidate_count,
        candidate_pool_digest=snapshot.candidate_pool_digest_value(),
        ranking_digest=snapshot.ranking_digest_value(),
        features=tuple(features),
        coverage=CatalogueCoverage(
            candidate_count=snapshot.candidate_count,
            metadata_found_count=snapshot.candidate_count,
            metadata_missing_count=0,
            readable_fields={"details.Color": snapshot.candidate_count},
            supported_inferred_features=("color",),
        ),
    )


def make_ranker_projection(
    *, task_id: str = "task-1", state_version: int = 1
) -> RankerStateProjection:
    """A minimal valid ``for_ranker`` projection for the frozen task contract."""
    return RankerStateProjection(task_id=task_id, state_version=state_version)


def scenario(
    *,
    assessments: dict[str, EligibilityState] | None = None,
    spec: dict[str, dict[str, Any]] | None = None,
    labels: Sequence[Any] = (),
    label_protocol: str | None = None,
    scores: dict[str, float] | None = None,
    task_id: str = "task-1",
    user_key: str = "user-1",
    turn_key: str = "turn-1",
) -> tuple[CandidateSnapshot, CandidateFeatureSet, Any]:
    """Build ``(snapshot, features, FrozenRankingTask)`` for one test scenario."""
    from recommendation.ranking.schemas import FrozenRankingTask

    snapshot = build_snapshot(
        assessments=assessments,
        scores=scores,
        user_key=user_key,
        task_id=task_id,
        turn_key=turn_key,
    )
    features = make_feature_set(snapshot, spec if spec is not None else default_feature_spec())
    task = FrozenRankingTask(
        task_id=task_id,
        user_key=user_key,
        turn_key=turn_key,
        snapshot=snapshot,
        ranker_projection=make_ranker_projection(task_id=task_id),
        labels=tuple(labels),
        label_protocol=label_protocol,
        dataset_source="synthetic_policy_tests",
    )
    return snapshot, features, task


def positions(ranking: ArmRanking) -> dict[str, int]:
    """Candidate id -> 1-based position in one arm ranking."""
    return {candidate_id: index for index, candidate_id in enumerate(ranking.candidate_ids, 1)}


# --------------------------------------------------------------------------- #
# A0: trusted original order
# --------------------------------------------------------------------------- #


def test_a0_is_the_trusted_original_order() -> None:
    snapshot, features, _ = scenario()
    a0 = rank_arm(RankingArm.A0_ORIGINAL, features, snapshot)
    assert a0.arm is RankingArm.A0_ORIGINAL
    assert a0.candidate_ids == snapshot.candidate_ids
    assert a0.candidate_count == snapshot.candidate_count
    assert [item.rank for item in a0.ranked] == list(range(1, snapshot.candidate_count + 1))
    assert [item.original_rank for item in a0.ranked] == list(range(1, snapshot.candidate_count + 1))
    assert {item.reason for item in a0.ranked} == {REASON_ORIGINAL_ORDER}
    assert a0.policy_description == POLICY_DESCRIPTION_A0


# --------------------------------------------------------------------------- #
# A1: the real unmodified M10B reranker
# --------------------------------------------------------------------------- #


def test_a1_equals_a_direct_real_m10b_rerank_call() -> None:
    snapshot, features, _ = scenario()
    a0, a1, _ = rank_all_arms(features, snapshot)
    records = {record.candidate_id: record for record in snapshot.candidates}
    expected_report = PreferenceEvidenceReport(
        candidates=tuple(
            CandidatePreferenceEvidence(
                original_rank=feature.original_rank,
                item_id=feature.item_id,
                parent_asin=feature.candidate_id,
                sasrec_score=records[feature.candidate_id].score,
                evidence=feature.explicit_evidence,
            )
            for feature in sorted(
                features.features, key=lambda item: (item.original_rank, item.item_id)
            )
        ),
    )
    expected = rerank_candidates(report=expected_report)
    assert a1.candidate_ids == expected.parent_asins
    assert [item.item_id for item in a1.ranked] == list(expected.item_ids)
    assert a1.arm is RankingArm.A1_EXPLICIT_RERANK
    assert a1.policy_description == POLICY_DESCRIPTION_A1
    # Sanity: the scenario is interesting -- A1 really differs from the original order.
    assert a1.candidate_ids != a0.candidate_ids
    # A1 ranks exactly the same candidates (never filters, even a violation).
    assert set(a1.candidate_ids) == set(snapshot.candidate_ids)


def test_a1_evidence_is_the_feature_evidence_verbatim() -> None:
    snapshot, features, _ = scenario()
    _, a1, _ = rank_all_arms(features, snapshot)
    feature_by_id = {feature.candidate_id: feature for feature in features.features}
    for item in a1.ranked:
        feature = feature_by_id[item.candidate_id]
        assert item.explicit_matches == feature.explicit_matches
        assert item.explicit_violations == feature.explicit_violations
        assert item.original_rank == feature.original_rank
        assert item.item_id == feature.item_id


# --------------------------------------------------------------------------- #
# A2: exact key, determinism, explicit-priority preservation
# --------------------------------------------------------------------------- #


def test_a2_is_deterministic() -> None:
    snapshot, features, _ = scenario()
    first = rank_arm(RankingArm.A2_BELIEF_TIEBREAK, features, snapshot)
    second = rank_arm(RankingArm.A2_BELIEF_TIEBREAK, features, snapshot)
    assert first == second
    assert first.candidate_ids == second.candidate_ids
    assert first.candidate_set_digest == second.candidate_set_digest
    # Feeding the same features in a different order also ranks identically.
    shuffled = make_feature_set(snapshot, default_feature_spec(), feature_order=tuple(reversed(_ALL_IDS)))
    third = rank_arm(RankingArm.A2_BELIEF_TIEBREAK, shuffled, snapshot)
    assert third.candidate_ids == first.candidate_ids


def test_a2_uses_inferred_only_after_explicit_ties() -> None:
    snapshot, features, _ = scenario()
    _, _, a2 = rank_all_arms(features, snapshot)
    placed = positions(a2)
    # cand-a and cand-d tie on explicit evidence; only there may inferred support promote d.
    assert placed["cand-d"] < placed["cand-a"]
    # cand-b has the best inferred support but an explicit violation: it must stay after
    # every candidate with no explicit violation.
    assert placed["cand-b"] > placed["cand-a"]
    assert placed["cand-b"] > placed["cand-d"]
    assert placed["cand-b"] > placed["cand-f"]
    # cand-f is an explicit tie with an ordinal penalty: it stays inside its tie group.
    assert placed["cand-f"] < placed["cand-b"]
    assert placed["cand-f"] > placed["cand-d"]
    # Global invariant: the sequence of explicit keys is non-decreasing in A2 order.
    feature_by_id = {feature.candidate_id: feature for feature in features.features}
    keys = [feature_by_id[candidate_id].explicit_key for candidate_id in a2.candidate_ids]
    assert keys == sorted(keys)


def test_a2_cannot_override_an_explicit_preference() -> None:
    """Better inferred evidence can never move a candidate ahead of better explicit evidence."""
    spec = default_feature_spec()
    # Give the violating candidate maximal inferred signal and strip it from the best
    # explicit candidate: explicit priority must still dominate.
    spec["cand-b"]["inferred_ordinal_support"] = 1000
    spec["cand-a"]["inferred_ordinal_support"] = 0
    snapshot, features, _ = scenario(spec=spec)
    _, a1, a2 = rank_all_arms(features, snapshot)
    assert positions(a2)["cand-b"] > positions(a2)["cand-a"]
    assert a1.candidate_ids.index("cand-b") > a1.candidate_ids.index("cand-a")
    # A2 is not "A1 re-sorted from scratch": every explicit-key order is preserved.
    feature_by_id = {feature.candidate_id: feature for feature in features.features}
    keys = [feature_by_id[candidate_id].explicit_key for candidate_id in a2.candidate_ids]
    assert keys == sorted(keys)


def test_a2_demotes_within_a_tie_but_a1_keeps_original_order() -> None:
    snapshot, features, _ = scenario()
    _, a1, a2 = rank_all_arms(features, snapshot)
    assert positions(a1)["cand-d"] > positions(a1)["cand-a"]
    assert positions(a2)["cand-d"] < positions(a2)["cand-a"]


def test_a2_reason_labels_use_the_documented_vocabulary() -> None:
    snapshot, features, _ = scenario()
    _, _, a2 = rank_all_arms(features, snapshot)
    allowed = {
        REASON_ORIGINAL_ORDER,
        REASON_EXPLICIT_PREFERENCE_ORDER,
        REASON_INFERRED_ORDINAL_TIEBREAK,
        REASON_ORIGINAL_RANK_TIEBREAK,
    }
    assert {item.reason for item in a2.ranked} <= allowed
    # The promotion of cand-d over cand-a is an inferred ordinal tie-break.
    promoted = next(item for item in a2.ranked if item.candidate_id == "cand-d")
    assert promoted.reason == REASON_INFERRED_ORDINAL_TIEBREAK


# --------------------------------------------------------------------------- #
# shared eligibility projection
# --------------------------------------------------------------------------- #


def test_inferred_evidence_never_filters_a_candidate() -> None:
    """Soft inferred evidence can reorder; it can never remove or add a candidate."""
    spec = default_feature_spec()
    spec["cand-e"] = {"explicit": (), "inferred_violations": 3, "inferred_ordinal_penalty": 5}
    snapshot, features, _ = scenario(spec=spec)
    projection = shared_eligibility_projection(features)
    a0, a1, a2 = rank_all_arms(features, snapshot)
    for arm in (a0, a1, a2):
        assert set(arm.candidate_ids) == set(projection.eligible_ids)
        assert arm.candidate_count == snapshot.candidate_count
        assert "cand-e" in arm.candidate_ids
    # The soft "violation" is not a hard constraint: it simply sorts last inside its
    # explicit tie group (no explicit evidence => explicit key ties with cand-c).
    assert positions(a2)["cand-e"] > positions(a2)["cand-a"]


def test_unknown_explicit_evidence_is_neutral_not_a_violation() -> None:
    """An UNKNOWN explicit evidence record is neither rewarded nor penalised."""
    snapshot, features, _ = scenario()
    _, a1, _ = rank_all_arms(features, snapshot)
    placed = positions(a1)
    # cand-c (UNKNOWN explicit) ranks ahead of cand-b (VIOLATION), and both are behind
    # candidates with at least one explicit match.
    assert placed["cand-c"] < placed["cand-b"]
    assert placed["cand-c"] > placed["cand-a"]
    assert placed["cand-e"] == placed["cand-c"] + 1  # zero evidence sorts by original rank


def test_explicit_priority_order_is_identical_between_a1_and_a2() -> None:
    """A1 and A2 may only disagree about candidates tied on explicit priority."""
    snapshot, features, _ = scenario()
    _, a1, a2 = rank_all_arms(features, snapshot)
    feature_by_id = {feature.candidate_id: feature for feature in features.features}
    for left_index in range(len(a1.candidate_ids)):
        for right_index in range(left_index + 1, len(a1.candidate_ids)):
            left = a1.candidate_ids[left_index]
            right = a1.candidate_ids[right_index]
            if feature_by_id[left].explicit_key != feature_by_id[right].explicit_key:
                assert a2.candidate_ids.index(left) < a2.candidate_ids.index(right)


def test_eligibility_projection_is_applied_once_and_recorded() -> None:
    snapshot, features, _ = scenario()
    projection = shared_eligibility_projection(features)
    assert projection.assessed is False
    assert projection.eligible_ids == snapshot.candidate_ids
    assert projection.excluded_ids == ()
    assert "no eligibility assessment" in projection.reason
    # Every arm ranks exactly the projected set; no arm can filter anything else.
    a0, a1, a2 = rank_all_arms(features, snapshot)
    assert a0.candidate_ids == projection.eligible_ids
    for arm in (a1, a2):
        assert set(arm.candidate_ids) == set(projection.eligible_ids)
        assert arm.candidate_count == projection.eligible_count


def test_hard_constraint_eligibility_dominates_all_arms() -> None:
    assessments = {
        "cand-a": EligibilityState.VERIFIED_ELIGIBLE,
        "cand-b": EligibilityState.INELIGIBLE,
        "cand-c": EligibilityState.UNRESOLVED,
        "cand-d": EligibilityState.VERIFIED_ELIGIBLE,
        "cand-e": EligibilityState.INELIGIBLE,
        "cand-f": EligibilityState.UNRESOLVED,
    }
    snapshot, features, _ = scenario(assessments=assessments)
    projection = shared_eligibility_projection(features)
    assert projection.assessed is True
    assert projection.eligible_ids == ("cand-a", "cand-d")
    assert set(projection.excluded_ids) == {"cand-b", "cand-c", "cand-e", "cand-f"}
    a0, a1, a2 = rank_all_arms(features, snapshot)
    assert a0.candidate_ids == ("cand-a", "cand-d")
    for arm in (a0, a1, a2):
        assert set(arm.candidate_ids) == {"cand-a", "cand-d"}
        # No excluded identity may appear in any arm.
        assert arm.candidate_count == 2
        assert not (set(arm.candidate_ids) & set(projection.excluded_ids))


def test_unresolved_is_not_presentable_when_an_assessment_exists() -> None:
    """UNRESOLVED and NOT_ASSESSED are both excluded once eligibility ran; only VERIFIED is."""
    assessments = {
        "cand-a": EligibilityState.VERIFIED_ELIGIBLE,
        "cand-b": EligibilityState.UNRESOLVED,
        # cand-c..f remain NOT_ASSESSED
    }
    snapshot, features, _ = scenario(assessments=assessments)
    projection = shared_eligibility_projection(features)
    assert projection.eligible_ids == ("cand-a",)
    assert set(projection.excluded_ids) == {"cand-b", "cand-c", "cand-d", "cand-e", "cand-f"}
    for arm in rank_all_arms(features, snapshot):
        assert arm.candidate_ids == ("cand-a",)


def test_empty_eligible_pool_yields_empty_arms() -> None:
    assessments = {candidate_id: EligibilityState.INELIGIBLE for candidate_id in _ALL_IDS}
    snapshot, features, _ = scenario(assessments=assessments)
    a0, a1, a2 = rank_all_arms(features, snapshot)
    for arm in (a0, a1, a2):
        assert arm.candidate_ids == ()
        assert arm.candidate_count == 0
        assert arm.ranked == ()
    assert a0.candidate_set_digest == a1.candidate_set_digest == a2.candidate_set_digest


# --------------------------------------------------------------------------- #
# candidate set, identity and digest invariants
# --------------------------------------------------------------------------- #


def test_all_arms_rank_the_same_set_and_digest() -> None:
    snapshot, features, _ = scenario()
    a0, a1, a2 = rank_all_arms(features, snapshot)
    assert set(a0.candidate_ids) == set(a1.candidate_ids) == set(a2.candidate_ids)
    assert a0.candidate_set_digest == a1.candidate_set_digest == a2.candidate_set_digest
    assert a0.candidate_count == a1.candidate_count == a2.candidate_count
    assert len(a0.candidate_ids) == len(set(a0.candidate_ids))


def test_candidate_set_digest_is_order_insensitive() -> None:
    snapshot, features, _ = scenario()
    _, a1, _ = rank_all_arms(features, snapshot)
    shuffled = make_feature_set(
        snapshot, default_feature_spec(), feature_order=tuple(reversed(_ALL_IDS))
    )
    _, shuffled_a1, _ = rank_all_arms(shuffled, snapshot)
    assert shuffled_a1.candidate_set_digest == a1.candidate_set_digest


def test_identity_mismatch_is_refused() -> None:
    snapshot, features, _ = scenario()
    foreign = features.model_copy(
        update={
            "features": (
                features.features[0].model_copy(update={"candidate_id": "cand-unknown"}),
                *features.features[1:],
            )
        }
    )
    with pytest.raises(RankingPolicyError):
        rank_all_arms(foreign, snapshot)
    wrong_task = features.model_copy(update={"task_id": "other-task"})
    with pytest.raises(RankingPolicyError):
        rank_all_arms(wrong_task, snapshot)
    wrong_rank = features.model_copy(
        update={
            "features": (
                features.features[0].model_copy(update={"original_rank": 99}),
                *features.features[1:],
            )
        }
    )
    with pytest.raises(RankingPolicyError):
        rank_all_arms(wrong_rank, snapshot)


def test_unknown_arm_is_refused() -> None:
    snapshot, features, _ = scenario()
    with pytest.raises(RankingPolicyError):
        rank_arm("a9_unknown", features, snapshot)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# no score mixing: raw scores are provenance only
# --------------------------------------------------------------------------- #


def test_no_raw_score_can_change_any_arm_order() -> None:
    snapshot, features, _ = scenario()
    baseline = rank_all_arms(features, snapshot)
    # Perturb every raw score field that is *not* a policy input: snapshot ordering score,
    # source scores/ranks and the copied rrf score.
    loud = {
        candidate_id: 1000.0 - index
        for index, candidate_id in enumerate(reversed(_ALL_IDS))
    }
    loud_snapshot = build_snapshot(scores=loud)
    loud_features = make_feature_set(loud_snapshot, default_feature_spec())
    loud_features = loud_features.model_copy(
        update={
            "features": tuple(
                feature.model_copy(
                    update={
                        "rrf_score": 99.0 - feature.original_rank,
                        "source_ranks": (
                            SourceRank(
                                source="catalog_search",
                                rank=99 - feature.original_rank,
                                score=123.456,
                                score_kind="bm25",
                            ),
                        ),
                    }
                )
                for feature in loud_features.features
            )
        }
    )
    perturbed = rank_all_arms(loud_features, loud_snapshot)
    for baseline_arm, perturbed_arm in zip(baseline, perturbed):
        assert baseline_arm.candidate_ids == perturbed_arm.candidate_ids
        assert baseline_arm.candidate_set_digest == perturbed_arm.candidate_set_digest


def test_policy_descriptions_are_exposed_on_every_arm() -> None:
    snapshot, features, _ = scenario()
    a0, a1, a2 = rank_all_arms(features, snapshot)
    assert a0.policy_description == POLICY_DESCRIPTION_A0
    assert a1.policy_description == POLICY_DESCRIPTION_A1
    assert a2.policy_description == POLICY_DESCRIPTION_A2
    for ranking in (a0, a1, a2):
        assert isinstance(ranking, ArmRanking)
        assert all(isinstance(item, RankedCandidate) for item in ranking.ranked)
