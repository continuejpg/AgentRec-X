"""Focused tests for the M6.1 trusted candidate snapshot resolver.

The resolver is the trust join between the accepted recommendation pipeline and the
offline ranking lab: identity, catalogue metadata, ownership/freshness/count/digest
verification and the single shared eligibility projection.  Every test here exercises
that contract through the real frozen schemas and the real synthetic fixture; nothing in
the tests re-implements resolver behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.metadata import MetadataIndex  # noqa: E402
from recommendation.catalog.schemas import ProductMetadata  # noqa: E402
from recommendation.ranking.errors import CandidateSnapshotError  # noqa: E402
from recommendation.ranking.resolver import (  # noqa: E402
    TrustedCandidateFeatureResolver,
    TrustedCandidateInput,
)
from recommendation.ranking.schemas import (  # noqa: E402
    CandidateSnapshot,
    EligibilityState,
    SnapshotExpectation,
)
from recommendation.state.shopping_adapter import (  # noqa: E402
    candidate_pool_snapshot,
    ranking_snapshot,
)
from tests import m6_1_fixture as fx  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolver() -> TrustedCandidateFeatureResolver:
    """The fixture's trusted resolver (metadata index + identity map wired)."""
    return fx.build_trusted_resolver()


def _build_kwargs(variant: str = "default") -> dict[str, object]:
    """The trusted snapshot-identity arguments for one fixture variant."""
    return {
        "snapshot_id": fx.variant_snapshot_id(variant),
        "user_key": fx.USER_KEY,
        "task_id": fx.variant_task_id(variant),
        "turn_key": fx.variant_turn_key(variant),
    }


def _replace_candidate(
    candidates: tuple[TrustedCandidateInput, ...],
    index: int,
    **changes: object,
) -> tuple[TrustedCandidateInput, ...]:
    """Return the candidate tuple with one candidate replaced by an invalid copy."""
    updated = list(candidates)
    updated[index] = updated[index].model_copy(update=changes)
    return tuple(updated)


def _tampered_snapshot(
    snapshot: CandidateSnapshot,
    *,
    candidates: object | None = None,
    **changes: object,
) -> CandidateSnapshot:
    """Return a copy of a valid snapshot carrying a deliberately invalid value.

    ``model_copy`` bypasses validation on purpose: this models a corrupted/foreign
    snapshot arriving from outside the trusted build path, which verification must reject.
    """
    payload = dict(changes)
    if candidates is not None:
        payload["candidates"] = candidates
    return snapshot.model_copy(update=payload)


def _expectation(snapshot: CandidateSnapshot, **changes: object) -> SnapshotExpectation:
    """Build an expectation matching ``snapshot``, optionally overridden."""
    payload: dict[str, object] = {
        "user_key": snapshot.user_key,
        "task_id": snapshot.task_id,
        "turn_key": snapshot.turn_key,
        "candidate_count": snapshot.candidate_count,
        "candidate_pool_ref": snapshot.candidate_pool_ref,
        "ranking_snapshot_ref": snapshot.ranking_snapshot_ref,
        "candidate_pool_digest": snapshot.candidate_pool_digest,
        "ranking_snapshot_digest": snapshot.ranking_snapshot_digest,
    }
    payload.update(changes)
    return SnapshotExpectation(**payload)


# --------------------------------------------------------------------------- #
# Construction: trusted identity + metadata join + provenance
# --------------------------------------------------------------------------- #


def test_build_snapshot_joins_trusted_identity_and_metadata() -> None:
    snapshot = fx.build_snapshot()
    resolver = _resolver()
    assert isinstance(snapshot, CandidateSnapshot)
    assert resolver.verify_snapshot(snapshot) is snapshot
    assert snapshot.user_key == fx.USER_KEY
    assert snapshot.task_id == fx.variant_task_id("default")
    assert snapshot.turn_key == fx.variant_turn_key("default")
    assert snapshot.source == "trusted_candidate_snapshot"
    assert snapshot.candidate_count == len(fx.CANDIDATE_ROWS) == 7
    assert snapshot.candidate_ids == fx.CANDIDATE_IDS

    joined = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    assert [candidate.original_rank for candidate in snapshot.candidates] == list(range(1, 8))
    assert [candidate.item_id for candidate in snapshot.candidates] == [
        row["item_id"] for row in fx.CANDIDATE_ROWS
    ]
    assert [candidate.score for candidate in snapshot.candidates] == [
        row["rrf_score"] for row in fx.CANDIDATE_ROWS
    ]
    for candidate in snapshot.candidates:
        assert candidate.score_kind == "rrf_rank_fusion"
        assert candidate.rrf_score == candidate.score
        assert len(candidate.source_ranks) == 2
        assert candidate.source_ranks[0].source == "history_sasrec"
        assert candidate.source_ranks[1].source == "catalog_search"

    first = joined["B0M6HIKE01"]
    assert first.metadata_status == "found"
    assert first.metadata is not None
    assert first.metadata.parent_asin == first.candidate_id
    assert first.metadata.title == "Ultralight Comfort Hiking Backpack 40L"
    assert first.metadata.price_text == "129.99"
    assert first.metadata.store == "TrailForge"

    # The trusted eligibility projection is copied verbatim.
    assert joined["B0M6HIKE01"].eligibility is EligibilityState.VERIFIED_ELIGIBLE
    assert joined["B0M6HIKE03"].eligibility is EligibilityState.INELIGIBLE
    assert joined["B0M6HIKE04"].eligibility is EligibilityState.UNRESOLVED


def test_build_snapshot_missing_metadata_is_explicit() -> None:
    snapshot = fx.build_snapshot()
    joined = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
    missing = joined["B0M6HIKE06"]
    assert missing.metadata_status == "missing"
    assert missing.metadata is None
    assert sum(1 for candidate in snapshot.candidates if candidate.metadata_status == "missing") == 1
    found = [candidate for candidate in snapshot.candidates if candidate.metadata_status == "found"]
    assert len(found) == 6
    assert all(candidate.metadata is not None for candidate in found)


def test_snapshot_refs_and_digests_use_m5_algorithms() -> None:
    snapshot = fx.build_snapshot()
    pool = candidate_pool_snapshot(fx.CANDIDATE_IDS)
    ranking = ranking_snapshot(fx.CANDIDATE_IDS)
    assert snapshot.candidate_pool_ref == pool.ref
    assert snapshot.candidate_pool_digest == pool.digest
    assert snapshot.candidate_pool_digest_value() == pool.digest
    assert snapshot.ranking_snapshot_ref == ranking.ref
    assert snapshot.ranking_snapshot_digest == ranking.digest
    assert snapshot.ranking_digest_value() == ranking.digest
    assert len(snapshot.candidate_pool_digest) == 64
    assert len(snapshot.ranking_snapshot_digest) == 64


def test_build_snapshot_without_refs_carries_no_digests() -> None:
    scope = _build_kwargs()
    snapshot = _resolver().build_snapshot(
        snapshot_id="m6-1-snapshot-no-refs",
        user_key=scope["user_key"],
        task_id=scope["task_id"],
        turn_key=scope["turn_key"],
        candidates=fx.build_candidate_inputs(),
    )
    assert snapshot.candidate_pool_ref is None
    assert snapshot.ranking_snapshot_ref is None
    assert snapshot.candidate_pool_digest is None
    assert snapshot.ranking_snapshot_digest is None
    assert _resolver().verify_snapshot(snapshot) is snapshot


# --------------------------------------------------------------------------- #
# Construction: rejections
# --------------------------------------------------------------------------- #


def test_trusted_candidate_input_rejects_pad_and_non_finite_score() -> None:
    base = fx.build_candidate_inputs()[0]
    with pytest.raises(ValidationError):
        TrustedCandidateInput(
            candidate_id=base.candidate_id,
            item_id=0,
            original_rank=1,
            score=base.score,
        )
    with pytest.raises(ValidationError):
        TrustedCandidateInput(
            candidate_id=base.candidate_id,
            item_id=base.item_id,
            original_rank=1,
            score=float("inf"),
        )
    with pytest.raises(ValidationError):
        TrustedCandidateInput(
            candidate_id="   ",
            item_id=base.item_id,
            original_rank=1,
            score=base.score,
        )


def test_build_snapshot_rejects_pad_item_id() -> None:
    candidates = _replace_candidate(fx.build_candidate_inputs(), 0, item_id=0)
    with pytest.raises(CandidateSnapshotError, match="PAD|positive"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_non_finite_score() -> None:
    candidates = _replace_candidate(fx.build_candidate_inputs(), 2, score=float("nan"))
    with pytest.raises(CandidateSnapshotError, match="finite"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_duplicate_candidate_ids() -> None:
    candidates = _replace_candidate(
        fx.build_candidate_inputs(), 1, candidate_id=fx.CANDIDATE_IDS[0]
    )
    with pytest.raises(CandidateSnapshotError, match="candidate ids must be unique"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_duplicate_item_ids() -> None:
    candidates = _replace_candidate(
        fx.build_candidate_inputs(), 1, item_id=fx.CANDIDATE_ROWS[0]["item_id"]
    )
    with pytest.raises(CandidateSnapshotError, match="item ids must be unique"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_rank_collision() -> None:
    candidates = _replace_candidate(fx.build_candidate_inputs(), 1, original_rank=1)
    with pytest.raises(CandidateSnapshotError, match="collide|contiguous"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_rank_gap() -> None:
    candidates = _replace_candidate(fx.build_candidate_inputs(), 4, original_rank=9)
    with pytest.raises(CandidateSnapshotError, match="contiguous"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_blank_candidate_id() -> None:
    candidates = _replace_candidate(fx.build_candidate_inputs(), 0, candidate_id="  ")
    with pytest.raises(CandidateSnapshotError, match="blank"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_foreign_identity_via_identity_map() -> None:
    candidates = _replace_candidate(
        fx.build_candidate_inputs(), 0, candidate_id="B0NOTINMAPPING"
    )
    with pytest.raises(CandidateSnapshotError, match="trusted identity mapping"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_item_id_identity_mismatch() -> None:
    candidates = _replace_candidate(fx.build_candidate_inputs(), 0, item_id=999)
    with pytest.raises(CandidateSnapshotError, match="identity mapping"):
        _resolver().build_snapshot(candidates=candidates, **_build_kwargs())


def test_build_snapshot_rejects_metadata_identity_mismatch() -> None:
    class _ForeignMetadataIndex:
        """A broken index returning a record for the wrong product."""

        def lookup(self, parent_asin: str) -> ProductMetadata:
            return ProductMetadata(parent_asin="B0DIFFERENT", source="test")

    resolver = TrustedCandidateFeatureResolver(
        metadata_index=_ForeignMetadataIndex(),
        identity_map=fx.build_identity_map(),
    )
    with pytest.raises(CandidateSnapshotError, match="different parent_asin"):
        resolver.build_snapshot(candidates=fx.build_candidate_inputs(), **_build_kwargs())


def test_build_snapshot_rejects_candidate_pool_ref_mismatch() -> None:
    kwargs = dict(_build_kwargs())
    kwargs["candidates"] = fx.build_candidate_inputs()
    kwargs["candidate_pool_ref"] = "cp_000000000000000000000000"
    with pytest.raises(CandidateSnapshotError, match="candidate_pool_ref"):
        _resolver().build_snapshot(**kwargs)


def test_build_snapshot_rejects_ranking_ref_mismatch() -> None:
    kwargs = dict(_build_kwargs())
    kwargs["candidates"] = fx.build_candidate_inputs()
    kwargs["ranking_snapshot_ref"] = "rs_000000000000000000000000"
    with pytest.raises(CandidateSnapshotError, match="ranking_snapshot_ref"):
        _resolver().build_snapshot(**kwargs)


def test_build_snapshot_rejects_short_identity_metadata_index() -> None:
    with pytest.raises(TypeError, match="lookup"):
        TrustedCandidateFeatureResolver(metadata_index=object())
    with pytest.raises(TypeError, match="identity_map"):
        TrustedCandidateFeatureResolver(
            metadata_index=MetadataIndex.from_records([]),
            identity_map=object(),
        )


# --------------------------------------------------------------------------- #
# Verification: ownership, freshness, count, refs and digests
# --------------------------------------------------------------------------- #


def test_verify_snapshot_accepts_matching_expectation() -> None:
    snapshot = fx.build_snapshot()
    expectation = _expectation(snapshot)
    resolver = _resolver()
    assert resolver.verify_snapshot(snapshot, expectation) is snapshot


def test_verify_snapshot_rejects_cross_user() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="different user"):
        _resolver().verify_snapshot(snapshot, _expectation(snapshot, user_key="someone-else"))


def test_verify_snapshot_rejects_cross_task() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="different task"):
        _resolver().verify_snapshot(
            snapshot, _expectation(snapshot, task_id="another-task")
        )


def test_verify_snapshot_rejects_stale_turn() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="stale"):
        _resolver().verify_snapshot(
            snapshot, _expectation(snapshot, turn_key="m6-1-hiking-backpack:turn-0")
        )


def test_verify_snapshot_rejects_count_mismatch() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="candidate count"):
        _resolver().verify_snapshot(snapshot, _expectation(snapshot, candidate_count=6))


def test_verify_snapshot_rejects_expected_ref_mismatch() -> None:
    snapshot = fx.build_snapshot()
    wrong_pool = candidate_pool_snapshot(fx.CANDIDATE_IDS[:-1]).ref
    with pytest.raises(CandidateSnapshotError, match="candidate pool reference"):
        _resolver().verify_snapshot(
            snapshot, _expectation(snapshot, candidate_pool_ref=wrong_pool)
        )


def test_verify_snapshot_rejects_expected_digest_mismatch() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="candidate-pool digest"):
        _resolver().verify_snapshot(
            snapshot, _expectation(snapshot, candidate_pool_digest="0" * 64)
        )


def test_verify_snapshot_rejects_tampered_declared_digest() -> None:
    snapshot = fx.build_snapshot()
    tampered = _tampered_snapshot(snapshot, candidate_pool_digest="0" * 64)
    with pytest.raises(CandidateSnapshotError, match="recomputed M5 digest"):
        _resolver().verify_snapshot(tampered)


def test_verify_snapshot_rejects_invalid_metadata_identity() -> None:
    snapshot = fx.build_snapshot()
    candidates = list(snapshot.candidates)
    foreign = ProductMetadata(parent_asin="B0DIFFERENT", source="test")
    candidates[0] = candidates[0].model_copy(update={"metadata": foreign})
    tampered = _tampered_snapshot(snapshot, candidates=tuple(candidates))
    with pytest.raises(CandidateSnapshotError, match="different parent_asin"):
        _resolver().verify_snapshot(tampered)


def test_verify_snapshot_rejects_duplicate_candidates_in_tampered_snapshot() -> None:
    snapshot = fx.build_snapshot()
    duplicated = (snapshot.candidates[0], snapshot.candidates[0])
    tampered = _tampered_snapshot(snapshot, candidates=duplicated)
    with pytest.raises(CandidateSnapshotError, match="duplicate identities"):
        _resolver().verify_snapshot(tampered)


def test_verify_snapshot_rejects_pad_in_tampered_snapshot() -> None:
    snapshot = fx.build_snapshot()
    candidates = list(snapshot.candidates)
    candidates[2] = candidates[2].model_copy(update={"item_id": 0})
    tampered = _tampered_snapshot(snapshot, candidates=tuple(candidates))
    with pytest.raises(CandidateSnapshotError, match="PAD|positive"):
        _resolver().verify_snapshot(tampered)


def test_verify_snapshot_rejects_non_snapshot_input() -> None:
    with pytest.raises(CandidateSnapshotError, match="CandidateSnapshot"):
        _resolver().verify_snapshot(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Shared eligibility projection
# --------------------------------------------------------------------------- #


def test_apply_eligibility_projects_trusted_decisions() -> None:
    snapshot = fx.build_snapshot()
    original = snapshot.candidates[0].eligibility
    decisions: dict[str, object] = {
        "B0M6HIKE01": (EligibilityState.INELIGIBLE, "reassessed: price 129.99 > current budget"),
        "B0M6HIKE04": EligibilityState.VERIFIED_ELIGIBLE,
        "B0M6HIKE06": ("unresolved", "still no catalogue metadata"),
    }
    projected = _resolver().apply_eligibility(snapshot, decisions)  # type: ignore[arg-type]
    joined = {candidate.candidate_id: candidate for candidate in projected.candidates}

    assert projected.candidate_ids == snapshot.candidate_ids
    assert joined["B0M6HIKE01"].eligibility is EligibilityState.INELIGIBLE
    assert "reassessed" in (joined["B0M6HIKE01"].eligibility_detail or "")
    assert joined["B0M6HIKE04"].eligibility is EligibilityState.VERIFIED_ELIGIBLE
    assert joined["B0M6HIKE04"].eligibility_detail is None
    assert joined["B0M6HIKE06"].eligibility is EligibilityState.UNRESOLVED
    # The projection never mutates the frozen input snapshot.
    assert snapshot.candidates[0].eligibility is original
    assert snapshot.candidates[0].eligibility_detail != joined["B0M6HIKE01"].eligibility_detail
    # Only the eligibility view changes; identity and verification material are intact.
    assert projected.candidate_pool_digest == snapshot.candidate_pool_digest
    assert projected.ranking_snapshot_digest == snapshot.ranking_snapshot_digest
    assert _resolver().verify_snapshot(projected) is projected


def test_apply_eligibility_rejects_foreign_decision() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="outside the snapshot"):
        _resolver().apply_eligibility(
            snapshot, {"B0NOTINPOOL": EligibilityState.VERIFIED_ELIGIBLE}
        )


def test_apply_eligibility_rejects_invalid_state_and_oversized_detail() -> None:
    snapshot = fx.build_snapshot()
    with pytest.raises(CandidateSnapshotError, match="unknown eligibility state"):
        _resolver().apply_eligibility(snapshot, {"B0M6HIKE01": "maybe"})
    with pytest.raises(CandidateSnapshotError, match="exceeds the allowed length"):
        _resolver().apply_eligibility(
            snapshot, {"B0M6HIKE01": (EligibilityState.INELIGIBLE, "x" * 400)}
        )
    with pytest.raises(CandidateSnapshotError, match="mapping"):
        _resolver().apply_eligibility(snapshot, object())


def test_apply_eligibility_with_no_decisions_is_identity() -> None:
    snapshot = fx.build_snapshot()
    projected = _resolver().apply_eligibility(snapshot, {})
    assert projected == snapshot
