"""D7 fail-closed tests: the M4 runner must refuse to produce a result on bad input.

The preregistration makes the cohort, the head table, the source universe, the protocol hash and
the statistics non-negotiable. A runner that warned and continued past a mismatch would still emit
plausible-looking numbers, which is the failure mode these tests exist to prevent.

Everything here is offline: synthetic heads and the real frozen primitives, no provider, no
20,000-user cohort.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest

from recommendation.control.arguments import CandidateSource
from recommendation.evaluation.metrics import DEFAULT_K_VALUES

from experiments.m4_benchmark import (
    M4ContractViolation,
    M4CohortRef,
    assert_preregistration_hash,
    cohort_identity_sha256,
    exact_mcnemar,
    load_frozen_heads,
    paired_statistics,
)
from experiments.m4_policies import FIXED_SOURCE_ORDER

SOURCES = (
    CandidateSource.CATALOG_SEARCH,
    CandidateSource.HISTORY,
    CandidateSource.SIMILAR_ITEM,
    CandidateSource.TWO_TOWER,
)
USERS = (11, 22, 33)
TARGETS = (5, 6, 7)
NUM_ITEMS = 20


def _cohort(users: Sequence[int] = USERS, targets: Sequence[int] = TARGETS) -> M4CohortRef:
    cases = tuple(
        type("C", (), {"user_int_id": uid, "test_target": target})() for uid, target in zip(users, targets)
    )
    return M4CohortRef(
        cases=cases,
        user_int_ids=tuple(users),
        target_ids=tuple(targets),
        identity_sha256=cohort_identity_sha256(users, targets),
    )


def _write_heads(
    path: Any,
    *,
    user_ids: Sequence[int],
    lengths: dict[CandidateSource, list[int]],
    provenance: Any | None = None,
) -> None:
    arrays: dict[str, Any] = {"user_int_ids": np.asarray(user_ids, dtype=np.int64)}
    for source in SOURCES:
        per_user = lengths[source]
        flat_ids: list[str] = []
        flat_ranks: list[int] = []
        flat_scores: list[float] = []
        for uid, count in zip(user_ids, per_user):
            for rank in range(count):
                item = ((uid + rank) % NUM_ITEMS) + 1
                flat_ids.append(f"B{item:04d}")
                flat_ranks.append(rank + 1)
                flat_scores.append(1.0 / (rank + 1))
        arrays[f"{source.value}_lengths"] = np.asarray(per_user, dtype=np.int32)
        arrays[f"{source.value}_identities"] = np.asarray(flat_ids, dtype=object)
        arrays[f"{source.value}_ranks"] = np.asarray(flat_ranks, dtype=np.int32)
        arrays[f"{source.value}_scores"] = np.asarray(flat_scores, dtype=np.float64)
        arrays[f"{source.value}_queries"] = np.asarray(
            ["q" for _ in user_ids], dtype=object
        )
    np.savez_compressed(path, **arrays)
    if provenance is not None:
        import hashlib
        import json

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance.write_text(
            json.dumps(
                {
                    "sha256": digest,
                    "score_kinds": {s.value: "k" for s in SOURCES},
                    "depth": 100,
                }
            ),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# protocol hash gate
# --------------------------------------------------------------------------- #


def test_the_preregistration_hash_is_the_frozen_one() -> None:
    assert assert_preregistration_hash().startswith("7a18ea72")


def test_a_changed_preregistration_is_refused() -> None:
    with pytest.raises(M4ContractViolation, match="preregistration sha256"):
        assert_preregistration_hash(expected="0" * 64)


# --------------------------------------------------------------------------- #
# statistics reproduce the inherited protocol
# --------------------------------------------------------------------------- #


def test_mcnemar_reproduces_the_m3_recorded_value() -> None:
    """M3 recorded 28 gained / 182 lost with p = 7.516255230127073e-29."""
    assert exact_mcnemar(28, 182) == pytest.approx(7.516255230127073e-29, rel=1e-12)


def test_paired_statistics_delta_convention_is_adaptive_minus_fixed() -> None:
    """A strictly better adaptive arm must produce a positive delta."""
    fixed = [(1, 2, 3)] * 4
    adaptive = [(5, 6, 7)] * 4
    targets = [5] * 4
    stats = paired_statistics(arm_a=fixed, arm_b=adaptive, targets=targets, resamples=200)
    assert stats["delta_convention"] == "adaptive - fixed"
    assert stats["per_k"]["10"]["ndcg_delta"] > 0
    assert stats["per_k"]["10"]["recall_delta"] > 0
    assert stats["verdict"] in {"SUPPORTED", "DIRECTIONAL ONLY"}


def test_paired_statistics_are_symmetric_under_arm_swap() -> None:
    """Swapping the arms flips the sign of every delta and swaps gained/lost."""
    a = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    b = [(2, 3, 4), (4, 5, 6), (9, 8, 7)]
    targets = [2, 4, 9]
    forward = paired_statistics(arm_a=a, arm_b=b, targets=targets, resamples=200)
    backward = paired_statistics(arm_a=b, arm_b=a, targets=targets, resamples=200)
    for k in forward["per_k"]:
        assert forward["per_k"][k]["gained"] == backward["per_k"][k]["lost"]
        assert forward["per_k"][k]["ndcg_delta"] == pytest.approx(
            -backward["per_k"][k]["ndcg_delta"]
        )


def test_paired_statistics_refuse_unequal_arms() -> None:
    with pytest.raises(M4ContractViolation, match="equal-length"):
        paired_statistics(arm_a=[(1,)], arm_b=[(1,), (2,)], targets=[1, 2], resamples=10)


def test_the_frozen_k_values_are_used() -> None:
    assert tuple(DEFAULT_K_VALUES) == (5, 10, 20)


# --------------------------------------------------------------------------- #
# head-table gates
# --------------------------------------------------------------------------- #


def test_a_missing_head_artifact_is_refused(tmp_path: Any) -> None:
    with pytest.raises(M4ContractViolation, match="missing assembled heads"):
        load_frozen_heads(
            _cohort(), path=tmp_path / "absent.npz", verify_provenance=False
        )


def test_a_wrong_user_order_is_refused(tmp_path: Any) -> None:
    """Order is part of the cohort's identity; a permutation must not be accepted."""
    path = tmp_path / "heads.npz"
    _write_heads(
        path,
        user_ids=[33, 22, 11],
        lengths={s: [3, 3, 3] for s in SOURCES},
    )
    with pytest.raises(M4ContractViolation, match="user order"):
        load_frozen_heads(_cohort(), path=path, verify_provenance=False)


def test_a_missing_user_is_refused(tmp_path: Any) -> None:
    path = tmp_path / "heads.npz"
    _write_heads(path, user_ids=[11, 22], lengths={s: [3, 3] for s in SOURCES})
    with pytest.raises(M4ContractViolation, match="holds 2 users"):
        load_frozen_heads(_cohort(), path=path, verify_provenance=False)


def test_a_duplicate_user_is_refused(tmp_path: Any) -> None:
    path = tmp_path / "heads.npz"
    _write_heads(path, user_ids=[11, 11, 33], lengths={s: [3, 3, 3] for s in SOURCES})
    with pytest.raises(M4ContractViolation, match="duplicate"):
        load_frozen_heads(_cohort(), path=path, verify_provenance=False)


def test_a_missing_source_is_refused(tmp_path: Any) -> None:
    """Dropping a source must fail: it would silently shrink the frozen universe."""
    path = tmp_path / "heads.npz"
    arrays = {"user_int_ids": np.asarray(USERS, dtype=np.int64)}
    for source in SOURCES[:-1]:  # leave TWO_TOWER out
        arrays[f"{source.value}_lengths"] = np.asarray([3, 3, 3], dtype=np.int32)
        arrays[f"{source.value}_identities"] = np.asarray(["B0001"] * 9, dtype=object)
        arrays[f"{source.value}_ranks"] = np.asarray([1, 2, 3] * 3, dtype=np.int32)
        arrays[f"{source.value}_scores"] = np.asarray([1.0, 0.5, 0.33] * 3, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    with pytest.raises(M4ContractViolation, match="no arrays for source"):
        load_frozen_heads(_cohort(), path=path, verify_provenance=False)


def test_a_head_deeper_than_the_frozen_depth_is_refused(tmp_path: Any) -> None:
    path = tmp_path / "heads.npz"
    _write_heads(path, user_ids=list(USERS), lengths={s: [101, 3, 3] for s in SOURCES})
    with pytest.raises(M4ContractViolation, match="exceeds the frozen depth"):
        load_frozen_heads(_cohort(), path=path, verify_provenance=False)


def test_a_hash_mismatch_against_provenance_is_refused(tmp_path: Any) -> None:
    path = tmp_path / "heads.npz"
    provenance = tmp_path / "prov.json"
    _write_heads(
        path,
        user_ids=list(USERS),
        lengths={s: [3, 3, 3] for s in SOURCES},
        provenance=provenance,
    )
    import json

    payload = json.loads(provenance.read_text())
    payload["sha256"] = "0" * 64
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    import experiments.m4_benchmark as B

    original = B.HEADS_PROVENANCE
    B.HEADS_PROVENANCE = provenance
    try:
        with pytest.raises(M4ContractViolation, match="assembled heads sha256"):
            load_frozen_heads(_cohort(), path=path, verify_provenance=True)
    finally:
        B.HEADS_PROVENANCE = original


def test_a_well_formed_head_table_loads(tmp_path: Any) -> None:
    """The positive control: a correct artifact must load and carry all four sources."""
    path = tmp_path / "heads.npz"
    provenance = tmp_path / "prov.json"
    _write_heads(
        path,
        user_ids=list(USERS),
        lengths={s: [3, 2, 1] for s in SOURCES},
        provenance=provenance,
    )
    import experiments.m4_benchmark as B

    original = B.HEADS_PROVENANCE
    B.HEADS_PROVENANCE = provenance
    try:
        table = load_frozen_heads(_cohort(), path=path)
    finally:
        B.HEADS_PROVENANCE = original
    assert set(table.heads) == set(SOURCES)
    assert table.heads[CandidateSource.HISTORY][11]
    assert table.users(CandidateSource.HISTORY) == 3


# --------------------------------------------------------------------------- #
# cohort identity hashing
# --------------------------------------------------------------------------- #


def test_cohort_identity_hash_covers_identity_and_order() -> None:
    base = cohort_identity_sha256((1, 2, 3), (4, 5, 6))
    assert base == cohort_identity_sha256((1, 2, 3), (4, 5, 6))
    assert base != cohort_identity_sha256((3, 2, 1), (4, 5, 6)), "order must matter"
    assert base != cohort_identity_sha256((1, 2, 3), (4, 5, 7)), "targets must matter"
    assert base != cohort_identity_sha256((1, 2, 9), (4, 5, 6)), "users must matter"


def test_the_frozen_source_order_is_the_universe() -> None:
    assert set(FIXED_SOURCE_ORDER) == set(SOURCES)
