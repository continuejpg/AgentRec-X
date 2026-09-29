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


# --------------------------------------------------------------------------- #
# the queried_sources accounting repair: accounting only, nothing else moves
# --------------------------------------------------------------------------- #

ALL_FOUR = ("catalog_search", "history", "similar_item", "two_tower")


def _tiny_harness(tmp_path: Any = None):
    """A three-user harness over synthetic heads that cover all four sources."""
    from experiments.m4_harness import M4HarnessFactory, M4HeadTable, M4IdentityMap
    from experiments.m4_policies import FixedTraversalPolicy

    ids = [f"B{i:04d}" for i in range(1, NUM_ITEMS + 1)]
    identity = M4IdentityMap({v: i + 1 for i, v in enumerate(ids)}, [None, *ids])
    heads = {
        source: {
            uid: tuple(
                (ids[(uid + rank) % NUM_ITEMS], rank + 1, 1.0 / (rank + 1)) for rank in range(4)
            )
            for uid in USERS
        }
        for source in SOURCES
    }
    table = M4HeadTable(
        heads=heads,
        score_kinds={s_: "k" for s_ in SOURCES},
        queries={CandidateSource.CATALOG_SEARCH: {uid: "camping lantern" for uid in USERS}},
        indexed_records=NUM_ITEMS,
        search_fields=("title",),
    )
    return M4HarnessFactory(
        identity_map=identity, table=table, policy_factory=FixedTraversalPolicy
    )


def test_queried_sources_names_every_source_that_was_dispatched() -> None:
    """The defect: a source that ran but carries no consumption flag was omitted.

    ``tool_calls`` proves four sources were asked; ``queried_sources`` must therefore name four.
    """
    factory = _tiny_harness()
    harness = factory(type("C", (), {"user_int_id": USERS[0], "test_history": (1, 2), "test_target": 3})())
    result = harness.controller.run("recommend something", harness.trusted_history)
    from experiments.m4_benchmark import _behaviour, _ranking_from_ledger

    item2id = {f"B{i:04d}": i for i in range(1, NUM_ITEMS + 1)}
    ranking = _ranking_from_ledger(harness.ledger, item2id)
    record = _behaviour(result, harness.ledger, ranking, harness.plane)

    assert record["tool_calls"] == len(SOURCES)
    assert record["queried_sources"] == sorted(ALL_FOUR), (
        f"queried_sources={record['queried_sources']} but {record['tool_calls']} sources ran"
    )
    _ = result


def test_the_dispatch_record_equals_the_ledger_sources_for_a_full_traversal() -> None:
    """Two independent signals must agree: what was dispatched, and what reached the ledger."""
    factory = _tiny_harness()
    harness = factory(type("C", (), {"user_int_id": USERS[0], "test_history": (1, 2), "test_target": 3})())
    harness.controller.run("recommend something", harness.trusted_history)
    dispatched = sorted(s.value for s in harness.plane.queried_sources)
    present = sorted(s.value for s in harness.ledger.sources_present())
    assert dispatched == present == sorted(ALL_FOUR)


def test_accounting_is_invariant_to_reruns() -> None:
    """Accounting must be a pure function of one run: repeating it changes nothing.

    The fixed arm is deterministic, so every scientific output is identical across runs; only a
    broken accounting path could differ. This is the invariance evidence for the repair.
    """
    def once() -> dict[str, Any]:
        factory = _tiny_harness()
        harness = factory(
            type("C", (), {"user_int_id": USERS[1], "test_history": (2, 3), "test_target": 4})()
        )
        result = harness.controller.run("recommend something", harness.trusted_history)
        item2id = {f"B{i:04d}": i for i in range(1, NUM_ITEMS + 1)}
        from experiments.m4_benchmark import _behaviour, _ranking_from_ledger

        ranking = _ranking_from_ledger(harness.ledger, item2id)
        record = _behaviour(result, harness.ledger, ranking, harness.plane)
        return {
            "ranking": tuple(ranking),
            "ranking_size": record["ranking_size"],
            "action_sequence": tuple(record["action_sequence"]),
            "tool_calls": record["tool_calls"],
            "steps": record["steps"],
            "ledger_sources": tuple(record["sources_present"]),
            "candidates": record["candidates"],
            "grounded": record["grounded_candidates"],
            "queried_sources": tuple(record["queried_sources"]),
        }

    first, second = once(), once()
    # Scientific outputs: candidates, ranking, trajectory, ledger contents.
    for field in ("ranking", "ranking_size", "action_sequence", "tool_calls", "steps",
                  "ledger_sources", "candidates", "grounded"):
        assert first[field] == second[field], f"{field} is not stable across runs"
    # The accounting field is stable too, and now complete.
    assert first["queried_sources"] == second["queried_sources"] == tuple(sorted(ALL_FOUR))


def test_a_legitimately_empty_source_is_still_reported_as_queried() -> None:
    """An empty result must remain distinguishable from 'never consulted'."""
    from experiments.m4_harness import M4HarnessFactory, M4HeadTable, M4IdentityMap
    from experiments.m4_policies import FixedTraversalPolicy

    ids = [f"B{i:04d}" for i in range(1, NUM_ITEMS + 1)]
    identity = M4IdentityMap({v: i + 1 for i, v in enumerate(ids)}, [None, *ids])
    heads = {
        source: {
            uid: (
                ()
                if source is CandidateSource.SIMILAR_ITEM
                else tuple((ids[(uid + r) % NUM_ITEMS], r + 1, 1.0 / (r + 1)) for r in range(4))
            )
            for uid in USERS
        }
        for source in SOURCES
    }
    table = M4HeadTable(
        heads=heads,
        score_kinds={s_: "k" for s_ in SOURCES},
        queries={CandidateSource.CATALOG_SEARCH: {uid: "q" for uid in USERS}},
        indexed_records=NUM_ITEMS,
        search_fields=("title",),
    )
    factory = M4HarnessFactory(
        identity_map=identity, table=table, policy_factory=FixedTraversalPolicy
    )
    harness = factory(
        type("C", (), {"user_int_id": USERS[0], "test_history": (1, 2), "test_target": 3})()
    )
    result = harness.controller.run("recommend something", harness.trusted_history)
    from experiments.m4_benchmark import _behaviour, _ranking_from_ledger

    item2id = {f"B{i:04d}": i for i in range(1, NUM_ITEMS + 1)}
    ranking = _ranking_from_ledger(harness.ledger, item2id)
    record = _behaviour(result, harness.ledger, ranking, harness.plane)

    assert "similar_item" in record["queried_sources"], "an empty source was hidden"
    assert "similar_item" not in record["sources_present"], "an empty source reached the ledger"
    assert record["tool_calls"] == len(SOURCES)
