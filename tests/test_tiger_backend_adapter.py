"""Adapter behaviour tests for the Step-2.3 handoff bridge.

Covers the gates the specification names for Step 2.3:

* **G2** the materialiser writes an item-id-only handoff and no canonical identity;
* **G3** seven negative refusals at the contract boundary;
* **G6** the ``case_ids`` receipt, shard offsets and totals;
* **G8** the stub stream feeds the **unchanged** frozen evaluator;
* **G9** the identity round-trip and the shuffled-``item_ids`` refusal;
* **G10** PAD is structurally impossible as a SID.

Everything runs on synthetic data, so the suite needs no catalogue artifact, no model, no GPU
and no backend virtual environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.backends.tiger_backend import (  # noqa: E402
    CONTRACT_VERSION,
    PAD_SENTINEL,
    BackendProcessError,
    ContractViolation,
    SIDResolver,
    TigerBackendAdapter,
    build_token_layout,
    collision_audit,
    frontier_stats,
    required_frontier,
)

HASHES = {"mappings_sha256": "a" * 64, "sequences_sha256": "b" * 64, "products_sha256": "c" * 64}
NUM_ITEMS = 12


def make_handoff(root: Path, *, num_items: int = NUM_ITEMS, cohort: int = 4) -> TigerBackendAdapter:
    """Materialise a tiny, complete, target-free handoff on disk.

    The catalogue record is threaded through explicitly so materialisation never re-reads a
    half-written directory: the manifest that guards a *read* is written once, at the end.
    """
    adapter = TigerBackendAdapter(root, timeout_seconds=120.0)
    catalogue = adapter.materialise_catalogue(
        item_ids=list(range(1, num_items + 1)), num_users=3, **HASHES
    )
    adapter.materialise_products_text(
        [f"item {index}" for index in range(1, num_items + 1)], catalogue=catalogue
    )
    adapter.materialise_train_exposure(
        train_histories=[[1, 2, 3, 4], [2, 3, 5], [1, 4, 6, 7, 8]], catalogue=catalogue
    )
    adapter.materialise_eval_cohort(
        test_histories=[[1, 2, 3, 4, 5, 6][: 2 + index] for index in range(cohort)],
        catalogue=catalogue,
    )
    adapter.write_score_request(status="APPROXIMATE", batch_size=2, sample_items=5)
    adapter.write_manifest()
    return adapter


def run_stub_stages(adapter: TigerBackendAdapter, *, status: str = "approximate") -> dict:
    """Run the four placeholder stages by subprocess, as the adapter does in production."""
    root = adapter.paths.root
    adapter.run_stage("build-features", "--catalogue", str(root), "--out", str(root / "features"))
    adapter.run_stage(
        "fit-sid",
        "--catalogue",
        str(root),
        "--features",
        str(root / "features"),
        "--out",
        str(root / "sid"),
        "--codebook-size",
        "8",
    )
    adapter.run_stage(
        "train",
        "--catalogue",
        str(root),
        "--exposure",
        str(adapter.paths.exposure),
        "--sid",
        str(root / "sid"),
        "--out",
        str(root / "ckpt"),
    )
    for name in ("semantic_ids.json", "layout.json"):
        (root / name).write_bytes((root / "sid" / name).read_bytes())
    (root / "tiger.json").write_bytes((root / "ckpt" / "tiger.json").read_bytes())
    adapter.write_manifest()
    completed = adapter.run_stage(
        "score",
        "--cohort",
        str(root),
        "--sid",
        str(root),
        "--ckpt",
        str(root),
        "--out",
        str(root),
        "--status",
        status,
        "--batch-size",
        "2",
        "--sample-items",
        "5",
    )
    return json.loads(completed.stdout.decode("utf-8"))


# --------------------------------------------------------------------------- #
# G2 - the materialiser
# --------------------------------------------------------------------------- #


def test_g2_materialise_writes_a_target_free_item_id_only_handoff(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    manifest = json.loads(adapter.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == CONTRACT_VERSION
    for name in (
        "catalogue.json",
        "catalogue_items.jsonl",
        "products_text.jsonl",
        "train_exposure.json",
        "train_exposure.jsonl",
        "eval_cohort.json",
        "eval_cohort.jsonl",
        "score_request.json",
    ):
        assert name in manifest["files"], name

    # No target-shaped key and no canonical identity in any artifact.
    for name in manifest["files"]:
        text = (tmp_path / name).read_text(encoding="utf-8")
        for forbidden in ("target", "label", "seen", "parent_asin", "validation"):
            assert forbidden not in text.lower(), f"{name} mentions {forbidden!r}"

    catalogue = adapter.read_catalogue()
    assert catalogue.num_items == NUM_ITEMS
    assert catalogue.item_ids == tuple(range(1, NUM_ITEMS + 1))
    assert catalogue.pad_id == 0 and catalogue.first_real_id == 1


def test_g2_cohort_carries_a_frontier_integer_and_never_a_seen_set(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    cohort = adapter.read_cohort()
    # Case i has a 2+i item history, all distinct, so the frontier is 20 + 2 + i.
    assert cohort.required_frontier == (22, 23, 24, 25)
    assert all(
        len(history) < frontier
        for history, frontier in zip(
            cohort.test_histories, cohort.required_frontier, strict=True
        )
    )
    payload = json.loads(adapter.paths.cohort.read_text(encoding="utf-8"))
    assert payload["required_frontier_stats"]["min"] == 22
    assert payload["required_frontier_stats"]["max"] == 25
    assert "seen" not in json.dumps(payload).lower()


def test_g2_train_exposure_declares_train_history_as_its_only_source(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    record = json.loads(adapter.paths.exposure_record.read_text(encoding="utf-8"))
    assert record["field_source"] == "EvaluationCase.train_history"
    assert record["protocol_version"] == "agentrecx.eval_protocol.v1"
    rows = [json.loads(line) for line in adapter.paths.exposure.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["items"] == [1, 2, 3, 4]


def test_g2_frontier_stats_reports_percentiles_and_refuses_a_short_frontier() -> None:
    stats = frontier_stats([20, 21, 22, 23, 24], k_max=20)
    assert (stats.minimum, stats.maximum, stats.cases) == (20, 24, 5)
    assert stats.p95 >= stats.p99 - 1
    with pytest.raises(ContractViolation):
        frontier_stats([19], k_max=20)


# --------------------------------------------------------------------------- #
# G3 - negative refusals
# --------------------------------------------------------------------------- #


def test_g3_a_pad_item_id_in_the_catalogue_is_refused(tmp_path: Path) -> None:
    adapter = TigerBackendAdapter(tmp_path, timeout_seconds=30.0)
    with pytest.raises(ContractViolation):
        adapter.materialise_catalogue(item_ids=[0, 1, 2], num_users=1, **HASHES)


def test_g3_an_item_id_gap_is_refused(tmp_path: Path) -> None:
    adapter = TigerBackendAdapter(tmp_path, timeout_seconds=30.0)
    with pytest.raises(ContractViolation):
        adapter.materialise_catalogue(item_ids=[1, 2, 4], num_users=1, **HASHES)


def test_g3_a_blank_source_digest_is_refused(tmp_path: Path) -> None:
    adapter = TigerBackendAdapter(tmp_path, timeout_seconds=30.0)
    with pytest.raises(ContractViolation):
        adapter.materialise_catalogue(
            item_ids=[1, 2],
            num_users=1,
            mappings_sha256="",
            sequences_sha256="b" * 64,
            products_sha256="c" * 64,
        )


def test_g3_a_zero_in_a_cohort_history_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(ContractViolation):
        adapter.materialise_eval_cohort(
            test_histories=[[1, 0, 2]], catalogue=adapter.read_catalogue()
        )


def test_g3_a_contract_version_mismatch_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    payload = json.loads(adapter.paths.catalogue.read_text(encoding="utf-8"))
    payload["contract_version"] = "agentrecx.tiger_backend.v2"
    adapter.paths.catalogue.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    adapter.write_manifest()
    with pytest.raises(ContractViolation):
        adapter.read_catalogue()


def test_g3_a_missing_manifest_entry_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    manifest = json.loads(adapter.paths.manifest.read_text(encoding="utf-8"))
    del manifest["files"]["catalogue.json"]
    adapter.paths.manifest.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ContractViolation):
        adapter.read_catalogue()


def test_g3_a_digest_mismatch_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    adapter.paths.catalogue.write_text('{"format": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ContractViolation):
        adapter.read_catalogue()


def test_g3_a_target_shaped_record_key_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(ContractViolation):
        _read_forbidden({"case_id": 0, "history": [1, 2], "target": 9})


def _read_forbidden(payload: dict) -> None:
    from recommendation.backends.tiger_backend import _assert_no_forbidden_keys

    _assert_no_forbidden_keys(payload, context="test")


def test_g3_an_out_of_span_item_id_in_an_exposure_row_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(ContractViolation):
        adapter.materialise_train_exposure(
            train_histories=[[1, NUM_ITEMS + 5]], catalogue=adapter.read_catalogue()
        )


def test_g3_a_single_item_exposure_row_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(ContractViolation):
        adapter.materialise_train_exposure(
            train_histories=[[7]], catalogue=adapter.read_catalogue()
        )


# --------------------------------------------------------------------------- #
# G6 - the case_ids receipt and shard bookkeeping
# --------------------------------------------------------------------------- #


def test_g6_the_stub_run_produces_a_valid_shard_index(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=6)
    summary = run_stub_stages(adapter)
    assert summary["stage"] == "score"
    assert summary["transport"] == "shards"
    assert summary["cases"] == 6
    index = json.loads((tmp_path / "scores" / "index.json").read_text(encoding="utf-8"))
    assert index["total_rows"] == 6 == index["cohort_size"]
    assert index["row_offsets"] == [0, 2, 4]
    assert sum(shard["rows"] for shard in index["shards"]) == 6
    batches = list(adapter.read_score_shards())
    receipts = [case_id for batch in batches for case_id in batch.case_ids]
    assert receipts == list(range(6))


def test_g6_a_reordered_receipt_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=4)
    run_stub_stages(adapter)
    batches = list(adapter.read_score_shards())
    # Reverse the first batch's receipt: the conversion to evaluator input must refuse it.
    first = batches[0]
    swapped = type(first)(
        case_ids=tuple(reversed(first.case_ids)),
        scores=first.scores,
        reachable=first.reachable,
        reachable_real_count=first.reachable_real_count,
        status=first.status,
        certification=first.certification,
        retrieval=first.retrieval,
        generation=first.generation,
    )
    with pytest.raises(ContractViolation):
        list(
            adapter.evaluation_batches(
                [swapped, *batches[1:]], targets=[1, 2, 3, 4], cohort=adapter.read_cohort()
            )
        )


def test_g6_an_incomplete_stream_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=4)
    run_stub_stages(adapter)
    batches = list(adapter.read_score_shards())
    with pytest.raises(ContractViolation):
        list(adapter.evaluation_batches(batches[:1], targets=[1, 2, 3, 4]))


# --------------------------------------------------------------------------- #
# G8 - the stub stream feeds the unchanged frozen evaluator
# --------------------------------------------------------------------------- #


def test_g8_the_adapter_output_satisfies_the_frozen_evaluator(tmp_path: Path) -> None:
    """The whole point of Step 2.3: raw scores reach ``evaluate_batched`` unmodified."""
    import torch

    from recommendation.evaluation.batched import evaluate_batched

    adapter = make_handoff(tmp_path, cohort=8)
    run_stub_stages(adapter)
    batches = list(adapter.read_score_shards())
    cohort = adapter.read_cohort()
    targets = [1 + (index % NUM_ITEMS) for index in range(cohort.cohort_size)]

    converted = list(
        adapter.evaluation_batches(batches, targets=targets, cohort=cohort)
    )
    assert sum(len(histories) for histories, _t, _s in converted) == cohort.cohort_size
    for _histories, _targets, scores in converted:
        assert isinstance(scores, torch.Tensor)
        assert scores.dtype == torch.float32
        assert scores.shape[1] == NUM_ITEMS + 1

    result = evaluate_batched(
        num_items=NUM_ITEMS, score_batches=converted, k_values=(5, 10)
    )
    report = result.report
    assert report.num_cases == cohort.cohort_size
    assert report.catalog_size == NUM_ITEMS
    for name in ("Recall", "NDCG", "HR"):
        for k in (5, 10):
            value = report.metrics[name][k]
            assert 0.0 <= value <= 1.0
    # The evaluator, not the adapter, produced these: the adapter never masks or ranks.
    assert report.mean_num_candidates > 0


def test_g8_every_score_including_the_tail_is_finite(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=4)
    run_stub_stages(adapter)
    for batch in adapter.read_score_shards():
        assert np.isfinite(batch.scores).all()
        # The PAD column is contract-only and must be exactly zero.
        assert np.all(batch.scores[:, 0] == 0.0)
        assert not batch.reachable[:, 0].any()


def test_g8_unreachable_items_sit_strictly_below_every_reachable_item(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=2)
    run_stub_stages(adapter)
    for batch in adapter.read_score_shards():
        for row in range(len(batch.case_ids)):
            mask = batch.reachable[row, 1:]
            assert mask.any()
            reachable_min = batch.scores[row, 1:][mask].min()
            tail = batch.scores[row, 1:][~mask]
            if tail.size:
                assert (tail < reachable_min).all()


def test_g8_streaming_transport_matches_the_sharded_transport(tmp_path: Path) -> None:
    """The stream path and the shard path must produce identical scores for the same input."""
    adapter = make_handoff(tmp_path, cohort=4)
    run_stub_stages(adapter)
    sharded = list(adapter.read_score_shards())
    streamed = list(
        adapter.score_stream(status="APPROXIMATE", batch_size=2, sample_items=5)
    )
    assert [batch.case_ids for batch in streamed] == [batch.case_ids for batch in sharded]
    for left, right in zip(streamed, sharded, strict=True):
        assert np.array_equal(left.scores, right.scores), "stream and shard scores differ"
        assert np.array_equal(left.reachable, right.reachable)


def test_g8_a_backend_failure_is_reported_with_its_stderr(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(BackendProcessError) as error:
        adapter.run_stage("score", "--cohort", str(tmp_path))  # missing required flags
    assert "exit code" in str(error.value)


def test_g8_an_unknown_stage_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(ContractViolation):
        adapter.run_stage("evaluate", "--out", str(tmp_path))


def test_g8_an_empty_frontier_is_refused_rather_than_filled(tmp_path: Path) -> None:
    """A row with no reachable real item has no defined tail score, so the run must fail."""
    adapter = make_handoff(tmp_path, cohort=2)
    run_stub_stages(adapter)
    batches = list(adapter.read_score_shards())
    first = batches[0]
    hollow = type(first)(
        case_ids=first.case_ids,
        scores=first.scores,
        reachable=first.reachable,
        reachable_real_count=first.reachable_real_count,
        status=first.status,
        certification=first.certification,
        retrieval=first.retrieval,
        generation=first.generation,
    )
    # Build a batch whose first row reaches nothing at all and hand it to the validator.
    object.__setattr__(hollow, "reachable", np.zeros_like(first.reachable))
    object.__setattr__(hollow, "reachable_real_count", np.zeros_like(first.reachable_real_count))
    with pytest.raises(ContractViolation) as error:
        adapter._validate_batch(
            header={
                "format": "agentrecx.tiger.scores.v3",
                "contract_version": CONTRACT_VERSION,
                "case_ids": list(hollow.case_ids),
                "status": "APPROXIMATE",
                "reachable_real_count": [0] * len(hollow.case_ids),
            },
            scores=hollow.scores,
            reachable=hollow.reachable,
            catalogue=adapter.read_catalogue(),
            cohort=adapter.read_cohort(),
        )
    assert "no real item is reachable" in str(error.value)


def test_g8_a_nan_score_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=2)
    run_stub_stages(adapter)
    first = next(iter(adapter.read_score_shards()))
    poisoned = np.array(first.scores, copy=True)
    poisoned[0, 1] = np.nan
    with pytest.raises(ContractViolation):
        adapter._validate_batch(
            header={
                "format": "agentrecx.tiger.scores.v3",
                "contract_version": CONTRACT_VERSION,
                "case_ids": list(first.case_ids),
                "status": "APPROXIMATE",
            },
            scores=poisoned,
            reachable=first.reachable,
            catalogue=adapter.read_catalogue(),
            cohort=adapter.read_cohort(),
        )


def test_g8_a_wrong_shaped_score_matrix_is_refused(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path, cohort=2)
    run_stub_stages(adapter)
    first = next(iter(adapter.read_score_shards()))
    with pytest.raises(ContractViolation):
        adapter._validate_batch(
            header={
                "format": "agentrecx.tiger.scores.v3",
                "contract_version": CONTRACT_VERSION,
                "case_ids": list(first.case_ids),
                "status": "APPROXIMATE",
            },
            scores=np.zeros((len(first.case_ids), NUM_ITEMS), dtype=np.float32),
            reachable=first.reachable,
            catalogue=adapter.read_catalogue(),
            cohort=adapter.read_cohort(),
        )


# --------------------------------------------------------------------------- #
# G9 - identity round-trip and the shuffled-table refusal
# --------------------------------------------------------------------------- #


def test_g9_item_id_row_sid_token_round_trip() -> None:
    catalogue_ids = list(range(1, 9))
    adapter = TigerBackendAdapter("/tmp/unused", timeout_seconds=1.0)
    assignment = [[PAD_SENTINEL] * 3] + [[index, index, index] for index in range(8)]
    resolver = SIDResolver(assignment, item_ids=catalogue_ids)
    layout = build_token_layout(levels=2, codebook_size=8, dedup_levels=1)
    assert layout["per_item_tokens"] == 3 == len(assignment[1])
    for row, item_id in enumerate(catalogue_ids):
        codes = assignment[row + 1]
        tokens = _tokenise(layout, codes)
        assert len(tokens) == layout["per_item_tokens"]
        assert resolver.resolve(codes) == item_id
    assert adapter is not None


def _tokenise(layout: dict, codes: list[int]) -> tuple[int, ...]:
    """Tokenise like the backend does, refusing a negative digit rather than aliasing it."""
    tokens: list[int] = []
    for index, digit in enumerate(codes):
        if digit < 0:
            raise ContractViolation("a negative digit (the PAD sentinel) is not tokenisable")
        size = layout["codebook_size"] if index < layout["levels"] else layout["dedup_vocab_size"]
        if digit >= size:
            raise ContractViolation(f"digit {digit} is outside [0, {size})")
        tokens.append(layout["level_offsets"][index] + digit)
    return tuple(tokens)


def test_g9_a_shuffled_item_id_table_is_detected(tmp_path: Path) -> None:
    """The backend must never infer ``item_id = row + 1``; a permutation must be visible."""
    adapter = make_handoff(tmp_path)
    rows = adapter.paths.catalogue_items.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in rows]
    records[0], records[1] = records[1], records[0]  # rows still ascend, ids permuted
    adapter.paths.catalogue_items.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    adapter.write_manifest()
    with pytest.raises(ContractViolation):
        adapter.read_catalogue()


def test_g9_a_non_contiguous_backend_row_is_detected(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    rows = [json.loads(line) for line in adapter.paths.catalogue_items.read_text().splitlines()]
    rows[1]["backend_row"] = 7
    adapter.paths.catalogue_items.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows), encoding="utf-8"
    )
    adapter.write_manifest()
    with pytest.raises(ContractViolation):
        adapter.read_catalogue()


# --------------------------------------------------------------------------- #
# G10 - PAD is structurally impossible as a SID
# --------------------------------------------------------------------------- #


def test_g10_the_pad_row_is_a_sentinel_and_is_ignored_by_the_resolver() -> None:
    layout = build_token_layout(levels=3, codebook_size=8, dedup_levels=1)
    pad_row = [PAD_SENTINEL] * layout["per_item_tokens"]
    assert all(value < 0 for value in pad_row)
    # Tokenising the sentinel is an error, not a silently aliased token.
    with pytest.raises(ContractViolation):
        _tokenise(layout, pad_row)

    # Row 0 is excluded from the resolver's table, so a sentinel lookup is a closed miss.
    resolver = SIDResolver([pad_row, [0, 0, 0, 0]], item_ids=[1])
    assert resolver.resolve(pad_row) is None
    assert resolver.resolve([0, 0, 0, 0]) == 1
    assert resolver.collision_sizes() == (1,)


def test_g10_real_digits_are_never_negative() -> None:
    """Only the reserved row may carry the sentinel."""
    with pytest.raises(ContractViolation):
        SIDResolver(
            [[PAD_SENTINEL] * 3, [PAD_SENTINEL, 0, 0]],
            item_ids=[1],
        )


def test_g10_pad_is_never_a_catalogue_item(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    catalogue = adapter.read_catalogue()
    assert 0 not in catalogue.item_ids
    with pytest.raises(ContractViolation):
        catalogue.row_of(0)


def test_g10_required_frontier_refuses_a_pad_history() -> None:
    with pytest.raises(ContractViolation):
        required_frontier([1, 2, 0])


def test_g10_collision_audit_keeps_both_blocks_and_flags_overflow() -> None:
    """Six items, two pre-dedup codes: the audit keeps the quality signal *and* the guarantee.

    ``pre_dedup`` reports how well the codes separate items (one collision group of five), and
    ``post_dedup`` reports the uniqueness the resolver relies on: the dedup digit makes every
    item distinct, so there are six distinct SIDs and no collision group remaining.
    """
    audit = collision_audit([(1, 1)] * 5 + [(2, 2)], dedup_levels=1, dedup_vocab_size=4)
    assert audit["pre_dedup"]["largest_group"] == 5
    assert audit["pre_dedup"]["collision_groups"] == 1
    assert audit["pre_dedup"]["distinct_sids"] == 2
    assert audit["post_dedup"]["collision_groups"] == 0
    assert audit["post_dedup"]["distinct_sids"] == 6
    # The overflow check reads the PRE-dedup group, which is the quantizer's own property.
    assert audit["largest_group_le_dedup_vocab"] is False
    assert audit["overflow_refusal_triggered"] is True


def test_g10_the_dedup_overflow_refuses_the_build(tmp_path: Path) -> None:
    """A dedup level that cannot disambiguate a group must refuse, never wrap or share.

    A 64-item catalogue with an 8-code book yields collision groups of eight, so a dedup width
    of 1 cannot disambiguate them and the build must refuse with the documented reason.
    """
    adapter = make_handoff(tmp_path, num_items=64)
    adapter.run_stage(
        "build-features", "--catalogue", str(tmp_path), "--out", str(tmp_path / "features")
    )
    with pytest.raises(BackendProcessError) as error:
        adapter.run_stage(
            "fit-sid",
            "--catalogue",
            str(tmp_path),
            "--features",
            str(tmp_path / "features"),
            "--out",
            str(tmp_path / "tight"),
            "--codebook-size",
            "8",
            "--dedup-vocab-size",
            "1",
        )
    assert "exceeds dedup_vocab_size" in str(error.value)


def test_g10_a_dead_code_level_is_refused_as_a_catastrophic_collapse(tmp_path: Path) -> None:
    """A level with more than 20% dead codes is a hard stop, not a reportable observation.

    A codebook wider than the catalogue genuinely *has* dead codes - most codes are never
    selected because there are not enough items to select them - which is exactly the
    catastrophic-collapse case the gate exists for.
    """
    adapter = make_handoff(tmp_path)
    adapter.run_stage(
        "build-features", "--catalogue", str(tmp_path), "--out", str(tmp_path / "features")
    )
    with pytest.raises(BackendProcessError) as error:
        adapter.run_stage(
            "fit-sid",
            "--catalogue",
            str(tmp_path),
            "--features",
            str(tmp_path / "features"),
            "--out",
            str(tmp_path / "wide"),
            "--codebook-size",
            "100",
            "--dedup-levels",
            "0",
        )
    assert "catastrophic collapse" in str(error.value)


# --------------------------------------------------------------------------- #
# Reproducibility and the module's own contract
# --------------------------------------------------------------------------- #


def test_the_stub_run_is_bit_identical_on_a_second_invocation(tmp_path: Path) -> None:
    first = make_handoff(tmp_path / "a")
    run_stub_stages(first)
    second = make_handoff(tmp_path / "b")
    run_stub_stages(second)
    left = sorted((tmp_path / "a" / "scores").glob("part-*.npz"))
    right = sorted((tmp_path / "b" / "scores").glob("part-*.npz"))
    assert len(left) == len(right) and left
    for one, two in zip(left, right, strict=True):
        with np.load(one, allow_pickle=False) as a, np.load(two, allow_pickle=False) as b:
            assert np.array_equal(a["scores"], b["scores"])
            assert np.array_equal(a["reachable"], b["reachable"])


def test_reading_semantic_ids_is_not_implemented_in_step_2_3(tmp_path: Path) -> None:
    """Step 2.3 has no stage-2 producer; the method says so rather than guessing."""
    adapter = make_handoff(tmp_path)
    with pytest.raises(NotImplementedError):
        adapter.read_semantic_ids()


def test_product_text_is_keyed_by_item_id_only(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    rows = [json.loads(line) for line in adapter.paths.products_text.read_text().splitlines()]
    assert rows[0] == {"item_id": 1, "text": "item 1"}
    assert all(set(row) == {"item_id", "text"} for row in rows)


def test_products_text_length_must_match_the_catalogue(tmp_path: Path) -> None:
    adapter = make_handoff(tmp_path)
    with pytest.raises(ContractViolation):
        adapter.materialise_products_text(
            ["only one"], catalogue=adapter.read_catalogue()
        )


def test_the_adapter_accepts_a_separate_backend_interpreter(tmp_path: Path) -> None:
    """Production points ``backend_python`` at the backend venv; a test can point it anywhere."""
    adapter = TigerBackendAdapter(tmp_path, backend_python="/nonexistent/python")
    with pytest.raises((BackendProcessError, OSError)):
        adapter.run_stage("build-features", "--catalogue", str(tmp_path), "--out", str(tmp_path))


def test_materialize_cli_help_is_available() -> None:
    """The materialiser must be runnable as a module without touching any artifact."""
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.materialize_tiger_backend", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"--mode" in completed.stdout
