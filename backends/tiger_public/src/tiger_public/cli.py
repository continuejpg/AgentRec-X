"""Stage CLI for the public-TIGER backend (Step 2.3 skeleton).

Four stages, matching ``docs/TIGER_BACKEND.md`` section 1::

    build-features  -> ItemFeatureArtifact     (Step 2.4 replaces the placeholder)
    fit-sid         -> SemanticIdArtifact      (Step 2.4 replaces the placeholder)
    train           -> TigerCheckpoint         (Step 2.5 replaces the placeholder)
    score           -> Iterator[ScoreBatch]    (Step 2.6 replaces the stub scorer)

**Step 2.3 implements no ML.**  The artifact stages write deterministic *placeholders* so the
process, schema, digest and score-batch boundary can be proven end to end today, and the
scoring stage emits the historical bounded stub from
:func:`tiger_public.contracts.compute_scores`.  Nothing here downloads a model, touches a GPU,
or trains anything.

Two invariants this file demonstrates:

* the backend reads only opaque ``item_id`` values - no ``parent_asin`` appears anywhere in
  this package, and no stage accepts a parameter that could carry one;
* the backend reads ``history`` and ``required_frontier`` and nothing else - there is no code
  path by which a target could reach it, because no such field is ever read.

Streaming protocol
------------------

``score --stream`` writes to stdout:

1. one JSON line: ``{"format", "contract_version", "case_ids", "status", "batch", "shape",
   "reachable_real_count", "certification", "retrieval", "generation"}``;
2. 4-byte little-endian length + the ``.npy`` bytes of ``scores`` (float32);
3. 4-byte little-endian length + the ``.npy`` bytes of ``reachable`` (bool).

All diagnostics go to stderr, so stdout stays a pure data channel.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from tiger_public.contracts import (
    CONTRACT_VERSION,
    PAD_SENTINEL,
    STUB_MODULUS,
    CatalogueHandoff,
    EvalCohortHandoff,
    ItemFeatureArtifact,
    ScoreBatch,
    SemanticIdArtifact,
    TigerCheckpoint,
    build_token_layout,
    collision_audit,
    compute_scores,
)
from tiger_public.dedup import (
    DEFAULT_DEDUP_VOCAB_SIZE,
    DedupOverflowError,
    assign_dedup_digits,
)
from tiger_public.features import (
    ENCODER_SENTENCE_T5,
    FeatureBuildOutcome,
    build_item_features,
    iter_text_batches,
)
from tiger_public.io import (
    ArtifactError,
    read_catalogue,
    read_eval_cohort,
    read_train_exposure,
    sha256_bytes,
    sha256_file,
    verify_manifest,
    write_json,
    write_jsonl,
    write_manifest,
)
from tiger_public.quantizer import (
    QuantizerConfig,
    RqVae,
    layer_utilisation,
    train_quantizer,
)
from tiger_public.scoring import SCORE_RULE, validate_score_rule

__all__ = ["main", "stage_build_features", "stage_fit_sid", "stage_score", "stage_train"]

#: Marker in every Step-2.3 artifact.  A reader can therefore tell a placeholder from a real
#: Step-2.4+ artifact without guessing, and no measurement can be mistaken for a result.
STUB_MARKER = "step-2.3-placeholder-no-ml"

#: The deterministic placeholder feature width.  Step 2.4 replaces this with the frozen text
#: encoder's real dimension (768 for sentence-t5-base).
STUB_FEATURE_DIM = 32

#: How many bytes of one streamed array precede the payload.
_LENGTH_PREFIX = struct.Struct("<I")


class CliError(RuntimeError):
    """Raised for a stage-level failure the caller must not ignore."""


# --------------------------------------------------------------------------- #
# Stage 1 - build features (placeholder)
# --------------------------------------------------------------------------- #


def stage_build_features(
    *,
    catalogue_dir: Path,
    out_dir: Path,
    encoder: str = ENCODER_SENTENCE_T5,
    batch_size: int = 256,
    dim: int | None = None,
    max_chars: int = 1000,
    device: str = "cpu",
    local_files_only: bool = False,
    revision: str | None = None,
) -> ItemFeatureArtifact:
    """Stage 1: encode ``products_text.jsonl`` into ``item_features.npy``.

    The handoff's manifest is verified first, so a modified or partially written materialisation
    is refused before any time is spent encoding.
    """
    verify_manifest(catalogue_dir, required=("catalogue.json", "products_text.jsonl"))
    catalogue = read_catalogue(catalogue_dir)
    outcome, record = build_item_features(
        products_text=catalogue_dir / "products_text.jsonl",
        out_dir=out_dir,
        encoder=encoder,
        dim=dim,
        batch_size=batch_size,
        device=device,
        max_chars=max_chars,
        expected_items=catalogue.num_items,
        local_files_only=local_files_only,
        revision=revision,
    )
    write_manifest(
        out_dir,
        extra={
            "stage": "build-features",
            "environment": environment_metadata(device=device),
            "timing": {
                "seconds": round(outcome.seconds, 3),
                "peak_allocated_bytes": outcome.peak_allocated_bytes,
            },
            "encoder_is_model": bool(record["encoder"].get("is_model")),
        },
    )
    return ItemFeatureArtifact(
        format=record["format"],
        num_items=catalogue.num_items,
        dim=int(record["dim"]),
        path=str(Path(out_dir) / "item_features.npy"),
        sha256=str(record["sha256"]),
        encoder=record["encoder"],
        empty_text_items=int(record["empty_text_items"]),
        truncated_items=int(record["truncated_items"]),
        catalogue_sha256=catalogue.mappings_sha256,
    )


# --------------------------------------------------------------------------- #
# Stage 2 - fit semantic ids (placeholder)
# --------------------------------------------------------------------------- #


def _load_features(
    features_dir: Path, *, verify: bool = True, require_manifest: bool = True
) -> tuple[np.ndarray, dict]:
    """Read ``item_features.npy`` and its metadata, refusing an inconsistent artifact.

    The array must be two-dimensional, ``float32``, hold exactly ``num_items`` rows **and no PAD
    row**, and contain no non-finite value.  Each of those is a silent-corruption path if left
    unchecked: a NaN would poison a codebook, and a PAD row would let the reserved sentinel be
    quantised.

    ``require_manifest`` defaults to **True**: a production ``fit-sid`` must be able to point at
    digest-verified features, because a directory whose manifest is absent could have been
    produced by an interrupted build.  ``--no-require-features-manifest`` restores the earlier
    optional behaviour, and exists only so pre-existing test fixtures keep working; it is
    recorded in the artifact so a relaxed run is visible.
    """
    manifest_path = features_dir / "manifest.json"
    if require_manifest and not manifest_path.is_file():
        raise ArtifactError(
            f"{features_dir} has no manifest.json; a production fit-sid requires digest-verified "
            "features. Rebuild them with build-features, or pass "
            "--no-require-features-manifest for a deliberately relaxed run."
        )
    if verify and manifest_path.is_file():
        verify_manifest(features_dir)
    record = read_json_file(features_dir / "item_features.json")
    array_path = features_dir / "item_features.npy"
    if not array_path.is_file():
        raise ArtifactError(f"missing {array_path}")
    features = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2:
        raise ArtifactError(f"item_features must be 2-D, got shape {tuple(features.shape)}")
    if features.dtype != np.float32:
        raise ArtifactError(f"item_features dtype must be float32, got {features.dtype}")
    if int(features.shape[0]) != int(record["num_items"]):
        raise ArtifactError(
            f"item_features holds {features.shape[0]} rows but its record declares "
            f"{record['num_items']}"
        )
    if int(features.shape[1]) != int(record["dim"]):
        raise ArtifactError(
            f"item_features has width {features.shape[1]} but its record declares {record['dim']}"
        )
    if bool(record.get("pad_row_present", False)):
        raise ArtifactError("item_features declares a PAD row; PAD must never be quantised")
    # Checked in slices so a full-catalogue artifact does not have to be materialised.
    for start in range(0, features.shape[0], 100_000):
        block = np.asarray(features[start : start + 100_000], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ArtifactError(
                f"item_features holds a non-finite value near row {start}; refusing to "
                "quantise it"
            )
    return features, record


def stage_fit_sid(
    *,
    catalogue_dir: Path,
    features_dir: Path,
    out_dir: Path,
    encoder: str = ENCODER_SENTENCE_T5,
    levels: int = 3,
    codebook_size: int = 256,
    latent_dim: int = 64,
    encoder_dims: Sequence[int] | None = None,
    beta: float = 0.25,
    epochs: int = 50,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    dedup_vocab_size: int = DEFAULT_DEDUP_VOCAB_SIZE,
    normalize_input: bool = True,
    revive_dead: bool = True,
    seed: int = 2026,
    device: str = "cpu",
    determinism: bool = False,
    max_items: int | None = None,
    allow_dead_codes: bool = False,
    require_features_manifest: bool = True,
    log: Any = None,
) -> SemanticIdArtifact:
    """Stage 2: train the RQ-VAE, assign codes, dedup, and audit.

    The order of the guards matters and is deliberate:

    1. the handoff and the feature artifact are both verified, including a check that the
       feature artifact was built from *this* ``products_text.jsonl``;
    2. the quantizer is trained on real item rows only;
    3. each level's dead-code fraction is checked, and more than 20 % refuses the build;
    4. the pre-dedup collision statistics are computed **before** any ordinal is assigned;
    5. the dedup pass runs, and an oversize group refuses rather than sharing an identity;
    6. only then is the artifact written.
    """
    verify_manifest(catalogue_dir, required=("catalogue.json", "products_text.jsonl"))
    catalogue = read_catalogue(catalogue_dir)
    features, feature_record = _load_features(
        features_dir, require_manifest=require_features_manifest
    )

    products_sha = sha256_file(catalogue_dir / "products_text.jsonl")
    recorded = str(feature_record.get("products_text_sha256", ""))
    if recorded != products_sha:
        raise ArtifactError(
            "item_features was built from a different products_text.jsonl: it records "
            f"{recorded[:12]}... but the handoff hashes to {products_sha[:12]}...; rebuild the "
            "features rather than mixing artifacts"
        )
    if int(feature_record["num_items"]) != catalogue.num_items:
        raise ArtifactError(
            f"item_features covers {feature_record['num_items']} items but the catalogue "
            f"declares {catalogue.num_items}"
        )

    if max_items is not None and max_items < catalogue.num_items:
        raise CliError(
            f"fit-sid must cover the whole catalogue ({catalogue.num_items} items); "
            f"--max-items {max_items} would leave items without a Semantic ID"
        )
    # A codebook wider than the catalogue cannot be filled no matter what the model learns:
    # with N items at most N distinct codes can ever be selected, so more than 20% would be
    # dead by geometry.  Refusing before training is both cheaper and more honest than letting
    # the collapse gate report a configuration error as a modelling failure.
    if int(codebook_size) > catalogue.num_items:
        raise CliError(
            f"codebook_size {codebook_size} exceeds the catalogue size {catalogue.num_items}; "
            "at most one code per item can be selected, so the level would necessarily be more "
            "than 20% dead. Reduce codebook_size or use a larger catalogue."
        )

    resolved_dims = tuple(encoder_dims or (768, 256, 128))
    config = QuantizerConfig(
        input_dim=int(features.shape[1]),
        encoder_dims=resolved_dims,
        latent_dim=int(latent_dim),
        levels=int(levels),
        codebook_size=int(codebook_size),
        beta=float(beta),
        normalize_input=bool(normalize_input),
        revive_dead=bool(revive_dead),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        seed=int(seed),
    )
    # Dedup feasibility is a property of the configuration alone, so it is checked *before*
    # training.  Two independent reasons a build cannot produce unique identities:
    #   * the dedup level cannot even address one item per code when the code space is at least
    #     as large as the catalogue (a group of 2 needs ordinal 1, which width 1 cannot express);
    #   * with fewer codes than items, a collision is guaranteed by pigeonhole.
    # Refusing here keeps a configuration error from masquerading as a modelling failure, and
    # keeps it from costing a training run.
    code_space = config.code_space
    if code_space >= catalogue.num_items and dedup_vocab_size < 2:
        raise CliError(
            f"dedup_vocab_size {dedup_vocab_size} cannot disambiguate even a two-item collision "
            f"group; widen the dedup level"
        )
    guaranteed_group = -(-catalogue.num_items // code_space)
    if guaranteed_group > dedup_vocab_size:
        raise CliError(
            f"a code space of {code_space} for {catalogue.num_items} items guarantees a "
            f"collision group of at least {guaranteed_group} (pigeonhole), which exceeds "
            f"dedup_vocab_size {dedup_vocab_size}; re-fit with a larger codebook, more levels, or "
            "a wider dedup level - a shared final Semantic ID is not an acceptable fallback"
        )
    torch.manual_seed(config.seed)
    model = RqVae(config)
    # ``features`` is a read-only memory map.  ``np.array(..., copy=True)`` makes it writable so
    # ``torch.from_numpy`` neither warns nor hands the trainer an array it could be told not to
    # write.  A C-contiguous memmap is backed by the page cache, so this is a mapped read account
    # rather than a 481 MB anonymous allocation; the trainer only ever *reads* the source
    # (``features[index]`` is an advanced-index copy), and the copy makes that structural.
    feature_matrix = np.array(features, dtype=np.float32, copy=True)
    outcome = train_quantizer(
        model,
        torch.from_numpy(feature_matrix),
        config=config,
        device=device,
        determinism=determinism,
        log=log,
    )
    final = outcome.final
    for term in ("loss", "reconstruction_loss", "quantization_loss"):
        value = float(final.get(term, float("nan")))
        if not np.isfinite(value):
            raise CliError(f"the RQ-VAE reported a non-finite {term} ({value}); refusing to write")

    codes = model.codes_for_features(torch.from_numpy(feature_matrix)).numpy()
    if codes.shape[0] != catalogue.num_items:
        raise CliError(
            f"assigned {codes.shape[0]} codes for {catalogue.num_items} items; every catalogue "
            "item must receive a Semantic ID"
        )
    utilisation = layer_utilisation(codes, codebook_size=config.codebook_size)
    collapsed = [block for block in utilisation if block["collapsed"]]
    if collapsed and not allow_dead_codes:
        first = collapsed[0]
        raise CliError(
            f"level {first['level']} has {first['dead_codes']} dead codes "
            f"(> 20% of {config.codebook_size}); that is a catastrophic collapse, not a "
            "reportable observation. Pass --allow-dead-codes to waive the stop for a "
            "SMALL-SCALE mechanism check, which is recorded in the artifact and is never a "
            "result."
        )
    dead_code_waiver = (
        {
            "waived": True,
            "collapsed_levels": [block["level"] for block in collapsed],
            "reason": (
                "explicit small-scale mechanism check; this run reports no quality metric and "
                "no comparison, and the waived stop is recorded so it cannot be silently "
                "inherited by a full run"
            ),
        }
        if collapsed
        else {"waived": False, "collapsed_levels": []}
    )

    # Pre-dedup statistics are computed on the quantizer's own codes, before any ordinal exists.
    pre_audit = _collision_blocks(codes.tolist(), dedup_levels=0, dedup_vocab_size=dedup_vocab_size)
    if pre_audit["pre_dedup"]["largest_group"] > dedup_vocab_size:
        raise CliError(
            f"largest pre-dedup collision group {pre_audit['pre_dedup']['largest_group']} "
            f"exceeds dedup_vocab_size {dedup_vocab_size}; re-fit with a larger codebook, more "
            "levels, or a wider dedup level - a shared final Semantic ID is not an acceptable "
            "fallback"
        )
    dedup = assign_dedup_digits(codes.tolist(), dedup_vocab_size=dedup_vocab_size)
    audit = _collision_blocks(
        codes.tolist(), dedup_levels=1, dedup_vocab_size=dedup_vocab_size
    )
    audit["dedup_vocab_size"] = dedup_vocab_size
    audit["dedup_scheme"] = "collision_ordinal"
    audit["first_member_zero"] = True
    audit["largest_group_le_dedup_vocab"] = (
        audit["pre_dedup"]["largest_group"] <= dedup_vocab_size
    )
    audit["overflow_refusal_triggered"] = False
    if audit["post_dedup"]["collision_groups"] != 0:
        raise CliError(
            f"the dedup pass left {audit['post_dedup']['collision_groups']} collision group(s); "
            "final identity must be unique"
        )
    if audit["post_dedup"]["distinct_sids"] != catalogue.num_items:
        raise CliError(
            f"the dedup pass produced {audit['post_dedup']['distinct_sids']} distinct SIDs for "
            f"{catalogue.num_items} items"
        )

    layout = build_token_layout(
        levels=config.levels,
        codebook_size=config.codebook_size,
        dedup_levels=1,
        dedup_vocab_size=dedup_vocab_size,
    )
    pad_row = [PAD_SENTINEL] * layout.per_item_tokens
    assignment: list[list[int]] = [list(pad_row)]
    for code, ordinal in zip(dedup.pre_codes, dedup.ordinals, strict=True):
        assignment.append([*code, ordinal])
    # Assert the sentinel invariants on the artifact that is about to be written.
    for item_id in range(1, len(assignment)):
        layout.tokenise(assignment[item_id])

    # ------------------------------------------------------------------ #
    # Transactional write.
    #
    # Every artifact goes into a staging directory whose name marks it incomplete, and the
    # directory is promoted to its final name only after the manifest has been written and the
    # manifest has been re-verified.  An interrupted run therefore leaves either an explicitly
    # incomplete `*.partial` directory or nothing - never a final directory that could be
    # mistaken for a completed run.
    # ------------------------------------------------------------------ #
    staging_dir = out_dir.parent / (out_dir.name + ".partial")
    if staging_dir.exists():
        # A previous attempt died here; its contents are by definition untrusted.
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = staging_dir / "tokenizer.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.as_dict()}, checkpoint_path)
    write_json(
        staging_dir / "tokenizer.json",
        {
            "format": "agentrecx.tiger.tokenizer.v3",
            "contract_version": CONTRACT_VERSION,
            "config": config.as_dict(),
            "item_features_sha256": feature_record["sha256"],
            "products_text_sha256": products_sha,
            "item_features_sha256_expected": feature_record["sha256"],
        },
    )
    record = {
        "format": "agentrecx.tiger.semantic_ids.v3",
        "contract_version": CONTRACT_VERSION,
        "num_items": catalogue.num_items,
        "levels": config.levels,
        "dedup_levels": 1,
        "codebook_size": config.codebook_size,
        "dedup_vocab_size": dedup_vocab_size,
        "code_space": config.code_space,
        "code_space_with_dedup": config.code_space * dedup_vocab_size,
        "pad_row": list(pad_row),
        "assignment": assignment,
        "collision_audit": audit,
        "utilisation": list(utilisation),
        "quantizer": config.as_dict(),
        "training": {
            "history": outcome.history,
            "seconds": round(outcome.seconds, 3),
            "revived_codes_total": outcome.revived_total,
            "seed": config.seed,
            "seed_info": outcome.seed_info,
            "device": outcome.device,
            "determinism_requested": bool(determinism),
            "peak_allocated_bytes": outcome.peak_allocated_bytes,
        },
        "content_features": {
            "path": "item_features/item_features.npy",
            "sha256": feature_record["sha256"],
            "encoder": feature_record["encoder"]["id"],
            "encoder_is_model": bool(feature_record["encoder"].get("is_model")),
            "dim": int(feature_record["dim"]),
        },
        "dead_code_waiver": dead_code_waiver,
        "features_manifest_required": bool(require_features_manifest),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(staging_dir / "semantic_ids.json", record)
    write_json(staging_dir / "layout.json", layout.as_dict())
    write_manifest(
        staging_dir,
        extra={
            "stage": "fit-sid",
            "environment": environment_metadata(device=device),
            "timing": {
                "seconds": round(outcome.seconds, 3),
                "peak_allocated_bytes": outcome.peak_allocated_bytes,
            },
        },
    )
    # The manifest is written last, then re-verified from disk, and only then promoted.
    promote_staging_directory(staging_dir, out_dir)
    return SemanticIdArtifact(
        format=record["format"],
        contract_version=CONTRACT_VERSION,
        num_items=catalogue.num_items,
        levels=config.levels,
        dedup_levels=1,
        codebook_size=config.codebook_size,
        dedup_vocab_size=dedup_vocab_size,
        pad_row=tuple(pad_row),
        assignment=tuple(tuple(row) for row in assignment),
        collision_audit=audit,
        utilisation=utilisation,
        content_features=record["content_features"],
    )


# --------------------------------------------------------------------------- #
# Stage 3 - train (placeholder)
# --------------------------------------------------------------------------- #


def stage_train(
    *,
    catalogue_dir: Path,
    exposure_path: Path,
    sid_dir: Path,
    out_dir: Path,
    epochs: int = 0,
) -> TigerCheckpoint:
    """Write a placeholder checkpoint that records the frozen score rule and the layout.

    No model is trained.  ``validation_used`` is ``false`` because the backend has no
    validation split **and** no code path that could compute one.
    """
    catalogue = read_catalogue(catalogue_dir)
    exposure = read_train_exposure(exposure_path.parent, catalogue=catalogue)
    layout_payload = json.loads((sid_dir / "layout.json").read_text(encoding="utf-8"))
    # The frozen rule is validated through its single definition rather than re-checked here.
    validate_score_rule(dict(SCORE_RULE))
    checkpoint = {
        "format": "agentrecx.tiger.checkpoint.v3",
        "contract_version": CONTRACT_VERSION,
        "layout": layout_payload,
        "score_rule": dict(SCORE_RULE),
        "model": {"family": STUB_MARKER, "init": "none", "params": 0},
        "training": {
            "objective": "none (Step 2.3 placeholder)",
            "epochs": epochs,
            "examples": exposure.examples,
            "labels_from": exposure.field_source,
            # An assertion, not a report field: the backend has no validation split.
            "validation_used": False,
        },
        "exposure_sha256": exposure.examples_sha256,
        "semantic_ids_sha256": sha256_file(sid_dir / "semantic_ids.json"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "tiger.json", checkpoint)
    write_manifest(out_dir, extra={"stage": "train", "stub": True})
    return TigerCheckpoint(
        format=checkpoint["format"],
        contract_version=CONTRACT_VERSION,
        layout=layout_payload,
        score_rule=dict(SCORE_RULE),
        model=checkpoint["model"],
        training=checkpoint["training"],
        exposure_sha256=exposure.examples_sha256,
        semantic_ids_sha256=checkpoint["semantic_ids_sha256"],
    )


# --------------------------------------------------------------------------- #
# Stage 4 - score (stub scorer)
# --------------------------------------------------------------------------- #


def _certification_block(
    *, required_frontier: Sequence[int], reachable: np.ndarray, method: str
) -> dict[str, Any]:
    counts = [int(value) for value in reachable[:, 1:].sum(axis=1)]
    enough = all(count >= want for count, want in zip(counts, required_frontier, strict=True))
    return {
        "status": "CERTIFIED",
        "method": method,
        "score_rule_version": int(SCORE_RULE["version"]),
        "admissible_bound": (
            "full-vocabulary log-probability prefix sum <= 0 for all continuations"
        ),
        "k_prime_field": "required_frontier",
        "certified_items_per_case": {
            "min": min(counts),
            "max": max(counts),
            "mean": round(sum(counts) / len(counts), 6),
        },
        "certificate_holds_for_all_cases": bool(enough),
        "budget_exhausted_cases": 0 if enough else 1,
    }


def iter_score_batches(
    cohort: EvalCohortHandoff,
    catalogue: CatalogueHandoff,
    *,
    batch_size: int,
    status: str,
    sample_items: int,
    seed: int,
) -> Iterator[ScoreBatch]:
    """Yield the deterministic stub scores as :class:`ScoreBatch` objects.

    The scorer is a placeholder, but everything around it is the real contract: the PAD column
    is pinned to zero and never reachable, unreachable items are exactly the complement of
    ``reachable``, ``reachable_real_count`` is emitted explicitly so the adapter can refuse an
    empty frontier without inferring anything, and ``case_ids`` is a per-batch receipt.
    """
    if batch_size < 1:
        raise CliError(f"batch_size must be >= 1, got {batch_size}")
    if status not in ("APPROXIMATE", "CERTIFIED"):
        raise CliError(f"status must be APPROXIMATE or CERTIFIED, got {status!r}")
    num_items = catalogue.num_items
    if not 1 <= sample_items <= num_items:
        raise CliError(
            f"sample-items must be in [1, {num_items}] for APPROXIMATE, got {sample_items}"
        )
    grid = np.asarray(catalogue.item_ids, dtype=np.int64)
    for start in range(0, cohort.cohort_size, batch_size):
        stop = min(start + batch_size, cohort.cohort_size)
        case_ids = np.asarray(cohort.case_ids[start:stop], dtype=np.int64)
        frontier = cohort.required_frontier[start:stop]

        raw = compute_scores(grid, case_ids)
        scores = np.zeros((len(case_ids), num_items + 1), dtype=np.float32)
        reachable = np.zeros((len(case_ids), num_items + 1), dtype=bool)
        if status == "CERTIFIED":
            scores[:, 1:] = raw.T
            reachable[:, 1:] = True
        else:
            rng = np.random.default_rng(seed)
            for row in range(len(case_ids)):
                picked = rng.choice(num_items, size=sample_items, replace=False)
                scores[row, 1 + picked] = raw[picked, row]
                reachable[row, 1 + picked] = True

        counts = reachable[:, 1:].sum(axis=1).astype(np.int32)
        certification = (
            _certification_block(
                required_frontier=frontier,
                reachable=reachable,
                method="full_catalogue_exact_stub" if status == "CERTIFIED" else "none",
            )
            if status == "CERTIFIED"
            else {}
        )
        yield ScoreBatch(
            format="agentrecx.tiger.scores.v3",
            contract_version=CONTRACT_VERSION,
            case_ids=tuple(int(value) for value in case_ids),
            scores=scores,
            reachable=reachable,
            reachable_real_count=counts,
            status=status,
            certification=certification,
            retrieval={
                "scored_share": round(float(counts.mean()) / num_items, 6),
                "expanded_nodes": 0,
                "beam": 0,
                "prefix_budget": 0,
                "scorer": STUB_MARKER,
            },
            generation={"generated": (), "invalid_generations": 0},
        )


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_stream(batches: Iterator[ScoreBatch], out) -> int:
    written = 0
    for batch in batches:
        header = {
            "format": batch.format,
            "contract_version": batch.contract_version,
            "case_ids": list(batch.case_ids),
            "status": batch.status,
            "batch": len(batch.case_ids),
            "shape": list(batch.scores.shape),
            "reachable_real_count": [int(v) for v in batch.reachable_real_count],
            "certification": dict(batch.certification),
            "retrieval": dict(batch.retrieval),
            "generation": dict(batch.generation),
        }
        out.write((json.dumps(header, sort_keys=True) + "\n").encode("utf-8"))
        for array in (batch.scores, batch.reachable):
            payload = _npy_bytes(array)
            out.write(_LENGTH_PREFIX.pack(len(payload)))
            out.write(payload)
        written += len(batch.case_ids)
    return written


def stage_score(
    *,
    cohort_dir: Path,
    sid_dir: Path,
    checkpoint_dir: Path,
    out_dir: Path | None,
    status: str,
    batch_size: int,
    sample_items: int,
    seed: int,
    stream: bool,
    stdout=None,
) -> dict[str, Any]:
    """Score the cohort, either streaming to stdout or writing sharded ``.npz`` artifacts."""
    catalogue = read_catalogue(cohort_dir)
    cohort = read_eval_cohort(cohort_dir, catalogue=catalogue)
    if not (checkpoint_dir / "tiger.json").is_file():
        raise ArtifactError(f"missing {checkpoint_dir / 'tiger.json'}")
    if not (sid_dir / "semantic_ids.json").is_file():
        raise ArtifactError(f"missing {sid_dir / 'semantic_ids.json'}")
    # The score rule must be the frozen one; a checkpoint that recorded anything else is
    # refused rather than scored under a different definition.
    import tiger_public.scoring as scoring

    payload = json.loads((checkpoint_dir / "tiger.json").read_text(encoding="utf-8"))
    scoring.validate_score_rule(payload.get("score_rule"))

    batches = iter_score_batches(
        cohort,
        catalogue,
        batch_size=batch_size,
        status=status,
        sample_items=sample_items,
        seed=seed,
    )
    started = time.perf_counter()
    if stream:
        handle = sys.stdout.buffer if stdout is None else stdout
        cases = _write_stream(batches, handle)
        return {
            "stage": "score",
            "transport": "stream",
            "cases": cases,
            "status": status,
            "batch_size": batch_size,
            "seconds": round(time.perf_counter() - started, 3),
        }

    if out_dir is None:
        raise CliError("a non-streaming score run requires --out")
    scores_dir = Path(out_dir) / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    offsets: list[int] = []
    row_offset = 0
    cases = 0
    for index, batch in enumerate(batches):
        name = f"part-{index:05d}.npz"
        np.savez_compressed(
            scores_dir / name,
            scores=batch.scores,
            reachable=batch.reachable,
            reachable_real_count=batch.reachable_real_count,
            case_ids=np.asarray(batch.case_ids, dtype=np.int64),
        )
        shards.append({"name": f"scores/{name}", "rows": len(batch.case_ids), "status": batch.status})
        offsets.append(row_offset)
        row_offset += len(batch.case_ids)
        cases += len(batch.case_ids)
    index_record = {
        "format": "agentrecx.tiger.scores_index.v3",
        "contract_version": CONTRACT_VERSION,
        "num_items": catalogue.num_items,
        "cohort_size": cohort.cohort_size,
        "batch_size": batch_size,
        "shards": shards,
        "row_offsets": offsets,
        "total_rows": row_offset,
        "status": status,
        "certification": {},
        "retrieval": {"scorer": STUB_MARKER},
    }
    write_json(scores_dir / "index.json", index_record)
    write_manifest(out_dir, extra={"stage": "score", "stub": True})
    return {
        "stage": "score",
        "transport": "shards",
        "cases": cases,
        "status": status,
        "batch_size": batch_size,
        "shards": len(shards),
        "seconds": round(time.perf_counter() - started, 3),
    }


# --------------------------------------------------------------------------- #
# Shared helpers for the ML stages
# --------------------------------------------------------------------------- #


def promote_staging_directory(staging_dir: Path, final_dir: Path) -> None:
    """Verify a staged stage-2 directory, then move it into place as a single rename.

    The verification is the point: the manifest is re-hashed from disk *before* promotion, so a
    directory that would fail a later read is refused here rather than being published.  The
    promotion itself is one ``os.replace`` on the directory, which is atomic on POSIX when the
    destination does not exist.
    """
    if not staging_dir.is_dir():
        raise CliError(f"staging directory {staging_dir} does not exist")
    required = ("tokenizer.pt", "tokenizer.json", "semantic_ids.json", "layout.json")
    missing = [name for name in required if not (staging_dir / name).is_file()]
    if missing:
        raise CliError(
            f"staging directory is incomplete: missing {missing}; refusing to promote a partial "
            "Semantic-ID artifact"
        )
    # Re-hash what was written.  A failure here means the bytes on disk are not the bytes the
    # manifest describes, so the directory must not become a final artifact.
    verify_manifest(staging_dir, required=required)
    if final_dir.exists():
        # Replace a previous complete run, but never merge into it: a stale file that the new
        # manifest does not list would otherwise survive and be read as part of this run.
        previous = final_dir.parent / (final_dir.name + ".superseded")
        if previous.exists():
            shutil.rmtree(previous)
        os.replace(final_dir, previous)
    os.replace(staging_dir, final_dir)


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object, refusing a missing file or a non-object payload."""
    if not path.is_file():
        raise ArtifactError(f"missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path.name} must hold a JSON object")
    return payload


def environment_metadata(*, device: str) -> dict[str, Any]:
    """Record what produced an artifact: the versions, the device, and the determinism posture.

    A GPU run that cannot promise bitwise reproducibility should say so here rather than imply
    it.  ``torch.use_deterministic_algorithms(warn_only=True)`` is advisory, and several CUDA
    reductions remain non-deterministic, so the flag is reported as *requested* and the note
    records the limitation instead of claiming identity.
    """
    info: dict[str, Any] = {
        "device": device,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "backend_git": _backend_git(),
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_bytes"] = int(props.total_memory)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
        else:
            info["note"] = (
                "no CUDA device visible; a run on this host is a CPU run regardless of --device"
            )
    except Exception as error:  # pragma: no cover - torch is a backend dependency
        info["torch"] = f"unavailable: {error}"
    for name in ("transformers", "sentence_transformers"):
        try:
            module = __import__(name)  # noqa: PLC0415

            info[name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[name] = "not installed"
    return info


def _backend_git() -> str:
    """The backend repository's commit, when it is inside a git work tree."""
    import subprocess  # noqa: PLC0415

    try:
        completed = subprocess.run(  # noqa: S603, S607 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.decode("utf-8", "replace").strip() or "unavailable"


def _collision_blocks(
    codes: Sequence[Sequence[int]], *, dedup_levels: int, dedup_vocab_size: int
) -> dict[str, Any]:
    """Thin delegation to the shared audit, kept here so the stage reads as one flow."""
    return collision_audit(
        codes, dedup_levels=dedup_levels, dedup_vocab_size=dedup_vocab_size
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiger_public.cli",
        description="public-TIGER backend stages (Step 2.3: no ML; all stages are placeholders)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-epoch progress")
    sub = parser.add_subparsers(dest="stage", required=True)

    features = sub.add_parser("build-features", help="stage 1: encode item text into features")
    features.add_argument("--catalogue", type=Path, required=True)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument(
        "--encoder",
        default=ENCODER_SENTENCE_T5,
        help=(
            "a sentence-transformers model id, or 'smoke' for the deterministic network-free "
            "stand-in (labelled in the artifact)"
        ),
    )
    features.add_argument("--batch-size", type=int, default=256)
    features.add_argument("--dim", type=int, default=None)
    features.add_argument("--max-chars", type=int, default=1000)
    features.add_argument("--device", default="cpu")
    features.add_argument(
        "--local-files-only",
        action="store_true",
        help="refuse to reach the network for encoder weights",
    )
    features.add_argument(
        "--encoder-revision",
        default=None,
        help=(
            "immutable HuggingFace snapshot SHA-1 to pin the weights to. A production run must "
            "pass this; the artifact records the resolved commit and refuses to claim a pin it "
            "cannot confirm."
        ),
    )

    sid = sub.add_parser("fit-sid", help="stage 2: train the RQ-VAE and assign Semantic IDs")
    sid.add_argument("--catalogue", type=Path, required=True)
    sid.add_argument("--features", type=Path, required=True)
    sid.add_argument("--out", type=Path, required=True)
    sid.add_argument("--levels", type=int, default=3)
    sid.add_argument("--codebook-size", type=int, default=256)
    sid.add_argument("--latent-dim", type=int, default=64)
    sid.add_argument("--encoder-dims", default="768,256,128")
    sid.add_argument("--beta", type=float, default=0.25)
    sid.add_argument("--epochs", type=int, default=50)
    sid.add_argument("--batch-size", type=int, default=4096)
    sid.add_argument("--learning-rate", type=float, default=1e-3)
    sid.add_argument("--dedup-vocab-size", type=int, default=DEFAULT_DEDUP_VOCAB_SIZE)
    sid.add_argument("--no-normalize-input", dest="normalize_input", action="store_false")
    sid.add_argument("--no-revive-dead", dest="revive_dead", action="store_false")
    sid.add_argument("--seed", type=int, default=2026)
    sid.add_argument("--device", default="cpu")
    sid.add_argument(
        "--no-require-features-manifest",
        dest="require_features_manifest",
        action="store_false",
        help=(
            "accept a features directory without a manifest.json (relaxed; recorded in the "
            "artifact). A production run must not pass this."
        ),
    )
    sid.add_argument(
        "--allow-dead-codes",
        action="store_true",
        help=(
            "waive the >20%% dead-code stop for a small-scale mechanism check; the waiver is "
            "recorded in the artifact and must never be inherited by a full run"
        ),
    )
    sid.add_argument(
        "--determinism",
        action="store_true",
        help="request deterministic algorithms (advisory on CUDA; recorded, not promised)",
    )

    train = sub.add_parser("train", help="stage 3: generator (placeholder)")
    train.add_argument("--catalogue", type=Path, required=True)
    train.add_argument("--exposure", type=Path, required=True)
    train.add_argument("--sid", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=0)

    score = sub.add_parser("score", help="stage 4: scores (deterministic stub)")
    score.add_argument("--cohort", type=Path, required=True)
    score.add_argument("--sid", type=Path, required=True)
    score.add_argument("--ckpt", type=Path, required=True)
    score.add_argument("--out", type=Path, default=None)
    score.add_argument("--status", choices=("approximate", "certified"), required=True)
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--sample-items", type=int, default=64)
    score.add_argument("--seed", type=int, default=2026)
    score.add_argument("--stream", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.stage == "build-features":
            artifact = stage_build_features(
                catalogue_dir=args.catalogue,
                out_dir=args.out,
                encoder=args.encoder,
                batch_size=args.batch_size,
                dim=args.dim,
                max_chars=args.max_chars,
                device=args.device,
                local_files_only=args.local_files_only,
                revision=args.encoder_revision,
            )
            record = {
                "stage": args.stage,
                "num_items": artifact.num_items,
                "dim": artifact.dim,
                "encoder": artifact.encoder.get("id"),
                "encoder_is_model": bool(artifact.encoder.get("is_model")),
                "encoder_requested_revision": artifact.encoder.get("requested_revision"),
                "encoder_revision": artifact.encoder.get("revision"),
                "encoder_weights_pinned": artifact.encoder.get("weights_pinned"),
                "empty_text_items": artifact.empty_text_items,
                "truncated_items": artifact.truncated_items,
            }
        elif args.stage == "fit-sid":
            artifact = stage_fit_sid(
                catalogue_dir=args.catalogue,
                features_dir=args.features,
                out_dir=args.out,
                levels=args.levels,
                codebook_size=args.codebook_size,
                latent_dim=args.latent_dim,
                encoder_dims=tuple(
                    int(value) for value in str(args.encoder_dims).split(",") if value.strip()
                ),
                beta=args.beta,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dedup_vocab_size=args.dedup_vocab_size,
                normalize_input=args.normalize_input,
                revive_dead=args.revive_dead,
                seed=args.seed,
                device=args.device,
                determinism=args.determinism,
                allow_dead_codes=args.allow_dead_codes,
                require_features_manifest=args.require_features_manifest,
                log=(lambda entry: print(f"[fit-sid] {entry}", file=sys.stderr))
                if not args.quiet
                else None,
            )
            dead_code_waiver_record = json.loads(
                (args.out / "semantic_ids.json").read_text(encoding="utf-8")
            ).get("dead_code_waiver", {"waived": False})
            record = {
                "stage": args.stage,
                "num_items": artifact.num_items,
                "levels": artifact.levels,
                "dedup_levels": artifact.dedup_levels,
                "dedup_vocab_size": artifact.dedup_vocab_size,
                "pre_dedup": dict(artifact.collision_audit["pre_dedup"]),
                "post_dedup": dict(artifact.collision_audit["post_dedup"]),
                "utilisation": [dict(block) for block in artifact.utilisation],
                "dead_code_waiver": dead_code_waiver_record,
            }
        elif args.stage == "train":
            checkpoint = stage_train(
                catalogue_dir=args.catalogue,
                exposure_path=args.exposure,
                sid_dir=args.sid,
                out_dir=args.out,
                epochs=args.epochs,
            )
            record = {
                "stage": args.stage,
                "validation_used": checkpoint.training.get("validation_used"),
            }
        else:
            record = stage_score(
                cohort_dir=args.cohort,
                sid_dir=args.sid,
                checkpoint_dir=args.ckpt,
                out_dir=args.out,
                status=args.status.upper(),
                batch_size=args.batch_size,
                sample_items=args.sample_items,
                seed=args.seed,
                stream=args.stream,
            )
    except (ArtifactError, CliError, DedupOverflowError, ValueError, RuntimeError) as error:
        print(f"tiger_public.cli: {error}", file=sys.stderr)
        return 2
    streaming = bool(getattr(args, "stream", False))
    if streaming:
        # stdout is the data channel in streaming mode, so the status goes to stderr.
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
    else:
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
