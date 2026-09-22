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
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

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
from tiger_public.io import (
    ArtifactError,
    read_catalogue,
    read_eval_cohort,
    read_train_exposure,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
    write_manifest,
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


def _placeholder_features(num_items: int, dim: int, *, seed: int) -> np.ndarray:
    """Deterministic, content-free item features.  **Not** embeddings, and labelled as such.

    A small integer hash rather than a random draw, so the bytes depend only on
    ``(item_id, dim, seed)`` and the artifact is reproducible across machines and NumPy
    versions without relying on a bit-generator's stability.
    """
    rows = np.arange(num_items, dtype=np.int64)
    columns = np.arange(dim, dtype=np.int64)
    mixed = (rows[:, None] * 2654435761 + columns[None, :] * 40503 + seed * 2246822519) % 65521
    return (mixed.astype(np.float32) / 65521.0).astype(np.float32)


def stage_build_features(
    *, catalogue_dir: Path, out_dir: Path, seed: int = 2026, dim: int = STUB_FEATURE_DIM
) -> ItemFeatureArtifact:
    """Write a placeholder ``item_features.npy`` with **no PAD row**."""
    catalogue = read_catalogue(catalogue_dir)
    if dim < 1:
        raise CliError(f"dim must be >= 1, got {dim}")
    features = _placeholder_features(catalogue.num_items, dim, seed=seed)
    if features.shape != (catalogue.num_items, dim):
        raise CliError(
            f"placeholder features have shape {features.shape}, expected "
            f"({catalogue.num_items}, {dim})"
        )
    if not np.isfinite(features).all():
        raise CliError("placeholder features are not finite")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = features.tobytes(order="C")
    (out_dir / "item_features.npy").write_bytes(payload)
    record = {
        "format": "agentrecx.tiger.item_features.v3",
        "contract_version": CONTRACT_VERSION,
        "num_items": catalogue.num_items,
        "dim": dim,
        "dtype": "float32",
        "order": "C",
        "endianness": "little",
        "encoder": {
            "id": STUB_MARKER,
            "revision": "none",
            "pooling": "none",
            "normalize": False,
            "frozen": True,
        },
        "text_template": "not used by the Step-2.3 placeholder",
        "max_chars": 0,
        "empty_text_items": 0,
        "truncated_items": 0,
        "nan_inf_scan": "clean",
        "catalogue_sha256": catalogue.mappings_sha256,
        "device": "cpu",
        "seconds": 0.0,
        "seed": seed,
        "pad_row_present": False,
    }
    write_json(out_dir / "item_features.json", record)
    write_manifest(out_dir, extra={"stage": "build-features", "stub": True})
    return ItemFeatureArtifact(
        format=record["format"],
        num_items=catalogue.num_items,
        dim=dim,
        path=str(out_dir / "item_features.npy"),
        sha256=sha256_bytes(payload),
        encoder=record["encoder"],
        catalogue_sha256=catalogue.mappings_sha256,
    )


# --------------------------------------------------------------------------- #
# Stage 2 - fit semantic ids (placeholder)
# --------------------------------------------------------------------------- #


def _placeholder_codes(num_items: int, levels: int, codebook_size: int) -> np.ndarray:
    """Deterministic codes that spread across each codebook, with realistic collisions.

    A multiplicative permutation of the item index rather than a structured index.  Two
    properties matter and a naive scheme gets one of them wrong:

    * **codes spread**, so a catalogue at least as large as the codebook uses every code.  A
      structured scheme (``row // 4``) leaves most codes dead and would trip the
      catastrophic-collapse gate for an artefact of the placeholder rather than a real
      collapse.  Note the honest consequence: a codebook wider than the catalogue genuinely
      *has* dead codes, and the gate correctly refuses that configuration.
    * **collisions still occur**, so the dedup digit has work to do.  The stride is
      ``codebook_size * 2**20 + 1``, which is coprime to the codebook size for any power of
      two; its projection onto a 1024-slot window therefore repeats after roughly a thousand
      items, which is what puts collisions into the audit at a realistic rate.
    """
    stride = codebook_size * (1 << 20) + 1
    window = 1024 * (1 << 20) + 1
    rows = np.arange(num_items, dtype=np.int64)
    columns = np.arange(levels, dtype=np.int64)
    projected = (rows * stride) % window
    per_level = (projected % codebook_size)[:, None] + columns[None, :] * 7
    return np.ascontiguousarray(per_level % codebook_size).astype(np.int64)


def _utilisation(codes: np.ndarray, *, codebook_size: int) -> tuple[dict[str, Any], ...]:
    """Per-level coverage and entropy, reported without becoming a superiority criterion."""
    blocks: list[dict[str, Any]] = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=codebook_size).astype(np.float64)
        used = int((counts > 0).sum())
        fractions = counts[counts > 0] / counts.sum()
        entropy = float(-(fractions * np.log(fractions)).sum()) if fractions.size else 0.0
        dead = codebook_size - used
        blocks.append(
            {
                "level": level,
                "codebook_size": codebook_size,
                "used_codes": used,
                "dead_codes": dead,
                "frac_layer_coverage": round(used / codebook_size, 6),
                "id_entropy": round(entropy, 6),
                # The only hard stop: a level that has catastrophically collapsed.
                "collapsed": dead > 0.2 * codebook_size,
            }
        )
    return tuple(blocks)


def stage_fit_sid(
    *,
    catalogue_dir: Path,
    features_dir: Path,
    out_dir: Path,
    levels: int = 3,
    codebook_size: int = 64,
    dedup_levels: int = 1,
    dedup_vocab_size: int | None = None,
) -> SemanticIdArtifact:
    """Write a placeholder ``semantic_ids.json`` with a PAD sentinel and a dedup digit.

    The overflow rule of ``docs/TIGER_BACKEND.md`` section 12.1 is enforced **here**, before any
    dedup digit is assigned: a pre-dedup collision group larger than the dedup vocabulary
    refuses the build rather than wrapping, sharing, or truncating.
    """
    catalogue = read_catalogue(catalogue_dir)
    if not (features_dir / "item_features.npy").is_file():
        raise ArtifactError(f"missing {features_dir / 'item_features.npy'}")
    dedup_size = codebook_size if dedup_vocab_size is None else int(dedup_vocab_size)
    layout = build_token_layout(
        levels=levels,
        codebook_size=codebook_size,
        dedup_levels=dedup_levels,
        dedup_vocab_size=dedup_size,
    )
    codes = _placeholder_codes(catalogue.num_items, levels, codebook_size)
    audit = collision_audit(
        codes.tolist(), dedup_levels=dedup_levels, dedup_vocab_size=dedup_size
    )
    if audit["overflow_refusal_triggered"]:
        raise CliError(
            f"largest pre-dedup collision group {audit['pre_dedup']['largest_group']} exceeds "
            f"dedup_vocab_size {dedup_size}; re-fit with a larger codebook, more levels, or a "
            "wider dedup level"
        )
    for block in _utilisation(codes, codebook_size=codebook_size):
        if block["collapsed"]:
            raise CliError(
                f"level {block['level']} has {block['dead_codes']} dead codes "
                f"(> 20% of {codebook_size}); that is a catastrophic collapse, not a "
                "reportable observation"
            )

    pad_row = [PAD_SENTINEL] * layout.per_item_tokens
    assignment: list[list[int]] = [list(pad_row)]
    for row in codes.tolist():
        item = list(row)
        if dedup_levels:
            item.append(0)
        assignment.append(item)
    if dedup_levels:
        # One O(N) pass, first member of each group assigned 0 - the corrected form of GRID's
        # scheme, which assigns 1..k to every member and scans the array once per group.
        seen: dict[tuple[int, ...], int] = {}
        for index in range(1, len(assignment)):
            key = tuple(assignment[index][:levels])
            ordinal = seen.get(key, 0)
            seen[key] = ordinal + 1
            assignment[index][levels] = ordinal

    record = {
        "format": "agentrecx.tiger.semantic_ids.v3",
        "contract_version": CONTRACT_VERSION,
        "num_items": catalogue.num_items,
        "levels": levels,
        "dedup_levels": dedup_levels,
        "codebook_size": codebook_size,
        "dedup_vocab_size": dedup_size,
        "code_space": codebook_size**levels,
        "code_space_with_dedup": (codebook_size**levels) * (dedup_size if dedup_levels else 1),
        "pad_row": list(pad_row),
        "assignment": assignment,
        "collision_audit": audit,
        "utilisation": list(_utilisation(codes, codebook_size=codebook_size)),
        "quantizer": {
            "init": STUB_MARKER,
            "normalize_input": False,
            "revive_dead": False,
            "beta": 0.25,
            "latent": 0,
            "epochs": 0,
            "batch_size": 0,
            "lr": 0.0,
        },
        "content_features": {
            "path": "item_features/item_features.npy",
            "sha256": sha256_file(features_dir / "item_features.npy"),
            "encoder": STUB_MARKER,
            "dim": STUB_FEATURE_DIM,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "semantic_ids.json", record)
    write_json(out_dir / "layout.json", layout.as_dict())
    write_manifest(out_dir, extra={"stage": "fit-sid", "stub": True})
    return SemanticIdArtifact(
        format=record["format"],
        contract_version=CONTRACT_VERSION,
        num_items=catalogue.num_items,
        levels=levels,
        dedup_levels=dedup_levels,
        codebook_size=codebook_size,
        dedup_vocab_size=dedup_size,
        pad_row=tuple(pad_row),
        assignment=tuple(tuple(row) for row in assignment),
        collision_audit=audit,
        utilisation=tuple(record["utilisation"]),
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
# Argument parsing
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tiger_public.cli",
        description="public-TIGER backend stages (Step 2.3: no ML; all stages are placeholders)",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    features = sub.add_parser("build-features", help="stage 1: item features (placeholder)")
    features.add_argument("--catalogue", type=Path, required=True)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--seed", type=int, default=2026)

    sid = sub.add_parser("fit-sid", help="stage 2: semantic ids (placeholder)")
    sid.add_argument("--catalogue", type=Path, required=True)
    sid.add_argument("--features", type=Path, required=True)
    sid.add_argument("--out", type=Path, required=True)
    sid.add_argument("--levels", type=int, default=3)
    sid.add_argument("--codebook-size", type=int, default=64)
    sid.add_argument("--dedup-levels", type=int, default=1)
    sid.add_argument("--dedup-vocab-size", type=int, default=None)

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
                catalogue_dir=args.catalogue, out_dir=args.out, seed=args.seed
            )
            record = {"stage": args.stage, "num_items": artifact.num_items, "dim": artifact.dim}
        elif args.stage == "fit-sid":
            artifact = stage_fit_sid(
                catalogue_dir=args.catalogue,
                features_dir=args.features,
                out_dir=args.out,
                levels=args.levels,
                codebook_size=args.codebook_size,
                dedup_levels=args.dedup_levels,
                dedup_vocab_size=args.dedup_vocab_size,
            )
            record = {
                "stage": args.stage,
                "num_items": artifact.num_items,
                "levels": artifact.levels,
                "dedup_levels": artifact.dedup_levels,
                "post_dedup": dict(artifact.collision_audit["post_dedup"]),
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
    except (ArtifactError, CliError, ValueError) as error:
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
