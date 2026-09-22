"""Read-only SID diagnostics for a fitted public-TIGER tokenizer (Step 2.4F).

Consumes an existing Semantic-ID artifact and reports what the single aggregate
``reconstruction_loss`` in ``semantic_ids.json`` cannot show: **how much of the reconstruction
each residual level actually contributes**, and whether a level has collapsed onto a handful of
codes.

```text
read-only, no training, no artifact modification
    item_features.npy  +  tokenizer.pt  +  semantic_ids.json
        -> prefix reconstruction MSE at L0, L0+L1, L0+L1+L2
        -> latent / input norm statistics and the residual norm after each level
        -> per-level occupancy histogram, coverage, entropy, largest-code fraction
        -> pre-dedup collision-group size distribution
```

Two quantities are deliberately distinguished, because conflating them is the usual mistake:

```text
codebook coverage     = used_codes / codebook_size      how much of the codebook is touched
largest_code_fraction = max(code_count) / num_items     how concentrated the assignment is
```

A level can have full coverage and still be badly skewed, or low coverage and evenly used.

The tool never writes into the artifact directory; ``--out`` writes a separate JSON report.

Usage::

    cd backends/tiger_public
    PYTHONPATH=src .venv/bin/python ../../experiments/audit_tiger_sid.py \
        --features ../../runs/tiger_public_2026/features \
        --sid      ../../runs/tiger_public_2026/sid \
        --out     ../../runs/tiger_public_2026/sid_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

#: The audit reads the same architecture description the trainer wrote, so it cannot drift.
DEFAULT_BATCH_SIZE = 4096


class AuditError(RuntimeError):
    """Raised when the artifacts are missing, inconsistent, or not readable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts(features_dir: Path, sid_dir: Path) -> tuple[np.ndarray, dict, dict, Any]:
    """Load features, the SID record and the tokenizer, verifying the recorded digests.

    The check that matters: ``semantic_ids.json.content_features.sha256`` must match the feature
    file actually being read.  Auditing a feature matrix against a SID assignment built from a
    different one would produce confident, meaningless numbers.
    """
    record_path = sid_dir / "semantic_ids.json"
    if not record_path.is_file():
        raise AuditError(f"missing {record_path}")
    sid = json.loads(record_path.read_text(encoding="utf-8"))

    features_path = features_dir / "item_features.npy"
    if not features_path.is_file():
        raise AuditError(f"missing {features_path}")
    actual = sha256_file(features_path)
    recorded = str((sid.get("content_features") or {}).get("sha256") or "")
    if recorded and recorded != actual:
        raise AuditError(
            "the Semantic-ID artifact was built from different features: it records "
            f"{recorded[:12]}... but {features_path} hashes to {actual[:12]}...; refusing to "
            "audit a mismatched pair"
        )

    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    feature_record = json.loads((features_dir / "item_features.json").read_text(encoding="utf-8"))

    checkpoint_path = sid_dir / "tokenizer.pt"
    if not checkpoint_path.is_file():
        raise AuditError(f"missing {checkpoint_path}")
    from tiger_public.quantizer import QuantizerConfig, RqVae  # noqa: PLC0415

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = QuantizerConfig.from_artifact(payload["config"])
    model = RqVae(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return features, sid, feature_record, model


def prefix_diagnostics(
    model: Any, features: np.ndarray, *, batch_size: int, device: str
) -> dict[str, Any]:
    """Reconstruction MSE and residual norms for each cumulative prefix of the code path.

    For a code path ``(c1, c2, c3)`` the decoder was trained on the *sum* of the chosen code
    vectors, so the honest prefix quantities are:

    ```text
    L0            decoder(sum of level 0)                    vs the normalised target
    L0+L1         decoder(level 0 + level 1)                 vs the normalised target
    L0+L1+L2      decoder(level 0 + level 1 + level 2)       vs the normalised target
    ```

    which is exactly the tensor the trainer fed the decoder at each stage.  The residual norm
    after level *l* is the norm of what levels ``0..l`` failed to explain, so the drop between
    consecutive values is that level's contribution.
    """
    torch_device = torch.device(device)
    model = model.to(torch_device)
    levels = int(model.config.levels)
    rows = int(features.shape[0])

    squared_error: list[float] = [0.0] * (levels + 1)
    target_squared: list[float] = [0.0] * (levels + 1)
    input_norm_sum = 0.0
    target_norm_sum = 0.0
    latent_norm_sum = 0.0
    residual_norm_sums: list[float] = [0.0] * (levels + 1)
    total = 0
    baseline_mse_sum = 0.0  # MSE of the zero prediction, for scale context

    with torch.no_grad():
        for start in range(0, rows, batch_size):
            # ``np.asarray`` on a memory-mapped file yields a read-only view, and torch warns
            # that writing to such a tensor is undefined.  The audit only reads, but copying is
            # cheap at this batch size and removes the ambiguity entirely.
            block = np.array(features[start : start + batch_size], dtype=np.float32, copy=True)
            tensor = torch.from_numpy(block).to(torch_device)
            target = model.normalise(tensor)
            latents = model.encoder(target)
            residual = latents
            accumulated = torch.zeros_like(latents)
            count = tensor.shape[0]
            total += count

            input_norm_sum += float(tensor.norm(dim=-1).sum())
            target_norm_sum += float(target.norm(dim=-1).sum())
            latent_norm_sum += float(latents.norm(dim=-1).sum())
            residual_norm_sums[0] += float(residual.norm(dim=-1).sum())
            baseline_mse_sum += float((target**2).sum())

            for level in range(levels):
                codebook = model.quantizer.codebooks[level]
                distances = (
                    residual.pow(2).sum(dim=-1, keepdim=True)
                    - 2.0 * residual @ codebook.t()
                    + codebook.pow(2).sum(dim=-1).unsqueeze(0)
                )
                chosen = codebook[distances.argmin(dim=-1)]
                accumulated = accumulated + chosen
                residual = residual - chosen
                reconstruction = model.decoder(accumulated)
                squared_error[level + 1] += float((reconstruction - target).pow(2).sum())
                target_squared[level + 1] += float(target.pow(2).sum())
                residual_norm_sums[level + 1] += float(residual.norm(dim=-1).sum())

    divisor = max(1, total)
    element_divisor = max(1, total * int(features.shape[1]))
    return {
        "num_items": total,
        "dim": int(features.shape[1]),
        "mse": {
            "L0": squared_error[1] / element_divisor if levels >= 1 else None,
            "L0+L1": squared_error[2] / element_divisor if levels >= 2 else None,
            "L0+L1+L2": squared_error[3] / element_divisor if levels >= 3 else None,
        },
        "zero_prediction_mse": baseline_mse_sum / element_divisor,
        "norms": {
            "input_norm_mean": input_norm_sum / divisor,
            "normalised_target_norm_mean": target_norm_sum / divisor,
            "latent_norm_mean": latent_norm_sum / divisor,
            "residual_norm_after_L0": residual_norm_sums[1] / divisor,
            "residual_norm_after_L1": residual_norm_sums[2] / divisor if levels >= 2 else None,
            "residual_norm_after_L2": residual_norm_sums[3] / divisor if levels >= 3 else None,
            "initial_latent_norm_mean": residual_norm_sums[0] / divisor,
        },
    }


def level_diagnostics(codes: np.ndarray, *, codebook_size: int) -> tuple[dict[str, Any], ...]:
    """Per-level occupancy, coverage and concentration.

    ``occupancy_histogram`` is the **complete** distribution: index ``i`` holds the number of
    codes selected exactly ``i`` times, and the final entry counts every code selected more than
    the histogram's cap.  ``largest_code_fraction`` is the share of the catalogue sitting on the
    single most-used code, which is a different question from coverage.
    """
    table = np.asarray(codes)
    if table.ndim != 2:
        raise AuditError(f"codes must be [N, levels], got shape {tuple(table.shape)}")
    num_items = int(table.shape[0])
    blocks: list[dict[str, Any]] = []

    for level in range(table.shape[1]):
        counts = np.bincount(table[:, level], minlength=codebook_size).astype(np.int64)
        used = int((counts > 0).sum())
        fractions = counts[counts > 0] / counts.sum()
        entropy = float(-(fractions * np.log(fractions)).sum()) if fractions.size else 0.0
        largest = int(counts.max()) if counts.size else 0
        tail = counts[counts > 10]
        histogram: dict[str, int] = {}
        for occurrences in range(0, 11):
            histogram[str(occurrences)] = int((counts == occurrences).sum())
        histogram[">10"] = int(tail.size)
        blocks.append(
            {
                "level": level,
                "codebook_size": int(codebook_size),
                "used_codes": used,
                "dead_codes": int(codebook_size) - used,
                "codebook_coverage": round(used / int(codebook_size), 6),
                "id_entropy": round(entropy, 6),
                "max_entropy": round(math.log(int(codebook_size)), 6),
                "entropy_fraction": (
                    round(entropy / math.log(int(codebook_size)), 6)
                    if int(codebook_size) > 1
                    else 0.0
                ),
                "largest_code_count": largest,
                "largest_code_fraction": round(largest / num_items, 6) if num_items else 0.0,
                "codes_used_once": int((counts == 1).sum()),
                "occupancy_histogram": histogram,
            }
        )
    return tuple(blocks)


def collision_diagnostics(assignment: Sequence[Sequence[int]], *, levels: int) -> dict[str, Any]:
    """Pre-dedup collision statistics, including the full group-size distribution.

    ``assignment[0]`` is the reserved PAD sentinel row and is excluded.  Only the first
    ``levels`` digits form the quantizer's code; the trailing digit is the dedup ordinal, which
    is removed before grouping so the reported collisions are the quantizer's own.
    """
    real = [tuple(int(value) for value in row) for row in assignment[1:]]
    if not real:
        raise AuditError("the assignment holds no real items")
    pre: dict[tuple[int, ...], int] = {}
    for row in real:
        pre[row[:levels]] = pre.get(row[:levels], 0) + 1
    sizes = np.asarray(list(pre.values()), dtype=np.int64)
    distribution = {str(size): int((sizes == size).sum()) for size in sorted(set(sizes.tolist()))}
    collided = int((sizes > 1).sum())
    post = len(set(real))
    return {
        "num_items": len(real),
        "levels": int(levels),
        "dedup_levels": len(real[0]) - int(levels),
        "pre_dedup_distinct_sids": len(pre),
        "pre_dedup_collision_groups": collided,
        "pre_dedup_items_in_collision": int(sizes[sizes > 1].sum()),
        "pre_dedup_collision_rate": round(float(sizes[sizes > 1].sum()) / len(real), 6),
        "pre_dedup_largest_group": int(sizes.max()),
        "pre_dedup_group_size_distribution": distribution,
        "post_dedup_distinct_sids": post,
        "post_dedup_collisions": len(real) - post,
        "post_dedup_unique": post == len(real) and len(set(real)) == len(real),
    }


def render(report: dict[str, Any]) -> str:
    """A compact human-readable rendering; the JSON report holds the same numbers."""
    lines: list[str] = []
    prefix = report["prefix"]
    mse = prefix["mse"]
    norms = prefix["norms"]
    lines.append("PREFIX RECONSTRUCTION (MSE against the normalised target)")
    lines.append(f"  items {prefix['num_items']}, dim {prefix['dim']}")
    for label in ("L0", "L0+L1", "L0+L1+L2"):
        value = mse.get(label)
        lines.append(f"  {label:<10} {value if value is None else round(value, 8)}")
    lines.append(f"  {'zero-pred':<10} {round(prefix['zero_prediction_mse'], 8)}  (scale reference)")
    lines.append("")
    lines.append("NORMS (mean per item)")
    for key in (
        "input_norm_mean",
        "normalised_target_norm_mean",
        "initial_latent_norm_mean",
        "latent_norm_mean",
        "residual_norm_after_L0",
        "residual_norm_after_L1",
        "residual_norm_after_L2",
    ):
        value = norms.get(key)
        lines.append(f"  {key:<30} {value if value is None else round(value, 6)}")
    lines.append("")
    lines.append("PER-LEVEL OCCUPANCY")
    header = f"  {'lvl':<4}{'used':>7}{'dead':>7}{'coverage':>10}{'H':>9}{'H/Hmax':>9}{'maxcnt':>8}{'maxfrac':>9}"
    lines.append(header)
    for block in report["levels"]:
        lines.append(
            f"  {block['level']:<4}{block['used_codes']:>7}{block['dead_codes']:>7}"
            f"{block['codebook_coverage']:>10.4f}{block['id_entropy']:>9.4f}"
            f"{block['entropy_fraction']:>9.4f}{block['largest_code_count']:>8}"
            f"{block['largest_code_fraction']:>9.4f}"
        )
    lines.append("")
    lines.append("PRE-DEDUP COLLISION GROUP SIZES")
    collisions = report["collisions"]
    for size, count in collisions["pre_dedup_group_size_distribution"].items():
        lines.append(f"  group size {size:>3} -> {count}")
    lines.append(
        f"  largest group {collisions['pre_dedup_largest_group']}, "
        f"groups {collisions['pre_dedup_collision_groups']}, "
        f"items {collisions['pre_dedup_items_in_collision']}, "
        f"rate {collisions['pre_dedup_collision_rate']}"
    )
    lines.append(
        f"  post-dedup: distinct {collisions['post_dedup_distinct_sids']}, "
        f"collisions {collisions['post_dedup_collisions']}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="read-only Semantic-ID diagnostics (never trains, never modifies artifacts)"
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sid", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        features, sid, feature_record, model = load_artifacts(args.features, args.sid)
        prefix = prefix_diagnostics(
            model, features, batch_size=args.batch_size, device=args.device
        )
        assignment = sid["assignment"]
        levels = int(sid["levels"])
        codes = np.asarray([row[:levels] for row in assignment[1:]], dtype=np.int64)
        report = {
            "format": "agentrecx.tiger.sid_audit.v1",
            "features_dir": str(args.features),
            "sid_dir": str(args.sid),
            "item_features_sha256": sha256_file(args.features / "item_features.npy"),
            "tokenizer_sha256": sha256_file(args.sid / "tokenizer.pt"),
            "semantic_ids_sha256": sha256_file(args.sid / "semantic_ids.json"),
            "encoder": (feature_record.get("encoder") or {}).get("id"),
            "encoder_revision": (feature_record.get("encoder") or {}).get("revision"),
            "quantizer": sid.get("quantizer"),
            "prefix": prefix,
            "levels": level_diagnostics(codes, codebook_size=int(sid["codebook_size"])),
            "collisions": collision_diagnostics(assignment, levels=levels),
            "read_only": True,
        }
    except (AuditError, OSError, KeyError, ValueError) as error:
        print(f"audit_tiger_sid: {error}", file=sys.stderr)
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not args.quiet:
            print(f"wrote {args.out}")
    if not args.quiet:
        print(render(report))
    else:
        print(json.dumps({"read_only": True, "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
