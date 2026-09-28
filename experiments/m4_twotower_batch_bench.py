"""Equivalence + throughput benchmark: batched vs per-user Two-Tower retrieval.

Answers one question only: is **exact** batched full-catalogue Two-Tower retrieval fast enough to
keep the frozen M4 design, or must the source be reconsidered?

It does not change recommendation semantics and does not touch the formal cohort. It reads the
frozen inputs, runs the repository's own reference path
(:class:`~recommendation.control.two_tower_source.TwoTowerSourceTool`) as the correctness oracle,
and compares the batched path against it.

Usage
-----
    python -m experiments.m4_twotower_batch_bench --equivalence 30
    python -m experiments.m4_twotower_batch_bench --sweep
    python -m experiments.m4_twotower_batch_bench --steady 300 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.m4_materialize_heads import (  # noqa: E402
    MAPPINGS,
    TWO_TOWER_CKPT,
    load_cohort,
)
from experiments.m4_twotower_batch import build_scorer  # noqa: E402

#: Batch sizes to sweep, per the task. Sizes that fail are reported, not silently dropped.
SWEEP = (1, 16, 32, 64, 128, 256)

TOP_K = 100


def peak_rss_gib() -> float:
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def current_rss_gib() -> float:
    """Current resident set size, in GiB."""
    import os

    try:
        text = pathlib.Path(f"/proc/{os.getpid()}/status").read_text()
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024 / 1024
    except OSError:  # pragma: no cover - non-Linux
        pass
    return peak_rss_gib()


def gpu_peak_gib() -> float | None:
    """Peak CUDA memory in GiB, or ``None`` when running on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
    except Exception:  # noqa: BLE001
        return None
    return None


def cohort_histories(limit: int) -> list[tuple[int, tuple[str, ...]]]:
    """The first ``limit`` users of the frozen cohort, in canonical order."""
    slices = load_cohort(limit=limit)
    return [(s.user_int_id, s.history) for s in slices]


def histories_as_item_ids(
    scorer: Any, users: Sequence[tuple[int, tuple[str, ...]]]
) -> list[tuple[int, list[int]]]:
    """Map ``parent_asin`` histories to integer ids, dropping unresolvable entries.

    Mirrors ``TwoTowerSourceTool.propose``: an unresolvable identity is a source failure there, so
    such users are excluded from the comparison rather than silently shortened.
    """
    out: list[tuple[int, list[int]]] = []
    for user_int_id, history in users:
        try:
            ids = [scorer.parent_asin_to_item_id(value) for value in history]
        except Exception:  # noqa: BLE001
            continue
        if ids:
            out.append((user_int_id, ids))
    return out


# --------------------------------------------------------------------------- #
# A. correctness / equivalence
# --------------------------------------------------------------------------- #


def equivalence(count: int, batch_size: int = 0) -> dict[str, Any]:
    """Compare the batched path against the per-user reference on ``count`` users.

    ``batch_size`` of 0 means "one batch holding every user". Set it to 1 to test the
    numerically faithful path, where each user's forward pass is identical in shape to the
    reference's.
    """
    from recommendation.control.two_tower_source import TwoTowerSourceTool

    scorer = build_scorer(TWO_TOWER_CKPT, MAPPINGS)
    users = cohort_histories(count)
    pairs = histories_as_item_ids(scorer, users)
    print(f"equivalence: {len(pairs)} users with resolvable history", flush=True)

    reference = TwoTowerSourceTool(scorer._engine)  # noqa: SLF001 - the oracle is the same engine

    mismatch_orders: list[dict[str, Any]] = []
    score_deltas: list[float] = []
    compared = 0

    # Reference first, one user at a time (this is the slow path being replaced).
    # NOTE: `TwoTowerSourceTool.propose` takes **parent_asin identities** and resolves them itself,
    # so the reference is fed the raw history while the batched path is fed integer ids below.
    started = time.time()
    reference_rows: list[list[tuple[str, int, float]]] = []
    raw_by_user = dict(users)
    for user_int_id, _ids in pairs:
        history = raw_by_user[user_int_id]
        rows = reference.propose(arguments=None, limit=TOP_K, history=tuple(history))
        reference_rows.append(list(rows))
    reference_seconds = time.time() - started
    print(
        f"reference: {len(pairs)} users in {reference_seconds:.1f}s "
        f"({reference_seconds / max(1, len(pairs)):.2f} s/user)",
        flush=True,
    )

    if batch_size and batch_size > 0:
        batched = []
        for start in range(0, len(pairs), batch_size):
            batched.extend(
                scorer.heads_for_batch(pairs[start : start + batch_size], k=TOP_K)
            )
    else:
        batched = scorer.heads_for_batch([(uid, ids) for uid, ids in pairs], k=TOP_K)

    for (user_int_id, _ids), ref_rows, batch_head in zip(pairs, reference_rows, batched):
        ref_ids = [r[0] for r in ref_rows]
        got_ids = [r[0] for r in batch_head.rows]
        compared += 1
        if ref_ids != got_ids:
            mismatch_orders.append(
                {
                    "user_int_id": user_int_id,
                    "reference_len": len(ref_ids),
                    "batched_len": len(got_ids),
                    "first_divergence": next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(ref_ids, got_ids))
                            if a != b
                        ),
                        min(len(ref_ids), len(got_ids)),
                    ),
                    "reference_head": ref_ids[:5],
                    "batched_head": got_ids[:5],
                }
            )
        for ref, got in zip(ref_rows, batch_head.rows):
            score_deltas.append(abs(float(ref[2]) - float(got[2])))

    # The engine reports scores as float32 tensors; `float()` on them is exact for the stored
    # value, and the ulp is ~1e-5 at these magnitudes. Report both the raw delta and the
    # float32-ulp band so a reader can see whether a delta is precision or a real difference.
    max_f32_ulp = max(
        (abs(float(np.spacing(np.float32(s)))) for s in score_deltas if s > 0), default=0.0
    )
    exact = compared - len(mismatch_orders)
    result = {
        "users_compared": compared,
        "topk": TOP_K,
        "exact_order_matches": exact,
        "match_rate": exact / compared if compared else 0.0,
        "identical_scores": sum(1 for d in score_deltas if d == 0.0),
        "score_pairs_compared": len(score_deltas),
        "max_abs_score_delta": max(score_deltas) if score_deltas else 0.0,
        "mean_abs_score_delta": (
            sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
        ),
        "float32_ulp_at_observed_magnitudes": float(max_f32_ulp),
        "mismatches": mismatch_orders[:10],
        "reference_seconds": reference_seconds,
        "reference_seconds_per_user": reference_seconds / max(1, len(pairs)),
    }
    return result


# --------------------------------------------------------------------------- #
# B. batch-size sweep
# --------------------------------------------------------------------------- #


def sweep(count: int, batch_sizes: Sequence[int] = SWEEP) -> list[dict[str, Any]]:
    """Time the batched path at several batch sizes on ``count`` users."""
    scorer = build_scorer(TWO_TOWER_CKPT, MAPPINGS)
    users = cohort_histories(count)
    pairs = histories_as_item_ids(scorer, users)
    print(f"sweep: {len(pairs)} users", flush=True)

    rows: list[dict[str, Any]] = []
    for size in batch_sizes:
        base_rss = current_rss_gib()
        started = time.time()
        processed = 0
        failure: str | None = None
        try:
            for start in range(0, len(pairs), size):
                chunk = pairs[start : start + size]
                scorer.heads_for_batch(chunk, k=TOP_K)
                processed += len(chunk)
        except MemoryError as exc:
            failure = f"MemoryError: {exc}"
        except Exception as exc:  # noqa: BLE001
            failure = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started
        ups = processed / elapsed if elapsed > 0 and processed else 0.0
        rows.append(
            {
                "batch_size": size,
                "users_processed": processed,
                "total_seconds": elapsed,
                "users_per_second": ups,
                "ms_per_user": (1000.0 / ups) if ups else None,
                "peak_rss_gib": peak_rss_gib(),
                "rss_before_gib": base_rss,
                "peak_gpu_gib": gpu_peak_gib(),
                "failure": failure,
            }
        )
        label = f"batch={size:4d} users={processed:5d} {elapsed:8.2f}s {ups:7.2f} u/s"
        print(
            f"{label}  peakRSS={peak_rss_gib():.2f} GiB"
            + (f"  FAILURE={failure}" if failure else ""),
            flush=True,
        )
        if failure:
            break
    return rows


# --------------------------------------------------------------------------- #
# C. steady-state
# --------------------------------------------------------------------------- #


def steady(count: int, batch_size: int) -> dict[str, Any]:
    """Longer run at the chosen batch size so startup cost does not dominate."""
    scorer = build_scorer(TWO_TOWER_CKPT, MAPPINGS)
    users = cohort_histories(count)
    pairs = histories_as_item_ids(scorer, users)
    # Warm up so the first batch's lazy allocation is not counted as steady state.
    scorer.heads_for_batch(pairs[: min(batch_size, len(pairs))], k=TOP_K)
    started = time.time()
    processed = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        scorer.heads_for_batch(chunk, k=TOP_K)
        processed += len(chunk)
    elapsed = time.time() - started
    ups = processed / elapsed if elapsed else 0.0
    projected = 20000 / ups / 3600 if ups else float("inf")
    out = {
        "batch_size": batch_size,
        "users_processed": processed,
        "total_seconds": elapsed,
        "users_per_second": ups,
        "ms_per_user": (1000.0 / ups) if ups else None,
        "peak_rss_gib": peak_rss_gib(),
        "peak_gpu_gib": gpu_peak_gib(),
        "projected_20000_hours": projected,
    }
    print(
        f"steady: batch={batch_size} users={processed} {elapsed:.2f}s "
        f"{ups:.2f} u/s -> projected 20k = {projected:.2f} h  peakRSS={peak_rss_gib():.2f} GiB",
        flush=True,
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="M4 batched Two-Tower benchmark")
    parser.add_argument("--equivalence", type=int, default=0)
    parser.add_argument("--equiv-batch-size", type=int, default=0, help="0 = one batch for all")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-users", type=int, default=32)
    parser.add_argument("--steady", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"checkpoint": str(TWO_TOWER_CKPT), "mappings": str(MAPPINGS)}
    if args.equivalence:
        report["equivalence"] = equivalence(args.equivalence, args.equiv_batch_size)
        print(json.dumps(report["equivalence"], indent=1)[:1600], flush=True)
    if args.sweep:
        report["sweep"] = sweep(args.sweep_users)
    if args.steady:
        report["steady"] = steady(args.steady, args.batch_size)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
