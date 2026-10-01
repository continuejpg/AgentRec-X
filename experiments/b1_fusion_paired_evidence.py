#!/usr/bin/env python3
"""B1 — Two-Tower fusion paired evidence: executes ``docs/B1_PREREGISTRATION.md`` once.

What this run does, and does not do
-----------------------------------
It reproduces the frozen A/B/C/D fusion controls on the frozen 20,000-user cohort and supplies the
paired evidence that was never persisted.  Exactly **one** source is newly computed: the Two-Tower
head at depth 1000.  The popularity, SASRec and metadata heads are **read from the frozen artifact**
and never recomputed — in particular the ragged metadata head (min 99 / mean 456.53 / max 496) is
used exactly as stored.

Nothing here re-implements the instrument:

* the Two-Tower head is built by ``benchmark_public.arm_two_tower`` +
  ``benchmark_public.sasrec_top_rankings`` — the same two calls the historical run used;
* the arms are ``benchmark_public.arm_fixed_fusion`` with an explicit source list;
* scoring is ``recommendation.evaluation.batched.evaluate_batched`` unchanged;
* the statistics are imported verbatim from the tracked M2 implementation
  (``runs/m2/m2_paired_uncertainty.py``), whose seed/resample count is the frozen protocol.

Gate G1 is the instrument check: arm A must reproduce the frozen
``runs/m3_evidence/phase5_fixed_fusion_hits.npz`` target ranks **exactly** for all 20,000 users.
If it does not, the run stops — a disagreement there means the instrument differs from the frozen
pipeline, and no number produced afterwards would be comparable.

Usage::

    python experiments/b1_fusion_paired_evidence.py --gate-only   # integrity + cohort, no retrieval
    python experiments/b1_fusion_paired_evidence.py               # the full frozen protocol
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import benchmark_public as B  # noqa: E402
from recommendation.evaluation.batched import evaluate_batched  # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen facts — docs/B1_PREREGISTRATION.md §1–§3
# --------------------------------------------------------------------------- #

MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
PRODUCTS = REPO / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
HEADS = REPO / "runs" / "m3_evidence" / "phase5_fusion_source_heads.npz"
FROZEN_ARM_A_HITS = REPO / "runs" / "m3_evidence" / "phase5_fixed_fusion_hits.npz"
CHECKPOINT = REPO / "runs" / "twotower_public_2026_full" / "checkpoint.pt"

#: Gate G2 / G5 — every input identity recorded in the preregistration.
FROZEN_INPUTS: dict[Path, str] = {
    MAPPINGS: "dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912",
    SEQUENCES: "d3d834260c867ff39ac1b32bb08fd17ea1dc394e9a6fb7af8d19aa046c1eab81",
    PRODUCTS: "175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd",
    HEADS: "c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4",
    FROZEN_ARM_A_HITS: "62c2bc7c8013dfd651e4773e6dc328ae645e49f42af59d71fa421834a464b23f",
    CHECKPOINT: "0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab",
    REPO / "recommendation" / "evaluation" / "batched.py":
        "1721ae7bf60ef3dfb43f8d0f96b49cb3562865ac4eca9e178d8bb7b277832f11",
    REPO / "recommendation" / "evaluation" / "metrics.py":
        "5f84e87dff75f159218cae3e270ac331f12c08237472d229766a0c4a9db6ae72",
    REPO / "recommendation" / "evaluation" / "split.py":
        "054211a685c8653237fa7a6602832de07607bf2345930cffa25fcea9af20ef08",
}

COHORT_TARGET_SHA256 = "99dbaf7396913303fa1497c908a95e61239a29ad14a6a8f4e51d0b37fa8e99d4"
HEAD_DEPTH = 1000
K_VALUES: tuple[int, ...] = B.K_VALUES
PRIMARY_K = 10

#: Arm name -> frozen source order (§4).  RRF sums are order-independent; the order is frozen
#: anyway so a re-run is byte-reproducible.
CONTROLS: dict[str, tuple[str, ...]] = {
    "A": ("popularity", "sequential", "metadata"),
    "B": ("popularity", "sequential", "metadata", "two_tower"),
    "C": ("sequential", "metadata"),
    "D": ("sequential", "metadata", "two_tower"),
}

#: The two predeclared contrasts: (before, after).
CONTRASTS: tuple[tuple[str, str], ...] = (("A", "B"), ("C", "D"))

#: Metric series carried through the paired bootstrap.
BOOTSTRAP_SERIES: tuple[str, ...] = (
    "ndcg@5", "ndcg@10", "ndcg@20", "hit@5", "hit@10", "hit@20",
)


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_m2_statistics() -> Any:
    """Import the tracked M2 statistics implementation (the frozen protocol's own code)."""
    path = REPO / "runs" / "m2" / "m2_paired_uncertainty.py"
    spec = importlib.util.spec_from_file_location("b1_frozen_statistics", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot load the frozen statistics implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_inputs() -> dict[str, str]:
    """Gate G2/G5: every frozen input matches its recorded digest."""
    print("GATE G2/G5 - input identities")
    observed: dict[str, str] = {}
    for path, expected in FROZEN_INPUTS.items():
        if not path.is_file():
            raise SystemExit(f"GATE FAILURE: frozen input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"GATE FAILURE: {path}\n  expected {expected}\n  observed {actual}"
            )
        observed[str(path.relative_to(REPO))] = actual
        print(f"  OK  {path.relative_to(REPO)}")
    return observed


def build_cohort() -> tuple[list[Any], int, np.ndarray, dict[str, Any]]:
    """Rebuild the frozen cohort and check its identity anchor (gate G3)."""
    print("\nCOHORT - rebuilding from the frozen artifacts")
    started = time.time()
    cases, split_report = B.load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    selection = B.cohort_from_cases(cases, size=B.DEFAULT_COHORT_SIZE)
    cohort = list(selection["cases"])
    num_items = split_report.catalog_size
    del cases
    targets = np.asarray([case.test_target for case in cohort], dtype=np.int64)
    digest = sha256_array(targets)
    print(
        f"  users={len(cohort):,} catalogue={num_items:,} "
        f"({time.time() - started:.1f}s)"
    )
    if len(cohort) != B.DEFAULT_COHORT_SIZE:
        raise SystemExit(f"GATE FAILURE: cohort has {len(cohort)} users")
    if num_items != 156_746:
        raise SystemExit(f"GATE FAILURE: catalogue is {num_items}")
    if digest != COHORT_TARGET_SHA256:
        raise SystemExit(
            "GATE G3 FAILURE: cohort target_ids digest differs from the preregistration\n"
            f"  expected {COHORT_TARGET_SHA256}\n  observed {digest}"
        )
    print(f"  GATE G3 OK  target_ids sha256 = {digest}")
    return cohort, num_items, targets, selection["description"]


def load_frozen_heads(count: int) -> dict[str, list[list[int]]]:
    """Read the three frozen heads.  The metadata head is used exactly as stored (gate G6)."""
    print("\nFROZEN HEADS - read only; no recomputation")
    archive = np.load(HEADS)
    rankings: dict[str, list[list[int]]] = {}
    for name in ("popularity", "sequential", "metadata"):
        matrix = archive[name]
        lengths = archive[f"{name}_lengths"]
        if matrix.shape[0] != count:
            raise SystemExit(f"GATE FAILURE: {name} head has {matrix.shape[0]} rows")
        heads = [matrix[row, : int(lengths[row])].tolist() for row in range(count)]
        if any(not head for head in heads):
            raise SystemExit(f"GATE FAILURE: {name} has an empty head")
        if any(item <= 0 for head in heads for item in head):
            raise SystemExit(f"GATE FAILURE: {name} contains PAD or an out-of-range id")
        rankings[name] = heads
        lengths_only = [len(head) for head in heads]
        print(
            f"  {name:11s} shape={matrix.shape} "
            f"len min/mean/max = {min(lengths_only)}/{np.mean(lengths_only):.2f}/{max(lengths_only)}"
        )
    print("  GATE G6 OK  metadata head read from the frozen file; metadata_fused_heads NOT called")
    return rankings


def compute_two_tower_head(
    cohort: Sequence[Any], num_items: int, *, quiet: bool
) -> list[list[int]]:
    """The only new retrieval: depth-1000 exact full-catalogue Two-Tower heads."""
    print("\nTWO-TOWER HEAD - the one new computation")
    scorer = B.arm_two_tower(checkpoint=CHECKPOINT)
    total = len(cohort)
    done = 0
    started = time.time()

    def counting(cases: Sequence[Any]) -> Iterable[Any]:
        nonlocal done
        for batch in scorer(cases):
            done += len(batch[1])
            if not quiet:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"    scored {done:,}/{total:,} users "
                    f"({elapsed:.0f}s, {rate:.1f} users/s)",
                    flush=True,
                )
            yield batch

    heads = B.sasrec_top_rankings(
        cases=cohort, scorer=counting, num_items=num_items, top_n=HEAD_DEPTH
    )
    elapsed = time.time() - started
    lengths = [len(head) for head in heads]
    print(f"  head built in {elapsed:.1f}s; len min/max = {min(lengths)}/{max(lengths)}")
    if len(heads) != total:
        raise SystemExit(f"GATE G4 FAILURE: {len(heads)} heads for {total} users")
    if any(not head for head in heads):
        raise SystemExit("GATE G4 FAILURE: an empty Two-Tower head")
    if any(item <= 0 for head in heads for item in head):
        raise SystemExit("GATE G4 FAILURE: PAD or an out-of-range id in the Two-Tower head")
    if max(lengths) > HEAD_DEPTH:
        raise SystemExit(f"GATE G4 FAILURE: depth {max(lengths)} exceeds {HEAD_DEPTH}")
    print("  GATE G4 OK  shape/depth/legality verified")
    return heads


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #


def score_arm(
    arm: str,
    sources: Sequence[str],
    rankings: dict[str, list[list[int]]],
    cohort: Sequence[Any],
    num_items: int,
) -> dict[str, Any]:
    """Score one frozen arm through the shared evaluator and keep its per-user evidence."""
    batches = B.arm_fixed_fusion(
        num_items=num_items, source_rankings=rankings, sources=tuple(sources)
    )
    result = evaluate_batched(
        num_items=num_items,
        score_batches=batches(cohort, B.DEFAULT_BATCH_SIZE),
        k_values=K_VALUES,
    )
    ranks = np.asarray(result.target_ranks, dtype=np.int64)
    if ranks.shape[0] != len(cohort):
        raise SystemExit(f"arm {arm}: {ranks.shape[0]} ranks for {len(cohort)} users")
    return {
        "ranks": ranks,
        "report": result.report.as_dict(),
        "sources": tuple(sources),
    }


def arm_artifact(arm: str, payload: dict[str, Any], targets: np.ndarray) -> dict[str, np.ndarray]:
    ranks = payload["ranks"]
    return {
        "case_index": np.arange(ranks.size, dtype=np.int64),
        "target_ids": targets,
        "target_rank": ranks,
        **{f"hit{k}": (ranks <= k) for k in K_VALUES},
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def bootstrap_with_draws(delta: np.ndarray, seed: int, resamples: int) -> tuple[float, float, float, np.ndarray]:
    """The frozen percentile bootstrap, keeping the per-resample means for the artifact.

    The loop mirrors ``m2_paired_uncertainty.paired_bootstrap`` exactly (same RNG, same chunk of
    500, same ``rng.integers`` draw) and the caller asserts the two agree.
    """
    values = np.asarray(delta, dtype=np.float64)
    n = values.size
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 500
    done = 0
    while done < resamples:
        take = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(take, n))
        means[done : done + take] = values[idx].mean(axis=1)
        done += take
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        means,
    )


def paired_statistics(
    stats_module: Any,
    per_arm: dict[str, dict[str, Any]],
    seed: int,
    resamples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Per-K paired deltas, bootstrap CIs and exact McNemar for both predeclared contrasts."""
    results: dict[str, Any] = {}
    draws: dict[str, np.ndarray] = {}

    for before, after in CONTRASTS:
        key = f"{after}_minus_{before}"
        ranks_before = per_arm[before]["ranks"]
        ranks_after = per_arm[after]["ranks"]
        per_k: list[dict[str, Any]] = []

        for k in K_VALUES:
            ndcg_before = stats_module.ndcg_vector(ranks_before, k)
            ndcg_after = stats_module.ndcg_vector(ranks_after, k)
            hit_before = (ranks_before <= k).astype(np.float64)
            hit_after = (ranks_after <= k).astype(np.float64)

            ndcg_mean, ndcg_lo, ndcg_hi, ndcg_draws = bootstrap_with_draws(
                ndcg_after - ndcg_before, seed, resamples
            )
            hit_mean, hit_lo, hit_hi, hit_draws = bootstrap_with_draws(
                hit_after - hit_before, seed, resamples
            )

            # Cross-check against the frozen implementation: identical inputs must give the
            # identical triple, or the artifact's draws would not reproduce the reported CI.
            frozen_ndcg = stats_module.paired_bootstrap(ndcg_after, ndcg_before, seed, resamples)
            frozen_hit = stats_module.paired_bootstrap(hit_after, hit_before, seed, resamples)
            if not (
                math.isclose(ndcg_mean, frozen_ndcg[0], rel_tol=0, abs_tol=1e-15)
                and math.isclose(ndcg_lo, frozen_ndcg[1], rel_tol=0, abs_tol=1e-15)
                and math.isclose(ndcg_hi, frozen_ndcg[2], rel_tol=0, abs_tol=1e-15)
                and math.isclose(hit_mean, frozen_hit[0], rel_tol=0, abs_tol=1e-15)
                and math.isclose(hit_lo, frozen_hit[1], rel_tol=0, abs_tol=1e-15)
                and math.isclose(hit_hi, frozen_hit[2], rel_tol=0, abs_tol=1e-15)
            ):
                raise SystemExit(
                    f"STATISTICS FAILURE: the recorded draws disagree with the frozen "
                    f"implementation at K={k} for {key}"
                )

            b = int(np.sum((ranks_before <= k) & (ranks_after > k)))
            c = int(np.sum((ranks_before > k) & (ranks_after <= k)))
            p_value = stats_module.exact_mcnemar(b, c)

            per_k.append(
                {
                    "k": k,
                    "ndcg_before": float(ndcg_before.mean()),
                    "ndcg_after": float(ndcg_after.mean()),
                    "ndcg_delta": ndcg_mean,
                    "ndcg_ci": (ndcg_lo, ndcg_hi),
                    "recall_before": float(hit_before.mean()),
                    "recall_after": float(hit_after.mean()),
                    "recall_delta": hit_mean,
                    "recall_ci": (hit_lo, hit_hi),
                    "mcnemar_b": b,
                    "mcnemar_c": c,
                    "mcnemar_p": p_value,
                }
            )
            draws[f"{key}|ndcg@{k}"] = ndcg_draws
            draws[f"{key}|hit@{k}"] = hit_draws

        results[key] = {
            "before": before,
            "after": after,
            "held_constant": sorted(set(CONTROLS[before]) & set(CONTROLS[after])),
            "added": sorted(set(CONTROLS[after]) - set(CONTROLS[before])),
            "removed": sorted(set(CONTROLS[before]) - set(CONTROLS[after])),
            "per_k": per_k,
            "classification": stats_module.classify(per_k),
        }
    return results, draws


def promotion_support(contrasts: dict[str, Any]) -> tuple[bool, str]:
    """Apply the preregistration §6.3 rule, and nothing else."""
    primary = contrasts["B_minus_A"]
    replication = contrasts["D_minus_C"]
    primary_k10 = next(r for r in primary["per_k"] if r["k"] == PRIMARY_K)
    replication_k10 = next(r for r in replication["per_k"] if r["k"] == PRIMARY_K)

    if primary["classification"] != "SUPPORTED":
        return False, f"primary B_minus_A is {primary['classification']}, not SUPPORTED"
    if not primary_k10["ndcg_delta"] > 0:
        return False, "primary ΔNDCG@10 is not positive"
    if math.copysign(1.0, replication_k10["ndcg_delta"]) != math.copysign(
        1.0, primary_k10["ndcg_delta"]
    ):
        return False, "replication D_minus_C has the opposite sign at K=10"
    return True, "primary SUPPORTED with a positive ΔNDCG@10 and a same-sign replication"


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def refuse_to_overwrite(paths: Iterable[Path]) -> None:
    """Amendment A1: existing B1 evidence is never overwritten or deleted."""
    existing = [path for path in paths if path.exists()]
    if existing:
        raise SystemExit(
            "REFUSING TO OVERWRITE existing B1 evidence:\n  "
            + "\n  ".join(str(path) for path in existing)
            + "\nA re-run is a new milestone, not an edit of this one."
        )


def read_freeze_manifest(path: Path) -> dict[str, Any]:
    """Load the pre-execution freeze record and check that the protocol has not been edited.

    Two things are enforced here rather than trusted:

    * the preregistration on disk still hashes to the value the freeze record binds — so a protocol
      edited after freezing cannot be executed under the old record;
    * the record does not already describe a completed run, so a second execution cannot overwrite
      this milestone's evidence.
    """
    if not path.is_file():
        raise SystemExit(
            f"no freeze manifest at {path}: the protocol must be frozen before it is executed"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    frozen_hash = record.get("preregistration", {}).get("sha256")
    current_hash = sha256_file(REPO / "docs" / "B1_PREREGISTRATION.md")
    if frozen_hash != current_hash:
        raise SystemExit(
            "FREEZE FAILURE: docs/B1_PREREGISTRATION.md changed after it was frozen\n"
            f"  frozen   {frozen_hash}\n  current  {current_hash}\n"
            "An edit after freezing invalidates this protocol; stop and re-freeze explicitly."
        )
    if record.get("status") == "COMPLETE":
        raise SystemExit(
            f"{path} already records a completed run; a second execution is a new milestone"
        )
    print(f"\nFREEZE - protocol bound to sha256 {frozen_hash}")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "b1_evidence")
    parser.add_argument("--report", type=Path, default=REPO / "docs" / "reports" / "b1-fusion-paired-evidence.txt")
    parser.add_argument("--seed", type=int, default=20260201)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--gate-only", action="store_true", help="integrity + cohort, no retrieval")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    stats_module = _load_m2_statistics()
    if (args.seed, args.resamples) != (20260201, 10_000):
        raise SystemExit("the frozen protocol fixes seed=20260201 and resamples=10000")

    observed_inputs = gate_inputs()
    cohort, num_items, targets, cohort_description = build_cohort()
    if args.gate_only:
        print("\n--gate-only: integrity and cohort verified; no retrieval performed")
        return 0

    targets_digest = sha256_array(targets)
    rankings = load_frozen_heads(len(cohort))
    rankings["two_tower"] = compute_two_tower_head(cohort, num_items, quiet=args.quiet)

    out: Path = args.out
    manifest_path = out / "B1_EXECUTION_MANIFEST.json"
    paths = [
        out / "two_tower_head.npz",
        out / "arm_A_hits.npz",
        out / "arm_B_hits.npz",
        out / "arm_C_hits.npz",
        out / "arm_D_hits.npz",
        out / "paired_inputs.npz",
        out / "paired_bootstrap.npz",
        out / "mcnemar.json",
        out / "b1_paired_statistics.json",
        args.report,
    ]
    refuse_to_overwrite(paths)
    freeze_manifest = read_freeze_manifest(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # ---- the Two-Tower head artifact (the gap this milestone exists to close) ---- #
    tt_lengths = np.asarray([len(head) for head in rankings["two_tower"]], dtype=np.int32)
    tt_matrix = np.zeros((len(rankings["two_tower"]), HEAD_DEPTH), dtype=np.int32)
    for row, head in enumerate(rankings["two_tower"]):
        tt_matrix[row, : len(head)] = head
    np.savez_compressed(
        out / "two_tower_head.npz", two_tower=tt_matrix, two_tower_lengths=tt_lengths
    )

    # ---- arms ---- #
    print("\nARMS - frozen RRF over the frozen heads")
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in ("A", "B", "C", "D"):
        began = time.time()
        per_arm[arm] = score_arm(arm, CONTROLS[arm], rankings, cohort, num_items)
        recall10 = per_arm[arm]["report"]["metrics"]["Recall"]["@10"]
        print(
            f"  {arm} = {' + '.join(CONTROLS[arm]):50s} R@10={recall10:.5f} "
            f"({time.time() - began:.1f}s)"
        )
        np.savez_compressed(
            out / f"arm_{arm}_hits.npz", **arm_artifact(arm, per_arm[arm], targets)
        )

    # ---- gate G1: the instrument must agree with the frozen pipeline ---- #
    print("\nGATE G1 - arm A reproduces the frozen Phase-5 fixed-fusion evidence")
    frozen = np.load(FROZEN_ARM_A_HITS)
    frozen_rank = frozen["target_rank"].astype(np.int64)
    if not np.array_equal(frozen["target_ids"].astype(np.int64), targets):
        raise SystemExit("GATE G1 FAILURE: the frozen evidence is for a different cohort")
    mismatches = int(np.sum(frozen_rank != per_arm["A"]["ranks"]))
    print(f"  target_rank mismatches: {mismatches} / {frozen_rank.size}")
    if mismatches:
        raise SystemExit(
            "GATE G1 FAILURE: arm A does not reproduce the frozen pipeline. "
            "Stopping without reporting any statistic."
        )
    print("  GATE G1 OK  arm A reproduces the frozen target ranks exactly")

    # ---- paired statistics ---- #
    print(f"\nSTATISTICS - {args.resamples:,} resamples, seed {args.seed}")
    contrasts, draws = paired_statistics(stats_module, per_arm, args.seed, args.resamples)
    supported, reason = promotion_support(contrasts)

    np.savez_compressed(
        out / "paired_inputs.npz",
        case_index=np.arange(targets.size, dtype=np.int64),
        target_ids=targets,
        **{
            f"{arm}_target_rank": per_arm[arm]["ranks"]
            for arm in ("A", "B", "C", "D")
        },
    )
    np.savez_compressed(out / "paired_bootstrap.npz", **draws)
    (out / "mcnemar.json").write_text(
        json.dumps(
            {
                f"{key}": [
                    {"k": row["k"], "b": row["mcnemar_b"], "c": row["mcnemar_c"],
                     "p": row["mcnemar_p"]}
                    for row in payload["per_k"]
                ]
                for key, payload in contrasts.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for key, payload in contrasts.items():
        print(f"\n{key}: {payload['classification']}")
        for row in payload["per_k"]:
            print(
                f"  K={row['k']:2d}  NDCG {row['ndcg_before']:.6f} -> {row['ndcg_after']:.6f} "
                f"(Δ {row['ndcg_delta']:+.6f}, CI [{row['ndcg_ci'][0]:+.6f}, {row['ndcg_ci'][1]:+.6f}])"
                f"  hit {row['recall_delta']:+.5f} "
                f"(CI [{row['recall_ci'][0]:+.5f}, {row['recall_ci'][1]:+.5f}])"
                f"  McNemar b={row['mcnemar_b']} c={row['mcnemar_c']} p={row['mcnemar_p']:.6g}"
            )
    print(f"\nPROMOTION SUPPORT = {'YES' if supported else 'NO'} ({reason})")

    statistics_payload = {
        "format": "agentrecx.b1.paired_statistics.v1",
        "preregistration": "docs/B1_PREREGISTRATION.md",
        "primary": {"contrast": "B_minus_A", "k": PRIMARY_K, "endpoint": "ndcg"},
        "replication": {"contrast": "D_minus_C", "k": PRIMARY_K, "endpoint": "ndcg"},
        "seed": args.seed,
        "resamples": args.resamples,
        "execution_pass": True,
        "promotion_support": supported,
        "promotion_reason": reason,
        "contrasts": {
            key: {
                **{k: v for k, v in payload.items() if k != "per_k"},
                "per_k": [
                    {**row, "ndcg_ci": list(row["ndcg_ci"]), "recall_ci": list(row["recall_ci"])}
                    for row in payload["per_k"]
                ],
            }
            for key, payload in contrasts.items()
        },
    }
    (out / "b1_paired_statistics.json").write_text(
        json.dumps(statistics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- report ---- #
    lines = [
        "B1 - TWO-TOWER FUSION PAIRED EVIDENCE (frozen protocol, executed once)",
        "=" * 92,
        "",
        "Protocol      docs/B1_PREREGISTRATION.md (incl. amendment A1, persistence policy)",
        f"Cohort        {len(cohort):,} users, seed 20260201, target_ids sha256 {targets_digest}",
        f"Catalogue     {num_items:,} items, full-catalogue ranking, PAD = 0 never a candidate",
        f"Fusion        reciprocal_rank_fusion, RRF_K = 60, head = {HEAD_DEPTH}",
        f"Statistics    paired user bootstrap, {args.resamples:,} resamples, seed {args.seed};"
        " exact McNemar",
        "",
        "Arm metrics through the shared evaluator",
        "-" * 92,
    ]
    for arm in ("A", "B", "C", "D"):
        metrics = per_arm[arm]["report"]["metrics"]
        lines.append(
            f"  {arm} = {' + '.join(CONTROLS[arm])}"
        )
        lines.append(
            "      R@5/10/20 = "
            f"{metrics['Recall']['@5']:.5f} / {metrics['Recall']['@10']:.5f} / "
            f"{metrics['Recall']['@20']:.5f}    NDCG@5/10/20 = "
            f"{metrics['NDCG']['@5']:.6f} / {metrics['NDCG']['@10']:.6f} / "
            f"{metrics['NDCG']['@20']:.6f}"
        )
    lines += ["", "Paired contrasts", "-" * 92]
    for key, payload in contrasts.items():
        lines.append(f"  {key}  ({payload['classification']})")
        for row in payload["per_k"]:
            lines.append(
                f"    K={row['k']:2d}  ΔNDCG {row['ndcg_delta']:+.6f} "
                f"CI [{row['ndcg_ci'][0]:+.6f}, {row['ndcg_ci'][1]:+.6f}]   "
                f"Δhit {row['recall_delta']:+.5f} "
                f"CI [{row['recall_ci'][0]:+.5f}, {row['recall_ci'][1]:+.5f}]   "
                f"McNemar b={row['mcnemar_b']} c={row['mcnemar_c']} p={row['mcnemar_p']:.6g}"
            )
    lines += [
        "",
        "Verdicts (independent)",
        "-" * 92,
        "  EXECUTION PASS      = YES  (protocol executed; all gates passed; statistics produced)",
        f"  PROMOTION SUPPORT   = {'YES' if supported else 'NO'}  ({reason})",
        "",
        "  EXECUTION PASS does not imply PROMOTION SUPPORT.",
        "  A NO here is a result, not a failure: it means the predeclared promotion criterion was",
        "  not met, and Serving Fusion V1 stays NO-GO at P1'.",
        "",
        "Gates",
        "-" * 92,
        "  G1 arm A reproduced the frozen Phase-5 target ranks exactly (0 mismatches / 20,000)",
        f"  G2/G5 all frozen input digests matched",
        f"  G3 cohort target_ids sha256 = {targets_digest}",
        "  G4 Two-Tower head depth/legality verified",
        "  G6 metadata head read from the frozen artifact; never recomputed",
        "",
        "Instrument note",
        "-" * 92,
        "  The popularity, SASRec and metadata heads were READ from",
        "  runs/m3_evidence/phase5_fusion_source_heads.npz, not recomputed. The only new retrieval",
        "  is the Two-Tower head. Statistics are the tracked M2 implementation, imported verbatim,",
        "  and the stored bootstrap draws were asserted equal to it.",
        "",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")

    # ---- manifest ---- #
    large_artifacts = []
    for name in (
        "two_tower_head.npz", "arm_A_hits.npz", "arm_B_hits.npz", "arm_C_hits.npz",
        "arm_D_hits.npz", "paired_inputs.npz", "paired_bootstrap.npz",
    ):
        path = out / name
        large_artifacts.append(
            {
                "path": str(path.relative_to(REPO)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "durable_storage_pending": True,
            }
        )
    small_artifacts = []
    for path in (out / "mcnemar.json", out / "b1_paired_statistics.json", args.report):
        small_artifacts.append(
            {
                "path": str(path.relative_to(REPO)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "durable_storage_pending": False,
            }
        )

    # The freeze record is completed in place: the frozen blocks stay byte-identical and only the
    # execution results are added. A mismatch between the frozen protocol and what actually ran is
    # an execution failure, not a footnote.
    frozen_protocol = freeze_manifest.get("protocol", {})
    expected_protocol = {
        "cohort_users": len(cohort),
        "catalogue_items": num_items,
        "head_depth": HEAD_DEPTH,
        "rrf_k": 60,
        "seed": args.seed,
        "resamples": args.resamples,
        "k_values": list(K_VALUES),
        "arms": {arm: list(sources) for arm, sources in CONTROLS.items()},
        "cohort_target_ids_sha256": targets_digest,
        "metadata_head_recomputed": False,
    }
    for field, value in expected_protocol.items():
        if frozen_protocol.get(field) != value:
            raise SystemExit(
                f"FREEZE FAILURE: frozen protocol field {field!r} is "
                f"{frozen_protocol.get(field)!r}, but the run used {value!r}"
            )
    if freeze_manifest.get("checkpoint", {}).get("sha256") != FROZEN_INPUTS[CHECKPOINT]:
        raise SystemExit("FREEZE FAILURE: the frozen checkpoint identity differs from the input")

    manifest = dict(freeze_manifest)
    manifest["status"] = "COMPLETE"
    manifest["inputs_sha256"] = observed_inputs
    manifest["gates"] = {
        "G1_arm_A_reproduces_frozen": {
            "verdict": "PASS", "mismatches": 0, "users": int(targets.size),
        },
        "G2_G5_input_digests": {"verdict": "PASS"},
        "G3_cohort_identity": {"verdict": "PASS", "sha256": targets_digest},
        "G4_two_tower_head": {
            "verdict": "PASS",
            "depth_min": int(tt_lengths.min()),
            "depth_max": int(tt_lengths.max()),
        },
        "G6_metadata_head_not_recomputed": {"verdict": "PASS"},
    }
    manifest["verdicts"] = {
        "execution_pass": True,
        "promotion_support": supported,
        "promotion_reason": reason,
    }
    manifest["classification"] = {key: payload["classification"] for key, payload in contrasts.items()}
    manifest["artifacts"] = {"large_binary": large_artifacts, "small_text_json": small_artifacts}
    manifest["wall_seconds"] = round(time.time() - started, 1)
    (out / "B1_EXECUTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nartifacts written to {out}")
    print(f"report written to {args.report}")
    print(f"wall {manifest['wall_seconds']}s")
    print("EXECUTION PASS = YES")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
