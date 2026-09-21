"""Frozen A/B/C/D fusion controls for the Two-Tower incremental-value measurement.

Why this command exists
-----------------------
The Step-1 comparison fused ``SASRec + Two-Tower + metadata`` and compared it with the accepted
Phase-5 ``popularity + SASRec + metadata``.  That comparison **changed two things at once** - it
added a source *and* removed popularity - so it could not attribute the difference to Two-Tower.
This command runs the controlled pairs instead, with the rank-fusion rule, its constant, the head
size, the cohort, the catalogue and the evaluator all frozen:

===========================  ==========================================
A  popularity + SASRec + metadata         (the accepted Phase-5 fusion)
B  popularity + SASRec + metadata + Two-Tower
C  SASRec + metadata
D  SASRec + metadata + Two-Tower
===========================  ==========================================

``B - A`` and ``D - C`` each hold every previous source constant and vary exactly one thing: the
presence of Two-Tower.  ``A`` and ``B`` also keep the accepted Phase-5 source set intact, so the
accepted arm can be re-checked rather than merely quoted.

Nothing is tuned.  The RRF constant is the repository default, the head size is the Phase-5 value,
and no threshold, weight or source order is adjusted between arms.

Usage::

    python -m experiments.fusion_controls --out runs/twotower_fusion_controls
    python -m experiments.fusion_controls --two-tower-dir runs/twotower_public_2026_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.benchmark_public as B  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
SASREC_CHECKPOINT = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
TWOTOWER_DIR = REPO_ROOT / "runs" / "twotower_public_2026_full"
PHASE5_ARTIFACT = REPO_ROOT / "runs" / "phase5_benchmark_public" / "arms.json"
STEP1_ARTIFACT = REPO_ROOT / "runs" / "twotower_benchmark" / "comparison.json"

#: Frozen head size: the Phase-5 value, unchanged, so head-level numbers are comparable.
HEAD = 1000

#: The four controlled source sets.  Order is part of the definition and identical everywhere.
CONTROLS: dict[str, tuple[str, ...]] = {
    "A_popularity_sasrec_metadata": ("popularity", "sequential", "metadata"),
    "B_popularity_sasrec_metadata_two_tower": (
        "popularity",
        "sequential",
        "metadata",
        "two_tower",
    ),
    "C_sasrec_metadata": ("sequential", "metadata"),
    "D_sasrec_metadata_two_tower": ("sequential", "metadata", "two_tower"),
}

#: The controlled contrasts.  Each varies exactly one thing: Two-Tower's presence.
CONTRASTS: tuple[tuple[str, str], ...] = (
    ("A_popularity_sasrec_metadata", "B_popularity_sasrec_metadata_two_tower"),
    ("C_sasrec_metadata", "D_sasrec_metadata_two_tower"),
)


def rss_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1e6
    return -1.0


def metrics_of(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten one arm's metrics into ``{recall@10: ..., ndcg@10: ...}`` for contrast arithmetic."""
    metrics = payload["metrics"]
    flat: dict[str, float] = {}
    for name in ("Recall", "NDCG", "HR"):
        for cutoff, value in metrics[name].items():
            flat[f"{name.lower()}{cutoff}"] = float(value)
    flat["mean_target_rank"] = float(payload["mean_target_rank"])
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen A/B/C/D fusion controls")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "twotower_fusion_controls")
    parser.add_argument("--two-tower-dir", type=Path, default=TWOTOWER_DIR)
    parser.add_argument("--cohort", type=int, default=B.DEFAULT_COHORT_SIZE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    cases, split_report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    selection = B.cohort_from_cases(cases, size=args.cohort)
    cohort = list(selection["cases"])
    num_items = split_report.catalog_size
    targets = [case.test_target for case in cohort]
    del cases
    if not args.quiet:
        print(f"cohort={len(cohort):,} catalogue={num_items:,} rss={rss_gb():.2f}GB", flush=True)

    # ---- per-source heads (the fusion inputs) ----------------------------- #
    t0 = time.time()
    sasrec = B.arm_sasrec(checkpoint=SASREC_CHECKPOINT)
    two_tower = B.arm_two_tower(checkpoint=args.two_tower_dir / "checkpoint.pt")
    sasrec_heads = B.sasrec_top_rankings(
        cases=cohort, scorer=sasrec, num_items=num_items, top_n=HEAD
    )
    two_tower_heads = B.sasrec_top_rankings(
        cases=cohort, scorer=two_tower, num_items=num_items, top_n=HEAD
    )
    popularity_heads = B.popularity_rankings(cohort, num_items=num_items, top_n=HEAD)
    if not args.quiet:
        print(f"model heads {time.time() - t0:.1f}s rss={rss_gb():.2f}GB", flush=True)

    from recommendation.catalog import MetadataIndex

    metadata = MetadataIndex.load(PRODUCTS)
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    meta_heads = B.metadata_fused_heads(
        index=B.ItemTextIndex(metadata.records),
        records=metadata.records,
        id2item=mappings["id2item"],
        item2id=mappings["item2id"],
        cases=cohort,
        top_n=HEAD,
    )
    if not args.quiet:
        print(f"metadata heads ready rss={rss_gb():.2f}GB", flush=True)

    rankings = {
        "popularity": popularity_heads,
        "sequential": sasrec_heads,
        "metadata": meta_heads,
        "two_tower": two_tower_heads,
    }

    results: dict[str, Any] = {
        "benchmark_version": B.BENCHMARK_VERSION,
        "kind": "frozen_fusion_controls",
        "question": (
            "what is the incremental value of adding Two-Tower to an existing fusion, holding "
            "every other source constant?"
        ),
        "cohort": selection["description"],
        "catalogue": num_items,
        "k_values": list(B.K_VALUES),
        "protocol": split_report.as_dict(),
        "head": HEAD,
        "rrf_rule": "recommendation.control.candidate_ledger.reciprocal_rank_fusion",
        "rrf_constant": "accepted default (RRF_K), unchanged",
        "tuned_on_test": False,
        "controls": {name: list(sources) for name, sources in CONTROLS.items()},
        "arms": {},
        "contrasts": {},
        "timings": {},
    }
    arms: dict[str, Any] = results["arms"]
    timings: dict[str, float] = results["timings"]

    for name, sources in CONTROLS.items():
        t0 = time.time()
        arms[name] = B.evaluate_arm(
            cases=cohort,
            num_items=num_items,
            batches=B.arm_fixed_fusion(
                num_items=num_items, source_rankings=rankings, sources=sources
            ),
        )
        timings[name] = round(time.time() - t0, 1)
        recall10 = arms[name]["metrics"]["Recall"]["@10"]
        print(f"{name:42s} R@10={recall10:.5f}  ({timings[name]}s)", flush=True)

    # ---- controlled contrasts -------------------------------------------- #
    for left, right in CONTRASTS:
        before = metrics_of(arms[left])
        after = metrics_of(arms[right])
        delta = {
            key: {
                "before": before[key],
                "after": after[key],
                "absolute": round(after[key] - before[key], 6),
                "relative_percent": (
                    round((after[key] - before[key]) / before[key] * 100.0, 3)
                    if before[key]
                    else None
                ),
            }
            for key in ("recall@10", "recall@20", "ndcg@10", "hr@10")
        }
        results["contrasts"][f"{right}_minus_{left}"] = {
            "held_constant": sorted(set(CONTROLS[left]) & set(CONTROLS[right])),
            "added": sorted(set(CONTROLS[right]) - set(CONTROLS[left])),
            "removed": sorted(set(CONTROLS[left]) - set(CONTROLS[right])),
            "metrics": delta,
        }
        print(
            f"  {right} - {left}: "
            f"R@10 {before['recall@10']:.5f} -> {after['recall@10']:.5f} "
            f"({delta['recall@10']['relative_percent']:+.2f}%)",
            flush=True,
        )

    # ---- cross-checks against the frozen artifacts ------------------------- #
    cross: dict[str, Any] = {}
    if PHASE5_ARTIFACT.exists():
        stored = json.loads(PHASE5_ARTIFACT.read_text(encoding="utf-8"))["arms"]["fixed_fusion"]
        cross["phase5_fixed_fusion"] = {
            "stored_recall_at_10": stored["metrics"]["Recall"]["@10"],
            "control_A_recall_at_10": arms["A_popularity_sasrec_metadata"]["metrics"]["Recall"]["@10"],
            "matches": (
                stored["metrics"]["Recall"]["@10"]
                == arms["A_popularity_sasrec_metadata"]["metrics"]["Recall"]["@10"]
            ),
        }
    if STEP1_ARTIFACT.exists():
        step1 = json.loads(STEP1_ARTIFACT.read_text(encoding="utf-8"))
        if "sasrec_two_tower_metadata" in step1.get("arms", {}):
            cross["step1_uncontrolled_fusion"] = {
                "note": (
                    "Step 1 compared 'SASRec+TwoTower+metadata' against the Phase-5 "
                    "'popularity+SASRec+metadata' fusion, which changed two things at once; its "
                    "artifact is preserved unchanged and is not used for attribution here."
                ),
                "step1_recall_at_10": step1["arms"]["sasrec_two_tower_metadata"]["metrics"][
                    "Recall"
                ]["@10"],
            }
    results["cross_checks"] = cross

    results["wall_seconds"] = round(time.time() - started, 1)
    results["peak_rss_gb"] = round(rss_gb(), 2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "controls.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out / 'controls.json'} in {results['wall_seconds']}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
