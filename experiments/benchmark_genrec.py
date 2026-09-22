"""Three-way comparison: SASRec vs Two-Tower vs Semantic-ID GenRec.

What this measures
------------------
Standalone metrics for the three learned recommenders under the **frozen** protocol (same dataset,
mapping, `temporal_leave_two_out` split, 20 000-user cohort, full catalogue, `k in {5,10,20}`, and
the accepted evaluator), then the complementarity of their top-1000 heads.

No fusion is run here.  The generative arm is a *new* source, and the question this step asks is
whether it contributes distinct candidates - fusing it would confound "does it retrieve anything
new" with "does the fusion rule help", which is the mistake Step 1.1 had to undo for Two-Tower.

The invalid-generation rate
---------------------------
Generation is constrained to catalogue code prefixes, so a generated Semantic ID always lands on at
least one catalogue item.  The rate is still reported, because a rate that is *asserted* to be zero
is worth measuring: if it were ever non-zero, the constrained decoder would have a hole.

Usage::

    python -m experiments.benchmark_genrec --out runs/genrec_benchmark
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
from experiments.benchmark_public import hit_vector, overlap_table  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
SASREC_CHECKPOINT = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
TWOTOWER_DIR = REPO_ROOT / "runs" / "twotower_public_2026_full"
GENREC_DIR = REPO_ROOT / "runs" / "semantic_id_public_2026"
PHASE5_ARTIFACT = REPO_ROOT / "runs" / "phase5_benchmark_public" / "arms.json"

HEAD = 1000


def rss_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1e6
    return -1.0


def head_overlap(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], *, top: int = 10
) -> dict[str, Any]:
    """Mean Jaccard between two sources' top-``top`` heads, averaged over users."""
    values: list[float] = []
    for a, b in zip(left, right, strict=True):
        head_a, head_b = set(a[:top]), set(b[:top])
        union = head_a | head_b
        values.append(len(head_a & head_b) / len(union) if union else 0.0)
    return {
        "top": top,
        "mean_jaccard": round(sum(values) / len(values), 6) if values else 0.0,
        "users": len(values),
    }


def triple_overlap(flags: dict[str, list[bool]]) -> dict[str, int]:
    """The 2x2x2 partition of users by which of the three sources hit."""
    names = sorted(flags)
    counts = {
        "hit_by_all_three": 0,
        "hit_by_exactly_two": 0,
        "hit_by_exactly_one": 0,
        "hit_by_none": 0,
    }
    per_source_only = {name: 0 for name in names}
    for index in range(len(flags[names[0]])):
        hits = [name for name in names if flags[name][index]]
        if len(hits) == 3:
            counts["hit_by_all_three"] += 1
        elif len(hits) == 2:
            counts["hit_by_exactly_two"] += 1
        elif len(hits) == 1:
            counts["hit_by_exactly_one"] += 1
            per_source_only[hits[0]] += 1
        else:
            counts["hit_by_none"] += 1
    union = len(flags[names[0]]) - counts["hit_by_none"]
    counts["union"] = union
    counts["union_share"] = round(union / len(flags[names[0]]), 6)
    counts["users"] = len(flags[names[0]])
    counts["only_hits"] = per_source_only
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SASRec vs Two-Tower vs GenRec")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "genrec_benchmark")
    parser.add_argument("--cohort", type=int, default=B.DEFAULT_COHORT_SIZE)
    parser.add_argument("--genrec-dir", type=Path, default=GENREC_DIR)
    parser.add_argument("--two-tower-dir", type=Path, default=TWOTOWER_DIR)
    parser.add_argument("--beams-per-level", type=int, default=B.GENREC_BEAMS_PER_LEVEL)
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

    results: dict[str, Any] = {
        "benchmark_version": B.BENCHMARK_VERSION,
        "kind": "post_phase5_semantic_id_comparison",
        "note": (
            "Additive to the accepted Phase-5 artifact and the Step-1.1 Two-Tower artifact; "
            "neither is rewritten."
        ),
        "cohort": selection["description"],
        "catalogue": num_items,
        "k_values": list(B.K_VALUES),
        "protocol": split_report.as_dict(),
        "head": HEAD,
        "arms": {},
        "timings": {},
        "source_target_in_head": {},
        "complementarity": {},
    }
    arms: dict[str, Any] = results["arms"]
    timings: dict[str, float] = results["timings"]

    # ---- standalone arms -------------------------------------------------- #
    sasrec = B.arm_sasrec(checkpoint=SASREC_CHECKPOINT)
    t0 = time.time()
    arms["sequential"] = B.evaluate_arm(cases=cohort, num_items=num_items, batches=sasrec)
    timings["sequential"] = round(time.time() - t0, 1)
    print(f"sequential  {timings['sequential']}s R@10="
          f"{arms['sequential']['metrics']['Recall']['@10']:.5f}", flush=True)

    two_tower = B.arm_two_tower(checkpoint=args.two_tower_dir / "checkpoint.pt")
    t0 = time.time()
    arms["two_tower"] = B.evaluate_arm(cases=cohort, num_items=num_items, batches=two_tower)
    timings["two_tower"] = round(time.time() - t0, 1)
    print(f"two_tower   {timings['two_tower']}s R@10="
          f"{arms['two_tower']['metrics']['Recall']['@10']:.5f}", flush=True)

    t0 = time.time()
    genrec_batches, genrec_diagnostics = B.arm_semantic_id_genrec(
        artifact_dir=args.genrec_dir,
        batch_size=B.DEFAULT_BATCH_SIZE,
        beams_per_level=args.beams_per_level,
    )
    arms["semantic_id_genrec"] = B.evaluate_arm(
        cases=cohort, num_items=num_items, batches=genrec_batches
    )
    timings["semantic_id_genrec"] = round(time.time() - t0, 1)
    results["genrec_diagnostics"] = genrec_batches.finalize()  # type: ignore[attr-defined]
    print(f"genrec      {timings['semantic_id_genrec']}s R@10="
          f"{arms['semantic_id_genrec']['metrics']['Recall']['@10']:.5f}", flush=True)
    print(f"  invalid generation rate: "
          f"{results['genrec_diagnostics']['invalid_generation_rate']}", flush=True)

    # ---- heads and complementarity ---------------------------------------- #
    t0 = time.time()
    heads = {
        "sequential": B.sasrec_top_rankings(
            cases=cohort, scorer=sasrec, num_items=num_items, top_n=HEAD
        ),
        "two_tower": B.sasrec_top_rankings(
            cases=cohort, scorer=two_tower, num_items=num_items, top_n=HEAD
        ),
        "semantic_id_genrec": B.sasrec_top_rankings(
            cases=cohort, scorer=genrec_batches, num_items=num_items, top_n=HEAD
        ),
    }
    timings["heads"] = round(time.time() - t0, 1)
    if not args.quiet:
        print(f"heads {timings['heads']}s rss={rss_gb():.2f}GB", flush=True)

    flags = {name: hit_vector(targets, value) for name, value in heads.items()}
    results["source_target_in_head"] = {
        name: round(sum(values) / len(values), 6) for name, values in flags.items()
    }
    results["complementarity"] = {
        "three_way": triple_overlap(flags),
        "pairwise": {
            "sequential_vs_two_tower": B.overlap_table(flags, "sequential", "two_tower"),
            "sequential_vs_semantic_id_genrec": B.overlap_table(
                flags, "sequential", "semantic_id_genrec"
            ),
            "two_tower_vs_semantic_id_genrec": B.overlap_table(
                flags, "two_tower", "semantic_id_genrec"
            ),
        },
        "head_overlap": {
            "sequential_vs_two_tower": head_overlap(heads["sequential"], heads["two_tower"]),
            "sequential_vs_semantic_id_genrec": head_overlap(
                heads["sequential"], heads["semantic_id_genrec"]
            ),
            "two_tower_vs_semantic_id_genrec": head_overlap(
                heads["two_tower"], heads["semantic_id_genrec"]
            ),
        },
    }
    for name, table in results["complementarity"]["pairwise"].items():
        print(f"  {name}: {json.dumps(table)}", flush=True)
    print(f"  three-way: {json.dumps(results['complementarity']['three_way'])}", flush=True)

    results["phase5_cross_check"] = {}
    if PHASE5_ARTIFACT.exists():
        stored = json.loads(PHASE5_ARTIFACT.read_text(encoding="utf-8"))["arms"]
        for arm in ("sequential", "metadata_retrieval", "fixed_fusion"):
            if arm in stored and arm in arms:
                results["phase5_cross_check"][arm] = {
                    "stored_recall_at_10": stored[arm]["metrics"]["Recall"]["@10"],
                    "rerun_recall_at_10": arms[arm]["metrics"]["Recall"]["@10"],
                }

    results["wall_seconds"] = round(time.time() - started, 1)
    results["peak_rss_gb"] = round(rss_gb(), 2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out / 'comparison.json'} in {results['wall_seconds']}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
