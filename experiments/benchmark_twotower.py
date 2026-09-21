"""Post-Phase-5 Two-Tower comparison and complementarity analysis.

What this adds, and what it does not touch
-----------------------------------------
The accepted Phase-5 artifact (``runs/phase5_benchmark_public/arms.json``) is **not**
rewritten.  This command produces a *second*, additive artifact that re-runs the accepted
arms under the identical protocol - same dataset, mapping, ``temporal_leave_two_out`` split,
20 000-user deterministic cohort, full catalogue, ``k in {5,10,20}`` and evaluator
semantics - and adds Two-Tower beside them.

Why re-run the accepted arms instead of quoting the stored numbers
-----------------------------------------------------------------
So that both arms in one table were produced by one process in one sitting.  Re-running also
means the comparison is *verifiable*: if this command's SASRec row disagreed with the stored
Phase-5 row, that would be a wiring bug worth seeing rather than a number worth trusting.  The
stored Phase-5 values are therefore reported alongside as a cross-check.

The complementarity analysis
----------------------------
Standalone Recall is not the question; **whether Two-Tower retrieves targets SASRec misses**
is.  For each user the command records whether the target falls inside each source's top-``N``
head and reports the 2x2 hit table plus the candidate-head overlap, which is the quantity that
decides whether adding this arm to a fusion could help at all.

Fusion
------
``sasrec_two_tower_metadata`` fuses SASRec, Two-Tower and the metadata retriever with the
repository's accepted reciprocal-rank-fusion rule and the accepted default constant.  Nothing
is tuned on the test split: the fusion rule, the constant and the head size are the values
already in the codebase, and the only choice made here - which three sources to fuse - is
stated in the artifact.

Usage::

    python -m experiments.benchmark_twotower
    python -m experiments.benchmark_twotower --cohort 20000 --out runs/twotower_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.benchmark_public as B  # noqa: E402
from recommendation.evaluation.split import EvaluationCase, load_cohort_from_artifacts  # noqa: E402

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
SASREC_CHECKPOINT = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
TWOTOWER_DIR = REPO_ROOT / "runs" / "twotower_public_2026"
PHASE5_ARTIFACT = REPO_ROOT / "runs" / "phase5_benchmark_public" / "arms.json"

#: Where the fused arms' heads come from.  1000 is the Phase-5 value, kept identical so the
#: two artifacts' head-level numbers are comparable.
HEAD = 1000


def rss_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1e6
    return -1.0


def hit_vector(targets: Sequence[int], heads: Sequence[Sequence[int]]) -> list[bool]:
    """``True`` where a user's target falls inside that source's head."""
    return [target in set(head) for target, head in zip(targets, heads, strict=True)]


def overlap_table(flags: dict[str, list[bool]], left: str, right: str) -> dict[str, int]:
    """The 2x2 hit table between two sources, over the users both cover."""
    both = sum(1 for a, b in zip(flags[left], flags[right], strict=True) if a and b)
    only_left = sum(1 for a, b in zip(flags[left], flags[right], strict=True) if a and not b)
    only_right = sum(1 for a, b in zip(flags[left], flags[right], strict=True) if b and not a)
    neither = sum(1 for a, b in zip(flags[left], flags[right], strict=True) if not a and not b)
    union = both + only_left + only_right
    total = len(flags[left])
    return {
        "users": total,
        "hit_by_both": both,
        "hit_by_left_only": only_left,
        "hit_by_right_only": only_right,
        "hit_by_neither": neither,
        "hit_by_either": union,
        "jaccard": round(both / union, 6) if union else 0.0,
        "left_share": round((both + only_left) / total, 6) if total else 0.0,
        "right_share": round((both + only_right) / total, 6) if total else 0.0,
        "union_share": round(union / total, 6) if total else 0.0,
        "lift_over_best_single": (
            round(union / max(both + only_left, both + only_right), 6)
            if max(both + only_left, both + only_right)
            else 0.0
        ),
    }


def head_overlap(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], *, top: int = 10) -> dict[str, float]:
    """Mean overlap between two sources' top-``top`` candidate heads.

    Reported as a Jaccard index over that prefix, averaged across users: a low value means the
    two sources propose genuinely different candidates, which is the precondition for fusion to
    help at all.
    """
    if not left:
        return {"mean_jaccard_at_k": 0.0, "users": 0}
    values: list[float] = []
    for a, b in zip(left, right, strict=True):
        head_a, head_b = set(a[:top]), set(b[:top])
        union = head_a | head_b
        values.append(len(head_a & head_b) / len(union) if union else 0.0)
    return {"mean_jaccard_at_k": round(sum(values) / len(values), 6), "users": len(values)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-Tower vs SASRec comparison")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "twotower_benchmark")
    parser.add_argument("--cohort", type=int, default=B.DEFAULT_COHORT_SIZE)
    parser.add_argument("--two-tower-dir", type=Path, default=TWOTOWER_DIR)
    parser.add_argument("--sasrec-checkpoint", type=Path, default=SASREC_CHECKPOINT)
    parser.add_argument("--skip-fusion", action="store_true")
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
        "kind": "post_phase5_two_tower_comparison",
        "note": (
            "Additive to the accepted Phase-5 artifact; the accepted five arms and their "
            "stored results are unchanged."
        ),
        "cohort": selection["description"],
        "catalogue": num_items,
        "k_values": list(B.K_VALUES),
        "protocol": split_report.as_dict(),
        "arms": {},
        "timings": {},
        "complementarity": {},
        "source_target_in_head": {},
    }
    arms: dict[str, Any] = results["arms"]
    timings: dict[str, float] = results["timings"]

    # ---- SASRec (accepted artifact, re-run under this protocol) ---------- #
    sasrec = B.arm_sasrec(checkpoint=args.sasrec_checkpoint)
    t0 = time.time()
    arms["sequential"] = B.evaluate_arm(cases=cohort, num_items=num_items, batches=sasrec)
    timings["sequential"] = round(time.time() - t0, 1)
    print(f"sequential {timings['sequential']}s rss={rss_gb():.2f}GB", flush=True)

    # ---- Two-Tower ------------------------------------------------------- #
    t0 = time.time()
    two_tower_batches = B.arm_two_tower(checkpoint=args.two_tower_dir / "checkpoint.pt")
    arms["two_tower"] = B.evaluate_arm(
        cases=cohort, num_items=num_items, batches=two_tower_batches
    )
    timings["two_tower"] = round(time.time() - t0, 1)
    print(f"two_tower {timings['two_tower']}s rss={rss_gb():.2f}GB", flush=True)

    # ---- metadata retrieval (the accepted Phase-5 arm) ------------------- #
    from recommendation.catalog import MetadataIndex

    metadata = MetadataIndex.load(PRODUCTS)
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    index = B.ItemTextIndex(metadata.records)
    t0 = time.time()
    meta_heads = B.metadata_fused_heads(
        index=index,
        records=metadata.records,
        id2item=mappings["id2item"],
        item2id=mappings["item2id"],
        cases=cohort,
        top_n=HEAD,
    )
    arms["metadata_retrieval"] = B.evaluate_arm(
        cases=cohort,
        num_items=num_items,
        batches=B.arm_metadata_retrieval(num_items=num_items, heads=meta_heads),
    )
    timings["metadata_retrieval"] = round(time.time() - t0, 1)
    print(f"metadata_retrieval {timings['metadata_retrieval']}s rss={rss_gb():.2f}GB", flush=True)

    # ---- heads for the hit analysis and the fused arms ------------------- #
    t0 = time.time()
    sasrec_heads = B.sasrec_top_rankings(
        cases=cohort, scorer=sasrec, num_items=num_items, top_n=HEAD
    )
    two_tower_heads = B.sasrec_top_rankings(
        cases=cohort, scorer=two_tower_batches, num_items=num_items, top_n=HEAD
    )
    pop_heads = B.popularity_rankings(cohort, num_items=num_items, top_n=HEAD)
    timings["heads"] = round(time.time() - t0, 1)
    print(f"heads {timings['heads']}s rss={rss_gb():.2f}GB", flush=True)

    flags = {
        "sequential": hit_vector(targets, sasrec_heads),
        "two_tower": hit_vector(targets, two_tower_heads),
        "metadata_retrieval": hit_vector(targets, meta_heads),
        "popularity": hit_vector(targets, pop_heads),
    }
    results["source_target_in_head"] = {
        name: round(sum(values) / len(values), 6) for name, values in flags.items()
    }
    results["head"] = HEAD
    results["complementarity"] = {
        "sasrec_vs_two_tower": overlap_table(flags, "sequential", "two_tower"),
        "sasrec_vs_metadata": overlap_table(flags, "sequential", "metadata_retrieval"),
        "two_tower_vs_metadata": overlap_table(flags, "two_tower", "metadata_retrieval"),
        "sasrec_vs_two_tower_head_overlap": head_overlap(sasrec_heads, two_tower_heads),
        "sasrec_vs_two_tower_top1_head_overlap": head_overlap(
            sasrec_heads, two_tower_heads, top=1
        ),
    }
    for name, table in results["complementarity"].items():
        if isinstance(table, dict) and "hit_by_either" in table:
            print(f"  {name}: {json.dumps(table)}", flush=True)

    # ---- fusion ---------------------------------------------------------- #
    if not args.skip_fusion:
        # The accepted Phase-5 three-source fusion, re-run for the table.
        t0 = time.time()
        arms["fixed_fusion"] = B.evaluate_arm(
            cases=cohort,
            num_items=num_items,
            batches=B.arm_fixed_fusion(
                num_items=num_items,
                source_rankings={
                    "popularity": pop_heads,
                    "sequential": sasrec_heads,
                    "metadata": meta_heads,
                },
            ),
        )
        timings["fixed_fusion"] = round(time.time() - t0, 1)
        print(f"fixed_fusion {timings['fixed_fusion']}s rss={rss_gb():.2f}GB", flush=True)

        # The new fusion: SASRec + Two-Tower + metadata.  Same rule, same constant, same
        # head size as the accepted fusion - only the source set differs.
        t0 = time.time()
        arms["sasrec_two_tower_metadata"] = B.evaluate_arm(
            cases=cohort,
            num_items=num_items,
            batches=B.arm_fixed_fusion(
                num_items=num_items,
                source_rankings={
                    "sequential": sasrec_heads,
                    "two_tower": two_tower_heads,
                    "metadata": meta_heads,
                },
                sources=("sequential", "two_tower", "metadata"),
            ),
        )
        timings["sasrec_two_tower_metadata"] = round(time.time() - t0, 1)
        print(
            f"sasrec_two_tower_metadata {timings['sasrec_two_tower_metadata']}s "
            f"rss={rss_gb():.2f}GB",
            flush=True,
        )

        fused_flags = {
            "fixed_fusion": hit_vector(
                targets,
                _fused_heads(pop_heads, sasrec_heads, meta_heads),
            ),
            "sasrec_two_tower_metadata": hit_vector(
                targets, _fused_heads(sasrec_heads, two_tower_heads, meta_heads)
            ),
        }
        flags.update(fused_flags)
        results["source_target_in_head"].update(
            {
                name: round(sum(values) / len(values), 6)
                for name, values in fused_flags.items()
            }
        )
        results["fusion_definition"] = {
            "rule": "reciprocal_rank_fusion (recommendation.control.candidate_ledger)",
            "rrf_k": "accepted default, unchanged",
            "head": HEAD,
            "fixed_fusion_sources": ["popularity", "sequential", "metadata"],
            "new_fusion_sources": ["sequential", "two_tower", "metadata"],
            "tuned_on_test": False,
        }

    # ---- cross-check against the stored Phase-5 artifact ----------------- #
    if PHASE5_ARTIFACT.exists():
        stored = json.loads(PHASE5_ARTIFACT.read_text(encoding="utf-8"))
        cross: dict[str, Any] = {}
        for arm in ("sequential", "metadata_retrieval", "fixed_fusion"):
            if arm in stored.get("arms", {}) and arm in arms:
                cross[arm] = {
                    "stored_recall_at_10": stored["arms"][arm]["metrics"]["Recall"]["@10"],
                    "rerun_recall_at_10": arms[arm]["metrics"]["Recall"]["@10"],
                }
        results["phase5_cross_check"] = cross

    results["wall_seconds"] = round(time.time() - started, 1)
    results["peak_rss_gb"] = round(rss_gb(), 2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out / 'comparison.json'} in {results['wall_seconds']}s")
    return 0


def _fused_heads(*ranking_groups: Sequence[Sequence[int]]) -> list[list[int]]:
    """The fused head per user for the given aligned per-source rankings (hit analysis only)."""
    from recommendation.control.candidate_ledger import reciprocal_rank_fusion

    users = len(ranking_groups[0])
    return [
        [
            int(identity)
            for identity, _score in reciprocal_rank_fusion(
                [[str(item) for item in group[index]] for group in ranking_groups]
            )[:HEAD]
        ]
        for index in range(users)
    ]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
