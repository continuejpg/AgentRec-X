"""Phase-5 benchmark runner: every arm under one protocol, results to one artifact.

Usage::

    python experiments/phase5_benchmark.py [cohort_size] [out.json] [--with-metadata]

Defaults to the full 20 000-user cohort and writes
``runs/phase5_benchmark_public/arms.json``.  The ``runs/`` tree is git-ignored on purpose - the
artifact is an experiment output, regenerated from the processed data, the checkpoint and this
script - so the numbers it contains are transcribed into ``docs/PHASE5_HANDOFF.md`` rather than
committed as a file.

The arms run in a fixed order and each is scored by ``recommendation.evaluation.batched`` over the
same cohort, split, catalogue and ``k`` values, so the stored numbers are comparable by construction
rather than by convention.  The runner also records the dataset-manifest hashes and runs the leakage
gate, and it refuses to write metrics if any check fails.

Cost, measured on a CPU-only 24-core / 7 GB host: ~22 minutes and a 3.7 GB peak, of which ~12 minutes
is building the metadata retrieval heads (one BM25 query per recent history item, cached across
users).
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments.benchmark_public as B  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402


def rss_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1e6
    return -1.0


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _timestamps_by_user(sequences_path: Path) -> dict[str, list[int]] | None:
    """Interaction timestamps per user, read from the processed artifact.

    Returns ``None`` when the artifact does not carry timestamps for a user, so the leakage gate
    reports a structural pass rather than pretending it had timestamp evidence it did not have.
    """
    payload = json.loads(sequences_path.read_text(encoding="utf-8"))
    stamps: dict[str, list[int]] = {}
    for record in payload.get("sequences", ()):
        values = record.get("unix_ms")
        if values:
            stamps[str(record.get("user_id"))] = [int(value) for value in values]
    return stamps or None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    size = int(args[0]) if args else B.DEFAULT_COHORT_SIZE
    out = (
        Path(args[1])
        if len(args) > 1
        else REPO / "runs" / "phase5_benchmark_public" / "arms.json"
    )
    with_metadata = "--with-metadata" in flags or size >= B.DEFAULT_COHORT_SIZE

    started = time.time()
    cases, report = load_cohort_from_artifacts(
        REPO / "data/processed/Sports_and_Outdoors_sequences.json",
        REPO / "data/processed/Sports_and_Outdoors_mappings.json",
    )
    selection = B.cohort_from_cases(cases, size=size)
    cohort = list(selection["cases"])
    num_items = report.catalog_size
    print(f"cohort={len(cohort):,} catalogue={num_items:,} rss={rss_gb():.2f}GB", flush=True)

    # ---- dataset identity ------------------------------------------------ #
    # Content hashes, not file names: a name says nothing about which preprocessing run produced
    # these numbers, and the header of the sequences artifact carries the k-core filtering
    # parameters and interaction counts the cohort was drawn from.
    sequences_path = REPO / "data/processed/Sports_and_Outdoors_sequences.json"
    mappings_path = REPO / "data/processed/Sports_and_Outdoors_mappings.json"
    manifest = B.dataset_manifest(
        sequences_path=sequences_path, mappings_path=mappings_path
    )

    # ---- protocol / leakage gate ----------------------------------------- #
    # The experiment refuses to publish a metric when a check fails.  The checks are decided on
    # interaction timestamps when the artifact supplies them, because a repeat purchase legitimately
    # puts the target in the user's own history.
    timestamps = _timestamps_by_user(sequences_path)
    checks = B.leakage_checks(cohort, cases, timestamps=timestamps)
    B.assert_protocol(checks)
    print(
        f"leakage checks passed: timestamped={checks['timestamp_evidence'] is not None} "
        f"checked={(checks['timestamp_evidence'] or {}).get('cases_checked', 0):,} "
        f"repurchases={checks['of_which_repurchase_of_an_earlier_item']}",
        flush=True,
    )

    results: dict[str, object] = {
        "benchmark_version": B.BENCHMARK_VERSION,
        "dataset": manifest,
        "split": report.as_dict(),
        "evaluation_cases": len(cohort),
        "cohort": selection["description"],
        "num_items": num_items,
        "k_values": list(B.K_VALUES),
        "batch_size": B.DEFAULT_BATCH_SIZE,
        "leakage_checks": checks,
        "arms": {},
        "timings": {},
    }
    arms: dict[str, object] = results["arms"]  # type: ignore[assignment]
    timings: dict[str, float] = results["timings"]  # type: ignore[assignment]

    # ---- popularity ------------------------------------------------------ #
    popularity = B.arm_popularity(cohort, num_items=num_items)(cohort)[0]
    t0 = time.time()
    arms["popularity"] = B.evaluate_arm(
        cases=cohort,
        num_items=num_items,
        batches=lambda c: B.ranking_batches(popularity, c, batch_size=B.DEFAULT_BATCH_SIZE),
    )
    timings["popularity"] = round(time.time() - t0, 1)
    print(f"popularity {timings['popularity']}s rss={rss_gb():.2f}GB", flush=True)

    # ---- sequential (accepted SASRec artifact) --------------------------- #
    sasrec = B.arm_sasrec(checkpoint=REPO / "runs/sasrec_canonical_2026/best.pt")
    t0 = time.time()
    arms["sequential"] = B.evaluate_arm(cases=cohort, num_items=num_items, batches=sasrec)
    timings["sequential"] = round(time.time() - t0, 1)
    print(f"sequential {timings['sequential']}s rss={rss_gb():.2f}GB", flush=True)

    if not with_metadata:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"wrote {out} (popularity+sequential only) in {time.time()-started:.1f}s", flush=True)
        return 0

    # ---- metadata retrieval --------------------------------------------- #
    from recommendation.catalog.metadata import MetadataIndex

    metadata = MetadataIndex.load(REPO / "data/processed/Sports_and_Outdoors_products.jsonl")
    t0 = time.time()
    index = B.ItemTextIndex(metadata.records)
    timings["metadata_index_build"] = round(time.time() - t0, 1)
    print(
        f"metadata index {timings['metadata_index_build']}s size={index.size:,} "
        f"term_cap={index.max_document_frequency:,} rss={rss_gb():.2f}GB",
        flush=True,
    )
    mappings = json.loads(
        (REPO / "data/processed/Sports_and_Outdoors_mappings.json").read_text(encoding="utf-8")
    )
    # One metadata pass: its heads serve both the metadata-only arm and the fusion arms, so the two
    # rows of the results table describe one retriever rather than two.
    t0 = time.time()
    meta_heads = B.metadata_fused_heads(
        index=index,
        records=metadata.records,
        id2item=mappings["id2item"],
        item2id=mappings["item2id"],
        cases=cohort,
        top_n=1000,
    )
    timings["metadata_heads"] = round(time.time() - t0, 1)
    print(f"metadata heads {timings['metadata_heads']}s rss={rss_gb():.2f}GB", flush=True)

    t0 = time.time()
    arms["metadata_retrieval"] = B.evaluate_arm(
        cases=cohort,
        num_items=num_items,
        batches=B.arm_metadata_retrieval(num_items=num_items, heads=meta_heads),
    )
    timings["metadata_retrieval"] = round(time.time() - t0, 1)
    print(f"metadata {timings['metadata_retrieval']}s rss={rss_gb():.2f}GB", flush=True)

    # ---- fusion inputs --------------------------------------------------- #
    t0 = time.time()
    sasrec_heads = B.sasrec_top_rankings(
        cases=cohort, scorer=sasrec, num_items=num_items, top_n=1000
    )
    pop_heads = B.popularity_rankings(cohort, num_items=num_items, top_n=1000)
    timings["fusion_inputs"] = round(time.time() - t0, 1)
    print(f"fusion inputs {timings['fusion_inputs']}s rss={rss_gb():.2f}GB", flush=True)

    source_rankings = {
        "popularity": pop_heads,
        "sequential": sasrec_heads,
        "metadata": meta_heads,
    }

    # ---- source diagnostics ---------------------------------------------- #
    # How often each source's own head contains the test target.  This is not a metric the phase
    # asks for; it is what makes a fusion result interpretable, because a fusion cannot rank a
    # target no source retrieved.  It also explains why the content-retrieval arms have a large
    # mean rank: their unretrieved tail is a stated convention, not a ranking.
    results["source_target_in_head"] = {
        name: round(
            sum(
                1
                for ranking, case in zip(rankings, cohort, strict=True)
                if case.test_target in set(ranking)
            )
            / len(cohort),
            6,
        )
        for name, rankings in source_rankings.items()
    }
    results["heads"] = {
        "top_n": 1000,
        "metadata_seeds": 5,
        "metadata_per_seed": 100,
    }
    print(f"target-in-head {results['source_target_in_head']}", flush=True)

    # ---- fixed fusion ---------------------------------------------------- #
    plan = B.agent_selection_plan(cohort)

    # Both fusion arms are scored over the *same* three-source universe, so the only difference
    # between them is which sources each user's run consults.
    results["agent_selection"] = {
        "distribution": plan["distribution"],
        "thresholds": {
            "min_distinct_items": B.SELECTION_MIN_DISTINCT_ITEMS,
            "short_history_items": B.SELECTION_SHORT_HISTORY_ITEMS,
        },
    }

    t0 = time.time()
    arms["fixed_fusion"] = B.evaluate_arm(
        cases=cohort,
        num_items=num_items,
        batches=B.arm_fixed_fusion(num_items=num_items, source_rankings=source_rankings),
    )
    timings["fixed_fusion"] = round(time.time() - t0, 1)
    print(f"fixed_fusion {timings['fixed_fusion']}s rss={rss_gb():.2f}GB", flush=True)

    t0 = time.time()
    arms["agent_selected"] = B.evaluate_arm(
        cases=cohort,
        num_items=num_items,
        batches=B.arm_agent_selected(
            num_items=num_items, source_rankings=source_rankings, plan=plan
        ),
    )
    timings["agent_selected"] = round(time.time() - t0, 1)
    print(
        f"agent_selected {timings['agent_selected']}s rss={rss_gb():.2f}GB "
        f"dist={plan['distribution']}",
        flush=True,
    )

    results["peak_rss_gb"] = round(rss_gb(), 2)
    results["wall_seconds"] = round(time.time() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {out} in {results['wall_seconds']}s peak={peak_gb():.2f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
