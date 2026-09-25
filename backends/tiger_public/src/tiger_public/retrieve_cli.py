"""Step 2.6 retrieval producer: run APPROXIMATE or CERTIFIED retrieval over the frozen cohort.

This is the *backend side* of the Step-2.6 boundary.  It runs inside the backend virtual
environment (CUDA torch), reads only the frozen handoff artifacts, and writes a compact **sparse**
candidate artifact that the AgentRec-X side consumes - it never masks, never ranks for
presentation, and never computes a metric.

Why the artifact is sparse rather than a dense score matrix
-----------------------------------------------------------
Only the certified frontier carries an exact model score; every other catalogue item keeps a
documented floor.  A dense ``[cohort, num_items+1]`` float32 matrix would be ~12.5 GB for the
frozen cohort.  Writing the frontier instead is bounded and has the useful side effect of making
the certification boundary explicit in the artifact.

Artifacts written to ``--out``
------------------------------
``retrieval_candidates.jsonl``  one record per case, in case order
``retrieval_summary.json``      producer identity, hash chain, certificate, timings, tail policy

Usage
-----
``python -m tiger_public.retrieve_cli --checkpoint-dir DIR --sid DIR --catalogue DIR \\
       --cohort DIR --generator-layout FILE --out DIR --mode CERTIFIED [--limit N]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tiger_public.contracts import CONTRACT_VERSION
from tiger_public.generator_layout import load_generator_layout
from tiger_public.io import read_catalogue, read_eval_cohort, read_semantic_ids, sha256_file
from tiger_public.retrieve import (
    APPROXIMATE,
    CERTIFIED,
    METHOD_APPROXIMATE,
    METHOD_CERTIFIED,
    RetrievalConfig,
    RetrievalError,
    SidResolver,
    TigerCaseScorer,
    aggregate_certificate,
    build_history_input,
    retrieve,
)
from tiger_public.scoring import SCORE_RULE
from tiger_public.tiger import load_generator
from tiger_public.trie import CatalogueTrie

PRODUCER = "tiger_public.retrieve_cli"
#: The documented tail score policy.  A tail item is never a certified candidate, so its exact
#: value is irrelevant to any top-K that the frontier can satisfy - it only has to sit strictly
#: below every certified score and be deterministic.
TAIL_POLICY = "min_certified_score - 1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiger_public.retrieve_cli", description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="the accepted run directory holding tiger.pt and generator_layout.json")
    parser.add_argument("--sid", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=None,
                        help="handoff directory holding eval_cohort.json/; defaults to --catalogue")
    parser.add_argument("--exposure", type=Path, default=None,
                        help="defaults to <catalogue>/train_exposure.jsonl")
    parser.add_argument("--generator-layout", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=(APPROXIMATE, CERTIFIED), default=CERTIFIED)
    parser.add_argument("--k", type=int, default=20, help="APPROXIMATE target depth")
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--prefix-budget", type=int, default=200_000)
    parser.add_argument("--batch-prefixes", type=int, default=256)
    parser.add_argument("--max-hist-items", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N cases only; a gate/debug device, never the canonical run")
    parser.add_argument("--log-every", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout_path = args.generator_layout or (args.checkpoint_dir / "generator_layout.json")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    handoff = args.catalogue
    args.cohort = args.cohort or handoff
    args.exposure = args.exposure or (handoff / "train_exposure.jsonl")
    catalogue = read_catalogue(handoff)
    cohort = read_eval_cohort(args.cohort, catalogue=catalogue)
    semantic_ids = read_semantic_ids(args.sid)
    layout = load_generator_layout(layout_path)
    assignment = semantic_ids.assignment
    if len(assignment) - 1 != catalogue.num_items:
        raise RetrievalError(
            f"the assignment holds {len(assignment) - 1} items but the catalogue declares "
            f"{catalogue.num_items}; refusing to retrieve against a mismatched catalogue"
        )
    trie = CatalogueTrie(assignment, layout=layout)
    resolver = SidResolver(trie)

    generator, metadata, _resume = load_generator(
        out_dir=args.checkpoint_dir, sid_dir=args.sid, exposure_path=args.exposure,
        catalogue_dir=args.catalogue, generator_layout_path=layout_path, device=args.device,
    )
    config = RetrievalConfig(mode=args.mode, k=args.k, beam=args.beam,
                             prefix_budget=args.prefix_budget,
                             batch_prefixes=args.batch_prefixes)

    total_cases = (cohort.cohort_size if args.limit is None
                   else min(int(args.limit), cohort.cohort_size))

    started = time.perf_counter()
    results = []
    records_path = out_dir / "retrieval_candidates.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for index in range(total_cases):
            case_id = int(cohort.case_ids[index])
            history = cohort.test_histories[index]
            frontier = int(cohort.required_frontier[index])
            encoder_tokens = build_history_input(
                history, assignment=assignment, layout=layout,
                max_hist_items=args.max_hist_items,
            )
            scorer = TigerCaseScorer(generator, encoder_tokens=encoder_tokens,
                                     batch_prefixes=args.batch_prefixes, device=args.device)
            result = retrieve(trie, scorer, config=config, required_frontier=frontier,
                              case_id=case_id, resolver=resolver)
            results.append(result)
            handle.write(json.dumps(result.as_dict(), sort_keys=True) + "\n")
            if args.log_every and (index + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "cases_done": index + 1, "of": total_cases,
                    "cases_per_second": round((index + 1) / elapsed, 4),
                    "elapsed_seconds": round(elapsed, 2),
                }), flush=True)
    wall = time.perf_counter() - started

    summary = build_summary(
        args=args, layout=layout, trie=trie, cohort=cohort, catalogue=catalogue,
        metadata=metadata, results=results, wall=wall, total_cases=total_cases,
        candidates_path=records_path, resolver=resolver,
    )
    (out_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: summary[k] for k in (
        "producer", "mode", "method", "status", "comparison_eligible", "cases",
        "wall_seconds", "cases_per_second", "candidate_expansions", "nodes_pruned_by_bound",
        "tail_score")}, sort_keys=True))
    return 0


def build_summary(*, args, layout, trie, cohort, catalogue, metadata, results, wall,
                  total_cases, candidates_path, resolver) -> dict[str, Any]:
    all_scores = [c.score for r in results for c in r.candidates]
    if not all_scores:
        raise RetrievalError("retrieval returned no candidates at all; refusing to write a summary")
    tail_score = float(min(all_scores)) - 1.0
    method = METHOD_CERTIFIED if args.mode == CERTIFIED else METHOD_APPROXIMATE
    block = aggregate_certificate(results, requested_frontier=int(np.max(cohort.required_frontier[:total_cases])),
                                 method=method, status=args.mode)
    # Section 15.2: a budget-exhausted run is automatically downgraded and rewritten as
    # APPROXIMATE.  Certification is never claimed for a truncated search.
    status = args.mode
    if block.budget_exhausted_cases > 0:
        status = APPROXIMATE
    frontiers = [int(cohort.required_frontier[i]) for i in range(total_cases)]
    counts = [len(r.candidates) for r in results]
    return {
        "producer": PRODUCER,
        "contract_version": CONTRACT_VERSION,
        "mode_requested": args.mode,
        "mode": status,
        "method": method,
        "status": status,
        "comparison_eligible": bool(status == CERTIFIED and block.budget_exhausted_cases == 0),
        "cases": total_cases,
        "cohort_size_declared": int(cohort.cohort_size),
        "cohort_seed": int(cohort.cohort_seed),
        "protocol_version": str(cohort.protocol_version),
        "k_values": list(cohort.k_values),
        "num_items": int(catalogue.num_items),
        "catalogue_contract_version": str(catalogue.contract_version),
        "requested_frontier_stats": {
            "min": int(min(frontiers)), "max": int(max(frontiers)),
            "mean": round(sum(frontiers) / len(frontiers), 6),
            "p95": int(np.percentile(frontiers, 95)), "p99": int(np.percentile(frontiers, 99)),
        },
        "certified_items_per_case": {"min": int(min(counts)), "max": int(max(counts)),
                                     "mean": round(sum(counts) / len(counts), 6)},
        "tail_score_policy": TAIL_POLICY,
        "tail_score": tail_score,
        "score_rule": dict(SCORE_RULE),
        "certificate": block.as_dict(),
        "candidate_expansions": int(sum(r.prefixes_expanded for r in results)),
        "nodes_pruned_by_bound": int(sum(r.nodes_pruned_by_bound for r in results)),
        "budget_exhausted_cases": int(block.budget_exhausted_cases),
        "collisions_observed": int(resolver.collisions_observed),
        "trie_stats": trie.stats.as_dict(),
        "wall_seconds": round(wall, 3),
        "cases_per_second": round(total_cases / wall, 4) if wall > 0 else None,
        "retrieval_config": {
            "prefix_budget": int(args.prefix_budget), "batch_prefixes": int(args.batch_prefixes),
            "beam": int(args.beam), "max_hist_items": int(args.max_hist_items),
            "device": str(args.device), "limit": args.limit,
        },
        "hashes": {
            "checkpoint_tiger_pt": sha256_file(args.checkpoint_dir / "tiger.pt"),
            "generator_layout": sha256_file(args.generator_layout or
                                            (args.checkpoint_dir / "generator_layout.json")),
            "sid_layout": sha256_file(args.sid / "layout.json"),
            "sid_semantic_ids": sha256_file(args.sid / "semantic_ids.json"),
            "cohort_eval_cohort_json": sha256_file(args.cohort / "eval_cohort.json"),
            "catalogue_json": sha256_file(args.catalogue / "catalogue.json"),
            "candidates_jsonl": sha256_file(candidates_path),
        },
        "checkpoint_training": {
            "precision": (metadata.get("training") or {}).get("precision"),
            "global_step": (metadata.get("training") or {}).get("global_step"),
            "completed_epoch": (metadata.get("training") or {}).get("completed_epoch"),
        },
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
