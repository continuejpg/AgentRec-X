"""Derive TIGER-FP32 per-user hit@K indicators from the ACCEPTED certified H7 artifact.

TIGER side only: no baseline is recomputed, nothing is retrieved again, and the certified scores
are used exactly as published.  The hit rule is the frozen evaluator's own rule, restated here
because the batch evaluator reports aggregates rather than per-case ranks:

    candidates = 1..num_items, PAD 0 never a candidate;
    items in the history handed to the evaluator are excluded EXCEPT the target;
    higher score ranks first; equal scores -> lower item_id first; rank is 1-based.

The history the accepted arm hands the evaluator is `case.test_history`, which G16.2 proved
identical to the frozen handoff for all 20 000 cases.

Why a target outside the certified frontier cannot be a hit at K<=20:
the certified frontier is the true top-`required_frontier` by score, and
`required_frontier = K_max + |seen|`, so at least 20 frontier items survive the seen-mask and all
of them outrank every non-frontier item (which carries the documented tail floor).  A target that
is not in the frontier therefore has rank > 20.

The derivation is validated by reproducing the frozen aggregate Recall exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path("/root/autodl-tmp/agentrec-X")
import sys
sys.path.insert(0, str(REPO))

FROZEN = {"@5": 0.00925, "@10": 0.01385, "@20": 0.02260}
FROZEN_HITS = {"@5": 185, "@10": 277, "@20": 452}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=Path("/root/autodl-tmp/step26_h7_canonical"))
    ap.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/step26_tiger_hits.npz"))
    ap.add_argument("--report", type=Path, default=Path("/root/autodl-tmp/gate_d_step26_tiger_hits.json"))
    ap.add_argument("--size", type=int, default=20000)
    args = ap.parse_args()

    import experiments.benchmark_public as B
    from recommendation.evaluation.split import load_cohort_from_artifacts

    cases, split_report = load_cohort_from_artifacts(
        REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json",
        REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json",
    )
    selection = B.cohort_from_cases(cases, size=args.size)
    cohort = list(selection["cases"])
    num_items = int(split_report.catalog_size)

    summary = json.loads((args.artifact / "retrieval_summary.json").read_text())
    tail_score = float(summary["tail_score"])
    records: dict[int, dict] = {}
    with (args.artifact / "retrieval_candidates.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[int(record["case_id"])] = record

    hits = {k: np.zeros(len(cohort), dtype=bool) for k in (5, 10, 20)}
    target_in_frontier = np.zeros(len(cohort), dtype=bool)
    target_rank = np.full(len(cohort), -1, dtype=np.int64)
    ranks_known = np.zeros(len(cohort), dtype=bool)

    for index, case in enumerate(cohort):
        record = records.get(index)
        if record is None:
            raise SystemExit(f"artifact has no case_id {index}")
        target = int(case.test_target)
        seen = set(int(x) for x in case.test_history)
        seen.discard(target)  # the target always stays eligible
        candidates = []
        for candidate in record["candidates"]:
            item_id = int(candidate["item_id"])
            if item_id in seen:
                continue
            candidates.append((float(candidate["score"]), item_id))
        # descending score, then ascending item id - the frozen tie-break
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        frontier_ids = {item for _, item in candidates}
        if target in frontier_ids:
            target_in_frontier[index] = True
            target_score = None
            for score, item in candidates:
                if item == target:
                    target_score = score
                    break
            higher = sum(1 for score, item in candidates
                         if score > target_score or (score == target_score and item < target))
            rank = 1 + higher
            target_rank[index] = rank
            ranks_known[index] = True
            for k in (5, 10, 20):
                hits[k][index] = rank <= k
        else:
            # tail-scored target: rank is strictly greater than every surviving frontier item
            n_legal_frontier = len(candidates)
            target_rank[index] = n_legal_frontier + 1
            ranks_known[index] = True

    report = {"artifact": str(args.artifact), "users": len(cohort), "num_items": num_items,
              "tail_score": tail_score, "target_in_certified_frontier": int(target_in_frontier.sum())}
    ok = True
    for k in (5, 10, 20):
        count = int(hits[k].sum())
        share = count / len(cohort)
        frozen_share = FROZEN[f"@{k}"]
        match = (count == FROZEN_HITS[f"@{k}"]) and abs(share - frozen_share) < 5e-7
        ok = ok and match
        report[f"hit@{k}"] = {"hits": count, "share": round(share, 6),
                              "frozen_recall": frozen_share,
                              "frozen_hits": FROZEN_HITS[f"@{k}"],
                              "reproduces_frozen": bool(match)}
    report["derivation_validated"] = bool(ok)
    np.savez_compressed(args.out, **{f"tiger_hit{k}": hits[k] for k in (5, 10, 20)},
                        target_in_frontier=target_in_frontier,
                        target_rank=target_rank, ranks_known=ranks_known)
    args.report.write_text(json.dumps(report, indent=1))

    print("=== TIGER-FP32 per-user hit derivation (TIGER side only) ===")
    for k in (5, 10, 20):
        h = report[f"hit@{k}"]
        print("  hit@%-2d  hits %4d / %d  share %.5f   frozen Recall %.5f (%d)  reproduces: %s"
              % (k, h["hits"], len(cohort), h["share"], h["frozen_recall"], h["frozen_hits"],
                 h["reproduces_frozen"]))
    print("  target inside the certified frontier: %d / %d" % (report["target_in_certified_frontier"], len(cohort)))
    print("  DERIVATION VALIDATED:", report["derivation_validated"])
    print("  wrote", args.out, "and", args.report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
