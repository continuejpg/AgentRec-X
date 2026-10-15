"""T-03-lite statistical verification for ``t03lite_tiger_increment_v1``.

Reads the frozen experiment inputs (never the TIGER worker, never a new head), recomputes both
arms through the **same** evaluator path, and emits a separate verified artifact.  The original
``runs/t03lite_tiger_increment_v1.json`` is never overwritten.

What this script corrects
-------------------------
1. **Labels.**  The original report mixed two different quantities:

   * *candidate target coverage* - the target appears somewhere in a source's frozen head
     (rank 1..depth).  This is what ``incumbent 1079 / union 1279 / TIGER-only 200`` counted.
   * *final Top-10 recall hits* - the target survives the evaluator's mask and lands inside the
     fused top-10.  This is what Recall@10 counts (0.02020 x 20000 = 404 users).

   ``200`` is coverage, not a Recall gain; the measured Recall gain is ``406 - 404 = 2`` users.
2. **Leave-one-out.**  The original LOO fused the source lists and took the raw top-10, without
   the evaluator's PAD/seen masking and tie-breaking, so ``without_tiger`` (383) could not equal
   the baseline (404).  Here every LOO arm runs through the same evaluator path, and the
   no-TIGER arm is required to reproduce the baseline exactly.

Reported invariants (fail closed)
---------------------------------
* the no-TIGER leave-one-out arm equals the baseline (aggregate metrics and per-user hits);
* final Top-10 hits equal ``Recall@10 * users`` for both arms;
* candidate coverage is never reported as a Recall number;
* both arms use the same user order and target array;
* the bootstrap is seeded and its resample count is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments import benchmark_public as B  # noqa: E402
from experiments import m4_benchmark as M4  # noqa: E402
from experiments.t03lite_tiger_increment import (  # noqa: E402
    ALL_SOURCES,
    HEADS,
    INCUMBENTS,
    MAPPINGS,
    PROFILES,
    SEQUENCES,
    TIGER_HEADS,
    evaluate,
    fused_rankings,
    load_identity_map,
    sha256_file,
    source_rankings_for,
)
from recommendation.control.candidate_ledger import RRF_K  # noqa: E402
from recommendation.evaluation.batched import batched_target_ranks, build_candidate_mask  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

ORIGINAL = REPO / "runs/t03lite_tiger_increment_v1.json"
OUT = REPO / "runs/t03lite_tiger_increment_v1_verified.json"


def hit_vector(rankings: Sequence[Sequence[int]], *, cases: Sequence[Any], num_items: int,
               k: int = 10) -> np.ndarray:
    """Per-user Top-``k`` hits, computed with the **evaluator's own** mask/rank primitives.

    A ranking-derived score vector (strictly decreasing) is masked exactly as ``evaluate_batched``
    masks it, so seen/PAD handling and tie-breaking are the evaluator's, not a local reimplementation.
    """
    rankings = [list(row) for row in rankings]
    histories = [tuple(int(v) for v in case.test_history) for case in cases]
    targets = [int(case.test_target) for case in cases]
    batch = 512
    hits: list[int] = []
    for start in range(0, len(rankings), batch):
        chunk = rankings[start : start + batch]
        rows = len(chunk)
        # The evaluator's primitives operate on torch tensors, so the ranking-derived score vector
        # is built as one: an already-ranked item scores `width - position`, everything else -1.0.
        scores = torch.full((rows, num_items + 1), -1.0, dtype=torch.float64)
        for index, ranking in enumerate(chunk):
            width = len(ranking)
            for position, item in enumerate(ranking):
                if 1 <= int(item) <= num_items:
                    scores[index, int(item)] = float(width - position)
        row_targets = targets[start : start + batch]
        mask = build_candidate_mask(rows, num_items, histories[start : start + batch], row_targets)
        ranks = batched_target_ranks(scores, mask, row_targets)
        hits.extend(int(rank) <= k for rank in ranks)
    return np.asarray(hits, dtype=np.int64)


def coverage(rows: Sequence[Sequence[int]], *, cases: Sequence[Any]) -> dict[str, Any]:
    """Candidate target coverage inside a frozen head (NOT a Recall figure)."""
    targets = [int(case.test_target) for case in cases]
    covered = sum(1 for user, target in enumerate(targets) if target in set(rows[user]))
    return {
        "quantity": "candidate_target_coverage_in_head",
        "users": len(targets),
        "users_whose_target_is_in_the_head": covered,
        "note": "coverage is not Recall@10: the evaluator still masks seen items and ranks the fusion",
    }


def seen_in_head(rows: Sequence[Sequence[int]], *, cases: Sequence[Any]) -> dict[str, Any]:
    histories = [set(int(v) for v in case.test_history) for case in cases]
    users = sum(1 for user, row in enumerate(rows) if any(int(v) in histories[user] for v in row))
    total = sum(sum(1 for v in row if int(v) in histories[user]) for user, row in enumerate(rows))
    duplicates = sum(len(row) - len({int(v) for v in row}) for row in rows)
    return {"users_with_a_seen_item_in_head": users, "seen_candidates_in_head": total,
            "duplicate_candidates_in_head": duplicates}


def verify(*, out: pathlib.Path, resamples: int, seed: int) -> dict[str, Any]:
    cases_list, split = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    cases = list(B.cohort_from_cases(cases_list, size=B.DEFAULT_COHORT_SIZE)["cases"])
    num_items = int(split.catalog_size)
    identity = load_identity_map()
    targets = [int(case.test_target) for case in cases]

    report: dict[str, Any] = {
        "experiment": "t03lite_tiger_increment_v1_verified",
        "source_experiment": {
            "path": str(ORIGINAL.relative_to(REPO)),
            "sha256": sha256_file(ORIGINAL) if ORIGINAL.is_file() else None,
            "note": "read-only: the original artifact is preserved unmodified",
        },
        "artifacts": {
            "heads": {"path": str(HEADS.relative_to(REPO)), "sha256": sha256_file(HEADS)},
            "tiger_heads": {"path": str(TIGER_HEADS.relative_to(REPO)),
                            "sha256": sha256_file(TIGER_HEADS)},
            "mappings": {"path": str(MAPPINGS.relative_to(REPO)), "sha256": sha256_file(MAPPINGS)},
        },
        "protocol": {
            "order_of_operations": "truncate the frozen head to depth -> RRF over source lists -> "
                                   "the evaluator masks PAD/seen and ranks",
            "rrf_k": RRF_K,
            "profiles": {name: dict(depths) for name, depths in PROFILES.items()},
        },
        "profiles": {},
        "statistics": {"resamples": int(resamples), "seed": int(seed),
                       "kind": "user-level paired bootstrap + exact two-sided McNemar"},
        "invariants": {},
        "limitations": [
            "frozen heads replay; no model inference and no TIGER worker",
            "not an online A/B test",
            "primary truncates the three incumbent engines from depth 100 to 20 (baseline != Serving V1)",
            "secondary keeps 100/100/100 and truncates only TIGER to 20 (biased against TIGER)",
            "seen history items are still inside the frozen heads and only masked by the evaluator, "
            "so a source's effective depth is below its nominal depth (measured below); M4's own "
            "head materialisation and the live TIGER source instead filter seen items before ranking",
            "latency is out of scope: no candidate generation is timed",
        ],
    }

    for profile_name, depths in PROFILES.items():
        head = max(depths.values())
        baseline_sources = source_rankings_for(depths, identity=identity, include_tiger=False)
        tiger_sources = source_rankings_for(depths, identity=identity, include_tiger=True)
        baseline_rankings = fused_rankings(baseline_sources, users=len(cases))
        tiger_rankings = fused_rankings(tiger_sources, users=len(cases))

        baseline_payload = evaluate(source_rankings=baseline_sources, cases=cases,
                                    num_items=num_items, head=head)
        tiger_payload = evaluate(source_rankings=tiger_sources, cases=cases,
                                 num_items=num_items, head=head)

        def block(payload: dict[str, Any]) -> dict[str, Any]:
            metrics = payload.get("metrics")
            return metrics if isinstance(metrics, dict) else payload

        def read(metrics: dict[str, Any], name: str, k: int = 10) -> float:
            """Read one metric; the evaluator groups by name and then by K, values may be scalars."""
            node: Any = None
            for key in (name, name.upper(), name.lower()):
                if key in metrics:
                    node = metrics[key]
                    break
            if node is None:
                raise SystemExit(f"metric {name} missing; got {sorted(metrics)}")
            if isinstance(node, dict):
                for candidate in (str(k), k, f"@{k}"):
                    if candidate in node:
                        node = node[candidate]
                        break
                else:
                    raise SystemExit(f"{name} has no K={k}: keys {sorted(node)}")
            try:
                return float(node)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"metric {name} is unreadable ({type(node).__name__}: {node!r})"
                ) from exc

        baseline_hits = hit_vector(baseline_rankings, cases=cases, num_items=num_items)
        tiger_hits = hit_vector(tiger_rankings, cases=cases, num_items=num_items)
        base_recall = read(block(baseline_payload), "Recall")
        tiger_recall = read(block(tiger_payload), "Recall")
        base_ndcg = read(block(baseline_payload), "NDCG")
        tiger_ndcg = read(block(tiger_payload), "NDCG")

        # Invariant 1: final hits must equal Recall@10 x users (one positive per user).
        for label, hits, recall in (("baseline", baseline_hits, base_recall),
                                    ("tiger", tiger_hits, tiger_recall)):
            if abs(hits.sum() / len(hits) - recall) > 5e-9:
                raise SystemExit(
                    f"invariant failed: {label} Top-10 hits {int(hits.sum())} != "
                    f"Recall@10 {recall} x {len(hits)}"
                )

        paired = M4.paired_statistics(
            arm_a=baseline_rankings, arm_b=tiger_rankings, targets=targets,
            k_values=(10,), resamples=resamples, seed=seed,
        )

        # Invariant 2: dropping TIGER must reproduce the baseline exactly, via the *same* path.
        loo: dict[str, Any] = {}
        for dropped in ALL_SOURCES:
            kept = [name for name in ALL_SOURCES if name != dropped]
            subset = {name: tiger_sources[name] for name in kept}
            payload = evaluate(source_rankings=subset, cases=cases, num_items=num_items, head=head)
            subset_rankings = fused_rankings(subset, users=len(cases))
            subset_hits = hit_vector(subset_rankings, cases=cases, num_items=num_items)
            entry = {
                "sources": kept,
                "recall@10": read(block(payload), "Recall"),
                "ndcg@10": read(block(payload), "NDCG"),
                "top10_hits": int(subset_hits.sum()),
            }
            if dropped == "tiger":
                if not np.array_equal(subset_hits, baseline_hits):
                    raise SystemExit(
                        "invariant failed: the no-TIGER leave-one-out arm does not reproduce the "
                        "baseline Top-10 hits"
                    )
                if abs(entry["recall@10"] - base_recall) > 1e-12 or abs(entry["ndcg@10"] - base_ndcg) > 1e-12:
                    raise SystemExit(
                        "invariant failed: the no-TIGER arm's aggregate metrics differ from baseline"
                    )
                entry["matches_baseline"] = True
            loo[dropped] = entry

        report["profiles"][profile_name] = {
            "depth_per_source": dict(depths),
            "baseline": {
                "recall@10": base_recall, "ndcg@10": base_ndcg,
                "top10_hits": int(baseline_hits.sum()),
            },
            "tiger": {
                "recall@10": tiger_recall, "ndcg@10": tiger_ndcg,
                "top10_hits": int(tiger_hits.sum()),
            },
            "paired": {
                "gain_users": int(((tiger_hits == 1) & (baseline_hits == 0)).sum()),
                "loss_users": int(((tiger_hits == 0) & (baseline_hits == 1)).sum()),
                "unchanged_users": int((tiger_hits == baseline_hits).sum()),
                "recall_delta_absolute": tiger_recall - base_recall,
                "recall_delta_relative": ((tiger_recall - base_recall) / base_recall) if base_recall else None,
                "ndcg_delta_absolute": tiger_ndcg - base_ndcg,
                "ndcg_delta_relative": ((tiger_ndcg - base_ndcg) / base_ndcg) if base_ndcg else None,
                "mcnemar_and_bootstrap": paired,
            },
            "candidate_coverage": {
                "baseline_union": coverage(
                    [list(row) for source in INCUMBENTS for row in baseline_sources[source]], cases=cases
                ),
                "tiger_arm_union": coverage(
                    [list(row) for source in ALL_SOURCES for row in tiger_sources[source]], cases=cases
                ),
                "tiger_own_head": coverage(tiger_sources["tiger"], cases=cases),
            },
            "seen_items_inside_heads": {
                **{name: seen_in_head(tiger_sources[name], cases=cases) for name in ALL_SOURCES},
            },
            "leave_one_out": loo,
        }

    # Invariant 3/4/5: alignment and reproducibility are structural here; assert them explicitly.
    users = len(cases)
    if users != 20000:
        raise SystemExit(f"invariant failed: expected the frozen 20 000-user cohort, got {users}")
    with np.load(HEADS, allow_pickle=True) as archive:
        head_users = [int(v) for v in archive["user_int_ids"]]
    if head_users != [int(case.user_int_id) for case in cases]:
        raise SystemExit("invariant failed: head user order differs from the cohort order")
    report["invariants"] = {
        "top10_hits_equal_recall_at_10": True,
        "no_tiger_leave_one_out_equals_baseline": True,
        "coverage_reported_separately_from_recall": True,
        "arms_share_user_order_and_targets": True,
        "bootstrap_seeded": True,
        "users": users,
        "target_sha256": hashlib.sha256(
            np.asarray(targets, dtype=np.int64).tobytes()
        ).hexdigest(),
    }

    verdict_bits = []
    for name, entry in report["profiles"].items():
        paired = entry["paired"]
        bits = paired["mcnemar_and_bootstrap"]
        ci = bits.get("per_k", {}).get("10") or bits.get("per_k", {}).get(10) or {}
        verdict_bits.append({
            "profile": name,
            "recall_delta": paired["recall_delta_absolute"],
            "ndcg_delta": paired["ndcg_delta_absolute"],
            "gain_users": paired["gain_users"],
            "loss_users": paired["loss_users"],
            "ci_keys": sorted(ci) if isinstance(ci, dict) else None,
        })
    report["conclusion"] = {
        "verdict": "evidence_insufficient_for_a_positive_claim",
        "reason": "the paired deltas are ~1e-4 in magnitude with 2 net gain users at depth 20; the "
                  "paired interval and McNemar result are reported above and must be read before "
                  "any claim is made",
        "per_profile": verdict_bits,
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--resamples", type=int, default=M4.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=M4.BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    report = verify(out=args.out, resamples=args.resamples, seed=args.seed)
    for name, entry in report["profiles"].items():
        paired = entry["paired"]
        print(f"[{name}] baseline hits={entry['baseline']['top10_hits']} "
              f"tiger hits={entry['tiger']['top10_hits']} "
              f"gain={paired['gain_users']} loss={paired['loss_users']} "
              f"recallΔ={paired['recall_delta_absolute']:+.6f} ndcgΔ={paired['ndcg_delta_absolute']:+.6f}")
        print("   mcnemar/bootstrap:", json.dumps(paired["mcnemar_and_bootstrap"], default=str)[:400])
        print("   loo without_tiger:", json.dumps(entry["leave_one_out"]["tiger"], default=str)[:200])
        print("   seen-in-head:", json.dumps(entry["seen_items_inside_heads"], default=str)[:300])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
