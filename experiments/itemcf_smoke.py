"""ItemCF real-artifact integration smoke run (Milestone 2B).

Fits the ItemCF baseline on the train histories of the Milestone 2A evaluation
cohort derived from the Milestone 1.5 preprocessing artifacts, then evaluates
validation and test **exclusively** through the existing unified full-ranking
evaluator.  No evaluation logic is implemented here: this runner only wires the
frozen cohort to the frozen evaluator and prints the resulting report.

Usage::

    .venv/bin/python -m experiments.itemcf_smoke
    .venv/bin/python -m experiments.itemcf_smoke --json out.json

IMPORTANT: the preprocessing fixture is a 100k-record *prefix* of the Sports and
Outdoors category.  Every number printed here is an engineering smoke check on a
non-representative sample, not a benchmark result and not a statement about final
ItemCF quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.baselines.itemcf import fit_from_cohort, make_scorer  # noqa: E402
from recommendation.evaluation import (  # noqa: E402
    DEFAULT_K_VALUES,
    FullRankingEvaluator,
    build_cohort_from_artifacts,
    cohort_summary,
)

DEFAULT_SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_sequences.json"
DEFAULT_MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_mappings.json"


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}s"


def run(sequences_path: Path, mappings_path: Path, k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> dict[str, Any]:
    """Fit and evaluate ItemCF, returning a JSON-serialisable summary."""
    # ---- cohort (Milestone 2A owns the split) ---------------------------- #
    cases, split = build_cohort_from_artifacts(str(sequences_path), str(mappings_path))
    summary = cohort_summary(cases, split)
    if not cases:
        raise SystemExit("no evaluation-eligible users: cannot run the ItemCF smoke test")

    fit_users = len(cases)
    train_interactions = summary["train_history"]["total"]

    # ---- fit (train histories only) -------------------------------------- #
    started = time.perf_counter()
    model, stats = fit_from_cohort(cases, split.catalog_size)
    fit_seconds = time.perf_counter() - started

    # ---- cold-target statistics ------------------------------------------ #
    seen_during_fit = set(model.item_freq)
    validation_cold = sum(1 for c in cases if c.validation_target not in seen_during_fit)
    test_cold = sum(1 for c in cases if c.test_target not in seen_during_fit)

    # ---- evaluation through the frozen evaluator ------------------------- #
    evaluator = FullRankingEvaluator(num_items=split.catalog_size, k_values=k_values)
    score_fn = make_scorer(model)

    started = time.perf_counter()
    validation = evaluator.evaluate(cases, score_fn, mode="validation")
    validation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test = evaluator.evaluate(cases, score_fn, mode="test")
    test_seconds = time.perf_counter() - started

    # ---- metric sanity checks -------------------------------------------- #
    sanity: dict[str, Any] = {}
    checks: list[tuple[str, bool]] = []
    for label, outcome in (("validation", validation), ("test", test)):
        report = outcome.report
        values = [report.metrics[name][k] for name in ("HR", "Recall", "NDCG") for k in k_values]
        checks.append((f"{label}: all metrics finite", all(v == v and abs(v) != float("inf") for v in values)))
        checks.append((f"{label}: all metrics in [0,1]", all(0.0 <= v <= 1.0 for v in values)))
        checks.append((f"{label}: HR@K == Recall@K", all(outcome.hr_recall_agree.values())))
        checks.append((
            f"{label}: HR monotone in K",
            all(report.metrics["HR"][a] <= report.metrics["HR"][b] for a, b in zip(k_values, k_values[1:])),
        ))
        checks.append((
            f"{label}: Recall monotone in K",
            all(report.metrics["Recall"][a] <= report.metrics["Recall"][b] for a, b in zip(k_values, k_values[1:])),
        ))
        checks.append((
            f"{label}: NDCG monotone in K",
            all(report.metrics["NDCG"][a] <= report.metrics["NDCG"][b] for a, b in zip(k_values, k_values[1:])),
        ))
        checks.append((f"{label}: cases == eligible cohort", report.num_cases == fit_users))
    sanity = {"checks": {name: ok for name, ok in checks}, "all_passed": all(ok for _, ok in checks)}

    return {
        "cohort": split.as_dict(),
        "fit": stats.as_dict(),
        "training_coverage": {
            "unique_training_items": stats.num_unique_train_items,
            "catalog_size": split.catalog_size,
            "coverage": round(stats.num_unique_train_items / split.catalog_size, 6),
            "train_interactions": train_interactions,
            "fit_users": fit_users,
        },
        "cold_targets": {
            "validation_cold": validation_cold,
            "validation_cold_pct": round(100.0 * validation_cold / fit_users, 4),
            "test_cold": test_cold,
            "test_cold_pct": round(100.0 * test_cold / fit_users, 4),
        },
        "reports": {
            "validation": validation.report.as_dict(),
            "test": test.report.as_dict(),
        },
        "sanity": sanity,
        "runtime_seconds": {
            "fit": round(fit_seconds, 6),
            "validation_evaluation": round(validation_seconds, 6),
            "test_evaluation": round(test_seconds, 6),
        },
    }


def print_report(payload: dict[str, Any]) -> None:
    """Render the smoke summary as readable text."""
    cohort, fit = payload["cohort"], payload["fit"]
    cov, cold = payload["training_coverage"], payload["cold_targets"]

    print("=" * 74)
    print("ItemCF real-artifact integration smoke (Milestone 2B)")
    print("ENGINEERING SMOKE METRICS on a non-representative 100k prefix fixture.")
    print("NOT benchmark results; NOT a statement about final ItemCF quality.")
    print("=" * 74)
    print()
    print("Cohort (Milestone 2A split)")
    print(f"  total preprocessing users   : {cohort['num_users_total']}")
    print(f"  eligible users (len >= 3)   : {cohort['num_users_eligible']}")
    print(f"  excluded users (len < 3)    : {cohort['num_users_excluded']}")
    print(f"  validation / test cases     : {cohort['num_validation_cases']} / {cohort['num_test_cases']}")
    print(f"  catalog size (num_items)    : {cohort['catalog_size']}")
    print()
    print("Fit (train histories only)")
    print(f"  fit users                   : {cov['fit_users']}")
    print(f"  train interactions          : {cov['train_interactions']}")
    print(f"  unique training items       : {cov['unique_training_items']}")
    print(f"  item training coverage      : {cov['coverage']:.6f} "
          f"({100 * cov['coverage']:.2f}% of the catalog)")
    print(f"  learned similarity pairs    : {fit['num_similarity_pairs']} (undirected)")
    print(f"  min co-occurrence           : {fit['min_cooccurrence']}")
    print()
    print("Cold targets (item unseen during fit)")
    print(f"  validation targets unseen   : {cold['validation_cold']} ({cold['validation_cold_pct']:.2f}%)")
    print(f"  test targets unseen         : {cold['test_cold']} ({cold['test_cold_pct']:.2f}%)")
    print()
    for label in ("validation", "test"):
        report = payload["reports"][label]
        print(f"{label.capitalize()} engineering-smoke metrics")
        print(f"  cases evaluated             : {report['num_cases']}")
        print(f"  mean target rank            : {report['mean_target_rank']:.4f}")
        print(f"  mean candidate count        : {report['mean_num_candidates']:.2f}")
        ks = report["k_values"]
        header = "  metric    " + "".join(f"{f'@{k}':>12s}" for k in ks)
        print(header)
        for name in ("HR", "Recall", "NDCG"):
            row = f"  {name:<9s} " + "".join(f"{report['metrics'][name][f'@{k}']:>12.6f}" for k in ks)
            print(row)
        print()
    runtime = payload["runtime_seconds"]
    print("Runtime")
    print(f"  fit                         : {_fmt(runtime['fit'])}")
    print(f"  validation evaluation       : {_fmt(runtime['validation_evaluation'])}")
    print(f"  test evaluation             : {_fmt(runtime['test_evaluation'])}")
    print()
    print("Metric sanity checks")
    for name, ok in payload["sanity"]["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    print(f"ALL SANITY CHECKS PASSED: {payload['sanity']['all_passed']}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ItemCF integration smoke run (Milestone 2B)")
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--json", type=Path, default=None, help="also write the summary as JSON")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    args = parser.parse_args(argv)

    if not args.sequences.exists() or not args.mappings.exists():
        print(f"missing artifacts:\n  {args.sequences}\n  {args.mappings}", file=sys.stderr)
        return 2

    payload = run(args.sequences, args.mappings, tuple(args.k))
    print_report(payload)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if payload["sanity"]["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
