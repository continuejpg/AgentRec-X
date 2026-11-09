"""Post-hoc bootstrap uncertainty for P6 classical evaluation artifacts.

This is an **analysis** tool, not part of the frozen P6 metric definitions.  It reads a
result directory produced by :mod:`experiments.p6_benchmark.run_classical` and resamples
the per-case metric values with replacement to report a percentile confidence interval
for the cohort mean.

Why it exists
-------------
The P6 v0 cohort is small (160 dev / 40 holdout cases).  At the observed hit rates a
difference of one or two hits moves Recall@10 by more than a percentage point, so a point
estimate alone invites over-reading.  The interval makes that uncertainty explicit.  It
uses only the already-frozen per-case metrics; it cannot change a model score, a ranking
or a protocol decision, and it must never be used to tune the holdout.

The bootstrap is seeded and documented so a reader can reproduce it exactly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]

#: Metrics that are per-case means and therefore bootstrap-able.  CatalogueCoverage@10 is
#: a set-union over the cohort, not a mean of per-case values, so it is not resampled here.
BOOTSTRAP_METRICS: tuple[str, ...] = ("Recall@10", "Recall@50", "NDCG@10", "MRR@10")

CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20261107


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile!r}")
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> dict[str, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Returns ``mean``, ``ci_low`` and ``ci_high``.  The resampling is deterministic for a
    fixed seed and case order.
    """
    if not values:
        raise ValueError("bootstrap needs at least one case")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    rng = random.Random(seed)
    count = len(values)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _index in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": sum(values) / count,
        "ci_low": _percentile(means, alpha),
        "ci_high": _percentile(means, 1.0 - alpha),
    }


def _load_groups(results_dir: Path) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    per_case_path = results_dir / "per_case.jsonl"
    for line in per_case_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["model"]), str(row["variant"]))
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError(f"no per-case rows in {per_case_path}")
    return summary, groups


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc bootstrap uncertainty for a P6 classical result directory"
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    summary, groups = _load_groups(args.results_dir)
    payload: dict[str, Any] = {
        "schema_version": "agentrecx.p6.classical_uncertainty.v0",
        "note": (
            "post-hoc percentile bootstrap over frozen per-case metrics; NOT part of the "
            "frozen P6 protocol, never used for holdout tuning or model selection"
        ),
        "results_dir": str(args.results_dir),
        "split": summary.get("split"),
        "resamples": args.resamples,
        "seed": args.seed,
        "confidence": CONFIDENCE,
        "metrics": list(BOOTSTRAP_METRICS),
        "groups": {},
    }
    for (model, variant), rows in sorted(groups.items()):
        frozen = summary["models"][model]["variants"][variant]["metrics"]
        group: dict[str, Any] = {"cases": len(rows), "metrics": {}}
        for metric in BOOTSTRAP_METRICS:
            values = [float(row["metrics"][metric]) for row in rows]
            interval = bootstrap_ci(values, resamples=args.resamples, seed=args.seed)
            # The frozen aggregate must agree with the per-case mean; a mismatch means the
            # artifact is internally inconsistent and the interval must not be trusted.
            if abs(interval["mean"] - float(frozen[metric])) > 1e-12:
                raise ValueError(
                    f"{model}/{variant}/{metric}: per-case mean {interval['mean']!r} != "
                    f"frozen aggregate {frozen[metric]!r}"
                )
            group["metrics"][metric] = interval
        payload["groups"][f"{model}|{variant}"] = group

    out = args.out or (args.results_dir / "uncertainty.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, group in payload["groups"].items():
        print(f"{key} (n={group['cases']})")
        for metric, interval in group["metrics"].items():
            print(
                f"  {metric:<10s} mean={interval['mean']:.4f} "
                f"95%CI=[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
