"""Real-checkpoint SASRec inference smoke and CPU latency measurement (Milestone 6).

Loads the accepted Milestone 5 `best.pt`, verifies its identity, runs one bounded
inference example, and measures CPU latency after warmup.

This is an **inference integration and latency** check, not a recommendation-quality
benchmark.  No metric is recomputed and the formal Milestone 5 result is untouched.

Usage::

    .venv/bin/python -m experiments.sasrec_inference_smoke
    .venv/bin/python -m experiments.sasrec_inference_smoke --k 10 --requests 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from recommendation.inference import (  # noqa: E402
    InferenceConfig,
    RequestValidationError,
    SASRecInferenceEngine,
)

#: Accepted Milestone 5 artifacts.
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
DEFAULT_MANIFEST = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "run.json"
DEFAULT_MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
DEFAULT_SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

#: Expected digest of the accepted formal checkpoint.
ACCEPTED_SHA256 = "352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912"


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile over a sorted list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_benchmark_histories(limit: int) -> list[list[str]]:
    """Load a bounded set of real histories from the frozen cohort artifact.

    Uses the **test-style** inference history ``train_history + validation_target``,
    which is the documented protocol for reproducing a test-case input.  The test
    target is deliberately NOT appended.
    """
    if not DEFAULT_SEQUENCES.exists():
        return []
    with open(DEFAULT_SEQUENCES, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    histories: list[list[str]] = []
    for record in payload["sequences"][:limit]:
        asins = record["parent_asins"]
        if len(asins) >= 3:
            histories.append(list(asins[:-2]) + [asins[-2]])  # train_history + validation
    return histories


def main(argv: list[str] | None = None) -> int:
    """Run the smoke; returns 0 when every integration check passes."""
    parser = argparse.ArgumentParser(description="SASRec real-checkpoint inference smoke")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--requests", type=int, default=60, help="bounded latency sample size")
    parser.add_argument("--histories", type=int, default=25, help="real histories to use")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 6 SASRec inference smoke (real accepted checkpoint)")
    print("Integration + latency only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    for path in (args.checkpoint, args.mappings):
        if not path.exists():
            print(f"missing artifact: {path}", file=sys.stderr)
            return 2

    # ---- load ------------------------------------------------------------ #
    started = time.perf_counter()
    engine = SASRecInferenceEngine(
        InferenceConfig(
            checkpoint_path=args.checkpoint,
            mappings_path=args.mappings,
            manifest_path=args.manifest if args.manifest.exists() else None,
            device=args.device,
            expected_checkpoint_sha256=(
                ACCEPTED_SHA256 if args.checkpoint == DEFAULT_CHECKPOINT else None
            ),
        )
    )
    startup_seconds = time.perf_counter() - started

    checks: dict[str, bool] = {}
    checks["checkpoint SHA-256 matches accepted digest"] = (
        args.checkpoint != DEFAULT_CHECKPOINT or engine.checkpoint_sha256 == ACCEPTED_SHA256
    )
    checks["model in eval mode"] = engine.model.training is False
    checks["no parameter requires grad"] = all(
        not p.requires_grad for p in engine.model.parameters()
    )

    print(f"\ncheckpoint      : {args.checkpoint.name}")
    print(f"checkpoint sha  : {engine.checkpoint_sha256}")
    print(f"device          : {engine.device}")
    print(f"catalog size    : {engine.num_items:,}")
    print(f"max_seq_len     : {engine.max_seq_len}")
    parameter_count = engine.model_metadata()["parameter_count"]
    print(f"parameters      : {parameter_count:,} (serve-frozen)")
    print(f"startup (load)  : {startup_seconds:.2f}s")

    # ---- one real recommendation ---------------------------------------- #
    histories = load_benchmark_histories(args.histories)
    if not histories:
        print("no benchmark histories available; skipping real-history checks", file=sys.stderr)
        return 2

    example = histories[0]
    result = engine.recommend(example, k=args.k)
    checks["full score vector finite"] = True  # enforced inside score_catalog
    checks["returned <= requested k"] = result.returned_k <= result.requested_k
    checks["no PAD in recommendations"] = all(r.item_id != 0 for r in result.recommendations)
    checks["no seen item recommended"] = not (
        {r.parent_asin for r in result.recommendations} & set(example)
    )
    checks["asin/item_id mapping consistent"] = all(
        r.parent_asin == engine.item_id_to_parent_asin(r.item_id) for r in result.recommendations
    )

    print(f"\nexample history : {len(example)} items "
          f"(effective {result.effective_history_length}, truncated={result.history_truncated})")
    print(f"top-{min(args.k, result.returned_k)} recommendations:")
    for item in result.recommendations[: min(args.k, 10)]:
        print(f"  {item.rank:2d}. item_id={item.item_id:<7d} {item.parent_asin}  "
              f"score={item.score:+.4f}")

    # ---- determinism ----------------------------------------------------- #
    repeat = engine.recommend(example, k=args.k)
    checks["repeated request deterministic"] = (
        [r.as_dict() for r in repeat.recommendations]
        == [r.as_dict() for r in result.recommendations]
    )

    # ---- latency (bounded, after warmup) -------------------------------- #
    warmup = min(5, len(histories))
    for history in histories[:warmup]:
        engine.recommend(history, k=args.k)

    scoring_ms: list[float] = []
    ranking_ms: list[float] = []
    end_to_end_ms: list[float] = []
    for index in range(args.requests):
        history = histories[index % len(histories)]
        started = time.perf_counter()
        outcome = engine.recommend(history, k=args.k)
        end_to_end_ms.append((time.perf_counter() - started) * 1000.0)
        scoring_ms.append(outcome.timings_ms["scoring"])
        ranking_ms.append(outcome.timings_ms["ranking"])

    import os

    latency = {
        "requests": len(end_to_end_ms),
        "catalog_size": engine.num_items,
        "k": args.k,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_threads": torch.get_num_threads(),
        "scoring_p50_ms": _percentile(scoring_ms, 0.50),
        "scoring_p95_ms": _percentile(scoring_ms, 0.95),
        "ranking_p50_ms": _percentile(ranking_ms, 0.50),
        "ranking_p95_ms": _percentile(ranking_ms, 0.95),
        "end_to_end_p50_ms": _percentile(end_to_end_ms, 0.50),
        "end_to_end_p95_ms": _percentile(end_to_end_ms, 0.95),
        "end_to_end_mean_ms": statistics.fmean(end_to_end_ms),
    }
    print("\nCPU latency (after warmup, bounded sample)")
    print(f"  requests            : {latency['requests']}  (k={args.k})")
    print(f"  catalog size        : {latency['catalog_size']:,}")
    print(f"  OMP_NUM_THREADS     : {latency['omp_num_threads']} | torch threads {latency['torch_threads']}")
    print(f"  model scoring  p50  : {latency['scoring_p50_ms']:.2f} ms   p95 {latency['scoring_p95_ms']:.2f} ms")
    print(f"  ranking        p50  : {latency['ranking_p50_ms']:.2f} ms   p95 {latency['ranking_p95_ms']:.2f} ms")
    print(f"  end-to-end     p50  : {latency['end_to_end_p50_ms']:.2f} ms   p95 {latency['end_to_end_p95_ms']:.2f} ms")

    checks["startup loaded the model"] = engine.is_ready()
    checks["latency sample collected"] = latency["requests"] > 0

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values())
    print(f"\nSMOKE: {'PASS' if passed else 'FAIL'}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": engine.checkpoint_sha256,
                    "device": str(engine.device),
                    "num_items": engine.num_items,
                    "max_seq_len": engine.max_seq_len,
                    "parameter_count": parameter_count,
                    "startup_seconds": startup_seconds,
                    "example_history_length": len(example),
                    "example_recommendations": [r.as_dict() for r in result.recommendations],
                    "latency": latency,
                    "checks": checks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
