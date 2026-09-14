"""Recommendation Tool smoke + latency measurement (Milestone 7A).

Loads the accepted Milestone 5 checkpoint once, wraps it in the Agent-facing
Recommendation Tool, runs a bounded real-history example, and measures the Tool
wrapper overhead separately from the underlying engine inference.

This is integration and latency measurement only.  No recommendation-quality metric
is computed, no review text is printed, and no product metadata is invented.

Usage::

    .venv/bin/python -m experiments.recommendation_tool_smoke
    .venv/bin/python -m experiments.recommendation_tool_smoke --k 5 --calls 60
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

from recommendation.inference import InferenceConfig, SASRecInferenceEngine  # noqa: E402
from recommendation.tools import (  # noqa: E402
    RecommendationContext,
    RecommendationTool,
    RecommendationToolRequest,
)

CHECKPOINT = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
MANIFEST = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "run.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

ACCEPTED_SHA256 = "352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912"


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile over a list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_histories(limit: int) -> list[list[str]]:
    """Bounded real histories: ``train_history + validation_target`` (test target omitted)."""
    payload = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    histories: list[list[str]] = []
    for record in payload["sequences"][:limit]:
        asins = record["parent_asins"]
        if len(asins) >= 3:
            histories.append(list(asins[:-2]) + [asins[-2]])
    return histories


def main(argv: list[str] | None = None) -> int:
    """Run the smoke; returns 0 when every check passes."""
    parser = argparse.ArgumentParser(description="Recommendation Tool smoke (Milestone 7A)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--mappings", type=Path, default=MAPPINGS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--calls", type=int, default=60, help="bounded latency sample")
    parser.add_argument("--histories", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("=" * 78)
    print("Milestone 7A Recommendation Tool smoke (real accepted checkpoint)")
    print("Integration + latency only. NOT a recommendation-quality benchmark.")
    print("=" * 78)

    for path in (args.checkpoint, args.mappings):
        if not path.exists():
            print(f"missing artifact: {path}", file=sys.stderr)
            return 2

    # ---- one engine, loaded once ---------------------------------------- #
    started = time.perf_counter()
    engine = SASRecInferenceEngine(
        InferenceConfig(
            checkpoint_path=args.checkpoint,
            mappings_path=args.mappings,
            manifest_path=args.manifest if args.manifest.exists() else None,
            device=args.device,
            expected_checkpoint_sha256=(
                ACCEPTED_SHA256 if args.checkpoint == CHECKPOINT else None
            ),
        )
    )
    load_seconds = time.perf_counter() - started
    tool = RecommendationTool(engine)

    checks: dict[str, bool] = {}
    checks["checkpoint SHA-256 matches accepted digest"] = (
        args.checkpoint != CHECKPOINT or engine.checkpoint_sha256 == ACCEPTED_SHA256
    )
    checks["tool reuses the supplied engine"] = tool.engine is engine
    checks["model in eval mode"] = engine.model.training is False
    checks["tool metadata stable"] = tool.metadata()["name"] == "recommend_products"

    print(f"\ntool            : {tool.name} v{tool.version}")
    print(f"description     : {tool.description}")
    print(f"device          : {engine.device}")
    print(f"checkpoint sha  : {engine.checkpoint_sha256}")
    print(f"catalog size    : {engine.num_items:,}")
    print(f"engine load     : {load_seconds:.2f}s (once, outside the tool call)")

    # ---- representative real Tool call ---------------------------------- #
    histories = load_histories(args.histories)
    if not histories:
        print("no real histories available", file=sys.stderr)
        return 2

    history = histories[0]
    snapshot = list(history)
    context = RecommendationContext(user_history=tuple(history))
    result = tool.run(RecommendationToolRequest(k=args.k), context)

    checks["real history accepted"] = result.history_length == len(history)
    checks["source history unmodified"] = history == snapshot
    checks["structured result returned"] = result.requested_k == args.k
    checks["no PAD returned"] = all(r.item_id != 0 for r in result.recommendations)
    checks["no seen item returned"] = not (
        {r.parent_asin for r in result.recommendations} & set(history)
    )
    checks["ranks contiguous from 1"] = [r.rank for r in result.recommendations] == list(
        range(1, result.returned_k + 1)
    )
    checks["parent_asin/item_id consistent"] = all(
        r.parent_asin == engine.item_id_to_parent_asin(r.item_id) for r in result.recommendations
    )
    checks["scores finite"] = all(
        r.score == r.score and abs(r.score) != float("inf") for r in result.recommendations
    )

    print(f"\ntrusted history length = {result.history_length} "
          f"(effective {result.effective_history_length}, truncated={result.history_truncated})")
    print(f"requested k = {result.requested_k}   returned k = {result.returned_k}")
    print(f"eligible candidates    = {result.eligible_candidates:,}")
    print("\n  rank  parent_asin    item_id     score")
    for item in result.recommendations:
        print(f"  {item.rank:4d}  {item.parent_asin}  {item.item_id:7d}  {item.score:+.4f}")

    # ---- determinism ----------------------------------------------------- #
    repeat = tool.run(RecommendationToolRequest(k=args.k), context)
    checks["repeated Tool call deterministic"] = repeat.recommendations == result.recommendations

    # ---- latency: engine vs Tool ---------------------------------------- #
    for warmup_history in histories[: min(3, len(histories))]:
        tool.run(
            RecommendationToolRequest(k=args.k),
            RecommendationContext(user_history=tuple(warmup_history)),
        )

    engine_ms: list[float] = []
    tool_ms: list[float] = []
    for index in range(args.calls):
        sample = histories[index % len(histories)]
        sample_context = RecommendationContext(user_history=tuple(sample))

        started = time.perf_counter()
        engine.recommend(list(sample), k=args.k)
        engine_ms.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        tool.run(RecommendationToolRequest(k=args.k), sample_context)
        tool_ms.append((time.perf_counter() - started) * 1000.0)

    overhead_ms = [t - e for t, e in zip(tool_ms, engine_ms)]
    import os

    latency = {
        "calls": args.calls,
        "k": args.k,
        "catalog_size": engine.num_items,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_threads": torch.get_num_threads(),
        "engine_p50_ms": _percentile(engine_ms, 0.50),
        "engine_p95_ms": _percentile(engine_ms, 0.95),
        "tool_p50_ms": _percentile(tool_ms, 0.50),
        "tool_p95_ms": _percentile(tool_ms, 0.95),
        "overhead_p50_ms": _percentile(overhead_ms, 0.50),
        "overhead_p95_ms": _percentile(overhead_ms, 0.95),
        "overhead_mean_ms": statistics.fmean(overhead_ms),
    }
    print("\nCPU latency (after warmup, bounded sample)")
    print(f"  calls                     : {latency['calls']}  (k={args.k})")
    print(f"  OMP_NUM_THREADS / threads : {latency['omp_num_threads']} / {latency['torch_threads']}")
    print(f"  engine     p50 {latency['engine_p50_ms']:.2f} ms   p95 {latency['engine_p95_ms']:.2f} ms")
    print(f"  Tool       p50 {latency['tool_p50_ms']:.2f} ms   p95 {latency['tool_p95_ms']:.2f} ms")
    print(f"  wrapper    p50 {latency['overhead_p50_ms']:.3f} ms  p95 {latency['overhead_p95_ms']:.3f} ms")

    checks["latency sample collected"] = latency["calls"] > 0

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
                    "engine_load_seconds": load_seconds,
                    "tool": tool.metadata(),
                    "history_length": result.history_length,
                    "requested_k": result.requested_k,
                    "returned_k": result.returned_k,
                    "recommendations": [r.model_dump() for r in result.recommendations],
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
