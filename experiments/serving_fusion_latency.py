#!/usr/bin/env python3
"""Serving V1 latency smoke: engine initialisation, one request, and the median of a few.

This is an engineering acceptance measurement, **not** a benchmark: it does not evaluate quality,
does not touch the frozen 20,000-user cohort and produces no metric.  It answers one question —
does the fixed multi-source serving path start and answer within a usable budget on this host?

Usage::

    python -m experiments.serving_fusion_latency --requests 5

Real artifacts are required (the accepted SASRec checkpoint, the Two-Tower checkpoint and its
categorical metadata, the canonical mappings and the normalized catalogue), so this is deliberately
not part of the offline test suite.  If an artifact is missing the run reports that and exits
non-zero rather than measuring something else.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _history_from_catalogue(engine: Any, size: int) -> list[str]:
    """A deterministic synthetic history drawn from the catalogue.

    A latency measurement needs *a* valid history, not a real user's: the work per request depends
    on the history length and the catalogue, not on which items were bought.  Identities are taken
    in sorted order so the run is reproducible.
    """
    identities = sorted(engine.catalogue.keys())[:size]
    if not identities:
        raise SystemExit("the catalogue index is empty; cannot build a history")
    return identities


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=5, help="timed requests (default 5)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--history", type=int, default=20, help="synthetic history length")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--mode",
        default="rrf3",
        choices=("rrf3", "sasrec"),
        help="which serving engine to measure; sasrec gives the single-source baseline",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the result here")
    args = parser.parse_args(argv)

    from recommendation.api.app import ServiceSettings, build_engine

    print(f"mode={args.mode} requests={args.requests} k={args.k} history={args.history}")
    started = time.perf_counter()
    engine, error = build_engine(ServiceSettings(serving_fusion=args.mode))
    init_seconds = time.perf_counter() - started
    if engine is None:
        print(f"FAILED to build the serving engine: {error}", file=sys.stderr)
        return 2
    print(f"engine initialisation: {init_seconds:.2f}s")
    metadata = getattr(engine, "model_metadata", None)
    if callable(metadata):
        print(f"engine metadata: {json.dumps(metadata(), sort_keys=True)}")

    history = _history_from_catalogue(engine, args.history)
    print(f"history ({len(history)} identities) starts with {history[0]!r}")

    for _ in range(max(0, args.warmup)):
        engine.recommend(history, k=args.k)

    latencies: list[float] = []
    last: Any = None
    for index in range(args.requests):
        began = time.perf_counter()
        last = engine.recommend(history, k=args.k)
        elapsed = time.perf_counter() - began
        latencies.append(elapsed)
        print(f"  request {index + 1}: {elapsed * 1000.0:.1f} ms, {last.returned_k} items")

    assert last is not None
    result = {
        "mode": args.mode,
        "engine_init_seconds": round(init_seconds, 3),
        "requests": args.requests,
        "k": args.k,
        "history_length": len(history),
        "returned_k": last.returned_k,
        "latency_ms": {
            "single": round(latencies[0] * 1000.0, 1),
            "median": round(statistics.median(latencies) * 1000.0, 1),
            "min": round(min(latencies) * 1000.0, 1),
            "max": round(max(latencies) * 1000.0, 1),
        },
        "engine_timings_ms": last.timings_ms,
        "score_kind": last.score_kind,
        "sources_in_first_item": (last.provenance or {}).get(
            last.recommendations[0].parent_asin, {}
        ).get("sources", []) if last.recommendations else [],
        "padded_returned": any(item.item_id == 0 for item in last.recommendations),
    }
    print()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"written to {args.json}")
    if result["padded_returned"]:
        print("FAILED: PAD was returned as a recommendation", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
