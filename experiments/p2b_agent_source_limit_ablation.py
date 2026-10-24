"""Phase 2B exploratory ablation: agent source limit vs the final k.

Four arms share one request set, one plan, one catalogue, one engine and one final ``k``;
only ``AGENTRECX_AGENT_SOURCE_LIMIT`` changes.  The path executed is the real Agent
control plane (``DemoAgentService.run`` -> ``LoopController`` -> ``CandidatePlane``), not
``/v1/recommend``: the point of the experiment is the candidate pool the Agent fuses.

    .venv/bin/python -m experiments.p2b_agent_source_limit_ablation --cases 30 --out runs/p2b_agent_source_limit

Outputs (never written into an existing run directory):
    config.json      the frozen configuration of this execution
    cases.json       the request set (user id, history length, case index)
    samples.jsonl    one record per (arm, case)
    summary.json     per-arm aggregates + paired deltas with bootstrap CIs
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from recommendation import config as project_config

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
DEFAULT_OUT = REPO / "runs" / "p2b_agent_source_limit"

#: The four arms.  ``requested_k`` is identical by construction.
ARMS: tuple[int, ...] = (5, 20, 50, 100)
REQUESTED_K = 5
#: Cases are taken well after the 1,000-case prefix P3 used, so the two sets are disjoint.
DEFAULT_SKIP = 2000
BOOTSTRAP_DRAWS = 2000
SEED = 20261024


def hr_at_k(ranked: Sequence[str], target: str, k: int) -> float:
    """1.0 when the single positive appears in the first ``k``, else 0.0."""
    return 1.0 if target in list(ranked)[:k] else 0.0


def ndcg_at_k(ranked: Sequence[str], target: str, k: int) -> float:
    """Single-positive NDCG@k: 1/log2(rank+1) at the hit position, else 0."""
    for position, identity in enumerate(list(ranked)[:k], start=1):
        if identity == target:
            return 1.0 / math.log2(position + 1)
    return 0.0


def paired_ci(diffs: Sequence[float], *, draws: int = BOOTSTRAP_DRAWS) -> list[float]:
    """95% percentile bootstrap CI of the mean paired difference."""
    if not diffs:
        return [0.0, 0.0]
    rng = random.Random(SEED)
    n = len(diffs)
    means = []
    for _ in range(draws):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (draws - 1))]
    hi = means[int(0.975 * (draws - 1))]
    return [round(lo, 6), round(hi, 6)]


def load_cases(limit: int, skip: int) -> list[dict[str, Any]]:
    document = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    cases = []
    for index, sequence in enumerate(document["sequences"]):
        if index < skip:
            continue
        items = sequence["item_ids"]
        asins = sequence["parent_asins"]
        if len(items) < 4:
            continue
        # temporal leave-two-out: history = items[:-2], test target = the last one.
        cases.append(
            {
                "case_index": index,
                "user_id": sequence["user_id"],
                "history": asins[:-2],
                "target": asins[-1],
            }
        )
        if len(cases) >= limit:
            break
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")

    from recommendation.api.app import ServiceSettings
    from recommendation.demo import build_demo_runtime
    from recommendation.demo.agent_schemas import AgentRecommendRequest

    cases = load_cases(args.cases, args.skip)
    if not cases:
        raise SystemExit("no usable cases")

    started = time.time()
    runtime = build_demo_runtime(settings=ServiceSettings.from_env())
    service = runtime.agent_service()
    load_seconds = time.time() - started

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "arms": list(ARMS),
                "requested_k": REQUESTED_K,
                "sources": ["history", "catalog_search"],
                "cases": len(cases),
                "skip": args.skip,
                "seed": SEED,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "sequence_artifact": str(SEQUENCES.relative_to(REPO)),
                "load_seconds": round(load_seconds, 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import os

    from recommendation.demo.source_limit import ENV_SOURCE_LIMIT

    records: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    # A warm request keeps first-call allocation out of the measured latency.
    service.run(AgentRecommendRequest(message="warm up", history=cases[0]["history"][-5:],
                                      k=REQUESTED_K, sources=("history", "catalog_search")))

    for limit in ARMS:
        os.environ[ENV_SOURCE_LIMIT] = str(limit)
        arms_rows = []
        for case in cases:
            payload = AgentRecommendRequest(
                message="recommend something for me",
                history=case["history"],
                k=REQUESTED_K,
                sources=("history", "catalog_search"),
            )
            began = time.perf_counter()
            try:
                response = service.run(payload)
                elapsed = (time.perf_counter() - began) * 1000.0
                identities = [item.parent_asin for item in response.recommendations]
                grounded = getattr(response, "grounded", None)
                pool = [view.parent_asin for view in getattr(grounded, "candidates", ()) or ()]
                row = {
                    "arm": limit,
                    "case_index": case["case_index"],
                    "target": case["target"],
                    "returned": len(identities),
                    "ranked": identities,
                    "pool_size": len({*pool}),
                    "target_in_pool": case["target"] in set(pool),
                    "hr@5": hr_at_k(identities, case["target"], REQUESTED_K),
                    "ndcg@5": ndcg_at_k(identities, case["target"], REQUESTED_K),
                    "ms": round(elapsed, 2),
                    "status": "ok",
                    "sources": dict(response.observation.source_calls),
                }
            except Exception as exc:  # noqa: BLE001 - a failed case is recorded, never dropped
                elapsed = (time.perf_counter() - began) * 1000.0
                row = {"arm": limit, "case_index": case["case_index"], "status": f"error:{type(exc).__name__}",
                       "ms": round(elapsed, 2), "hr@5": 0.0, "ndcg@5": 0.0}
            arms_rows.append(row)
            records.append(row)
        ok = [row for row in arms_rows if row["status"] == "ok"]
        latencies = sorted(row["ms"] for row in ok)
        aggregates[str(limit)] = {
            "cases": len(arms_rows),
            "ok": len(ok),
            "hr@5": round(sum(row["hr@5"] for row in ok) / max(1, len(ok)), 4),
            "ndcg@5": round(sum(row["ndcg@5"] for row in ok) / max(1, len(ok)), 4),
            "pool_coverage": round(sum(row.get("target_in_pool", False) for row in ok) / max(1, len(ok)), 4),
            "mean_pool": round(sum(row.get("pool_size", 0) for row in ok) / max(1, len(ok)), 2),
            "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
            "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
        }

    paired = {}
    baseline = {row["case_index"]: row for row in records if row["arm"] == ARMS[0]}
    for limit in ARMS[1:]:
        arm = {row["case_index"]: row for row in records if row["arm"] == limit}
        shared = sorted(set(baseline) & set(arm))
        for metric in ("hr@5", "ndcg@5"):
            diffs = [arm[i][metric] - baseline[i][metric] for i in shared]
            paired[f"{limit}_minus_{ARMS[0]}_{metric}"] = {
                "delta": round(sum(diffs) / max(1, len(diffs)), 4),
                "ci95": paired_ci(diffs),
                "n": len(diffs),
            }

    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps({"arms": aggregates, "paired_vs_baseline": paired}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"arms": aggregates, "paired": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
