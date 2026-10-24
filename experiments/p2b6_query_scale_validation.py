"""Phase 2B.6: bounded scale validation of the catalog query strategies.

What this is
------------
Phase 2B.5 measured three query arms on 200 cases and found no established gain:
C (balanced) beat B (current) by one case (+0.005 coverage, 95% CI [-0.010, 0.025]).
This driver repeats **only the three history-bearing arms** on a fresh, larger sample,
using the same code path, budgets and 2B.5 helpers:

    A  history only
    B  history + current catalog query   (frozen M4 ``deterministic_terms``)
    C  history + balanced catalog query  (2B.5 ``balanced_title_terms``)

The catalog-only arm D is deliberately **not** rerun: 2B.5's diagnosis showed its hits are
either already in the history head or far outside the top-5, so it cannot change this
decision.  Re-running it would double the cost for no decision value.

Sample
------
1,000 valid cases starting at artifact index **6710**, i.e. immediately after 2B.5's
6500-6709 window, so this is a fresh sample and not a superset.  Users in the frozen
20,000-user cohort (M3/M4/Phase 5/B1) are excluded by ``user_int_id``.  P3 (first 1,000) and
the R-LR 5,000-group window (valid #1001-6000) are outside by position; R-LR's per-case
split is not stored, so that exclusion is positional only (recorded in config).

Stop rules (fixed **before** the run, copied from the 2B.5 report)
-----------------------------------------------------------------
R1  n >= 1,000 and the paired C-B coverage difference has a 95% bootstrap CI lower bound > 0
R2  at least two catalog-unique targets (in the catalog head, absent from the history head)
    reach the fused top-5 in arm B or C
R3  the catalog target hit rate reaches the pre-registered doubling of the 2B.5 baseline:
    B baseline 3/200 = 1.50% -> threshold 3.00%; C baseline 4/200 = 2.00% -> 4.00%

No rule is satisfied -> the balanced query is not promoted to production, the query-strategy
line ends, and the result is reported as "incremental gain not established".

    .venv/bin/python -m experiments.p2b6_query_scale_validation --cases 1000 \
        --skip 6710 --out runs/p2b6_query_scale_1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config

from experiments.p2b5_query_coverage import (
    ARMS as P25_ARMS,
    DEFAULT_OUT as P25_OUT,
    MAPPINGS,
    SEQUENCES,
    BALANCED_QUERY_SOURCE,
    CURRENT_QUERY_SOURCE,
    HISTORY_SEMANTICS,
    REQUESTED_K,
    SOURCE_DEPTH,
    _aggregate,
    _frozen_cohort_users,
    _git_head,
    _hit_attribution,
    _manifest,
    _paired,
    _sha256,
    attach_queries,
    build_sample,
    load_cases,
)
from experiments.p2b_agent_source_limit_ablation import (
    SourceProbe,
    instrument_catalog_search,
    instrument_history_source,
    paired_ci,
)

REPO = project_config.REPO_ROOT
DEFAULT_OUT = REPO / "runs" / "p2b6_query_scale_1000"
#: Fresh cases, immediately after 2B.5's 6500-6709 window.
DEFAULT_SKIP = 6710
DEFAULT_CASES = 1000
BOOTSTRAP_DRAWS = 5000
SEED = 20261024

#: Only the three history-bearing arms; order must match the 2B.5 names.
ARMS: tuple[dict[str, Any], ...] = tuple(
    arm for arm in P25_ARMS if arm["name"] in {
        "A_history_only",
        "B_history_current_catalog",
        "C_history_balanced_catalog",
    }
)

#: 2B.5 baselines, read from its recorded attribution (n=200).  Fixed here, before the run.
BASELINE_CASES = 200
BASELINE_CATALOG_HITS = {
    "B_history_current_catalog": 3,   # both 1 + catalog_only 2
    "C_history_balanced_catalog": 4,  # both 1 + catalog_only 3
}
STOP_RULES = {
    "R1": {
        "definition": "paired C-B pool-coverage difference, 95% bootstrap CI lower bound > 0",
        "min_cases": 1000,
    },
    "R2": {
        "definition": ">=2 catalog-unique targets reach the fused top-5 (arm B or C)",
        "threshold": 2,
    },
    "R3": {
        "definition": "catalog target hit rate >= 2x the 2B.5 baseline",
        "baseline_cases": BASELINE_CASES,
        "baseline_hits": dict(BASELINE_CATALOG_HITS),
        "threshold_rate": {
            name: round(2 * hits / BASELINE_CASES, 4)
            for name, hits in BASELINE_CATALOG_HITS.items()
        },
    },
}


def _by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        grouped.setdefault(str(row["arm"]), {})[int(row["case_index"])] = row
    return grouped


def _coverage(row: Mapping[str, Any]) -> float:
    return 1.0 if row["target_in_pool"] else 0.0


def _pair_between(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    grouped = _by_arm(rows)
    shared = sorted(set(grouped.get(left, {})) & set(grouped.get(right, {})))
    diffs = [_coverage(grouped[left][i]) - _coverage(grouped[right][i]) for i in shared]
    return {
        "delta": round(sum(diffs) / len(diffs), 4) if diffs else 0.0,
        "ci95": paired_ci(diffs, draws=BOOTSTRAP_DRAWS),
        "n": len(diffs),
    }


def _catalog_hit_rate(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    grouped = _by_arm(rows).get(arm, {})
    hits = sum(
        1
        for row in grouped.values()
        if row["sources"]["catalog_search"].get("rank_of_target") is not None
    )
    return {"hits": hits, "cases": len(grouped), "rate": round(hits / len(grouped), 4) if grouped else None}


def _catalog_unique_top5(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Catalog-unique target hits (catalog head yes, history head no) that fused into top-5."""
    found: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "ok" or row["arm"] == "A_history_only":
            continue
        history_rank = row["sources"]["history"].get("rank_of_target")
        catalog_rank = row["sources"]["catalog_search"].get("rank_of_target")
        final_rank = row["target_final_rank"]
        if catalog_rank is not None and history_rank is None and final_rank is not None:
            if int(final_rank) <= REQUESTED_K:
                found.append(
                    {
                        "arm": row["arm"],
                        "case_index": row["case_index"],
                        "catalog_rank": catalog_rank,
                        "final_rank": final_rank,
                    }
                )
    return found


def _evaluate_stop_rules(rows: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    c_vs_b = _pair_between(rows, "C_history_balanced_catalog", "B_history_current_catalog")
    unique_top5 = _catalog_unique_top5(rows)
    rates = {
        arm: _catalog_hit_rate(rows, arm)
        for arm in ("B_history_current_catalog", "C_history_balanced_catalog")
    }
    r1 = bool(n >= STOP_RULES["R1"]["min_cases"] and c_vs_b["ci95"][0] > 0)
    r2 = len(unique_top5) >= STOP_RULES["R2"]["threshold"]
    r3_checks = {
        arm: {
            "rate": rates[arm]["rate"],
            "threshold": STOP_RULES["R3"]["threshold_rate"][arm],
            "satisfied": rates[arm]["rate"] is not None
            and rates[arm]["rate"] >= STOP_RULES["R3"]["threshold_rate"][arm],
        }
        for arm in rates
    }
    r3 = any(check["satisfied"] for check in r3_checks.values())
    return {
        "R1": {"satisfied": r1, "c_vs_b": c_vs_b, "n": n,
               "min_cases": STOP_RULES["R1"]["min_cases"]},
        "R2": {"satisfied": r2, "catalog_unique_top5": unique_top5,
               "threshold": STOP_RULES["R2"]["threshold"]},
        "R3": {"satisfied": r3, "checks": r3_checks,
               "threshold_rate": STOP_RULES["R3"]["threshold_rate"]},
        "any_satisfied": bool(r1 or r2 or r3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")

    from recommendation.api.app import ServiceSettings
    from recommendation.demo import build_demo_runtime
    from recommendation.demo.agent_schemas import AgentRecommendRequest
    from recommendation.demo.source_limit import ENV_SOURCE_LIMIT

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    user_map = mappings.get("user2id") or {}
    excluded_users = _frozen_cohort_users()

    # Load a small buffer: a case whose queries are empty is dropped rather than weakening
    # the sample, and the buffer keeps the usable count at the requested size.
    cases, selection = load_cases(
        args.cases + 25, args.skip, excluded_users=excluded_users, user_map=user_map
    )
    if not cases:
        raise SystemExit("no usable cases")

    started = time.time()
    runtime = build_demo_runtime(settings=ServiceSettings.from_env())
    load_seconds = time.time() - started
    service = runtime.agent_service()

    records = runtime.metadata.records

    def title_of(parent_asin: str) -> str | None:
        record = records.get(parent_asin)
        return None if record is None else getattr(record, "title", None)

    attach_queries(cases, title_of)
    usable = [case for case in cases if case["usable"]][: args.cases]
    if not usable:
        raise SystemExit("no case has both proxy queries")

    engine = getattr(runtime, "engine", None)
    engine_config = getattr(engine, "config", None)
    config = {
        "task": "next_item_prediction",
        "history_semantics": HISTORY_SEMANTICS,
        "arms": [arm["name"] for arm in ARMS],
        "source_depth": SOURCE_DEPTH,
        "requested_k": REQUESTED_K,
        "cases_requested": args.cases,
        "cases_usable": len(usable),
        "skip": args.skip,
        "selection": {**selection, "usable": len(usable),
                      "unusable_queries": len(cases) - sum(1 for c in cases if c["usable"])},
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "stop_rules": STOP_RULES,
        "baseline_artifact": str(P25_OUT.relative_to(REPO)),
        "query_rules": {
            "current": {"source": CURRENT_QUERY_SOURCE,
                        "implementation": "experiments.m4_query_rules.deterministic_terms"},
            "balanced": {"source": BALANCED_QUERY_SOURCE,
                         "implementation": "experiments.p2b5_query_coverage.balanced_title_terms"},
        },
        "sample_isolation": {
            "frozen_20k_cohort_excluded_by_user_int_id": True,
            "frozen_20k_cohort_users": len(excluded_users),
            "p3_prefix_avoided": args.skip >= 1000,
            "r_lr_window": "valid cases #1001-6000; sample starts at artifact index "
                           f"{args.skip} (outside by position)",
            "r_lr_split_membership_verifiable": False,
            "disjoint_from_2b5_200": args.skip >= 6710,
        },
        "source_limit_env": ENV_SOURCE_LIMIT,
        "source_limit_value": SOURCE_DEPTH,
        "load_seconds": round(load_seconds, 3),
        "history_source": {
            "pre_fused": True,
            "engine_score_kind": getattr(engine, "score_kind", None),
            "profile": getattr(engine_config, "profile", None),
            "inner_sources": list(getattr(engine_config, "sources", ()) or ()),
            "inner_head_depth": getattr(engine_config, "head_depth", None),
            "inner_rrf_k": getattr(engine_config, "rrf_k", None),
        },
        "catalog_search_source": {
            "pre_fused": False,
            "score_kind": "bm25",
            "catalogue_size": int(getattr(service._catalog_search, "catalogue_size", 0) or 0),
        },
        "sequence_artifact": str(SEQUENCES.relative_to(REPO)),
        "python": sys.version.split()[0],
        "git_head": _git_head(),
        "driver_path": str(Path(__file__).resolve().relative_to(REPO)),
        "driver_sha256": _sha256(Path(__file__)),
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cases.json").write_text(json.dumps(usable, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Warm-up before the probes are attached; it never enters a sample.
    service.run(
        AgentRecommendRequest(
            message=usable[0]["current_query"],
            history=usable[0]["history"],
            k=REQUESTED_K,
            sources=("history", "catalog_search"),
        )
    )

    probe = SourceProbe()
    instrument_history_source(service._tool, probe)
    instrument_catalog_search(service._catalog_search, probe)

    records_out: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    previous_limit = os.environ.get(ENV_SOURCE_LIMIT)
    try:
        os.environ[ENV_SOURCE_LIMIT] = str(SOURCE_DEPTH)
        for arm in ARMS:
            arm_rows: list[dict[str, Any]] = []
            for case in usable:
                query = (
                    case["current_query"]
                    if arm["query_mode"] == "current"
                    else case["balanced_query"]
                )
                payload = AgentRecommendRequest(
                    message=query,
                    history=case["history"],
                    k=REQUESTED_K,
                    sources=tuple(arm["plan"]),
                )
                probe.reset()
                began = time.perf_counter()
                try:
                    response = service.run(payload)
                    elapsed_ms = (time.perf_counter() - began) * 1000.0
                    row = build_sample(
                        arm=arm,
                        case=case,
                        query=query,
                        response=response,
                        probe=probe,
                        elapsed_ms=elapsed_ms,
                    )
                except Exception as exc:  # noqa: BLE001 - a failed case is recorded, never dropped
                    elapsed_ms = (time.perf_counter() - began) * 1000.0
                    row = {
                        "arm": arm["name"],
                        "query_mode": arm["query_mode"],
                        "case_index": case["case_index"],
                        "user_id": case["user_id"],
                        "user_int_id": case["user_int_id"],
                        "target": case["target"],
                        "message": query,
                        "query_terms": list(case.get(f"{arm['query_mode']}_terms") or ()),
                        "history_semantics": HISTORY_SEMANTICS,
                        "history_length": int(case["history_length"]),
                        "status": f"error:{type(exc).__name__}",
                        "returned": 0,
                        "ranked": [],
                        "pool_size": 0,
                        "target_in_pool": False,
                        "target_pool_rank": None,
                        "target_final_rank": None,
                        "hr@5": 0.0,
                        "ndcg@5": 0.0,
                        "ms": round(elapsed_ms, 2),
                        "error": str(exc),
                        "sources": {
                            source: [call for call in probe.of(source)]
                            for source in ("history", "catalog_search")
                        },
                    }
                arm_rows.append(row)
                records_out.append(row)
            aggregates[arm["name"]] = _aggregate(arm_rows)
            print(json.dumps({"arm": arm["name"], "summary": aggregates[arm["name"]]}),
                  flush=True)
    finally:
        if previous_limit is None:
            os.environ.pop(ENV_SOURCE_LIMIT, None)
        else:
            os.environ[ENV_SOURCE_LIMIT] = previous_limit

    summary = {
        "task": "next_item_prediction",
        "history_semantics": HISTORY_SEMANTICS,
        "source_depth": SOURCE_DEPTH,
        "requested_k": REQUESTED_K,
        "cases_usable": len(usable),
        "arms": aggregates,
        "paired_vs_A_history_only": _paired(records_out, ARMS),
        "target_source_attribution": _hit_attribution(records_out, ARMS),
        "stop_rules": _evaluate_stop_rules(records_out, len(usable)),
    }

    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records_out), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "SHA256SUMS.json").write_text(
        json.dumps(_manifest(args.out), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
