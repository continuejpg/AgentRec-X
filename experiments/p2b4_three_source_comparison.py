"""Phase 2B.4: three-source comparison on the frozen next-item task.

Why this driver exists
----------------------
Phase 2B.3 measured the Agent's candidate sources with ``history = asins[:-2]``.  That is
the **train** history of ``agentrecx.eval_protocol.v1``: it drops the validation interaction,
which happened strictly before the test target and is legitimately visible at prediction
time.  Every frozen evaluator in this repository (P3, Phase 5, B1, M4) uses
``case.test_history = train_history + (validation_target,) = asins[:-1]``.  Comparing a
recall number measured on ``asins[:-2]`` with any frozen number would therefore compare two
different tasks.

This driver measures the three plans the 2B.4 brief asks for, on the **frozen task**:

    A  history only          (Serving V1: three inner sources fused by RRF, pre_fused)
    B  catalog_search only   (BM25 over the full catalogue)
    C  both, fused           (the Agent's multi-source plane)

Same 40 cases, same ``requested_k = 5``, same per-source depth of 50, paired by case.

The catalog query is a **proxy**: the frozen M4 rule ``deterministic_terms(h)`` (three
newest history items, at most eight terms, terms shorter than three characters dropped).
It is a pure function of the trusted history, not a user-authored shopping query, and every
sample is labelled ``query_source = "frozen-m4-deterministic-terms"``.  Nothing here is
evidence about natural-language product search.

    .venv/bin/python -m experiments.p2b4_three_source_comparison --cases 40 \
        --out runs/p2b4_three_source_40

Outputs (never written into an existing run directory):
    config.json       frozen configuration + artifact identity
    cases.json        the 40 cases, their test history and their proxy query
    samples.jsonl     one record per (plan, case)
    summary.json      per-plan aggregates, paired deltas and source-uniqueness counts
    SHA256SUMS.json   SHA-256 of every artifact above
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config

from experiments.m4_query_rules import deterministic_terms
from experiments.p2b_agent_source_limit_ablation import (
    SourceProbe,
    hr_at_k,
    instrument_catalog_search,
    instrument_history_source,
    ndcg_at_k,
    paired_ci,
)

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
DEFAULT_OUT = REPO / "runs" / "p2b4_three_source_40"

#: The three plans the brief asks for.  Order matters only for presentation.
PLANS: tuple[tuple[str, ...], ...] = (
    ("history",),
    ("catalog_search",),
    ("history", "catalog_search"),
)
#: Fixed per-source depth (the brief's "先固定候选深度 50").
SOURCE_DEPTH = 50
REQUESTED_K = 5
DEFAULT_SKIP = 2000
DEFAULT_CASES = 40
BOOTSTRAP_DRAWS = 2000
SEED = 20261024

#: Labelling.  The history is the frozen test history; the query is a frozen proxy rule.
HISTORY_SEMANTICS = "case.test_history (train_history + validation_target)"
QUERY_SOURCE = "frozen-m4-deterministic-terms"
LEGACY_MESSAGE = "recommend something for me"


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def load_cases(limit: int, skip: int) -> list[dict[str, Any]]:
    """The frozen test split: history = items[:-1], target = items[-1].

    The validation target is part of the history because it happened strictly before the
    test target; :class:`recommendation.evaluation.split.EvaluationCase` defines it the
    same way and the frozen evaluator masks with it.
    """
    document = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for index, sequence in enumerate(document["sequences"]):
        if index < skip:
            continue
        items = sequence["item_ids"]
        asins = sequence["parent_asins"]
        if len(items) < 4:
            continue
        cases.append(
            {
                "case_index": index,
                "user_id": sequence["user_id"],
                # test_history = train_history + (validation_target,)
                "history": asins[:-1],
                "validation_target": asins[-2],
                "target": asins[-1],
                "history_length": len(asins) - 1,
            }
        )
        if len(cases) >= limit:
            break
    return cases


def attach_proxy_queries(
    cases: list[dict[str, Any]],
    title_of: Any,
) -> list[dict[str, Any]]:
    """Attach the frozen M4 query rule's terms to every case, in place.

    ``deterministic_terms`` is the preregistered M4 construction (three newest history items,
    at most eight terms, tokens shorter than three characters dropped).  Reusing it keeps the
    query a frozen, testable function instead of a new recipe invented here.
    """
    for case in cases:
        terms = deterministic_terms(list(case["history"]), describe=title_of)
        case["query_terms"] = list(terms)
        case["query"] = " ".join(terms) if terms else None
        case["query_source"] = QUERY_SOURCE
        case["usable"] = bool(terms)
        case["excluded_reason"] = None if terms else "empty_query_terms"
    return cases


# --------------------------------------------------------------------------- #
# one sample
# --------------------------------------------------------------------------- #


def _first(calls: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return calls[0] if calls else None


def _rank_of(identities: Sequence[str], target: str) -> int | None:
    try:
        return list(identities).index(target) + 1
    except ValueError:
        return None


def _status(view: Any) -> str:
    value = getattr(view, "verification_status", None)
    return str(getattr(value, "value", value))


def build_sample(
    *,
    plan: Sequence[str],
    case: Mapping[str, Any],
    query: str,
    response: Any,
    probe: SourceProbe,
    elapsed_ms: float,
) -> dict[str, Any]:
    """One (plan, case) record: candidate counts, target ranks, metrics, latency."""
    target = str(case["target"])
    history_call = _first(probe.of("history"))
    catalog_call = _first(probe.of("catalog_search"))
    history_ids = list((history_call or {}).get("ranked") or ())
    catalog_ids = list((catalog_call or {}).get("ranked") or ())

    pool_order = [
        str(view.parent_asin)
        for view in response.grounded.candidates
        if _status(view) == "grounded"
    ]
    final_order = [str(item.parent_asin) for item in response.recommendations]
    pool_rank = _rank_of(pool_order, target)
    final_rank = _rank_of(final_order, target)

    def source_record(call: Mapping[str, Any] | None, *, effective_cap: int) -> dict[str, Any]:
        if call is None:
            return {"called": False}
        returned = int(call.get("returned") or 0)
        internal = call.get("internal") or {}
        return {
            "called": True,
            "requested_limit": SOURCE_DEPTH,
            "effective_limit": int(call.get("transport_limit") or 0),
            "returned": returned,
            "rank_of_target": _rank_of(list(call.get("ranked") or ()), target),
            "ms": call.get("ms"),
            "failed": call.get("failed"),
            "pre_fused": bool(call.get("pre_fused")),
            "score_kind": call.get("score_kind"),
            "query": call.get("query"),
            "short_of_limit": returned < effective_cap,
            # Serving V1's own view of the target, when its head contains it.
            "target_internal_provenance": (internal.get("provenance") or {}).get(target)
            if internal else None,
        }

    return {
        "plan": "+".join(plan),
        "case_index": case["case_index"],
        "user_id": case["user_id"],
        "target": target,
        "message": query,
        "query_source": case.get("query_source"),
        "query_terms": list(case.get("query_terms") or ()),
        "history_semantics": HISTORY_SEMANTICS,
        "history_length": int(case["history_length"]),
        "status": "ok",
        "returned": len(final_order),
        "ranked": final_order,
        "pool_size": len(pool_order),
        "target_in_pool": pool_rank is not None,
        "target_pool_rank": pool_rank,
        "target_final_rank": final_rank,
        "sources": {
            "history": source_record(history_call, effective_cap=SOURCE_DEPTH),
            "catalog_search": source_record(catalog_call, effective_cap=SOURCE_DEPTH),
        },
        # Unique candidate identities each source contributed to the fused pool.
        "unique_candidates": {
            "catalog_only": len(set(catalog_ids) - set(history_ids)),
            "history_only": len(set(history_ids) - set(catalog_ids)),
            "shared": len(set(history_ids) & set(catalog_ids)),
        },
        "hr@5": hr_at_k(final_order, target, REQUESTED_K),
        "ndcg@5": ndcg_at_k(final_order, target, REQUESTED_K),
        "ms": round(elapsed_ms, 2),
    }


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(ordered: Sequence[float], fraction: float) -> float | None:
    if not ordered:
        return None
    return round(ordered[int(fraction * (len(ordered) - 1))], 1)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    latencies = sorted(float(row["ms"]) for row in ok)
    pooled = [row for row in ok if row["target_in_pool"]]
    return {
        "cases": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "hr@5": _mean([float(row["hr@5"]) for row in ok]),
        "ndcg@5": _mean([float(row["ndcg@5"]) for row in ok]),
        "pool_coverage": _mean([1.0 if row["target_in_pool"] else 0.0 for row in ok]),
        "mean_pool": _mean([float(row["pool_size"]) for row in ok]),
        "mean_history_returned": _mean(
            [float(row["sources"]["history"].get("returned", 0)) for row in ok]
        ),
        "mean_catalog_returned": _mean(
            [float(row["sources"]["catalog_search"].get("returned", 0)) for row in ok]
        ),
        "pooled_cases": len(pooled),
        "pool_rank_when_pooled": sorted(
            int(row["target_pool_rank"]) for row in pooled
            if row["target_pool_rank"] is not None
        ),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "mean_history_ms": _mean(
            [float(row["sources"]["history"].get("ms") or 0.0) for row in ok]
        ),
        "mean_catalog_ms": _mean(
            [float(row["sources"]["catalog_search"].get("ms") or 0.0) for row in ok]
        ),
    }


def _paired(
    rows: Sequence[Mapping[str, Any]], plans: Sequence[Sequence[str]]
) -> dict[str, dict[str, Any]]:
    by_plan: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_plan.setdefault(str(row["plan"]), {})[int(row["case_index"])] = row
    baseline_name = "+".join(plans[0])
    baseline = by_plan.get(baseline_name, {})
    extractors = {
        "hr@5": lambda row: float(row["hr@5"]),
        "ndcg@5": lambda row: float(row["ndcg@5"]),
        "pool_coverage": lambda row: 1.0 if row["target_in_pool"] else 0.0,
    }
    paired: dict[str, dict[str, Any]] = {}
    for plan in plans[1:]:
        name = "+".join(plan)
        shared = sorted(set(baseline) & set(by_plan.get(name, {})))
        for metric, extract in extractors.items():
            diffs = [extract(by_plan[name][index]) - extract(baseline[index]) for index in shared]
            paired[f"{name}_minus_{baseline_name}_{metric}"] = {
                "delta": round(sum(diffs) / len(diffs), 4) if diffs else 0.0,
                "ci95": paired_ci(diffs),
                "n": len(diffs),
            }
    return paired


def _uniqueness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which source actually returned the target, from the fused arm's per-source capture.

    Uses the fused plan because it is the only record where both sources ran on the same
    case.  A target rank here is the source's own rank at the fixed depth.
    """
    fused = [row for row in rows if row["plan"] == "history+catalog_search" and row["status"] == "ok"]
    counts = {"both": 0, "history_only": 0, "catalog_only": 0, "neither": 0}
    details = {"history_only": [], "catalog_only": []}
    unique_candidates = {"catalog_only": [], "history_only": [], "shared": []}
    for row in fused:
        history_rank = row["sources"]["history"].get("rank_of_target")
        catalog_rank = row["sources"]["catalog_search"].get("rank_of_target")
        if history_rank is not None and catalog_rank is not None:
            counts["both"] += 1
        elif history_rank is not None:
            counts["history_only"] += 1
            details["history_only"].append(
                {"case_index": row["case_index"], "history_rank": history_rank}
            )
        elif catalog_rank is not None:
            counts["catalog_only"] += 1
            details["catalog_only"].append(
                {"case_index": row["case_index"], "catalog_rank": catalog_rank}
            )
        else:
            counts["neither"] += 1
        for key in unique_candidates:
            unique_candidates[key].append(int(row["unique_candidates"][key]))
    return {
        "cases": len(fused),
        "target_hit_by": counts,
        "examples": details,
        "mean_unique_candidates": {
            key: _mean([float(value) for value in values])
            for key, values in unique_candidates.items()
        },
    }


# --------------------------------------------------------------------------- #
# manifest / identity
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(out: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name in ("config.json", "cases.json", "samples.jsonl", "summary.json"):
        payload = (out / name).read_bytes()
        entries[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return entries


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - a missing git must not fail the experiment
        return None
    return completed.stdout.strip() or None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


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

    cases = load_cases(args.cases, args.skip)
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

    attach_proxy_queries(cases, title_of)
    run_cases = [case for case in cases if case["usable"]]
    if not run_cases:
        raise SystemExit("no case has a usable proxy query")

    engine = getattr(runtime, "engine", None)
    engine_config = getattr(engine, "config", None)
    config = {
        "task": "next_item_prediction",
        "brief_branch": 2,
        "plans": ["+".join(plan) for plan in PLANS],
        "source_depth": SOURCE_DEPTH,
        "requested_k": REQUESTED_K,
        "cases_requested": len(cases),
        "cases_usable": len(run_cases),
        "excluded_cases": [c["case_index"] for c in cases if not c["usable"]],
        "skip": args.skip,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "history_semantics": HISTORY_SEMANTICS,
        "query_source": QUERY_SOURCE,
        "query_rule": {
            "implementation": "experiments.m4_query_rules.deterministic_terms",
            "newest_items_scanned": 3,
            "max_terms": 8,
            "min_term_length": 3,
            "depends_on": "trusted test history only",
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
        "sequence_sha256": _sha256(SEQUENCES),
        "python": sys.version.split()[0],
        "git_head": _git_head(),
        "driver_path": str(Path(__file__).resolve().relative_to(REPO)),
        "driver_sha256": _sha256(Path(__file__)),
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Warm-up before the probes are attached; it never enters a sample.
    service.run(
        AgentRecommendRequest(
            message=run_cases[0]["query"],
            history=run_cases[0]["history"],
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
        for plan in PLANS:
            plan_rows: list[dict[str, Any]] = []
            for case in run_cases:
                payload = AgentRecommendRequest(
                    message=case["query"],
                    history=case["history"],
                    k=REQUESTED_K,
                    sources=plan,
                )
                probe.reset()
                began = time.perf_counter()
                try:
                    response = service.run(payload)
                    elapsed_ms = (time.perf_counter() - began) * 1000.0
                    row = build_sample(
                        plan=plan,
                        case=case,
                        query=case["query"],
                        response=response,
                        probe=probe,
                        elapsed_ms=elapsed_ms,
                    )
                except Exception as exc:  # noqa: BLE001 - a failed case is recorded, never dropped
                    elapsed_ms = (time.perf_counter() - began) * 1000.0
                    row = {
                        "plan": "+".join(plan),
                        "case_index": case["case_index"],
                        "user_id": case["user_id"],
                        "target": case["target"],
                        "message": case["query"],
                        "query_source": case.get("query_source"),
                        "query_terms": list(case.get("query_terms") or ()),
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
                plan_rows.append(row)
                records_out.append(row)
            name = "+".join(plan)
            aggregates[name] = _aggregate(plan_rows)
            print(json.dumps({"plan": name, "summary": aggregates[name]}), flush=True)
    finally:
        if previous_limit is None:
            os.environ.pop(ENV_SOURCE_LIMIT, None)
        else:
            os.environ[ENV_SOURCE_LIMIT] = previous_limit

    summary = {
        "task": "next_item_prediction",
        "history_semantics": HISTORY_SEMANTICS,
        "query_source": QUERY_SOURCE,
        "source_depth": SOURCE_DEPTH,
        "requested_k": REQUESTED_K,
        "cases_usable": len(run_cases),
        "arms": aggregates,
        "paired_vs_history_only": _paired(records_out, PLANS),
        "target_source_uniqueness": _uniqueness(records_out),
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
