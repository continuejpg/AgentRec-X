"""Phase 2B.5: catalog query coverage and source gain.

Question
--------
Phase 2B.4 showed that the Agent's independent ``catalog_search`` source returned no target
that the ``history`` source (Serving V1) did not already return, on 40 cases at depth 50.
It also showed *why* the catalog query can miss: the frozen M4 rule
(:func:`experiments.m4_query_rules.deterministic_terms`) spends its whole eight-term budget
on the newest title, so a verbose recent purchase can crowd out every other history item.

This driver separates those two effects on a fresh, protocol-correct sample:

    A  history only                 (Serving V1, pre-fused; the baseline)
    B  history + current catalog    (the frozen M4 query: newest-first, first-come budget)
    C  history + balanced catalog   (experiment-only: round-robin per-title term budget)
    D  catalog only, current query  (the raw lexical source on its own)

Fixed for every arm: ``agent_source_limit = 50``, ``requested_k = 5``, the same cases,
histories, targets, catalogue, checkpoints and ordering.  TIGER is not enabled.

Task and history
----------------
The task is **next-item prediction** under ``agentrecx.eval_protocol.v1``:
``history = case.test_history = items[:-1]`` and ``target = items[-1]``.  The query is a
proxy built only from titles of interactions that happened before the target; the target's
own identity/text is never read (asserted and tested).

Sample
------
About 200 valid cases starting at artifact index 6500.  Users in the frozen 20,000-user
cohort (M3 / M4 / Phase 5 / B1, seed 20260201) are excluded by ``user_int_id``.  Starting at
6500 also avoids the P3 prefix (first 1,000) and the R-LR 5,000-group window (valid cases
#1001-6000).  R-LR's per-case train/validation/test split is not stored in any artifact, so
the sample is labelled "outside the R-LR window by position; split membership unverifiable".
No other frozen cohort was identified.

    .venv/bin/python -m experiments.p2b5_query_coverage --cases 200 \
        --out runs/p2b5_query_coverage_200

Outputs (never written into an existing run directory):
    config.json       frozen configuration, query definitions, sample selection
    cases.json        the cases, their test history and both proxy queries
    samples.jsonl     one record per (arm, case)
    summary.json      per-arm aggregates, paired deltas, source-hit attribution
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
from typing import Any, Callable, Mapping, Sequence

from recommendation import config as project_config
from recommendation.control.similar_item import TOKEN_PATTERN

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
MAPPINGS = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_mappings.json"
M4_HEADS = REPO / "runs" / "m4_evidence" / "heads.npz"
DEFAULT_OUT = REPO / "runs" / "p2b5_query_coverage_200"

#: Four arms.  ``query_mode`` names which proxy construction the catalog source receives.
ARMS: tuple[dict[str, Any], ...] = (
    {"name": "A_history_only", "plan": ("history",), "query_mode": "current"},
    {"name": "B_history_current_catalog", "plan": ("history", "catalog_search"),
     "query_mode": "current"},
    {"name": "C_history_balanced_catalog", "plan": ("history", "catalog_search"),
     "query_mode": "balanced"},
    {"name": "D_catalog_only_current", "plan": ("catalog_search",), "query_mode": "current"},
)

#: Fixed budgets (the brief).  Changing these would be a different experiment.
SOURCE_DEPTH = 50
REQUESTED_K = 5
#: Case selection: outside P3 (0-999) and the R-LR window (valid #1001-6000).
DEFAULT_SKIP = 6500
DEFAULT_CASES = 200
BOOTSTRAP_DRAWS = 2000
SEED = 20261024

HISTORY_SEMANTICS = "case.test_history (train_history + validation_target)"
CURRENT_QUERY_SOURCE = "frozen-m4-deterministic-terms"
BALANCED_QUERY_SOURCE = "balanced-title-budget-experiment"

#: The balanced construction's constants.  They mirror the frozen rule's window and budget
#: (three newest items, eight terms, three-character minimum) and change only the allocation.
BALANCED_TITLES = 3
BALANCED_MAX_TERMS = 8
BALANCED_TERMS_PER_TITLE = 3
MIN_TERM_LENGTH = 3


# --------------------------------------------------------------------------- #
# query construction (experiment-only; production is untouched)
# --------------------------------------------------------------------------- #


def _title_tokens(text: object, *, limit: int) -> list[str]:
    """Distinct tokens of one title, same tokeniser as the frozen M4 rule."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.casefold()):
        if len(token) < MIN_TERM_LENGTH or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def balanced_title_terms(
    history: Sequence[str],
    *,
    describe: Callable[[str], str | None],
    titles: int = BALANCED_TITLES,
    max_terms: int = BALANCED_MAX_TERMS,
    per_title: int = BALANCED_TERMS_PER_TITLE,
) -> tuple[str, ...]:
    """Terms from the newest titles with an even per-title budget.

    The frozen rule fills its budget newest-title-first, so one verbose title can consume all
    eight slots.  This construction takes the newest ``titles`` titles and allocates
    round-robin: every title contributes its first term before any title contributes its
    second, and so on, stopping at ``max_terms``.  It is a pure function of the trusted
    history; it reads no target and no future interaction.
    """
    recent = list(history)[-titles:][::-1]  # newest first
    pools = [_title_tokens(describe(identity), limit=per_title) for identity in recent]
    terms: list[str] = []
    seen: set[str] = set()
    for round_index in range(per_title):
        for pool in pools:
            if round_index >= len(pool):
                continue
            term = pool[round_index]
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= max_terms:
                return tuple(terms)
    return tuple(terms)


def build_query(
    history: Sequence[str],
    *,
    describe: Callable[[str], str | None],
    mode: str,
) -> tuple[str, ...]:
    """The proxy query terms for one case under ``mode`` (``current`` or ``balanced``)."""
    if mode == "current":
        return deterministic_terms(list(history), describe=describe)
    if mode == "balanced":
        return balanced_title_terms(list(history), describe=describe)
    raise ValueError(f"unknown query mode {mode!r}")


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def _frozen_cohort_users() -> set[int]:
    """The frozen 20,000-user cohort's ``user_int_id`` values (M3/M4/Phase 5/B1)."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency
        raise SystemExit("numpy is required to read the frozen cohort") from exc
    data = np.load(M4_HEADS, allow_pickle=True)
    return {int(value) for value in data["user_int_ids"].tolist()}


def load_cases(
    limit: int,
    skip: int,
    *,
    excluded_users: set[int],
    user_map: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The protocol-correct sample: ``history = items[:-1]``, target ``items[-1]``.

    Users in the frozen 20k cohort are dropped.  The remaining cases are taken in artifact
    order from ``skip``; the returned stats record how many were scanned and excluded.
    """
    document = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    scanned = 0
    excluded_cohort = 0
    for index, sequence in enumerate(document["sequences"]):
        if index < skip:
            continue
        items = sequence["item_ids"]
        asins = sequence["parent_asins"]
        if len(items) < 4:
            continue
        scanned += 1
        user_identity = sequence.get("user_int_id")
        user_int_id = int(user_identity) if user_identity is not None else user_map.get(
            sequence["user_id"], -1
        )
        if user_int_id in excluded_users:
            excluded_cohort += 1
            continue
        cases.append(
            {
                "case_index": index,
                "user_id": sequence["user_id"],
                "user_int_id": user_int_id,
                "history": asins[:-1],
                "validation_target": asins[-2],
                "target": asins[-1],
                "history_length": len(asins) - 1,
            }
        )
        if len(cases) >= limit:
            break
    return cases, {
        "scanned": scanned,
        "excluded_by_frozen_cohort": excluded_cohort,
        "selected": len(cases),
    }


def attach_queries(cases: list[dict[str, Any]], title_of: Any) -> list[dict[str, Any]]:
    """Attach both proxy queries to every case, in place.

    A case is usable only when **both** queries are non-empty, so all four arms run on
    exactly the same cases.  The target is never passed to either construction; only history
    identities are read.
    """
    for case in cases:
        current = build_query(case["history"], describe=title_of, mode="current")
        balanced = build_query(case["history"], describe=title_of, mode="balanced")
        case["current_terms"] = list(current)
        case["balanced_terms"] = list(balanced)
        case["current_query"] = " ".join(current) if current else None
        case["balanced_query"] = " ".join(balanced) if balanced else None
        case["usable"] = bool(current) and bool(balanced)
        case["excluded_reason"] = None if case["usable"] else "empty_query"
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


def _source_record(call: Mapping[str, Any] | None, target: str) -> dict[str, Any]:
    if call is None:
        return {"called": False}
    ranked = list(call.get("ranked") or [])
    returned = int(call.get("returned") or 0)
    internal = call.get("internal") or {}
    return {
        "called": True,
        "pre_fused": bool(call.get("pre_fused")),
        "requested_limit": SOURCE_DEPTH,
        "effective_limit": int(call.get("transport_limit") or 0),
        "returned": returned,
        "rank_of_target": _rank_of(ranked, target),
        "ms": call.get("ms"),
        "failed": call.get("failed"),
        "score_kind": call.get("score_kind"),
        "query": call.get("query"),
        "short_of_limit": returned < SOURCE_DEPTH,
        "ranked": ranked,
        "target_internal_provenance": (internal.get("provenance") or {}).get(target)
        if internal
        else None,
    }


def build_sample(
    *,
    arm: Mapping[str, Any],
    case: Mapping[str, Any],
    query: str,
    response: Any,
    probe: SourceProbe,
    elapsed_ms: float,
) -> dict[str, Any]:
    target = str(case["target"])
    history = _source_record(_first(probe.of("history")), target)
    catalog = _source_record(_first(probe.of("catalog_search")), target)
    history_ids = list(history.get("ranked") or [])
    catalog_ids = list(catalog.get("ranked") or [])

    pool_order = [
        str(view.parent_asin)
        for view in response.grounded.candidates
        if _status(view) == "grounded"
    ]
    final_order = [str(item.parent_asin) for item in response.recommendations]
    pool_rank = _rank_of(pool_order, target)
    final_rank = _rank_of(final_order, target)

    return {
        "arm": arm["name"],
        "query_mode": arm["query_mode"],
        "case_index": case["case_index"],
        "user_id": case["user_id"],
        "user_int_id": case["user_int_id"],
        "target": target,
        "message": query,
        "query_source": (
            CURRENT_QUERY_SOURCE if arm["query_mode"] == "current" else BALANCED_QUERY_SOURCE
        ),
        "query_terms": list(case.get(f"{arm['query_mode']}_terms") or ()),
        "history_semantics": HISTORY_SEMANTICS,
        "history_length": int(case["history_length"]),
        "status": "ok",
        "returned": len(final_order),
        "ranked": final_order,
        "pool_size": len(pool_order),
        "target_in_pool": pool_rank is not None,
        "target_pool_rank": pool_rank,
        "target_final_rank": final_rank,
        "sources": {"history": history, "catalog_search": catalog},
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
        "mean_catalog_unique_candidates": _mean(
            [float(row["unique_candidates"]["catalog_only"]) for row in ok]
        ),
        "pooled_cases": len(pooled),
        "pooled_not_top_k": sum(
            1 for row in pooled if row["target_final_rank"] is None
        ),
        "pool_rank_when_pooled": sorted(
            int(row["target_pool_rank"])
            for row in pooled
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
        "source_failures": sum(
            1
            for row in ok
            for source in ("history", "catalog_search")
            if row["sources"][source].get("failed")
        ),
        "catalog_query_terms_mean": _mean(
            [float(len(row["query_terms"])) for row in ok if row["query_terms"]]
        ),
    }


def _paired(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_arm.setdefault(str(row["arm"]), {})[int(row["case_index"])] = row
    baseline_name = arms[0]["name"]
    baseline = by_arm.get(baseline_name, {})
    extractors = {
        "hr@5": lambda row: float(row["hr@5"]),
        "ndcg@5": lambda row: float(row["ndcg@5"]),
        "pool_coverage": lambda row: 1.0 if row["target_in_pool"] else 0.0,
    }
    paired: dict[str, dict[str, Any]] = {}
    for arm in arms[1:]:
        name = arm["name"]
        shared = sorted(set(baseline) & set(by_arm.get(name, {})))
        for metric, extract in extractors.items():
            diffs = [extract(by_arm[name][index]) - extract(baseline[index]) for index in shared]
            paired[f"{name}_minus_{baseline_name}_{metric}"] = {
                "delta": round(sum(diffs) / len(diffs), 4) if diffs else 0.0,
                "ci95": paired_ci(diffs),
                "n": len(diffs),
            }
    return paired


def _hit_attribution(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per arm: who returned the target, and whether C recovers what B missed."""
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_arm.setdefault(str(row["arm"]), {})[int(row["case_index"])] = row
    result: dict[str, Any] = {}
    for arm in arms:
        name = arm["name"]
        rows_for_arm = by_arm.get(name, {})
        counts = {"both": 0, "history_only": 0, "catalog_only": 0, "neither": 0}
        details = {"history_only": [], "catalog_only": []}
        for row in rows_for_arm.values():
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
        result[name] = {"target_hit_by": counts, "examples": details}

    # C vs B: cases where the balanced query recovers a target the frozen query missed.
    current = by_arm.get("B_history_current_catalog", {})
    balanced = by_arm.get("C_history_balanced_catalog", {})
    recovered = []
    lost = []
    for index in sorted(set(current) & set(balanced)):
        current_rank = current[index]["sources"]["catalog_search"].get("rank_of_target")
        balanced_rank = balanced[index]["sources"]["catalog_search"].get("rank_of_target")
        if balanced_rank is not None and current_rank is None:
            recovered.append({"case_index": index, "balanced_rank": balanced_rank})
        if current_rank is not None and balanced_rank is None:
            lost.append({"case_index": index, "current_rank": current_rank})
    result["C_vs_B_catalog_hits"] = {
        "recovered": recovered,
        "lost": lost,
        "n": len(set(current) & set(balanced)),
    }
    return result


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

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    user_map = mappings.get("user2id") or {}
    excluded_users = _frozen_cohort_users()

    cases, selection = load_cases(
        args.cases, args.skip, excluded_users=excluded_users, user_map=user_map
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
    run_cases = [case for case in cases if case["usable"]]
    if not run_cases:
        raise SystemExit("no case has both proxy queries")

    engine = getattr(runtime, "engine", None)
    engine_config = getattr(engine, "config", None)
    config = {
        "task": "next_item_prediction",
        "history_semantics": HISTORY_SEMANTICS,
        "arms": [arm["name"] for arm in ARMS],
        "source_depth": SOURCE_DEPTH,
        "requested_k": REQUESTED_K,
        "cases_requested": len(cases),
        "cases_usable": len(run_cases),
        "excluded_cases": [c["case_index"] for c in cases if not c["usable"]],
        "skip": args.skip,
        "selection": selection,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "query_rules": {
            "current": {
                "source": CURRENT_QUERY_SOURCE,
                "implementation": "experiments.m4_query_rules.deterministic_terms",
                "newest_items_scanned": 3,
                "max_terms": 8,
                "min_term_length": 3,
                "allocation": "newest-title-first, first-come budget",
            },
            "balanced": {
                "source": BALANCED_QUERY_SOURCE,
                "implementation": "experiments.p2b5_query_coverage.balanced_title_terms",
                "newest_items_scanned": BALANCED_TITLES,
                "max_terms": BALANCED_MAX_TERMS,
                "max_terms_per_title": BALANCED_TERMS_PER_TITLE,
                "min_term_length": MIN_TERM_LENGTH,
                "allocation": "round-robin across the newest titles",
            },
        },
        "sample_isolation": {
            "frozen_20k_cohort_excluded_by_user_int_id": True,
            "frozen_20k_cohort_users": len(excluded_users),
            "p3_prefix_avoided": args.skip >= 1000,
            "r_lr_window": "valid cases #1001-6000; sample starts at artifact index "
                           f"{args.skip} (outside by position)",
            "r_lr_split_membership_verifiable": False,
            "note": "R-LR's per-case split is not stored; exclusion there is positional only",
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
            message=run_cases[0]["current_query"],
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
        for arm in ARMS:
            arm_rows: list[dict[str, Any]] = []
            for case in run_cases:
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
                        "query_source": (
                            CURRENT_QUERY_SOURCE
                            if arm["query_mode"] == "current"
                            else BALANCED_QUERY_SOURCE
                        ),
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
        "cases_usable": len(run_cases),
        "arms": aggregates,
        "paired_vs_A_history_only": _paired(records_out, ARMS),
        "target_source_attribution": _hit_attribution(records_out, ARMS),
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
