"""Phase 5A: decouple the Agent's history source and attribute model contributions.

This driver is an **experiment**, not a serving path.  It answers one question on the
frozen 2B.6 sample:

    A  history = Serving V1 (SASRec + Two-Tower + lexical, fused by RRF)
    B  history = SASRec only
    C  history = SASRec only + an independent Two-Tower source

All three arms also consult the same ``catalog_search`` source, receive the same
``current_query``, use the same ``requested_k=5`` and ``source_limit=50``, and run
through the same CandidatePlane / GroundingVerifier / outer RRF.

No production default is changed.  The driver composes the existing engines and
profiles; the only new file is this experiment.  Results are written to a new run
directory and never overwrite an existing one.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config

from experiments.p2b5_query_coverage import (
    _frozen_cohort_users,
    load_cases as load_protocol_cases,
)
from experiments.p2b_agent_source_limit_ablation import (
    SourceProbe,
    hr_at_k,
    instrument_catalog_search,
    ndcg_at_k,
    paired_ci,
)

REPO = project_config.REPO_ROOT
CASES = REPO / "runs" / "p2b6_query_scale_1000" / "cases.json"
P2B6_DIR = REPO / "runs" / "p2b6_query_scale_1000"
P2B6_CONFIG = P2B6_DIR / "config.json"
P2B6_SUMS = P2B6_DIR / "SHA256SUMS.json"
SASREC_CHECKPOINT = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
SASREC_MANIFEST = REPO / "runs" / "sasrec_canonical_2026" / "run.json"
TWOTOWER_CHECKPOINT = REPO / "runs" / "twotower_public_2026_full" / "checkpoint.pt"
TWOTOWER_CATEGORIES = REPO / "runs" / "twotower_public_2026_full" / "metadata_categories.json"
CATALOG = REPO / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

DEFAULT_OUT = REPO / "runs" / "p5a_source_decoupling_1000"
SOURCE_LIMIT = 50
REQUESTED_K = 5
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261024

ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "A_serving_v1_history_catalog",
        "mode": "rrf3",
        "profile": "serving_v1",
        "plan": ("history", "catalog_search"),
        "two_tower_source": False,
        "describe": "Serving V1 pre-fused history + catalog_search",
    },
    {
        "name": "B_sasrec_history_catalog",
        "mode": "sasrec",
        "profile": None,
        "plan": ("history", "catalog_search"),
        "two_tower_source": False,
        "describe": "SASRec-only history + catalog_search",
    },
    {
        "name": "C_sasrec_twotower_history_catalog",
        "mode": "sasrec",
        "profile": None,
        "plan": ("history", "two_tower", "catalog_search"),
        "two_tower_source": True,
        "describe": "SASRec-only history + independent Two-Tower + catalog_search",
    },
)


class ExperimentBlocked(RuntimeError):
    """A required artifact or engine could not be used; never continue degraded."""


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(fraction * (len(ordered) - 1))], 2)


def _rank_of(identities: Sequence[str], target: str) -> int | None:
    try:
        return list(identities).index(target) + 1
    except ValueError:
        return None


@contextmanager
def patched_env(values: Mapping[str, str | None]):
    """Temporarily set/remove environment variables and restore them exactly."""
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _git_head() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _git_tracked_changes() -> list[str]:
    try:
        import subprocess

        output = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line for line in output.splitlines() if line and not line.startswith("??")]
    except Exception:  # noqa: BLE001 - provenance only
        return []


# --------------------------------------------------------------------------- #
# sample identity and query validation
# --------------------------------------------------------------------------- #


def load_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the frozen 2B.6 cases and verify their identity against the protocol source."""
    if not CASES.exists():
        raise ExperimentBlocked(f"missing 2B.6 cases: {CASES}")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 1000:
        raise ExperimentBlocked(f"expected 1,000 frozen 2B.6 cases, got {len(cases)}")

    required = {
        "case_index", "user_id", "user_int_id", "history", "target",
        "history_length", "current_query", "current_terms", "usable",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ExperimentBlocked(f"case {case.get('case_index')} missing {sorted(missing)}")
        if not case["usable"] or not case["current_query"] or not case["current_terms"]:
            raise ExperimentBlocked(f"case {case['case_index']} has no usable query")
        if " ".join(case["current_terms"]) != case["current_query"]:
            raise ExperimentBlocked(
                f"case {case['case_index']} query/terms disagree"
            )

    indices = [int(case["case_index"]) for case in cases]
    if len(set(indices)) != len(indices):
        raise ExperimentBlocked("duplicate case_index in 2B.6 cases")
    if any(case["target"] in case["history"] for case in cases):
        # The protocol permits a repeated item in principle; record it rather than fail.
        pass

    sums = json.loads(P2B6_SUMS.read_text(encoding="utf-8"))
    actual_hash = sha256_file(CASES)
    declared = sums.get("cases.json", {}).get("sha256")
    if actual_hash != declared:
        raise ExperimentBlocked(
            f"2B.6 cases hash mismatch: {actual_hash} != declared {declared}"
        )

    config = json.loads(P2B6_CONFIG.read_text(encoding="utf-8"))
    if int(config.get("source_depth", -1)) != SOURCE_LIMIT:
        raise ExperimentBlocked(f"2B.6 source_depth is not {SOURCE_LIMIT}")
    if int(config.get("requested_k", -1)) != REQUESTED_K:
        raise ExperimentBlocked(f"2B.6 requested_k is not {REQUESTED_K}")

    # Reconstruct the same protocol sample and compare every saved (history, target) pair.
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    user_map = mappings.get("user2id") or {}
    excluded = _frozen_cohort_users()
    rebuilt, selection = load_protocol_cases(
        1025, 6710, excluded_users=excluded, user_map=user_map
    )
    by_index = {int(item["case_index"]): item for item in rebuilt}
    mismatches = []
    for case in cases:
        reference = by_index.get(int(case["case_index"]))
        if reference is None:
            mismatches.append({"case_index": case["case_index"], "reason": "missing"})
            continue
        for field in ("user_id", "user_int_id", "history", "target",
                      "validation_target", "history_length"):
            if reference.get(field) != case.get(field):
                mismatches.append(
                    {"case_index": case["case_index"], "field": field,
                     "saved": case.get(field), "rebuilt": reference.get(field)}
                )
        if int(case["user_int_id"]) in excluded:
            mismatches.append({"case_index": case["case_index"], "reason": "cohort_user"})
    if mismatches:
        raise ExperimentBlocked(f"2B.6 case reconstruction mismatches: {mismatches[:5]}")

    validation = {
        "cases": len(cases),
        "case_indices_first_last": [indices[0], indices[-1]],
        "target_equals_last_sequence_item": True,
        "history_semantics": "case.test_history (train_history + validation_target)",
        "reconstructed_from": str(SEQUENCES.relative_to(REPO)),
        "reconstruction_stats": selection,
        "frozen_20k_cohort_excluded": True,
        "frozen_20k_cohort_users": len(excluded),
        "case_hash": actual_hash,
        "p2b6_config_hash": sha256_file(P2B6_CONFIG),
        "p2b6_driver_hash": config.get("driver_sha256"),
        "query_source": "runs/p2b6_query_scale_1000/cases.json:current_query",
        "queries_reconstructed": False,
        "queries_used_verbatim": True,
        "target_in_history_cases": sum(1 for c in cases if c["target"] in c["history"]),
    }
    return cases, validation


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #


def instrument_history(tool: Any, probe: SourceProbe, *, pre_fused: bool) -> None:
    """Observe the accepted history Tool without replacing the object.

    The captured ``provenance`` is the Serving V1 engine's own per-candidate inner-source
    record.  It is kept only when the engine actually returns it; B/C have no inner fusion
    and therefore record an empty provenance rather than inventing attribution.
    """
    original = tool.run

    def observed_run(request: Any = None, context: Any = None) -> Any:
        transport_k = int(getattr(request, "k", 0) or 0)
        began = time.perf_counter()
        try:
            result = original(request=request, context=context)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            probe.record(
                {
                    "source": "history",
                    "pre_fused": pre_fused,
                    "transport_limit": transport_k,
                    "returned": 0,
                    "ranked": [],
                    "scores": [],
                    "score_kind": None,
                    "query": None,
                    "failed": type(exc).__name__,
                    "ms": round((time.perf_counter() - began) * 1000.0, 2),
                }
            )
            raise
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        identities = [str(item.parent_asin) for item in result.recommendations]
        provenance = dict(getattr(result, "provenance", {}) or {})
        inner_provenance = {
            str(identity): {
                "sources": [str(name) for name in meta.get("sources", ())],
                "source_ranks": {
                    str(name): int(rank)
                    for name, rank in (meta.get("source_ranks") or {}).items()
                },
            }
            for identity, meta in provenance.items()
        }
        probe.record(
            {
                "source": "history",
                "pre_fused": pre_fused,
                "transport_limit": transport_k,
                "returned": len(identities),
                "ranked": identities,
                "scores": [float(item.score) for item in result.recommendations],
                "score_kind": getattr(result, "score_kind", None),
                "query": None,
                "failed": None,
                "ms": round(elapsed_ms, 2),
                "internal": {
                    "requested_k": int(getattr(result, "requested_k", 0) or 0),
                    "returned_k": int(getattr(result, "returned_k", len(identities)) or 0),
                    "history_length": int(getattr(result, "history_length", 0) or 0),
                    "effective_history_length": int(
                        getattr(result, "effective_history_length", 0) or 0
                    ),
                    "history_truncated": bool(getattr(result, "history_truncated", False)),
                    "timings_ms": dict(getattr(result, "timings_ms", {}) or {}),
                    "provenance_size": len(provenance),
                    "provenance": inner_provenance,
                },
            }
        )
        return result

    tool.run = observed_run


def instrument_two_tower(tool: Any, probe: SourceProbe) -> None:
    """Observe the independent Two-Tower source's own ranking and timing."""
    original = tool.propose

    def observed_propose(*, arguments: Any, limit: int, history: tuple[str, ...] = ()) -> Any:
        began = time.perf_counter()
        try:
            rows = original(arguments=arguments, limit=limit, history=history)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            probe.record(
                {
                    "source": "two_tower",
                    "pre_fused": False,
                    "transport_limit": int(limit),
                    "returned": 0,
                    "ranked": [],
                    "scores": [],
                    "score_kind": "two_tower_cosine",
                    "query": None,
                    "failed": type(exc).__name__,
                    "ms": round((time.perf_counter() - began) * 1000.0, 2),
                }
            )
            raise
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        probe.record(
            {
                "source": "two_tower",
                "pre_fused": False,
                "transport_limit": int(limit),
                "returned": len(rows),
                "ranked": [str(row[0]) for row in rows],
                "scores": [float(row[2]) for row in rows],
                "score_kind": "two_tower_cosine",
                "query": None,
                "failed": None,
                "ms": round(elapsed_ms, 2),
            }
        )
        return rows

    tool.propose = observed_propose


# --------------------------------------------------------------------------- #
# source view / one sample
# --------------------------------------------------------------------------- #


def _first(calls: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return calls[0] if calls else None


def _source_view(call: Mapping[str, Any] | None, target: str) -> dict[str, Any]:
    if call is None:
        return {"called": False}
    ranked = list(call.get("ranked") or ())
    effective = int(call.get("transport_limit") or 0)
    view: dict[str, Any] = {
        "called": True,
        "pre_fused": bool(call.get("pre_fused")),
        "requested_limit": SOURCE_LIMIT,
        "effective_limit": effective,
        "capped": effective < SOURCE_LIMIT,
        "returned": int(call.get("returned") or 0),
        "rank_of_target": _rank_of(ranked, target),
        "ms": call.get("ms"),
        "failed": call.get("failed"),
        "score_kind": call.get("score_kind"),
        "query": call.get("query"),
        "exhausted_short_of_limit": int(call.get("returned") or 0) < effective,
        "ranked": ranked,
    }
    internal = call.get("internal")
    if internal:
        provenance = dict(internal.get("provenance") or {})
        counts: dict[str, int] = {}
        only: dict[str, int] = {}
        for meta in provenance.values():
            sources = [str(name) for name in (meta.get("sources") or ())]
            for source in sources:
                counts[source] = counts.get(source, 0) + 1
            if len(sources) == 1:
                only[sources[0]] = only.get(sources[0], 0) + 1
        view["internal"] = {
            "requested_k": internal.get("requested_k"),
            "returned_k": internal.get("returned_k"),
            "history_length": internal.get("history_length"),
            "effective_history_length": internal.get("effective_history_length"),
            "history_truncated": internal.get("history_truncated"),
            "timings_ms": internal.get("timings_ms", {}),
            "provenance_size": internal.get("provenance_size"),
            "source_candidate_counts": counts,
            "source_only_counts": only,
            "target_provenance": provenance.get(target),
        }
    return view


def _source_only(lists: Mapping[str, set[str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, values in lists.items():
        others: set[str] = set()
        for other, other_values in lists.items():
            if other != name:
                others |= other_values
        result[name] = len(values - others)
    return result


def _pairwise_overlap(lists: Mapping[str, set[str]]) -> dict[str, int]:
    names = sorted(lists)
    result: dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            result[f"{left}&{right}"] = len(lists[left] & lists[right])
    return result


def _build_row(
    *,
    arm: Mapping[str, Any],
    case: Mapping[str, Any],
    response: Any,
    probe: SourceProbe,
    elapsed_ms: float,
) -> dict[str, Any]:
    target = str(case["target"])
    source_names = [name for name in arm["plan"]]
    views = {name: _source_view(_first(probe.of(name)), target) for name in source_names}

    pool_order = [
        str(view.parent_asin)
        for view in response.grounded.candidates
        if str(getattr(view.verification_status, "value", view.verification_status)) == "grounded"
    ]
    final_order = [str(item.parent_asin) for item in response.recommendations]
    pool_rank = _rank_of(pool_order, target)
    final_rank = _rank_of(final_order, target)

    lists = {
        name: set(view.get("ranked") or ())
        for name, view in views.items()
        if view.get("called")
    }
    union = set().union(*lists.values()) if lists else set()
    returned_total = sum(len(values) for values in lists.values())
    duplicate_count = returned_total - len(union)
    pairwise = _pairwise_overlap(lists)
    intersection_all = set.intersection(*lists.values()) if len(lists) > 1 else set(union)
    jaccard = round(len(union) / returned_total, 6) if returned_total else None
    if len(lists) > 1:
        # Mean pairwise Jaccard is more interpretable than all-source intersection for 3 lists.
        pair_scores = []
        for i, left in enumerate(sorted(lists)):
            for right in sorted(lists)[i + 1:]:
                pair_union = lists[left] | lists[right]
                pair_scores.append(len(lists[left] & lists[right]) / len(pair_union) if pair_union else 0.0)
        mean_pairwise_jaccard = round(sum(pair_scores) / len(pair_scores), 6) if pair_scores else None
    else:
        mean_pairwise_jaccard = None

    unique = _source_only(lists)
    hit_by_source = {
        name: views[name].get("rank_of_target") is not None
        for name in source_names
        if views[name].get("called")
    }
    inner = (views.get("history", {}).get("internal") or {})
    inner_target = inner.get("target_provenance")

    row = {
        "arm": arm["name"],
        "arm_describe": arm["describe"],
        "case_index": int(case["case_index"]),
        "user_id": case["user_id"],
        "user_int_id": case["user_int_id"],
        "target": target,
        "message": case["current_query"],
        "query_terms": list(case["current_terms"]),
        "history_semantics": "case.test_history (train_history + validation_target)",
        "history_length": int(case["history_length"]),
        "source_limit": SOURCE_LIMIT,
        "requested_k": REQUESTED_K,
        "status": "ok",
        "returned": len(final_order),
        "ranked": final_order,
        "pool_size": len(pool_order),
        "pool_before_grounding": len(union),
        "target_in_pool": pool_rank is not None,
        "target_pool_rank": pool_rank,
        "target_final_rank": final_rank,
        "sources": views,
        "source_hits_target": hit_by_source,
        "source_union_size": len(union),
        "source_returned_total": returned_total,
        "duplicate_before_dedup": duplicate_count,
        "dedup_after": len(union),
        "duplicate_rate": round(duplicate_count / returned_total, 6) if returned_total else None,
        "mean_pairwise_jaccard": mean_pairwise_jaccard,
        "all_source_intersection": len(intersection_all),
        "pairwise_overlap": pairwise,
        "unique_candidates": unique,
        "history_inner_target_provenance": inner_target,
        "history_inner_source_counts": inner.get("source_candidate_counts") or {},
        "history_inner_source_only": inner.get("source_only_counts") or {},
        "hr@5": hr_at_k(final_order, target, REQUESTED_K),
        "ndcg@5": ndcg_at_k(final_order, target, REQUESTED_K),
        "ms": round(elapsed_ms, 2),
    }
    return row


def _error_row(
    *,
    arm: Mapping[str, Any],
    case: Mapping[str, Any],
    probe: SourceProbe,
    elapsed_ms: float,
    error: str,
) -> dict[str, Any]:
    target = str(case["target"])
    views = {
        name: _source_view(_first(probe.of(name)), target)
        for name in arm["plan"]
    }
    return {
        "arm": arm["name"],
        "arm_describe": arm["describe"],
        "case_index": int(case["case_index"]),
        "user_id": case["user_id"],
        "user_int_id": case["user_int_id"],
        "target": target,
        "message": case["current_query"],
        "query_terms": list(case["current_terms"]),
        "history_semantics": "case.test_history (train_history + validation_target)",
        "history_length": int(case["history_length"]),
        "source_limit": SOURCE_LIMIT,
        "requested_k": REQUESTED_K,
        "status": f"error:{type(error).__name__ if not isinstance(error, str) else 'Exception'}",
        "returned": 0,
        "ranked": [],
        "pool_size": 0,
        "pool_before_grounding": 0,
        "target_in_pool": False,
        "target_pool_rank": None,
        "target_final_rank": None,
        "sources": views,
        "source_hits_target": {},
        "source_union_size": 0,
        "source_returned_total": 0,
        "duplicate_before_dedup": 0,
        "dedup_after": 0,
        "duplicate_rate": None,
        "mean_pairwise_jaccard": None,
        "all_source_intersection": 0,
        "pairwise_overlap": {},
        "unique_candidates": {},
        "history_inner_target_provenance": None,
        "history_inner_source_counts": {},
        "history_inner_source_only": {},
        "hr@5": 0.0,
        "ndcg@5": 0.0,
        "ms": round(elapsed_ms, 2),
        "error": str(error)[:500],
    }


# --------------------------------------------------------------------------- #
# aggregates and paired statistics
# --------------------------------------------------------------------------- #


def _aggregate(rows: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    latencies = [float(row["ms"]) for row in ok]
    source_names = list(arm["plan"])
    source_stats: dict[str, Any] = {}
    for source in source_names:
        calls = [row["sources"][source] for row in ok if row["sources"].get(source, {}).get("called")]
        source_stats[source] = {
            "called_cases": len(calls),
            "mean_returned": _mean([float(call.get("returned") or 0) for call in calls]),
            "mean_only": _mean(
                [float(row["unique_candidates"].get(source, 0)) for row in ok]
            ),
            "mean_ms": _mean([float(call.get("ms") or 0.0) for call in calls]),
            "p50_ms": _percentile([float(call.get("ms") or 0.0) for call in calls], 0.50),
            "p95_ms": _percentile([float(call.get("ms") or 0.0) for call in calls], 0.95),
            "failed_calls": sum(1 for call in calls if call.get("failed")),
            "target_hits": sum(
                1 for row in ok if row["sources"][source].get("rank_of_target") is not None
            ),
            "mean_requested_limit": _mean(
                [float(call.get("requested_limit") or 0) for call in calls]
            ),
            "mean_effective_limit": _mean(
                [float(call.get("effective_limit") or 0) for call in calls]
            ),
            "capped_calls": sum(1 for call in calls if call.get("capped")),
            "short_calls": sum(1 for call in calls if call.get("exhausted_short_of_limit")),
        }

    inner_counts: dict[str, int] = {}
    inner_only: dict[str, int] = {}
    for row in ok:
        for name, count in (row.get("history_inner_source_counts") or {}).items():
            inner_counts[name] = inner_counts.get(name, 0) + int(count)
        for name, count in (row.get("history_inner_source_only") or {}).items():
            inner_only[name] = inner_only.get(name, 0) + int(count)

    target_hit_by_source = {
        source: sum(
            1 for row in ok if row["sources"].get(source, {}).get("rank_of_target") is not None
        )
        for source in source_names
    }

    return {
        "cases": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "failure_rate": round((len(rows) - len(ok)) / len(rows), 6) if rows else None,
        "pool_coverage": _mean([1.0 if row["target_in_pool"] else 0.0 for row in ok]),
        "hr@5": _mean([float(row["hr@5"]) for row in ok]),
        "ndcg@5": _mean([float(row["ndcg@5"]) for row in ok]),
        "mean_pool_size": _mean([float(row["pool_size"]) for row in ok]),
        "mean_returned": _mean([float(row["returned"]) for row in ok]),
        "mean_source_returned": {
            source: source_stats[source]["mean_returned"] for source in source_names
        },
        "mean_source_only": {
            source: source_stats[source]["mean_only"] for source in source_names
        },
        "mean_duplicate_rate": _mean([float(row["duplicate_rate"] or 0.0) for row in ok]),
        "mean_pairwise_jaccard": _mean(
            [float(row["mean_pairwise_jaccard"] or 0.0) for row in ok]
        ),
        "mean_source_union": _mean([float(row["source_union_size"]) for row in ok]),
        "mean_pool_before_grounding": _mean(
            [float(row["pool_before_grounding"]) for row in ok]
        ),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "source_stats": source_stats,
        "source_failures_total": sum(
            stats["failed_calls"] for stats in source_stats.values()
        ),
        "target_hit_by_source": target_hit_by_source,
        "target_in_pool_cases": sum(1 for row in ok if row["target_in_pool"]),
        "target_in_top5_cases": sum(
            1 for row in ok if row["target_final_rank"] is not None and row["target_final_rank"] <= REQUESTED_K
        ),
        "pooled_target_pool_ranks": sorted(
            int(row["target_pool_rank"]) for row in ok if row["target_pool_rank"] is not None
        ),
        "history_inner_source_counts": inner_counts,
        "history_inner_source_only": inner_only,
        "history_inner_target_cases": sum(
            1 for row in ok if row["history_inner_target_provenance"] is not None
        ),
    }


def _paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            by_arm.setdefault(str(row["arm"]), {})[int(row["case_index"])] = row

    pairs = (
        ("B_sasrec_history_catalog", "A_serving_v1_history_catalog"),
        ("C_sasrec_twotower_history_catalog", "A_serving_v1_history_catalog"),
        ("C_sasrec_twotower_history_catalog", "B_sasrec_history_catalog"),
    )
    metrics = {
        "pool_coverage": lambda row: 1.0 if row["target_in_pool"] else 0.0,
        "hr@5": lambda row: float(row["hr@5"]),
        "ndcg@5": lambda row: float(row["ndcg@5"]),
    }
    result: dict[str, Any] = {}
    for left, right in pairs:
        shared = sorted(set(by_arm.get(left, {})) & set(by_arm.get(right, {})))
        for metric, extract in metrics.items():
            diffs = [extract(by_arm[left][index]) - extract(by_arm[right][index]) for index in shared]
            result[f"{left}_minus_{right}__{metric}"] = {
                "delta": round(sum(diffs) / len(diffs), 6) if diffs else 0.0,
                "ci95": paired_ci(diffs, draws=BOOTSTRAP_DRAWS),
                "n": len(diffs),
            }
    return result


def _attribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            by_arm.setdefault(str(row["arm"]), {})[int(row["case_index"])] = row

    result: dict[str, Any] = {}
    for arm, cases in by_arm.items():
        hit_counts = {"both": 0, "history_only": 0, "catalog_only": 0,
                      "two_tower_only": 0, "neither": 0}
        examples: dict[str, list[dict[str, Any]]] = {
            "history_only": [], "catalog_only": [], "two_tower_only": []
        }
        for index, row in cases.items():
            hit = {
                name: row["sources"].get(name, {}).get("rank_of_target") is not None
                for name in row["sources"]
                if row["sources"][name].get("called")
            }
            names = [name for name, present in hit.items() if present]
            if len(names) >= 2:
                hit_counts["both"] += 1
            elif names == ["history"]:
                hit_counts["history_only"] += 1
                examples["history_only"].append(
                    {"case_index": index, "history_rank": row["sources"]["history"]["rank_of_target"]}
                )
            elif names == ["catalog_search"]:
                hit_counts["catalog_only"] += 1
                examples["catalog_only"].append(
                    {"case_index": index, "catalog_rank": row["sources"]["catalog_search"]["rank_of_target"]}
                )
            elif names == ["two_tower"]:
                hit_counts["two_tower_only"] += 1
                examples["two_tower_only"].append(
                    {"case_index": index, "two_tower_rank": row["sources"]["two_tower"]["rank_of_target"]}
                )
            else:
                hit_counts["neither"] += 1
        result[arm] = {"target_hit_by": hit_counts, "examples": examples}

    current = by_arm.get("B_sasrec_history_catalog", {})
    with_tt = by_arm.get("C_sasrec_twotower_history_catalog", {})
    recovered = []
    lost = []
    tt_unique = []
    for index in sorted(set(current) & set(with_tt)):
        b_rank = current[index]["target_final_rank"]
        c_rank = with_tt[index]["target_final_rank"]
        tt_rank = with_tt[index]["sources"].get("two_tower", {}).get("rank_of_target")
        if (b_rank is None or b_rank > REQUESTED_K) and c_rank is not None and c_rank <= REQUESTED_K:
            recovered.append({"case_index": index, "B_final_rank": b_rank, "C_final_rank": c_rank})
        if b_rank is not None and b_rank <= REQUESTED_K and (c_rank is None or c_rank > REQUESTED_K):
            lost.append({"case_index": index, "B_final_rank": b_rank, "C_final_rank": c_rank})
        if tt_rank is not None and current[index]["sources"]["history"].get("rank_of_target") is None:
            tt_unique.append({"case_index": index, "two_tower_rank": tt_rank, "C_final_rank": c_rank})
    result["C_vs_B"] = {
        "recovered_top5": recovered,
        "lost_top5": lost,
        "two_tower_target_cases": tt_unique,
        "n": len(set(current) & set(with_tt)),
    }

    # A's inner-source target attribution, available only from Serving V1 provenance.
    a_cases = by_arm.get("A_serving_v1_history_catalog", {})
    inner_target = {"sequential": 0, "two_tower": 0, "catalog_lexical": 0, "none": 0}
    for row in a_cases.values():
        provenance = row.get("history_inner_target_provenance") or {}
        sources = set(provenance.get("sources") or ())
        if not sources:
            inner_target["none"] += 1
        else:
            for source in sources:
                inner_target[source] = inner_target.get(source, 0) + 1
    result["A_inner_target"] = inner_target
    return result


# --------------------------------------------------------------------------- #
# arm composition
# --------------------------------------------------------------------------- #


@dataclass
class ArmContext:
    spec: Mapping[str, Any]
    runtime: Any
    service: Any
    engine: Any
    engine_meta: Mapping[str, Any]
    tt_meta: Mapping[str, Any] | None
    probe: SourceProbe


def _compose_arm(spec: Mapping[str, Any], out: Path) -> ArmContext:
    from recommendation.api.app import ServiceSettings
    from recommendation.demo import build_demo_runtime

    env_values = {
        "AGENTRECX_FUSION_PROFILE": spec.get("profile") or "serving_v1",
        "AGENTRECX_TIGER": "0",
        "AGENTRECX_AGENT_SOURCE_LIMIT": str(SOURCE_LIMIT),
        "AGENTRECX_AGENT_SIMILAR_ITEMS": "0",
        "AGENTRECX_AGENT_MEMORY_RANKING": "0",
        "AGENTRECX_TWO_TOWER_CHECKPOINT": (
            str(TWOTOWER_CHECKPOINT) if spec["two_tower_source"] else None
        ),
        "AGENTRECX_TWO_TOWER_MAPPINGS_PATH": (
            str(MAPPINGS) if spec["two_tower_source"] else None
        ),
    }
    with patched_env(env_values):
        settings = ServiceSettings(serving_fusion=str(spec["mode"]))
        try:
            runtime = build_demo_runtime(
                settings=settings,
                memory_db=out / f"memory_{spec['name']}.sqlite",
            )
        except Exception as exc:  # noqa: BLE001 - reported as a blocked experiment
            raise ExperimentBlocked(
                f"{spec['name']}: runtime could not be composed: {type(exc).__name__}: {exc}"
            ) from exc

        engine = runtime.engine
        if engine is None:
            raise ExperimentBlocked(f"{spec['name']}: runtime returned no engine")
        try:
            meta = dict(engine.model_metadata())
        except Exception as exc:  # noqa: BLE001
            raise ExperimentBlocked(
                f"{spec['name']}: engine metadata failed: {type(exc).__name__}: {exc}"
            ) from exc

        if spec["name"].startswith("A_"):
            if meta.get("active_engine") != "fusion_rrf":
                raise ExperimentBlocked(
                    f"{spec['name']}: expected fusion_rrf, got {meta.get('active_engine')!r}"
                )
            if meta.get("serving_fusion_profile") != "serving_v1":
                raise ExperimentBlocked(
                    f"{spec['name']}: profile {meta.get('serving_fusion_profile')!r}, not serving_v1"
                )
            if tuple(meta.get("fusion_sources") or ()) != (
                "sequential", "two_tower", "catalog_lexical"
            ):
                raise ExperimentBlocked(
                    f"{spec['name']}: unexpected sources {meta.get('fusion_sources')!r}"
                )
            if int(meta.get("source_head_depth") or 0) != 100 or int(meta.get("rrf_k") or 0) != 60:
                raise ExperimentBlocked(
                    f"{spec['name']}: unexpected head/RRF {meta.get('source_head_depth')!r}/"
                    f"{meta.get('rrf_k')!r}"
                )
        else:
            if meta.get("model_type") != "SASRec":
                raise ExperimentBlocked(
                    f"{spec['name']}: expected SASRec, got {meta.get('model_type')!r}"
                )

        try:
            service = runtime.agent_service()
        except Exception as exc:  # noqa: BLE001
            raise ExperimentBlocked(
                f"{spec['name']}: Agent service could not be composed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        tt_meta = None
        if spec["two_tower_source"]:
            tt_tool = getattr(service, "_two_tower_tool", None)
            if tt_tool is None:
                raise ExperimentBlocked(
                    f"{spec['name']}: independent Two-Tower source was not registered"
                )
            try:
                tt_meta = dict(tt_tool.engine.model_metadata())
            except Exception as exc:  # noqa: BLE001
                raise ExperimentBlocked(
                    f"{spec['name']}: Two-Tower metadata failed: {type(exc).__name__}: {exc}"
                ) from exc
            if "two_tower" not in service.available_sources:
                raise ExperimentBlocked(
                    f"{spec['name']}: two_tower not in available_sources "
                    f"{service.available_sources!r}"
                )

        return ArmContext(
            spec=spec,
            runtime=runtime,
            service=service,
            engine=engine,
            engine_meta=meta,
            tt_meta=tt_meta,
            probe=SourceProbe(),
        )


def _preflight(context: ArmContext, case: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the configured engine/source can score a real case before the loop."""
    from recommendation.demo.agent_schemas import AgentRecommendRequest

    spec = context.spec
    if spec["two_tower_source"]:
        tt_tool = getattr(context.service, "_two_tower_tool", None)
        if tt_tool is None:
            raise ExperimentBlocked(f"{spec['name']}: Two-Tower tool disappeared")
        try:
            rows = tt_tool.propose(
                arguments=None, limit=5, history=tuple(case["history"])
            )
        except Exception as exc:  # noqa: BLE001
            raise ExperimentBlocked(
                f"{spec['name']}: Two-Tower preflight failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not rows:
            raise ExperimentBlocked(f"{spec['name']}: Two-Tower preflight returned no candidates")
        tt_candidates = len(rows)
    else:
        tt_candidates = None

    payload = AgentRecommendRequest(
        message=case["current_query"],
        history=case["history"],
        k=REQUESTED_K,
        sources=tuple(spec["plan"]),
    )
    began = time.perf_counter()
    try:
        response = context.service.run(payload)
    except Exception as exc:  # noqa: BLE001
        raise ExperimentBlocked(
            f"{spec['name']}: warm-up request failed: {type(exc).__name__}: {exc}"
        ) from exc
    warm_ms = (time.perf_counter() - began) * 1000.0
    return {
        "warmup_returned": len(response.recommendations),
        "warmup_pool_size": len(response.grounded.candidates),
        "warmup_ms": round(warm_ms, 2),
        "two_tower_preflight_candidates": tt_candidates,
        "available_sources": list(context.service.available_sources),
    }


def _run_arm(
    spec: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    context = _compose_arm(spec, out)
    preflight = _preflight(context, cases[0])

    probe = context.probe
    instrument_history(
        context.service._tool, probe, pre_fused=bool(spec["name"].startswith("A_"))
    )
    instrument_catalog_search(context.service._catalog_search, probe)
    if spec["two_tower_source"]:
        instrument_two_tower(context.service._two_tower_tool, probe)

    rows: list[dict[str, Any]] = []
    try:
        for position, case in enumerate(cases, start=1):
            probe.reset()
            began = time.perf_counter()
            try:
                from recommendation.demo.agent_schemas import AgentRecommendRequest

                payload = AgentRecommendRequest(
                    message=case["current_query"],
                    history=case["history"],
                    k=REQUESTED_K,
                    sources=tuple(spec["plan"]),
                )
                response = context.service.run(payload)
                elapsed_ms = (time.perf_counter() - began) * 1000.0
                row = _build_row(
                    arm=spec, case=case, response=response, probe=probe,
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
                elapsed_ms = (time.perf_counter() - began) * 1000.0
                row = _error_row(
                    arm=spec, case=case, probe=probe, elapsed_ms=elapsed_ms,
                    error=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            if position % 50 == 0 or position == len(cases):
                ok = sum(1 for r in rows if r["status"] == "ok")
                print(
                    f"[{spec['name']}] {position}/{len(cases)} ok={ok} "
                    f"last_ms={rows[-1]['ms']}",
                    flush=True,
                )
    finally:
        closer = getattr(context.runtime, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown must not hide the result
                pass
        del context.service
        del context.runtime
        del context.engine
        gc.collect()

    aggregate = _aggregate(rows, spec)
    metadata = {
        "engine": context.engine_meta,
        "two_tower_source": context.tt_meta,
        "preflight": preflight,
    }
    return rows, aggregate, metadata


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _write_manifest(out: Path) -> None:
    entries = {}
    for name in ("config.json", "cases.json", "samples.jsonl", "summary.json", "report.md"):
        path = out / name
        if path.exists():
            entries[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    (out / "SHA256SUMS.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _report(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    paired = summary["paired"]
    lines = [
        "# Phase 5A — recommendation source decoupling",
        "",
        "> Experiment only.  The default serving profile, RRF weights and candidate depth are unchanged.",
        "",
        "## Configuration",
        "",
        f"- cases: `{config['case_validation']['cases']}` from `runs/p2b6_query_scale_1000/cases.json`",
        f"- requested_k: `{config['requested_k']}`, source_limit: `{config['source_limit']}`",
        f"- SASRec: `{config['models']['sasrec_checkpoint']['path']}`",
        f"- Two-Tower: `{config['models']['two_tower_checkpoint']['path']}`",
        "",
        "## Arms",
        "",
        "| Arm | History source | Sources | Pool coverage | HR@5 | NDCG@5 | mean pool | p50 ms | p95 ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in config["arms"].items():
        agg = arms[name]
        lines.append(
            f"| {name} | {spec['history']} | {', '.join(spec['plan'])} | "
            f"{agg['pool_coverage']} | {agg['hr@5']} | {agg['ndcg@5']} | "
            f"{agg['mean_pool_size']} | {agg['p50_ms']} | {agg['p95_ms']} |"
        )
    lines += ["", "## Paired differences", "", "| Comparison | metric | delta | 95% CI | n |", "|---|---|---:|---|---:|"]
    for key, value in paired.items():
        comparison, metric = key.split("__", 1)
        lines.append(
            f"| {comparison} | {metric} | {value['delta']} | "
            f"[{value['ci95'][0]}, {value['ci95'][1]}] | {value['n']} |"
        )
    lines += ["", "## Source contribution", "", "```json", json.dumps(summary["target_attribution"], indent=2), "```", ""]
    lines += ["## Failures", "", "```json", json.dumps(summary["failures"], indent=2), "```", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    started = time.time()
    config: dict[str, Any] = {
        "task": "phase5a_source_decoupling",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git_head(),
        "git_tracked_changes_before": _git_tracked_changes(),
        "driver_path": str(Path(__file__).resolve().relative_to(REPO)),
        "driver_sha256": sha256_file(Path(__file__)),
        "requested_k": REQUESTED_K,
        "source_limit": SOURCE_LIMIT,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "query_source": "runs/p2b6_query_scale_1000/cases.json:current_query",
        "arms": {
            spec["name"]: {
                "describe": spec["describe"],
                "serving_fusion": spec["mode"],
                "fusion_profile": spec["profile"],
                "history": (
                    "Serving V1: SASRec + Two-Tower + catalog_lexical, inner RRF"
                    if spec["name"].startswith("A_")
                    else "SASRec only"
                ),
                "plan": list(spec["plan"]),
                "two_tower_source": bool(spec["two_tower_source"]),
            }
            for spec in ARMS
        },
    }

    cases, case_validation = load_cases()
    if args.cases < len(cases):
        cases = cases[: args.cases]
        case_validation["cases"] = len(cases)
        case_validation["subset_for_smoke"] = True
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")

    # Model/catalog identity is recorded before any engine is built.
    model_paths = {
        "sasrec_checkpoint": SASREC_CHECKPOINT,
        "sasrec_manifest": SASREC_MANIFEST,
        "two_tower_checkpoint": TWOTOWER_CHECKPOINT,
        "two_tower_categories": TWOTOWER_CATEGORIES,
        "catalog": CATALOG,
        "mappings": MAPPINGS,
        "sequences": SEQUENCES,
    }
    config["models"] = {
        key: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for key, path in model_paths.items()
    }
    config["case_validation"] = case_validation
    config["default_profile_unchanged"] = "serving_v1"
    config["default_rrf_k"] = 60
    config["default_source_head"] = 100
    config["notes"] = {
        "A_vs_B": "architecture comparison: all three inner sources + inner RRF vs SASRec only; not an RRF-layer-only contrast",
        "B_vs_C": "only difference is the independent Two-Tower source",
        "A_inner_attribution": "only available from Serving V1 provenance; B/C have no inner fusion",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(json.dumps({"cases": len(cases), "validation": case_validation}, indent=2), flush=True)

    rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    # The per-source budget is read by _compose_controller at run time, outside the
    # short composition context in _compose_arm; keep it pinned for every run.
    previous_limit = os.environ.get("AGENTRECX_AGENT_SOURCE_LIMIT")
    os.environ["AGENTRECX_AGENT_SOURCE_LIMIT"] = str(SOURCE_LIMIT)
    try:
        for spec in ARMS:
            arm_rows, arm_agg, arm_meta = _run_arm(spec, cases, args.out)
            rows.extend(arm_rows)
            aggregates[spec["name"]] = arm_agg
            metadata[spec["name"]] = arm_meta
            with (args.out / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            print(json.dumps({"arm": spec["name"], "summary": arm_agg}, indent=2), flush=True)
    except ExperimentBlocked as exc:
        (args.out / "blocked.json").write_text(
            json.dumps(
                {
                    "blocked": True,
                    "reason": str(exc),
                    "config": config,
                    "partial_rows": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"blocked": str(exc)}, indent=2), flush=True)
        return 2
    finally:
        if previous_limit is None:
            os.environ.pop("AGENTRECX_AGENT_SOURCE_LIMIT", None)
        else:
            os.environ["AGENTRECX_AGENT_SOURCE_LIMIT"] = previous_limit

    summary = {
        "task": "phase5a_source_decoupling",
        "git_head": config["git_head"],
        "cases": len(cases),
        "requested_k": REQUESTED_K,
        "source_limit": SOURCE_LIMIT,
        "arms": aggregates,
        "paired": _paired(rows),
        "target_attribution": _attribution(rows),
        "arm_metadata": metadata,
        "failures": {
            name: {
                "errors": agg["errors"],
                "failure_rate": agg["failure_rate"],
                "source_failures": agg["source_failures_total"],
            }
            for name, agg in aggregates.items()
        },
        "runtime_seconds": round(time.time() - started, 2),
        "git_tracked_changes_after": _git_tracked_changes(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(_report(summary, config), encoding="utf-8")
    _write_manifest(args.out)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
