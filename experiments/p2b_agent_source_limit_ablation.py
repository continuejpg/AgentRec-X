"""Phase 2B exploratory ablation: the agent's per-source budget against the final ``k``.

Four arms share one request set, one plan, one catalogue, one engine and one final ``k``;
only ``AGENTRECX_AGENT_SOURCE_LIMIT`` changes.  The path executed is the real Agent
control plane (``DemoAgentService.run`` -> ``LoopController`` -> ``CandidatePlane``), not
``/v1/recommend``: the point of the experiment is the candidate pool the Agent fuses.

Phase 2B.3 changed three things
-------------------------------
**The request is no longer one fixed message.**  Each case's message is a *proxy query*
built deterministically from the titles of the two most recent history items
(``history[-2:]``).  ``load_cases`` builds ``history`` as ``asins[:-2]``, so those seeds are
the third- and fourth-from-last interactions: neither the test target (``asins[-1]``) nor
the validation target (``asins[-2]``) is ever read.  Before this change every case sent
``"recommend something for me"``, so ``catalog_search`` returned the same products for all
40 cases; its "only 5% of targets reach the pool" reading was a property of that input, not
a recall result.

**Every source call is accounted for.**  The driver wraps the two trusted sources with
pass-through observers that record, per call: the budget the Agent asked for, the budget the
source was actually invoked with, how many candidates it returned, the target's rank inside
that list, the latency, whether it fell short, and - for ``history`` - the Serving V1
timings and the target's inner provenance.  ``history`` is labelled ``pre_fused``: the
default serving engine already fuses three inner sources by RRF before the Agent sees it, so
an Agent source budget only decides how much of that pre-fused head is kept.

**The outcome is attributed.**  Each case lands in exactly one branch: the target was never
returned by any source, was returned but filtered out (ungrounded), was in the fused pool
but outside the final top-k, or was in the final top-k.

Instrumentation never changes what a source returns, the fusion, the ordering or the final
truncation: the wrappers forward every call unchanged and only observe it.

    .venv/bin/python -m experiments.p2b_agent_source_limit_ablation \
        --cases 40 --out runs/p2b_agent_source_limit_40b

Outputs (never written into an existing run directory):
    config.json       the frozen configuration of this execution
    cases.json        the request set (identity, history, proxy query, target)
    samples.jsonl     one record per (arm, case): per-source accounting and attribution
    summary.json      per-arm aggregates + paired deltas with bootstrap CIs
    attribution.json  the four-branch attribution, per arm and per case
    SHA256SUMS.json   SHA-256 of every artifact above
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
DEFAULT_OUT = REPO / "runs" / "p2b_agent_source_limit_40b"

#: The four arms.  ``requested_k`` is identical by construction.
ARMS: tuple[int, ...] = (5, 20, 50, 100)
REQUESTED_K = 5
#: Cases are taken well after the 1,000-case prefix P3 used, so the two sets are disjoint.
DEFAULT_SKIP = 2000
BOOTSTRAP_DRAWS = 2000
SEED = 20261024

#: How a case's message was chosen.  ``proxy`` is the 2B.3 rule; ``fixed`` reproduces 2B.2.
PROXY_QUERY_SOURCE = "history-title-proxy"
FIXED_QUERY_SOURCE = "fixed-message"
LEGACY_MESSAGE = "recommend something for me"
#: How many trailing history items seed the proxy query.  ``items[-1]``/``items[-2]`` are
#: the held-out targets and are structurally unreachable from ``history``.
PROXY_SEED_ITEMS = 2
#: Tokens taken from each seed title.  2 x 6 = 12 <= ``MAX_SEARCH_TERMS`` (16).
PROXY_TERMS_PER_TITLE = 6

#: The four exclusive outcome branches, in the order the report uses them.
BRANCHES: tuple[str, ...] = ("never_returned", "filtered", "pooled_not_top_k", "top_k")
BRANCH_DEFINITIONS: dict[str, str] = {
    "never_returned": "no source returned the target at this arm's per-source depth",
    "filtered": "some source returned it, but it never became a grounded pool candidate",
    "pooled_not_top_k": "it was in the fused pool, but its fused rank was beyond the final k",
    "top_k": "it appeared in the final ranked answer",
}

#: The same token shape the trusted lexical source indexes: alphanumeric runs, lowercased.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


# --------------------------------------------------------------------------- #
# metrics (unchanged from 2B.1/2B.2; the tests import these by name)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# request set + deterministic proxy query
# --------------------------------------------------------------------------- #


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


def title_terms(title: object, *, limit: int = PROXY_TERMS_PER_TITLE) -> tuple[str, ...]:
    """Deterministic search terms from one catalogue title.

    The token shape is the trusted lexical source's own (``[^\\W_]+``, lowercased), so the
    recorded query is what the source will actually index.  No stemming, no stop-word list,
    no model: the first ``limit`` tokens of the title, or nothing when there is no title.
    """
    if not isinstance(title, str) or not title.strip():
        return ()
    return tuple(match.group(0) for match in _WORD_RE.finditer(title.lower()))[:limit]


def build_proxy_query(
    seed_asins: Sequence[str],
    title_of: Any,
    *,
    terms_per_title: int = PROXY_TERMS_PER_TITLE,
) -> tuple[str | None, dict[str, Any]]:
    """Build one case's proxy query from the titles of its seed history items.

    Deterministic: the seed identities are consumed in the order given, each contributes at
    most ``terms_per_title`` tokens, duplicates are removed preserving first occurrence, and
    the terms are joined with single spaces.  Returns ``(query, detail)``; ``query`` is
    ``None`` when no seed has a usable title, which is reported rather than patched with a
    fabricated query.
    """
    used: list[dict[str, Any]] = []
    parts: list[str] = []
    for parent_asin in seed_asins:
        title = title_of(str(parent_asin))
        terms = title_terms(title, limit=terms_per_title)
        used.append(
            {
                "parent_asin": str(parent_asin),
                "title": title if isinstance(title, str) else None,
                "terms": list(terms),
            }
        )
        parts.extend(terms)
    seen: set[str] = set()
    deduped: list[str] = []
    for term in parts:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    query = " ".join(deduped) if deduped else None
    return query, {"seed_items": used, "terms": deduped}


def attach_proxy_queries(
    cases: list[dict[str, Any]],
    title_of: Any,
    *,
    terms_per_title: int = PROXY_TERMS_PER_TITLE,
) -> list[dict[str, Any]]:
    """Attach each case's proxy query and its provenance, in place.

    The seeds are exactly ``history[-2:]``.  ``load_cases`` builds ``history`` as
    ``asins[:-2]``, so these are the third- and fourth-from-last interactions: the test
    target (``asins[-1]``) and the validation target (``asins[-2]``) cannot be read here.
    """
    for case in cases:
        seeds = list(case["history"][-PROXY_SEED_ITEMS:])
        if case["target"] in seeds:  # pragma: no cover - structural guard, never expected
            raise AssertionError("the proxy query would read the held-out target")
        query, detail = build_proxy_query(seeds, title_of, terms_per_title=terms_per_title)
        case["proxy_seed_asins"] = seeds
        case["proxy_query"] = query
        case["query_source"] = PROXY_QUERY_SOURCE
        case["proxy_titles"] = detail["seed_items"]
        case["proxy_terms"] = detail["terms"]
    return cases


def classify_target_branch(
    *,
    appeared: bool,
    in_pool: bool,
    final_rank: int | None,
) -> str:
    """Map one case's target fate onto exactly one of :data:`BRANCHES`."""
    if final_rank is not None:
        return "top_k"
    if in_pool:
        return "pooled_not_top_k"
    if appeared:
        return "filtered"
    return "never_returned"


def _rank_of(identities: Sequence[str], target: str) -> int | None:
    """1-based position of ``target`` in ``identities``, or ``None``."""
    try:
        return list(identities).index(target) + 1
    except ValueError:
        return None


def _status(view: Any) -> str:
    """The verification status of a grounded-view candidate, as a plain string."""
    value = getattr(view, "verification_status", None)
    return str(getattr(value, "value", value))


def _replay_fusion(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Replay the run's fused order with the accepted RRF rule, for reporting only.

    This does not decide anything: it re-derives the order the controller already wrote so
    the report can name a fused rank, and the sample records whether the replay agreed with
    the order the response actually published.
    """
    from recommendation.control.candidate_ledger import reciprocal_rank_fusion

    usable = [list(ranking) for ranking in rankings if ranking]
    if not usable:
        return []
    return [identity for identity, _ in reciprocal_rank_fusion(usable)]


# --------------------------------------------------------------------------- #
# pass-through source observers
# --------------------------------------------------------------------------- #


class SourceProbe:
    """Per-case capture of what each candidate source was asked for and returned."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.calls = []

    def record(self, call: dict[str, Any]) -> None:
        self.calls.append(call)

    def of(self, source: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["source"] == source]


def instrument_history_source(tool: Any, probe: SourceProbe) -> Any:
    """Attach per-call accounting to the accepted Tool instance, in place.

    The object is **not** wrapped: ``RecommendFromHistoryCapability`` type-checks its tool
    (``isinstance(..., RecommendationTool)``), and substituting a proxy would make a legal
    deployment look unsupported.  Replacing the instance's ``run`` attribute keeps the object
    and every other attribute exactly what they were; each call is forwarded unchanged.
    """
    original = tool.run

    def observed_run(request: Any = None, context: Any = None) -> Any:
        transport_k = int(getattr(request, "k", 0) or 0)
        began = time.perf_counter()
        try:
            result = original(request=request, context=context)
        except Exception as exc:  # noqa: BLE001 - the failure is recorded, then re-raised
            probe.record(
                {
                    "source": "history",
                    "pre_fused": True,
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
        probe.record(
            {
                "source": "history",
                # The Serving V1 engine fuses three inner sources by RRF before returning;
                # the Agent's history budget only decides how much of that head is kept.
                "pre_fused": True,
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
                    "provenance": {
                        str(identity): {
                            "fused_rank": int(meta.get("fused_rank", 0) or 0),
                            "sources": [str(name) for name in meta.get("sources", ())],
                            "source_ranks": {
                                str(name): int(rank)
                                for name, rank in (meta.get("source_ranks") or {}).items()
                            },
                        }
                        for identity, meta in provenance.items()
                    },
                },
            }
        )
        return result

    tool.run = observed_run
    return tool


def instrument_catalog_search(source: Any, probe: SourceProbe) -> Any:
    """Attach per-call accounting to the trusted lexical source, in place.

    Same rule as :func:`instrument_history_source`: the instance is kept and only its
    ``candidates`` entry point is observed, so the plane still calls the accepted source.
    """
    original = source.candidates

    def observed_candidates(query: str, *, limit: int = 10) -> list[tuple[str, int, float]]:
        began = time.perf_counter()
        try:
            rows = original(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 - the failure is recorded, then re-raised
            probe.record(
                {
                    "source": "catalog_search",
                    "pre_fused": False,
                    "transport_limit": int(limit),
                    "returned": 0,
                    "ranked": [],
                    "scores": [],
                    "score_kind": "bm25",
                    "query": str(query),
                    "failed": type(exc).__name__,
                    "ms": round((time.perf_counter() - began) * 1000.0, 2),
                }
            )
            raise
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        probe.record(
            {
                "source": "catalog_search",
                "pre_fused": False,
                "transport_limit": int(limit),
                "returned": len(rows),
                "ranked": [str(row[0]) for row in rows],
                "scores": [float(row[2]) for row in rows],
                "score_kind": "bm25",
                "query": str(query),
                "failed": None,
                "ms": round(elapsed_ms, 2),
            }
        )
        return rows

    source.candidates = observed_candidates
    return source


# --------------------------------------------------------------------------- #
# one sample
# --------------------------------------------------------------------------- #


def _source_view(
    call: Mapping[str, Any] | None,
    target: str,
    *,
    requested_limit: int,
) -> dict[str, Any]:
    """The serialisable record of one source call, with the target's place in it."""
    if call is None:
        return {"called": False}
    ranked = list(call.get("ranked") or ())
    effective = int(call.get("transport_limit") or 0)
    view: dict[str, Any] = {
        "called": True,
        "pre_fused": bool(call.get("pre_fused")),
        "requested_limit": int(requested_limit),
        "effective_limit": effective,
        "capped": effective < int(requested_limit),
        "returned": int(call.get("returned") or 0),
        "rank_of_target": _rank_of(ranked, target),
        "ms": call.get("ms"),
        "failed": call.get("failed"),
        "score_kind": call.get("score_kind"),
        "query": call.get("query"),
        # Fewer candidates than asked for is the source's own exhaustion, not a truncation
        # this driver performed; a source's internal cap is recorded under ``internal``.
        "exhausted_short_of_limit": int(call.get("returned") or 0) < effective,
        "ranked": ranked,
    }
    internal = call.get("internal")
    if internal:
        view["internal"] = {
            "requested_k": internal.get("requested_k"),
            "returned_k": internal.get("returned_k"),
            "history_length": internal.get("history_length"),
            "effective_history_length": internal.get("effective_history_length"),
            "history_truncated": internal.get("history_truncated"),
            "timings_ms": internal.get("timings_ms", {}),
            "provenance_size": internal.get("provenance_size"),
            "target_provenance": (internal.get("provenance") or {}).get(target),
        }
    return view


def _target_outcome(
    *,
    target: str,
    history_view: Mapping[str, Any],
    catalog_view: Mapping[str, Any],
    grounded_views: Mapping[str, Any],
    replay: Sequence[str],
    final_order: Sequence[str],
) -> dict[str, Any]:
    """Where the target ended up, and which of the four branches that is."""
    history_rank = history_view.get("rank_of_target")
    catalog_rank = catalog_view.get("rank_of_target")
    appeared = history_rank is not None or catalog_rank is not None
    view = grounded_views.get(target)
    status = None if view is None else _status(view)
    fused_rank = _rank_of(replay, target)
    final_rank = _rank_of(final_order, target)
    in_pool = fused_rank is not None
    if appeared and not in_pool:
        filtered_reason = "ungrounded" if status == "ungrounded" else "absent_from_grounded_pool"
    else:
        filtered_reason = None
    return {
        "appeared_in_any_source": appeared,
        "rank_in_history": history_rank,
        "rank_in_catalog_search": catalog_rank,
        # Serving V1's own view: only present when the history Tool returned the target.
        "history_internal": (history_view.get("internal") or {}).get("target_provenance"),
        "grounded_status": status,
        "filtered": bool(appeared and not in_pool),
        "filtered_reason": filtered_reason,
        "in_pool": in_pool,
        "fused_rank": fused_rank,
        "final_rank": final_rank,
        "branch": classify_target_branch(
            appeared=appeared, in_pool=in_pool, final_rank=final_rank
        ),
    }


def build_sample(
    *,
    arm: int,
    case: Mapping[str, Any],
    message: str,
    response: Any,
    probe: SourceProbe,
    elapsed_ms: float,
) -> dict[str, Any]:
    """One (arm, case) record: source accounting, de-duplication, fusion and attribution."""
    from recommendation.control.candidate_ledger import RRF_K

    target = str(case["target"])
    history_calls = probe.of("history")
    catalog_calls = probe.of("catalog_search")
    history_view = _source_view(
        history_calls[0] if history_calls else None, target, requested_limit=arm
    )
    catalog_view = _source_view(
        catalog_calls[0] if catalog_calls else None, target, requested_limit=arm
    )

    grounded_views = {
        str(view.parent_asin): view for view in response.grounded.candidates
    }
    grounded_ids = {
        identity
        for identity, view in grounded_views.items()
        if _status(view) == "grounded"
    }

    history_ids = list(history_view.get("ranked") or ())
    catalog_ids = list(catalog_view.get("ranked") or ())
    combined = history_ids + catalog_ids
    unique = list(dict.fromkeys(combined))
    replay = _replay_fusion(
        [
            [identity for identity in history_ids if identity in grounded_ids],
            [identity for identity in catalog_ids if identity in grounded_ids],
        ]
    )
    grounded_order = [
        identity
        for identity, view in grounded_views.items()
        if _status(view) == "grounded"
    ]
    final_order = [str(item.parent_asin) for item in response.recommendations]
    outcome = _target_outcome(
        target=target,
        history_view=history_view,
        catalog_view=catalog_view,
        grounded_views=grounded_views,
        replay=replay,
        final_order=final_order,
    )

    return {
        "arm": arm,
        "case_index": case["case_index"],
        "user_id": case["user_id"],
        "target": target,
        "message": message,
        "query_source": case.get("query_source"),
        "status": "ok",
        "returned": len(final_order),
        "ranked": final_order,
        "pool_size": len(replay),
        "target_in_pool": bool(outcome["in_pool"]),
        "target_outcome": outcome,
        "sources": {"history": history_view, "catalog_search": catalog_view},
        "dedup": {
            "returned_total": len(combined),
            "unique": len(unique),
            "duplicates": len(combined) - len(unique),
            "overlap_history_catalog": len(set(history_ids) & set(catalog_ids)),
            "grounded_pool": len(replay),
            "grounded_count": int(response.grounded.grounded_count),
            "ungrounded_count": int(response.grounded.ungrounded_count),
        },
        "fusion": {
            "method": "reciprocal_rank_fusion",
            "rrf_k": int(RRF_K),
            "replayed_size": len(replay),
            # Two independent checks that the replay is the run's own order, not a new one.
            "matches_presented_order": replay == grounded_order,
            "matches_final_prefix": final_order == replay[: len(final_order)],
        },
        "observation": {
            "source_calls": dict(response.observation.source_calls),
            "source_outcomes": dict(response.observation.source_outcomes),
            "presented_count": int(response.observation.presented_count),
            "candidates_sufficient": bool(response.observation.candidates_sufficient),
        },
        "terminal": {
            "status": str(response.terminal.status),
            "termination_reason": response.terminal.termination_reason,
            "steps": int(response.terminal.steps),
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


def _branch_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {branch: 0 for branch in BRANCHES}
    for row in rows:
        if row["status"] != "ok":
            continue
        counts[row["target_outcome"]["branch"]] += 1
    return counts


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    latencies = sorted(float(row["ms"]) for row in ok)
    pooled = [row for row in ok if row["target_outcome"]["in_pool"]]
    fused_ranks = sorted(
        int(row["target_outcome"]["fused_rank"]) for row in pooled
        if row["target_outcome"]["fused_rank"] is not None
    )
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
        "mean_overlap": _mean([float(row["dedup"]["overlap_history_catalog"]) for row in ok]),
        "mean_duplicates": _mean([float(row["dedup"]["duplicates"]) for row in ok]),
        "target_in_history_source": _mean(
            [
                1.0 if row["target_outcome"]["rank_in_history"] is not None else 0.0
                for row in ok
            ]
        ),
        "target_in_catalog_source": _mean(
            [
                1.0 if row["target_outcome"]["rank_in_catalog_search"] is not None else 0.0
                for row in ok
            ]
        ),
        "target_in_history_internal": _mean(
            [1.0 if row["target_outcome"]["history_internal"] else 0.0 for row in ok]
        ),
        "pooled_cases": len(pooled),
        "median_fused_rank_when_pooled": (
            round(statistics.median(fused_ranks), 1) if fused_ranks else None
        ),
        "fused_ranks_when_pooled": fused_ranks,
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "mean_history_ms": _mean(
            [float(row["sources"]["history"].get("ms") or 0.0) for row in ok]
        ),
        "mean_catalog_ms": _mean(
            [float(row["sources"]["catalog_search"].get("ms") or 0.0) for row in ok]
        ),
        "branches": _branch_counts(rows),
    }


def _paired(
    records: Sequence[Mapping[str, Any]], arms: Sequence[int]
) -> dict[str, dict[str, Any]]:
    by_arm: dict[int, dict[int, Mapping[str, Any]]] = {}
    for row in records:
        if row["status"] != "ok":
            continue
        by_arm.setdefault(int(row["arm"]), {})[int(row["case_index"])] = row
    extractors = {
        "hr@5": lambda row: float(row["hr@5"]),
        "ndcg@5": lambda row: float(row["ndcg@5"]),
        "pool_coverage": lambda row: 1.0 if row["target_in_pool"] else 0.0,
    }
    baseline = by_arm.get(arms[0], {})
    paired: dict[str, dict[str, Any]] = {}
    for arm in arms[1:]:
        shared = sorted(set(baseline) & set(by_arm.get(arm, {})))
        for metric, extract in extractors.items():
            diffs = [extract(by_arm[arm][index]) - extract(baseline[index]) for index in shared]
            paired[f"{arm}_minus_{arms[0]}_{metric}"] = {
                "delta": round(sum(diffs) / len(diffs), 4) if diffs else 0.0,
                "ci95": paired_ci(diffs),
                "n": len(diffs),
            }
    return paired


def _attribution(
    records: Sequence[Mapping[str, Any]], arms: Sequence[int]
) -> dict[str, Any]:
    by_arm = {str(arm): {branch: 0 for branch in BRANCHES} for arm in arms}
    examples: dict[str, dict[str, list[int]]] = {
        str(arm): {branch: [] for branch in BRANCHES} for arm in arms
    }
    per_case: dict[int, dict[str, Any]] = {}
    for row in records:
        arm = str(row["arm"])
        if row["status"] != "ok":
            by_arm[arm]["error"] = by_arm[arm].get("error", 0) + 1
            continue
        branch = row["target_outcome"]["branch"]
        by_arm[arm][branch] += 1
        examples[arm][branch].append(int(row["case_index"]))
        per_case.setdefault(int(row["case_index"]), {"case_index": row["case_index"]})[arm] = branch
    return {
        "branches": list(BRANCHES),
        "definitions": dict(BRANCH_DEFINITIONS),
        "by_arm": by_arm,
        "examples": examples,
        "per_case": [per_case[index] for index in sorted(per_case)],
    }


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def _manifest(out: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name in ("config.json", "cases.json", "samples.jsonl", "summary.json",
                 "attribution.json"):
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
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--message-mode",
        choices=("proxy", "fixed"),
        default="proxy",
        help="'proxy' is the 2B.3 per-case query; 'fixed' reproduces the 2B.2 message.",
    )
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
    for case in cases:
        case["usable"] = args.message_mode == "fixed" or case["proxy_query"] is not None
        case["excluded_reason"] = None if case["usable"] else "no_history_title"
    run_cases = [case for case in cases if case["usable"]]
    if not run_cases:
        raise SystemExit("no case has a usable proxy query")

    engine = getattr(runtime, "engine", None)
    engine_config = getattr(engine, "config", None)
    config = {
        "arms": list(ARMS),
        "requested_k": REQUESTED_K,
        "sources": ["history", "catalog_search"],
        "cases_requested": len(cases),
        "cases_usable": len(run_cases),
        "excluded_cases": [c["case_index"] for c in cases if not c["usable"]],
        "skip": args.skip,
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "message_mode": args.message_mode,
        "query_source": PROXY_QUERY_SOURCE if args.message_mode == "proxy" else FIXED_QUERY_SOURCE,
        "legacy_message": LEGACY_MESSAGE if args.message_mode == "fixed" else None,
        "proxy_query": {
            "seed": "history[-2:]",
            "seed_offsets_from_sequence_end": [-3, -4],
            "terms_per_title": PROXY_TERMS_PER_TITLE,
            "max_terms": PROXY_SEED_ITEMS * PROXY_TERMS_PER_TITLE,
            "tokenizer": r"[^\W_]+ lowercased",
            "held_out_items_read": [],
        },
        "source_limit_env": ENV_SOURCE_LIMIT,
        "source_limit_initial": os.environ.get(ENV_SOURCE_LIMIT),
        "sequence_artifact": str(SEQUENCES.relative_to(REPO)),
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
        "python": sys.version.split()[0],
        "git_head": _git_head(),
        "driver_path": str(Path(__file__).resolve().relative_to(REPO)),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "agentrecx_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("AGENTRECX_") and key != ENV_SOURCE_LIMIT
        },
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cases.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8"
    )
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # A warm request keeps first-call allocation out of the measured latency.  It runs
    # before the probes are attached, so the warm-up never enters a sample.
    service.run(
        AgentRecommendRequest(
            message=LEGACY_MESSAGE,
            history=run_cases[0]["history"][-5:],
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
        for arm in ARMS:
            os.environ[ENV_SOURCE_LIMIT] = str(arm)
            arm_rows: list[dict[str, Any]] = []
            for case in run_cases:
                message = LEGACY_MESSAGE if args.message_mode == "fixed" else case["proxy_query"]
                payload = AgentRecommendRequest(
                    message=message,
                    history=case["history"],
                    k=REQUESTED_K,
                    sources=("history", "catalog_search"),
                )
                probe.reset()
                began = time.perf_counter()
                try:
                    response = service.run(payload)
                    elapsed_ms = (time.perf_counter() - began) * 1000.0
                    row = build_sample(
                        arm=arm,
                        case=case,
                        message=message,
                        response=response,
                        probe=probe,
                        elapsed_ms=elapsed_ms,
                    )
                except Exception as exc:  # noqa: BLE001 - a failed case is recorded, never dropped
                    elapsed_ms = (time.perf_counter() - began) * 1000.0
                    row = {
                        "arm": arm,
                        "case_index": case["case_index"],
                        "user_id": case["user_id"],
                        "target": case["target"],
                        "message": message,
                        "query_source": case.get("query_source"),
                        "status": f"error:{type(exc).__name__}",
                        "returned": 0,
                        "ranked": [],
                        "pool_size": 0,
                        "target_in_pool": False,
                        "hr@5": 0.0,
                        "ndcg@5": 0.0,
                        "ms": round(elapsed_ms, 2),
                        "sources": {
                            source: [call for call in probe.of(source)]
                            for source in ("history", "catalog_search")
                        },
                        "error": str(exc),
                    }
                arm_rows.append(row)
                records_out.append(row)
            aggregates[str(arm)] = _aggregate(arm_rows)
            print(json.dumps({"arm": arm, "summary": aggregates[str(arm)]}), flush=True)
    finally:
        if previous_limit is None:
            os.environ.pop(ENV_SOURCE_LIMIT, None)
        else:
            os.environ[ENV_SOURCE_LIMIT] = previous_limit

    paired = _paired(records_out, ARMS)
    attribution = _attribution(records_out, ARMS)
    summary = {
        "arms": aggregates,
        "paired_vs_baseline": paired,
        "attribution": {"by_arm": attribution["by_arm"]},
        "cases_usable": len(run_cases),
        "message_mode": args.message_mode,
    }

    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records_out), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "attribution.json").write_text(
        json.dumps(attribution, indent=2), encoding="utf-8"
    )
    (args.out / "SHA256SUMS.json").write_text(
        json.dumps(_manifest(args.out), indent=2), encoding="utf-8"
    )
    print(json.dumps({"arms": aggregates, "paired": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
