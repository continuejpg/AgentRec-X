"""P8-D multi-user Source-Policy dataset builder and reachability/learnability analysis.

This extends the canonical P6 Dev cohort (the published 160 cases are only the first
160 eligible Dev users) with additional users from the same user_id hash split.  It uses
the existing temporal leave-two-out validation position, never the test position, and
never the frozen Holdout users.

For every case it runs the real Serving V1 history tool and real Hybrid BM25+BGE catalog
search once, caches the two immutable candidate rankings, then replays the three accepted
Source Policies through the exact same weighted RRF function used by the Agent.

The result is a reusable development dataset, not a training set.  Validation targets are
relevance labels for next-item evaluation only; they are never used to build queries or
features visible to the Planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/p8_d/multiuser"
INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
MODEL = ROOT / "models/bge-base-en-v1.5"
MAPPING = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
SEQUENCES = ROOT / "data/processed/Sports_and_Outdoors_sequences.json"
CANONICAL = ROOT / "data/p6_benchmark_v0/classical_dev.jsonl"
P8C_CACHE = ROOT / "runs/p8_c/source_policy_eval/candidate_cache.jsonl"
P8C_PER_USER = ROOT / "runs/p8_c/source_policy_eval/per_user.jsonl"

os.environ["AGENTRECX_CATALOG_SEARCH_MODE"] = "hybrid"
os.environ["AGENTRECX_DENSE_INDEX"] = str(INDEX)
os.environ["AGENTRECX_BGE_MODEL_PATH"] = str(MODEL)
os.environ["AGENTRECX_MAPPINGS_PATH"] = str(MAPPING)
os.environ["AGENTRECX_CATALOG_METADATA_PATH"] = str(PRODUCTS)
os.environ["AGENTRECX_SERVING_FUSION"] = "rrf3"
os.environ["AGENTRECX_STATE_MODE"] = "off"
os.environ["AGENTRECX_STATE_DB"] = str(OUT / "state.sqlite3")
os.environ["AGENTRECX_MEMORY_DB"] = str(OUT / "memory.sqlite3")

from recommendation.api.app import ServiceSettings, build_engine
from recommendation.control.candidate_ledger import (
    SOURCE_POLICIES,
    weighted_reciprocal_rank_fusion,
)
from recommendation.demo.product_role import assess_candidate, infer_target
from recommendation.demo.runtime import build_demo_runtime
from recommendation.tools.recommendation import RecommendationTool, RecommendationToolRequest
from recommendation.tools.schemas import RecommendationContext

POLICIES = list(SOURCE_POLICIES)


def split_for_user(user_id: str) -> str:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return "holdout" if int(digest, 16) % 5 == 0 else "dev"


def _valid_sequence(record: dict[str, Any]) -> tuple[list[str], list[int], list[int]] | None:
    items = list(record.get("item_ids") or ())
    asins = list(record.get("parent_asins") or ())
    times = list(record.get("unix_ms") or ())
    if len(items) < 3 or len(asins) != len(items) or len(times) != len(items):
        return None
    if 0 in items or any(not isinstance(item, int) for item in items):
        return None
    if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
        return None
    if any(not isinstance(asin, str) or not asin for asin in asins):
        return None
    return asins, items, times


def build_cases(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_rows = [
        row
        for row in (
            json.loads(line)
            for line in CANONICAL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("split") == "dev"
    ]
    cases: list[dict[str, Any]] = []
    users: set[str] = set()
    for row in canonical_rows:
        history = list(row.get("history_parent_asins") or ())
        target = str(row.get("validation_parent_asin") or "")
        if not history or not target or target in history:
            continue
        cases.append(
            {
                "case_id": str(row["case_id"]),
                "user_id": str(row["user_id"]),
                "source": "p6_canonical_dev",
                "is_canonical": True,
                "history": history,
                "target": target,
                "validation_time_ms": row.get("validation_time_ms"),
            }
        )
        users.add(str(row["user_id"]))
    canonical_count = len(cases)

    payload = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    scanned = 0
    for record in payload.get("sequences") or ():
        if len(cases) >= limit:
            break
        scanned += 1
        user_id = str(record.get("user_id") or "")
        if not user_id or user_id in users:
            continue
        if split_for_user(user_id) != "dev":
            continue
        valid = _valid_sequence(record)
        if valid is None:
            continue
        asins, items, times = valid
        history = asins[:-2]
        target = asins[-2]
        if not history or not target or target in history:
            continue
        case_id = f"p8d-dev-ext-{len(cases):05d}"
        cases.append(
            {
                "case_id": case_id,
                "user_id": user_id,
                "source": "p6_dev_extended_same_temporal_protocol",
                "is_canonical": False,
                "history": history,
                "target": target,
                "validation_time_ms": times[-2],
            }
        )
        users.add(user_id)
    return cases, {
        "canonical_cases": canonical_count,
        "requested_limit": limit,
        "actual_cases": len(cases),
        "sequences_scanned": scanned,
        "sequence_artifact": str(SEQUENCES.relative_to(ROOT)),
        "canonical_artifact": str(CANONICAL.relative_to(ROOT)),
        "split_policy": "sha256(user_id) % 5 == 0 is frozen holdout; never selected",
        "target_position": "items[-2] validation; items[-1] test is not loaded into the dataset",
        "target_in_history_excluded": True,
    }


def load_reuse_cache() -> dict[str, dict[str, Any]]:
    reused: dict[str, dict[str, Any]] = {}
    if P8C_CACHE.exists():
        for line in P8C_CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                reused[str(row["case_id"])] = row
    return reused


def load_p8c_results() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if P8C_PER_USER.exists():
        for line in P8C_PER_USER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["case_id"])] = row
    return rows


def build_query(history: list[str], records) -> tuple[str, str]:
    parts: list[str] = []
    for identity in reversed(history[-2:]):
        record = records.get(identity)
        if record is None:
            continue
        categories = [
            str(value)
            for value in (getattr(record, "categories", ()) or ())
            if str(value).casefold()
            not in {"sports & outdoors", "sports", "outdoors"}
        ]
        if categories:
            parts.extend(categories[-2:])
        else:
            title = str(getattr(record, "title", "") or "").strip()
            if title:
                parts.append(title)
    seen: set[str] = set()
    terms: list[str] = []
    for part in parts:
        folded = " ".join(part.casefold().split())
        if not folded or folded in seen:
            continue
        seen.add(folded)
        terms.append(folded)
    query = " ".join(terms).strip()[:200]
    return (query or "camping gear"), ("last_history_categories" if query else "fallback")


def history_candidates(tool: RecommendationTool, history: list[str], limit: int) -> list[dict]:
    result = tool.run(
        RecommendationToolRequest(k=limit),
        RecommendationContext(user_history=tuple(history)),
    )
    return [
        {
            "parent_asin": item.parent_asin,
            "rank": int(item.rank),
            "score": float(item.score) if item.score is not None else None,
            "score_kind": item.score_kind,
        }
        for item in result.recommendations
    ]


def catalog_candidates(catalog, query: str, limit: int) -> list[dict]:
    result = catalog.search(query, limit=limit)
    return [
        {
            "parent_asin": hit.parent_asin,
            "rank": pos,
            "score": float(hit.score),
            "score_kind": catalog.score_kind,
            "matched_fields": list(hit.matched_fields),
        }
        for pos, hit in enumerate(result.hits, start=1)
    ]


def _rank_metrics(ranked: list[tuple[str, float]], target: str, k: int = 10) -> dict:
    top = [identity for identity, _ in ranked[:k]]
    full = [identity for identity, _ in ranked]
    rank_top = None
    rank_full = None
    for pos, identity in enumerate(top, start=1):
        if identity == target:
            rank_top = pos
            break
    for pos, identity in enumerate(full, start=1):
        if identity == target:
            rank_full = pos
            break
    return {
        "top10": top,
        "target_rank_top10": rank_top,
        "target_rank_full": rank_full,
        "recall@10": 1.0 if rank_top is not None else 0.0,
        "ndcg@10": (1.0 / math.log2(rank_top + 1)) if rank_top else 0.0,
        "mrr@10": (1.0 / rank_top) if rank_top else 0.0,
    }


def history_category_features(history: list[str], records) -> dict:
    categories: list[str] = []
    titles: list[str] = []
    for identity in history:
        record = records.get(identity)
        if record is None:
            continue
        categories.extend(str(value) for value in (getattr(record, "categories", ()) or ()))
        titles.append(str(getattr(record, "title", "") or ""))
    counts = Counter(categories)
    total = sum(counts.values())
    entropy = 0.0
    if total:
        for value in counts.values():
            p = value / total
            entropy -= p * math.log2(p)
    return {
        "history_len": len(history),
        "history_category_count": len(counts),
        "history_category_entropy": round(entropy, 6),
        "history_title_length": sum(len(title) for title in titles),
    }


def _feature_row(case: dict, rows: dict, routed: dict, records, role) -> dict:
    history_feat = history_category_features(case["history"], records)
    history_rows = rows.get("history_rows") or []
    catalog_rows = rows.get("catalog_rows") or []
    overlap = len(
        {str(row["parent_asin"]) for row in history_rows}
        & {str(row["parent_asin"]) for row in catalog_rows}
    )
    query = str(rows.get("query") or routed.get("query") or "")
    union_ids = {str(row["parent_asin"]) for row in history_rows + catalog_rows}
    role_match = 0
    role_mismatch = 0
    role_unknown = 0
    for identity in union_ids:
        record = records.get(identity)
        if record is None:
            continue
        status = assess_candidate(identity, record, role).status.value
        if status == "MATCH":
            role_match += 1
        elif status == "MISMATCH":
            role_mismatch += 1
        else:
            role_unknown += 1
    return {
        "pre_retrieval": {
            **history_feat,
            "query_length": len(query),
            "query_token_count": len(query.split()),
            "query_role_type": role.type,
            "query_role": role.role.value,
            "query_has_price_constraint": any(
                token in query.casefold() for token in ("under $", "below $", "price")
            ),
        },
        "post_retrieval": {
            "history_candidate_count": len(history_rows),
            "catalog_candidate_count": len(catalog_rows),
            "union_candidate_count": len(union_ids),
            "overlap_count": overlap,
            "overlap_rate": overlap / max(1, len(union_ids)),
            "role_match_count": role_match,
            "role_mismatch_count": role_mismatch,
            "role_unknown_count": role_unknown,
            "catalog_bm25_bge_overlap": sum(
                1
                for row in catalog_rows
                if len(row.get("matched_fields") or ()) >= 2
            ),
        },
    }


def _policy_replay(history_rows: list[dict], catalog_rows: list[dict], target: str) -> dict:
    rankings = {
        "history": [str(row["parent_asin"]) for row in history_rows],
        "catalog_search": [str(row["parent_asin"]) for row in catalog_rows],
    }
    results = {}
    for policy in POLICIES:
        fused = weighted_reciprocal_rank_fusion(rankings, source_policy=policy)
        results[policy] = _rank_metrics(fused, target)
    scores = {policy: results[policy]["ndcg@10"] for policy in POLICIES}
    max_score = max(scores.values())
    winners = [policy for policy in POLICIES if scores[policy] == max_score]
    return {
        "policies": results,
        "best_ndcg": max_score,
        "best_policies": winners,
        "unique_best": winners[0] if len(winners) == 1 and max_score > 0 else None,
        "strict_tie": len(winners) > 1,
        "all_zero": max_score == 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--history-k", type=int, default=50)
    parser.add_argument("--catalog-k", type=int, default=50)
    parser.add_argument("--reuse-cache", action="store_true", default=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases, case_meta = build_cases(args.limit)
    reuse = load_reuse_cache() if args.reuse_cache else {}
    p8c = load_p8c_results()

    settings = ServiceSettings.from_env()
    engine, error = build_engine(settings)
    if engine is None:
        raise SystemExit(f"engine unavailable: {error}")
    runtime = build_demo_runtime(settings=settings, engine=engine)
    service = runtime.agent_service()
    catalog = service._catalog_search
    records = runtime.metadata.records
    tool = runtime.tool
    if catalog.mode != "hybrid":
        raise SystemExit(f"expected hybrid mode, got {catalog.mode}")

    candidate_cache: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for pos, case in enumerate(cases, start=1):
        cache = reuse.get(case["case_id"])
        if cache is not None and str(cache.get("target")) == str(case["target"]):
            rows = {
                "history_rows": list(cache.get("history_rows") or ()),
                "catalog_rows": list(cache.get("catalog_rows") or ()),
                "query": str(cache.get("query") or ""),
                "query_source": str(cache.get("query_source") or ""),
                "cache_reused_from": "runs/p8_c/source_policy_eval/candidate_cache.jsonl",
            }
            cache_reused = True
        else:
            query, query_source = build_query(case["history"], records)
            try:
                hist_rows = history_candidates(tool, case["history"], args.history_k)
            except Exception as exc:  # noqa: BLE001 - retain data failure reason
                policy_records.append(
                    {
                        "case_id": case["case_id"],
                        "user_id": case["user_id"],
                        "status": "skipped",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            began = time.perf_counter()
            cat_rows = catalog_candidates(catalog, query, args.catalog_k)
            query_seconds = time.perf_counter() - began
            rows = {
                "history_rows": hist_rows,
                "catalog_rows": cat_rows,
                "query": query,
                "query_source": query_source,
                "cache_reused_from": None,
                "catalog_seconds": round(query_seconds, 4),
            }
            cache_reused = False
        cache_row = {
            "schema_version": "agentrecx.p8d.candidate_cache.v1",
            "case_id": case["case_id"],
            "user_id": case["user_id"],
            "source": case["source"],
            "is_canonical": case["is_canonical"],
            "history": case["history"],
            "target": case["target"],
            "query": rows["query"],
            "query_source": rows["query_source"],
            "history_rows": rows["history_rows"],
            "catalog_rows": rows["catalog_rows"],
            "source_calls": {"history": 1, "catalog_search": 1},
            "catalog_score_kind": catalog.score_kind,
            "cache_reused": cache_reused,
            "cache_reused_from": rows["cache_reused_from"],
            "catalog_seconds": rows.get("catalog_seconds"),
            "history_k": args.history_k,
            "catalog_k": args.catalog_k,
        }
        candidate_cache.append(cache_row)
        history_ids = {str(row["parent_asin"]) for row in rows["history_rows"]}
        catalog_ids = {str(row["parent_asin"]) for row in rows["catalog_rows"]}
        reachable_by_source = (
            "both"
            if case["target"] in history_ids and case["target"] in catalog_ids
            else "history_only"
            if case["target"] in history_ids
            else "catalog_only"
            if case["target"] in catalog_ids
            else "none"
        )
        replay = _policy_replay(rows["history_rows"], rows["catalog_rows"], case["target"])
        role = infer_target(rows["query"])
        features = _feature_row(case, rows, rows, records, role)
        policy_records.append(
            {
                "schema_version": "agentrecx.p8d.policy_result.v1",
                "case_id": case["case_id"],
                "user_id": case["user_id"],
                "source": case["source"],
                "is_canonical": case["is_canonical"],
                "status": "evaluated",
                "history_len": len(case["history"]),
                "query": rows["query"],
                "query_source": rows["query_source"],
                "target": case["target"],
                "reachable_by_source": reachable_by_source,
                "target_in_history_pool": case["target"] in history_ids,
                "target_in_catalog_pool": case["target"] in catalog_ids,
                "target_in_union": reachable_by_source != "none",
                "history_candidate_count": len(rows["history_rows"]),
                "catalog_candidate_count": len(rows["catalog_rows"]),
                "union_candidate_count": len(history_ids | catalog_ids),
                "overlap_count": len(history_ids & catalog_ids),
                "catalog_seconds": rows.get("catalog_seconds"),
                "features": features,
                "replay": replay,
                "policies": replay["policies"],
            }
        )
        if pos % 25 == 0:
            print(
                f"{pos}/{len(cases)} {case['case_id']} reachable={reachable_by_source} "
                f"best={replay['unique_best']}",
                flush=True,
            )

    evaluated = [row for row in policy_records if row.get("status") == "evaluated"]
    reachable = [row for row in evaluated if row["target_in_union"]]
    summary: dict[str, Any] = {
        "schema_version": "agentrecx.p8d.multiuser_policy_dataset.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_kind": "next_item_relevance_proxy",
        "not_search_satisfaction": True,
        "target_label": "validation_parent_asin (items[-2]); items[-1]/frozen Holdout unused",
        "query_source": "last two history items' categories/title only; target never used",
        "config": {
            "history_k": args.history_k,
            "catalog_k": args.catalog_k,
            "catalog_mode": catalog.mode,
            "catalog_score_kind": catalog.score_kind,
            "bge_index": str(INDEX.relative_to(ROOT)),
            "bge_model": str(MODEL.relative_to(ROOT)),
            "policies": POLICIES,
            "source_policy_formula": "sum(weight_s / (60 + rank_s))",
            "case_meta": case_meta,
        },
        "seconds": round(time.perf_counter() - started, 2),
        "cases": len(evaluated),
        "skipped": len(policy_records) - len(evaluated),
        "canonical_cases": sum(1 for row in evaluated if row["is_canonical"]),
        "extended_cases": sum(1 for row in evaluated if not row["is_canonical"]),
        "cache_reused_cases": sum(1 for row in candidate_cache if row["cache_reused"]),
        "target_reachability": {
            "history_pool": sum(row["target_in_history_pool"] for row in evaluated),
            "catalog_pool": sum(row["target_in_catalog_pool"] for row in evaluated),
            "union_pool": len(reachable),
            "by_source": dict(Counter(row["reachable_by_source"] for row in reachable)),
        },
        "source_overlap": {
            "history_candidates_mean": (
                sum(row["history_candidate_count"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "catalog_candidates_mean": (
                sum(row["catalog_candidate_count"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "overlap_mean": (
                sum(row["overlap_count"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "union_mean": (
                sum(row["union_candidate_count"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
        },
        "metrics_all": {},
        "metrics_reachable": {},
        "strict_outcomes": {},
        "rank_change": {},
        "by_history_length": {},
        "oracle": {},
        "learnability": {},
    }
    for policy in POLICIES:
        summary["metrics_all"][policy] = {
            key: (sum(row["replay"]["policies"][policy][key] for row in evaluated) / len(evaluated)
                  if evaluated else 0.0)
            for key in ("recall@10", "ndcg@10", "mrr@10")
        }
        summary["metrics_reachable"][policy] = {
            key: (sum(row["replay"]["policies"][policy][key] for row in reachable) / len(reachable)
                  if reachable else 0.0)
            for key in ("recall@10", "ndcg@10", "mrr@10")
        }
        summary["rank_change"][policy] = {
            "order_changed_rate": (
                sum(
                    row["replay"]["policies"][policy]["top10"]
                    != row["replay"]["policies"]["balanced"]["top10"]
                    for row in evaluated
                ) / len(evaluated) if evaluated else 0.0
            ),
            "set_changed_rate": (
                sum(
                    set(row["replay"]["policies"][policy]["top10"])
                    != set(row["replay"]["policies"]["balanced"]["top10"])
                    for row in evaluated
                ) / len(evaluated) if evaluated else 0.0
            ),
        }
    strict_wins = Counter()
    strict_counts = {"strict_unique_win": 0, "tie": 0, "unreachable": len(evaluated) - len(reachable)}
    reachable_outside_top10 = 0
    for row in reachable:
        replay = row["replay"]
        if replay["unique_best"] is not None:
            strict_counts["strict_unique_win"] += 1
            strict_wins[replay["unique_best"]] += 1
        elif replay["strict_tie"]:
            strict_counts["tie"] += 1
        if all(row["policies"][policy]["target_rank_top10"] is None for policy in POLICIES):
            reachable_outside_top10 += 1
    summary["strict_outcomes"] = {
        **strict_counts,
        "strict_unique_win_by_policy": dict(strict_wins),
        "reachable_but_all_policies_outside_top10": reachable_outside_top10,
        "reachable_count": len(reachable),
        "note": "strict win is unique best NDCG@10 and >0 on a target-reachable sample",
    }
    bucket = defaultdict(list)
    for row in evaluated:
        length = row["history_len"]
        key = "0-3" if length <= 3 else "4-7" if length <= 7 else "8+"
        bucket[key].append(row)
    for key, rows in sorted(bucket.items()):
        summary["by_history_length"][key] = {
            "n": len(rows),
            "reachable": sum(1 for row in rows if row["target_in_union"]),
            "metrics": {
                policy: {
                    metric: sum(row["policies"][policy][metric] for row in rows) / len(rows)
                    for metric in ("recall@10", "ndcg@10", "mrr@10")
                }
                for policy in POLICIES
            },
        }
    best_counts = Counter()
    for row in reachable:
        if row["replay"]["unique_best"] is not None:
            best_counts[row["replay"]["unique_best"]] += 1
    summary["oracle"] = {
        "label": "validation-target oracle; ANALYSIS ONLY, not a Planner-achievable score",
        "reachable_n": len(reachable),
        "unique_best_counts": dict(best_counts),
        "tie_count": strict_counts["tie"],
    }

    # Learnability analysis: only target-reachable, unique-best samples.
    learn_rows = [
        row for row in reachable if row["replay"]["unique_best"] is not None
    ]
    feature_names = [
        "history_len",
        "history_category_count",
        "history_category_entropy",
        "query_length",
        "query_token_count",
    ]
    summary["learnability"] = {
        "sample_n": len(learn_rows),
        "feature_source": "pre-retrieval only; no target, no candidate outcomes",
        "feature_names": feature_names,
        "label": "unique best policy on NDCG@10 among target-reachable samples",
        "status": "insufficient",
    }
    if len(learn_rows) >= 8 and len({row["replay"]["unique_best"] for row in learn_rows}) >= 2:
        try:
            import numpy as np
            from sklearn.model_selection import cross_val_score
            from sklearn.tree import DecisionTreeClassifier

            labels = [row["replay"]["unique_best"] for row in learn_rows]
            x = np.asarray(
                [
                    [
                        float(row["features"]["pre_retrieval"][name])
                        for name in feature_names
                    ]
                    for row in learn_rows
                ],
                dtype=float,
            )
            counts = Counter(labels)
            majority = max(counts.values()) / len(labels)
            clf = DecisionTreeClassifier(max_depth=3, random_state=0)
            folds = min(5, len(labels))
            scores = cross_val_score(clf, x, labels, cv=folds)
            summary["learnability"].update(
                {
                    "status": "computed_but_small_sample",
                    "label_counts": dict(counts),
                    "majority_baseline_accuracy": round(majority, 6),
                    "decision_tree_cv_accuracy": round(float(scores.mean()), 6),
                    "decision_tree_cv_folds": folds,
                    "decision_tree_cv_std": round(float(scores.std()), 6),
                    "features": [row["features"]["pre_retrieval"] for row in learn_rows],
                }
            )
        except Exception as exc:  # noqa: BLE001 - report, do not fabricate learnability
            summary["learnability"]["error"] = f"{type(exc).__name__}: {exc}"

    (OUT / "cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )
    (OUT / "candidate_cache.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_cache),
        encoding="utf-8",
    )
    (OUT / "policy_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in policy_records),
        encoding="utf-8",
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
