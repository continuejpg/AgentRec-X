"""P8-C multi-user source-policy replay on real Dev histories.

For every sampled P6 classical Dev user:
  1. run the real Serving V1 history tool once with train_history;
  2. run the real Hybrid BM25+BGE catalog search once with a query built only from
     prior history metadata;
  3. cache both raw candidate rankings;
  4. replay balanced / intent_focus / history_focus over the same immutable snapshot.

This is explicitly a next-item relevance proxy, not a human search-satisfaction study.
The validation target is never used to build the query.  The frozen Holdout is untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/p8_c/source_policy_eval"
INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
MODEL = ROOT / "models/bge-base-en-v1.5"
MAPPING = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
CLASSICAL_DEV = ROOT / "data/p6_benchmark_v0/classical_dev.jsonl"

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
from recommendation.demo.runtime import build_demo_runtime
from recommendation.demo.product_role import TargetRole, assess_candidate, infer_target
from recommendation.tools.recommendation import RecommendationTool, RecommendationToolRequest
from recommendation.tools.schemas import RecommendationContext


def _load_cases(limit: int) -> list[dict]:
    cases = []
    with CLASSICAL_DEV.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            history = [str(item) for item in row.get("history_parent_asins") or ()]
            target = str(row.get("validation_parent_asin") or "")
            if not history or not target:
                continue
            cases.append(
                {
                    "case_id": row.get("case_id"),
                    "user_id": row.get("user_id"),
                    "history": history,
                    "target": target,
                    "history_len": len(history),
                }
            )
            if len(cases) >= limit:
                break
    return cases


def _build_query(history: list[str], records) -> tuple[str, str]:
    """Build a query only from the user's prior history metadata."""
    parts: list[str] = []
    for identity in reversed(history[-2:]):
        record = records.get(identity)
        if record is None:
            continue
        categories = [
            str(value)
            for value in (getattr(record, "categories", ()) or ())
            if str(value).casefold()
            not in {"sports & outdoors", "sports", "outdoors", "sports & outdoors"}
        ]
        if categories:
            parts.extend(categories[-2:])
        else:
            title = str(getattr(record, "title", "") or "").strip()
            if title:
                parts.append(title)
    # Stable, deduplicated, and bounded; this is a proxy query, not a label.
    seen: set[str] = set()
    terms: list[str] = []
    for part in parts:
        folded = " ".join(part.casefold().split())
        if not folded or folded in seen:
            continue
        seen.add(folded)
        terms.append(folded)
    query = " ".join(terms).strip()[:200]
    source = "last_history_categories" if query else "fallback"
    return (query or "camping gear"), source


def _history_candidates(tool: RecommendationTool, history: list[str], limit: int) -> list[dict]:
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


def _hit(candidates: list[dict], target: str, k: int = 10) -> tuple[int | None, float, float]:
    for item in candidates[:k]:
        if item["parent_asin"] == target:
            rank = int(item["rank"])
            return rank, 1.0 / rank, 1.0 / math.log2(rank + 1)
    return None, 0.0, 0.0


def _topk(ranked: list[tuple[str, float]], k: int = 10) -> list[str]:
    return [identity for identity, _ in ranked[:k]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--history-k", type=int, default=50)
    parser.add_argument("--catalog-k", type=int, default=50)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

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
        raise SystemExit(f"expected hybrid catalog mode, got {catalog.mode}")

    cases = _load_cases(args.limit)
    per_user = []
    cache_rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        history = case["history"]
        target = case["target"]
        query, query_source = _build_query(history, records)
        try:
            history_rows = _history_candidates(tool, history, args.history_k)
        except Exception as exc:  # noqa: BLE001 - skip unusable sample, keep provenance
            per_user.append(
                {
                    "case_id": case["case_id"],
                    "user_id": case["user_id"],
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        began = time.perf_counter()
        catalog_result = catalog.search(query, limit=args.catalog_k)
        catalog_seconds = time.perf_counter() - began
        catalog_rows = [
            {
                "parent_asin": hit.parent_asin,
                "rank": int(pos),
                "score": float(hit.score),
                "score_kind": catalog.score_kind,
                "matched_fields": list(hit.matched_fields),
            }
            for pos, hit in enumerate(catalog_result.hits, start=1)
        ]
        history_by_id = {row["parent_asin"]: row for row in history_rows}
        catalog_by_id = {row["parent_asin"]: row for row in catalog_rows}
        cache_rows.append(
            {
                "case_id": case["case_id"],
                "user_id": case["user_id"],
                "history": history,
                "target": target,
                "query": query,
                "query_source": query_source,
                "history_rows": history_rows,
                "catalog_rows": catalog_rows,
                "source_calls": {"history": 1, "catalog_search": 1},
                "catalog_score_kind": catalog.score_kind,
                "catalog_seconds": round(catalog_seconds, 4),
            }
        )
        policies = {}
        target_rank = {}
        for policy in SOURCE_POLICIES:
            fused = weighted_reciprocal_rank_fusion(
                {
                    "history": [row["parent_asin"] for row in history_rows],
                    "catalog_search": [row["parent_asin"] for row in catalog_rows],
                },
                source_policy=policy,
            )
            top = _topk(fused, 10)
            rank, rr, ndcg = _hit(
                [{"parent_asin": identity, "rank": pos} for pos, identity in enumerate(top, 1)],
                target,
            )
            policies[policy] = {
                "top10": top,
                "target_rank": rank,
                "recall@10": 1.0 if rank is not None else 0.0,
                "mrr@10": rr,
                "ndcg@10": ndcg,
                "fused_scores": {identity: score for identity, score in fused[:50]},
            }
            target_rank[policy] = rank
        balanced_top = policies["balanced"]["top10"]
        changes = {
            policy: {
                "order_changed": policies[policy]["top10"] != balanced_top,
                "set_changed": set(policies[policy]["top10"]) != set(balanced_top),
                "jaccard_vs_balanced": len(
                    set(policies[policy]["top10"]) & set(balanced_top)
                )
                / max(1, len(set(policies[policy]["top10"]) | set(balanced_top))),
            }
            for policy in SOURCE_POLICIES
        }
        role_target = infer_target(query)
        top10_union = set().union(*(set(policies[p]["top10"]) for p in SOURCE_POLICIES))
        role_status = Counter()
        for identity in top10_union:
            record = records.get(identity)
            if record is None:
                continue
            role_status[assess_candidate(identity, record, role_target).status.value] += 1
        per_user.append(
            {
                "case_id": case["case_id"],
                "user_id": case["user_id"],
                "status": "evaluated",
                "history_len": len(history),
                "query": query,
                "query_source": query_source,
                "target": target,
                "target_in_history_pool": target in {r["parent_asin"] for r in history_rows},
                "target_in_catalog_pool": target in {r["parent_asin"] for r in catalog_rows},
                "target_in_union": target in (
                    {r["parent_asin"] for r in history_rows}
                    | {r["parent_asin"] for r in catalog_rows}
                ),
                "history_candidate_count": len(history_rows),
                "catalog_candidate_count": len(catalog_rows),
                "catalog_seconds": round(catalog_seconds, 4),
                "union_candidate_count": len(
                    history_by_id.keys() | catalog_by_id.keys()
                ),
                "overlap_count": len(history_by_id.keys() & catalog_by_id.keys()),
                "policies": policies,
                "policy_changes": changes,
                "role_status_top10_union": dict(role_status),
                "role_target": {
                    "type": role_target.type,
                    "role": role_target.role.value,
                },
            }
        )
        print(
            f"{index}/{len(cases)} {case['case_id']} q={query!r} "
            f"hit={target in (history_by_id.keys() | catalog_by_id.keys())} "
            f"balanced={target_rank['balanced']} "
            f"intent={target_rank['intent_focus']} history={target_rank['history_focus']}",
            flush=True,
        )

    evaluated = [row for row in per_user if row["status"] == "evaluated"]
    summary = {
        "schema_version": "agentrecx.p8c.source_policy_eval.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(CLASSICAL_DEV.relative_to(ROOT)),
        "split": "dev",
        "evaluation_kind": "next_item_relevance_proxy",
        "not_a_search_satisfaction_metric": True,
        "users": len(cases),
        "evaluated": len(evaluated),
        "skipped": len(cases) - len(evaluated),
        "history_k": args.history_k,
        "catalog_k": args.catalog_k,
        "policies": list(SOURCE_POLICIES),
        "query_source": "last two history items' categories/title only; target never used",
        "target": "validation_parent_asin only; frozen test/Holdout never read",
        "catalog_source": {
            "mode": catalog.mode,
            "score_kind": catalog.score_kind,
            "index": str(INDEX.relative_to(ROOT)),
        },
        "seconds": round(time.perf_counter() - started, 2),
        "metrics": {},
        "target_reachability": {
            "history_pool": sum(row["target_in_history_pool"] for row in evaluated),
            "catalog_pool": sum(row["target_in_catalog_pool"] for row in evaluated),
            "union_pool": sum(row["target_in_union"] for row in evaluated),
        },
        "rank_change": {},
        "source_overlap": {
            "mean_history_candidates": (
                sum(row["history_candidate_count"] for row in evaluated) / len(evaluated)
                if evaluated
                else 0.0
            ),
            "mean_catalog_candidates": (
                sum(row["catalog_candidate_count"] for row in evaluated) / len(evaluated)
                if evaluated
                else 0.0
            ),
            "mean_overlap": (
                sum(row["overlap_count"] for row in evaluated) / len(evaluated)
                if evaluated
                else 0.0
            ),
        },
        "by_history_length": {},
        "oracle": {},
    }
    for policy in SOURCE_POLICIES:
        summary["metrics"][policy] = {
            "recall@10": (
                sum(row["policies"][policy]["recall@10"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "ndcg@10": (
                sum(row["policies"][policy]["ndcg@10"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "mrr@10": (
                sum(row["policies"][policy]["mrr@10"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
        }
        summary["rank_change"][policy] = {
            "order_changed_rate": (
                sum(row["policy_changes"][policy]["order_changed"] for row in evaluated)
                / len(evaluated) if evaluated else 0.0
            ),
            "set_changed_rate": (
                sum(row["policy_changes"][policy]["set_changed"] for row in evaluated)
                / len(evaluated) if evaluated else 0.0
            ),
            "mean_jaccard_vs_balanced": (
                sum(row["policy_changes"][policy]["jaccard_vs_balanced"] for row in evaluated)
                / len(evaluated) if evaluated else 0.0
            ),
        }
    buckets = defaultdict(list)
    for row in evaluated:
        length = row["history_len"]
        bucket = "0-3" if length <= 3 else "4-7" if length <= 7 else "8+"
        buckets[bucket].append(row)
    for bucket, rows in sorted(buckets.items()):
        summary["by_history_length"][bucket] = {
            "n": len(rows),
            "metrics": {
                policy: {
                    key: sum(row["policies"][policy][key] for row in rows) / len(rows)
                    for key in ("recall@10", "ndcg@10", "mrr@10")
                }
                for policy in SOURCE_POLICIES
            },
        }
    best = Counter()
    for row in evaluated:
        scores = {
            policy: row["policies"][policy]["ndcg@10"]
            for policy in SOURCE_POLICIES
        }
        best[max(SOURCE_POLICIES, key=lambda policy: (scores[policy], policy))] += 1
    summary["oracle"] = {
        "label": "uses validation target; ANALYSIS ONLY, not a Planner-achievable score",
        "best_policy_counts": dict(best),
    }
    (OUT / "per_user.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_user),
        encoding="utf-8",
    )
    (OUT / "candidate_cache.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cache_rows),
        encoding="utf-8",
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
