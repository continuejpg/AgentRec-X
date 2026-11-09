"""P8-D current shopping-intent retrieval check.

Runs the real BM25, BGE and Hybrid catalogue sources on the six synthetic Dev shopping
requests.  These tasks have no human Query-Product relevance labels; this script therefore
reports product-role and hard-price-constraint proxies plus synthetic seed coverage only.
It never calls this search-satisfaction evidence.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/p8_d/current_intent"
TASKS = ROOT / "data/p6_benchmark_v0/dynamic_dev.jsonl"
INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
MODEL = ROOT / "models/bge-base-en-v1.5"
MAPPING = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"

os.environ["AGENTRECX_CATALOG_SEARCH_MODE"] = "hybrid"
os.environ["AGENTRECX_DENSE_INDEX"] = str(INDEX)
os.environ["AGENTRECX_BGE_MODEL_PATH"] = str(MODEL)
os.environ["AGENTRECX_MAPPINGS_PATH"] = str(MAPPING)
os.environ["AGENTRECX_CATALOG_METADATA_PATH"] = str(PRODUCTS)
os.environ["AGENTRECX_SERVING_FUSION"] = "rrf3"
os.environ["AGENTRECX_STATE_MODE"] = "off"

from recommendation.api.app import ServiceSettings, build_engine
from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.demo.product_role import assess_candidate, infer_target
from recommendation.demo.runtime import build_demo_runtime

PRICE_RE = re.compile(r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_price_max(query: str) -> float | None:
    match = PRICE_RE.search(query)
    return float(match.group(1)) if match else None


def parse_price(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(match.group(0)) if match else None


def evaluate_hits(
    hits: list[Any],
    *,
    records: Any,
    target_role: Any,
    price_max: float | None,
    seeds: set[str],
    limit: int,
) -> dict:
    top = hits[:limit]
    role_counts = Counter()
    price_counts = Counter()
    seed_hits = 0
    inspected = 0
    for hit in top:
        record = records.get(hit.parent_asin)
        if record is None:
            continue
        inspected += 1
        role_counts[assess_candidate(hit.parent_asin, record, target_role).status.value] += 1
        if price_max is not None:
            value = parse_price(str(getattr(record, "price_text", "") or ""))
            if value is None:
                price_counts["unresolved"] += 1
            elif value <= price_max:
                price_counts["eligible"] += 1
            else:
                price_counts["ineligible"] += 1
        if hit.parent_asin in seeds:
            seed_hits += 1
    denominator = max(1, len(top))
    return {
        "hits": len(top),
        "inspected": inspected,
        "role_match": role_counts["MATCH"],
        "role_mismatch": role_counts["MISMATCH"],
        "role_unknown": role_counts["UNKNOWN"],
        "role_match_rate": role_counts["MATCH"] / denominator,
        "role_mismatch_rate": role_counts["MISMATCH"] / denominator,
        "role_unknown_rate": role_counts["UNKNOWN"] / denominator,
        "price_constraint": price_max,
        "price_eligible": price_counts["eligible"],
        "price_ineligible": price_counts["ineligible"],
        "price_unresolved": price_counts["unresolved"],
        "price_eligible_rate": (
            price_counts["eligible"] / denominator if price_max is not None else None
        ),
        "seed_count": len(seeds),
        "seed_hits": seed_hits,
        "seed_recall": seed_hits / len(seeds) if seeds else None,
        "top_ids": [hit.parent_asin for hit in top[:10]],
        "top_sources": [
            {"parent_asin": hit.parent_asin, "score": float(hit.score),
             "matched_fields": list(hit.matched_fields)}
            for hit in top[:10]
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [
        json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = ServiceSettings.from_env()
    engine, error = build_engine(settings)
    if engine is None:
        raise SystemExit(error)
    runtime = build_demo_runtime(settings=settings, engine=engine)
    service = runtime.agent_service()
    metadata = runtime.metadata
    hybrid = service._catalog_search
    dense = hybrid._dense
    if hybrid.mode != "hybrid" or dense is None:
        raise SystemExit("hybrid dense source unavailable")
    bm25 = CatalogSearchSource(metadata, mode="bm25")
    bge = CatalogSearchSource(metadata, mode="bge", dense_index=dense)
    sources = {"bm25": bm25, "bge": bge, "hybrid": hybrid}

    per_task = []
    for task in tasks:
        query = str(task.get("initial_request") or "")
        role = infer_target(query)
        price_max = parse_price_max(query)
        seeds = set(str(item) for item in (task.get("catalogue_seed_asins") or ()))
        task_result = {
            "task_id": task.get("task_id"),
            "template_id": task.get("template_id"),
            "query": query,
            "query_role_target": {"type": role.type, "role": role.role.value},
            "price_max": price_max,
            "synthetic_seed_count": len(seeds),
            "modes": {},
        }
        for mode, source in sources.items():
            hits = list(source.search(query, limit=20).hits)
            task_result["modes"][mode] = evaluate_hits(
                hits,
                records=metadata.records,
                target_role=role,
                price_max=price_max,
                seeds=seeds,
                limit=10,
            )
        per_task.append(task_result)

    aggregate: dict[str, Any] = {}
    for mode in sources:
        rows = [task["modes"][mode] for task in per_task]
        price_rows = [row["price_eligible_rate"] for row in rows
                      if row["price_eligible_rate"] is not None]
        seed_rows = [row["seed_recall"] for row in rows
                     if row["seed_recall"] is not None]
        aggregate[mode] = {
            "mean_role_match_rate": sum(row["role_match_rate"] for row in rows) / len(rows),
            "mean_role_mismatch_rate": sum(row["role_mismatch_rate"] for row in rows) / len(rows),
            "mean_role_unknown_rate": sum(row["role_unknown_rate"] for row in rows) / len(rows),
            "mean_price_eligible_rate": (
                sum(price_rows) / len(price_rows) if price_rows else None
            ),
            "price_constraint_tasks": len(price_rows),
            "mean_seed_recall": (
                sum(seed_rows) / len(seed_rows) if seed_rows else None
            ),
            "tasks": len(rows),
        }
    report = {
        "schema_version": "agentrecx.p8d.current_intent_retrieval.v1",
        "created_utc": "2026-11-08",
        "evaluation_kind": "current_shopping_intent_retrieval_proxy",
        "query_source": "P6 dynamic Dev synthetic initial_request",
        "negative_label_warning": {
            "real_query_product_relevance_labels": False,
            "synthetic_seed_is_not_human_relevance": True,
            "role_and_price_are_not_search_satisfaction": True,
            "esci_used": False,
            "esci_catalogue_identity_warning": "ESCI item ids would not be this catalogue; not used",
        },
        "config": {
            "source": "real catalogue, real BM25/BGE/Hybrid",
            "bge_index": str(INDEX.relative_to(ROOT)),
            "bge_model": str(MODEL.relative_to(ROOT)),
            "top_k_inspected": 10,
            "limit": 20,
            "tasks": len(tasks),
        },
        "aggregate": aggregate,
        "per_task": per_task,
    }
    (OUT / "retrieval_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate, "tasks": [t["task_id"] for t in per_task]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
