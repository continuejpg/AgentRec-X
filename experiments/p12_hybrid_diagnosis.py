"""P12 hybrid budget-boundary diagnosis.

Compares the original hybrid (full BM25 ranking in RRF) with an opt-in bounded BM25 head
and the two individual sources on the exact P11 tasks.  No Gold is defined by retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from recommendation.control.catalog_search import CatalogSearchSource

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data/p11_benchmark/tasks.jsonl"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
MODEL = ROOT / "models/bge-base-en-v1.5"
OUT = ROOT / "runs/p12/hybrid"


def load_catalog(path: Path) -> Any:
    records: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            asin = row.get("parent_asin")
            if not asin:
                continue
            records[str(asin)] = SimpleNamespace(
                parent_asin=str(asin),
                title=row.get("title"),
                store=row.get("store"),
                main_category=row.get("main_category"),
                categories=tuple(row.get("categories") or ()),
                features=tuple(row.get("features") or ()),
                description=tuple(row.get("description") or ()),
                price_text=row.get("price_text"),
            )
    return SimpleNamespace(records=records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=50)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--products", type=Path, default=PRODUCTS)
    parser.add_argument("--mappings", type=Path, default=MAPPINGS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    from recommendation.inference.dense_catalog import DenseCatalogIndex

    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    catalog = load_catalog(args.products)
    dense = DenseCatalogIndex.load(
        args.index,
        model_path=str(args.model),
        catalogue_records=catalog.records,
        mapping_sha=hashlib.sha256(args.mappings.read_bytes()).hexdigest(),
    )
    sources = {
        "bm25": CatalogSearchSource(catalog, mode="bm25"),
        "bge": CatalogSearchSource(catalog, mode="bge", dense_index=dense),
        "hybrid_legacy": CatalogSearchSource(catalog, mode="hybrid", dense_index=dense),
        "hybrid_bounded": CatalogSearchSource(
            catalog, mode="hybrid", dense_index=dense, hybrid_bm25_head=args.head
        ),
    }
    rows = []
    for task in tasks:
        pool = {str(item["parent_asin"]) for item in task.get("verified_candidate_pool") or ()}
        query = task["conversation_turns"][0]
        row = {"task_id": task["task_id"], "pool_size": len(pool), "query": query}
        for name, source in sources.items():
            result = source.search(query, limit=50)
            hits = list(result.hits)
            row[name] = {
                "reachable@10": bool(pool & {item.parent_asin for item in hits[:10]}),
                "reachable@50": bool(pool & {item.parent_asin for item in hits}),
                "found@10": len(pool & {item.parent_asin for item in hits[:10]}),
                "found@50": len(pool & {item.parent_asin for item in hits}),
                "n_hits": len(hits),
            }
        rows.append(row)
    pool_rows = [row for row in rows if row["pool_size"] > 0]
    summary = {
        "schema_version": "agentrecx.p12.hybrid_diagnosis.v1",
        "head": args.head,
        "tasks": len(rows),
        "pool_tasks": len(pool_rows),
        "results": {
            name: {
                "reachable@10": sum(1 for row in pool_rows if row[name]["reachable@10"]),
                "reachable@50": sum(1 for row in pool_rows if row[name]["reachable@50"]),
                "mean_found@10": float(np.mean([row[name]["found@10"] for row in pool_rows])),
            }
            for name in sources
        },
        "interpretation": (
            "hybrid_legacy keeps full BM25 ranking in RRF; hybrid_bounded limits BM25 to "
            f"the {args.head}-deep head to match the dense head. No retrieval output is "
            "used as Gold."
        ),
    }
    (args.out / "per_task.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
