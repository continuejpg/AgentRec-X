"""P11 retrieval/evidence execution on the Grounded Shopping Benchmark.

This evaluates catalogue-source reachability and history-source reachability against the
metadata-verified candidate pools.  It does not use a retrieval score to define Gold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.inference.dense_catalog import DenseCatalogIndex

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data/p11_benchmark/tasks.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
MODEL = ROOT / "models/bge-base-en-v1.5"
HISTORY_CACHE = ROOT / "runs/p8_d/multiuser/candidate_cache.jsonl"
OUT = ROOT / "runs/p11/eval"


def load_catalog(path: Path) -> Any:
    """Load only the catalogue fields needed by BM25/BGE, without pydantic wrappers."""
    records: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parent_asin = row.get("parent_asin")
            if not parent_asin:
                continue
            records[str(parent_asin)] = SimpleNamespace(
                parent_asin=str(parent_asin),
                title=row.get("title"),
                store=row.get("store"),
                main_category=row.get("main_category"),
                categories=tuple(row.get("categories") or ()),
                features=tuple(row.get("features") or ()),
                description=tuple(row.get("description") or ()),
                price_text=row.get("price_text"),
            )
    return SimpleNamespace(records=records)


def load_tasks(path: Path = TASKS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_history_cache(path: Path) -> dict[str, list[dict]]:
    by_user: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_user[str(row["user_id"])] = list(row.get("history_rows") or ())
    return by_user


def _pool_ids(task: dict) -> set[str]:
    return {str(item["parent_asin"]) for item in task.get("verified_candidate_pool") or ()}


def _hit(pool: set[str], hits: list[Any], k: int) -> dict[str, Any]:
    top = {str(hit.parent_asin) for hit in hits[:k]}
    return {
        "hits": len(top),
        "found": len(pool & top),
        "reachable": bool(pool & top),
        "found_ids": sorted(pool & top),
    }


def _affinity(task: dict, hits: list[Any], k: int, metadata: Any) -> bool:
    profile = task.get("history_profile")
    if not profile or not task.get("history_evidence_refs"):
        return False
    for hit in hits[:k]:
        record = metadata.records.get(hit.parent_asin)
        if record is None:
            continue
        if any(str(value) == profile for value in (getattr(record, "categories", ()) or ())):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--catalog-limit", type=int, default=50)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--products", type=Path, default=PRODUCTS)
    parser.add_argument("--mappings", type=Path, default=MAPPINGS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--history-cache", type=Path, default=HISTORY_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    metadata = load_catalog(args.products)
    history_cache = load_history_cache(args.history_cache)
    dense = DenseCatalogIndex.load(
        args.index,
        model_path=str(args.model),
        catalogue_records=metadata.records,
        mapping_sha=__import__("hashlib").sha256(args.mappings.read_bytes()).hexdigest(),
    )
    sources = {
        "bm25": CatalogSearchSource(metadata, mode="bm25"),
        "bge": CatalogSearchSource(metadata, mode="bge", dense_index=dense),
        "hybrid": CatalogSearchSource(metadata, mode="hybrid", dense_index=dense),
    }
    per_task = []
    for index, task in enumerate(tasks, start=1):
        query = str(task["conversation_turns"][0])
        pool = _pool_ids(task)
        history_rows = history_cache.get(str(task["user_id"]), [])
        history_ids = {str(row["parent_asin"]) for row in history_rows[: args.history_limit]}
        catalog_rows = {}
        for mode, source in sources.items():
            result = source.search(query, limit=args.catalog_limit)
            hits = list(result.hits)
            catalog_rows[mode] = hits
        per_task.append(
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "user_id": task["user_id"],
                "query": query,
                "preference_evidence_level": task["preference_evidence_level"],
                "pool_size": len(pool),
                "history_pool_found": len(pool & history_ids),
                "history_reachable": bool(pool & history_ids),
                "history_candidates": len(history_rows),
                "history_limit": args.history_limit,
                "catalog": {
                    mode: {
                        **_hit(pool, hits, 10),
                        "top50": _hit(pool, hits, 50),
                        "n_hits": len(hits),
                        "score_kind": sources[mode].score_kind,
                    }
                    for mode, hits in catalog_rows.items()
                },
                "history_affinity": {
                    mode: _affinity(task, hits, 10, metadata)
                    for mode, hits in catalog_rows.items()
                },
            }
        )
        if index % 25 == 0:
            print(f"{index}/{len(tasks)}", flush=True)
    summary = {
        "schema_version": "agentrecx.p11.retrieval_eval.v1",
        "tasks": len(tasks),
        "task_type_counts": dict(Counter(task["task_type"] for task in tasks)),
        "pool_tasks": sum(1 for task in tasks if task.get("verified_candidate_pool")),
        "empty_pool_tasks": sum(1 for task in tasks if not task.get("verified_candidate_pool")),
        "history": {
            "tasks_with_history_cache": sum(1 for task in tasks if task["user_id"] in history_cache),
            "reachable_at_50": sum(1 for row in per_task if row["history_reachable"]),
            "reachable_at_100": sum(
                1 for row in per_task if row["history_reachable"]
            ),
            "note": "history source from real P8-D model cache, top-50; no BGE query",
        },
        "catalog": {},
        "personalization_proxy": {},
    }
    pool_rows = [row for row in per_task if row["pool_size"] > 0]
    for mode in sources:
        summary["catalog"][mode] = {
            "reachable_at_10": sum(1 for row in pool_rows if row["catalog"][mode]["reachable"]),
            "reachable_at_50": sum(1 for row in pool_rows if row["catalog"][mode]["top50"]["reachable"]),
            "pool_tasks": len(pool_rows),
            "mean_found_at_10": (
                float(np.mean([row["catalog"][mode]["found"] for row in pool_rows]))
                if pool_rows else 0.0
            ),
        }
    level_b = [row for row in pool_rows if row["preference_evidence_level"] == "B"]
    unknown = [row for row in pool_rows if row["preference_evidence_level"] == "UNKNOWN"]
    for mode in sources:
        summary["personalization_proxy"][mode] = {
            "level_b_tasks": len(level_b),
            "level_b_history_affinity_at_10": sum(
                1 for row in level_b if row["history_affinity"][mode]
            ),
            "unknown_tasks": len(unknown),
            "narrative": "history-affinity proxy only; not human-confirmed preference satisfaction",
        }
    (args.out / "per_task.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_task),
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
