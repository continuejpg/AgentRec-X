"""Phase 2 dense-retrieval comparison: Original BGE vs Enriched BGE and hybrids.

A. Original C4 BGE (stored real top-100 from the existing index)
B. Enriched BGE v2 (actual query encoding against the new NPZ index)
C. Enriched BGE + Enriched BM25, RRF k=60
D. Original C4 BGE + Original C4 BM25, RRF k=60 (existing reference)

The target label is one observed positive; all other products are unjudged.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.p14c1.search_backend import LocalBGEDenseIndex
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.control.catalog_search import reciprocal_rank_fusion
from recommendation.retrieval import load_enriched_bm25

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "data/p14d1/dev_tasks.jsonl"
TARGETS = REPO / "data/p14d1/dev_targets.jsonl"
OLD_BGE_RESULTS = REPO / "runs/p14d2/bge_retrieval.json"
OLD_BGE_INDEX = REPO / "runs/p14d2/bge_index.npz"
ENRICHED_DOCS = REPO / "data/enrichment/electronics_retrieval_docs_v2.jsonl"
DEFAULT_NEW_INDEX = REPO / "runs/advanced_retrieval_v2/bge_enriched_v2.npz"
KS = (1, 10, 20, 100)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _strip(identity: str) -> str:
    return str(identity).split(":", 1)[-1]


def _metrics(ranked: dict[str, list[str]], targets: dict[str, str]) -> dict[str, Any]:
    n = len(ranked)
    result: dict[str, Any] = {"tasks": n}
    for k in KS:
        hits = sum(1 for task_id, ids in ranked.items() if targets[task_id] in ids[:k])
        result[f"hit@{k}"] = round(hits / max(1, n), 6)
        result[f"coverage@{k}"] = hits
    reciprocal = []
    ndcg = []
    for task_id, ids in ranked.items():
        target = targets[task_id]
        if target in ids:
            rank = ids.index(target) + 1
            reciprocal.append(1.0 / rank if rank <= 10 else 0.0)
            ndcg.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
        else:
            reciprocal.append(0.0)
            ndcg.append(0.0)
    result["mrr@10"] = round(statistics.mean(reciprocal), 6) if reciprocal else 0.0
    result["ndcg@10"] = round(statistics.mean(ndcg), 6) if ndcg else 0.0
    return result


def _method(name: str, ranked: dict[str, list[str]], targets: dict[str, str],
            latency_s: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "method": name,
        "metrics": _metrics(ranked, targets),
        "latency_seconds": round(latency_s, 3),
        "target_ranks": {
            task_id: (ids.index(targets[task_id]) + 1 if targets[task_id] in ids else None)
            for task_id, ids in ranked.items()
        },
        "top1": {task_id: (ids[0] if ids else None) for task_id, ids in ranked.items()},
    }
    if extra:
        payload.update(extra)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge-index", type=Path, default=DEFAULT_NEW_INDEX)
    parser.add_argument("--bge-model", type=Path, default=REPO / "models/bge-base-en-v1.5")
    parser.add_argument("--out", type=Path,
                        default=REPO / "runs/advanced_retrieval_v2/phase2_retrieval_comparison.json")
    args = parser.parse_args()
    if not args.bge_index.is_file():
        raise SystemExit(f"enriched BGE index not found: {args.bge_index}; build it first")
    if not args.bge_model.is_dir():
        raise SystemExit(f"BGE model not found: {args.bge_model}")
    tasks = _load(TASKS)
    targets = {row["task_id"]: row["target_parent_asin"] for row in _load(TARGETS)}
    old_bge_payload = json.loads(OLD_BGE_RESULTS.read_text(encoding="utf-8"))
    catalogue = ElectronicsCatalogue.load()
    old_bm25 = catalogue.search_source
    enriched_bm25 = load_enriched_bm25(ENRICHED_DOCS)
    dense = LocalBGEDenseIndex.load(args.bge_index, args.bge_model)
    result: dict[str, Any] = {
        "schema": "agentrecx.advanced_retrieval.v2_phase2_comparison.v1",
        "task_set": str(TASKS), "targets": str(TARGETS),
        "tasks": len(tasks), "ks": list(KS), "rrf_k": 60,
        "label_status": "provisional_one_observed_positive_other_items_unjudged",
        "hard_verified_recall": None,
        "hard_verified_note": "one observed positive per task; no exhaustive hard-constraint labels",
        "indexes": {
            "old_bge_index": str(OLD_BGE_INDEX),
            "old_bge_index_bytes": OLD_BGE_INDEX.stat().st_size if OLD_BGE_INDEX.is_file() else None,
            "new_bge_index": str(args.bge_index),
            "new_bge_index_bytes": args.bge_index.stat().st_size,
            "new_bge_index_manifest": str(args.bge_index.with_suffix(".json")),
            "enriched_docs": str(ENRICHED_DOCS),
            "enriched_docs_bytes": ENRICHED_DOCS.stat().st_size,
            "enriched_bm25_documents": len(enriched_bm25.documents),
            "bge_model": str(args.bge_model),
        },
        "variants": {},
    }
    for variant in ("original", "rewritten"):
        old_bge = {
            task["task_id"]: [row["item_id"] for row in
                              old_bge_payload["top100"][variant][task["task_id"]]]
            for task in tasks
        }
        old_bm25_ranks: dict[str, list[str]] = {}
        enriched_bm25_ranks: dict[str, list[str]] = {}
        bm25_started = time.perf_counter()
        for task in tasks:
            query = task[f"query_{variant}"]
            task_id = task["task_id"]
            old_bm25_ranks[task_id] = [
                _strip(hit.parent_asin) for hit in old_bm25.search(query, limit=100).hits
            ]
            enriched_bm25_ranks[task_id] = [
                _strip(hit.parent_asin) for hit in enriched_bm25.search(query, limit=100).hits
            ]
        bm25_seconds = round(time.perf_counter() - bm25_started, 3)
        enriched_bge: dict[str, list[str]] = {}
        dense_started = time.perf_counter()
        for task in tasks:
            enriched_bge[task["task_id"]] = [
                _strip(hit.parent_asin)
                for hit in dense.search(task[f"query_{variant}"], limit=100)
            ]
        bge_seconds = round(time.perf_counter() - dense_started, 3)
        old_hybrid_started = time.perf_counter()
        old_hybrid = {task_id: [item for item, _ in reciprocal_rank_fusion(
            [old_bm25_ranks[task_id], old_bge[task_id]], k=60)] for task_id in old_bm25_ranks}
        old_hybrid_seconds = round(time.perf_counter() - old_hybrid_started, 6)
        enriched_hybrid_started = time.perf_counter()
        enriched_hybrid = {task_id: [item for item, _ in reciprocal_rank_fusion(
            [enriched_bm25_ranks[task_id], enriched_bge[task_id]], k=60)]
                          for task_id in enriched_bge}
        enriched_hybrid_seconds = round(time.perf_counter() - enriched_hybrid_started, 6)
        methods = {
            "A_original_bge": _method("A_original_bge", old_bge, targets, 0.0,
                                     {"execution_note": "stored real C4 BGE top-100"}),
            "B_enriched_bge": _method("B_enriched_bge", enriched_bge, targets, bge_seconds,
                                     {"execution_note": "actual query encoding against enriched NPZ"}),
            "C_enriched_hybrid_rrf": _method("C_enriched_hybrid_rrf", enriched_hybrid, targets,
                                            bm25_seconds + enriched_hybrid_seconds,
                                            {"fusion_seconds": enriched_hybrid_seconds,
                                             "bm25_seconds": bm25_seconds}),
            "D_original_hybrid_rrf": _method("D_original_hybrid_rrf", old_hybrid, targets,
                                            bm25_seconds + old_hybrid_seconds,
                                            {"fusion_seconds": old_hybrid_seconds}),
        }
        lost = [
            {"task_id": task["task_id"],
             "original_rank": methods["A_original_bge"]["target_ranks"][task["task_id"]],
             "enriched_rank": methods["B_enriched_bge"]["target_ranks"][task["task_id"]]}
            for task in tasks
            if methods["A_original_bge"]["target_ranks"][task["task_id"]] is not None
            and methods["B_enriched_bge"]["target_ranks"][task["task_id"]] is None
        ]
        result["variants"][variant] = {"methods": methods, "targets_lost_enriched": lost}
        print(json.dumps({"variant": variant, "methods":
                          {name: methods[name]["metrics"] for name in methods},
                          "targets_lost": len(lost)}, ensure_ascii=False), flush=True)
    result["cpu"] = {"max_rss_kb_after": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    result["notes"] = [
        "A uses the existing C4 BGE top-100 artifact; B actually encodes queries against the v2 index",
        "C and D are fixed equal-weight RRF k=60; no target tuning",
        "one observed positive per task; other products are unjudged, not negatives",
        "hard-verified candidate recall is unavailable for this label set",
        "reranker/selection improvements cannot recover targets absent from the candidate pool",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cpu": result["cpu"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
