"""CPU retrieval comparison for Advanced Retrieval v2.

Methods:
  A. original C4 BM25
  B. original C4 BGE dense (precomputed query vectors/ranks from the existing index)
  C. original C4 BM25 + BGE, fixed RRF k=60
  D. enriched retrieval-document BM25

The target is an observed positive, not an exhaustive relevance judgment.  All other
catalogue items are unjudged; metrics are labelled provisional and no unjudged item is
treated as a negative.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p14c1.search_backend import LocalBGEDenseIndex
from recommendation.control.catalog_search import reciprocal_rank_fusion
from recommendation.retrieval import load_enriched_bm25

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "data/p14d1/dev_tasks.jsonl"
TARGETS = REPO / "data/p14d1/dev_targets.jsonl"
BGE_RESULTS = REPO / "runs/p14d2/bge_retrieval.json"
ENRICHED_DOCS = REPO / "data/enrichment/electronics_retrieval_docs_v2.jsonl"
OUT = REPO / "runs/advanced_retrieval_v2"
KS = (10, 20, 100)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _strip(identity: str) -> str:
    return str(identity).split(":", 1)[-1]


def _metrics(ranked_by_task: dict[str, list[str]], targets: dict[str, str]) -> dict[str, Any]:
    tasks = len(ranked_by_task)
    out: dict[str, Any] = {"tasks": tasks}
    for k in KS:
        hits = sum(1 for task_id, ids in ranked_by_task.items()
                   if targets[task_id] in ids[:k])
        out[f"recall@{k}"] = round(hits / max(1, tasks), 6)
        out[f"target_coverage@{k}"] = hits
    out["hit@1"] = round(sum(1 for task_id, ids in ranked_by_task.items()
                             if ids and ids[0] == targets[task_id]) / max(1, tasks), 6)
    return out


def _run_source(name: str, source: Any, tasks: list[dict[str, Any]], variant: str,
                targets: dict[str, str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    ranked: dict[str, list[str]] = {}
    latency_ms: list[float] = []
    started = time.perf_counter()
    for task in tasks:
        before = time.perf_counter()
        result = source.search(task[f"query_{variant}"], limit=100)
        latency_ms.append(round((time.perf_counter() - before) * 1000.0, 3))
        ranked[task["task_id"]] = [_strip(hit.parent_asin) for hit in result.hits]
    total = round(time.perf_counter() - started, 3)
    payload = {
        "method": name, "variant": variant, "metrics": _metrics(ranked, targets),
        "latency_ms": {"total": round(sum(latency_ms), 3),
                       "mean": round(statistics.mean(latency_ms), 3) if latency_ms else 0.0,
                       "p50": round(statistics.median(latency_ms), 3) if latency_ms else 0.0,
                       "max": max(latency_ms) if latency_ms else 0.0},
        "target_ranks": {task_id: (ids.index(targets[task_id]) + 1
                                   if targets[task_id] in ids else None)
                         for task_id, ids in ranked.items()},
        "top1": {task_id: (ids[0] if ids else None) for task_id, ids in ranked.items()},
        "elapsed_s": total,
    }
    return payload, ranked


def _run_rank_lists(name: str, rank_lists: dict[str, list[str]],
                    latency_s: float, targets: dict[str, str]) -> dict[str, Any]:
    return {
        "method": name,
        "metrics": _metrics(rank_lists, targets),
        "latency_ms": {"total": round(latency_s * 1000.0, 3), "mean": 0.0,
                       "p50": 0.0, "max": 0.0},
        "target_ranks": {task_id: (ids.index(targets[task_id]) + 1
                                   if targets[task_id] in ids else None)
                         for task_id, ids in rank_lists.items()},
        "top1": {task_id: (ids[0] if ids else None) for task_id, ids in rank_lists.items()},
        "elapsed_s": latency_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["original", "rewritten"],
                        choices=("original", "rewritten"))
    parser.add_argument("--out", type=Path, default=OUT / "cpu_retrieval_comparison.json")
    parser.add_argument("--bge-index", type=Path, default=None,
                        help="optional actual dense index for a Phase-2 query-encoding pass")
    parser.add_argument("--bge-model", type=Path, default=None,
                        help="local BGE model matching --bge-index")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = _load_jsonl(TASKS)
    targets = {row["task_id"]: row["target_parent_asin"] for row in _load_jsonl(TARGETS)}
    bge_payload = json.loads(BGE_RESULTS.read_text(encoding="utf-8"))
    catalogue = ElectronicsCatalogue.load()
    bm25 = catalogue.search_source
    if bm25.mode != "bm25":
        raise RuntimeError("expected the catalogue default source to be BM25")
    enriched = load_enriched_bm25(ENRICHED_DOCS)
    actual_dense = None
    if args.bge_index is not None:
        if args.bge_model is None:
            parser.error("--bge-model is required with --bge-index")
        actual_dense = LocalBGEDenseIndex.load(args.bge_index, args.bge_model)
    result: dict[str, Any] = {
        "schema": "agentrecx.advanced_retrieval.v2_cpu_comparison.v1",
        "task_set": str(TASKS), "targets": str(TARGETS),
        "tasks": len(tasks), "ks": list(KS), "rrf_k": 60,
        "label_status": "provisional_one_observed_positive_other_items_unjudged",
        "hard_verified_recall": None,
        "hard_verified_note": "these retrieval tasks have one observed positive only; no exhaustive hard-constraint labels",
        "indexes": {
            "c4_bge_index": "runs/p14d2/bge_index.npz",
            "c4_bge_index_bytes": (REPO / "runs/p14d2/bge_index.npz").stat().st_size,
            "enriched_docs": str(ENRICHED_DOCS),
            "enriched_docs_bytes": ENRICHED_DOCS.stat().st_size,
            "enriched_bm25_documents": len(enriched.documents),
            "bge_results": str(BGE_RESULTS),
            "bge_model_revision": bge_payload["model"].get("revision"),
        },
        "variants": {}, "cpu": {},
    }
    for variant in args.variants:
        if actual_dense is not None:
            bge_started = time.perf_counter()
            bge_rank = {}
            for task in tasks:
                hits = actual_dense.search(task[f"query_{variant}"], limit=100)
                bge_rank[task["task_id"]] = [_strip(hit.parent_asin) for hit in hits]
            bge_latency = round(time.perf_counter() - bge_started, 3)
            actual_note = "actual dense query encoding via LocalBGEDenseIndex"
        else:
            bge_rank = {
                task["task_id"]: [row["item_id"] for row in
                                  bge_payload["top100"][variant][task["task_id"]]]
                for task in tasks
            }
            bge_latency = 0.0
            actual_note = "precomputed BGE ranks from existing index/query encoding"
        # B is a real dense ranking from the existing BGE index; the query vectors were
        # already encoded by that model, so this CPU comparison does not re-encode them.
        result["variants"][variant] = {
            "B_bge_c4": _run_rank_lists("B_bge_c4", bge_rank, bge_latency, targets)
        }
        result["variants"][variant]["B_bge_c4"]["execution_note"] = actual_note
        a_payload, a_ranked = _run_source("A_bm25_c4", bm25, tasks, variant, targets)
        result["variants"][variant]["A_bm25_c4"] = a_payload
        d_payload, d_ranked = _run_source("D_bm25_enriched_v2", enriched, tasks, variant, targets)
        result["variants"][variant]["D_bm25_enriched_v2"] = d_payload
        fusion_started = time.perf_counter()
        c_ranked = {
            task["task_id"]: [item_id for item_id, _score in reciprocal_rank_fusion(
                [a_ranked[task["task_id"]], bge_rank[task["task_id"]]], k=60
            )]
            for task in tasks
        }
        fusion_s = round(time.perf_counter() - fusion_started, 6)
        result["variants"][variant]["C_hybrid_rrf_c4"] = _run_rank_lists(
            "C_hybrid_rrf_c4", c_ranked, fusion_s, targets)
        result["variants"][variant]["C_hybrid_rrf_c4"]["fusion_seconds"] = fusion_s
        print(json.dumps({"variant": variant,
                          **{name: result["variants"][variant][name]["metrics"]
                             for name in ("A_bm25_c4", "B_bge_c4", "C_hybrid_rrf_c4",
                                          "D_bm25_enriched_v2")}},
                         ensure_ascii=False), flush=True)
    # Rank movement summaries (target rank B -> C/D).
    movements = {}
    for variant in args.variants:
        methods = result["variants"][variant]
        def ranks(name):
            return [methods[name]["target_ranks"][task["task_id"]] for task in tasks]
        movements[variant] = {}
        for target_name in ("C_hybrid_rrf_c4", "D_bm25_enriched_v2"):
            a, b = ranks("B_bge_c4"), ranks(target_name)
            improved = sum(1 for x, y in zip(a, b) if y is not None and (x is None or y < x))
            worsened = sum(1 for x, y in zip(a, b) if x is not None and (y is None or y > x))
            movements[variant][f"bge_to_{target_name}"] = {
                "improved": improved, "worsened": worsened,
                "same": len(tasks) - improved - worsened,
            }
    result["rank_movements"] = movements
    result["cpu"]["max_rss_kb_after"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["comparison_notes"] = [
        "A/B/C use the same original C4 catalogue and existing identity-aligned BGE index",
        "B reuses real BGE top-100 ranks produced by the existing index/model query encoding",
        "C is fixed RRF k=60 equal weights; it is also exercised by the hybrid source tests",
        "D is new enriched-document BM25; C4 catalogue/BGE index are untouched",
        "only one observed target is labelled; all other products are unjudged, not negatives",
        "hard-verified candidate recall is unavailable for these one-positive tasks",
    ]
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cpu": result["cpu"],
                      "movements": movements}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
