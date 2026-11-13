"""Reranker benchmark runner for Advanced Retrieval v2.

``existing`` is a no-op control.  ``cross_encoder`` loads an explicit local model and is
a Phase-2 GPU experiment; it fails loudly when the model is unavailable.  This runner
does not change the default Agent path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.retrieval.reranker import (
    RerankCandidate, build_reranker,
)

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "data/p14d1/dev_tasks.jsonl"
TARGETS = REPO / "data/p14d1/dev_targets.jsonl"
BGE_RESULTS = REPO / "runs/p14d2/bge_retrieval.json"
OUT = REPO / "runs/advanced_retrieval_v2"


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("existing", "cross_encoder"), default="existing")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "cross_encoder" and not args.model:
        parser.error("--model is required for cross_encoder mode")

    catalogue = ElectronicsCatalogue.load()
    tasks = _load(TASKS)
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    targets = {row["task_id"]: row["target_parent_asin"] for row in _load(TARGETS)}
    bge_top = json.loads(BGE_RESULTS.read_text(encoding="utf-8"))["top100"]["original"]
    reranker = build_reranker(args.mode, model_path=args.model, device=args.device,
                              batch_size=args.batch_size, max_length=args.max_length)
    rows = []
    started = time.perf_counter()
    for task in tasks:
        task_id = task["task_id"]
        candidates = []
        for rank, row in enumerate(bge_top[task_id][:args.depth], start=1):
            product = catalogue.get(row["item_id"])
            if product is None:
                continue
            candidates.append(RerankCandidate(
                item_id=row["item_id"], text=product.text[:1200],
                existing_score=float(row["score"]), existing_rank=rank,
            ))
        ranked = reranker.rerank(task["query_original"], candidates, top_n=args.depth)
        ids = [item.item_id for item in ranked]
        target = targets[task_id]
        rows.append({
            "task_id": task_id, "target": target,
            "target_reranked_rank": ids.index(target) + 1 if target in ids else None,
            "hit_at_1": bool(ids and ids[0] == target),
            "hit_at_5": target in ids[:5],
            "hit_at_10": target in ids[:10],
            "candidate_ids": ids,
            "scores": [item.as_dict() for item in ranked[:10]],
        })
    latency = round(time.perf_counter() - started, 3)
    n = len(rows)
    payload = {
        "schema": "agentrecx.advanced_retrieval.reranker_benchmark.v1",
        "mode": args.mode,
        "model": args.model,
        "device": args.device,
        "depth": args.depth,
        "tasks": n,
        "label_status": "provisional_one_observed_positive_other_items_unjudged",
        "metrics": {
            "hit@1": round(sum(1 for row in rows if row["hit_at_1"]) / max(1, n), 6),
            "hit@5": round(sum(1 for row in rows if row["hit_at_5"]) / max(1, n), 6),
            "hit@10": round(sum(1 for row in rows if row["hit_at_10"]) / max(1, n), 6),
            "target_rank_median": statistics.median([
                row["target_reranked_rank"] for row in rows
                if row["target_reranked_rank"] is not None
            ]) if any(row["target_reranked_rank"] for row in rows) else None,
            "latency_seconds": latency,
        },
        "reranker_class": type(reranker).__name__,
        "available": bool(getattr(reranker, "available", True)),
        "rows": rows,
    }
    output = args.out or OUT / f"reranker_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("mode", "model", "tasks", "metrics")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
