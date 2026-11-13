"""Run the frozen Qwen3-Reranker-4B over the fixed preference-ranking pool.

No training, no Gate/Scorer changes, no candidate addition.  The query input contains the
real task query, legal active preferences and task hard requirements.  The model only
reorders the fixed candidate ids supplied by the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recommendation.retrieval.reranker import Qwen3Reranker, RerankCandidate


def _query(row: dict[str, Any]) -> str:
    prefs = row.get("historical_memory") or []
    hard = row.get("hard_requirements") or []
    preference_text = "; ".join(
        f"{item.get('polarity', 'positive')}:{item.get('value')}" for item in prefs
        if item.get("value")
    ) or "none"
    hard_text = "; ".join(
        f"{item.get('kind')}:{item.get('value')}" for item in hard if item.get("value")
    ) or "none"
    return (
        f"User request: {row['query']}\n"
        f"Active remembered preferences: {preference_text}\n"
        f"Hard requirements: {hard_text}\n"
        "Rank candidates by how well they satisfy the request, the active preferences, "
        "and the hard requirements. Prefer explicit catalogue evidence; do not infer "
        "unstated attributes."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limits", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row.get("expected_decision") == "recommend"]
    if args.limits:
        rows = rows[:args.limits]
    reranker = Qwen3Reranker(
        model_path=str(args.model), device=args.device, batch_size=args.batch_size,
        max_length=args.max_length, quantization="4bit",
    )
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        candidates = [
            RerankCandidate(
                item_id=item["item_id"], text=str(item.get("text") or item.get("title") or ""),
                existing_score=float(item.get("original_bge_score") or 0.0),
                existing_rank=int(item.get("original_bge_rank") or 0),
            )
            for item in row["candidate_pool"]
        ]
        ranked = reranker.rerank(_query(row), candidates, top_n=None)
        out_rows.append({
            "task_id": row["task_id"],
            "query_for_reranker": _query(row),
            "candidate_ids": [item.item_id for item in candidates],
            "ranked": [item.as_dict() for item in ranked],
            "resources": {
                "load_seconds": getattr(reranker, "load_seconds", None),
                "scoring_seconds": getattr(reranker, "scoring_seconds", None),
                "pairs_scored": getattr(reranker, "pairs_scored", None),
                "peak_vram_mib": getattr(reranker, "peak_vram_mib", None),
            },
        })
        print(json.dumps({"task_id": row["task_id"], "top": ranked[0].item_id if ranked else None,
                          "n": len(ranked)}, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows),
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out), "tasks": len(out_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
