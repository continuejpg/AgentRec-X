"""P13-R retrieval/history diagnosis only.  No ranker training, no test retrieval."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES = ROOT / "runs/p13r/retrieval/queries.jsonl"
OUT = ROOT / "runs/p13r/diagnosis"


def rank_of(scores: dict[str, float], target: str) -> int:
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    return next((i + 1 for i, item in enumerate(ordered) if item == target), len(ordered) + 1)


def metrics(ranks: list[int], total: int = 0) -> dict[str, float]:
    return {
        "queries": float(len(ranks)),
        "recall@10": sum(1 for rank in ranks if rank <= 10) / max(1, len(ranks)),
        "recall@50": sum(1 for rank in ranks if rank <= 50) / max(1, len(ranks)),
        "recall@100": sum(1 for rank in ranks if rank <= 100) / max(1, len(ranks)),
        "mrr@10": sum(1.0 / rank for rank in ranks if rank <= 10) / max(1, len(ranks)),
        "ndcg@10": sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10) / max(1, len(ranks)),
        "mean_target_rank": sum(ranks) / max(1, len(ranks)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary: dict = {
        "schema_version": "agentrecx.p13r.diagnosis.v1",
        "test_retrieved": False,
        "scope": "observed-positive rank proxy; one positive per query; unjudged products are not judged negatives",
        "splits": {},
    }
    for split in ("train", "dev"):
        subset = [row for row in rows if row["split"] == split]
        pool_subset = [row for row in subset if row["target_in_pool"]]
        source_metrics = {}
        unique = Counter()
        for source in ("bm25", "bge"):
            ranks = [row["source"][source]["rank"] for row in subset if row["source"][source]["rank"] is not None]
            source_metrics[source] = {
                "hit@10": sum(1 for rank in ranks if rank <= 10),
                "hit@50": sum(1 for rank in ranks if rank <= 50),
                "hit@100": sum(1 for rank in ranks if rank <= 100),
                "queries": len(subset),
                "available_rank_queries": len(ranks),
                "mrr@10": sum(1.0 / rank for rank in ranks if rank <= 10) / max(1, len(subset)),
                "ndcg@10": sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10) / max(1, len(subset)),
            }
        for row in subset:
            hits = [source for source in ("bm25", "bge") if row["source"][source]["rank"] is not None]
            if len(hits) == 1:
                unique[hits[0]] += 1
        query_ranks: list[int] = []
        history_ranks: list[int] = []
        changed = 0
        for row in pool_subset:
            target = row["target_parent_asin"]
            query_scores = {}
            history_scores = {}
            for candidate in row["pool"]:
                bm25_rank = candidate.get("bm25_rank")
                bge_rank = candidate.get("bge_rank")
                query_score = 1.0 / (60 + (bm25_rank if bm25_rank else 1001))
                query_score += 1.0 / (60 + (bge_rank if bge_rank else 1001))
                history_score = query_score
                history_score += 0.25 * float(candidate.get("history_category_match") or 0.0)
                history_score += 0.25 * float(candidate.get("history_text_overlap") or 0.0)
                history_score += 0.05 * math.log1p(float(candidate.get("history_time_decay") or 0.0))
                query_scores[candidate["parent_asin"]] = query_score
                history_scores[candidate["parent_asin"]] = history_score
            q_rank = rank_of(query_scores, target)
            h_rank = rank_of(history_scores, target)
            query_ranks.append(q_rank)
            history_ranks.append(h_rank)
            changed += q_rank != h_rank
        summary["splits"][split] = {
            "queries": len(subset),
            "target_in_pool@100": len(pool_subset),
            "target_in_pool_rate": len(pool_subset) / max(1, len(subset)),
            "source": source_metrics,
            "unique_source_hits": dict(unique),
            "query_only_fixed_ranking": metrics(query_ranks),
            "history_fixed_ranking": metrics(history_ranks),
            "rank_changes_by_history": changed,
            "history_feature_coverage": {
                "tasks_with_history": sum(1 for row in subset if row["history_length"] > 0),
                "tasks_with_history_text": sum(1 for row in subset if row["history_text_items"] > 0),
                "total_history_items": sum(row["history_length"] for row in subset),
                "history_items_with_text": sum(row["history_text_items"] for row in subset),
            },
        }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
