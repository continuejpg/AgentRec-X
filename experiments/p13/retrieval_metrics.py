"""P13 retrieval benchmark metrics from fixed per-source ranks.

Only observed-positive ranks are available; these are hit rates / rank proxies, not full
relevance recall.  The reserved test split is reported but not used for model selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "runs/p13/retrieval/source_target_ranks.jsonl"
POOL = ROOT / "runs/p13/retrieval/queries.jsonl"
OUT = ROOT / "runs/p13/retrieval"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary: dict = {
        "schema_version": "agentrecx.p13.retrieval_metrics.v1",
        "scope": "observed-positive hit/rank proxy; one positive per query; all other products unjudged",
        "splits": {},
        "fusion_fixed_pool": {},
    }
    for split in ("train", "dev", "test"):
        subset = [row for row in rows if row["split"] == split]
        source_metrics = {}
        for source in ("bm25", "bge", "blair"):
            ranks = [row[source]["rank"] for row in subset if row[source]["rank"] is not None]
            source_metrics[source] = {
                "queries": len(subset),
                "hit@10": sum(1 for rank in ranks if rank <= 10),
                "hit@50": sum(1 for rank in ranks if rank <= 50),
                "hit@100": sum(1 for rank in ranks if rank <= 100),
                "mrr@10": sum(1.0 / rank for rank in ranks if rank <= 10) / max(1, len(subset)),
                "ndcg@10": sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
                / max(1, len(subset)),
                "hit_queries": len(ranks),
            }
        # Any-source and fixed-pool reachability.
        any_hit = 0
        unique = Counter()
        for row in subset:
            hits = [source for source in ("bm25", "bge", "blair") if row[source]["rank"] is not None]
            if hits:
                any_hit += 1
            if len(hits) == 1:
                unique[hits[0]] += 1
        summary["splits"][split] = {
            "queries": len(subset),
            "any_source_hit@100": any_hit,
            "unique_source_hit_queries": dict(unique),
            "source": source_metrics,
        }
    pool_rows = [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines() if line.strip()]
    for split in ("train", "dev", "test"):
        subset = [row for row in pool_rows if row["split"] == split]
        target_in_pool = sum(1 for row in subset if row["pool_target_rank_by_source_fusion"] is not None)
        summary["fusion_fixed_pool"][split] = {
            "queries": len(subset),
            "target_in_fixed_pool@100": target_in_pool,
            "fixed_pool_quota": "BM25 50 + BGE 25 + BLaIR 25, deduplicated, max 100",
            "note": "fixed-budget fusion reachability; source ranks above are untruncated top-100",
        }
    (args.out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
