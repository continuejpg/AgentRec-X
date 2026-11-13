"""Personalized preference benchmark baseline for Dataset B.

The only positive is the user's real next observed interaction.  All other catalogue items
are UNJUDGED; no sampled item is treated as a negative.  The baseline below is a
non-personalized full-catalogue popularity reference, explicitly labelled as such.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/gold/amazon_reviews_electronics_v1"
OUT = DATA / "benchmark_metrics.json"


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    train = _rows(DATA / "train.jsonl.gz")
    val = _rows(DATA / "val.jsonl.gz")
    test = _rows(DATA / "test.jsonl.gz")
    popularity: Counter[str] = Counter()
    for row in train + val:
        popularity[row["target_item_id"]] += 1
        for item in row["history"]:
            popularity[item["item_id"]] += 1
    ordered = [item for item, _ in sorted(popularity.items(), key=lambda pair: (-pair[1], pair[0]))]
    rank = {item: index + 1 for index, item in enumerate(ordered)}
    cold_rank = len(rank) + 1

    def metric(rows: list[dict]) -> dict:
        hits10 = hits50 = 0
        rr = []
        ndcg = []
        ratings = Counter()
        for row in rows:
            target = row["target_item_id"]
            ratings[str(row["rating"])] += 1
            r = rank.get(target, cold_rank)
            hits10 += int(r <= 10)
            hits50 += int(r <= 50)
            rr.append(1.0 / r if r <= 10 else 0.0)
            ndcg.append(1.0 / math.log2(r + 1) if r <= 10 else 0.0)
        n = len(rows)
        return {
            "tasks": n,
            "recall@10": round(hits10 / n, 6),
            "recall@50": round(hits50 / n, 6),
            "mrr@10": round(sum(rr) / n, 6),
            "ndcg@10": round(sum(ndcg) / n, 6),
            "target_rating_counts": dict(sorted(ratings.items())),
            "popularity_rank_coverage": round(sum(1 for row in rows if row["target_item_id"] in rank) / n, 6),
        }

    positive = [row for row in test if row["rating"] >= 4.0]
    payload = {
        "schema": "agentrecx.gold.amazon_reviews_electronics.benchmark.v1",
        "benchmark": "next observed interaction, full 76,918-item catalogue",
        "data": str(DATA),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "candidate_universe": "full current Electronics catalogue (76,918 items)",
        "label": "known observed next item; positive subset rating>=4 for sensitivity",
        "unjudged_policy": "all non-target items are unjudged, never negatives",
        "candidate_sampling": "none; rank over full catalogue; no random negatives",
        "baselines": {
            "non_personalized_popularity": {
                "all_test": metric(test),
                "rating_ge_4_test": metric(positive),
                "ranking_source": "item count over train+val targets and histories; no test data",
            },
            "random_expected_recall@10": round(10 / 76918, 6),
        },
        "limitations": [
            "one observed next item; no relevance labels for the other 76,917 items",
            "Recall@K is known-positive retrieval, not full relevance recall",
            "the files are Electronics.valid/test ratings, not an official train/val/test recommendation split",
            "no review text is present in the source files",
        ],
        "training_use": "not used for ranking training in this phase",
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
