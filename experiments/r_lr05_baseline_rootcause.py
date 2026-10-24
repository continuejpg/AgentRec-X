"""Phase 2C: reproduce the RRF baseline discrepancy (3/766 vs 30/766).

The LightGBM training scripts evaluated the RRF baseline by sorting candidates on
``1 / (60 + rrf_score)``.  Because that transform is strictly decreasing in the RRF score, and
``rank_metrics`` sorts primarily by descending ``score``, the baseline was ranked in **reverse
RRF order**.  This script rebuilds the exact 766-group test split from the frozen materialisation
and prints the three orderings side by side:

    correct RRF          order by rrf_score desc (then rrf, then item_id)
    training-script RRF  order by 1/(60 + rrf_score) desc (the inverted baseline)
    frozen LR            order by model score desc (22 trees), then rrf, then item_id

Expected: correct RRF HR@20 = 30/766, training-script RRF HR@20 = 3/766, LR HR@20 = 21/766,
matching runs/lr_v1_formal/corrected_test_comparison.json and metrics.json.

    .venv/bin/python -m experiments.r_lr05_baseline_rootcause
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "runs" / "lr_v1_r2" / "rlr01_groups.jsonl"
MODEL = REPO / "runs" / "lr_v1_formal" / "model.txt"
K_GROUPS = 5000
BEST_ITERATION = 22
FEATURES = [
    "seq_mask", "seq_rank", "seq_inv_rank", "seq_score",
    "tt_mask", "tt_rank", "tt_inv_rank", "tt_score",
    "lex_mask", "lex_rank", "lex_inv_rank", "lex_score",
    "rrf_score", "rrf_rank", "source_count",
]


def main() -> int:
    X, y, item, rrf, sizes, cases = [], [], [], [], [], []
    with SRC.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["query_id"]) >= K_GROUPS:
                continue
            X.append([float(row[name]) for name in FEATURES])
            y.append(int(row["label"]))
            item.append(int(row["item_id"]))
            rrf.append(float(row["rrf_score"]))
            if not cases or cases[-1] != int(row["case"]):
                cases.append(int(row["case"]))
                sizes.append(0)
            sizes[-1] += 1
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    item = np.asarray(item, dtype=np.int64)
    rrf = np.asarray(rrf, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.int32)
    cases = np.asarray(cases, dtype=np.int64)

    bucket = np.asarray([
        int(hashlib.sha256(str(int(case)).encode()).hexdigest()[:8], 16) % 100
        for case in cases
    ])
    test_groups = np.where(bucket >= 85)[0]
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    test_starts = starts[test_groups]
    test_lengths = sizes[test_groups]

    def metrics(score: np.ndarray, label: str) -> None:
        hits = {k: 0 for k in (5, 10, 20)}
        ndcg = {k: 0.0 for k in (5, 10, 20)}
        positives = 0
        for start, length in zip(test_starts, test_lengths):
            a, b = int(start), int(start + length)
            order = np.lexsort((item[a:b], -rrf[a:b], -score[a:b]))
            labels = y[a:b]
            if not labels.any():
                continue
            positives += 1
            rank = int(np.where(labels[order] == 1)[0][0]) + 1
            for k in hits:
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)
        n = len(test_starts)
        print(
            f"{label}: n={n} positives={positives} "
            f"HR@5={hits[5]}/{n} HR@10={hits[10]}/{n} HR@20={hits[20]}/{n} "
            f"NDCG@5={ndcg[5] / n:.6f} NDCG@20={ndcg[20] / n:.6f}"
        )

    print("groups", len(sizes), "rows", len(y), "test groups", len(test_groups))
    metrics(rrf.copy(), "correct RRF (score = rrf_score)")
    metrics(np.where(rrf > 0, 1.0 / (60.0 + rrf), 0.0),
            "training-script RRF (score = 1/(60+rrf_score))")
    booster = lgb.Booster(model_file=str(MODEL))
    metrics(booster.predict(X, num_iteration=BEST_ITERATION), "frozen LR (22 trees)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
