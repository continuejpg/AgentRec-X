"""P13 fixed-pool ranking and history ablations.

The candidate pool is fixed by retrieval.py.  Every query has one observed positive and
all other pool items are unjudged/weak negatives.  LambdaMART training is therefore
explicitly weak-supervision, not feedback-trained personalization.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES = ROOT / "runs/p13/retrieval/queries.jsonl"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
OUT = ROOT / "runs/p13/ranking"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text or "").casefold()))


def load_products() -> dict[str, Any]:
    records: dict[str, Any] = {}
    with PRODUCTS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            asin = row.get("parent_asin")
            if not asin:
                continue
            records[str(asin)] = {
                "title": str(row.get("title") or ""),
                "categories": tuple(str(v) for v in (row.get("categories") or ())),
            }
    return records


def candidate_features(
    row: dict,
    products: dict[str, Any],
    history_variant: str,
) -> np.ndarray:
    query_tokens = tokens(row["query"])
    history_features = row["history_features"][history_variant]
    values = []
    for candidate in row["pool"]:
        asin = candidate["parent_asin"]
        product = products.get(asin, {"title": "", "categories": ()})
        title_tokens = tokens(product["title"])
        overlap = query_tokens & title_tokens
        category_text = " ".join(product["categories"])
        category_tokens = tokens(category_text)
        h = history_features.get(asin, {})
        values.append(
            [
                float(candidate.get("bm25_rank") or 1001),
                float(candidate.get("bm25_score") or 0.0),
                float(candidate.get("bge_rank") or 1001),
                float(candidate.get("bge_score") or 0.0),
                float(candidate.get("blair_rank") or 1001),
                float(candidate.get("blair_score") or 0.0),
                1.0 if candidate.get("bm25_rank") is None else 0.0,
                1.0 if candidate.get("bge_rank") is None else 0.0,
                1.0 if candidate.get("blair_rank") is None else 0.0,
                len(query_tokens),
                len(title_tokens),
                len(overlap),
                len(overlap) / max(1, len(query_tokens)),
                len(category_tokens & query_tokens),
                h.get("history_len", 0.0),
                h.get("history_in_catalog", 0.0),
                h.get("history_category_match", 0.0),
                h.get("history_category_fraction", 0.0),
                h.get("history_time_decay_match", 0.0),
                h.get("history_bge_max", 0.0),
                h.get("history_bge_mean", 0.0),
                h.get("history_blair_max", 0.0),
                h.get("history_blair_mean", 0.0),
            ]
        )
    return np.asarray(values, dtype=np.float32)


FEATURE_NAMES = [
    "bm25_rank", "bm25_score", "bge_rank", "bge_score", "blair_rank", "blair_score",
    "bm25_missing", "bge_missing", "blair_missing", "query_len", "title_len",
    "title_overlap", "title_overlap_ratio", "category_query_overlap",
    "history_len", "history_in_catalog", "history_category_match",
    "history_category_fraction", "history_time_decay_match",
    "history_bge_max", "history_bge_mean", "history_blair_max", "history_blair_mean",
]
QUERY_FEATURES = list(range(0, 14))
HISTORY_FEATURES = list(range(14, len(FEATURE_NAMES)))


def metric_rows(scores: np.ndarray, target_index: int) -> dict[str, Any]:
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    rank = next(
        (position + 1 for position, index in enumerate(order.tolist()) if index == target_index),
        len(scores) + 1,
    )
    return {
        "rank": rank,
        "recall@10": 1.0 if rank <= 10 else 0.0,
        "recall@50": 1.0 if rank <= 50 else 0.0,
        "recall@100": 1.0 if rank <= 100 else 0.0,
        "ndcg@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "mrr@10": 1.0 / rank if rank <= 10 else 0.0,
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {
        "queries": float(len(rows)),
        "recall@10": float(np.mean([row["recall@10"] for row in rows])) if rows else 0.0,
        "recall@50": float(np.mean([row["recall@50"] for row in rows])) if rows else 0.0,
        "recall@100": float(np.mean([row["recall@100"] for row in rows])) if rows else 0.0,
        "ndcg@10": float(np.mean([row["ndcg@10"] for row in rows])) if rows else 0.0,
        "mrr@10": float(np.mean([row["mrr@10"] for row in rows])) if rows else 0.0,
        "mean_target_rank": float(np.mean([row["rank"] for row in rows])) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--product", type=Path, default=PRODUCTS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    products = load_products()
    train = [row for row in rows if row["split"] == "train" and row["pool_target_rank_by_source_fusion"] is not None]
    dev = [
        row for row in rows
        if row["split"] == "dev" and row["pool_target_rank_by_source_fusion"] is not None
    ]  # reserved test intentionally excluded
    if not train or not dev:
        raise SystemExit("need train and dev queries with targets in candidate pool")
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm is required for P13 ranking") from exc

    def build(rows_subset: list[dict], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        groups: list[int] = []
        target_indices: list[int] = []
        for row in rows_subset:
            matrix = candidate_features(row, products, variant)
            target_index = next(
                index for index, candidate in enumerate(row["pool"])
                if candidate["label_observed_positive"]
            )
            y = np.zeros(len(matrix), dtype=np.int32)
            y[target_index] = 1
            features.append(matrix)
            labels.append(y)
            groups.append(len(matrix))
            target_indices.append(target_index)
        return np.vstack(features), np.concatenate(labels), np.asarray(groups), target_indices

    def train_ranker(variant: str, history: bool) -> Any:
        x, y, group, _ = build(train, variant)
        if not history:
            x = x[:, QUERY_FEATURES]
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            random_state=2026,
            verbosity=-1,
        )
        ranker.fit(x, y, group=group)
        return ranker

    models = {
        "query_only": (train_ranker("true", False), False),
        "history_true": (train_ranker("true", True), True),
        "history_unrelated": (train_ranker("unrelated", True), True),
        "history_removed": (train_ranker("removed", True), True),
    }
    result_names = (
        "bm25_order", "bge_order", "blair_order", "pool_order",
        "query_only", "history_true", "history_true_unrelated_features",
        "history_true_removed_features", "history_unrelated_model", "history_removed_model",
    )
    results: dict[str, list[dict]] = {name: [] for name in result_names}
    for row in dev:
        # Retrieval-order baselines.
        for source in ("bm25", "bge", "blair"):
            ranking = sorted(
                range(len(row["pool"])),
                key=lambda i: (
                    row["pool"][i].get(f"{source}_rank") is None,
                    row["pool"][i].get(f"{source}_rank") or 10_000,
                    row["pool"][i]["parent_asin"],
                ),
            )
            target_index = next(i for i, c in enumerate(row["pool"]) if c["label_observed_positive"])
            rank = next(i + 1 for i, index in enumerate(ranking) if index == target_index)
            results[f"{source}_order"].append({
                "rank": rank,
                "recall@10": 1.0 if rank <= 10 else 0.0,
                "recall@50": 1.0 if rank <= 50 else 0.0,
                "recall@100": 1.0 if rank <= 100 else 0.0,
                "ndcg@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
                "mrr@10": 1.0 / rank if rank <= 10 else 0.0,
            })
        target_index = next(i for i, c in enumerate(row["pool"]) if c["label_observed_positive"])
        results["pool_order"].append(metric_rows(np.arange(len(row["pool"]), 0, -1), target_index))
        for name, (model, uses_history) in models.items():
            if name == "query_only":
                variants = [("query_only", "true")]
            elif name == "history_true":
                variants = [
                    ("history_true", "true"),
                    ("history_true_unrelated_features", "unrelated"),
                    ("history_true_removed_features", "removed"),
                ]
            elif name == "history_unrelated":
                variants = [("history_unrelated_model", "unrelated")]
            else:
                variants = [("history_removed_model", "removed")]
            for result_name, variant in variants:
                matrix = candidate_features(row, products, variant)
                if not uses_history:
                    matrix = matrix[:, QUERY_FEATURES]
                scores = model.predict(matrix)
                results[result_name].append(metric_rows(scores, target_index))
    summary = {
        "schema_version": "agentrecx.p13.ranking.v1",
        "train_queries": len(train),
        "dev_queries": len(dev),
        "reserved_test_used": False,
        "label_scope": (
            "one observed positive per query; all other pool candidates are unjudged and "
            "used as weak negatives only; this is not feedback-trained personalization"
        ),
        "feature_names": FEATURE_NAMES,
        "query_only_features": [FEATURE_NAMES[i] for i in QUERY_FEATURES],
        "history_features": [FEATURE_NAMES[i] for i in HISTORY_FEATURES],
        "ranking": {name: aggregate(rows) for name, rows in results.items()},
        "history_ablation": {
            "query_only": aggregate(results["query_only"]),
            "history_true": aggregate(results["history_true"]),
            "history_true_with_unrelated_features": aggregate(
                results["history_true_unrelated_features"]
            ),
            "history_true_with_removed_features": aggregate(
                results["history_true_removed_features"]
            ),
            "history_trained_unrelated": aggregate(results["history_unrelated_model"]),
            "history_trained_removed": aggregate(results["history_removed_model"]),
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in dev:
            handle.write(json.dumps({"sequence_id": row["sequence_id"], "qid": row["qid"]}) + "\n")
    for name, (model, _) in models.items():
        model.booster_.save_model(str(args.out / f"{name}.txt"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
