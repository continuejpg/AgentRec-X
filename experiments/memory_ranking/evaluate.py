"""Evaluate the two Memory-Aware Ranking v1 models on the held-out test split.

This script produces:

* ``runs/memory_ranking_v1/comparison.json``  -- Memory OFF/ON metrics and deltas;
* ``runs/memory_ranking_v1/feature_importance.json`` -- gain importance;
* ``runs/memory_ranking_v1/ranking_group_metrics.json`` -- observed-pair ranking.

The test split is the last observed catalog rating of every user.  All reported
ranking metrics use only real rated items; no unobserved item is a negative.

Run from the repository root::

    python -m experiments.memory_ranking.evaluate
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import platform
import sys
from math import log2
from pathlib import Path
from typing import Any, Callable, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from experiments.memory_ranking import features

OUT = features.REPO / "runs/memory_ranking_v1"
DATA = features.OUT_DIR

HISTORY_BUCKETS = ("2", "3", "4", "5-9", "10+")
COVERAGE_BUCKETS = ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")


def _history_bucket(length: int) -> str:
    value = int(length)
    if value <= 2:
        return "2"
    if value == 3:
        return "3"
    if value == 4:
        return "4"
    if value <= 9:
        return "5-9"
    return "10+"


def _coverage_bucket(value: float) -> str:
    value = float(value)
    if value < 0.25:
        return "0.00-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def _regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "mean_actual": round(float(np.mean(y_true)), 6),
        "mean_predicted": round(float(np.mean(y_pred)), 6),
    }


def _classification(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y_true)) < 2:
        return {
            "n": int(len(y_true)),
            "positive_rate": round(float(np.mean(y_true)), 6),
            "roc_auc": None,
            "pr_auc": None,
            "log_loss": None,
            "accuracy": None,
            "confusion_matrix": None,
            "note": "single-class group; AUC/PR-AUC undefined",
        }
    labels = (probability >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
        "pr_auc": round(float(average_precision_score(y_true, probability)), 6),
        "log_loss": round(float(log_loss(y_true, probability)), 6),
        "accuracy": round(float(accuracy_score(y_true, labels)), 6),
        "confusion_matrix": [
            [int(value) for value in row]
            for row in confusion_matrix(y_true, labels, labels=[0, 1])
        ],
    }


def _delta(off: Mapping[str, Any], on: Mapping[str, Any], key: str, *, higher_is_better: bool) -> float | None:
    left = off.get(key)
    right = on.get(key)
    if left is None or right is None:
        return None
    value = float(right) - float(left)
    return round(value, 6)


def _grouped(
    frame: pd.DataFrame,
    bucket_column: str,
    *,
    metric_kind: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bucket in (HISTORY_BUCKETS if bucket_column == "history_bucket" else COVERAGE_BUCKETS):
        subset = frame[frame[bucket_column] == bucket]
        if subset.empty:
            continue
        if metric_kind == "regression":
            off = _regression(subset["target_rating"].to_numpy(), subset["memory_off_rating"].to_numpy())
            on = _regression(subset["target_rating"].to_numpy(), subset["memory_on_rating"].to_numpy())
            output[bucket] = {
                "n": int(len(subset)),
                "memory_off": off,
                "memory_on": on,
                "delta_mae_memory_off_minus_on": round(off["mae"] - on["mae"], 6),
                "delta_rmse_memory_off_minus_on": round(off["rmse"] - on["rmse"], 6),
            }
        else:
            binary = subset[subset["target_high"].notna()]
            if binary.empty:
                continue
            off = _classification(
                binary["target_high"].to_numpy(dtype=int),
                binary["memory_off_high_prob"].to_numpy(dtype=float),
            )
            on = _classification(
                binary["target_high"].to_numpy(dtype=int),
                binary["memory_on_high_prob"].to_numpy(dtype=float),
            )
            output[bucket] = {
                "n": int(len(binary)),
                "memory_off": off,
                "memory_on": on,
                "delta_roc_auc": _delta(off, on, "roc_auc", higher_is_better=True),
                "delta_pr_auc": _delta(off, on, "pr_auc", higher_is_better=True),
                "delta_log_loss": _delta(off, on, "log_loss", higher_is_better=False),
                "delta_accuracy": _delta(off, on, "accuracy", higher_is_better=True),
            }
    return output


def _bootstrap_metric_delta(
    y_true: np.ndarray,
    off_pred: np.ndarray,
    on_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        off = metric(y_true[indices], off_pred[indices])
        on = metric(y_true[indices], on_pred[indices])
        values.append(float(on - off))
    array = np.asarray(values, dtype=float)
    return {
        "metric": metric.__name__,
        "delta_memory_on_minus_off_mean": round(float(array.mean()), 6),
        "ci95_low": round(float(np.quantile(array, 0.025)), 6),
        "ci95_high": round(float(np.quantile(array, 0.975)), 6),
        "n_boot": n_boot,
        "note": "paired row bootstrap; positive delta means Memory ON is better for the metric as defined",
    }


def _bootstrap_auc_delta(
    y_true: np.ndarray,
    off_prob: np.ndarray,
    on_prob: np.ndarray,
    *,
    n_boot: int = 300,
    seed: int = 43,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sample = y_true[indices]
        if len(np.unique(sample)) < 2:
            continue
        values.append(
            float(
                roc_auc_score(sample, on_prob[indices])
                - roc_auc_score(sample, off_prob[indices])
            )
        )
    array = np.asarray(values, dtype=float)
    return {
        "delta_memory_on_minus_off_mean": round(float(array.mean()), 6),
        "ci95_low": round(float(np.quantile(array, 0.025)), 6),
        "ci95_high": round(float(np.quantile(array, 0.975)), 6),
        "n_boot": len(values),
    }


def _load_booster(name: str) -> lgb.Booster:
    return lgb.Booster(model_file=str(OUT / name))


def _feature_importance() -> dict[str, Any]:
    off_features = features.non_memory_feature_names()
    on_features = off_features + features.memory_feature_names()
    output: dict[str, Any] = {
        "schema": "agentrecx.memory_ranking.feature_importance.v1",
        "importance_type": "gain",
        "models": {},
    }
    for model_name, feature_names in (
        ("memory_off_regression", off_features),
        ("memory_on_regression", on_features),
        ("memory_off_classifier", off_features),
        ("memory_on_classifier", on_features),
    ):
        filename = {
            "memory_off_regression": "memory_off_model.txt",
            "memory_on_regression": "memory_on_model.txt",
            "memory_off_classifier": "memory_off_classifier.txt",
            "memory_on_classifier": "memory_on_classifier.txt",
        }[model_name]
        booster = _load_booster(filename)
        gains = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        total = float(np.sum(gains))
        normalized = {
            name: round(float(value) / total, 8) if total > 0 else 0.0
            for name, value in zip(names, gains)
        }
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        memory_gain = sum(value for name, value in normalized.items() if name.startswith("mem_"))
        output["models"][model_name] = {
            "model_file": filename,
            "feature_count": len(feature_names),
            "memory_feature_gain_share": round(memory_gain, 6),
            "non_memory_feature_gain_share": round(1.0 - memory_gain, 6),
            "top30": [
                {"feature": name, "gain_share": value} for name, value in ranked[:30]
            ],
            "all_features": dict(ranked),
        }
    return output


def _np_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _np_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _ndcg_at_2(true_gains: np.ndarray, predicted_order: np.ndarray) -> float:
    ideal = np.sort(true_gains)[::-1]
    dcg = true_gains[predicted_order[0]] / log2(2) + true_gains[predicted_order[1]] / log2(3)
    idcg = ideal[0] / log2(2) + ideal[1] / log2(3)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _ranking_groups(pairs: pd.DataFrame) -> dict[str, Any]:
    off_booster = _load_booster("memory_off_model.txt")
    on_booster = _load_booster("memory_on_model.txt")
    off_features = off_booster.feature_name()
    on_features = on_booster.feature_name()
    pairs = pairs.copy()
    pairs["pred_off"] = off_booster.predict(pairs[off_features].astype(float))
    pairs["pred_on"] = on_booster.predict(pairs[on_features].astype(float))

    stats = {
        "groups": 0,
        "groups_with_rating_difference": 0,
        "groups_with_high_rating": 0,
        "group_sizes": {},
        "rating_pair_counts": {},
    }
    pairwise_off: list[float] = []
    pairwise_on: list[float] = []
    pairwise_high_low_off: list[float] = []
    pairwise_high_low_on: list[float] = []
    pairwise_high_not_off: list[float] = []
    pairwise_high_not_on: list[float] = []
    mrr_off: list[float] = []
    mrr_on: list[float] = []
    ndcg_off: list[float] = []
    ndcg_on: list[float] = []
    ndcg_binary_off: list[float] = []
    ndcg_binary_on: list[float] = []
    for group_id, group in pairs.groupby("pair_group_id", sort=False):
        group = group.reset_index(drop=True)
        size = len(group)
        stats["groups"] += 1
        stats["group_sizes"][str(size)] = stats["group_sizes"].get(str(size), 0) + 1
        ratings = group["target_rating"].to_numpy(dtype=float)
        key = "|".join(str(int(value)) for value in sorted(ratings))
        stats["rating_pair_counts"][key] = stats["rating_pair_counts"].get(key, 0) + 1
        if len(np.unique(ratings)) > 1:
            stats["groups_with_rating_difference"] += 1
        if (ratings >= 4).any():
            stats["groups_with_high_rating"] += 1
        high_indices = np.where(ratings >= 4)[0]
        low_indices = np.where(ratings <= 2)[0]
        not_high_indices = np.where(ratings < 4)[0]
        for prediction_column, pairwise, high_low, high_not, mrr, ndcg, ndcg_binary in (
            (
                "pred_off", pairwise_off, pairwise_high_low_off, pairwise_high_not_off,
                mrr_off, ndcg_off, ndcg_binary_off,
            ),
            (
                "pred_on", pairwise_on, pairwise_high_low_on, pairwise_high_not_on,
                mrr_on, ndcg_on, ndcg_binary_on,
            ),
        ):
            scores = group[prediction_column].to_numpy(dtype=float)
            order = np.argsort(-scores, kind="mergesort")
            if len(np.unique(ratings)) > 1:
                best = int(np.argmax(ratings))
                worst = int(np.argmin(ratings))
                if scores[best] > scores[worst]:
                    pairwise.append(1.0)
                elif scores[best] == scores[worst]:
                    pairwise.append(0.5)
                else:
                    pairwise.append(0.0)
            if len(high_indices) and len(low_indices):
                best_high = int(high_indices[np.argmax(ratings[high_indices])])
                worst_low = int(low_indices[np.argmin(ratings[low_indices])])
                if scores[best_high] > scores[worst_low]:
                    high_low.append(1.0)
                elif scores[best_high] == scores[worst_low]:
                    high_low.append(0.5)
                else:
                    high_low.append(0.0)
            if len(high_indices) and len(not_high_indices):
                best_high = int(high_indices[np.argmax(ratings[high_indices])])
                best_not_high = int(not_high_indices[np.argmax(ratings[not_high_indices])])
                if scores[best_high] > scores[best_not_high]:
                    high_not.append(1.0)
                elif scores[best_high] == scores[best_not_high]:
                    high_not.append(0.5)
                else:
                    high_not.append(0.0)
            relevant = np.where(ratings >= 4)[0]
            if len(relevant):
                first_relevant = next(
                    index + 1 for index, candidate in enumerate(order) if candidate in set(relevant)
                )
                mrr.append(1.0 / first_relevant)
            ndcg.append(_ndcg_at_2(ratings, order))
            ndcg_binary.append(_ndcg_at_2((ratings >= 4).astype(float), order))
    return {
        "schema": "agentrecx.memory_ranking.ranking_groups.v1",
        "note": (
            "ranking_pairs groups the last two real-rated items of users with >=4 "
            "interactions under one common history cutoff.  Group size is 2; only "
            "observed rated items are ranked.  nDCG@5/@10 is not meaningful for "
            "group size 2 and is not reported."
        ),
        "groups": stats["groups"],
        "candidates": int(len(pairs)),
        "group_size_distribution": stats["group_sizes"],
        "groups_with_rating_difference": stats["groups_with_rating_difference"],
        "groups_with_high_rating": stats["groups_with_high_rating"],
        "rating_pair_distribution": dict(sorted(stats["rating_pair_counts"].items())),
        "metrics": {
            "pairwise_ordering_accuracy": {
                "memory_off": round(float(np.mean(pairwise_off)), 6) if pairwise_off else None,
                "memory_on": round(float(np.mean(pairwise_on)), 6) if pairwise_on else None,
                "delta_on_minus_off": (
                    round(float(np.mean(pairwise_on) - np.mean(pairwise_off)), 6)
                    if pairwise_off and pairwise_on
                    else None
                ),
                "n_groups_with_different_ratings": len(pairwise_off),
            },
            "pairwise_ordering_accuracy_high_vs_low": {
                "memory_off": round(float(np.mean(pairwise_high_low_off)), 6)
                if pairwise_high_low_off
                else None,
                "memory_on": round(float(np.mean(pairwise_high_low_on)), 6)
                if pairwise_high_low_on
                else None,
                "delta_on_minus_off": (
                    round(float(np.mean(pairwise_high_low_on) - np.mean(pairwise_high_low_off)), 6)
                    if pairwise_high_low_off and pairwise_high_low_on
                    else None
                ),
                "n_groups": len(pairwise_high_low_off),
            },
            "pairwise_ordering_accuracy_high_vs_not_high": {
                "memory_off": round(float(np.mean(pairwise_high_not_off)), 6)
                if pairwise_high_not_off
                else None,
                "memory_on": round(float(np.mean(pairwise_high_not_on)), 6)
                if pairwise_high_not_on
                else None,
                "delta_on_minus_off": (
                    round(float(np.mean(pairwise_high_not_on) - np.mean(pairwise_high_not_off)), 6)
                    if pairwise_high_not_off and pairwise_high_not_on
                    else None
                ),
                "n_groups": len(pairwise_high_not_off),
            },
            "mrr_high_relevance": {
                "memory_off": round(float(np.mean(mrr_off)), 6) if mrr_off else None,
                "memory_on": round(float(np.mean(mrr_on)), 6) if mrr_on else None,
                "delta_on_minus_off": (
                    round(float(np.mean(mrr_on) - np.mean(mrr_off)), 6)
                    if mrr_off and mrr_on
                    else None
                ),
                "n_groups_with_high_rating": len(mrr_off),
            },
            "ndcg_at_2_graded_rating_gain": {
                "memory_off": round(float(np.mean(ndcg_off)), 6),
                "memory_on": round(float(np.mean(ndcg_on)), 6),
                "delta_on_minus_off": round(float(np.mean(ndcg_on) - np.mean(ndcg_off)), 6),
                "n_groups": len(ndcg_off),
            },
            "ndcg_at_2_binary_high_relevance": {
                "memory_off": round(float(np.mean(ndcg_binary_off)), 6),
                "memory_on": round(float(np.mean(ndcg_binary_on)), 6),
                "delta_on_minus_off": round(
                    float(np.mean(ndcg_binary_on) - np.mean(ndcg_binary_off)), 6
                ),
                "n_groups": len(ndcg_binary_off),
            },
        },
        "metric_definitions": {
            "pairwise_ordering_accuracy": "fraction of pairs where the predicted order matches the true rating order; prediction ties count 0.5",
            "pairwise_ordering_accuracy_high_vs_low": "pairs with one rating>=4 and one rating<=2",
            "pairwise_ordering_accuracy_high_vs_not_high": "pairs with one rating>=4 and the other rating<4",
            "mrr_high_relevance": "reciprocal rank of the first candidate with true rating >=4",
            "ndcg_at_2_graded_rating_gain": "nDCG@2 with gain = observed 1-5 rating",
            "ndcg_at_2_binary_high_relevance": "nDCG@2 with gain = 1 for rating>=4, 0 otherwise",
        },
        "limitations": [
            "Only observed rated candidates are ranked; unobserved items are not negatives.",
            "Group size is 2, so nDCG@5/@10 is not a meaningful metric here.",
            "This is a secondary held-out pair benchmark, not a full-catalogue Top-K evaluation.",
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(DATA / "test.parquet").reset_index(drop=True)
    predictions = pd.read_parquet(OUT / "test_predictions.parquet").reset_index(drop=True)
    if len(test) != len(predictions):
        raise ValueError("test predictions are not aligned with the test split")
    frame = test.copy()
    frame["memory_off_rating"] = predictions["memory_off_rating"].to_numpy()
    frame["memory_on_rating"] = predictions["memory_on_rating"].to_numpy()
    frame["memory_off_high_prob"] = predictions["memory_off_high_prob"].to_numpy()
    frame["memory_on_high_prob"] = predictions["memory_on_high_prob"].to_numpy()
    frame["history_bucket"] = frame["history_length"].map(_history_bucket)
    frame["coverage_bucket"] = frame["target_attribute_coverage"].map(_coverage_bucket)
    train = pd.read_parquet(DATA / "train.parquet")
    category_means = train.groupby("target_main_category")["target_rating"].mean()
    global_mean = float(train["target_rating"].mean())
    frame["constant_mean_prediction"] = global_mean

    binary = frame[frame["target_high"].notna()]
    off_reg = _regression(frame["target_rating"].to_numpy(), frame["memory_off_rating"].to_numpy())
    on_reg = _regression(frame["target_rating"].to_numpy(), frame["memory_on_rating"].to_numpy())
    constant_reg = _regression(
        frame["target_rating"].to_numpy(), frame["constant_mean_prediction"].to_numpy()
    )
    off_clf = _classification(
        binary["target_high"].to_numpy(dtype=int),
        binary["memory_off_high_prob"].to_numpy(dtype=float),
    )
    on_clf = _classification(
        binary["target_high"].to_numpy(dtype=int),
        binary["memory_on_high_prob"].to_numpy(dtype=float),
    )
    comparison: dict[str, Any] = {
        "schema": "agentrecx.memory_ranking.comparison.v1",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "memory_ranking_v1",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "gpu_used": False,
        },
        "test_rows": len(frame),
        "test_binary_rows": len(binary),
        "test_neutral_rating_rows": int((frame["target_rating"] == 3).sum()),
        "baselines": {
            "constant_train_mean": {
                **constant_reg,
                "note": "predict the training-split global mean for every test row",
            },
            "majority_class_accuracy": round(float(max(np.mean(binary["target_high"]), 1 - np.mean(binary["target_high"]))), 6),
        },
        "regression": {
            "memory_off": off_reg,
            "memory_on": on_reg,
            "delta_memory_on_minus_off": {
                "mae": round(on_reg["mae"] - off_reg["mae"], 6),
                "rmse": round(on_reg["rmse"] - off_reg["rmse"], 6),
                "relative_mae_reduction": round(
                    (off_reg["mae"] - on_reg["mae"]) / off_reg["mae"], 6
                ),
                "relative_rmse_reduction": round(
                    (off_reg["rmse"] - on_reg["rmse"]) / off_reg["rmse"], 6
                ),
            },
            "bootstrap_delta": {
                "mae": _bootstrap_metric_delta(
                    frame["target_rating"].to_numpy(),
                    frame["memory_off_rating"].to_numpy(),
                    frame["memory_on_rating"].to_numpy(),
                    _np_mae,
                ),
                "rmse": _bootstrap_metric_delta(
                    frame["target_rating"].to_numpy(),
                    frame["memory_off_rating"].to_numpy(),
                    frame["memory_on_rating"].to_numpy(),
                    _np_rmse,
                ),
            },
            "by_history_length": _grouped(frame, "history_bucket", metric_kind="regression"),
            "by_target_attribute_coverage": _grouped(
                frame, "coverage_bucket", metric_kind="regression"
            ),
        },
        "classification": {
            "memory_off": off_clf,
            "memory_on": on_clf,
            "delta_memory_on_minus_off": {
                "roc_auc": _delta(off_clf, on_clf, "roc_auc", higher_is_better=True),
                "pr_auc": _delta(off_clf, on_clf, "pr_auc", higher_is_better=True),
                "log_loss": _delta(off_clf, on_clf, "log_loss", higher_is_better=False),
                "accuracy": _delta(off_clf, on_clf, "accuracy", higher_is_better=True),
            },
            "bootstrap_auc_delta": _bootstrap_auc_delta(
                binary["target_high"].to_numpy(dtype=int),
                binary["memory_off_high_prob"].to_numpy(dtype=float),
                binary["memory_on_high_prob"].to_numpy(dtype=float),
            ),
            "by_history_length": _grouped(frame, "history_bucket", metric_kind="classification"),
            "by_target_attribute_coverage": _grouped(
                frame, "coverage_bucket", metric_kind="classification"
            ),
        },
        "memory_gain_summary": {
            "regression_mae_memory_off": off_reg["mae"],
            "regression_mae_memory_on": on_reg["mae"],
            "regression_mae_absolute_reduction": round(off_reg["mae"] - on_reg["mae"], 6),
            "regression_rmse_absolute_reduction": round(off_reg["rmse"] - on_reg["rmse"], 6),
            "classification_roc_auc_memory_off": off_clf["roc_auc"],
            "classification_roc_auc_memory_on": on_clf["roc_auc"],
            "classification_roc_auc_absolute_delta": _delta(
                off_clf, on_clf, "roc_auc", higher_is_better=True
            ),
            "classification_pr_auc_absolute_delta": _delta(
                off_clf, on_clf, "pr_auc", higher_is_better=True
            ),
            "history_length_note": "test history length starts at 2 because users need at least 3 interactions",
        },
        "claims": {
            "pointwise_memory_gain_on_observed_rating": (
                "measured on the held-out chronological test split" if on_reg["mae"] < off_reg["mae"] else "not observed"
            ),
            "full_topk_recommendation_gain": "not established",
            "real_user_satisfaction": "not established",
            "unobserved_items_policy": "UNOBSERVED/UNJUDGED, never negative",
            "label_status": "observed Amazon rating, not human pairwise relevance",
        },
    }

    ranking_groups = _ranking_groups(pd.read_parquet(DATA / "ranking_pairs.parquet"))
    comparison["ranking_groups"] = ranking_groups
    feature_importance = _feature_importance()
    comparison["feature_importance_summary"] = {
        model: {
            "memory_feature_gain_share": value["memory_feature_gain_share"],
            "non_memory_feature_gain_share": value["non_memory_feature_gain_share"],
            "top5": value["top30"][:5],
        }
        for model, value in feature_importance["models"].items()
    }
    comparison["dataset_manifest"] = json.loads(
        (DATA / "manifest.json").read_text(encoding="utf-8")
    )

    (OUT / "comparison.json").write_text(
        json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "feature_importance.json").write_text(
        json.dumps(feature_importance, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "ranking_group_metrics.json").write_text(
        json.dumps(ranking_groups, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "regression": comparison["regression"]["delta_memory_on_minus_off"],
        "classification": comparison["classification"]["delta_memory_on_minus_off"],
        "ranking_groups": ranking_groups["metrics"],
        "memory_gain_summary": comparison["memory_gain_summary"],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
