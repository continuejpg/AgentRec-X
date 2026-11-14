"""Train the two pointwise CPU baselines for Memory-Aware Ranking v1.

Model A (Memory OFF) uses item metadata + training-split popularity only.
Model B (Memory ON) uses exactly the same features plus features inferred from the
user's earlier ratings.  Both models use the same rows, labels, splits and training
budget; no GPU, no neural network, no LambdaMART.

Run from the repository root::

    python -m experiments.memory_ranking.train_baseline
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from experiments.memory_ranking import features

OUT = features.REPO / "runs/memory_ranking_v1"
DATA = features.OUT_DIR
RANDOM_STATE = 42
COMMON_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
    "verbosity": -1,
    "force_col_wise": True,
}


def _load(part: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{part}.parquet")


def _xy(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing[:5]}")
    return frame[feature_names].astype(float), frame["target_rating"].to_numpy(dtype=float)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "n": int(len(y_true)),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mean_actual": round(float(np.mean(y_true)), 6),
        "mean_predicted": round(float(np.mean(y_pred)), 6),
    }


def _classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )

    labels = (probabilities >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 6),
        "log_loss": round(float(log_loss(y_true, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_true, labels)), 6),
        "confusion_matrix": [
            [int(value) for value in row]
            for row in confusion_matrix(y_true, labels, labels=[0, 1])
        ],
    }


def train_regression(
    name: str, feature_names: list[str], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    X_train, y_train = _xy(train, feature_names)
    X_val, y_val = _xy(val, feature_names)
    X_test, y_test = _xy(test, feature_names)
    model = lgb.LGBMRegressor(objective="regression", metric="rmse", **COMMON_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    booster = model.booster_
    booster.save_model(str(OUT / f"{name}_model.txt"))
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    return {
        "model": f"{name}_model.txt",
        "feature_count": len(feature_names),
        "best_iteration": int(model.best_iteration_ or COMMON_PARAMS["n_estimators"]),
        "train": _regression_metrics(y_train, train_pred),
        "val": _regression_metrics(y_val, val_pred),
        "test": _regression_metrics(y_test, test_pred),
        "test_predictions": test_pred,
    }


def train_classifier(
    name: str, feature_names: list[str], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    train_bin = train[train["target_high"].notna()].copy()
    val_bin = val[val["target_high"].notna()].copy()
    test_bin = test[test["target_high"].notna()].copy()
    X_train, _ = _xy(train_bin, feature_names)
    X_val, _ = _xy(val_bin, feature_names)
    X_test, _ = _xy(test_bin, feature_names)
    y_train = train_bin["target_high"].to_numpy(dtype=int)
    y_val = val_bin["target_high"].to_numpy(dtype=int)
    y_test = test_bin["target_high"].to_numpy(dtype=int)
    model = lgb.LGBMClassifier(objective="binary", metric="auc", **COMMON_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    booster = model.booster_
    booster.save_model(str(OUT / f"{name}_classifier.txt"))
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": f"{name}_classifier.txt",
        "feature_count": len(feature_names),
        "best_iteration": int(model.best_iteration_ or COMMON_PARAMS["n_estimators"]),
        "train": _classification_metrics(y_train, train_prob),
        "val": _classification_metrics(y_val, val_prob),
        "test": _classification_metrics(y_test, test_prob),
        "test_probabilities": test_prob,
        "test_index": test_bin.index.to_numpy(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train = _load("train")
    val = _load("val")
    test = _load("test")
    off_features = features.non_memory_feature_names()
    on_features = off_features + features.memory_feature_names()
    forbidden = {
        "user_id", "target_item_id", "target_rating", "target_high",
        "history_item_ids", "history_ratings", "history_timestamps",
        "pair_group_id",
    }
    assert not (set(off_features) & forbidden)
    assert not (set(on_features) & forbidden)
    assert off_features == [name for name in on_features if not name.startswith("mem_")]

    regression_off = train_regression("memory_off", off_features, train, val, test)
    regression_on = train_regression("memory_on", on_features, train, val, test)
    classification_off = train_classifier("memory_off", off_features, train, val, test)
    classification_on = train_classifier("memory_on", on_features, train, val, test)

    # Persist test predictions as untracked local artifacts for evaluate.py; the
    # parquet files are not training inputs.
    prediction_frame = test[
        ["user_id", "target_item_id", "target_timestamp", "history_length", "target_rating", "target_high"]
    ].copy()
    prediction_frame["memory_off_rating"] = regression_off.pop("test_predictions")
    prediction_frame["memory_on_rating"] = regression_on.pop("test_predictions")
    prediction_frame = prediction_frame.reset_index(drop=True)
    prediction_frame["memory_off_high_prob"] = np.nan
    prediction_frame["memory_on_high_prob"] = np.nan
    prediction_frame.loc[
        classification_off.pop("test_index"), "memory_off_high_prob"
    ] = classification_off.pop("test_probabilities")
    prediction_frame.loc[
        classification_on.pop("test_index"), "memory_on_high_prob"
    ] = classification_on.pop("test_probabilities")
    prediction_frame.to_parquet(OUT / "test_predictions.parquet", index=False, compression="zstd")

    summary = {
        "schema": "agentrecx.memory_ranking.training_summary.v1",
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "memory_ranking_v1",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "gpu_used": False,
        },
        "training_params": COMMON_PARAMS,
        "rows": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_regression": len(train),
            "train_classification": int(train["target_high"].notna().sum()),
            "val_regression": len(val),
            "val_classification": int(val["target_high"].notna().sum()),
            "test_regression": len(test),
            "test_classification": int(test["target_high"].notna().sum()),
        },
        "feature_counts": {
            "memory_off": len(off_features),
            "memory_on": len(on_features),
            "memory_only": len(features.memory_feature_names()),
        },
        "regression": {
            "memory_off": regression_off,
            "memory_on": regression_on,
        },
        "classification": {
            "memory_off": classification_off,
            "memory_on": classification_on,
        },
        "notes": [
            "Both models use the same rows, labels, split and training budget.",
            "Memory OFF has 95 item/popularity features; Memory ON adds 118 inferred-from-ratings features.",
            "Classification excludes rating==3 from training/evaluation; regression keeps it.",
        ],
    }
    (OUT / "training_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
