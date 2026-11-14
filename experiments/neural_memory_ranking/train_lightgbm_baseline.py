"""Expanded-data LightGBM baselines on the common next-item candidate protocol.

This is a pointwise rating model used as a ranking baseline: candidates are ranked by
predicted observed rating.  It is trained on the same expanded training targets and
evaluated on exactly the same 1+99 candidate arrays as the neural models.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("/root/autodl-tmp/neural_memory_v1")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/neural_memory_v1")
PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
    force_col_wise=True,
)


def ranking_metrics(scores: np.ndarray, positive_positions: np.ndarray, k: int = 10) -> dict:
    positive_scores = scores[np.arange(len(scores)), positive_positions]
    ranks = 1 + (scores > positive_scores[:, None]).sum(axis=1)
    return {
        "n": int(len(scores)),
        "hit@10": round(float((ranks <= k).mean()), 6),
        "ndcg@10": round(float(np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0).mean()), 6),
        "mrr@10": round(float((1.0 / ranks).mean()), 6),
        "mean_rank": round(float(ranks.mean()), 6),
        "candidates": int(scores.shape[1]),
    }


def _load() -> dict:
    item_features = np.load(DATA / "item_features.npy").astype(np.float32)
    item_names = json.loads((DATA / "item_feature_names.json").read_text())
    memory_names = json.loads((DATA / "memory_feature_names.json").read_text())
    attrs = ["over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth",
             "microphone", "rechargeable", "noise_cancelling", "waterproof", "foldable",
             "portable", "gaming", "mechanical", "silent", "ergonomic", "touch", "rgb",
             "led", "usb", "hdmi", "4k", "remote"]
    attr_idx = [item_names.index(f"meta_attr_{attr}") for attr in attrs]
    high_idx = [memory_names.index(f"h_high_attr_{attr}") for attr in attrs]
    low_idx = [memory_names.index(f"h_low_attr_{attr}") for attr in attrs]
    train = pd.read_parquet(DATA / "train.parquet")
    val = pd.read_parquet(DATA / "val.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    return {
        "item_features": item_features,
        "item_names": item_names,
        "memory_names": memory_names,
        "attr_idx": attr_idx,
        "high_idx": high_idx,
        "low_idx": low_idx,
        "train": train,
        "val": val,
        "test": test,
        "memory_train": np.load(DATA / "memory_train.npy"),
        "memory_val": np.load(DATA / "memory_val.npy"),
        "memory_test": np.load(DATA / "memory_test.npy"),
        "common": np.load(DATA / "test_common_indices.npy"),
        "test_candidates": np.load(DATA / f"test_candidates{os.environ.get('CANDIDATE_SUFFIX', '')}.npy"),
        "test_positive": np.load(DATA / f"test_positive_position{os.environ.get('CANDIDATE_SUFFIX', '')}.npy").astype(np.int64),
        "pairs": pd.read_parquet(DATA / "pairs.parquet"),
    }


def _interactions(item_features: np.ndarray, memory: np.ndarray, data: dict) -> np.ndarray:
    attr = item_features[:, data["attr_idx"]]
    high = memory[:, data["high_idx"]] / np.maximum(memory[:, :1], 1.0)
    low = memory[:, data["low_idx"]] / np.maximum(memory[:, :1], 1.0)
    return np.concatenate([attr * high, attr * low], axis=1).astype(np.float32)


def train_model(data: dict, memory_on: bool):
    item_features = data["item_features"]
    train = data["train"]
    val = data["val"]
    target_item = train["target_item_index"].to_numpy(dtype=np.int64)
    val_item = val["target_item_index"].to_numpy(dtype=np.int64)
    if memory_on:
        x_train = np.concatenate(
            [data["memory_train"], item_features[target_item], _interactions(item_features[target_item], data["memory_train"], data)],
            axis=1,
        )
        x_val = np.concatenate(
            [data["memory_val"], item_features[val_item], _interactions(item_features[val_item], data["memory_val"], data)],
            axis=1,
        )
    else:
        x_train = item_features[target_item]
        x_val = item_features[val_item]
    model = lgb.LGBMRegressor(objective="regression", metric="rmse", **PARAMS)
    model.fit(
        x_train,
        train["target_rating"].to_numpy(),
        eval_set=[(x_val, val["target_rating"].to_numpy())],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    path = OUT / ("expanded_lightgbm_on.txt" if memory_on else "expanded_lightgbm_off.txt")
    model.booster_.save_model(str(path))
    return model, x_train.shape[1], val_item


def predict_candidates(model, data, indices, candidate_array, memory_on: bool, chunk=1000):
    item_features = data["item_features"]
    all_scores = []
    memory = data["memory_test"]
    for start in range(0, len(indices), chunk):
        stop = min(start + chunk, len(indices))
        row_idx = indices[start:stop]
        candidates = candidate_array[start:stop]
        flat_candidates = candidates.reshape(-1)
        item = item_features[flat_candidates]
        if memory_on:
            mem = np.repeat(memory[row_idx], candidates.shape[1], axis=0)
            x = np.concatenate([mem, item, _interactions(item, mem, data)], axis=1)
        else:
            x = item
        scores = model.predict(x).reshape(candidates.shape)
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)


def pair_metrics(model, data, memory_on: bool) -> dict:
    pairs = data["pairs"]
    groups = list(pairs.groupby("pair_group_id", sort=False))
    row_indices = np.asarray([int(group.iloc[0]["val_row_index"]) for _, group in groups], dtype=np.int64)
    candidate_array = np.stack([group["candidate_item_index"].to_numpy(dtype=np.int64) for _, group in groups], axis=0)
    item = data["item_features"][candidate_array.reshape(-1)].reshape(len(groups), 2, -1)
    memory = data["memory_val"][row_indices]
    if memory_on:
        mem = np.repeat(memory[:, None, :], 2, axis=1)
        x = np.concatenate([mem, item, _interactions(item.reshape(-1, item.shape[-1]), mem.reshape(-1, mem.shape[-1]), data).reshape(len(groups), 2, -1)], axis=2)
    else:
        x = item
    scores = model.predict(x.reshape(-1, x.shape[-1])).reshape(len(groups), 2)
    pairwise = []
    high_low = []
    ndcg = []
    mrr = []
    for row_scores, (_, group) in zip(scores, groups):
        ratings = group["candidate_rating"].to_numpy(dtype=float)
        if ratings[0] == ratings[1]:
            continue
        predicted = int(np.argmax(row_scores))
        true = int(np.argmax(ratings))
        pairwise.append(1.0 if predicted == true else 0.0)
        if (ratings >= 4).any() and (ratings <= 2).any():
            high_low.append(1.0 if predicted == true else 0.0)
        order = np.argsort(-row_scores)
        ideal = np.sort(ratings)[::-1]
        ndcg.append((ratings[order[0]] + ratings[order[1]] / math.log2(3)) / (ideal[0] + ideal[1] / math.log2(3)))
        relevant = np.where(ratings >= 4)[0]
        if len(relevant):
            first = next(i + 1 for i, candidate in enumerate(order) if candidate in set(relevant))
            mrr.append(1.0 / first)
    return {
        "pairwise_ordering_accuracy": round(float(np.mean(pairwise)), 6) if pairwise else None,
        "n_pairs_rating_different": len(pairwise),
        "high_vs_low_pairwise_accuracy": round(float(np.mean(high_low)), 6) if high_low else None,
        "n_high_vs_low_pairs": len(high_low),
        "ndcg_at_2_graded": round(float(np.mean(ndcg)), 6) if ndcg else None,
        "mrr_high": round(float(np.mean(mrr)), 6) if mrr else None,
    }


def main() -> int:
    data = _load()
    common = data["common"]
    candidates = data["test_candidates"]
    positive = data["test_positive"]
    results = {}
    run_suffix = os.environ.get("RUN_SUFFIX", "")
    for memory_on in (False, True):
        name = "memory_on" if memory_on else "memory_off"
        model, features, _ = train_model(data, memory_on)
        scores = predict_candidates(model, data, common, candidates[common], memory_on)
        common_metrics = ranking_metrics(scores, positive[common])
        pair = pair_metrics(model, data, memory_on)
        # full test popularity baseline and LightGBM common only (full feature matrix avoided)
        results[name] = {
            "features": int(features),
            "test_common_14157": common_metrics,
            "pairwise": pair,
        }
        print(json.dumps({name: results[name]}, indent=1), flush=True)

    # Popularity baseline on common and full candidate sets.
    item_names = data["item_names"]
    pop_index = item_names.index("pop_count_log")
    popularity = data["item_features"][:, pop_index]
    for scope, row_indices in (("test_common_14157", common), ("test_full_77029", np.arange(len(candidates)))):
        scores = popularity[candidates[row_indices]]
        results[f"popularity_{scope}"] = ranking_metrics(scores, positive[row_indices])
    # Popularity pairwise
    pairs = data["pairs"]
    groups = list(pairs.groupby("pair_group_id", sort=False))
    pairwise = []
    high_low = []
    for _, group in groups:
        ratings = group["candidate_rating"].to_numpy(dtype=float)
        if ratings[0] == ratings[1]:
            continue
        item_scores = popularity[group["candidate_item_index"].to_numpy(dtype=np.int64)]
        predicted = int(np.argmax(item_scores))
        true = int(np.argmax(ratings))
        pairwise.append(1.0 if predicted == true else 0.0)
        if (ratings >= 4).any() and (ratings <= 2).any():
            high_low.append(1.0 if predicted == true else 0.0)
    results["popularity_pairwise"] = {
        "pairwise_ordering_accuracy": round(float(np.mean(pairwise)), 6) if pairwise else None,
        "n_pairs_rating_different": len(pairwise),
        "high_vs_low_pairwise_accuracy": round(float(np.mean(high_low)), 6) if high_low else None,
        "n_high_vs_low_pairs": len(high_low),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "agentrecx.neural_memory_ranking.lightgbm_baseline.v1",
        "note": "Pointwise rating model ranked by predicted observed rating; same 1+99 candidate arrays as neural models. Common 14,157-user test subset for feature-heavy LightGBM.",
        "results": results,
    }
    (OUT / f"lightgbm_comparison{run_suffix}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
