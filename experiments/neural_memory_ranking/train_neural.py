"""Train and evaluate the Neural Memory OFF / ON rankers on AutoDL GPU.

Data are produced by ``build_sequences.py`` from full Electronics interactions.
Main objective: sampled next-observed-catalog-item ranking over 1 positive + 99
unobserved catalog candidates.  Unobserved candidates are SAMPLED, not confirmed
negative feedback.

Run remotely:

    cd /root/autodl-tmp/AgentRec-X-phase2
    /root/autodl-tmp/agentrecx-sft/bin/python -m experiments.neural_memory_ranking.train_neural
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.neural_memory_ranking.model import ModelConfig, NeuralMemoryRanker

DATA = Path("/root/autodl-tmp/neural_memory_v1")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/neural_memory_v1")
MS_PER_DAY = 86_400_000.0
MAX_HISTORY = 64
TRAIN_NEGATIVES = 32
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
RATING_LOSS_WEIGHT = 0.1
SEED = 42
CANDIDATE_SUFFIX = os.environ.get("CANDIDATE_SUFFIX", "")
RUN_SUFFIX = os.environ.get("RUN_SUFFIX", "")
ATTR_ABLATION = os.environ.get("ATTR_ABLATION", "0") == "1"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_split(split: str) -> dict[str, Any]:
    frame = pd.read_parquet(DATA / f"{split}.parquet")
    memory = np.load(DATA / f"memory_{split}.npy").astype(np.float32)
    n = len(frame)
    hist_idx = np.zeros((n, MAX_HISTORY), dtype=np.int32)
    hist_rating = np.zeros((n, MAX_HISTORY), dtype=np.float32)
    hist_time = np.zeros((n, MAX_HISTORY), dtype=np.int64)
    hist_len = np.zeros(n, dtype=np.int32)
    for position, (indices, ratings, timestamps) in enumerate(
        zip(frame["history_item_index"], frame["history_rating"], frame["history_timestamp"])
    ):
        values = list(indices)
        ratings_list = list(ratings)
        timestamps_list = list(timestamps)
        length = len(values)
        hist_len[position] = length
        if length:
            hist_idx[position, :length] = values
            hist_rating[position, :length] = ratings_list
            hist_time[position, :length] = timestamps_list
    return {
        "frame": frame,
        "memory": memory,
        "hist_idx": hist_idx,
        "hist_rating": hist_rating,
        "hist_time": hist_time,
        "hist_len": hist_len,
        "target_item": frame["target_item_index"].to_numpy(dtype=np.int64),
        "target_rating": frame["target_rating"].to_numpy(dtype=np.float32),
        "target_ts": frame["target_timestamp"].to_numpy(dtype=np.int64),
    }


def _history_batch(data: dict[str, Any], indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    idx = data["hist_idx"][indices]
    rating = data["hist_rating"][indices]
    time_values = data["hist_time"][indices]
    lengths = data["hist_len"][indices]
    mask = torch.arange(MAX_HISTORY, device=device).unsqueeze(0) < torch.as_tensor(lengths, device=device).unsqueeze(1)
    features = item_matrix[torch.as_tensor(idx, device=device).long()]
    known = (torch.as_tensor(idx, device=device) < n_items).float()
    days = (torch.as_tensor(data["target_ts"][indices], device=device).float().unsqueeze(1) - torch.as_tensor(time_values, device=device).float()) / MS_PER_DAY
    days = torch.clamp(days, min=0.0)
    return (
        features,
        torch.as_tensor(rating, device=device),
        days,
        known,
        mask,
        torch.as_tensor(data["memory"][indices], device=device),
    )


def _candidate_features(candidate_indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return item_matrix[torch.as_tensor(candidate_indices, device=device).long()]


def _train_batch(model, optimizer, data, indices, device, memory_on):
    candidates = np.concatenate(
        [
            data["target_item"][indices, None],
            train_negatives[indices],
        ],
        axis=1,
    )
    cand_features = _candidate_features(candidates, device)
    hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory_features = _history_batch(data, indices, device)
    scores, rating_pred = model(
        cand_features, hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory_features
    )
    ranking_loss = nn.functional.cross_entropy(
        scores, torch.zeros(scores.shape[0], dtype=torch.long, device=device)
    )
    rating_target = torch.as_tensor((data["target_rating"][indices] - 3.0) / 2.0).to(device)
    rating_loss = nn.functional.mse_loss(rating_pred[:, 0], rating_target)
    loss = ranking_loss + RATING_LOSS_WEIGHT * rating_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), float(ranking_loss.detach()), float(rating_loss.detach())


@torch.no_grad()
def _score_candidates(model, data, candidate_indices, device, batch_size=512):
    model.eval()
    outputs = []
    for start in range(0, len(candidate_indices), batch_size):
        stop = min(start + batch_size, len(candidate_indices))
        indices = np.arange(start, stop)
        candidates = candidate_indices[start:stop]
        cand_features = _candidate_features(candidates, device)
        hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory_features = _history_batch(data, indices, device)
        scores, rating_pred = model(
            cand_features, hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory_features
        )
        outputs.append((scores.cpu().numpy(), rating_pred.cpu().numpy()))
    all_scores = np.concatenate([value[0] for value in outputs], axis=0)
    all_ratings = np.concatenate([value[1] for value in outputs], axis=0)
    return all_scores, all_ratings


def _ranking_metrics(scores: np.ndarray, positive_positions: np.ndarray, k=10) -> dict[str, float]:
    # rank of the positive candidate, counting strictly better scores; ties count pessimistically
    positive_scores = scores[np.arange(len(scores)), positive_positions]
    ranks = 1 + (scores > positive_scores[:, None]).sum(axis=1)
    hits = (ranks <= k).astype(float)
    ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    mrr = 1.0 / ranks
    return {
        "n": int(len(scores)),
        "hit@10": round(float(hits.mean()), 6),
        "ndcg@10": round(float(ndcg.mean()), 6),
        "mrr@10": round(float(mrr.mean()), 6),
        "mean_rank": round(float(ranks.mean()), 6),
        "candidates": int(scores.shape[1]),
    }


@torch.no_grad()
def _pair_metrics(model, data, pairs: pd.DataFrame, device) -> dict[str, Any]:
    group_indices = []
    groups = []
    for group_id, group in pairs.groupby("pair_group_id", sort=False):
        val_row = int(group.iloc[0]["val_row_index"])
        candidates = group["candidate_item_index"].to_numpy(dtype=np.int64)
        ratings = group["candidate_rating"].to_numpy(dtype=np.float64)
        group_indices.append(val_row)
        groups.append((candidates, ratings))
    all_scores = []
    for start in range(0, len(group_indices), 512):
        stop = min(start + 512, len(group_indices))
        row_indices = np.asarray(group_indices[start:stop], dtype=np.int64)
        candidate_array = np.stack([group[0] for group in groups[start:stop]], axis=0)
        cand_features = _candidate_features(candidate_array, device)
        hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory_features = _history_batch(data, row_indices, device)
        scores, _ = model(
            cand_features, hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory_features
        )
        all_scores.append(scores.cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    pairwise_all = []
    pairwise_high_low = []
    ndcg = []
    mrr = []
    for row_scores, (_candidate_indices, ratings) in zip(scores, groups):
        if ratings[0] == ratings[1]:
            continue
        predicted_high = int(np.argmax(row_scores))
        true_high = int(np.argmax(ratings))
        pairwise_all.append(1.0 if predicted_high == true_high else 0.0)
        if (ratings >= 4).any() and (ratings <= 2).any():
            pairwise_high_low.append(1.0 if predicted_high == true_high else 0.0)
        order = np.argsort(-row_scores)
        ideal = np.sort(ratings)[::-1]
        dcg = ratings[order[0]] + ratings[order[1]] / math.log2(3)
        idcg = ideal[0] + ideal[1] / math.log2(3)
        ndcg.append(dcg / idcg if idcg else 0.0)
        relevant = np.where(ratings >= 4)[0]
        if len(relevant):
            first = next(index + 1 for index, candidate in enumerate(order) if candidate in set(relevant))
            mrr.append(1.0 / first)
    return {
        "groups": int(len(groups)),
        "pairwise_ordering_accuracy": round(float(np.mean(pairwise_all)), 6) if pairwise_all else None,
        "n_pairs_rating_different": len(pairwise_all),
        "high_vs_low_pairwise_accuracy": round(float(np.mean(pairwise_high_low)), 6) if pairwise_high_low else None,
        "n_high_vs_low_pairs": len(pairwise_high_low),
        "ndcg_at_2_graded": round(float(np.mean(ndcg)), 6) if ndcg else None,
        "mrr_high": round(float(np.mean(mrr)), 6) if mrr else None,
    }


def _rating_aux_metrics(rating_pred: np.ndarray, target_rating: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

    predicted = 3.0 + 2.0 * rating_pred
    high = (target_rating >= 4).astype(int)
    low = (target_rating <= 2).astype(int)
    binary = (high + low) == 1
    return {
        "mae": round(float(mean_absolute_error(target_rating, predicted)), 6),
        "rmse": round(float(math.sqrt(mean_squared_error(target_rating, predicted))), 6),
        "roc_auc_high_low": round(
            float(roc_auc_score(high[binary], predicted[binary])), 6
        ) if binary.sum() > 1 and len(np.unique(high[binary])) > 1 else None,
        "n_binary": int(binary.sum()),
    }


def _train_one(memory_on: bool) -> dict[str, Any]:
    _set_seed(SEED)
    device = torch.device("cuda")
    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")
    pairs = pd.read_parquet(DATA / "pairs.parquet")
    val_candidates = np.load(DATA / f"val_candidates{CANDIDATE_SUFFIX}.npy")
    val_positive = np.load(DATA / f"val_positive_position{CANDIDATE_SUFFIX}.npy").astype(np.int64)
    test_candidates = np.load(DATA / f"test_candidates{CANDIDATE_SUFFIX}.npy")
    test_positive = np.load(DATA / f"test_positive_position{CANDIDATE_SUFFIX}.npy").astype(np.int64)
    common_indices = np.load(DATA / "test_common_indices.npy")
    item_feature_names = json.loads((DATA / "item_feature_names.json").read_text())
    memory_names = json.loads((DATA / "memory_feature_names.json").read_text())
    attr_indices = [item_feature_names.index(f"meta_attr_{attr}") for attr in
                    ["over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth",
                     "microphone", "rechargeable", "noise_cancelling", "waterproof", "foldable",
                     "portable", "gaming", "mechanical", "silent", "ergonomic", "touch", "rgb",
                     "led", "usb", "hdmi", "4k", "remote"]]
    high_indices = [memory_names.index(f"h_high_attr_{attr}") for attr in
                    ["over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth",
                     "microphone", "rechargeable", "noise_cancelling", "waterproof", "foldable",
                     "portable", "gaming", "mechanical", "silent", "ergonomic", "touch", "rgb",
                     "led", "usb", "hdmi", "4k", "remote"]]
    low_indices = [memory_names.index(f"h_low_attr_{attr}") for attr in
                   ["over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth",
                    "microphone", "rechargeable", "noise_cancelling", "waterproof", "foldable",
                    "portable", "gaming", "mechanical", "silent", "ergonomic", "touch", "rgb",
                    "led", "usb", "hdmi", "4k", "remote"]]
    config = ModelConfig(
        item_dim=item_matrix.shape[1],
        memory_dim=len(memory_names),
        hidden=128,
        dropout=0.1,
        memory_on=memory_on,
        use_attributes=not ATTR_ABLATION,
    )
    model = NeuralMemoryRanker(config, attr_indices, high_indices, low_indices).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    # Real forward/backward smoke before the full run.
    smoke_indices = np.arange(min(BATCH_SIZE, len(train["frame"])), dtype=np.int64)
    smoke_loss, smoke_rank, smoke_rating = _train_batch(model, optimizer, train, smoke_indices, device, memory_on)
    smoke = {
        "batch": int(len(smoke_indices)),
        "loss": round(smoke_loss, 6),
        "ranking_loss": round(smoke_rank, 6),
        "rating_loss": round(smoke_rating, 6),
        "backward_ok": True,
    }

    best_state = None
    best_val = -1.0
    best_epoch = -1
    history = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = np.random.permutation(len(train["frame"]))
        total_loss = total_rank = total_rating = 0.0
        batches = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            if len(indices) < 2:
                continue
            loss, ranking_loss, rating_loss = _train_batch(model, optimizer, train, indices, device, memory_on)
            total_loss += loss
            total_rank += ranking_loss
            total_rating += rating_loss
            batches += 1
        val_scores, _ = _score_candidates(model, val, val_candidates, device)
        val_metrics = _ranking_metrics(val_scores, val_positive)
        history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / batches, 6),
            "train_ranking_loss": round(total_rank / batches, 6),
            "train_rating_loss": round(total_rating / batches, 6),
            "val_hit@10": val_metrics["hit@10"],
            "val_ndcg@10": val_metrics["ndcg@10"],
            "val_mrr@10": val_metrics["mrr@10"],
        })
        print(json.dumps(history[-1]), flush=True)
        if val_metrics["ndcg@10"] > best_val:
            best_val = val_metrics["ndcg@10"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    training_seconds = time.time() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    model.eval()

    test_scores, test_rating_pred = _score_candidates(model, test, test_candidates, device)
    test_metrics = _ranking_metrics(test_scores, test_positive)
    common_scores = test_scores[common_indices]
    common_positive = test_positive[common_indices]
    common_metrics = _ranking_metrics(common_scores, common_positive)
    positive_test_ratings = test_rating_pred[np.arange(len(test_rating_pred)), test_positive]
    test_aux = _rating_aux_metrics(positive_test_ratings, test["target_rating"])
    pair_metrics = _pair_metrics(model, val, pairs, device)

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / (f"memory_on{RUN_SUFFIX}.pt" if memory_on else f"memory_off{RUN_SUFFIX}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "attr_indices": attr_indices,
            "high_attr_indices": high_indices,
            "low_attr_indices": low_indices,
            "best_epoch": best_epoch,
        },
        model_path,
    )
    return {
        "memory_on": memory_on,
        "model": model_path.name,
        "parameters": int(n_params),
        "config": asdict(config),
        "smoke": smoke,
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_val,
        "epochs": history,
        "training_seconds": round(training_seconds, 3),
        "peak_gpu_memory_mb": round(peak_memory, 2),
        "train_rows": int(len(train["frame"])),
        "negatives_per_positive": TRAIN_NEGATIVES,
        "test": test_metrics,
        "test_common_14157": common_metrics,
        "test_rating_aux": test_aux,
        "pairwise": pair_metrics,
    }


def main() -> int:
    global item_matrix, n_items, train_negatives
    item_matrix = torch.from_numpy(np.load(DATA / "item_features.npy")).cuda()
    n_items = item_matrix.shape[0] - 1
    train_negatives = np.load(DATA / f"train_negatives{CANDIDATE_SUFFIX}.npy")
    results = {}
    for memory_on in (False, True):
        name = "memory_on" if memory_on else "memory_off"
        print(f"=== training {name} ===", flush=True)
        results[name] = _train_one(memory_on)
        print(json.dumps({key: value for key, value in results[name].items() if key not in {"epochs"}}, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = {
        "schema": "agentrecx.neural_memory_ranking.comparison.v1",
        "dataset_manifest": json.loads((DATA / "manifest.json").read_text()),
        "models": results,
        "protocol": {
            "test": "77,029 test targets; each scored against 1 positive + 99 sampled unobserved catalog candidates",
            "test_common_14157": "the 14,157 users with >=3 catalog interactions (same user scope as Phase 2), same 1+99 sampled protocol",
            "pairwise": "14,157 same-user held-out pairs (last two catalog events, common history before the first candidate)",
            "negatives": "sampled unobserved catalog items; not confirmed negative feedback",
            "targets": "catalog items only",
        },
    }
    (OUT / f"comparison{RUN_SUFFIX}.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
