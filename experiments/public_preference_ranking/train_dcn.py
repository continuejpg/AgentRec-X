"""DCN-V2 three-arm training on the ESCI query-derived preference proxy.

Arms:
  A base_query_only
  B preference_on
  C full_query
The labels are official ESCI E/S/C/I; no generated labels.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/public_preference_ranking_v1"
OUT = REPO / "runs/public_preference_ranking_v1"
sys.path.insert(0, str(REPO))
from experiments.dcn_v2.model import DCNConfig, DCNV2Ranker

GAIN = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def load():
    frame = pd.read_parquet(DATA / "dcn_features.parquet")
    schema = json.loads((DATA / "dcn_feature_schema.json").read_text())
    groups = {}
    for group_id, part in frame.groupby("group_id", sort=True):
        part = part.sort_values("product_id").reset_index(drop=True)
        groups[group_id] = part
    return groups, schema


def pair_index(part: pd.DataFrame) -> list[tuple[int, int, int]]:
    candidates = part.index.to_numpy()
    grades = part["grade"].to_numpy()
    pairs = []
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j:
                continue
            if grades[i] > grades[j]:
                pairs.append((int(candidates[i]), int(candidates[j]), 1))
    return pairs


def evaluate(model, frame: pd.DataFrame, features: list[str], mean: np.ndarray, std: np.ndarray) -> dict:
    model.eval()
    values = []
    with torch.no_grad():
        for group_id, part in frame.groupby("group_id", sort=True):
            x = part[features].astype(np.float32).fillna(0.0).to_numpy()
            x = np.clip((x - mean) / std, -8, 8)
            scores = model(torch.from_numpy(x).cuda()).cpu().numpy()
            part = part.reset_index(drop=True)
            part["score"] = scores
            part = part.sort_values(["score", "product_id"], ascending=[False, True]).reset_index(drop=True)
            labels = part["esci_label"].tolist()
            gains = [GAIN.get(label, 0.0) for label in labels]
            dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
            ideal = sorted(gains, reverse=True)
            idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
            ndcg10 = dcg / idcg if idcg else 0.0
            first_e = next((index + 1 for index, label in enumerate(labels) if label == "E"), None)
            mrr = 1.0 / first_e if first_e else 0.0
            pair_correct = pair_total = 0
            for i in range(len(part)):
                for j in range(len(part)):
                    if i == j or part.loc[i, "grade"] <= part.loc[j, "grade"]:
                        continue
                    pair_total += 1
                    pair_correct += int(part.loc[i, "score"] > part.loc[j, "score"])
            values.append({
                "group_id": group_id,
                "ndcg@3": dcg_at(labels, gains, 3), "ndcg@5": dcg_at(labels, gains, 5),
                "ndcg@10": ndcg10, "mrr": mrr,
                "pair_correct": pair_correct, "pair_total": pair_total,
                "preference_polarity": part["preference_polarity"].iloc[0],
                "preference_attr": part["preference_attr"].iloc[0],
                "preference_coverage": float(part["preference_present"].mean()),
            })
    result = {
        "groups": len(values),
        "ndcg@3": round(float(np.mean([row["ndcg@3"] for row in values])), 6),
        "ndcg@5": round(float(np.mean([row["ndcg@5"] for row in values])), 6),
        "ndcg@10": round(float(np.mean([row["ndcg@10"] for row in values])), 6),
        "mrr": round(float(np.mean([row["mrr"] for row in values])), 6),
        "pairwise_accuracy": round(float(sum(row["pair_correct"] for row in values) / max(1, sum(row["pair_total"] for row in values))), 6),
        "pairwise_pairs": int(sum(row["pair_total"] for row in values)),
        "by_polarity": {},
        "by_coverage": {},
        "per_group": values,
    }
    for polarity in ("positive", "negative"):
        subset = [row for row in values if row["preference_polarity"] == polarity]
        if subset:
            result["by_polarity"][polarity] = {
                "groups": len(subset),
                "ndcg@10": round(float(np.mean([row["ndcg@10"] for row in subset])), 6),
                "mrr": round(float(np.mean([row["mrr"] for row in subset])), 6),
            }
    for bucket, predicate in (("coverage<0.5", lambda value: value < 0.5), ("coverage>=0.5", lambda value: value >= 0.5)):
        subset = [row for row in values if predicate(row["preference_coverage"])]
        if subset:
            result["by_coverage"][bucket] = {
                "groups": len(subset),
                "ndcg@10": round(float(np.mean([row["ndcg@10"] for row in subset])), 6),
                "mrr": round(float(np.mean([row["mrr"] for row in subset])), 6),
            }
    return result


def dcg_at(labels: list[str], gains: list[float], k: int) -> float:
    actual = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains[:k]))
    ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(sorted(gains, reverse=True)[:k]))
    return actual / ideal if ideal else 0.0


def train_arm(groups, features, arm_name, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_groups = [part for part in groups.values() if part["split"].iloc[0] == "train"]
    val_groups = [part for part in groups.values() if part["split"].iloc[0] == "val"]
    test_frame = pd.concat([part for part in groups.values() if part["split"].iloc[0] == "test"], ignore_index=True)
    train_frame = pd.concat(train_groups, ignore_index=True)
    all_train_x = train_frame[features].astype(np.float32).fillna(0.0).to_numpy()
    mean = all_train_x.mean(axis=0, keepdims=True)
    std = all_train_x.std(axis=0, keepdims=True) + 1e-6
    train_parts = {part["group_id"].iloc[0]: part for part in train_groups}
    val_parts = {part["group_id"].iloc[0]: part for part in val_groups}
    # Materialize pairwise rows: winner feature vector, loser feature vector.
    winners, losers = [], []
    for part in train_parts.values():
        for i, j, _ in pair_index(part):
            winners.append(part.loc[i, features].astype(float).fillna(0.0).to_numpy())
            losers.append(part.loc[j, features].astype(float).fillna(0.0).to_numpy())
    winners = np.clip((np.asarray(winners, dtype=np.float32) - mean) / std, -8, 8)
    losers = np.clip((np.asarray(losers, dtype=np.float32) - mean) / std, -8, 8)
    model = DCNV2Ranker(DCNConfig(input_dim=len(features), hidden=64, cross_layers=3, dropout=0.1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(winners), torch.from_numpy(losers)), batch_size=256, shuffle=True)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    started = time.perf_counter()
    history = []
    for epoch in range(1, 31):
        model.train()
        losses = []
        for winner, loser in loader:
            winner, loser = winner.cuda(), loser.cuda()
            scores = model(winner) - model(loser)
            loss = loss_fn(scores, torch.ones_like(scores))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, pd.concat(val_groups, ignore_index=True), features, mean, std)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6), "val_ndcg@10": val["ndcg@10"], "val_mrr": val["mrr"]})
        if val["ndcg@10"] > best_val:
            best_val = val["ndcg@10"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    train_metrics = evaluate(model, train_frame, features, mean, std)
    val_metrics = evaluate(model, pd.concat(val_groups, ignore_index=True), features, mean, std)
    test_metrics = evaluate(model, test_frame, features, mean, std)
    torch.save(model.state_dict(), OUT / f"{arm_name}.pt")
    test_metrics["per_group"] = test_metrics["per_group"]
    return {
        "arm": arm_name,
        "features": len(features),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_groups": len(train_parts),
        "val_groups": len(val_parts),
        "test_groups": int(test_frame["group_id"].nunique()),
        "train_pairs": int(len(winners)),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_seconds": round(elapsed, 3),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups, schema = load()
    torch.cuda.reset_peak_memory_stats()
    results = {
        "schema": "agentrecx.public_preference_ranking_v1.dcn_training.v1",
        "dataset": "ESCI official train/test, query-derived preference proxy",
        "label_source": "official ESCI E/S/C/I only",
        "preference_source": "query_derived_proxy (not historical memory)",
        "gain_map": GAIN,
        "nDCG": "official Task 1 style graded nDCG; C is grade 1, never collapsed into S",
        "seed": 42,
        "splits": {split: int(sum(1 for part in groups.values() if part["split"].iloc[0] == split)) for split in ("train", "val", "test")},
        "arms": {},
    }
    for arm_name, view in (("A_base_query_only", "A_base_query_only"), ("B_preference_on", "B_preference_on"), ("C_full_query", "C_full_query")):
        results["arms"][arm_name] = train_arm(groups, schema["views"][view], arm_name, seed=42)
        print(arm_name, json.dumps({key: results["arms"][arm_name][key] for key in ("features", "best_epoch", "train_seconds", "peak_gpu_mb")}), flush=True)
    (OUT / "dcn_training_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {
        "splits": results["splits"],
        "arms": {
            name: {
                "features": arm["features"], "parameters": arm["parameters"],
                "best_epoch": arm["best_epoch"], "train_seconds": arm["train_seconds"],
                "peak_gpu_mb": arm["peak_gpu_mb"],
                "test": {key: arm["test_metrics"][key] for key in ("groups", "ndcg@3", "ndcg@5", "ndcg@10", "mrr", "pairwise_accuracy", "pairwise_pairs")},
            }
            for name, arm in results["arms"].items()
        },
    }
    print(json.dumps(compact, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
