"""DCN-V2 Memory OFF / ON pilot on Silver Benchmark v2 pairs.

This is a pilot on rule-derived SILVER labels, not human preference gold.
One group has exactly two real candidates (A/B); the label is which one better
matches the synthetic memory state according to the fixed Silver rule.
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
DATA = REPO / "data/memory_ranking_benchmark_v2/dcn_v2_pairs"
OUT = REPO / "runs/memory_ranking_benchmark_v2"
sys.path.insert(0, str(REPO))
from experiments.dcn_v2.model import DCNConfig, DCNV2Ranker


def load():
    frame = pd.read_parquet(DATA / "pairs.parquet")
    schema = json.loads((DATA / "feature_schema.json").read_text())
    features = list(schema["numeric_features"])
    memory = set(schema["memory_features"])
    off = [name for name in features if name not in memory]
    groups = []
    for group_id, part in frame.groupby("group_id", sort=True):
        part = part.set_index("side")
        if "A" not in part.index or "B" not in part.index:
            continue
        split = str(part.iloc[0]["split"])
        winner = int(part.loc["A"]["winner"])
        groups.append({
            "group_id": group_id,
            "base_query_id": str(part.iloc[0]["base_query_id"]),
            "scenario_type": str(part.iloc[0]["scenario_type"]),
            "split": split,
            "a": part.loc["A"][features].astype(float).to_numpy(),
            "b": part.loc["B"][features].astype(float).to_numpy(),
            "y": float(winner),
        })
    return groups, features, off, memory


def stack(groups, indices):
    x = np.stack([np.stack([group["a"][indices], group["b"][indices]]) for group in groups]).astype(np.float32)
    y = np.asarray([group["y"] for group in groups], dtype=np.float32)
    return x, y


def train_arm(groups, off_features, features, memory_on: bool, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    indices = list(range(len(features))) if memory_on else [features.index(name) for name in off_features]
    by_split = {split: [group for group in groups if group["split"] == split] for split in ("train", "val", "test")}
    train_x, train_y = stack(by_split["train"], indices)
    val_x, val_y = stack(by_split["val"], indices)
    test_x, test_y = stack(by_split["test"], indices)
    mean = train_x.reshape(-1, len(indices)).mean(axis=0, keepdims=True)
    std = train_x.reshape(-1, len(indices)).std(axis=0, keepdims=True) + 1e-6
    for array in (train_x, val_x, test_x):
        array[:] = np.clip((array - mean) / std, -8, 8)
    device = torch.device("cuda")
    model = DCNV2Ranker(DCNConfig(input_dim=len(indices), hidden=64, cross_layers=3, dropout=0.1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=32, shuffle=True)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, 61):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            scores = model(xb)  # [B, 2]
            logits = scores[:, 0] - scores[:, 1]
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
        val_accuracy, val_logits = evaluate(model, val_x, val_y, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_accuracy": val_accuracy})
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 12:
            break
    model.load_state_dict(best_state)
    train_time = time.perf_counter() - started
    train_accuracy, _ = evaluate(model, train_x, train_y, device)
    val_accuracy, val_logits = evaluate(model, val_x, val_y, device)
    test_accuracy, test_logits = evaluate(model, test_x, test_y, device)
    test_groups = by_split["test"]
    per_group = []
    for group, logit in zip(test_groups, test_logits):
        per_group.append({
            "group_id": group["group_id"],
            "base_query_id": group["base_query_id"],
            "scenario_type": group["scenario_type"],
            "label_is_a": group["y"],
            "logit_a_minus_b": float(logit),
            "predicted_is_a": bool(logit > 0),
            "correct": bool((logit > 0) == (group["y"] > 0.5)),
        })
    torch.save(model.state_dict(), OUT / f"dcn_v2_{'on' if memory_on else 'off'}.pt")
    return {
        "memory_on": memory_on,
        "input_features": len(indices),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_groups": len(train_x),
        "val_groups": len(val_x),
        "test_groups": len(test_x),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_loss_last": history[-1]["train_loss"],
        "train_accuracy_group": train_accuracy,
        "val_accuracy_group": val_accuracy,
        "test_accuracy_group": test_accuracy,
        "train_seconds": round(train_time, 3),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        "history": history,
        "test_predictions": per_group,
    }


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    tensor = torch.from_numpy(x).to(device)
    scores = model(tensor)
    logits = (scores[:, 0] - scores[:, 1]).cpu().numpy()
    # y=1 means A is preferred; logit>0 predicts A.
    accuracy = float(np.mean((logits > 0) == (y > 0.5)))
    return accuracy, logits


def main():
    groups, features, off_features, memory = load()
    torch.cuda.reset_peak_memory_stats()
    results = {
        "schema": "agentrecx.dcn_v2.benchmark_v2_pilot.v1",
        "dataset": "memory_ranking_benchmark_v2/dcn_v2_pairs",
        "label_provenance": "SILVER only; HUMAN_GOLD=0",
        "groups": len(groups),
        "features": len(features),
        "memory_features": len(memory),
        "off_features": len(off_features),
        "split_groups": {split: sum(1 for group in groups if group["split"] == split) for split in ("train", "val", "test")},
        "arms": {},
    }
    for memory_on in (False, True):
        results["arms"]["memory_on" if memory_on else "memory_off"] = train_arm(groups, off_features, features, memory_on)
    (OUT / "dcn_v2_pilot.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {key: results[key] for key in ("groups", "features", "memory_features", "split_groups")}
    compact["arms"] = {name: {key: value for key, value in arm.items() if key not in {"history", "test_predictions"}} for name, arm in results["arms"].items()}
    print(json.dumps(compact, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
