"""Evaluate neural OFF/ON on history-length and attribute-coverage subsets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path("/root/autodl-tmp/neural_memory_v1")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/neural_memory_v1")


def bucket_length(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 9:
        return "5-9"
    return "10+"


def bucket_coverage(value: float) -> str:
    if value < 0.25:
        return "0.00-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def metrics(scores, positive_positions, indices):
    sub_scores = scores[indices]
    sub_positive = positive_positions[indices]
    positive_scores = sub_scores[np.arange(len(sub_scores)), sub_positive]
    ranks = 1 + (sub_scores > positive_scores[:, None]).sum(axis=1)
    return {
        "n": int(len(sub_scores)),
        "hit@10": round(float((ranks <= 10).mean()), 6),
        "ndcg@10": round(float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0).mean()), 6),
        "mrr@10": round(float((1.0 / ranks).mean()), 6),
    }


def main():
    import sys
    sys.path.insert(0, "/root/autodl-tmp/AgentRec-X-phase2")
    from experiments.neural_memory_ranking import train_neural as tn
    from experiments.neural_memory_ranking.model import ModelConfig, NeuralMemoryRanker

    device = torch.device("cuda")
    item_matrix = torch.from_numpy(np.load(DATA / "item_features.npy")).cuda()
    n_items = item_matrix.shape[0] - 1
    tn.item_matrix = item_matrix
    tn.n_items = n_items
    test = tn._load_split("test")
    candidates = np.load(DATA / "test_candidates_hard.npy")
    positive = np.load(DATA / "test_positive_position_hard.npy").astype(np.int64)
    item_names = json.loads((DATA / "item_feature_names.json").read_text())
    coverage_index = item_names.index("meta_attr_coverage")
    positive_item = test["target_item"]
    coverage = np.load(DATA / "item_features.npy")[positive_item, coverage_index]
    lengths = test["frame"]["history_length"].to_numpy()
    groups_length = np.asarray([bucket_length(int(v)) for v in lengths])
    groups_coverage = np.asarray([bucket_coverage(float(v)) for v in coverage])

    outputs = {}
    for name, file in (("memory_off", "memory_off_hard.pt"), ("memory_on", "memory_on_hard.pt")):
        checkpoint = torch.load(OUT / file, map_location="cpu")
        config = ModelConfig(**checkpoint["config"])
        model = NeuralMemoryRanker(config, checkpoint["attr_indices"], checkpoint["high_attr_indices"], checkpoint["low_attr_indices"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scores, _ = tn._score_candidates(model, test, candidates, device)
        outputs[name] = scores
    results = {"by_history_length": {}, "by_target_attribute_coverage": {}}
    for field, groups, keys in (
        ("by_history_length", groups_length, ["1", "2", "3-4", "5-9", "10+"]),
        ("by_target_attribute_coverage", groups_coverage, ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]),
    ):
        for key in keys:
            indices = np.where(groups == key)[0]
            if not len(indices):
                continue
            off = metrics(outputs["memory_off"], positive, indices)
            on = metrics(outputs["memory_on"], positive, indices)
            results[field][key] = {
                "n": int(len(indices)),
                "memory_off": off,
                "memory_on": on,
                "delta_hit@10": round(on["hit@10"] - off["hit@10"], 6),
                "delta_ndcg@10": round(on["ndcg@10"] - off["ndcg@10"], 6),
                "delta_mrr@10": round(on["mrr@10"] - off["mrr@10"], 6),
            }
    (OUT / "neural_subsets.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
