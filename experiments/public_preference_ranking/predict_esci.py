"""Export per-query ESCI test predictions for the three DCN arms."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/public_preference_ranking_v1"
OUT = REPO / "runs/public_preference_ranking_v1"
sys.path.insert(0, str(REPO))
from experiments.dcn_v2.model import DCNConfig, DCNV2Ranker

GAIN = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def main():
    frame = pd.read_parquet(DATA / "dcn_features.parquet")
    schema = json.loads((DATA / "dcn_feature_schema.json").read_text())
    train = frame[frame["split"] == "train"]
    rows = []
    for arm in ("A_base_query_only", "B_preference_on", "C_full_query"):
        features = schema["views"][arm]
        mean = train[features].astype(np.float32).fillna(0.0).to_numpy().mean(axis=0, keepdims=True)
        std = train[features].astype(np.float32).fillna(0.0).to_numpy().std(axis=0, keepdims=True) + 1e-6
        model = DCNV2Ranker(DCNConfig(input_dim=len(features), hidden=64, cross_layers=3, dropout=0.1)).cuda()
        model.load_state_dict(torch.load(OUT / f"{arm}.pt", map_location="cuda"))
        model.eval()
        test = frame[frame["split"] == "test"]
        with torch.no_grad():
            for group_id, part in test.groupby("group_id", sort=True):
                x = np.clip((part[features].astype(np.float32).fillna(0.0).to_numpy() - mean) / std, -8, 8)
                scores = model(torch.from_numpy(x).cuda()).cpu().numpy()
                products = []
                for (_, item), score in zip(part.iterrows(), scores):
                    products.append({
                        "product_id": str(item["product_id"]),
                        "esci_label": str(item["esci_label"]),
                        "score": round(float(score), 6),
                    })
                products.sort(key=lambda entry: (-entry["score"], entry["product_id"]))
                for rank, entry in enumerate(products, start=1):
                    entry["rank"] = rank
                gains = [GAIN.get(entry["esci_label"], 0.0) for entry in products]
                dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
                ideal = sorted(gains, reverse=True)
                idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
                first_e = next((index + 1 for index, entry in enumerate(products) if entry["esci_label"] == "E"), None)
                rows.append({
                    "arm": arm,
                    "group_id": group_id,
                    "query_id": str(part["query_id"].iloc[0]),
                    "query": str(part["query"].iloc[0]),
                    "base_query": str(part["base_query"].iloc[0]),
                    "preference_attr": str(part["preference_attr"].iloc[0]),
                    "preference_polarity": str(part["preference_polarity"].iloc[0]),
                    "ndcg@10": round(dcg / idcg, 6) if idcg else 0.0,
                    "mrr": round(1.0 / first_e, 6) if first_e else 0.0,
                    "products": products,
                })
        print("exported", arm, flush=True)
    (OUT / "esci_test_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print("rows", len(rows))


if __name__ == "__main__":
    main()
