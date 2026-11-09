"""P10 four-model Dev evaluation under the fixed P8-D protocol.

Models: SASRec, BGE-enhanced SASRec, Two-Tower, LightGCN.  All are scored on the same
300 histories, with the same seen-item masking and metrics.  The LightGCN arm is
transductive and uses the user id from the case; unseen users are not scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recommendation.datasets.bge_features import build_bge_item_matrix
from recommendation.datasets.lightgcn import build_normalized_adjacency
from recommendation.datasets.sasrec import encode_inference_history
from recommendation.datasets.twotower import load_metadata_categories, metadata_tensor
from recommendation.evaluation.metrics import evaluate_case
from recommendation.models.bge_sasrec import BGEEnhancedSASRec
from recommendation.models.lightgcn import LightGCN, LightGCNConfig
from recommendation.models.sasrec import SASRec, SASRecConfig
from recommendation.models.twotower import TwoTower, TwoTowerConfig

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "runs/p8_d/multiuser/cases.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
BASE_SASREC = ROOT / "runs/sasrec_canonical_2026/best.pt"
BASE_TWOTOWER = ROOT / "runs/twotower_public_2026_full/checkpoint.pt"
TWOTOWER_METADATA = ROOT / "runs/twotower_public_2026_full/metadata_categories.json"
BGE_INDEX = Path("/root/autodl-tmp/bge_build/bge_base_dense.npz")
OUT = ROOT / "runs/p10/eval"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sasrec_config(raw: dict[str, Any]) -> SASRecConfig:
    return SASRecConfig(
        num_items=int(raw["num_items"]),
        max_seq_len=int(raw["max_seq_len"]),
        hidden_size=int(raw["hidden_size"]),
        num_blocks=int(raw["num_blocks"]),
        num_heads=int(raw["num_heads"]),
        dropout=float(raw["dropout"]),
    )


def _topk(scores: np.ndarray, seen: set[int], k: int) -> list[int]:
    """Top-k real item ids, excluding seen history; ties break on smaller id."""
    array = torch.from_numpy(np.asarray(scores, dtype=np.float64))
    eligible = np.ones(array.numel(), dtype=bool)
    eligible[0] = False
    if seen:
        eligible[np.asarray(sorted(seen), dtype=np.int64)] = False
    ids = np.flatnonzero(eligible)
    if ids.size == 0:
        return []
    # Exact top-k ordering for the small selected prefix.
    limit = min(k, ids.size)
    order = np.lexsort((ids, -array[ids].numpy()))[:limit]
    return [int(item) for item in ids[order]]


def _metrics(scores: np.ndarray, target: int, history: list[int], num_items: int) -> dict[str, Any]:
    result = evaluate_case(
        scores.tolist(),
        target,
        history=history,
        k_values=(10, 50, 100),
        num_items=num_items,
    )
    rank = int(result.rank)
    return {
        "rank": rank,
        "recall@10": 1.0 if rank <= 10 else 0.0,
        "recall@50": 1.0 if rank <= 50 else 0.0,
        "recall@100": 1.0 if rank <= 100 else 0.0,
        "ndcg@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        "mrr@10": 1.0 / rank if rank <= 10 else 0.0,
    }


def _fuse(rankings: dict[str, list[int]], quotas: dict[str, int]) -> list[int]:
    totals: dict[int, float] = {}
    for source, quota in quotas.items():
        for rank, item in enumerate(rankings[source][:quota], start=1):
            totals[item] = totals.get(item, 0.0) + 1.0 / (60.0 + rank)
    return [item for item, _ in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sasrec", type=Path, default=BASE_SASREC)
    parser.add_argument("--base-twotower", type=Path, default=BASE_TWOTOWER)
    parser.add_argument("--twotower-metadata", type=Path, default=TWOTOWER_METADATA)
    parser.add_argument("--enhanced-sasrec", type=Path, default=ROOT / "runs/p10/bge_sasrec/checkpoint.pt")
    parser.add_argument("--lightgcn", type=Path, default=ROOT / "runs/p10/lightgcn/checkpoint.pt")
    parser.add_argument("--bge-index", type=Path, default=BGE_INDEX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    cases = load_cases()
    mapping = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    item2id = mapping["item2id"]
    user2id = mapping["user2id"]
    num_items = int(mapping["num_items"])
    histories = [[int(item2id[value]) for value in case["history"]] for case in cases]
    targets = [int(item2id[case["target"]]) for case in cases]
    user_indices = [int(user2id[case["user_id"]]) - 1 for case in cases]

    # ---- SASRec ---------------------------------------------------------- #
    base_payload = torch.load(args.base_sasrec, map_location="cpu", weights_only=False)
    base_config = _sasrec_config(base_payload["model_config"])
    base = SASRec(base_config)
    base.load_state_dict(base_payload["model_state_dict"])
    base.to(device).eval()

    # ---- BGE-enhanced SASRec --------------------------------------------- #
    enhanced_payload = torch.load(args.enhanced_sasrec, map_location="cpu", weights_only=False)
    enhanced_config = _sasrec_config(enhanced_payload["model_config"])
    bge_matrix = build_bge_item_matrix(args.bge_index, MAPPINGS, num_items=num_items)
    enhanced = BGEEnhancedSASRec(enhanced_config, torch.from_numpy(bge_matrix))
    enhanced.load_state_dict(enhanced_payload["model_state_dict"], strict=True)
    enhanced.to(device).eval()

    # ---- Two-Tower -------------------------------------------------------- #
    tw_payload = torch.load(args.base_twotower, map_location="cpu", weights_only=False)
    tw_config = TwoTowerConfig(**tw_payload["model_config"])
    two_tower = TwoTower(tw_config)
    two_tower.load_state_dict(tw_payload["model_state_dict"], strict=False)
    two_tower.set_categories(metadata_tensor(load_metadata_categories(args.twotower_metadata)))
    two_tower.to(device).eval()

    # ---- LightGCN --------------------------------------------------------- #
    light_payload = torch.load(args.lightgcn, map_location="cpu", weights_only=False)
    light_config = LightGCNConfig(**light_payload["model_config"])
    lightgcn = LightGCN(light_config)
    lightgcn.load_state_dict(light_payload["model_state_dict"], strict=True)
    lightgcn.to(device).eval()
    edge_index = light_payload["edge_index"].numpy()
    U = int(light_payload["num_users"])
    N = int(light_payload["num_items"])
    adjacency = build_normalized_adjacency(
        edge_index, num_users=U, num_items=N, device=device
    )

    model_scores: dict[str, list[np.ndarray]] = {name: [] for name in ("sasrec", "bge_sasrec", "two_tower", "lightgcn")}
    with torch.inference_mode():
        for start in range(0, len(cases), args.batch_size):
            batch_histories = histories[start : start + args.batch_size]
            batch_targets = targets[start : start + args.batch_size]
            encoded = torch.tensor(
                [encode_inference_history(h, base_config.max_seq_len, num_items) for h in batch_histories],
                dtype=torch.long,
                device=device,
            )
            model_scores["sasrec"].extend(base.full_catalog_scores(encoded).cpu().numpy())
            model_scores["bge_sasrec"].extend(enhanced.full_catalog_scores(encoded).cpu().numpy())
            model_scores["two_tower"].extend(two_tower.full_catalog_scores(encoded).cpu().numpy())
            user_emb, item_emb = lightgcn.propagate(adjacency)
            light_scores = lightgcn.scores(
                user_emb, item_emb,
                torch.tensor(user_indices[start : start + args.batch_size], dtype=torch.long, device=device),
            )
            model_scores["lightgcn"].extend(light_scores.cpu().numpy())

    model_results: dict[str, list[dict[str, Any]]] = {name: [] for name in model_scores}
    top_rankings: dict[str, dict[str, list[int]]] = {name: {} for name in model_scores}
    for position, case in enumerate(cases):
        case_id = str(case["case_id"])
        history = histories[position]
        target = targets[position]
        seen = set(history)
        for name, scores in model_scores.items():
            result = _metrics(np.asarray(scores[position], dtype=np.float64), target, history, num_items)
            result["case_id"] = case_id
            result["target"] = case["target"]
            result["top10"] = _topk(scores[position], seen, 10)
            result["top50"] = _topk(scores[position], seen, 50)
            result["top100"] = _topk(scores[position], seen, 100)
            model_results[name].append(result)
            top_rankings[name][case_id] = result["top100"]

    summary: dict[str, Any] = {
        "schema_version": "agentrecx.p10.four_model_eval.v1",
        "cases": len(cases),
        "protocol": "P8-D 300 dev cases; validation target; seen-history masking; k=10/50/100",
        "models": {},
        "unique_hits": {},
        "top50_overlap": {},
        "fixed_budget_fusion": {},
    }
    for name, rows in model_results.items():
        summary["models"][name] = {
            "recall@10": float(np.mean([row["recall@10"] for row in rows])),
            "recall@50": float(np.mean([row["recall@50"] for row in rows])),
            "recall@100": float(np.mean([row["recall@100"] for row in rows])),
            "ndcg@10": float(np.mean([row["ndcg@10"] for row in rows])),
            "mrr@10": float(np.mean([row["mrr@10"] for row in rows])),
            "top50_reachability": int(sum(row["rank"] <= 50 for row in rows)),
            "top100_reachability": int(sum(row["rank"] <= 100 for row in rows)),
            "top10_hits": int(sum(row["rank"] <= 10 for row in rows)),
        }
    # Unique Top-10 hits: only one model got the target.
    for name in model_scores:
        others = [other for other in model_scores if other != name]
        unique_cases = []
        for index, case in enumerate(cases):
            own = model_results[name][index]["rank"] <= 10
            if own and all(model_results[other][index]["rank"] > 10 for other in others):
                unique_cases.append(case["case_id"])
        summary["unique_hits"][name] = {"count": len(unique_cases), "case_ids": unique_cases}

    # Top-50 overlap between model candidate sets.
    for a in model_scores:
        for b in model_scores:
            if a >= b:
                continue
            overlaps = []
            for index, case in enumerate(cases):
                left = set(model_results[a][index]["top50"])
                right = set(model_results[b][index]["top50"])
                overlaps.append(len(left & right) / max(1, len(left | right)))
            summary["top50_overlap"][f"{a}__{b}"] = float(np.mean(overlaps))

    # Fixed total candidate budgets.  Each quota is the number of candidates that
    # source may contribute; RRF is the accepted rank fusion.
    fusion_specs = {
        "budget100_base_sasrec50_twotower50": {
            "sasrec": top_rankings["sasrec"],
            "two_tower": top_rankings["two_tower"],
        },
        "budget100_four_25_each": {
            "sasrec": top_rankings["sasrec"],
            "bge_sasrec": top_rankings["bge_sasrec"],
            "two_tower": top_rankings["two_tower"],
            "lightgcn": top_rankings["lightgcn"],
        },
        "budget200_base_sasrec100_twotower100": {
            "sasrec": top_rankings["sasrec"],
            "two_tower": top_rankings["two_tower"],
        },
        "budget200_four_50_each": {
            "sasrec": top_rankings["sasrec"],
            "bge_sasrec": top_rankings["bge_sasrec"],
            "two_tower": top_rankings["two_tower"],
            "lightgcn": top_rankings["lightgcn"],
        },
    }
    quotas_by_name = {
        "budget100_base_sasrec50_twotower50": {"sasrec": 50, "two_tower": 50},
        "budget100_four_25_each": {"sasrec": 25, "bge_sasrec": 25, "two_tower": 25, "lightgcn": 25},
        "budget200_base_sasrec100_twotower100": {"sasrec": 100, "two_tower": 100},
        "budget200_four_50_each": {"sasrec": 50, "bge_sasrec": 50, "two_tower": 50, "lightgcn": 50},
    }
    for label, rankings_by_model in fusion_specs.items():
        quotas = quotas_by_name[label]
        rows = []
        for index, case in enumerate(cases):
            case_id = case["case_id"]
            fused = _fuse(
                {source: rankings[case_id] for source, rankings in rankings_by_model.items()},
                quotas,
            )
            target = item2id[case["target"]]
            rank = next(
                (position + 1 for position, item in enumerate(fused) if item == target),
                num_items + 1,
            )
            rows.append(
                {
                    "rank": rank,
                    "recall@10": 1.0 if rank <= 10 else 0.0,
                    "recall@50": 1.0 if rank <= 50 else 0.0,
                    "recall@100": 1.0 if rank <= 100 else 0.0,
                    "ndcg@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
                    "mrr@10": 1.0 / rank if rank <= 10 else 0.0,
                }
            )
        summary["fixed_budget_fusion"][label] = {
            "quotas": quotas,
            "total_budget": sum(quotas.values()),
            "recall@10": float(np.mean([row["recall@10"] for row in rows])),
            "recall@50": float(np.mean([row["recall@50"] for row in rows])),
            "recall@100": float(np.mean([row["recall@100"] for row in rows])),
            "ndcg@10": float(np.mean([row["ndcg@10"] for row in rows])),
            "mrr@10": float(np.mean([row["mrr@10"] for row in rows])),
            "top50_reachability": int(sum(row["rank"] <= 50 for row in rows)),
            "top100_reachability": int(sum(row["rank"] <= 100 for row in rows)),
        }

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            row = {"case_id": case["case_id"]}
            for name in model_results:
                row[name] = model_results[name][index]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
