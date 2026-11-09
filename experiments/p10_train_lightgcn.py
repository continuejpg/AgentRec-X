"""Train LightGCN on train-history-only user-item edges (P10).

One primary configuration, no hyper-parameter search.  Validation/test targets are never
used to build graph edges.  The model is transductive: users must exist in the training
graph; there is no fallback embedding for unseen users.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recommendation.datasets.lightgcn import build_normalized_adjacency
from recommendation.evaluation import build_cohort_from_artifacts
from recommendation.models.lightgcn import LightGCN, LightGCNConfig

ROOT = Path(__file__).resolve().parent.parent
SEQUENCES = ROOT / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
DEFAULT_OUT = ROOT / "runs/p10/lightgcn"


def build_graph(cases: list[Any], num_items: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build symmetric normalized adjacency from train histories only."""
    max_user = max(int(case.user_int_id) for case in cases)
    num_users = max_user
    edges: set[tuple[int, int]] = set()
    for case in cases:
        user_index = int(case.user_int_id) - 1
        for item_id in case.train_history:
            item_id = int(item_id)
            if not 1 <= item_id <= num_items:
                raise ValueError(
                    f"train history for user {case.user_int_id} has item {item_id} outside catalogue"
                )
            edges.add((user_index, item_id))
    edge_index = np.asarray(sorted(edges), dtype=np.int64)
    if edge_index.size == 0:
        raise ValueError("training graph has no edges")
    adjacency = build_normalized_adjacency(
        edge_index, num_users=num_users, num_items=num_items
    )
    return adjacency, torch.tensor(edge_index, dtype=torch.long), num_users


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA is required for the P10 LightGCN run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.time()
    cases, split = build_cohort_from_artifacts(str(SEQUENCES), str(MAPPINGS))
    num_items = int(split.catalog_size)
    adjacency, edge_index, num_users = build_graph(cases, num_items)
    print(
        f"cohort={len(cases):,} users={num_users:,} items={num_items:,} "
        f"edges={edge_index.shape[0]:,}",
        flush=True,
    )

    device = torch.device(args.device)
    adjacency = adjacency.to(device)
    edge_index = edge_index.to(device)
    config = LightGCNConfig(
        num_users=num_users,
        num_items=num_items,
        embedding_size=args.embedding_size,
        num_layers=args.num_layers,
    )
    model = LightGCN(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Per-user positive sets for BPR negative rejection.  This is train-only data.
    positives: dict[int, set[int]] = {}
    users_np = edge_index[:, 0].detach().cpu().numpy()
    items_np = edge_index[:, 1].detach().cpu().numpy()
    for user, item in zip(users_np.tolist(), items_np.tolist()):
        positives.setdefault(int(user), set()).add(int(item))
    rng = np.random.default_rng(args.seed)
    epochs = []
    steps_per_epoch = max(1, (len(users_np) + args.batch_size - 1) // args.batch_size)
    print(f"steps/epoch={steps_per_epoch:,}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = rng.permutation(len(users_np))
        for step in range(steps_per_epoch):
            batch = order[step * args.batch_size : (step + 1) * args.batch_size]
            if batch.size == 0:
                continue
            users = torch.tensor(users_np[batch], dtype=torch.long, device=device)
            positives_t = torch.tensor(items_np[batch], dtype=torch.long, device=device)
            negatives_np = rng.integers(1, num_items + 1, size=batch.size)
            for position, user in enumerate(users_np[batch].tolist()):
                user_positives = positives.get(int(user), set())
                while int(negatives_np[position]) in user_positives:
                    negatives_np[position] = int(rng.integers(1, num_items + 1))
            negatives = torch.tensor(negatives_np, dtype=torch.long, device=device)
            user_emb, item_emb = model.propagate(adjacency)
            loss = model.bpr_loss(user_emb, item_emb, users, positives_t, negatives)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite LightGCN loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (step + 1) % 20 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{steps_per_epoch} "
                    f"loss={np.mean(losses[-20:]):.5f}",
                    flush=True,
                )
        epochs.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "steps": len(losses),
            }
        )
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "agentrecx.lightgcn.checkpoint.v1",
                "model_state_dict": model.state_dict(),
                "model_config": config.as_dict(),
                "epoch": epoch + 1,
                "seed": args.seed,
                "num_users": num_users,
                "num_items": num_items,
                "edge_index": edge_index.detach().cpu(),
                "training": {"epochs": epochs},
            },
            args.out / "checkpoint.pt",
        )
    run = {
        "schema_version": "agentrecx.p10.lightgcn_run.v1",
        "model": "LightGCN",
        "config": config.as_dict(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "loss": "BPR",
            "negative_sampling": "uniform item; exclude user's known train positives",
            "epoch_metrics": epochs,
        },
        "dataset": {
            "source": "EvaluationCase.train_history only (no validation/test edges)",
            "users": len(cases),
            "items": num_items,
            "edges": int(edge_index.shape[0]),
            "cohort_protocol": split.protocol if hasattr(split, "protocol") else "temporal_leave_two_out",
        },
        "seconds": round(time.time() - started, 3),
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "checkpoint": "checkpoint.pt",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
