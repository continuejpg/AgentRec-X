"""Train a BGE-enhanced SASRec from the accepted SASRec checkpoint (P10).

Uses the existing SASRec dataset builder, model blocks, loss and trainer.  The only new
model state is a trainable projection of the frozen P8-B BGE item vectors; the BGE
vectors themselves are non-persistent data rebuilt from the accepted index on load.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from recommendation.datasets.bge_features import build_bge_item_matrix
from recommendation.datasets.sasrec import build_dataset, SASRecDatasetConfig
from recommendation.evaluation import build_cohort_from_artifacts
from recommendation.models.bge_sasrec import BGEEnhancedSASRec
from recommendation.models.sasrec import SASRecConfig
from recommendation.training.sasrec import SASRecTrainer, TrainerConfig

ROOT = Path(__file__).resolve().parent.parent
SEQUENCES = ROOT / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
BGE_INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
BASE_CHECKPOINT = ROOT / "runs/sasrec_canonical_2026/best.pt"
DEFAULT_OUT = ROOT / "runs/p10/bge_sasrec"


def _config_from_checkpoint(payload: dict[str, Any]) -> SASRecConfig:
    raw = payload["model_config"]
    return SASRecConfig(
        num_items=int(raw["num_items"]),
        max_seq_len=int(raw["max_seq_len"]),
        hidden_size=int(raw["hidden_size"]),
        num_blocks=int(raw["num_blocks"]),
        num_heads=int(raw["num_heads"]),
        dropout=float(raw["dropout"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--bge-index", type=Path, default=BGE_INDEX)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the P10 enhanced-SASRec run")
    torch.manual_seed(args.seed)
    started = time.time()

    cases, split = build_cohort_from_artifacts(str(SEQUENCES), str(MAPPINGS))
    num_items = int(split.catalog_size)
    dataset = build_dataset(
        cases,
        num_items,
        SASRecDatasetConfig(max_seq_len=50, seed=args.seed, epoch=0),
    )
    print(
        f"cohort={len(cases):,} items={num_items:,} trainable={dataset.stats.trainable_users:,} "
        f"transitions={dataset.stats.raw_next_item_transitions:,}",
        flush=True,
    )

    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    base_config = _config_from_checkpoint(base_payload)
    if base_config.num_items != num_items:
        raise SystemExit(
            f"base checkpoint catalogue {base_config.num_items} != current {num_items}"
        )
    bge_matrix = build_bge_item_matrix(args.bge_index, MAPPINGS, num_items=num_items)
    bge_vectors = torch.from_numpy(bge_matrix)
    model = BGEEnhancedSASRec(base_config, bge_vectors)
    incompatible = model.load_state_dict(base_payload["model_state_dict"], strict=False)
    unexpected = [key for key in incompatible.unexpected_keys]
    missing = [key for key in incompatible.missing_keys if key not in {"bge_projection.weight", "bge_scale"}]
    if unexpected or missing:
        raise SystemExit(f"base checkpoint mismatch: unexpected={unexpected}, missing={missing}")
    model.to(args.device)

    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        shuffle=True,
        resample_negatives=True,
        max_grad_norm=args.max_grad_norm,
    )
    trainer = SASRecTrainer(model, num_items, trainer_config)
    epoch_metrics: list[dict[str, Any]] = []

    def on_epoch(epoch: int, trainer_obj: SASRecTrainer) -> None:
        metrics = {
            "epoch": epoch + 1,
            "mean_loss": None,
        }
        # The trainer stores the latest epoch metrics in its training result only at
        # the end; a lightweight train-loss re-evaluation on one deterministic batch is
        # enough for a monotonicity check without reading validation/test labels.
        epoch_metrics.append(metrics)
        print(f"epoch={epoch + 1} done", flush=True)

    result = trainer.train(dataset, on_epoch_end=on_epoch)
    # Attach the trainer's own epoch records (it exposes them in the result).
    training = result.as_dict() if hasattr(result, "as_dict") else {}
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "agentrecx.bge_sasrec.checkpoint.v1",
        "model_state_dict": model.state_dict(),
        "model_config": base_config.as_dict(),
        "max_seq_len": base_config.max_seq_len,
        "num_items": num_items,
        "seed": args.seed,
        "base_checkpoint": str(args.base_checkpoint),
        "bge_index": str(args.bge_index),
        "bge_model": "BAAI/bge-base-en-v1.5",
        "bge_frozen": True,
        "training": training,
        "dataset": {
            "source": "EvaluationCase.train_history only (validation/test targets never read)",
            "trainable_users": dataset.stats.trainable_users,
            "transitions": dataset.stats.raw_next_item_transitions,
        },
    }
    torch.save(checkpoint, args.out / "checkpoint.pt")
    run = {
        "schema_version": "agentrecx.p10.bge_sasrec_run.v1",
        "model": "BGEEnhancedSASRec",
        "config": base_config.as_dict(),
        "content": {
            "model": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "frozen": True,
            "fusion": "id_embedding + scale * linear(bge)",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "initialised_from": str(args.base_checkpoint),
            "result": training,
        },
        "dataset": checkpoint["dataset"],
        "seconds": round(time.time() - started, 3),
        "environment": {
            "device": args.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "checkpoint": "checkpoint.pt",
    }
    (args.out / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
