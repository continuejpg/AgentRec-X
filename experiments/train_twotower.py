"""Train and checkpoint the Two-Tower retrieval baseline on the public dataset.

What it does
------------
1. loads the preprocessed Amazon Reviews 2023 *Sports & Outdoors* artifacts through the
   accepted loader (``recommendation.evaluation.split``);
2. takes the **same deterministic 20 000-user cohort** the Phase-5 benchmark uses
   (``experiments.benchmark_public.cohort_from_cases``), so the arm is trained and
   evaluated on one population;
3. builds the categorical item metadata from the normalised catalogue artifact;
4. builds training pairs from ``train_history`` only and trains
   :class:`~recommendation.models.twotower.TwoTower` with the in-batch sampled-softmax
   objective;
5. writes a checkpoint in the accepted checkpoint envelope plus a JSON run record.

What it deliberately does not do
--------------------------------
No validation or test label is read, and no epoch is selected on an evaluation split: the
epoch count is a recorded configuration value.  That is the same boundary the accepted
SASRec protocol enforces; see ``docs/EXPERIMENTS.md``.

Usage::

    python -m experiments.train_twotower --out runs/twotower_public_2026
    python -m experiments.train_twotower --cohort 20000 --epochs 8 --batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.datasets.twotower import (  # noqa: E402
    TwoTowerDatasetConfig,
    build_metadata_categories,
    build_training_pairs,
    metadata_tensor,
    write_metadata_categories,
)
from recommendation.datasets.sasrec import PAD_ID  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402
from recommendation.models.twotower import TwoTower, TwoTowerConfig  # noqa: E402
from recommendation.training.checkpoint import (  # noqa: E402
    TrainingState,
    environment_metadata,
    git_metadata,
    save_checkpoint,
)
from recommendation.training.twotower import (  # noqa: E402
    TwoTowerTrainerConfig,
    train_two_tower,
)

#: Default output directory.  ``runs/`` is git-ignored: a checkpoint is a regenerable
#: experiment output, not source.
DEFAULT_OUT = REPO_ROOT / "runs" / "twotower_public_2026"

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
PRODUCTS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Two-Tower retrieval baseline")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cohort", type=int, default=20_000, help="users in the train cohort")
    parser.add_argument("--cohort-seed", type=int, default=20260201)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--pooling", default="gru", choices=("gru", "mean"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metadata-min-frequency", type=int, default=20)
    parser.add_argument("--no-logq", action="store_true", help="disable the logQ correction")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    cases, split_report = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    num_items = split_report.catalog_size

    # The cohort is the benchmark's cohort.  Imported lazily so this command does not pull
    # the benchmark module's optional dependencies unless it runs.
    from experiments.benchmark_public import cohort_from_cases

    selection = cohort_from_cases(cases, size=args.cohort, seed=args.cohort_seed)
    cohort = list(selection["cases"])
    if not args.quiet:
        print(f"cohort={len(cohort):,} catalogue={num_items:,} eligible={len(cases):,}")

    # ---- categorical item metadata -------------------------------------- #
    from recommendation.catalog import MetadataIndex  # local: needs the catalogue artifact

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    metadata = MetadataIndex.load(PRODUCTS)
    categories = build_metadata_categories(
        metadata.records,
        mappings["item2id"],
        num_items=num_items,
        min_frequency=args.metadata_min_frequency,
    )
    category_path = args.out / "metadata_categories.json"
    write_metadata_categories(categories, category_path)
    if not args.quiet:
        print(f"metadata cardinalities={categories['cardinalities']}")
        print(f"metadata coverage={categories['coverage']}")

    # ---- pairs ----------------------------------------------------------- #
    dataset_config = TwoTowerDatasetConfig(
        max_seq_len=args.max_seq_len, seed=args.seed
    )
    dataset = build_training_pairs(cohort, num_items=num_items, config=dataset_config)
    if not args.quiet:
        print(f"training pairs={len(dataset):,} stats={json.dumps(dataset.stats, sort_keys=True)}")

    # ---- model ----------------------------------------------------------- #
    model_config = TwoTowerConfig(
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        pooling=args.pooling,
        num_categories=tuple(categories["cardinalities"]),
        logq_correction=not args.no_logq,
    )
    import torch

    torch.manual_seed(args.seed)
    model = TwoTower(model_config)
    model.set_categories(metadata_tensor(categories))

    trainer_config = TwoTowerTrainerConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        logq_correction=not args.no_logq,
    )

    def on_epoch(epoch: int, metrics: dict[str, Any]) -> None:
        if not args.quiet:
            print(
                f"epoch {metrics['epoch']:>3} loss={metrics['loss']:.6f} "
                f"acc={metrics['in_batch_accuracy']:.4f} steps={metrics['steps']} "
                f"pairs={metrics['pairs_used']} temp={metrics['temperature']:.4f}",
                flush=True,
            )

    result = train_two_tower(model, dataset, config=trainer_config, on_epoch=on_epoch)

    # ---- checkpoint ------------------------------------------------------ #
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        state=TrainingState(
            epoch=trainer_config.epochs - 1,
            global_step=int(result.dataset_stats.get("total_steps", 0)),
        ),
        model_config=model_config.as_dict(),
        trainer_config=trainer_config.as_dict(),
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        num_items=num_items,
        dataset_identity={
            "source": "Amazon Reviews 2023 / Sports_and_Outdoors",
            "cohort_users": len(cohort),
            "cohort_seed": args.cohort_seed,
            "pairs": len(dataset),
            "pairs_stats": dataset.stats,
            "labels_from": "EvaluationCase.train_history only",
        },
        extra={
            "architecture": "two_tower_dssm",
            "objective": "in_batch_sampled_softmax",
            "embedding_size": args.embedding_size,
            "hidden_size": args.hidden_size,
            "pooling": args.pooling,
            "metadata_artifact": category_path.name,
            "metadata_attributes": categories["attributes"],
            "metadata_cardinalities": categories["cardinalities"],
            "metadata_coverage": categories["coverage"],
            "padding_id": PAD_ID,
        },
    )

    run_record = {
        "run_id": f"twotower-public-{args.pooling}-seed{args.seed}",
        "architecture": "two_tower_dssm",
        "objective": "in_batch_sampled_softmax",
        "model_config": model_config.as_dict(),
        "trainer_config": trainer_config.as_dict(),
        "training": result.as_dict(),
        "dataset": {
            "name": "Amazon Reviews 2023 / Sports_and_Outdoors",
            "num_users_total": split_report.num_users_total,
            "num_users_eligible": split_report.num_users_eligible,
            "catalog_size": num_items,
            "cohort_users": len(cohort),
            "cohort_seed": args.cohort_seed,
            "cohort_description": selection["description"],
            "split_report": split_report.as_dict(),
            "labels_from": "EvaluationCase.train_history only",
        },
        "metadata": {
            "attributes": categories["attributes"],
            "cardinalities": categories["cardinalities"],
            "coverage": categories["coverage"],
            "min_frequency": categories["min_frequency"],
        },
        "checkpoint": {"path": str(checkpoint_path), "format": "agentrecx.sasrec.checkpoint.v1"},
        "environment": environment_metadata(),
        "git": git_metadata(REPO_ROOT),
        "wall_seconds": round(time.time() - started, 3),
    }
    (args.out / "run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not args.quiet:
        print(f"wrote {checkpoint_path} and {args.out / 'run.json'}")
        print(f"wall={run_record['wall_seconds']}s final_loss={result.as_dict()['final_loss']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
