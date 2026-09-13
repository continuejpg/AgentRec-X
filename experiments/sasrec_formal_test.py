"""Sealed formal test + run manifest for the canonical Milestone 5B run.

Run **once**, after canonical training has finished and the best checkpoint has been
selected from validation alone.  It:

1. loads ``best.pt`` and records its SHA-256;
2. re-runs the full validation cohort for the record;
3. runs exactly one formal test evaluation with
   ``history = train_history + validation_target`` and ``target = test_target``;
4. writes the formal run manifest.

The test target is never appended to its own history, no refitting happens on the
validation interaction, and no training epoch follows checkpoint selection.

Usage::

    .venv/bin/python -m experiments.sasrec_formal_test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from experiments.sasrec_canonical import (
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    NUM_BLOCKS,
    NUM_HEADS,
    PATIENCE,
    SEED,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    GRAD_CLIP,
    MAPPINGS,
    RUN_DIR,
    SEQUENCES,
    batched_validate,
    build_cohort_and_dataset,
    metrics_summary,
)
from recommendation.datasets.window import select_max_seq_len, train_history_statistics
from recommendation.evaluation import build_cohort_from_artifacts
from recommendation.models import build_model
from recommendation.training.checkpoint import (
    MANIFEST_FORMAT,
    environment_metadata,
    git_metadata,
    load_checkpoint,
    sha256_file,
    state_from_payload,
    write_json_atomic,
)
from recommendation.training.sasrec import SASRecTrainer, TrainerConfig


def main(argv: list[str] | None = None) -> int:
    """Run the sealed formal test and write the manifest."""
    parser = argparse.ArgumentParser(description="Milestone 5B sealed formal test")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    if not best_path.exists():
        print(f"missing best checkpoint: {best_path}", file=sys.stderr)
        return 2

    started = time.time()
    print("=" * 78)
    print("Milestone 5B FORMAL TEST - exactly one sealed test evaluation")
    print("=" * 78)

    cases, split, dataset = build_cohort_and_dataset()
    num_items = split.catalog_size

    history_stats = train_history_statistics(cases)
    selected_window, window_evidence = select_max_seq_len(cases)
    assert selected_window == MAX_SEQ_LEN, (
        f"frozen max_seq_len {MAX_SEQ_LEN} disagrees with the rule's {selected_window}"
    )

    # ---- 2-3. load the frozen best checkpoint ---------------------------- #
    model = build_model(num_items=num_items, seed=SEED, max_seq_len=MAX_SEQ_LEN,
                        hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS,
                        num_heads=NUM_HEADS, dropout=DROPOUT)
    config = TrainerConfig(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        batch_size=TRAIN_BATCH_SIZE, epochs=1, seed=SEED, device=args.device,
        shuffle=True, resample_negatives=False, max_grad_norm=GRAD_CLIP,
    )
    trainer = SASRecTrainer(model, num_items, config)
    payload = load_checkpoint(best_path, model=model, optimizer=trainer.optimizer)
    state = state_from_payload(payload)
    best_hash = sha256_file(best_path)
    last_hash = sha256_file(last_path) if last_path.exists() else None
    print(f"best.pt  sha256 = {best_hash}")
    print(f"last.pt  sha256 = {last_hash}")
    print(f"best epoch (from checkpoint) = {state.best_epoch} | "
          f"patience state = {state.patience_counter} | negative epoch = {state.negative_epoch}")

    # ---- 5. recordkeeping validation ------------------------------------ #
    val_started = time.perf_counter()
    val_result = batched_validate(model, cases, num_items, mode="validation", device=args.device)
    val_seconds = time.perf_counter() - val_started
    val_metrics = metrics_summary(val_result.report)
    print(f"\nfinal validation (recordkeeping): {val_result.report.num_cases:,} cases "
          f"in {val_seconds:.1f}s")
    for k in (5, 10, 20):
        print(f"  NDCG@{k:<2d} {val_metrics[f'NDCG@{k}']:.6f}   "
              f"HR@{k:<2d} {val_metrics[f'HR@{k}']:.6f}   "
              f"Recall@{k:<2d} {val_metrics[f'Recall@{k}']:.6f}")

    # ---- 6. the ONE formal test ----------------------------------------- #
    print("\n--- SEALED TEST (first and only evaluation of the test split) ---")
    test_started = time.perf_counter()
    test_result = batched_validate(model, cases, num_items, mode="test", device=args.device)
    test_seconds = time.perf_counter() - test_started
    test_metrics = metrics_summary(test_result.report)
    for k in (5, 10, 20):
        print(f"  NDCG@{k:<2d} {test_metrics[f'NDCG@{k}']:.6f}   "
              f"HR@{k:<2d} {test_metrics[f'HR@{k}']:.6f}   "
              f"Recall@{k:<2d} {test_metrics[f'Recall@{k}']:.6f}")

    checks = {
        "test cases == evaluation cohort": test_result.report.num_cases == len(cases),
        "validation cases == evaluation cohort": val_result.report.num_cases == len(cases),
        "all test metrics finite": all(math.isfinite(v) for v in test_metrics.values()),
        "all validation metrics finite": all(math.isfinite(v) for v in val_metrics.values()),
        "HR == Recall @5": test_metrics["HR@5"] == test_metrics["Recall@5"],
        "HR == Recall @10": test_metrics["HR@10"] == test_metrics["Recall@10"],
        "HR == Recall @20": test_metrics["HR@20"] == test_metrics["Recall@20"],
        "test metrics in [0,1]": all(0.0 <= v <= 1.0 for v in test_metrics.values()),
    }
    print("\n  checks:")
    for name, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")

    # ---- training history ------------------------------------------------ #
    history_path = run_dir / "history.jsonl"
    history = []
    if history_path.exists():
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs_completed = len(history)
    peak_memory = max((h.get("gpu_peak_allocated", 0) for h in history), default=0)
    train_seconds = sum(h.get("train_seconds", 0.0) for h in history)
    val_runtimes = [h.get("validation_seconds", 0.0) for h in history]
    best_entry = max(history, key=lambda h: h["val_metrics"]["NDCG@10"]) if history else {}
    stop_reason = (
        f"early stopping: patience {PATIENCE} exhausted"
        if history and history[-1]["patience_counter"] >= PATIENCE
        else f"completed {epochs_completed} epochs (max_epochs {200})"
    )

    manifest = {
        "format": MANIFEST_FORMAT,
        "run_id": "m5b-sasrec-canonical-seed2026",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seed": SEED,
        "git": git_metadata(REPO_ROOT),
        "environment": environment_metadata(),
        "raw_data": {
            "path": str(RAW := REPO_ROOT / "data" / "raw" / "Sports_and_Outdoors.jsonl.gz"),
            "sha256": sha256_file(RAW),
            "size_bytes": RAW.stat().st_size,
        },
        "processed_artifacts": {
            "sequences": {"path": str(SEQUENCES), "sha256": sha256_file(SEQUENCES)},
            "mappings": {"path": str(MAPPINGS), "sha256": sha256_file(MAPPINGS)},
        },
        "cohort": split.as_dict(),
        "num_users": split.num_users_total,
        "num_items": num_items,
        "evaluation_users": split.num_users_eligible,
        "trainable_users": dataset.stats.trainable_users,
        "train_transitions": dataset.stats.raw_next_item_transitions,
        "zero_transition_users": dataset.stats.users_with_zero_transitions,
        "train_history_statistics": history_stats,
        "max_seq_len": MAX_SEQ_LEN,
        "max_seq_len_selection": window_evidence,
        "model_config": payload.get("model_config"),
        "optimizer_config": config.as_dict(),
        "training_batch_size": TRAIN_BATCH_SIZE,
        "evaluation_batch_size": 2048,
        "precision": "fp32",
        "training": {
            "epochs_completed": epochs_completed,
            "global_steps": history[-1]["optimizer_steps"] * epochs_completed if history else 0,
            "stop_reason": stop_reason,
            "best_epoch": best_entry.get("epoch"),
            "best_validation_ndcg@10": best_entry.get("val_metrics", {}).get("NDCG@10"),
            "best_validation_hr@10": best_entry.get("val_metrics", {}).get("HR@10"),
            "train_seconds": train_seconds,
            "validation_runtimes": val_runtimes,
            "peak_gpu_memory_bytes": peak_memory,
            "history_path": str(history_path),
        },
        "formal_validation_metrics": val_metrics,
        "formal_test_metrics": test_metrics,
        "validation_runtime_seconds": val_seconds,
        "test_runtime_seconds": test_seconds,
        "test_throughput_users_per_second": test_result.report.num_cases / test_seconds,
        "checkpoints": {
            "best": {"path": str(best_path), "sha256": best_hash},
            "last": {"path": str(last_path), "sha256": last_hash},
            "best_epoch": state.best_epoch,
            "best_metric": state.best_metric,
        },
        "test_sealed": {
            "test_used_for_selection": False,
            "statement": (
                "The formal test set was not used for model selection, early stopping, "
                "hyperparameter selection, or restart decisions."
            ),
        },
        "verification": {
            "pytest": "see final report",
            "compileall": "see final report",
        },
        "itemcf_comparison": "PENDING (no same-artifact full-data ItemCF benchmark exists)",
        "checks": checks,
    }

    manifest_path = run_dir / "run.json"
    write_json_atomic(manifest, manifest_path)
    print(f"\nmanifest written: {manifest_path}")
    print(f"ALL CHECKS PASSED: {all(checks.values())}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
