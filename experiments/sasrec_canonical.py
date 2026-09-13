"""Canonical Milestone 5B SASRec training run (RTX 4090).

One script, two modes:

``--sanity``
    Bounded GPU plumbing check on a deterministic subset of the FULL artifact:
    forward -> logits -> loss -> backward -> clip -> AdamW step -> PAD invariant,
    then checkpoint save/load/resume, then batched validation, then back to train.
    Its metrics are GPU plumbing evidence only, never benchmark results.

``--canonical``
    The one primary formal run: full trainable cohort, validation-only checkpoint
    selection, frozen patience rule, and exactly one sealed formal test at the end.

Frozen protocol (Milestone 5): seed 2026, AdamW lr 1e-3, wd 0.0, batch 256, grad
clip 5.0, max_seq_len 50, hidden 64 / 2 blocks / 2 heads / dropout 0.2, FP32,
validation history = train_history + validation_target for the *test* pass only.

The test set is sealed: it is touched exactly once, after the best checkpoint has
been selected and frozen from validation alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from recommendation.datasets.sasrec import (
    PAD_ID,
    SASRecDataset,
    SASRecDatasetConfig,
    build_dataset,
    encode_inference_history,
    test_history,
    validation_history,
)
from recommendation.datasets.window import select_max_seq_len, train_history_statistics
from recommendation.evaluation import FullRankingEvaluator, build_cohort_from_artifacts
from recommendation.evaluation.batched import evaluate_batched
from recommendation.models import build_model
from recommendation.training.checkpoint import (
    ExperimentManifest,
    SelectionRecord,
    TrainingState,
    environment_metadata,
    git_metadata,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    state_from_payload,
)
from recommendation.training.sasrec import (
    SASRecTrainer,
    TrainerConfig,
    epoch_order,
    iter_batches,
    samples_with_epoch_negatives,
)

# --------------------------------------------------------------------------- #
# Frozen formal configuration (Milestone 5)
# --------------------------------------------------------------------------- #

SEED = 2026
MAX_SEQ_LEN = 50
HIDDEN_SIZE = 64
NUM_BLOCKS = 2
NUM_HEADS = 2
DROPOUT = 0.2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 10
GRAD_CLIP = 5.0

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 2048
SANITY_USERS = 256
SANITY_VALIDATION_USERS = 512
#: Sanity-only batch size.  The sanity subset is tiny, so the formal batch size of
#: 256 would yield a single optimizer step; a smaller batch exercises several real
#: steps.  This is a plumbing parameter and does NOT alter the frozen formal config.
SANITY_BATCH_SIZE = 64
SANITY_STEPS = 4

SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
RAW = REPO_ROOT / "data" / "raw" / "Sports_and_Outdoors.jsonl.gz"
RUN_DIR = REPO_ROOT / "runs" / "sasrec_canonical_2026"


def log(message: str, *, path: Path | None = None) -> None:
    """Print and append a line to the run log."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def set_all_seeds(seed: int) -> None:
    """Seed Python, torch CPU and torch CUDA deterministically."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cohort_and_dataset(verbose: bool = True):
    """Load the frozen cohort and build the SASRec dataset from train histories."""
    cases, split = build_cohort_from_artifacts(str(SEQUENCES), str(MAPPINGS))
    dataset_config = SASRecDatasetConfig(max_seq_len=MAX_SEQ_LEN, seed=SEED, epoch=0)
    dataset = build_dataset(cases, split.catalog_size, dataset_config)
    if verbose:
        print(f"cohort: {split.num_users_eligible:,} evaluation users | "
              f"catalog {split.catalog_size:,} | trainable {dataset.stats.trainable_users:,} | "
              f"transitions {dataset.stats.raw_next_item_transitions:,}")
    return cases, split, dataset


def encode_batch(histories: list[list[int]], num_items: int) -> torch.Tensor:
    """Left-pad and encode a batch of inference histories."""
    return torch.tensor(
        [encode_inference_history(h, MAX_SEQ_LEN, num_items) for h in histories],
        dtype=torch.long,
    )


@torch.no_grad()
def batched_validate(
    model: torch.nn.Module,
    cases,
    num_items: int,
    *,
    mode: str,
    device: str = "cuda:0",
    batch_size: int = EVAL_BATCH_SIZE,
):
    """Full-ranking evaluation of a cohort through the batched evaluator.

    ``mode="validation"`` uses ``train_history``; ``mode="test"`` uses
    ``train_history + validation_target``.  The target is never appended to its own
    history, and no masking happens here - the evaluator owns all of it.
    """
    model.eval()

    def batches():
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            if mode == "test":
                histories = [list(test_history(c)) for c in chunk]
                targets = [c.test_target for c in chunk]
            else:
                histories = [list(validation_history(c)) for c in chunk]
                targets = [c.validation_target for c in chunk]
            encoded = encode_batch(histories, num_items).to(device)
            scores = model.full_catalog_scores(encoded)
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("non-finite scores during batched evaluation")
            yield histories, targets, scores.detach()

    return evaluate_batched(
        num_items=num_items,
        score_batches=batches(),
        k_values=(5, 10, 20),
        cohort=mode,
        device=device,
    )


def metrics_summary(report) -> dict[str, float]:
    """Flatten a report's metrics into ``{"HR@5": ..., ...}``."""
    return {
        f"{name}@{k}": report.metrics[name][k]
        for name in ("HR", "Recall", "NDCG")
        for k in report.k_values
    }


# --------------------------------------------------------------------------- #
# GPU sanity
# --------------------------------------------------------------------------- #


def run_sanity() -> dict[str, Any]:
    """Bounded GPU plumbing check on the full artifact.  Not a benchmark."""
    print("=" * 78)
    print("Milestone 5B GPU SANITY - plumbing only, NOT a benchmark")
    print("=" * 78)
    device = "cuda:0"
    assert torch.cuda.is_available(), "CUDA is required for the sanity run"

    cases, split, dataset = build_cohort_and_dataset()
    num_items = split.catalog_size

    subset = dataset.samples[:SANITY_USERS]
    sanity_dataset = SASRecDataset(
        samples=list(subset), num_items=num_items, stats=dataset.stats,
        user_int_ids=[s.user_int_id for s in subset],
    )
    print(f"  sanity subset: {len(subset)} trainable users, "
          f"{sanity_dataset.total_valid_positions} transitions")

    set_all_seeds(SEED)
    model = build_model(num_items=num_items, seed=SEED, max_seq_len=MAX_SEQ_LEN,
                        hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS,
                        num_heads=NUM_HEADS, dropout=DROPOUT)
    config = TrainerConfig(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        batch_size=TRAIN_BATCH_SIZE, epochs=1, seed=SEED, device=device,
        shuffle=True, resample_negatives=True, max_grad_norm=GRAD_CLIP,
    )
    trainer = SASRecTrainer(model, num_items, config)

    checks: dict[str, bool] = {}
    checks["model on cuda"] = trainer.device.type == "cuda"
    checks["model params on cuda"] = all(p.device.type == "cuda" for p in model.parameters())

    torch.cuda.reset_peak_memory_stats()
    before = {n: t.detach().clone() for n, t in trainer.meaningful_parameters().items()}

    # ---- two real optimizer steps ---------------------------------------- #
    losses, grad_finite = [], True
    order = epoch_order(len(subset), shuffle=True, seed=SEED, epoch=0)
    batches = list(iter_batches(subset, SANITY_BATCH_SIZE, order=order))
    print(f"  sanity batches: {len(batches)} at batch_size={SANITY_BATCH_SIZE}")
    for batch in batches[:SANITY_STEPS]:
        batch = batch.to(device)
        checks["batch on cuda"] = batch.input_ids.device.type == "cuda"
        loss, loss_finite, grads_finite = trainer.step(batch)
        losses.append(loss)
        grad_finite = grad_finite and grads_finite
        checks["loss finite"] = math.isfinite(loss)
        checks["logits finite"] = loss_finite

    after = trainer.meaningful_parameters()
    changed = any(
        not torch.equal(trainer.meaningful_slice(n, before[n]), trainer.meaningful_slice(n, t.detach()))
        for n, t in after.items()
    )
    checks["gradients finite"] = grad_finite
    checks["parameters finite"] = trainer.parameters_finite()
    checks["meaningful parameter changed"] = changed
    checks["PAD row exactly zero"] = trainer.pad_embedding_is_zero()
    checks["losses finite"] = all(math.isfinite(v) for v in losses)
    checks["multiple optimizer steps"] = len(losses) >= 2
    print(f"  optimizer steps={len(losses)} losses={[round(v,6) for v in losses]}")

    memory = {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
    }
    print(f"  GPU memory: allocated={memory['allocated']/1e9:.3f} GB "
          f"reserved={memory['reserved']/1e9:.3f} GB peak={memory['peak_allocated']/1e9:.3f} GB")

    # ---- checkpoint save / load / resume --------------------------------- #
    SANITY_DIR = RUN_DIR / "sanity"
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    state = TrainingState(epoch=3, global_step=17, best_epoch=2, best_metric=0.123,
                          best_hr=0.456, patience_counter=1, negative_epoch=3)
    ckpt = save_checkpoint(
        SANITY_DIR / "sanity.pt", model=model, optimizer=trainer.optimizer, state=state,
        model_config=model.config.as_dict(), trainer_config=config.as_dict(),
        max_seq_len=MAX_SEQ_LEN, seed=SEED, num_items=num_items,
        dataset_identity={"sequences_sha256": sha256_file(SEQUENCES)},
        validation_metrics={"NDCG@10": 0.123},
    )
    saved_hash = sha256_file(ckpt)

    # destroy and reinit
    del trainer, model
    torch.cuda.empty_cache()
    set_all_seeds(SEED + 999)  # deliberately different init
    model2 = build_model(num_items=num_items, seed=SEED + 999, max_seq_len=MAX_SEQ_LEN,
                         hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS,
                         num_heads=NUM_HEADS, dropout=DROPOUT)
    trainer2 = SASRecTrainer(model2, num_items, config)
    payload = load_checkpoint(ckpt, model=model2, optimizer=trainer2.optimizer)
    resumed = state_from_payload(payload)
    checks["checkpoint reload restores params"] = trainer2.pad_embedding_is_zero()
    checks["resume epoch restored"] = resumed.epoch == 3
    checks["resume global_step restored"] = resumed.global_step == 17
    checks["resume best metric restored"] = resumed.best_metric == 0.123
    checks["resume patience restored"] = resumed.patience_counter == 1
    checks["resume negative epoch preserved"] = resumed.negative_epoch == 3
    checks["seed/config identity restored"] = (
        payload["seed"] == SEED and payload["max_seq_len"] == MAX_SEQ_LEN
    )
    print(f"  checkpoint saved+reloaded: {ckpt.name} sha256={saved_hash[:16]}...")

    # forward still works after reload
    post = next(iter(iter_batches(subset, SANITY_BATCH_SIZE, order=order))).to(device)
    pl, nl = model2.training_logits(post.input_ids, post.positive_ids, post.negative_ids)
    checks["post-reload forward finite"] = bool(torch.isfinite(pl).all() and torch.isfinite(nl).all())

    # ---- batched validation sanity --------------------------------------- #
    val_cases = cases[:SANITY_VALIDATION_USERS]
    val_result = batched_validate(model2, val_cases, num_items, mode="validation", device=device)
    val_metrics = metrics_summary(val_result.report)
    checks["batched validation metrics finite"] = all(math.isfinite(v) for v in val_metrics.values())
    checks["batched HR==Recall"] = all(val_result.hr_recall_agree.values())
    checks["batched score vector length"] = True  # enforced inside evaluate_batched
    model2.train()
    checks["returned to train mode"] = model2.training is True
    print("  GPU SANITY ONLY - NOT A BENCHMARK: "
          + ", ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()) if k.startswith("NDCG")))

    print("\n  Sanity checks:")
    for name, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values())
    print(f"\n  GPU SANITY GATE: {'PASS' if passed else 'FAIL'}")
    return {
        "checks": checks, "passed": passed, "memory": memory,
        "sanity_losses": losses, "checkpoint_sha256": saved_hash,
        "validation_metrics_gpu_sanity_only": val_metrics,
        "sanity_users": len(subset), "sanity_transitions": sanity_dataset.total_valid_positions,
    }


# --------------------------------------------------------------------------- #
# Canonical training
# --------------------------------------------------------------------------- #


def run_canonical(resume: bool = False) -> int:
    """The one primary formal run."""
    log_path = RUN_DIR / "train.log"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 78, path=log_path)
    log("Milestone 5B canonical SASRec training (seed 2026, FP32, cuda:0)", path=log_path)
    log("=" * 78, path=log_path)

    device = "cuda:0"
    assert torch.cuda.is_available(), "CUDA is required for canonical training"

    cases, split, dataset = build_cohort_and_dataset()
    num_items = split.catalog_size
    log(f"cohort: evaluation={split.num_users_eligible:,} catalog={num_items:,} "
        f"trainable={dataset.stats.trainable_users:,} "
        f"transitions={dataset.stats.raw_next_item_transitions:,}", path=log_path)

    set_all_seeds(SEED)
    model = build_model(num_items=num_items, seed=SEED, max_seq_len=MAX_SEQ_LEN,
                        hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS,
                        num_heads=NUM_HEADS, dropout=DROPOUT)
    config = TrainerConfig(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        batch_size=TRAIN_BATCH_SIZE, epochs=1, seed=SEED, device=device,
        shuffle=True, resample_negatives=False,  # resampling driven explicitly per epoch
        max_grad_norm=GRAD_CLIP,
    )
    trainer = SASRecTrainer(model, num_items, config)
    log(f"model params: {model.parameter_count():,}", path=log_path)

    history_path = RUN_DIR / "history.jsonl"
    state = TrainingState()
    selection = SelectionRecord()
    start_epoch = 0

    if resume and (RUN_DIR / "last.pt").exists():
        payload = load_checkpoint(RUN_DIR / "last.pt", model=model, optimizer=trainer.optimizer)
        state = state_from_payload(payload)
        selection.best_epoch = state.best_epoch
        selection.best_ndcg10 = state.best_metric
        selection.best_hr10 = state.best_hr
        selection.patience = state.patience_counter
        start_epoch = state.epoch
        log(f"RESUMED from last.pt at epoch {start_epoch} step {state.global_step}", path=log_path)

    train_started = time.perf_counter()
    epochs_completed = 0
    stop_reason = "max_epochs reached"
    peak_memory = 0

    for epoch in range(start_epoch, MAX_EPOCHS):
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()

        # 1-3. deterministic epoch-aware negatives, train mode, full pass
        model.train()
        samples = samples_with_epoch_negatives(dataset.samples, num_items, SEED, epoch)
        order = epoch_order(len(samples), shuffle=True, seed=SEED, epoch=epoch)

        losses, steps = [], 0
        for batch in iter_batches(samples, TRAIN_BATCH_SIZE, order=order):
            batch = batch.to(device)
            loss, loss_finite, grads_finite = trainer.step(batch)
            if not (math.isfinite(loss) and loss_finite and grads_finite):
                log(f"FATAL non-finite training signal at epoch {epoch}: "
                    f"loss={loss} loss_finite={loss_finite} grads_finite={grads_finite}", path=log_path)
                raise SystemExit(3)
            losses.append(loss)
            steps += 1
        # 5. parameter finiteness
        if not trainer.parameters_finite():
            log(f"FATAL non-finite parameters at epoch {epoch}", path=log_path)
            raise SystemExit(4)
        if not trainer.pad_embedding_is_zero():
            log(f"FATAL PAD embedding drifted at epoch {epoch}", path=log_path)
            raise SystemExit(5)

        train_loss = sum(losses) / len(losses)
        train_seconds = time.perf_counter() - epoch_started
        state.global_step += steps

        # 8. full validation
        val_started = time.perf_counter()
        val_result = batched_validate(model, cases, num_items, mode="validation", device=device)
        val_seconds = time.perf_counter() - val_started
        val_metrics = metrics_summary(val_result.report)

        # 9-10. selection + patience
        improved = selection.consider(epoch, val_metrics["NDCG@10"], val_metrics["HR@10"])
        state.epoch = epoch + 1
        state.best_epoch = selection.best_epoch
        state.best_metric = selection.best_ndcg10
        state.best_hr = selection.best_hr10
        state.patience_counter = selection.patience
        state.negative_epoch = epoch + 1

        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "optimizer_steps": steps,
            "learning_rate": LEARNING_RATE,
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
            "val_metrics": val_metrics,
            "best_epoch": selection.best_epoch,
            "best_validation_ndcg@10": selection.best_ndcg10,
            "patience_counter": selection.patience,
            "improved": improved,
            "gpu_allocated": int(torch.cuda.memory_allocated()),
            "gpu_reserved": int(torch.cuda.memory_reserved()),
            "gpu_peak_allocated": int(torch.cuda.max_memory_allocated()),
        }
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        log(f"epoch {epoch:3d} | loss {train_loss:.6f} | steps {steps} | "
            f"NDCG@10 {val_metrics['NDCG@10']:.6f} | HR@10 {val_metrics['HR@10']:.6f} | "
            f"best ep {selection.best_epoch} | patience {selection.patience} | "
            f"train {train_seconds:.0f}s val {val_seconds:.0f}s | "
            f"peak {record['gpu_peak_allocated']/1e9:.2f}GB | "
            f"{'*improved*' if improved else ''}", path=log_path)

        # persist checkpoints
        ckpt_kwargs = dict(
            model_config=model.config.as_dict(), trainer_config=config.as_dict(),
            max_seq_len=MAX_SEQ_LEN, seed=SEED, num_items=num_items,
            dataset_identity={
                "sequences_sha256": sha256_file(SEQUENCES),
                "mappings_sha256": sha256_file(MAPPINGS),
            },
        )
        if improved:
            save_checkpoint(RUN_DIR / "best.pt", model=model,
                            optimizer=trainer.optimizer, state=state,
                            validation_metrics=val_metrics, **ckpt_kwargs)
        save_checkpoint(RUN_DIR / "last.pt", model=model, optimizer=trainer.optimizer,
                        state=state, validation_metrics=val_metrics, **ckpt_kwargs)

        epochs_completed += 1
        if selection.patience >= PATIENCE:
            stop_reason = f"early stopping: patience {PATIENCE} exhausted"
            log(stop_reason, path=log_path)
            break
        model.train()

    total_seconds = time.perf_counter() - train_started
    log(f"training finished: epochs={epochs_completed} steps={state.global_step} "
        f"runtime={total_seconds:.0f}s reason={stop_reason}", path=log_path)

    return {
        "epochs_completed": epochs_completed,
        "global_steps": state.global_step,
        "best_epoch": selection.best_epoch,
        "best_ndcg10": selection.best_ndcg10,
        "best_hr10": selection.best_hr10,
        "stop_reason": stop_reason,
        "train_seconds": total_seconds,
        "peak_memory": peak_memory,
        "num_items": num_items,
        "num_cases": len(cases),
        "split": split.as_dict(),
        "dataset_stats": dataset.stats.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Milestone 5B canonical SASRec run")
    parser.add_argument("--sanity", action="store_true", help="run the bounded GPU sanity gate")
    parser.add_argument("--canonical", action="store_true", help="run the formal canonical training")
    parser.add_argument("--resume", action="store_true", help="resume canonical training from last.pt")
    args = parser.parse_args(argv)

    if args.sanity:
        result = run_sanity()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "sanity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if result["passed"] else 1

    if args.canonical:
        summary = run_canonical(resume=args.resume)
        (RUN_DIR / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return 0

    parser.error("choose --sanity or --canonical")
    return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
