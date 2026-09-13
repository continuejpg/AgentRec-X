"""Deterministic tiny-overfit experiment + bounded real-artifact smoke (Milestone 4).

Two clearly separated runs:

1. **Tiny synthetic overfit** - the milestone's central acceptance evidence.  A tiny
   SASRec on a hand-inspectable fixture must drive the training loss far below its
   initial value and rank every training positive above its paired negative.
2. **Bounded real-artifact smoke** - a short CPU smoke on a subset of the existing
   Milestone 1.5 / 2A engineering artifact, followed by validation/test scoring
   through the frozen evaluator.

Usage::

    .venv/bin/python -m experiments.sasrec_tiny_overfit
    .venv/bin/python -m experiments.sasrec_tiny_overfit --json out.json

NOTHING here is a benchmark.  No recommendation-quality conclusion may be drawn from
either run, and the real-artifact smoke is not the Milestone 5 training run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

from recommendation.datasets.sasrec import (  # noqa: E402
    PAD_ID,
    SASRecDataset,
    SASRecDatasetConfig,
    build_dataset,
)
from recommendation.evaluation import (  # noqa: E402
    EvaluationCase,
    FullRankingEvaluator,
    build_cohort_from_artifacts,
)
from recommendation.models.sasrec import build_model, make_score_fn  # noqa: E402
from recommendation.training.losses import ranking_accuracy  # noqa: E402
from recommendation.training.sasrec import (  # noqa: E402
    SASRecTrainer,
    TrainerConfig,
    train_sasrec,
)

DEFAULT_SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_sequences.json"
DEFAULT_MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sample_mappings.json"

# --------------------------------------------------------------------------- #
# Tiny synthetic overfit fixture (self-contained; mirrors the unit-test fixture)
# --------------------------------------------------------------------------- #

TINY_NUM_ITEMS = 12
TINY_MAX_SEQ_LEN = 5
TINY_HIDDEN = 16
TINY_BLOCKS = 1
TINY_HEADS = 2
TINY_DROPOUT = 0.0
TINY_EPOCHS = 200
TINY_BATCH_SIZE = 4
TINY_LEARNING_RATE = 0.1
TINY_WEIGHT_DECAY = 0.0
TINY_SEED = 0

#: Every distinct input item has exactly ONE next item across all users, so the
#: fixture is unambiguously memorisable.  The last two items of each sequence are the
#: held-out validation and test targets and never enter the fit data.
TINY_SEQUENCES = {
    "u1": [1, 2, 3, 4, 5, 6],
    "u2": [1, 2, 3, 4, 6, 7],
    "u3": [2, 3, 4, 5, 8, 9],
    "u4": [3, 4, 5, 6, 9, 10],
    "u5": [4, 5, 6, 7, 11, 12],
    "u6": [8, 9, 10, 11, 3, 4],
}

# Real-artifact smoke settings (documented bounded run, NOT Milestone 5).
SMOKE_TRAINABLE_USERS = 128
SMOKE_MAX_SEQ_LEN = 20          # existing development setting; NOT a benchmark value
SMOKE_EPOCHS = 2
SMOKE_BATCH_SIZE = 32
SMOKE_LEARNING_RATE = 0.01
SMOKE_HIDDEN = 32
SMOKE_SEED = 4242


def tiny_cases() -> list[EvaluationCase]:
    """Build the tiny cohort from :data:`TINY_SEQUENCES`."""
    cases = []
    for index, (user_id, items) in enumerate(sorted(TINY_SEQUENCES.items()), start=1):
        cases.append(
            EvaluationCase(
                user_id=user_id,
                user_int_id=index,
                train_history=tuple(items[:-2]),
                validation_target=items[-2],
                test_target=items[-1],
                sequence_length=len(items),
            )
        )
    return cases


def run_tiny_overfit(seed: int = TINY_SEED) -> dict[str, Any]:
    """Run the deterministic tiny-overfit experiment and return its evidence."""
    config = SASRecDatasetConfig(max_seq_len=TINY_MAX_SEQ_LEN, seed=11, epoch=0)
    dataset = build_dataset(tiny_cases(), TINY_NUM_ITEMS, config)

    model = build_model(
        num_items=TINY_NUM_ITEMS,
        seed=seed,
        max_seq_len=TINY_MAX_SEQ_LEN,
        hidden_size=TINY_HIDDEN,
        num_blocks=TINY_BLOCKS,
        num_heads=TINY_HEADS,
        dropout=TINY_DROPOUT,
    )
    trainer_config = TrainerConfig(
        learning_rate=TINY_LEARNING_RATE,
        weight_decay=TINY_WEIGHT_DECAY,
        batch_size=TINY_BATCH_SIZE,
        epochs=TINY_EPOCHS,
        seed=seed,
        shuffle=True,
        resample_negatives=False,  # fixed negatives -> fixed optimisation target
    )
    trainer = SASRecTrainer(model, TINY_NUM_ITEMS, trainer_config)

    initial_parameters = {
        name: tensor.detach().clone() for name, tensor in model.named_parameters()
    }
    result = train_sasrec(model, dataset, trainer_config)

    positives, negatives = trainer.batch_logits(dataset.samples)
    beaten = int((positives > negatives).sum().item())
    total = int(positives.numel())

    return {
        "fixture": {
            "sequences": {k: list(v) for k, v in TINY_SEQUENCES.items()},
            "num_items": TINY_NUM_ITEMS,
            "num_trainable_users": len(dataset.samples),
            "num_transitions": dataset.total_valid_positions,
            "max_seq_len": TINY_MAX_SEQ_LEN,
        },
        "model_config": model.config.as_dict(),
        "optimizer_config": trainer_config.as_dict(),
        "result": result.as_dict(),
        "logits": {
            "num_positions": total,
            "positions_beaten": beaten,
            "ranking_accuracy": ranking_accuracy(positives, negatives),
            "min_logit_gap": float((positives - negatives).min().item()),
            "max_logit_gap": float((positives - negatives).max().item()),
        },
        "initial_parameters_digest": {
            name: float(tensor.double().sum().item()) for name, tensor in initial_parameters.items()
        },
        "criteria": {
            "loss_reduction_ratio": result.loss_reduction_ratio,
            "ratio_at_most_0_25": result.loss_reduction_ratio <= 0.25,
            "all_positions_beaten": beaten == total,
            "losses_finite": result.all_losses_finite,
            "parameters_finite": result.parameters_finite_after_training,
            "pad_embedding_zero": result.pad_embedding_zero_after_training,
            "parameter_update_observed": result.parameter_update_observed,
            "nonzero_gradient_observed": result.nonzero_gradient_observed,
        },
    }


def run_real_smoke(
    sequences_path: Path,
    mappings_path: Path,
    *,
    users: int = SMOKE_TRAINABLE_USERS,
    seed: int = SMOKE_SEED,
) -> dict[str, Any]:
    """Run the bounded real-artifact smoke training + inference checks."""
    cases, split = build_cohort_from_artifacts(str(sequences_path), str(mappings_path))
    config = SASRecDatasetConfig(max_seq_len=SMOKE_MAX_SEQ_LEN, seed=seed, epoch=0)
    full = build_dataset(cases, split.catalog_size, config)

    subset = list(full.samples[:users])
    subset_dataset = SASRecDataset(
        samples=subset,
        num_items=full.num_items,
        stats=full.stats,
        user_int_ids=[s.user_int_id for s in subset],
    )

    model = build_model(
        num_items=split.catalog_size,
        seed=seed + 1,
        max_seq_len=SMOKE_MAX_SEQ_LEN,
        hidden_size=SMOKE_HIDDEN,
        num_blocks=1,
        num_heads=2,
        dropout=0.1,
    )
    trainer_config = TrainerConfig(
        learning_rate=SMOKE_LEARNING_RATE,
        weight_decay=0.0,
        batch_size=SMOKE_BATCH_SIZE,
        epochs=SMOKE_EPOCHS,
        seed=seed,
        shuffle=True,
        resample_negatives=True,   # exercise epoch-aware deterministic resampling
    )
    result = train_sasrec(model, subset_dataset, trainer_config)

    # ---- small deterministic evaluation subset --------------------------- #
    evaluator = FullRankingEvaluator(num_items=split.catalog_size, k_values=(5, 10, 20))
    score_fn = make_score_fn(model, SMOKE_MAX_SEQ_LEN, split.catalog_size)

    trainable_cases = [c for c in cases if len(c.train_history) >= 2][:8]
    zero_transition_cases = [c for c in cases if len(c.train_history) == 1][:1]
    eval_subset = trainable_cases + zero_transition_cases

    validation = evaluator.evaluate(eval_subset, score_fn, mode="validation")
    test = evaluator.evaluate(eval_subset, score_fn, mode="test")

    # the zero-transition user must remain evaluable
    zero = zero_transition_cases[0]
    zero_scores = score_fn(zero.validation_history, zero.validation_target)
    evaluator.validate_scores(zero_scores)

    return {
        "cohort": split.as_dict(),
        "subset": {
            "trainable_users_selected": len(subset),
            "transitions": subset_dataset.total_valid_positions,
            "max_seq_len": SMOKE_MAX_SEQ_LEN,
        },
        "model_config": model.config.as_dict(),
        "optimizer_config": trainer_config.as_dict(),
        "result": result.as_dict(),
        "evaluation_subset": {
            "num_cases": len(eval_subset),
            "trainable": len(trainable_cases),
            "zero_transition": len(zero_transition_cases),
            "zero_transition_user_evaluable": True,
            "zero_transition_score_vector_length": len(zero_scores),
            "zero_transition_scores_finite": all(math.isfinite(v) for v in zero_scores),
        },
        "smoke_diagnostics_validation": validation.report.as_dict(),
        "smoke_diagnostics_test": test.report.as_dict(),
        "diagnostic_label": "SMOKE DIAGNOSTIC ONLY - NOT A BENCHMARK RESULT",
    }


def print_report(tiny: dict[str, Any], smoke: dict[str, Any] | None) -> None:
    """Render both runs as readable text."""
    fixture, result, logits, criteria = (
        tiny["fixture"],
        tiny["result"],
        tiny["logits"],
        tiny["criteria"],
    )
    print("=" * 78)
    print("Milestone 4 - tiny deterministic overfit (acceptance evidence)")
    print("NO TRAINING BENCHMARK. Loss/logit evidence only; no quality claim.")
    print("=" * 78)
    print()
    print("Tiny fixture")
    print(f"  users / transitions / catalog : {fixture['num_trainable_users']} / "
          f"{fixture['num_transitions']} / {fixture['num_items']}")
    for name, items in fixture["sequences"].items():
        print(f"    {name}: {items}   train_history={items[:-2]}")
    print()
    print("Tiny model / optimizer")
    mc, oc = tiny["model_config"], tiny["optimizer_config"]
    print(f"  hidden={mc['hidden_size']} blocks={mc['num_blocks']} heads={mc['num_heads']} "
          f"dropout={mc['dropout']} max_seq_len={mc['max_seq_len']}")
    print(f"  {oc['optimizer']} lr={oc['learning_rate']} wd={oc['weight_decay']} "
          f"batch={oc['batch_size']} epochs={oc['epochs']} seed={oc['seed']}")
    print()
    print("Tiny overfit result")
    print(f"  initial loss                  : {result['initial_loss']:.6f}")
    print(f"  final loss                    : {result['final_loss']:.8f}")
    print(f"  loss reduction ratio          : {result['loss_reduction_ratio']:.6f} "
          f"(<= 0.25 required)")
    print(f"  optimizer steps               : {result['optimizer_steps']}")
    print(f"  positives > negatives         : {logits['positions_beaten']}/{logits['num_positions']} "
          f"({logits['ranking_accuracy']:.4f})")
    print(f"  min / max logit gap           : {logits['min_logit_gap']:+.4f} / {logits['max_logit_gap']:+.4f}")
    print()
    print("Acceptance criteria")
    for name, ok in criteria.items():
        if name == "loss_reduction_ratio":
            continue
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()

    if smoke is not None:
        sub, sres = smoke["subset"], smoke["result"]
        print("=" * 78)
        print("Bounded real-artifact smoke training (pipeline integration only)")
        print("NOT full training. NOT Milestone 5. NOT a benchmark.")
        print("=" * 78)
        print()
        print(f"  cohort evaluation users       : {smoke['cohort']['num_users_eligible']}")
        print(f"  smoke subset (trainable users): {sub['trainable_users_selected']}")
        print(f"  smoke subset transitions      : {sub['transitions']}")
        print(f"  max_seq_len (dev setting)     : {sub['max_seq_len']}")
        print(f"  epochs / optimizer steps      : {smoke['optimizer_config']['epochs']} / "
              f"{sres['optimizer_steps']}")
        print(f"  initial loss                  : {sres['initial_loss']:.6f}")
        print(f"  final loss                    : {sres['final_loss']:.6f}")
        print(f"  losses finite / grads finite  : {sres['all_losses_finite']} / {sres['all_gradients_finite']}")
        print(f"  parameter update occurred     : {sres['parameter_update_observed']}")
        print(f"  PAD embedding still zero      : {sres['pad_embedding_zero_after_training']}")
        print(f"  runtime                       : {sres['seconds']:.2f}s")
        print()
        ev = smoke["evaluation_subset"]
        print(f"  evaluation subset             : {ev['num_cases']} cases "
              f"({ev['trainable']} trainable + {ev['zero_transition']} zero-transition)")
        print(f"  zero-transition user scoreable: {ev['zero_transition_user_evaluable']} "
              f"(vector length {ev['zero_transition_score_vector_length']}, "
              f"finite={ev['zero_transition_scores_finite']})")
        print()
        print("  SMOKE DIAGNOSTIC ONLY - NOT A BENCHMARK RESULT")
        for label, key in (("validation", "smoke_diagnostics_validation"),
                           ("test", "smoke_diagnostics_test")):
            report = smoke[key]
            print(f"    {label}: cases={report['num_cases']} "
                  f"HR@20={report['metrics']['HR']['@20']:.6f} "
                  f"NDCG@20={report['metrics']['NDCG']['@20']:.6f}")
        print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns 0 when every tiny-overfit criterion passes."""
    parser = argparse.ArgumentParser(description="SASRec tiny overfit + bounded smoke (Milestone 4)")
    parser.add_argument("--seed", type=int, default=TINY_SEED)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--mappings", type=Path, default=DEFAULT_MAPPINGS)
    parser.add_argument("--skip-real-smoke", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    tiny = run_tiny_overfit(seed=args.seed)
    smoke = None
    if not args.skip_real_smoke and args.sequences.exists() and args.mappings.exists():
        smoke = run_real_smoke(args.sequences, args.mappings)

    print_report(tiny, smoke)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"tiny_overfit": tiny, "real_smoke": smoke}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    passed = all(v for k, v in tiny["criteria"].items() if k != "loss_reduction_ratio")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
