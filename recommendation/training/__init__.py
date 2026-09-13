"""Minimal training infrastructure (Milestone 4).

Provides the SASRec training objective and a small deterministic trainer.  It
contains **no** evaluation logic: masking, ranking and metrics stay in
:mod:`recommendation.evaluation`.

Responsibility split::

    Dataset    -> training examples / negatives
    Model      -> hidden states / logits / scores
    Trainer    -> loss / backward / optimizer
    Evaluator  -> masking / ranking / HR / Recall / NDCG

Deliberately not implemented in this milestone: schedulers, mixed precision,
distributed or multi-GPU training, checkpoint management, early stopping, and
experiment tracking.
"""

from __future__ import annotations

from .losses import (
    ZERO_LOGIT_LOSS,
    TrainingLossError,
    binary_logistic_loss,
    ensure_finite_logits,
    logit_diagnostics,
    mean_logit_gap,
    per_position_loss,
    ranking_accuracy,
    stable_softplus,
)
from .checkpoint import (
    CHECKPOINT_FORMAT,
    MANIFEST_FORMAT,
    CheckpointError,
    ExperimentManifest,
    SelectionRecord,
    TrainingState,
    environment_metadata,
    load_checkpoint,
    save_checkpoint,
    state_from_payload,
)
from .sasrec import (
    SUPPORTED_DEVICES,
    EpochMetrics,
    SASRecBatch,
    SASRecTrainer,
    TrainerConfig,
    TrainingError,
    TrainingResult,
    collate_samples,
    epoch_order,
    iter_batches,
    resolve_device,
    samples_with_epoch_negatives,
    train_sasrec,
)

__all__ = [
    "ZERO_LOGIT_LOSS",
    "CHECKPOINT_FORMAT",
    "MANIFEST_FORMAT",
    "CheckpointError",
    "EpochMetrics",
    "ExperimentManifest",
    "SASRecBatch",
    "SelectionRecord",
    "TrainingState",
    "environment_metadata",
    "load_checkpoint",
    "save_checkpoint",
    "state_from_payload",
    "SUPPORTED_DEVICES",
    "SASRecTrainer",
    "TrainerConfig",
    "TrainingError",
    "TrainingLossError",
    "TrainingResult",
    "binary_logistic_loss",
    "collate_samples",
    "ensure_finite_logits",
    "epoch_order",
    "iter_batches",
    "logit_diagnostics",
    "mean_logit_gap",
    "per_position_loss",
    "ranking_accuracy",
    "resolve_device",
    "samples_with_epoch_negatives",
    "stable_softplus",
    "train_sasrec",
]

__version__ = "0.1.0"
