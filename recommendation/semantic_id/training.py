"""Trainer for the Semantic-ID generator.

The objective
-------------
Per-position cross-entropy over the **code tokens** of the target item's Semantic ID::

    tokens = [ history items..., SEP, c1 target, c2 target, c3 target ]
    loss   = mean over the target positions  -log P(token | everything before it)

Only the target positions contribute.  The history tokens are context, and a padding token is never
a label - a loss that trained the model to predict its own padding would be measuring the wrong
thing and would reward degenerate continuations.

Why not in-batch negatives
--------------------------
Two-Tower's objective is contrastive because a dual encoder needs negatives to learn a shared
metric space.  A generative model's next-token likelihood already discriminates: the softmax at each
position is over the whole code vocabulary, so the "negatives" are every other code.  That
difference in objective is the point of adding this arm; making it contrastive too would blur what
is being compared.

Seeding and selection
---------------------
Initialisation and batch order are seeded and recorded.  No epoch is selected on an evaluation
split: the epoch count is a configuration value, and the trainer reports per-epoch diagnostics so a
run's convergence can be judged without touching test data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from recommendation.semantic_id.dataset import GeneratorDataset, iter_batches
from recommendation.semantic_id.generator import SemanticIdGenerator
from recommendation.training.losses import ensure_finite_logits
from recommendation.training.sasrec import resolve_device

__all__ = [
    "GeneratorTrainerConfig",
    "GeneratorTrainingError",
    "GeneratorTrainingResult",
    "build_example_arrays",
    "train_generator",
]


class GeneratorTrainingError(ValueError):
    """Raised when a generator training run is configured or fed unusably."""


@dataclass(frozen=True)
class GeneratorTrainerConfig:
    """Training hyper-parameters, all recorded in the checkpoint."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 4
    seed: int = 2026
    device: str = "cpu"
    max_grad_norm: float | None = 5.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise GeneratorTrainingError(
                f"learning_rate must be finite and > 0, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise GeneratorTrainingError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise GeneratorTrainingError("batch_size must be an int")
        if self.batch_size < 1:
            raise GeneratorTrainingError(
                f"batch_size must be >= 1, got {self.batch_size!r}"
            )
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise GeneratorTrainingError(f"epochs must be a positive int, got {self.epochs!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise GeneratorTrainingError("seed must be an int")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0
        ):
            raise GeneratorTrainingError(
                f"max_grad_norm must be finite and > 0 when set, got {self.max_grad_norm}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.device,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": "AdamW",
            "objective": "per_position_code_cross_entropy",
        }


@dataclass
class GeneratorTrainingResult:
    """The trained model plus the auditable record of the run."""

    model: SemanticIdGenerator
    config: GeneratorTrainerConfig
    dataset_stats: dict[str, Any]
    epoch_metrics: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "trainer_config": self.config.as_dict(),
            "dataset": self.dataset_stats,
            "epochs": self.epoch_metrics,
            "seconds": round(self.seconds, 3),
            "final_loss": self.epoch_metrics[-1]["loss"] if self.epoch_metrics else None,
        }


def build_example_arrays(
    model: SemanticIdGenerator,
    histories: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(input_ids, labels, loss_mask)`` for a batch of examples.

    The target codes are appended after a separator, so the model predicts ``c1`` from the history
    and the separator, ``c2`` from ``c1``, and ``c3`` from ``c2`` - the same decomposition the
    decoder walks at inference time.  ``loss_mask`` is ``True`` only at those target positions.
    """
    if len(histories) != len(targets):
        raise GeneratorTrainingError(
            f"histories and targets must be the same length, got {len(histories)} and "
            f"{len(targets)}"
        )
    levels = model.layout.levels
    sep = model.layout.sep_token
    tokens = model.build_sequences(histories)  # [B, (max_items + 1) * per_item]
    if tokens.shape[1] < levels + 1:
        raise GeneratorTrainingError("the model window is too small to hold a target")

    # Replace the final (empty) item slot with [SEP, target codes, PAD...].
    slot = model.per_item
    start = tokens.shape[1] - slot
    tokens = tokens.clone()
    tokens[:, start:] = 0
    tokens[:, start] = sep
    for offset, target in enumerate(targets):
        codes = tuple(int(value) for value in target)
        if len(codes) != levels:
            raise GeneratorTrainingError(
                f"target {offset} has {len(codes)} codes, expected levels={levels}"
            )
        tokens[offset, start + 1 : start + 1 + levels] = torch.tensor(codes, dtype=torch.long)

    # Labels are the next token at each position; only the target-code positions are supervised.
    labels = torch.zeros_like(tokens)
    labels[:, :-1] = tokens[:, 1:]
    labels[:, -1] = 0
    loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
    loss_mask[:, start : start + levels] = True
    return tokens, labels, loss_mask


def train_generator(
    model: SemanticIdGenerator,
    dataset: GeneratorDataset,
    *,
    config: GeneratorTrainerConfig | None = None,
    on_epoch: Callable[[int, dict[str, Any]], None] | None = None,
) -> GeneratorTrainingResult:
    """Train ``model`` in place and return it in ``eval()`` mode."""
    resolved = config or GeneratorTrainerConfig()
    device = resolve_device(resolved.device)
    torch.manual_seed(resolved.seed)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay
    )
    started = time.time()
    epoch_metrics: list[dict[str, Any]] = []
    steps_total = 0

    for epoch in range(resolved.epochs):
        total_loss = 0.0
        total_correct = 0.0
        total_positions = 0
        epoch_steps = 0
        for batch in iter_batches(
            dataset, batch_size=resolved.batch_size, epoch=epoch, seed=resolved.seed
        ):
            histories = [dataset.histories[index] for index in batch]
            targets = [dataset.targets[index] for index in batch]
            tokens, labels, loss_mask = build_example_arrays(model, histories, targets)
            tokens, labels, loss_mask = (
                tokens.to(device),
                labels.to(device),
                loss_mask.to(device),
            )

            logits = model(tokens)
            ensure_finite_logits("semantic-id generator logits", logits)
            # Cross-entropy over the supervised positions only.
            flat_logits = logits[loss_mask]
            flat_labels = labels[loss_mask]
            if flat_labels.numel() == 0:
                raise GeneratorTrainingError("a batch produced no supervised positions")
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if resolved.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resolved.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                predictions = flat_logits.argmax(dim=-1)
                total_correct += float((predictions == flat_labels).sum())
                total_positions += int(flat_labels.numel())
            total_loss += float(loss.detach())
            epoch_steps += 1

        if epoch_steps == 0:
            raise GeneratorTrainingError("no batch could be formed from the dataset")
        steps_total += epoch_steps
        metrics = {
            "epoch": epoch,
            "loss": round(total_loss / epoch_steps, 6),
            "code_accuracy": round(total_correct / max(1, total_positions), 6),
            "steps": epoch_steps,
            "supervised_positions": total_positions,
        }
        epoch_metrics.append(metrics)
        if on_epoch is not None:
            on_epoch(epoch, metrics)

    model.eval()
    result = GeneratorTrainingResult(
        model=model,
        config=resolved,
        dataset_stats=dict(dataset.stats),
        epoch_metrics=epoch_metrics,
        seconds=time.time() - started,
    )
    result.dataset_stats["total_steps"] = steps_total
    return result
