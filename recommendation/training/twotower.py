"""Two-tower trainer: deterministic, offline, checkpointed.

The objective itself lives in :mod:`recommendation.training.twotower_loss`; this module is
the loop around it.

Design decisions worth stating
------------------------------
**The item tower is supervised through the batch's own positives only.**  The ``B - 1``
in-batch negatives are other rows' positives, which already carry gradients, so there is no
second forward pass over all items.  The batch is duplicate-free by construction
(:func:`~recommendation.datasets.twotower.iter_training_batches`), which is what makes the
diagonal labelling correct.

**No validation or test label is ever read.**  The trainer consumes pre-built pairs, and the
builder reads ``train_history`` only.  There is no "pick the best epoch on validation NDCG"
here: selecting an epoch on an evaluation split is exactly the leakage this repository's
protocol forbids, so the epoch count is a recorded configuration value instead.

**Every stochastic decision is seeded and recorded.**  Initialisation, batch order and the
optional pair cap all derive from ``TwoTowerTrainerConfig.seed``, and the resolved config
plus per-epoch diagnostics go into the checkpoint.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from recommendation.datasets.twotower import (
    TwoTowerDataset,
    iter_training_batches,
    item_prior_from_pairs,
)
from recommendation.models.twotower import TwoTower
from recommendation.training.losses import ensure_finite_logits
from recommendation.training.sasrec import resolve_device
from recommendation.training.twotower_loss import sampled_softmax_loss

__all__ = [
    "TwoTowerTrainerConfig",
    "TwoTowerTrainingError",
    "TwoTowerTrainingResult",
    "train_two_tower",
]


class TwoTowerTrainingError(ValueError):
    """Raised when a two-tower training run is configured or fed unusably."""


@dataclass(frozen=True)
class TwoTowerTrainerConfig:
    """Training hyper-parameters, all recorded in the checkpoint.

    Attributes
    ----------
    batch_size:
        Pairs per step, i.e. in-batch negatives plus one.  This is the main quality dial of
        the objective, so it is a first-class configuration value rather than a constant.
    epochs:
        Passes over the pair set.  Fixed, never selected on the evaluation split.
    seed:
        Seeds initialisation, batch order and the pair-subsample cap.
    logq_correction:
        Subtract the log sampling prior from the sampled logits.
    """

    learning_rate: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 1024
    epochs: int = 8
    seed: int = 2026
    device: str = "cpu"
    max_grad_norm: float | None = 5.0
    logq_correction: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise TwoTowerTrainingError(
                f"learning_rate must be finite and > 0, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise TwoTowerTrainingError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TwoTowerTrainingError("batch_size must be an int")
        if self.batch_size < 2:
            raise TwoTowerTrainingError(
                "batch_size must be >= 2: in-batch negatives need at least one other row"
            )
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise TwoTowerTrainingError(f"epochs must be a positive int, got {self.epochs!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TwoTowerTrainingError("seed must be an int")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0
        ):
            raise TwoTowerTrainingError(
                f"max_grad_norm must be finite and > 0 when set, got {self.max_grad_norm}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.device,
            "max_grad_norm": self.max_grad_norm,
            "logq_correction": self.logq_correction,
            "optimizer": "AdamW",
            "objective": "in_batch_sampled_softmax",
        }


@dataclass
class TwoTowerTrainingResult:
    """The trained model plus the auditable record of the run."""

    model: TwoTower
    config: TwoTowerTrainerConfig
    dataset_stats: dict[str, Any]
    epoch_metrics: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "trainer_config": self.config.as_dict(),
            "dataset": self.dataset_stats,
            "epochs": self.epoch_metrics,
            "seconds": round(self.seconds, 3),
            "final_loss": self.epoch_metrics[-1]["loss"] if self.epoch_metrics else None,
        }


def train_two_tower(
    model: TwoTower,
    dataset: TwoTowerDataset,
    *,
    config: TwoTowerTrainerConfig | None = None,
    on_epoch: Callable[[int, dict[str, Any]], None] | None = None,
) -> TwoTowerTrainingResult:
    """Train ``model`` in place on ``dataset`` and return it in ``eval()`` mode.

    Raises
    ------
    TwoTowerTrainingError
        No batch could be formed.  A run that silently trains on nothing is worse than one
        that refuses loudly.
    """
    resolved = config or TwoTowerTrainerConfig()
    device = resolve_device(resolved.device)
    # Reproducible parameter initialisation is the caller's responsibility (build the model
    # under the same seed); this seeds torch for any residual randomness in the loop, exactly
    # as the accepted SASRec trainer does.
    torch.manual_seed(resolved.seed)
    model = model.to(device)
    model.train()

    log_prior = None
    if resolved.logq_correction:
        # ``item_prior_from_pairs`` smooths over the whole catalogue, so every real item has
        # a strictly positive probability and this logarithm is finite for all of them.
        # Index 0 (PAD) is exactly 0 - correctly, because PAD is never a positive - and is
        # never indexed by the loss.
        prior = item_prior_from_pairs(dataset, num_items=model.num_items)
        # The prior's index 0 is exactly 0.0 (PAD is never a positive), so its logarithm is
        # -inf.  Only real ids are ever indexed, but a vector containing a non-finite entry
        # would be rejected by the loss's fail-fast check - correctly, since that check must
        # not have exceptions - so index 0 is set to a finite sentinel that is never read.
        log_prior = torch.log(torch.tensor(prior, dtype=torch.float32, device=device))
        log_prior[0] = 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=resolved.learning_rate, weight_decay=resolved.weight_decay
    )

    started = time.time()
    epoch_metrics: list[dict[str, Any]] = []
    steps_total = 0
    for epoch in range(resolved.epochs):
        totals = {"loss": 0.0, "accuracy": 0.0, "positive": 0.0, "negative": 0.0}
        epoch_steps = 0
        pairs_used = 0
        for windows, targets, _indices in iter_training_batches(
            dataset, batch_size=resolved.batch_size, epoch=epoch, seed=resolved.seed
        ):
            input_ids = torch.tensor(windows, dtype=torch.long, device=device)
            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
            categories = None if model.categories is None else model.categories[target_tensor]

            user_embeddings = model.encode_users(input_ids)
            item_embeddings = model.encode_items(target_tensor, categories)
            loss, stats = sampled_softmax_loss(
                user_embeddings,
                item_embeddings,
                targets,
                temperature=model.temperature(),
                log_prior=log_prior,
                num_items=model.num_items,
            )
            ensure_finite_logits("two-tower loss", loss.reshape(1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if resolved.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resolved.max_grad_norm)
            optimizer.step()

            totals["loss"] += stats.loss
            totals["accuracy"] += stats.accuracy
            totals["positive"] += stats.mean_positive_logit
            totals["negative"] += stats.mean_negative_logit
            epoch_steps += 1
            pairs_used += len(targets)

        if epoch_steps == 0:
            raise TwoTowerTrainingError(
                f"no batch could be formed with batch_size={resolved.batch_size} from "
                f"{len(dataset)} pairs; lower the batch size or add data"
            )

        steps_total += epoch_steps
        metrics = {
            "epoch": epoch,
            "loss": round(totals["loss"] / epoch_steps, 6),
            "in_batch_accuracy": round(totals["accuracy"] / epoch_steps, 6),
            "mean_positive_logit": round(totals["positive"] / epoch_steps, 6),
            "mean_negative_logit": round(totals["negative"] / epoch_steps, 6),
            "steps": epoch_steps,
            "pairs_used": pairs_used,
            "temperature": round(float(model.temperature().detach()), 6),
        }
        epoch_metrics.append(metrics)
        if on_epoch is not None:
            on_epoch(epoch, metrics)

    model.eval()
    result = TwoTowerTrainingResult(
        model=model,
        config=resolved,
        dataset_stats=dict(dataset.stats),
        epoch_metrics=epoch_metrics,
        seconds=time.time() - started,
    )
    result.dataset_stats["total_steps"] = steps_total
    return result
