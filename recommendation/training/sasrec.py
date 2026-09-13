"""Minimal deterministic SASRec trainer (Milestone 4).

Responsibility split (kept strict — no evaluation semantics live here):

======================  ==========================================================
Dataset                 training examples and negatives
Model                   hidden states, logits, full-catalog scores
**Trainer (here)**      loss, backward, optimizer step, batching, loss accounting
Evaluator               masking, ranking, HR / Recall / NDCG
======================  ==========================================================

The trainer never masks candidates, never ranks, and never computes a metric.  It
consumes the Milestone 3 dataset/model interfaces unchanged:

* batching is built on :class:`~recommendation.datasets.sasrec.SASRecSample`;
* the objective uses :meth:`~recommendation.models.sasrec.SASRec.training_logits`
  through :mod:`recommendation.training.losses`.

Batching is hand-rolled rather than using ``torch.utils.data.DataLoader``: the
samples are already fixed-length integer arrays, so a custom collate would only add
indirection, and doing it directly keeps sample ordering and determinism explicit
(`num_workers=0` semantics by construction).  No multiprocessing.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from recommendation.datasets.sasrec import (
    SASRecDataError,
    SASRecDataset,
    SASRecSample,
    sample_negative,
)
from recommendation.models.sasrec import PAD_ID, SASRec
from recommendation.training.losses import (
    binary_logistic_loss,
    logit_diagnostics,
)


class TrainingError(ValueError):
    """Raised when the trainer is configured or driven incorrectly."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainerConfig:
    """Trainer hyper-parameters.

    ``weight_decay`` defaults to ``0.0``: this milestone trains tiny fixtures to
    convergence, where any regularisation only fights memorisation.  It is exposed
    so later milestones can enable it deliberately.
    """

    learning_rate: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 1
    seed: int = 0
    device: str = "cpu"
    shuffle: bool = True
    resample_negatives: bool = False
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise TrainingError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise TrainingError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise TrainingError(f"batch_size must be a positive int, got {self.batch_size!r}")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise TrainingError(f"epochs must be a positive int, got {self.epochs!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TrainingError(f"seed must be an int, got {type(self.seed).__name__}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise TrainingError(f"max_grad_norm must be > 0 when set, got {self.max_grad_norm}")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the configuration."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.device,
            "shuffle": self.shuffle,
            "resample_negatives": self.resample_negatives,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": "AdamW",
        }



#: Device strings accepted by :func:`resolve_device`.
SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda", "cuda:0")


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string, failing fast when it is unusable.

    * ``"cpu"`` is always accepted.
    * ``"cuda"`` / ``"cuda:0"`` are accepted only when CUDA is actually available;
      otherwise the run stops with an explicit message rather than silently falling
      back to CPU (which would make a "GPU" run meaningless).
    * Anything else - including malformed strings such as ``"cuda:abc"`` or
      ``"gpu"`` - is rejected.
    """
    if not isinstance(requested, str) or not requested.strip():
        raise TrainingError(f"device must be a non-empty string, got {requested!r}")

    normalised = requested.strip().lower()
    if normalised not in SUPPORTED_DEVICES:
        raise TrainingError(
            f"unsupported device {requested!r}; supported values are {SUPPORTED_DEVICES}"
        )

    if normalised == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise TrainingError(
            f"device {requested!r} requested but torch.cuda.is_available() is False; "
            "refusing to silently fall back to CPU"
        )
    if normalised == "cuda":
        return torch.device("cuda:0")
    return torch.device(normalised)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SASRecBatch:
    """One aligned batch of fixed-length integer arrays."""

    input_ids: torch.Tensor
    positive_ids: torch.Tensor
    negative_ids: torch.Tensor
    user_int_ids: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        """Sequence length of the batch (always ``max_seq_len``)."""
        return int(self.input_ids.shape[1])

    def to(self, device: str) -> SASRecBatch:
        """Return the batch with all tensors moved to ``device``."""
        return SASRecBatch(
            input_ids=self.input_ids.to(device),
            positive_ids=self.positive_ids.to(device),
            negative_ids=self.negative_ids.to(device),
            user_int_ids=self.user_int_ids,
        )


def collate_samples(samples: Sequence[SASRecSample]) -> SASRecBatch:
    """Collate samples into ``[batch, max_seq_len]`` long tensors.

    All arrays are already ``max_seq_len`` long and left-padded, so collation is a
    pure stack.  Input/positive/negative alignment and padding come straight from the
    dataset and are therefore preserved rather than recomputed.
    """
    if not samples:
        raise TrainingError("cannot collate an empty list of samples")

    seq_len = len(samples[0].input_ids)
    for sample in samples:
        if len(sample.input_ids) != seq_len:
            raise TrainingError(
                "all samples must share one max_seq_len; got lengths "
                f"{sorted({len(s.input_ids) for s in samples})}"
            )

    def stack(values: list[tuple[int, ...]]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long)

    return SASRecBatch(
        input_ids=stack([s.input_ids for s in samples]),
        positive_ids=stack([s.positive_ids for s in samples]),
        negative_ids=stack([s.negative_ids for s in samples]),
        user_int_ids=tuple(s.user_int_id for s in samples),
    )


def iter_batches(
    samples: Sequence[SASRecSample],
    batch_size: int,
    *,
    order: Sequence[int] | None = None,
) -> Iterable[SASRecBatch]:
    """Yield batches in ``order`` (or natural order), with a final partial batch.

    ``drop_last`` is deliberately **not** used: dropping the remainder would discard
    real training transitions, which matters for tiny fixtures where the last batch
    may hold most of the data.
    """
    if batch_size < 1:
        raise TrainingError(f"batch_size must be >= 1, got {batch_size}")
    indices = list(range(len(samples))) if order is None else list(order)
    for start in range(0, len(indices), batch_size):
        chunk = [samples[i] for i in indices[start : start + batch_size]]
        yield collate_samples(chunk)


def epoch_order(num_samples: int, *, shuffle: bool, seed: int, epoch: int) -> list[int]:
    """Return the deterministic sample order for one epoch.

    Shuffling uses a generator seeded from ``(seed, epoch)`` only, so the order is
    reproducible and independent of global RNG state and of ``PYTHONHASHSEED``.
    """
    if num_samples < 0:
        raise TrainingError(f"num_samples must be >= 0, got {num_samples}")
    order = list(range(num_samples))
    if shuffle:
        random.Random(f"agentrecx.sasrec.train|{seed}|{epoch}").shuffle(order)
    return order


# --------------------------------------------------------------------------- #
# Epoch-aware negative resampling
# --------------------------------------------------------------------------- #


def samples_with_epoch_negatives(
    samples: Sequence[SASRecSample],
    num_items: int,
    seed: int,
    epoch: int,
) -> list[SASRecSample]:
    """Return the samples with negatives resampled for ``epoch``.

    Inputs and positives are copied through untouched, so the optimization structure
    (and therefore the set of learned preferences) is unchanged; only the negatives
    are redrawn.  Determinism comes from the dataset's own
    ``(seed, epoch, user, position)``-derived sampling, and validation/test targets
    remain invisible because only train histories are involved.
    """
    # Build the per-epoch config ONCE: it is a pure function of (seed, epoch), and
    # rebuilding it per position dominated this function's runtime.
    epoch_config = _epoch_config(seed, epoch)

    resampled: list[SASRecSample] = []
    for sample in samples:
        history = tuple(v for v in sample.input_ids if v != PAD_ID)
        if len(history) < 1:
            raise SASRecDataError(
                f"sample for user {sample.user_int_id} has no real input items"
            )
        exclusion = frozenset(history)
        negatives: list[int] = []
        position = 0
        for positive in sample.positive_ids:
            if positive == PAD_ID:
                negatives.append(PAD_ID)
                continue
            negatives.append(
                sample_negative(
                    sample.user_int_id,
                    position,
                    exclusion,
                    num_items,
                    epoch_config,
                )
            )
            position += 1
        resampled.append(
            SASRecSample(
                user_int_id=sample.user_int_id,
                input_ids=sample.input_ids,
                positive_ids=sample.positive_ids,
                negative_ids=tuple(negatives),
                num_valid_positions=sample.num_valid_positions,
            )
        )
    return resampled


def _epoch_config(seed: int, epoch: int):
    """Build the minimal dataset config carrying ``(seed, epoch)`` for sampling."""
    from recommendation.datasets.sasrec import SASRecDatasetConfig

    return SASRecDatasetConfig(seed=seed, epoch=epoch)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class EpochMetrics:
    """Loss accounting for one training epoch."""

    epoch: int
    num_batches: int
    num_positions: int
    mean_loss: float
    first_batch_loss: float
    last_batch_loss: float
    all_losses_finite: bool
    ranking_accuracy: float
    mean_logit_gap: float
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "epoch": self.epoch,
            "num_batches": self.num_batches,
            "num_positions": self.num_positions,
            "mean_loss": self.mean_loss,
            "first_batch_loss": self.first_batch_loss,
            "last_batch_loss": self.last_batch_loss,
            "all_losses_finite": self.all_losses_finite,
            "ranking_accuracy": self.ranking_accuracy,
            "mean_logit_gap": self.mean_logit_gap,
            "seconds": round(self.seconds, 6),
        }


@dataclass
class TrainingResult:
    """Outcome of a training run, including the evidence Milestone 4 requires."""

    config: TrainerConfig
    model_config: dict[str, Any]
    initial_loss: float
    final_loss: float
    optimizer_steps: int
    epochs: list[EpochMetrics]
    loss_trajectory: list[float]
    all_losses_finite: bool
    all_gradients_finite: bool
    nonzero_gradient_observed: bool
    parameter_update_observed: bool
    pad_embedding_zero_after_training: bool
    parameters_finite_after_training: bool
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def loss_reduction_ratio(self) -> float:
        """``final_loss / initial_loss`` (lower is better; 0.25 is the bar)."""
        if self.initial_loss == 0.0:
            return 0.0
        return self.final_loss / self.initial_loss

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the run."""
        payload = {
            "trainer_config": self.config.as_dict(),
            "model_config": self.model_config,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_reduction_ratio": self.loss_reduction_ratio,
            "optimizer_steps": self.optimizer_steps,
            "epochs": [e.as_dict() for e in self.epochs],
            "loss_trajectory": [round(v, 8) for v in self.loss_trajectory],
            "all_losses_finite": self.all_losses_finite,
            "all_gradients_finite": self.all_gradients_finite,
            "nonzero_gradient_observed": self.nonzero_gradient_observed,
            "parameter_update_observed": self.parameter_update_observed,
            "pad_embedding_zero_after_training": self.pad_embedding_zero_after_training,
            "parameters_finite_after_training": self.parameters_finite_after_training,
            "seconds": round(self.seconds, 6),
        }
        payload.update(self.extra)
        return payload


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class SASRecTrainer:
    """Minimal deterministic trainer: forward, loss, backward, optimizer step.

    Parameters
    ----------
    model:
        A :class:`~recommendation.models.sasrec.SASRec`.
    num_items:
        Catalogue size; must match the model.
    config:
        A :class:`TrainerConfig`.
    """

    def __init__(self, model: SASRec, num_items: int, config: TrainerConfig | None = None) -> None:
        if not isinstance(model, SASRec):
            raise TrainingError(f"model must be a SASRec, got {type(model).__name__}")
        if num_items != model.num_items:
            raise TrainingError(
                f"num_items {num_items} does not match the model's {model.num_items}"
            )
        self.model = model
        self.num_items = num_items
        self.config = config or TrainerConfig()

        # Device policy (Milestone 5): CPU and CUDA are supported; anything else is
        # rejected.  CUDA is only accepted when it is genuinely available, so a
        # misconfigured run fails immediately instead of silently training on CPU.
        self.device = resolve_device(self.config.device)
        self.model.to(self.device)

        # Reproducible parameter initialization is the caller's responsibility (via
        # build_model(seed=...)); the trainer seeds torch for any residual randomness.
        torch.manual_seed(self.config.seed)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    # -- helpers ----------------------------------------------------------- #

    def pad_embedding_is_zero(self) -> bool:
        """True when the item embedding's PAD row is exactly zero."""
        with torch.no_grad():
            row = self.model.item_embedding.weight[PAD_ID]
            return bool(torch.equal(row, torch.zeros_like(row)))

    def meaningful_parameters(self) -> dict[str, torch.Tensor]:
        """Leaf parameters that must actually learn.

        All leaf parameter tensors are returned *unsliced*: slicing a parameter
        produces a non-leaf tensor whose ``.grad`` is never populated.  Callers that
        need to exclude the item embedding's PAD row (which is supposed to stay zero
        and must not count as evidence of learning) should apply
        :meth:`meaningful_slice` to the gradient instead.
        """
        return {name: tensor for name, tensor in self.model.named_parameters()}

    @staticmethod
    def meaningful_slice(name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` without the PAD row when it is the item embedding table."""
        if name == "item_embedding.weight":
            return tensor[1:]
        return tensor

    def parameters_finite(self) -> bool:
        """True when every parameter tensor is finite."""
        with torch.no_grad():
            return all(bool(torch.isfinite(p).all()) for p in self.model.parameters())

    # -- one optimisation step --------------------------------------------- #

    def step(self, batch: SASRecBatch) -> tuple[float, bool, bool]:
        """Run one forward/backward/step on ``batch``.

        Returns ``(loss, loss_finite, gradients_finite)``.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        positive_logits, negative_logits = self.model.training_logits(
            batch.input_ids, batch.positive_ids, batch.negative_ids
        )
        loss = binary_logistic_loss(positive_logits, negative_logits)
        loss_value = float(loss.item())

        loss.backward()

        grads_finite = True
        for parameter in self.model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                grads_finite = False
                break

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

        # `padding_idx` only zeroes the gradient *into* PAD; with AdamW the weight
        # decay term can still move that row, so restore the invariant explicitly.
        self.enforce_pad_embedding_zero()

        return loss_value, bool(torch.isfinite(loss).item()), grads_finite

    @torch.no_grad()
    def enforce_pad_embedding_zero(self) -> None:
        """Force the item embedding's PAD row back to exactly zero."""
        self.model.item_embedding.weight[PAD_ID].zero_()

    # -- epochs ------------------------------------------------------------ #

    def samples_for_epoch(self, dataset: SASRecDataset, epoch: int) -> list[SASRecSample]:
        """Return the samples to train on for ``epoch`` (optionally resampled negatives)."""
        if self.config.resample_negatives:
            return samples_with_epoch_negatives(
                dataset.samples, self.num_items, self.config.seed, epoch
            )
        return list(dataset.samples)

    def run_epoch(self, dataset: SASRecDataset, epoch: int) -> tuple[EpochMetrics, list[bool]]:
        """Train for one epoch, returning its metrics and per-step finiteness flags."""
        started = time.perf_counter()
        samples = self.samples_for_epoch(dataset, epoch)
        order = epoch_order(
            len(samples), shuffle=self.config.shuffle, seed=self.config.seed, epoch=epoch
        )

        losses: list[float] = []
        finite_flags: list[bool] = []
        positive_parts: list[torch.Tensor] = []
        negative_parts: list[torch.Tensor] = []
        positions = 0

        for batch in iter_batches(samples, self.config.batch_size, order=order):
            batch = batch.to(self.config.device)
            loss_value, loss_finite, grads_finite = self.step(batch)
            losses.append(loss_value)
            finite_flags.append(bool(loss_finite and grads_finite))

            with torch.no_grad():
                positive_logits, negative_logits = self.model.training_logits(
                    batch.input_ids, batch.positive_ids, batch.negative_ids
                )
            positive_parts.append(positive_logits.detach().cpu())
            negative_parts.append(negative_logits.detach().cpu())
            positions += int(positive_logits.numel())

        if not losses:
            raise TrainingError("epoch produced no batches; the dataset is empty")

        positives = torch.cat(positive_parts)
        negatives = torch.cat(negative_parts)
        diagnostics = logit_diagnostics(positives, negatives)

        metrics = EpochMetrics(
            epoch=epoch,
            num_batches=len(losses),
            num_positions=positions,
            mean_loss=sum(losses) / len(losses),
            first_batch_loss=losses[0],
            last_batch_loss=losses[-1],
            all_losses_finite=all(finite_flags),
            ranking_accuracy=diagnostics["ranking_accuracy"],
            mean_logit_gap=diagnostics["mean_logit_gap"],
            seconds=time.perf_counter() - started,
        )
        return metrics, finite_flags

    # -- full run ---------------------------------------------------------- #

    def train(
        self,
        dataset: SASRecDataset,
        *,
        on_epoch_end: Callable[[int, SASRecTrainer], None] | None = None,
    ) -> TrainingResult:
        """Train for ``config.epochs`` epochs and return the evidence summary.

        ``on_epoch_end`` is invoked after each epoch and is never fired mid-epoch, so
        a callback that snapshots parameters or runs inference cannot accidentally
        observe a partially updated epoch.
        """
        if not dataset.samples:
            raise TrainingError(
                "dataset contains no trainable samples (all users have zero transitions)"
            )

        started = time.perf_counter()
        initial_parameters = {
            name: tensor.detach().clone() for name, tensor in self.meaningful_parameters().items()
        }

        # initial loss on the fixed first epoch's data, before any update
        first_samples = self.samples_for_epoch(dataset, 0)
        first_order = epoch_order(
            len(first_samples), shuffle=self.config.shuffle, seed=self.config.seed, epoch=0
        )
        initial_batch = next(iter_batches(first_samples, self.config.batch_size, order=first_order))
        initial_loss = self.evaluate_loss(initial_batch.to(self.config.device))

        epoch_metrics: list[EpochMetrics] = []
        trajectory: list[float] = [initial_loss]
        all_finite = bool(torch.isfinite(torch.tensor(initial_loss)).item())

        for epoch in range(self.config.epochs):
            metrics, finite_flags = self.run_epoch(dataset, epoch)
            epoch_metrics.append(metrics)
            trajectory.append(metrics.mean_loss)
            all_finite = all_finite and metrics.all_losses_finite and all(finite_flags)
            if on_epoch_end is not None:
                on_epoch_end(epoch, self)

        final_loss = epoch_metrics[-1].mean_loss

        # evidence
        update_observed = any(
            not torch.equal(
                self.meaningful_slice(name, before),
                self.meaningful_slice(name, after),
            )
            for (name, before), (_, after) in zip(
                initial_parameters.items(), self.meaningful_parameters().items()
            )
        )
        nonzero_gradient = False
        self.optimizer.zero_grad(set_to_none=True)
        probe_batch = next(
            iter_batches(first_samples, min(self.config.batch_size, len(first_samples)), order=first_order)
        ).to(self.config.device)
        positive_logits, negative_logits = self.model.training_logits(
            probe_batch.input_ids, probe_batch.positive_ids, probe_batch.negative_ids
        )
        binary_logistic_loss(positive_logits, negative_logits).backward()
        for name, parameter in self.meaningful_parameters().items():
            if parameter.grad is None:
                continue
            if bool((self.meaningful_slice(name, parameter.grad) != 0).any()):
                nonzero_gradient = True
                break
        self.optimizer.zero_grad(set_to_none=True)

        return TrainingResult(
            config=self.config,
            model_config=self.model.config.as_dict(),
            initial_loss=initial_loss,
            final_loss=final_loss,
            optimizer_steps=sum(m.num_batches for m in epoch_metrics),
            epochs=epoch_metrics,
            loss_trajectory=trajectory,
            all_losses_finite=all_finite,
            all_gradients_finite=all(m.all_losses_finite for m in epoch_metrics),
            nonzero_gradient_observed=nonzero_gradient,
            parameter_update_observed=update_observed,
            pad_embedding_zero_after_training=self.pad_embedding_is_zero(),
            parameters_finite_after_training=self.parameters_finite(),
            seconds=time.perf_counter() - started,
            extra={"num_samples": len(dataset.samples)},
        )

    # -- inference-free diagnostics ---------------------------------------- #

    @torch.no_grad()
    def evaluate_loss(self, batch: SASRecBatch) -> float:
        """Return the loss on ``batch`` without touching gradients or the optimizer."""
        was_training = self.model.training
        self.model.eval()
        try:
            positive_logits, negative_logits = self.model.training_logits(
                batch.input_ids, batch.positive_ids, batch.negative_ids
            )
            return float(binary_logistic_loss(positive_logits, negative_logits).item())
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def batch_logits(self, samples: Sequence[SASRecSample]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated positive/negative logits for ``samples`` (no updates)."""
        was_training = self.model.training
        self.model.eval()
        try:
            positives: list[torch.Tensor] = []
            negatives: list[torch.Tensor] = []
            for batch in iter_batches(list(samples), self.config.batch_size):
                batch = batch.to(self.device)
                positive_logits, negative_logits = self.model.training_logits(
                    batch.input_ids, batch.positive_ids, batch.negative_ids
                )
                positives.append(positive_logits.detach().cpu())
                negatives.append(negative_logits.detach().cpu())
            return torch.cat(positives), torch.cat(negatives)
        finally:
            self.model.train(was_training)


def train_sasrec(
    model: SASRec,
    dataset: SASRecDataset,
    config: TrainerConfig | None = None,
    *,
    on_epoch_end: Callable[[int, SASRecTrainer], None] | None = None,
) -> TrainingResult:
    """Convenience wrapper: build a :class:`SASRecTrainer` and run it."""
    trainer = SASRecTrainer(model, dataset.num_items, config)
    return trainer.train(dataset, on_epoch_end=on_epoch_end)


__all__ = [
    "EpochMetrics",
    "SUPPORTED_DEVICES",
    "resolve_device",
    "SASRecBatch",
    "SASRecTrainer",
    "TrainerConfig",
    "TrainingError",
    "TrainingResult",
    "collate_samples",
    "epoch_order",
    "iter_batches",
    "samples_with_epoch_negatives",
    "train_sasrec",
]
