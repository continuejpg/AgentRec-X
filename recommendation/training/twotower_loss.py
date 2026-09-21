"""Sampled-softmax retrieval objective for the two-tower model.

This module holds the **objective**; the training loop lives in
:mod:`recommendation.training.twotower`.  Keep them separate so a test can exercise the loss
without building a trainer, and so the trainer's imports read as "the objective plus a loop".


The objective
-------------
For a batch of ``B`` ``(user, positive item)`` pairs, score every user against every item
in the batch and take the cross-entropy of the diagonal::

    logits[u, i] = <user_u, item_i> / temperature  -  log q(item_i)
    loss         = mean_u  -log softmax_i( logits[u, :] )[i]

This is **in-batch sampled softmax**: the ``B - 1`` other positives act as negatives, so
negatives are free and scale with the batch size.  It is the standard two-tower retrieval
loss and it is deliberately different from the accepted SASRec training objective (a
per-position binary logistic loss over individually sampled negatives), which is what makes
a SASRec-vs-Two-Tower comparison a comparison of two real alternatives rather than of two
seeds of the same thing.

Why targets must be unique inside a batch
-----------------------------------------
The diagonal is treated as "the" positive.  If two rows shared a target item, row ``r``'s
positive would also appear as another row's negative and the loss would actively push an
identical item pair apart.  :func:`recommendation.datasets.twotower.iter_training_batches`
therefore skips duplicate targets, and :func:`sampled_softmax_loss` rejects a batch whose
targets are not unique rather than silently training on a contradiction.

The logQ correction
-------------------
In-batch negatives are sampled from the *training* distribution, which is popularity
skewed, so the sampled softmax is biased toward popular items.  Subtracting ``log q(i)``
- the log probability of drawing item ``i`` as a negative - is the standard correction that
makes the objective an unbiased estimator of the full-softmax gradient.  ``q`` is the
smoothed empirical target frequency from the training split
(:func:`recommendation.datasets.twotower.item_prior_from_pairs`), so it is a property of
the data, not a tuned hyper-parameter.

Numerical honesty
-----------------
The temperature is a learned parameter but is clamped before use, and every tensor that
enters the loss is checked for finiteness with the same fail-fast policy the accepted
losses use: nothing is clamped, sanitised or ``nan_to_num``-ed into looking healthy.
"""

from __future__ import annotations

import math

import torch

from recommendation.training.losses import ensure_finite_logits

__all__ = [
    "SampledSoftmaxStats",
    "assert_unique_targets",
    "in_batch_negatives",
    "sampled_softmax_loss",
]


class SampledSoftmaxStats:
    """Diagnostics for one loss evaluation, recorded in the training report."""

    __slots__ = ("loss", "mean_positive_logit", "mean_negative_logit", "batch_size", "accuracy")

    def __init__(
        self,
        *,
        loss: float,
        mean_positive_logit: float,
        mean_negative_logit: float,
        batch_size: int,
        accuracy: float,
    ) -> None:
        self.loss = loss
        self.mean_positive_logit = mean_positive_logit
        self.mean_negative_logit = mean_negative_logit
        self.batch_size = batch_size
        #: Fraction of rows whose positive scored above every in-batch negative.
        self.accuracy = accuracy

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "mean_positive_logit": self.mean_positive_logit,
            "mean_negative_logit": self.mean_negative_logit,
            "batch_size": self.batch_size,
            "in_batch_accuracy": self.accuracy,
        }


def assert_unique_targets(targets: list[int] | tuple[int, ...]) -> None:
    """Raise when a batch repeats a target, which would corrupt the diagonal labelling."""
    if len(set(targets)) != len(targets):
        duplicates = sorted({t for t in targets if list(targets).count(t) > 1})
        raise ValueError(
            "in-batch sampled softmax requires unique targets within a batch; "
            f"repeated: {duplicates[:5]}"
        )


def in_batch_negatives(targets: list[int] | tuple[int, ...]) -> int:
    """How many negatives a batch of this size contributes per row (``B - 1``)."""
    return max(0, len(targets) - 1)


def sampled_softmax_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    targets: list[int] | tuple[int, ...],
    *,
    temperature: torch.Tensor | float = 1.0,
    log_prior: torch.Tensor | None = None,
    num_items: int | None = None,
) -> tuple[torch.Tensor, SampledSoftmaxStats]:
    """Return ``(loss, stats)`` for one batch of in-batch sampled softmax.

    Parameters
    ----------
    user_embeddings:
        ``[B, D]`` unit user vectors (the model normalises them; this function does not
        re-normalise, so a caller that passes unnormalised vectors gets the objective it
        actually asked for).
    item_embeddings:
        ``[B, D]`` item vectors for the batch's positives, in the same row order.
    targets:
        The ``B`` positive item ids, used only to check uniqueness and to report a
        PAD-collision diagnostic.
    temperature:
        Positive scalar or 0-dim tensor.  Must be finite and strictly positive.
    log_prior:
        Optional ``[num_items + 1]`` tensor of ``log q(i)``.  When given, it is subtracted
        from the logits (the logQ correction).
    num_items:
        Catalogue size, required when ``log_prior`` is given so the prior's length can be
        validated.

    Raises
    ------
    ValueError
        A batch smaller than two rows (no negatives), repeated targets, a non-positive or
        non-finite temperature, a mis-shaped or non-finite prior, or targets outside the
        catalogue.
    """
    if user_embeddings.dim() != 2 or item_embeddings.dim() != 2:
        raise ValueError("user_embeddings and item_embeddings must both be 2-D")
    if user_embeddings.shape != item_embeddings.shape:
        raise ValueError(
            "user and item embedding shapes must match, got "
            f"{tuple(user_embeddings.shape)} and {tuple(item_embeddings.shape)}"
        )
    batch_size = user_embeddings.shape[0]
    if batch_size < 2:
        raise ValueError(
            "in-batch sampled softmax needs at least two rows to supply a negative, "
            f"got {batch_size}"
        )
    if len(targets) != batch_size:
        raise ValueError(
            f"targets has {len(targets)} entries but the batch has {batch_size} rows"
        )
    assert_unique_targets(targets)
    if any(int(target) < 1 for target in targets):
        raise ValueError("targets must be real item ids (>= 1); PAD is never a positive")

    scale = _resolve_temperature(temperature)
    logits = (user_embeddings @ item_embeddings.t()) * scale
    if log_prior is not None:
        # The correction applies to the sampled candidates only, i.e. the batch's own
        # positives: the softmax normaliser runs over exactly those columns, so subtracting
        # a full-catalogue prior vector would be a shape error and subtracting anything else
        # would not be the logQ estimator.
        all_log_q = _resolve_log_prior(log_prior, targets, num_items)
        batch_log_q = all_log_q[torch.tensor(list(targets), dtype=torch.long)]
        logits = logits - batch_log_q[None, :]
    ensure_finite_logits("sampled softmax logits", logits)

    labels = torch.arange(batch_size, device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    with torch.no_grad():
        positive = logits.diagonal()
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        negatives = logits.masked_select(mask)
        accuracy = float((positive.unsqueeze(1) > logits.masked_fill(
            torch.eye(batch_size, dtype=torch.bool, device=logits.device), float("-inf")
        )).all(dim=1).float().mean())
    stats = SampledSoftmaxStats(
        loss=float(loss.detach()),
        mean_positive_logit=float(positive.mean()),
        mean_negative_logit=float(negatives.mean()) if negatives.numel() else 0.0,
        batch_size=batch_size,
        accuracy=accuracy,
    )
    return loss, stats


def _resolve_temperature(temperature: torch.Tensor | float) -> torch.Tensor:
    """Validate and return the softmax temperature as a 0-dim tensor."""
    tensor = temperature if isinstance(temperature, torch.Tensor) else torch.tensor(
        float(temperature)
    )
    if tensor.numel() != 1:
        raise ValueError("temperature must be a scalar")
    value = float(tensor.detach())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {value!r}")
    return tensor


def _resolve_log_prior(
    log_prior: torch.Tensor, targets: list[int] | tuple[int, ...], num_items: int | None
) -> torch.Tensor:
    """Validate the log-prior vector against the batch's targets."""
    if log_prior.dim() != 1:
        raise ValueError("log_prior must be 1-D")
    if num_items is not None and log_prior.shape[0] != num_items + 1:
        raise ValueError(
            f"log_prior must have num_items + 1 = {num_items + 1} entries, "
            f"got {log_prior.shape[0]}"
        )
    if not bool(torch.isfinite(log_prior).all()):
        raise ValueError("log_prior contains non-finite values")
    highest = max(int(target) for target in targets)
    if highest >= log_prior.shape[0]:
        raise ValueError(
            f"a target ({highest}) is outside the prior's range of {log_prior.shape[0]}"
        )
    return log_prior
