"""SASRec training objective (Milestone 4).

One loss only: the binary logistic (BPR-style) objective over the aligned
positive/negative logits produced by
:meth:`recommendation.models.sasrec.SASRec.training_logits`.

For each valid (non-padding) position with positive logit ``p`` and negative logit
``n``::

    per_position = softplus(-p) + softplus(n)
    loss         = mean(per_position)

Padding never reaches this function: ``training_logits`` already drops padding
positions, so no fake labels are invented for PAD.  For zero logits the loss is
exactly ``2 * log(2)``.

``softplus`` is implemented as ``log1p(exp(-|x|)) + max(x, 0)``, which is
numerically stable and free of the overflow that a naive ``log(1 + exp(x))`` would
produce for large ``x``.
"""

from __future__ import annotations

import math

import torch

#: Expected value of the loss when every positive and negative logit is zero.
ZERO_LOGIT_LOSS = 2.0 * math.log(2.0)


class TrainingLossError(ValueError):
    """Raised when the training loss is given unusable logits."""


def stable_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``softplus(x) = log(1 + exp(x))``.

    Equivalent to ``torch.nn.functional.softplus(x, beta=1.0, threshold=inf)`` but
    written out so the exact formula used by the loss is inspectable, and so it does
    not switch to the ``x`` approximation above any threshold.
    """
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp_min(x, 0.0)


def ensure_finite_logits(name: str, logits: torch.Tensor) -> torch.Tensor:
    """Return ``logits`` unchanged, or raise if any value is NaN/Inf.

    This is a **fail-fast** guard, not a sanitising step.  It exists because the
    mathematically stable ``softplus`` saturates at infinity::

        softplus(-(+inf)) = softplus(-inf) = 0        -> finite
        softplus(+inf)    = +inf                      -> infinite

    so an infinite *positive* logit yields the perfectly finite loss ``log 2``.
    That saturation is mathematically correct but would hide a genuinely broken
    forward pass, so non-finite logits are rejected before any loss arithmetic runs.
    Nothing is clamped, replaced, ``nan_to_num``-ed or averaged around.
    """
    if logits.is_complex():
        raise TrainingLossError(f"{name} must be real-valued")
    finite = torch.isfinite(logits)
    if not bool(finite.all()):
        total = int(logits.numel())
        bad = int((~finite).sum().item())
        first = int((~finite).reshape(-1).nonzero()[0].item())
        value = logits.reshape(-1)[first].item()
        raise TrainingLossError(
            f"{name} contains {bad} non-finite value(s) out of {total} "
            f"(first at flat index {first}: {value!r}); training must not continue "
            "with NaN/Inf logits, because stable softplus can return a finite loss "
            "for an infinite logit and hide the numerical failure"
        )
    return logits


def _validate_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate a pair of aligned logit tensors and return them flat and finite."""
    if not isinstance(positive_logits, torch.Tensor) or not isinstance(
        negative_logits, torch.Tensor
    ):
        raise TrainingLossError("positive_logits and negative_logits must be tensors")
    if positive_logits.shape != negative_logits.shape:
        raise TrainingLossError(
            f"positive/negative logit shapes must match, got "
            f"{tuple(positive_logits.shape)} and {tuple(negative_logits.shape)}"
        )
    if positive_logits.numel() == 0:
        raise TrainingLossError(
            "no valid (non-padding) positions: refusing to return a NaN loss"
        )
    if positive_logits.is_complex() or negative_logits.is_complex():
        raise TrainingLossError("logits must be real-valued")

    # Fail fast on non-finite logits *before* any loss arithmetic.
    ensure_finite_logits("positive_logits", positive_logits)
    ensure_finite_logits("negative_logits", negative_logits)

    return positive_logits.reshape(-1), negative_logits.reshape(-1)


def binary_logistic_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar mean ``softplus(-p) + softplus(n)`` over all positions.

    Raises
    ------
    TrainingLossError
        On shape mismatch, an empty position set, or any non-finite logit.  Nothing
        is clamped or sanitised: a broken forward pass fails fast instead of
        producing a plausible-looking loss.  For **finite** inputs the arithmetic is
        unchanged.
    """
    positive, negative = _validate_logits(positive_logits, negative_logits)
    per_position = stable_softplus(-positive) + stable_softplus(negative)
    return per_position.mean()


def per_position_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the per-position losses (no reduction), for diagnostics."""
    positive, negative = _validate_logits(positive_logits, negative_logits)
    return stable_softplus(-positive) + stable_softplus(negative)


def ranking_accuracy(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> float:
    """Fraction of positions where ``positive_logit > negative_logit``.

    A diagnostic for the tiny-overfit acceptance test, not a recommendation metric.
    """
    positive, negative = _validate_logits(positive_logits, negative_logits)
    return float((positive > negative).to(torch.float64).mean().item())


def mean_logit_gap(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> float:
    """Mean ``positive_logit - negative_logit`` (diagnostic)."""
    positive, negative = _validate_logits(positive_logits, negative_logits)
    return float((positive - negative).mean().item())


def logit_diagnostics(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> dict[str, float]:
    """Return finite/loss/ranking diagnostics for one set of logits."""
    positive, negative = _validate_logits(positive_logits, negative_logits)
    loss = binary_logistic_loss(positive, negative)
    return {
        "num_positions": float(positive.numel()),
        "loss": float(loss.item()),
        "loss_finite": float(loss.isfinite().item()),
        "logits_finite": float(
            bool(torch.isfinite(positive).all() and torch.isfinite(negative).all())
        ),
        "ranking_accuracy": ranking_accuracy(positive, negative),
        "mean_logit_gap": mean_logit_gap(positive, negative),
    }


__all__ = [
    "ZERO_LOGIT_LOSS",
    "TrainingLossError",
    "binary_logistic_loss",
    "ensure_finite_logits",
    "logit_diagnostics",
    "mean_logit_gap",
    "per_position_loss",
    "ranking_accuracy",
    "stable_softplus",
]
