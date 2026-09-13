"""Batched / vectorized full-ranking evaluation (Milestone 5).

The canonical per-case evaluator in :mod:`recommendation.evaluation.metrics` remains
the semantic oracle.  This module adds a *batched* path with the same ranking
semantics, so that a full-category cohort can be evaluated in reasonable time
without changing any result.

Frozen semantics reproduced exactly
-----------------------------------
* catalog = item ids ``1..num_items``;
* PAD ``0`` is never a candidate;
* items in the case history are excluded, **except** the target, which always stays
  eligible;
* higher score ranks first;
* equal scores: lower ``item_id`` ranks first;
* rank is 1-based.

Why a scatter mask rather than gathering candidates
---------------------------------------------------
The obvious vectorization - gather each case's legal candidate scores into a
``[batch, num_candidates]`` matrix - is unusable at full-category scale
(``sum(num_candidates)`` reaches billions of elements).  Instead this implementation
keeps the natural ``[batch, num_items + 1]`` score matrix and applies the mask with a
scatter, which costs ``O(batch * history_len)`` extra memory and one extra full-width
comparison.  Index 0 is excluded positionally, so PAD never needs to be masked out.

Target rank for unique targets is computed as::

    1
    + count(score >  target_score)   over legal candidates
    + count(score == target_score AND item_id < target)

For a case whose target also *appears earlier in its own history*, the rank is
computed with the shared histogram path, which is exactly equivalent - see
:func:`batched_target_ranks`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from recommendation import config

from .metrics import (
    DEFAULT_K_VALUES,
    METRIC_NAMES,
    EvaluationError,
    EvaluationReport,
    compare_hr_recall,
    hit_at_k,
    ndcg_at_k,
    validate_history_items,
    validate_k_values,
)


@dataclass
class BatchedEvaluationResult:
    """Aggregated outcome of a batched evaluation pass."""

    report: EvaluationReport
    target_ranks: list[int]
    num_cases: int
    num_batches: int
    seconds: float
    peak_device_bytes: int | None = None

    @property
    def hr_recall_agree(self) -> dict[int, bool]:
        """Per K, whether HR@K and Recall@K coincide (they must here)."""
        return compare_hr_recall(self.report)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload = self.report.as_dict()
        payload.update(
            {
                "num_batches": self.num_batches,
                "seconds": round(self.seconds, 6),
                "peak_device_bytes": self.peak_device_bytes,
            }
        )
        return payload


def ensure_finite_scores_tensor(scores: torch.Tensor, context: str = "scores") -> None:
    """Fail fast if a batched score matrix contains NaN/Inf.

    Reuses the canonical evaluator's policy: the whole vector is validated (PAD
    included), and nothing is clamped or replaced.
    """
    if scores.dim() != 2:
        raise EvaluationError(f"{context} must be 2-D [batch, num_items+1], got {scores.shape}")
    if not bool(torch.isfinite(scores).all()):
        flat = scores.reshape(-1)
        offending = flat[~torch.isfinite(flat)]
        raise EvaluationError(
            f"{context} contains {int(offending.numel())} non-finite value(s); "
            "batched evaluation refuses NaN/Inf scores"
        )


def build_candidate_mask(
    batch_size: int,
    num_items: int,
    histories: Sequence[Sequence[int]],
    target_ids: Sequence[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a ``[batch, num_items + 1]`` bool mask of *legal candidates*.

    ``True`` means "rankable candidate".  PAD (index 0) is always ``False``; every
    history item is ``False`` except the target, which is forced ``True``.
    """
    if len(histories) != batch_size or len(target_ids) != batch_size:
        raise EvaluationError(
            "histories, target_ids and batch_size must agree, got "
            f"{len(histories)}, {len(target_ids)}, {batch_size}"
        )

    mask = torch.ones((batch_size, num_items + 1), dtype=torch.bool, device=device)
    mask[:, config.PAD_ID] = False  # PAD is never a candidate

    for row, history in enumerate(histories):
        items = validate_history_items(history, num_items)
        if items:
            mask[row, torch.tensor(items, dtype=torch.long, device=device)] = False
        target = target_ids[row]
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target == config.PAD_ID
            or not config.FIRST_REAL_ID <= target <= num_items
        ):
            raise EvaluationError(
                f"target item id {target!r} is not a real catalog item in "
                f"[{config.FIRST_REAL_ID}, {num_items}]"
            )
        mask[row, target] = True  # the target always stays eligible
    return mask


def batched_target_ranks(
    scores: torch.Tensor,
    mask: torch.Tensor,
    target_ids: Sequence[int],
) -> list[int]:
    """Return 1-based target ranks for a batch under the frozen tie rule.

    The count is done per row against that row's own mask, which handles every case
    uniformly - including a target that also appears earlier in its own history, and
    a target shared by two rows (where it is a legal candidate in one row and a seen
    item in the other).  The target's own entry is excluded by index, so retaining it
    in ``mask`` costs nothing.

    Verified to agree with the canonical per-case evaluator across unique targets,
    duplicate targets, repeated targets, ties (including all-equal scores) and
    PAD/seen high-score traps - see ``tests/test_batched_evaluator.py``.
    """
    batch_size = scores.shape[0]
    if batch_size == 0:
        return []
    if mask.shape != scores.shape:
        raise EvaluationError(f"mask shape {tuple(mask.shape)} must match scores")

    targets = torch.tensor(list(target_ids), dtype=torch.long, device=scores.device)
    target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)

    item_ids = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)

    # candidates that legitimately beat the target
    greater = (scores > target_scores.unsqueeze(1)) & mask
    equal_lower = (scores == target_scores.unsqueeze(1)) & mask & (item_ids < targets.unsqueeze(1))
    ranks = 1 + greater.sum(dim=1) + equal_lower.sum(dim=1)

    # the target itself satisfies "equal and lower id"? no (id == target), and
    # "greater than target"? no. So its retention contributes nothing here, which is
    # precisely why the unique-target path needs no special handling.

    return [int(value) for value in ranks.tolist()]


def batched_metrics_from_ranks(
    ranks: Sequence[int],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Aggregate metric sums from target ranks (mirrors :func:`evaluate_case`)."""
    k_values = validate_k_values(k_values)
    totals = {f"{name}@{k}": 0.0 for name in METRIC_NAMES for k in k_values}
    for rank in ranks:
        for k in k_values:
            hit = hit_at_k(rank, k)
            totals[f"HR@{k}"] += hit
            totals[f"Recall@{k}"] += hit
            totals[f"NDCG@{k}"] += ndcg_at_k(rank, k)
    return totals


def evaluate_batched(
    *,
    num_items: int,
    score_batches: Iterable[tuple[Sequence[Sequence[int]], Sequence[int], torch.Tensor]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    cohort: str = "test",
    protocol: str = "temporal_leave_two_out",
    device: torch.device | str = "cpu",
    release: bool = True,
) -> BatchedEvaluationResult:
    """Evaluate an iterable of ``(histories, targets, scores)`` batches.

    ``scores`` must be a ``[batch, num_items + 1]`` tensor of raw model scores; the
    caller (model) never masks, this function owns masking and ranking.

    Batches are consumed one at a time and each batch's score matrix is released
    before the next is requested, so peak memory stays bounded by the evaluation
    batch size rather than the cohort size.
    """
    import time

    k_values = validate_k_values(k_values)
    started = time.perf_counter()

    all_ranks: list[int] = []
    num_batches = 0
    peak_device_bytes: int | None = None
    metric_totals = {f"{name}@{k}": 0.0 for name in METRIC_NAMES for k in k_values}
    rank_histogram: dict[int, int] = {}
    candidate_total = 0

    for histories, targets, scores in score_batches:
        num_batches += 1
        batch_size = len(targets)
        if scores.shape != (batch_size, num_items + 1):
            raise EvaluationError(
                f"scores shape {tuple(scores.shape)} must be "
                f"[{batch_size}, {num_items + 1}]"
            )
        ensure_finite_scores_tensor(scores, context="batched scores")

        mask = build_candidate_mask(batch_size, num_items, histories, targets, device=device)
        ranks = batched_target_ranks(scores, mask, targets)

        for rank in ranks:
            all_ranks.append(rank)
            rank_histogram[rank] = rank_histogram.get(rank, 0) + 1
            for k in k_values:
                hit = hit_at_k(rank, k)
                metric_totals[f"HR@{k}"] += hit
                metric_totals[f"Recall@{k}"] += hit
                metric_totals[f"NDCG@{k}"] += ndcg_at_k(rank, k)
        candidate_total += int(mask.sum().item())

        if device != "cpu" and torch.cuda.is_available():
            peak_device_bytes = max(
                peak_device_bytes or 0, int(torch.cuda.max_memory_allocated())
            )

        if release:
            del scores, mask
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_ranks:
        raise EvaluationError(
            "cannot aggregate an empty evaluation cohort; check the split policy and "
            "the minimum sequence length (no user was evaluation-eligible)"
        )

    n = len(all_ranks)
    metrics = {
        name: {k: metric_totals[f"{name}@{k}"] / n for k in k_values} for name in METRIC_NAMES
    }
    report = EvaluationReport(
        cohort=cohort,
        protocol=protocol,
        num_cases=n,
        catalog_size=num_items,
        k_values=k_values,
        metrics=metrics,
        rank_histogram=dict(sorted(rank_histogram.items())),
        mean_target_rank=sum(all_ranks) / n,
        mean_num_candidates=candidate_total / n,
    )
    return BatchedEvaluationResult(
        report=report,
        target_ranks=all_ranks,
        num_cases=n,
        num_batches=num_batches,
        seconds=time.perf_counter() - started,
        peak_device_bytes=peak_device_bytes,
    )


def canonical_ranks_for_reference(
    scores: Sequence[Sequence[float]] | torch.Tensor,
    histories: Sequence[Sequence[int]],
    targets: Sequence[int],
    *,
    num_items: int,
) -> list[int]:
    """Reference ranks from the canonical per-case evaluator (equivalence oracle).

    **Numeric-precision note.** Ranking compares scores for exact equality to apply
    the tie rule, so the two evaluators must be given *bit-identical* values or a
    score that differs only by a rounding step can flip an exact tie and change a
    rank by one.  When a float32 tensor is supplied it is converted with
    ``.tolist()``, so the oracle sees exactly the same Python floats the batched
    path compares - making the equivalence check meaningful rather than
    precision-dependent.
    """
    from .metrics import rank_of_target

    if isinstance(scores, torch.Tensor):
        scores = scores.tolist()
    if not (len(scores) == len(histories) == len(targets)):
        raise EvaluationError(
            "scores, histories and targets must have the same length, got "
            f"{len(scores)}, {len(histories)}, {len(targets)}"
        )

    ranks: list[int] = []
    for row, target in enumerate(targets):
        ranks.append(
            rank_of_target(scores[row], target, history=histories[row], num_items=num_items)
        )
    return ranks


__all__ = [
    "BatchedEvaluationResult",
    "batched_metrics_from_ranks",
    "batched_target_ranks",
    "build_candidate_mask",
    "canonical_ranks_for_reference",
    "ensure_finite_scores_tensor",
    "evaluate_batched",
]
