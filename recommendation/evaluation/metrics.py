"""Ranking kernel, metrics and deterministic aggregation.

This module owns the *scoring* half of the unified evaluation protocol.  It is
model independent: any recommender later supplies one score per catalog item, and
everything about ranking and metric computation happens here so that ItemCF,
SASRec and SID-OneRec cannot drift into incompatible evaluation semantics.

Canonical full-ranking semantics
--------------------------------
For a single positive target and a history of already-consumed items::

    excluded_seen = set(history) - {target}
    candidates    = {1, ..., num_items} - excluded_seen

* PAD (item id :data:`recommendation.config.PAD_ID`, i.e. 0) is never a candidate.
* The target stays eligible even when it also occurs earlier in the history; the
  history set is computed with the target removed, which is what makes the
  repeated-target case well defined.
* Full ranking: every catalog item is a candidate.  Sampled negatives are
  deliberately *not* implemented - they are a different protocol and results from
  the two are not comparable (AGENTS.md section 6).

Deterministic tie rule
----------------------
Ranking is frozen as:

1. higher score ranks first;
2. on equal score, lower item id ranks first.

Metrics
-------
With a 1-based target rank ``r`` and cutoff ``K``::

    HR@K      = 1.0 if r <= K else 0.0
    Recall@K  = 1.0 if r <= K else 0.0
    NDCG@K    = 1 / log2(r + 1) if r <= K else 0.0

Under a single-positive protocol HR@K and Recall@K are numerically identical
(there is exactly one relevant item, so hit rate and recall coincide).  Both are
kept because future protocols may have multiple positives, and their equivalence
here is documented rather than "fixed".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from recommendation import config

#: Canonical cutoff values used throughout the project.
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)

#: Metric names in their canonical report order.
METRIC_NAMES: tuple[str, ...] = ("HR", "Recall", "NDCG")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class EvaluationError(ValueError):
    """Raised when scores, targets or candidate sets violate the protocol."""


# --------------------------------------------------------------------------- #
# Ranking kernel
# --------------------------------------------------------------------------- #


def rank_of_target(
    scores: Sequence[float],
    target_item_id: int,
    *,
    history: Iterable[int] = (),
    num_items: int | None = None,
) -> int:
    """Return the 1-based rank of ``target_item_id`` under canonical ranking.

    ``scores`` is indexed by item id, so ``scores[item_id]`` is the score of that
    item and ``scores[0]`` is the PAD slot (ignored, but required to be present so
    that indexing mistakes surface immediately).

    This is an ``O(num_items)`` implementation: rather than sorting the catalog,
    it counts how many eligible candidates outrank the target.  That count plus one
    *is* the target's rank, because the ranking is a total order under the frozen
    tie rule.  :func:`sorted_ranking` provides the equivalent explicit-sort oracle
    used by the tests to prove the two agree.

    Parameters
    ----------
    scores:
        One score per catalog item, length ``num_items + 1`` (index 0 = PAD).
    target_item_id:
        The single positive item.
    history:
        Items already consumed; excluded from candidates.  The target is always
        removed from this set before masking, so a repeated target stays eligible.
    num_items:
        Catalogue size.  Defaults to ``len(scores) - 1``.

    Raises
    ------
    EvaluationError
        On a malformed score vector, or an invalid target.
    """
    if num_items is None:
        num_items = len(scores) - 1
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise EvaluationError(f"num_items must be a positive int, got {num_items!r}")
    if len(scores) != num_items + 1:
        raise EvaluationError(
            f"score vector must have length num_items + 1 = {num_items + 1} "
            f"(index 0 is the PAD slot), got {len(scores)}"
        )
    validate_scores_finite(scores, context="scores")
    if isinstance(target_item_id, bool) or not isinstance(target_item_id, int):
        raise EvaluationError(
            f"target item id must be an int, got {type(target_item_id).__name__}"
        )
    if target_item_id == config.PAD_ID:
        raise EvaluationError("target item id 0 is the PAD slot and cannot be a target")
    if not config.FIRST_REAL_ID <= target_item_id <= num_items:
        raise EvaluationError(
            f"target item id {target_item_id} outside catalog "
            f"[{config.FIRST_REAL_ID}, {num_items}]"
        )

    excluded = excluded_seen(validate_history_items(history, num_items), target_item_id)

    target_score = scores[target_item_id]
    rank = 1
    for item_id in range(config.FIRST_REAL_ID, num_items + 1):
        if item_id == target_item_id or item_id in excluded:
            continue
        candidate_score = scores[item_id]
        if candidate_score > target_score:
            rank += 1
        elif candidate_score == target_score and item_id < target_item_id:
            rank += 1
    return rank


def validate_scores_finite(scores: Sequence[float], *, context: str = "scores") -> None:
    """Raise :class:`EvaluationError` if any score is NaN or infinite.

    Non-finite scores silently corrupt every ranking built on them, because every
    IEEE-754 comparison against ``NaN`` is false:

    * a ``NaN`` target compares as "not better" against every candidate, so it is
      ranked **first** and every metric reports a hit;
    * a ``NaN`` candidate is skipped as a competitor, so real competitors below it
      are never counted;
    * ``sorted_ranking`` places ``NaN`` in a position that depends on the sort's
      comparison order rather than on the frozen tie rule, breaking determinism.

    ``+inf``/``-inf`` are rejected too, so a model cannot smuggle in an "always
    rank first" sentinel: express confidence with large *finite* values instead.
    The whole vector is checked, including the PAD slot at index 0, so a vector
    padded with ``NaN`` cannot slip through.
    """
    for index, value in enumerate(scores):
        try:
            finite = math.isfinite(value)
        except TypeError as exc:
            raise EvaluationError(
                f"{context}[{index}] is {type(value).__name__}, expected a real number"
            ) from exc
        if not finite:
            raise EvaluationError(
                f"{context}[{index}] is {value!r}; scores must be finite "
                "(NaN and +/-inf are rejected because they corrupt ranking)"
            )


def excluded_seen(history: Iterable[int], target_item_id: int) -> frozenset[int]:
    """Return the items masked out of the candidate set for one evaluation case.

    ``set(history) - {target_item_id}``.  Subtracting the target is what keeps a
    repeated target eligible.
    """
    return frozenset(history) - {target_item_id}


def validate_history_items(history: Iterable[int], num_items: int) -> tuple[int, ...]:
    """Validate a history and return it as a tuple.

    Every history entry must be a real catalog item.  Out-of-range or PAD entries
    are rejected loudly: they would otherwise be silently ignored by the mask loop
    and hide a corrupt artifact or a model-side indexing bug.
    """
    items = tuple(history)
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise EvaluationError(f"num_items must be a positive int, got {num_items!r}")
    for item_id in items:
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise EvaluationError(
                f"history item must be an int, got {type(item_id).__name__}: {item_id!r}"
            )
        if item_id == config.PAD_ID:
            raise EvaluationError(
                "history contains the PAD slot 0; PAD must never be part of a history"
            )
        if not config.FIRST_REAL_ID <= item_id <= num_items:
            raise EvaluationError(
                f"history item {item_id} outside catalog "
                f"[{config.FIRST_REAL_ID}, {num_items}]"
            )
    return items


def valid_candidates(
    num_items: int,
    history: Iterable[int] = (),
    target_item_id: int | None = None,
) -> list[int]:
    """Return the ranked candidate list for one case, in ascending item id.

    Useful for tests and for inspecting the candidate contract; the metric path
    does not need to materialise this list.
    """
    history_items = validate_history_items(history, num_items)
    if target_item_id is not None:
        if (
            isinstance(target_item_id, bool)
            or not isinstance(target_item_id, int)
            or target_item_id == config.PAD_ID
            or not config.FIRST_REAL_ID <= target_item_id <= num_items
        ):
            raise EvaluationError(
                f"target item id {target_item_id!r} is not a real catalog item in "
                f"[{config.FIRST_REAL_ID}, {num_items}]"
            )
        excluded = excluded_seen(history_items, target_item_id)
    else:
        excluded = frozenset(history_items)
    return [i for i in range(config.FIRST_REAL_ID, num_items + 1) if i not in excluded]


def sorted_ranking(
    scores: Sequence[float],
    *,
    history: Iterable[int] = (),
    target_item_id: int | None = None,
    num_items: int | None = None,
) -> list[int]:
    """Reference implementation: explicitly sort candidates by the canonical rule.

    Returns candidate item ids, best first, ordered by ``(-score, item_id)``.  This
    exists as the *oracle* for the ``O(num_items)`` :func:`rank_of_target`; the
    tests assert the two always agree.
    """
    if num_items is None:
        num_items = len(scores) - 1
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise EvaluationError(f"num_items must be a positive int, got {num_items!r}")
    if len(scores) != num_items + 1:
        raise EvaluationError(
            f"score vector must have length num_items + 1 = {num_items + 1}, "
            f"got {len(scores)}"
        )
    validate_scores_finite(scores, context="scores")
    candidates = valid_candidates(num_items, history, target_item_id)
    return sorted(candidates, key=lambda item_id: (-scores[item_id], item_id))


def rank_of_target_via_sort(
    scores: Sequence[float],
    target_item_id: int,
    *,
    history: Iterable[int] = (),
    num_items: int | None = None,
) -> int:
    """Target rank computed from the explicit-sort oracle (1-based)."""
    ranking = sorted_ranking(
        scores,
        history=history,
        target_item_id=target_item_id,
        num_items=num_items,
    )
    try:
        return ranking.index(target_item_id) + 1
    except ValueError as exc:  # pragma: no cover - guarded by the caller
        raise EvaluationError(
            f"target {target_item_id} is not in the candidate list"
        ) from exc


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def validate_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    """Return ``k_values`` as a validated, ascending, duplicate-free tuple."""
    cleaned: list[int] = []
    for k in k_values:
        if isinstance(k, bool) or not isinstance(k, int):
            raise EvaluationError(f"K must be an int, got {type(k).__name__}: {k!r}")
        if k < 1:
            raise EvaluationError(f"K must be >= 1, got {k}")
        cleaned.append(k)
    if not cleaned:
        raise EvaluationError("at least one K value is required")
    return tuple(sorted(set(cleaned)))


def hit_at_k(rank: int, k: int) -> float:
    """HR@K / Recall@K for a single positive: 1.0 if ``rank <= k`` else 0.0."""
    _validate_rank(rank)
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """NDCG@K for a single positive: ``1 / log2(rank + 1)`` if ``rank <= k``.

    With ``rank == 1`` this is exactly 1.0, because ``log2(2) == 1``.
    """
    _validate_rank(rank)
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _validate_rank(rank: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise EvaluationError(f"rank must be an int, got {type(rank).__name__}")
    if rank < 1:
        raise EvaluationError(f"rank is 1-based and must be >= 1, got {rank}")


@dataclass(frozen=True)
class CaseResult:
    """Per-case evaluation outcome for one target."""

    user_id: str
    target_item_id: int
    rank: int
    num_candidates: int
    num_masked: int
    metrics: dict[str, float]


def evaluate_case(
    scores: Sequence[float],
    target_item_id: int,
    *,
    history: Iterable[int] = (),
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    num_items: int | None = None,
    user_id: str = "",
) -> CaseResult:
    """Evaluate one target and return its rank plus per-K metrics."""
    k_values = validate_k_values(k_values)
    history_items = tuple(history)
    rank = rank_of_target(
        scores,
        target_item_id,
        history=history_items,
        num_items=num_items,
    )
    if num_items is None:
        num_items = len(scores) - 1
    num_candidates = len(valid_candidates(num_items, history_items, target_item_id))

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"HR@{k}"] = hit_at_k(rank, k)
        metrics[f"Recall@{k}"] = hit_at_k(rank, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(rank, k)

    return CaseResult(
        user_id=user_id,
        target_item_id=target_item_id,
        rank=rank,
        num_candidates=num_candidates,
        num_masked=num_items - num_candidates,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass
class EvaluationReport:
    """Deterministic aggregation of per-case results over one cohort.

    ``metrics[name][k]`` holds the mean metric value across the cohort, and
    ``num_cases`` records the cohort size the mean was taken over.
    """

    cohort: str
    protocol: str
    num_cases: int
    catalog_size: int
    k_values: tuple[int, ...]
    metrics: dict[str, dict[int, float]] = field(default_factory=dict)
    rank_histogram: dict[int, int] = field(default_factory=dict)
    mean_target_rank: float | None = None
    mean_num_candidates: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "cohort": self.cohort,
            "protocol": self.protocol,
            "num_cases": self.num_cases,
            "catalog_size": self.catalog_size,
            "k_values": list(self.k_values),
            "metrics": {
                name: {f"@{k}": self.metrics[name][k] for k in self.k_values}
                for name in sorted(self.metrics)
            },
            "mean_target_rank": self.mean_target_rank,
            "mean_num_candidates": self.mean_num_candidates,
        }

    def format(self) -> str:
        """Render the report as a compact aligned table."""
        lines = [
            f"cohort={self.cohort}  cases={self.num_cases}  "
            f"catalog={self.catalog_size}  protocol={self.protocol}"
        ]
        header = "  metric   " + "".join(f"{f'@{k}':>12s}" for k in self.k_values)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for name in METRIC_NAMES:
            if name not in self.metrics:
                continue
            row = f"  {name:<8s} " + "".join(
                f"{self.metrics[name][k]:>12.6f}" for k in self.k_values
            )
            lines.append(row)
        return "\n".join(lines)


def aggregate(
    results: Iterable[CaseResult],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    catalog_size: int,
    cohort: str = "test",
    protocol: str = "temporal_leave_two_out",
) -> EvaluationReport:
    """Aggregate per-case results into a report.

    Raise
    -----
    EvaluationError
        If ``results`` is empty.  An empty evaluation cohort is a configuration
        error (usually a split/threshold mistake), so it fails loudly instead of
        silently producing NaN metrics.
    """
    k_values = validate_k_values(k_values)
    collected = list(results)
    if not collected:
        raise EvaluationError(
            "cannot aggregate an empty evaluation cohort; check the split policy "
            "and the minimum sequence length (no user was evaluation-eligible)"
        )

    totals: dict[str, dict[int, float]] = {name: {k: 0.0 for k in k_values} for name in METRIC_NAMES}
    histogram: dict[int, int] = {}
    rank_sum = 0
    candidate_sum = 0
    for result in collected:
        for name in METRIC_NAMES:
            for k in k_values:
                totals[name][k] += result.metrics[f"{name}@{k}"]
        histogram[result.rank] = histogram.get(result.rank, 0) + 1
        rank_sum += result.rank
        candidate_sum += result.num_candidates

    n = len(collected)
    means = {name: {k: totals[name][k] / n for k in k_values} for name in METRIC_NAMES}
    return EvaluationReport(
        cohort=cohort,
        protocol=protocol,
        num_cases=n,
        catalog_size=catalog_size,
        k_values=k_values,
        metrics=means,
        rank_histogram=dict(sorted(histogram.items())),
        mean_target_rank=rank_sum / n,
        mean_num_candidates=candidate_sum / n,
    )


def compare_hr_recall(report: EvaluationReport) -> dict[int, bool]:
    """Return, per K, whether HR@K and Recall@K agree (they must in this protocol)."""
    return {
        k: report.metrics["HR"][k] == report.metrics["Recall"][k]
        for k in report.k_values
    }
