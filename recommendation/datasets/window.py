"""Train-history statistics and the frozen ``max_seq_len`` selection rule.

Milestone 5 fixes the sequence window **before** any formal training, using only
training data - never validation or test recommendation metrics.

Rule (Milestone 5 section 10)::

    candidate windows = [20, 50, 100, 200]
    select the smallest candidate that retains >= 95% of raw training transitions
    if none reaches 95%, select the largest candidate and report the shortfall

"Raw training transitions" is ``sum(len(train_history) - 1)`` over evaluation users,
i.e. the number of next-item transitions available before any SASRec window is
applied.  Retention for a window ``L`` is computed by clipping each user's shifted
arrays to the newest ``L`` transitions - exactly what :func:`build_arrays` does - so
the statistic matches the dataset that will actually be trained on.

Everything here reads ``EvaluationCase.train_history`` only.  Validation and test
targets are never consulted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from recommendation.evaluation.split import EvaluationCase

#: Candidate windows considered by the frozen rule.
CANDIDATE_WINDOWS: tuple[int, ...] = (20, 50, 100, 200)

#: Required fraction of raw transitions retained by the selected window.
RETENTION_TARGET = 0.95


class SequenceWindowError(ValueError):
    """Raised when the window statistics cannot be computed."""


def train_history_lengths(cases: Iterable[EvaluationCase]) -> list[int]:
    """Return ``len(case.train_history)`` for each case (training data only)."""
    return [len(case.train_history) for case in cases]


def _percentile(sorted_values: Sequence[int], fraction: float) -> float:
    """Linear-interpolation percentile over an already sorted sequence."""
    if not sorted_values:
        raise SequenceWindowError("cannot take a percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def train_history_statistics(cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    """Return length statistics over train histories only.

    Included keys: ``count``, ``min``, ``median``, ``mean``, ``p75``, ``p90``,
    ``p95``, ``p99``, ``max``, plus ``total_interactions`` and
    ``raw_transitions`` (= ``sum(len - 1)`` over users with at least two items).
    """
    lengths = sorted(train_history_lengths(cases))
    if not lengths:
        raise SequenceWindowError("no evaluation cases supplied")

    total = sum(lengths)
    raw_transitions = sum(max(0, length - 1) for length in lengths)
    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": _percentile(lengths, 0.50),
        "mean": total / len(lengths),
        "p75": _percentile(lengths, 0.75),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": lengths[-1],
        "total_interactions": total,
        "raw_transitions": raw_transitions,
        "users_with_zero_transitions": sum(1 for length in lengths if length < 2),
    }


def retained_transitions(lengths: Sequence[int], max_seq_len: int) -> int:
    """Transitions retained for a window ``L``, matching :func:`build_arrays`.

    For a history of ``m`` items the shifted arrays hold ``m - 1`` transitions; a
    window ``L`` keeps the newest ``min(m - 1, L)`` of them.
    """
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise SequenceWindowError(f"max_seq_len must be a positive int, got {max_seq_len!r}")
    return sum(min(max(0, length - 1), max_seq_len) for length in lengths)


def transition_retention(
    cases: Sequence[EvaluationCase],
    windows: Sequence[int] = CANDIDATE_WINDOWS,
) -> dict[int, dict[str, float]]:
    """Return retained transitions and retention fraction for each candidate window."""
    lengths = train_history_lengths(cases)
    raw = sum(max(0, length - 1) for length in lengths)
    report: dict[int, dict[str, float]] = {}
    for window in windows:
        retained = retained_transitions(lengths, window)
        report[window] = {
            "retained_transitions": retained,
            "raw_transitions": raw,
            "retention": (retained / raw) if raw else 0.0,
        }
    return report


def select_max_seq_len(
    cases: Sequence[EvaluationCase],
    windows: Sequence[int] = CANDIDATE_WINDOWS,
    target: float = RETENTION_TARGET,
) -> tuple[int, dict[str, Any]]:
    """Apply the frozen rule and return ``(selected, evidence)``.

    The selected window is the smallest candidate retaining at least ``target`` of the
    raw training transitions; if none qualifies, the largest candidate is selected and
    the shortfall is reported rather than hidden.
    """
    if not windows:
        raise SequenceWindowError("at least one candidate window is required")
    ordered = tuple(sorted(set(windows)))
    retention = transition_retention(cases, ordered)

    selected = ordered[-1]
    reason = f"no candidate reached {target:.0%}; selected the largest candidate"
    for window in ordered:
        if retention[window]["retention"] >= target:
            selected = window
            reason = f"smallest candidate retaining >= {target:.0%} of raw transitions"
            break

    evidence = {
        "candidate_windows": list(ordered),
        "retention_target": target,
        "retention_by_window": {
            str(window): {
                "retained_transitions": retention[window]["retained_transitions"],
                "raw_transitions": retention[window]["raw_transitions"],
                "retention": round(retention[window]["retention"], 6),
            }
            for window in ordered
        },
        "selected_max_seq_len": selected,
        "selected_retention": round(retention[selected]["retention"], 6),
        "reason": reason,
    }
    return selected, evidence


__all__ = [
    "CANDIDATE_WINDOWS",
    "RETENTION_TARGET",
    "SequenceWindowError",
    "retained_transitions",
    "select_max_seq_len",
    "train_history_lengths",
    "train_history_statistics",
    "transition_retention",
]
