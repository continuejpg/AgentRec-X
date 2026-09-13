"""Deterministic reference baselines.

Milestone 2B contains a single baseline, :class:`recommendation.baselines.itemcf.ItemCF`.
Baselines are *models*: they expose a scoring function and contain no evaluation
logic.  Every baseline is consumed through the Milestone 2A unified evaluator, which
owns PAD exclusion, seen-item masking, target retention, tie handling, ranking and
metrics.

Later milestones add further reference models here (e.g. a sequential model); this
package deliberately does not create them early.
"""

from __future__ import annotations

from .itemcf import (
    DEFAULT_SCORE,
    MODEL_NAME,
    ItemCF,
    ItemCFError,
    ItemCFFitStats,
    fit_from_cohort,
    make_scorer,
    similarity_pair_count,
    unique_items,
    validate_history,
)

__all__ = [
    "DEFAULT_SCORE",
    "MODEL_NAME",
    "ItemCF",
    "ItemCFError",
    "ItemCFFitStats",
    "fit_from_cohort",
    "make_scorer",
    "similarity_pair_count",
    "unique_items",
    "validate_history",
]

__version__ = "0.1.0"
