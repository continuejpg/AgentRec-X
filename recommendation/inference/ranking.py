"""Deterministic production ranking over raw SASRec catalog scores.

Serving semantics are deliberately **not** the evaluator's semantics:

* the evaluator retains the held-out target even if it appears in the history;
* production has no target, so the candidate set is simply::

      catalog 1..num_items  -  seen items  -  PAD

Everyone who has already interacted with an item must not be recommended it again.

Ranking order is frozen::

    1. higher score first
    2. exact score tie -> lower item_int_id first

The tie rule is enforced explicitly with a lexicographic sort key rather than relying
on the stability of any particular ``topk`` implementation.

Scores are raw SASRec model scores, **not** probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from recommendation import config

#: PAD is not a real item and can never be recommended.
PAD_ID = config.PAD_ID


class RankingError(ValueError):
    """Raised when a score vector or request is unusable for ranking."""


@dataclass(frozen=True)
class RankedItem:
    """One ranked recommendation."""

    rank: int
    item_id: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (``item_id`` is the integer id)."""
        return {"rank": self.rank, "item_id": self.item_id, "score": self.score}


def validate_score_vector(scores: Sequence[float] | np.ndarray, num_items: int) -> np.ndarray:
    """Validate a raw catalog score vector and return it as float64 ``ndarray``.

    The vector must have length ``num_items + 1`` (index 0 is the PAD slot) and must
    be entirely finite.  Non-finite values fail fast: ranking NaN/Inf would silently
    produce a meaningless order, and production must not clamp or replace them.
    """
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise RankingError(f"num_items must be a positive int, got {num_items!r}")

    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1:
        raise RankingError(f"score vector must be 1-D, got shape {array.shape}")
    expected = num_items + 1
    if array.shape[0] != expected:
        raise RankingError(
            f"score vector must have length num_items + 1 = {expected} "
            f"(index 0 is the PAD slot), got {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        first = int(np.flatnonzero(~np.isfinite(array))[0])
        raise RankingError(
            f"score vector contains {bad} non-finite value(s); "
            f"first at index {first}. NaN/Inf scores cannot be ranked."
        )
    return array


def rank_top_k(
    scores: Sequence[float] | np.ndarray,
    *,
    num_items: int,
    seen_item_ids: Sequence[int] = (),
    k: int = 10,
) -> list[RankedItem]:
    """Return the deterministic top-``k`` recommendations.

    Parameters
    ----------
    scores:
        Raw model scores, length ``num_items + 1``, index 0 = PAD.
    num_items:
        Catalogue size; candidates are ``1..num_items``.
    seen_item_ids:
        Items the caller has already interacted with.  These are excluded from the
        candidate set.  Out-of-catalog ids are ignored rather than fatal, because the
        serving mask must tolerate a caller history that mentions items the catalog
        does not contain (they cannot be recommended anyway).
    k:
        Number of recommendations requested; validated against ``1..100``.

    Returns at most ``min(k, eligible_candidate_count)`` items, possibly empty.
    """
    validate_k(k)
    array = validate_score_vector(scores, num_items)

    # Candidate mask: skip PAD (index 0) and every seen item.
    eligible = np.zeros(num_items + 1, dtype=bool)
    eligible[1:] = True
    if seen_item_ids:
        seen = np.asarray(list(seen_item_ids), dtype=np.int64)
        seen = seen[(seen >= 1) & (seen <= num_items)]
        if seen.size:
            eligible[seen] = False

    candidate_ids = np.flatnonzero(eligible)
    if candidate_ids.size == 0:
        return []

    candidate_scores = array[candidate_ids]
    # Lexicographic key: descending score, then ascending item id.  ``lexsort`` uses
    # the LAST key as primary, so (item_id, -score) sorts by -score first and breaks
    # exact ties by the smaller item id - the frozen rule, enforced explicitly.
    order = np.lexsort((candidate_ids, -candidate_scores))
    chosen = candidate_ids[order][:k]
    chosen_scores = array[chosen]

    return [
        RankedItem(rank=position, item_id=int(item_id), score=float(score))
        for position, (item_id, score) in enumerate(zip(chosen, chosen_scores), start=1)
    ]


def reference_rank_top_k(
    scores: Sequence[float],
    *,
    num_items: int,
    seen_item_ids: Sequence[int] = (),
    k: int = 10,
) -> list[RankedItem]:
    """Explicit-sort oracle for :func:`rank_top_k`, used by the equivalence tests.

    Written as a plain Python full sort with the canonical ``(-score, item_id)`` key.
    """
    validate_k(k)
    array = validate_score_vector(scores, num_items)
    seen = {int(i) for i in seen_item_ids}
    candidates = [i for i in range(1, num_items + 1) if i not in seen]
    candidates.sort(key=lambda item_id: (-array[item_id], item_id))
    return [
        RankedItem(rank=position, item_id=item_id, score=float(array[item_id]))
        for position, item_id in enumerate(candidates[:k], start=1)
    ]


def validate_k(k: int, *, minimum: int = 1, maximum: int = 100) -> int:
    """Validate a requested ``k`` and return it."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise RankingError(f"k must be an integer, got {type(k).__name__}")
    if not minimum <= k <= maximum:
        raise RankingError(f"k must be between {minimum} and {maximum}, got {k}")
    return k


def eligible_candidate_count(num_items: int, seen_item_ids: Sequence[int] = ()) -> int:
    """Number of recommendable items: the catalog minus the seen items."""
    seen = {int(i) for i in seen_item_ids if 1 <= int(i) <= num_items}
    return num_items - len(seen)


__all__ = [
    "PAD_ID",
    "RankedItem",
    "RankingError",
    "eligible_candidate_count",
    "rank_top_k",
    "reference_rank_top_k",
    "validate_k",
    "validate_score_vector",
]
