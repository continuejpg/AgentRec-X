"""Deterministic implicit-feedback ItemCF baseline.

This is the first reference recommender for AgentRec-X: intentionally simple,
interpretable and reproducible.  It contains **no evaluation logic** - scoring is
its only output, and it is always consumed through the Milestone 2A unified
evaluator.

Training-data contract
----------------------
The model is fitted **once** from the train histories of the evaluation cohort::

    for an eligible sequence [i1, ..., i(n-2), i(n-1), in]:
        fit data = [i1, ..., i(n-2)]        (train_history only)

The validation target ``i(n-1)`` and the test target ``in`` are never used to build
item frequencies, co-occurrences or similarities.  At *inference* time the
validation interaction may be supplied as history (it precedes the test target),
but it is never folded back into the similarity matrix - this milestone does not
refit on train+validation.

Feedback semantics
------------------
Pure implicit feedback: the presence of an interaction is the signal, and ratings
are ignored.  Within one user, an item counts **once** no matter how many times it
was consumed::

    unique_items = unique(train_history)

Similarity
----------
For items ``i`` and ``j``::

    freq(i)   = number of training users whose unique train history contains i
    cooc(i,j) = number of training users whose unique train history contains both
    sim(i,j)  = cooc(i,j) / sqrt(freq(i) * freq(j))     for i != j
    sim(i,i)  = 0                                      (self-similarity excluded)

No rating weighting, time decay, IUF, popularity prior or learned parameter is
applied.  Similarity is symmetric by construction.

Representation
--------------
Similarities are stored sparsely as ``item_id -> {neighbour_id: similarity}`` and
``item_id -> {neighbour_id: cooccurrence}``; a dense ``num_items x num_items``
matrix is never materialised.

Scoring
-------
For an inference history::

    history_items = unique(history)
    score(j)      = sum(sim(i, j) for i in history_items)

Items with no learned similarity score exactly ``0.0`` (no popularity fallback).
The scorer returns one score per catalog item, aligned by item id, with length
``num_items + 1`` (index 0 is PAD).  The scorer performs **no** masking: it does not
remove PAD, seen items or targets.  PAD exclusion, seen-item masking, target
retention, tie handling, ranking and metrics all belong to the Milestone 2A
evaluator and stay there.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from recommendation import config

#: Human-readable identifier stored in fit statistics.
MODEL_NAME = "itemcf_cosine"

#: Score given to every item that shares no similarity with the history.
DEFAULT_SCORE = 0.0


class ItemCFError(ValueError):
    """Raised when ItemCF is given data that violates the id or history contract."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def validate_history(
    history: Iterable[int],
    num_items: int,
    *,
    context: str = "history",
) -> tuple[int, ...]:
    """Validate one inference/training history and return it as a tuple.

    Every entry must be a real catalog item (``1..num_items``).  PAD, negatives,
    out-of-range ids and non-integers are rejected loudly: silently dropping them
    would hide a corrupt artifact or an indexing bug.
    """
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
        raise ItemCFError(f"num_items must be a positive int, got {num_items!r}")

    items = tuple(history)
    for item_id in items:
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise ItemCFError(
                f"{context} entry must be an int, got {type(item_id).__name__}: {item_id!r}"
            )
        if item_id == config.PAD_ID:
            raise ItemCFError(
                f"{context} contains the PAD slot {config.PAD_ID}; "
                "PAD is not a real catalog item"
            )
        if not config.FIRST_REAL_ID <= item_id <= num_items:
            raise ItemCFError(
                f"{context} entry {item_id} outside catalog "
                f"[{config.FIRST_REAL_ID}, {num_items}]"
            )
    return items


def unique_items(history: Iterable[int]) -> tuple[int, ...]:
    """Return the distinct items of a history, in ascending id order.

    Ordering is explicit (ascending id) rather than insertion order, so nothing
    downstream can depend on how the history happened to be iterated.  Repetition
    is collapsed here, which is what makes repeated interactions count once.
    """
    return tuple(sorted(set(history)))


# --------------------------------------------------------------------------- #
# Fit statistics
# --------------------------------------------------------------------------- #


@dataclass
class ItemCFFitStats:
    """Auditable description of what a fit used and produced."""

    model: str
    num_items: int
    num_fit_users: int
    num_train_interactions: int
    num_unique_train_items: int
    num_similarity_pairs: int
    fit_seconds: float
    min_cooccurrence: int
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def item_training_coverage(self) -> float:
        """Fraction of the catalog that appears in at least one train history."""
        if self.num_items <= 0:
            return 0.0
        return self.num_unique_train_items / self.num_items

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the statistics."""
        payload = {
            "model": self.model,
            "num_items": self.num_items,
            "num_fit_users": self.num_fit_users,
            "num_train_interactions": self.num_train_interactions,
            "num_unique_train_items": self.num_unique_train_items,
            "item_training_coverage": round(self.item_training_coverage, 6),
            "num_similarity_pairs": self.num_similarity_pairs,
            "min_cooccurrence": self.min_cooccurrence,
            "fit_seconds": round(self.fit_seconds, 6),
        }
        payload.update(self.extra)
        return payload


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class ItemCF:
    """Cosine item-item similarity over implicit feedback.

    Parameters
    ----------
    num_items:
        Catalogue size; the model scores items ``1..num_items``.
    min_cooccurrence:
        Minimum raw co-occurrence count for a pair to be retained.  Defaults to
        ``1`` (keep every observed pair).  This is a *noise floor on observations*,
        not a tuned hyperparameter: this milestone performs no hyperparameter
        search.
    """

    def __init__(self, num_items: int, *, min_cooccurrence: int = 1) -> None:
        if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items < 1:
            raise ItemCFError(f"num_items must be a positive int, got {num_items!r}")
        if isinstance(min_cooccurrence, bool) or not isinstance(min_cooccurrence, int):
            raise ItemCFError(
                f"min_cooccurrence must be an int, got {type(min_cooccurrence).__name__}"
            )
        if min_cooccurrence < 1:
            raise ItemCFError(f"min_cooccurrence must be >= 1, got {min_cooccurrence}")

        self.num_items = num_items
        self.min_cooccurrence = min_cooccurrence

        self._item_freq: dict[int, int] = {}
        self._cooccurrence: dict[int, dict[int, int]] = {}
        self._similarity: dict[int, dict[int, float]] = {}
        self._stats: ItemCFFitStats | None = None

    # -- properties -------------------------------------------------------- #

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._stats is not None

    @property
    def item_freq(self) -> dict[int, int]:
        """Item support: number of training users whose unique history contains it."""
        return self._item_freq

    @property
    def similarity(self) -> dict[int, dict[int, float]]:
        """Sparse similarity map ``item -> {neighbour: sim}``."""
        return self._similarity

    @property
    def cooccurrence(self) -> dict[int, dict[int, int]]:
        """Sparse co-occurrence map ``item -> {neighbour: cooc}``."""
        return self._cooccurrence

    @property
    def stats(self) -> ItemCFFitStats:
        """Fit statistics; raises if the model has not been fitted."""
        if self._stats is None:
            raise ItemCFError("model has not been fitted; call fit() first")
        return self._stats

    def neighbors(self, item_id: int) -> dict[int, float]:
        """Return the learned neighbours of ``item_id`` (empty when unseen)."""
        return self._similarity.get(item_id, {})

    def similarity_between(self, i: int, j: int) -> float:
        """Similarity of the unordered pair ``{i, j}``; ``0.0`` when unobserved.

        Similarity is symmetric, but it is *stored* under a single canonical key per
        pair (the lower ``(frequency, id)`` item) to keep the representation sparse.
        These accessors hide that detail so callers never have to know which side the
        pair was filed under.
        """
        if i == j:
            return 0.0  # self-similarity is excluded by definition
        neighbours = self._similarity.get(i)
        if neighbours is not None and j in neighbours:
            return neighbours[j]
        return self._similarity.get(j, {}).get(i, 0.0)

    def cooccurrence_between(self, i: int, j: int) -> int:
        """Raw co-occurrence count of the unordered pair ``{i, j}``; 0 if unobserved."""
        if i == j:
            return 0
        neighbours = self._cooccurrence.get(i)
        if neighbours is not None and j in neighbours:
            return neighbours[j]
        return self._cooccurrence.get(j, {}).get(i, 0)

    # -- fitting ----------------------------------------------------------- #

    def fit(
        self,
        train_histories: Iterable[Sequence[int]],
        *,
        validate: bool = True,
    ) -> ItemCFFitStats:
        """Fit item frequencies and cosine similarities from train histories.

        ``train_histories`` must be exactly the ``train_history`` of each
        evaluation-eligible user and nothing else.  Those histories may contain
        repeated items; each distinct item counts once per user.

        The pass structure is:

        1. compute per-user unique item sets and item frequencies;
        2. accumulate co-occurrence by iterating unordered item pairs per user, with
           each user's items ordered by ascending frequency before pair generation
           so that small buckets are combined first;
        3. compute ``cooc / sqrt(freq_i * freq_j)`` for the retained pairs.

        Every structure is a ``dict`` keyed by item id and written in ascending id
        order, so results never depend on set/dict iteration order.
        """
        started = time.perf_counter()

        item_freq: dict[int, int] = defaultdict(int)
        cooc: dict[int, dict[int, int]] = {}
        num_users = 0
        num_interactions = 0
        unique_all: set[int] = set()

        # Items ordered by ascending frequency, ascending id - the shared order for
        # both the final similarity lists and the pair-generation inner list.
        ordered_items: list[int] = []
        user_unique_sets: list[list[int]] = []

        for user_index, history in enumerate(train_histories):
            raw = validate_history(
                history, self.num_items, context=f"train history for user #{user_index}"
            ) if validate else tuple(history)
            distinct = sorted(set(raw))
            if not distinct:
                continue

            num_users += 1
            num_interactions += len(raw)
            for item_id in distinct:
                item_freq[item_id] += 1
                unique_all.add(item_id)
            user_unique_sets.append(distinct)

        ordered_items = sorted(item_freq, key=lambda i: (item_freq[i], i))
        rank_of = {item_id: position for position, item_id in enumerate(ordered_items)}

        for distinct in user_unique_sets:
            # ascending (frequency, id) so the outer loop has the small buckets
            ordered = sorted(distinct, key=lambda i: (item_freq[i], rank_of[i]))
            size = len(ordered)
            for left in range(size):
                a = ordered[left]
                a_count = item_freq[a]
                inner = cooc.get(a)
                if inner is None:
                    inner = {}
                    cooc[a] = inner
                for right in range(left + 1, size):
                    b = ordered[right]
                    # skip pairs that cannot clear the retention floor
                    if a_count < self.min_cooccurrence and item_freq[b] < self.min_cooccurrence:
                        continue
                    inner[b] = inner.get(b, 0) + 1

        # Store every retained pair under *both* items.  Scoring is
        # ``sum(sim(i, j) for i in history)``, so a pair must be reachable from
        # either endpoint; a single-canonical-key layout would make the score depend
        # on which endpoint happened to be iterated first.  The memory cost is 2x the
        # pair count, which is still far below a dense matrix.
        similarity: dict[int, dict[int, float]] = {}
        cooccurrence: dict[int, dict[int, int]] = {}
        num_pairs = 0
        for a in sorted(cooc):
            a_freq = item_freq[a]
            for b in sorted(cooc[a]):
                count = cooc[a][b]
                if count < self.min_cooccurrence:
                    # Drop the pair from *both* views: keeping it in one and not the
                    # other would make them disagree (the spec's
                    # "no-cooccurrence pair gives zero" case).
                    continue
                denominator = math.sqrt(a_freq * item_freq[b])
                if denominator <= 0.0:  # pragma: no cover - freq is always >= 1
                    continue
                value = count / denominator
                similarity.setdefault(a, {})[b] = value
                similarity.setdefault(b, {})[a] = value
                cooccurrence.setdefault(a, {})[b] = count
                cooccurrence.setdefault(b, {})[a] = count
                num_pairs += 1

        self._item_freq = {i: item_freq[i] for i in sorted(item_freq)}
        self._cooccurrence = {
            a: {b: cooccurrence[a][b] for b in sorted(cooccurrence[a])}
            for a in sorted(cooccurrence)
        }
        self._similarity = {
            a: {b: similarity[a][b] for b in sorted(similarity[a])}
            for a in sorted(similarity)
        }
        self._stats = ItemCFFitStats(
            model=MODEL_NAME,
            num_items=self.num_items,
            num_fit_users=num_users,
            num_train_interactions=num_interactions,
            num_unique_train_items=len(unique_all),
            num_similarity_pairs=num_pairs,
            fit_seconds=time.perf_counter() - started,
            min_cooccurrence=self.min_cooccurrence,
            extra={
                "similarity": "cosine",
                "feedback": "implicit",
                "rating_weighting": False,
                "self_similarity": 0.0,
                "id_dtype": "int",
            },
        )
        return self._stats

    # -- scoring ----------------------------------------------------------- #

    def score(
        self,
        history: Sequence[int],
        *,
        validate: bool = True,
    ) -> list[float]:
        """Return a score vector for ``history``, aligned by item id, length N+1.

        ``scores[item_id]`` is the summed similarity between that item and the
        *distinct* history items.  The vector always covers the whole catalog, PAD
        slot included, and contains only finite values.

        The scorer does **not** mask anything: seen items, PAD and targets may all
        carry a score here, because masking is the evaluator's responsibility.
        """
        if self._stats is None:
            raise ItemCFError("model has not been fitted; call fit() first")

        raw = (
            validate_history(history, self.num_items, context="inference history")
            if validate
            else tuple(history)
        )
        distinct = unique_items(raw)

        scores = [DEFAULT_SCORE] * (self.num_items + 1)
        if not distinct:
            return scores

        # Accumulate only over items that actually have learned neighbours.
        for item_id in distinct:
            neighbours = self._similarity.get(item_id)
            if not neighbours:
                continue
            for neighbour_id, similarity in neighbours.items():
                scores[neighbour_id] += similarity
        return scores

    # -- persistence ------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the fitted model."""
        if self._stats is None:
            raise ItemCFError("model has not been fitted; call fit() first")
        return {
            "model": MODEL_NAME,
            "num_items": self.num_items,
            "min_cooccurrence": self.min_cooccurrence,
            "stats": self._stats.as_dict(),
            "item_freq": {str(k): self._item_freq[k] for k in sorted(self._item_freq)},
            "similarity": {
                str(a): {str(b): self._similarity[a][b] for b in sorted(self._similarity[a])}
                for a in sorted(self._similarity)
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "fitted" if self.is_fitted else "unfitted"
        pairs = self._stats.num_similarity_pairs if self._stats else 0
        return (
            f"ItemCF(num_items={self.num_items}, {state}, "
            f"items={len(self._item_freq)}, pairs={pairs})"
        )


# --------------------------------------------------------------------------- #
# Convenience: fit from an evaluation cohort
# --------------------------------------------------------------------------- #


def fit_from_cohort(
    cases: Sequence[Any],
    num_items: int,
    *,
    min_cooccurrence: int = 1,
) -> tuple[ItemCF, ItemCFFitStats]:
    """Fit ItemCF from the ``train_history`` of Milestone 2A evaluation cases.

    This is the only sanctioned way to build the fit data: it reads
    ``case.train_history`` and deliberately never touches ``validation_target`` or
    ``test_target``, which is what guarantees the temporal contract.
    """
    model = ItemCF(num_items, min_cooccurrence=min_cooccurrence)
    stats = model.fit([case.train_history for case in cases])
    return model, stats


def make_scorer(model: ItemCF):
    """Return a Milestone 2A compatible scorer for ``model``.

    The returned callable has the evaluator's signature
    ``score_fn(history, target_item_id) -> Sequence[float]``.  The target argument is
    intentionally unused: the model scores the whole catalog and the evaluator
    applies masking, target retention and ranking.
    """

    def score_fn(history: tuple[int, ...], target_item_id: int) -> list[float]:
        return model.score(history)

    return score_fn


def similarity_pair_count(model: ItemCF) -> int:
    """Number of directed ``(i, j)`` pairs with a nonzero learned similarity."""
    return sum(len(neighbours) for neighbours in model.similarity.values())
