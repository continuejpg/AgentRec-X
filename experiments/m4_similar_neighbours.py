"""Exact-accelerated similar-item neighbours.

`SimilarItemIndex.neighbours` is correct but scans **every** catalogue identity per query
(``similar_item.py:174``), which measured 2.39 s per seed on this machine -- 13.3 h for the 20,000
M4 users, for the similar-item source alone.

This module computes the **same** result far faster by only scoring identities that share at least
one term with the seed, using an inverted index built from the frozen index's own weight vectors.
A pair with no shared term has a dot product of exactly ``0.0``, which ``neighbours`` already
discards (``similar_item.py:178-181``), so restricting the scan cannot change *which* identities are
eligible.

Floating point is deliberately mirrored, not "improved"
------------------------------------------------------
The reference computes ``dot`` over **raw** weights and divides once by the product of the two
norms (``_cosine``, ``similar_item.py:144-156``), iterating the *smaller* vector and using
``sum(value * right.get(term, 0.0) for ...)`` in ``left.items()`` order. Summation order therefore
affects the last bits, and the last bits decide ties.

This module reproduces that arithmetic exactly:

* raw weights are used for the dot product (no pre-normalisation);
* the same "iterate the smaller vector" rule is applied;
* the same operand order and the same ``sum(...)`` over a generator are kept;
* the division is a single ``dot / (left_norm * right_norm)``, as in the reference.

Consequently the output is **bit-identical**, not merely close, and the equality is asserted by
``tests/test_m4_similar_neighbours.py`` rather than argued. If that test ever fails, the
optimisation must be discarded: the frozen heads are only valid if they are exactly what the
trusted tool would have produced.

Ordering and truncation are copied verbatim from ``similar_item.py:182-189``:
``sorted((-score, catalogue_position, identity))`` then ``[:limit]``, so ties break by catalogue
order.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from recommendation.control.similar_item import SimilarItemIndex

__all__ = ["SparseNeighbourIndex"]


class SparseNeighbourIndex:
    """An inverted index over a :class:`SimilarItemIndex`, with bit-identical output.

    Built once per process from the frozen index; read-only afterwards.
    """

    def __init__(self, index: SimilarItemIndex) -> None:
        self._index = index
        identities: Sequence[str] = index.identities
        self._position: dict[str, int] = {identity: i for i, identity in enumerate(identities)}
        vectors = index._vectors  # noqa: SLF001 - the frozen index owns these; we only read them
        self._vectors = vectors

        postings: dict[str, list[str]] = {}
        for identity, vector in vectors.items():
            for term in vector:
                postings.setdefault(term, []).append(identity)
        # Tuples keep iteration deterministic and immutable.
        self._postings = {term: tuple(members) for term, members in postings.items()}
        self.total = len(identities)
        self.vocabulary_size = index.vocabulary_size

    @property
    def identities(self) -> Sequence[str]:
        """The indexed identities, in catalogue order."""
        return self._index.identities

    @property
    def postings_terms(self) -> int:
        """How many distinct terms the inverted index holds."""
        return len(self._postings)

    def neighbours(self, seed: str, *, limit: int = 10) -> list[tuple[str, int, float]]:
        """Return ``(parent_asin, rank, cosine)`` for ``seed``, matching the reference exactly."""
        seed_vector = self._vectors.get(seed)
        if seed_vector is None:
            return []
        # Candidate set = union of the seed's posting lists. Every identity outside it shares no
        # term with the seed, so its dot product is 0.0 and the reference discards it.
        candidates: set[str] = set()
        for term in seed_vector:
            members = self._postings.get(term)
            if members:
                candidates.update(members)
        candidates.discard(seed)

        norms = self._index._norms  # noqa: SLF001
        seed_norm = norms.get(seed) or 1.0
        scored: list[tuple[float, int, str]] = []
        get = self._vectors.get
        for identity in candidates:
            vector = get(identity)
            # `candidates` only ever holds identities that are in `_vectors`, so this is a
            # defensive check rather than a reachable branch.
            if vector is None:
                continue
            # Mirror the reference: iterate the smaller vector, keeping operand order.
            if len(seed_vector) > len(vector):
                left, right = vector, seed_vector
            else:
                left, right = seed_vector, vector
            dot = sum(value * right.get(term, 0.0) for term, value in left.items())
            if dot == 0.0:
                continue
            # `_cosine` multiplies the two norms then falls back to 1.0 when that product is zero,
            # and it is symmetric in the two norms. The product is formed the same way here.
            other_norm = norms.get(identity) or 1.0
            score = dot / ((seed_norm * other_norm) or 1.0)
            if score <= 0.0:
                continue
            scored.append((-score, self._position[identity], identity))
        scored.sort()
        return [
            (identity, rank, -negative_score)
            for rank, (negative_score, _position, identity) in enumerate(
                scored[: max(0, limit)], start=1
            )
        ]

    def neighbour_counts(self, seeds: Iterable[str], *, limit: int = 100) -> list[int]:
        """Debug helper: neighbour counts for a handful of seeds (tests only)."""
        return [len(self.neighbours(seed, limit=limit)) for seed in seeds]
