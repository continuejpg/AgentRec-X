"""A real item-to-item retrieval source (Phase 4).

`FIND_SIMILAR` had a complete contract and no implementation: the action kind, the
`CandidateSource.SIMILAR_ITEM` member, the `CandidateSourceTool` protocol and the
`CandidatePlane(similar_item_tool=...)` registration point all existed, and nothing supplied a
tool.  This module supplies one.

What kind of retrieval this is, stated plainly
---------------------------------------------
**Lexical item-to-item similarity over catalogue text**, computed as TF-IDF vectors with cosine
similarity.  It is *not* learned embedding retrieval and *not* ANN: there is no trained encoder, no
approximate nearest-neighbour index and no vector database.  `numpy` is the only dependency, and
it is already in the project's environment.

That choice is deliberate.  A lexical index is reproducible from the catalogue alone, needs no
model artifact, no training run and no network, and its similarity is explainable - two products
score because they share weighted terms.  Calling it "embedding ANN retrieval" would overstate it,
and the repository's documentation rules require measurements to be attributed rather than
inflated.

How identity is constrained
---------------------------
The seed is **explicit** and comes from the action's arguments, and this tool does not choose one.
It never returns the seed itself, and it never returns an identity it did not read from the trusted
catalogue.  Grounding is still the plane's job: this tool returns catalogue identities and its own
ordered scores, and the `GroundingVerifier` decides whether each one is real.  An identity the
trusted mapping does not know is recorded as an UNGROUNDED audit entry by the plane, exactly as for
any other source - so this tool cannot smuggle a candidate in either.

Scores are not fused
--------------------
The scores here are cosine similarities in ``[0, 1]``.  They are recorded with
``score_kind="item_item_similarity"`` and never compared against a SASRec logit or a BM25 value;
the ledger's rank-based fusion is the only cross-source combination, and it uses order alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from .arguments import CandidateSource

__all__ = ["SimilarItemIndex", "SimilarItemSource", "TOKEN_PATTERN"]

#: What counts as a term: runs of letters and digits, lowercased.  Deliberately simple and
#: deterministic - no stemming, no stop-word list, no language model.  A term is a token, and the
#: weighting below is what makes shared tokens meaningful.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

#: Text fields combined into a product's document, in a fixed order.  ``details`` is included
#: because the normalised catalogue keeps colour, weight and material there.
_DOCUMENT_FIELDS: tuple[str, ...] = (
    "title",
    "subtitle",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
)


def _terms(text: str) -> list[str]:
    """Tokenise one text value into lowercase terms."""
    return TOKEN_PATTERN.findall(str(text).casefold())


class SimilarItemIndex:
    """A TF-IDF index over catalogue text, with cosine nearest-neighbour lookup.

    The index is built once from the trusted catalogue and is **read-only** afterwards: there is no
    method to add, remove or reweight a document, so nothing in a run can change what "similar"
    means for that run.

    TF-IDF is used rather than raw token overlap because a term appearing in every product ("widget"
    in this repository's fixture) carries no discriminative information; the inverse-document
    frequency factor is what makes a shared rare term score higher than a shared common one.  The
    vectors are L2-normalised, so the dot product is the cosine similarity.
    """

    def __init__(self, documents: dict[str, str], *, identity_order: Sequence[str] = ()) -> None:
        if not documents:
            raise ValueError("a similar-item index needs at least one document")
        # Preserve the caller's order so ties break deterministically by catalogue order rather
        # than by dictionary insertion.
        self._identities: tuple[str, ...] = tuple(
            identity for identity in (identity_order or documents) if identity in documents
        )
        if not self._identities:
            raise ValueError("identity_order names no known document")
        self._documents = {identity: documents[identity] for identity in self._identities}

        tokenised = {identity: _terms(self._documents[identity]) for identity in self._identities}
        self._document_frequency: Counter[str] = Counter()
        for terms in tokenised.values():
            self._document_frequency.update(set(terms))

        total = len(self._identities)
        # Smoothed inverse document frequency: a term in every document gets a small positive
        # weight rather than zero, so a degenerate corpus does not produce all-zero vectors.
        self._idf: dict[str, float] = {
            term: math.log((1.0 + total) / (1.0 + count)) + 1.0
            for term, count in self._document_frequency.items()
        }
        self._vectors: dict[str, dict[str, float]] = {
            identity: self._weight(terms) for identity, terms in tokenised.items()
        }
        self._norms: dict[str, float] = {
            identity: math.sqrt(sum(value * value for value in vector.values())) or 1.0
            for identity, vector in self._vectors.items()
        }

    @property
    def identities(self) -> tuple[str, ...]:
        """The indexed identities, in catalogue order."""
        return self._identities

    @property
    def vocabulary_size(self) -> int:
        """How many distinct terms the index knows."""
        return len(self._idf)

    def _weight(self, terms: Iterable[str]) -> dict[str, float]:
        """TF-IDF weight the terms of one document."""
        counts = Counter(terms)
        if not counts:
            return {}
        length = sum(counts.values())
        return {
            term: (count / length) * self._idf.get(term, 0.0) for term, count in counts.items()
        }

    def vector_for(self, text: str) -> dict[str, float]:
        """Weight arbitrary text against this index, for a query that is not a document."""
        return self._weight(_terms(text))

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity between two indexed identities."""
        return self._cosine(self._vectors.get(left, {}), self._vectors.get(right, {}))

    def _cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        """Cosine similarity of two sparse vectors (unnormalised input)."""
        if not left or not right:
            return 0.0
        # Iterate the smaller vector: the dot product only has terms in common.
        if len(left) > len(right):
            left, right = right, left
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        if dot == 0.0:
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left.values())) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right.values())) or 1.0
        return dot / (left_norm * right_norm)

    def neighbours(
        self, seed: str, *, limit: int = 10
    ) -> list[tuple[str, int, float]]:
        """Return ``(parent_asin, rank, cosine)`` for the seed's nearest neighbours.

        Ordered by descending similarity, then by catalogue order for ties, so the result is
        reproducible without relying on dictionary iteration.  The seed is excluded - "products
        like this one" that includes the product itself would waste the retrieval.

        A seed the index does not know yields an empty list rather than an error: the caller
        (the plane) owns the decision of what an unknown seed means, and this tool must not
        invent a neighbour for it.
        """
        if seed not in self._vectors:
            return []
        scored: list[tuple[float, int, str]] = []
        for position, identity in enumerate(self._identities):
            if identity == seed:
                continue
            score = self._cosine(self._vectors[seed], self._vectors[identity])
            if score <= 0.0:
                # A zero-similarity neighbour is not a neighbour.  Returning it would pad the
                # result with unrelated products and make the rank meaningless.
                continue
            scored.append((-score, position, identity))
        scored.sort()
        return [
            (identity, rank, -negative_score)
            for rank, (negative_score, _position, identity) in enumerate(
                scored[: max(0, limit)], start=1
            )
        ]


class SimilarItemSource:
    """The trusted ``SIMILAR_ITEM`` candidate source over a :class:`SimilarItemIndex`.

    Implements the repository's :class:`CandidateSourceTool` protocol, so the candidate plane can
    register it exactly like the history recommender and the catalogue search.  It does not ground
    identities, does not touch the ledger and does not read run state: it answers one question -
    which catalogue products are lexically nearest this one - and the plane does the rest.
    """

    def __init__(self, index: SimilarItemIndex, *, source: str = "similar_item_index") -> None:
        self._index = index
        #: Where the index came from, for provenance and diagnostics.
        self._origin = str(source)

    @property
    def source(self) -> CandidateSource:
        """Which :class:`CandidateSource` this tool implements."""
        return CandidateSource.SIMILAR_ITEM

    @property
    def score_kind(self) -> str:
        """What this tool's scores mean.

        Named explicitly so a consumer cannot mistake a cosine similarity for a SASRec logit or a
        BM25 value, and so the ledger never fuses them numerically.
        """
        return "item_item_similarity"

    @property
    def index(self) -> SimilarItemIndex:
        """The read-only index behind this source."""
        return self._index

    @property
    def origin(self) -> str:
        """A description of where the index came from."""
        return self._origin

    def covers(self, seed: str) -> bool:
        """True when the index knows this seed, so the plane can tell "no neighbours" from
        "unknown seed"."""
        return seed in self._index.identities

    def propose(self, *, arguments: Any, limit: int) -> list[tuple[str, int, float]]:
        """Return the seed's neighbours as ``(parent_asin, rank, score)`` triples.

        The seed is read from the action's arguments and is **never** chosen here.  An action
        without a seed cannot be served, because the tool has no basis on which to pick one and
        picking would be exactly the implicit target selection Phase 2.1 removed.
        """
        seed = getattr(arguments, "seed_parent_asin", None)
        if not seed:
            raise ValueError(
                "find_similar requires an explicit seed_parent_asin; this source never chooses a "
                "seed itself"
            )
        return self._index.neighbours(str(seed), limit=int(limit))


def build_similar_item_index(
    catalogue: Any, *, fields: Sequence[str] = _DOCUMENT_FIELDS
) -> SimilarItemIndex:
    """Build a :class:`SimilarItemIndex` from a trusted catalogue.

    ``catalogue`` is the normalised metadata index, so the documents are exactly the trusted
    records the rest of the system reads - this builder invents no text and reads no other source.
    """
    records = getattr(catalogue, "records", None)
    if records is None:
        raise TypeError("build_similar_item_index needs a catalogue exposing 'records'")
    documents: dict[str, str] = {}
    order: list[str] = []
    for identity, record in records.items():
        order.append(str(identity))
        documents[str(identity)] = _document_text(record, fields)
    return SimilarItemIndex(documents, identity_order=order)


def _document_text(record: Any, fields: Sequence[str]) -> str:
    """Flatten the configured fields of one catalogue record into one document string."""
    parts: list[str] = []
    for field in fields:
        value = getattr(record, field, None)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    # ``details`` is a mapping of attribute -> value in the normalised catalogue; including it
    # lets a shared material or colour contribute to similarity.
    details = getattr(record, "details", None)
    if details:
        for key, value in details:
            parts.append(str(key))
            parts.append(str(value))
    return " ".join(parts)
