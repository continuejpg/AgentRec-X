"""Full-catalogue lexical search: a trusted candidate source (Stage 2).

This is the first candidate source that is not the sequential recommender, so it is the
place the multi-source trust model is most likely to be got wrong.  Two things are
deliberately different from the accepted Milestone 8 retriever
(:mod:`recommendation.rag.retrieval`):

**Scope.**  The Milestone 8 retriever is *candidate-scoped*: it may only select evidence
from metadata belonging to candidates the Tool already produced, which is why RAG cannot
widen a candidate set.  This module is the opposite by design - it searches the **whole**
catalogue - and that is exactly why it is a separate component with its own trust story
rather than a flag on the existing one.  Widening the candidate scope is a capability the
policy may *select*, and it happens only here.

**What it returns.**  Milestone 8 returns evidence passages attributed to a candidate.
This returns *candidate identities*: the caller gets products, and it is the grounding
verifier's job to confirm each identity before the ledger accepts it.  Search never
asserts that a product exists; it reports what the catalogue contains.

Shared with Milestone 8: the tokenizer, the BM25 constants and the idf function are
imported from :mod:`recommendation.rag.retrieval`, so the two retrievers cannot drift apart
in how they score text.  What differs is the document scope and the output type, not the
maths.

Scoring honesty: the score this source reports is a BM25 value over catalogue text.  It is
not a SASRec logit and not a probability, and the ledger records it under
``score_kind="bm25"`` precisely so that no later stage can compare it against another
source's score.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from recommendation.rag.retrieval import BM25_B, BM25_K1, _idf, tokenize

__all__ = [
    "CATALOG_SEARCH_SCORE_KIND",
    "CatalogSearchHit",
    "CatalogSearchResult",
    "CatalogSearchSource",
    "CatalogTextLike",
]

#: How this source's scores must be labelled wherever they are recorded.
CATALOG_SEARCH_SCORE_KIND = "bm25"

#: Catalogue fields searched, in a fixed evaluation order.  Every field is normalized
#: catalogue text; nothing here is generated, summarised or inferred.
SEARCHABLE_FIELDS: tuple[str, ...] = (
    "title",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
)


@runtime_checkable
class CatalogTextLike(Protocol):
    """The trusted catalogue as this source needs it: identity plus normalized text.

    Satisfied by :class:`~recommendation.catalog.metadata.MetadataIndex` (its ``records``
    mapping) without modification.
    """

    @property
    def records(self) -> dict[str, Any]:
        """Mapping of canonical identity to a normalized metadata record."""
        ...


class CatalogSearchHit:
    """One catalogue record matched by a query, with its BM25 score and matched fields."""

    __slots__ = ("parent_asin", "score", "matched_fields", "title")

    def __init__(
        self,
        *,
        parent_asin: str,
        score: float,
        matched_fields: tuple[str, ...],
        title: str | None,
    ) -> None:
        self.parent_asin = parent_asin
        self.score = score
        self.matched_fields = matched_fields
        #: A display title copied verbatim from the catalogue record, or ``None``.  Never
        #: generated: a record without a title reports ``None``.
        self.title = title

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "parent_asin": self.parent_asin,
            "score": self.score,
            "matched_fields": list(self.matched_fields),
            "title": self.title,
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"CatalogSearchHit({self.parent_asin!r}, {self.score:.4f})"


class CatalogSearchResult:
    """The outcome of one full-catalogue search."""

    __slots__ = ("hits", "query", "matched", "catalogue_size")

    def __init__(
        self,
        *,
        hits: tuple[CatalogSearchHit, ...],
        query: str,
        matched: bool,
        catalogue_size: int,
    ) -> None:
        self.hits = hits
        self.query = query
        #: False when no catalogue record contained any query term.  An honest "no
        #: lexical match" rather than a fabricated fallback list: a zero-hit search is a
        #: legitimate observation that lets the policy choose a different source.
        self.matched = matched
        self.catalogue_size = catalogue_size

    @property
    def parent_asins(self) -> tuple[str, ...]:
        """Matched identities in score order."""
        return tuple(hit.parent_asin for hit in self.hits)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "query": self.query,
            "matched": self.matched,
            "hits": [hit.as_dict() for hit in self.hits],
            "catalogue_size": self.catalogue_size,
        }


class CatalogSearchSource:
    """Search the full catalogue lexically, deterministically and offline.

    The index is built **once** from catalogue records and then reused, so a per-turn search
    is a scoring pass rather than a re-tokenisation of the catalogue.

    Parameters
    ----------
    catalog:
        Any :class:`CatalogTextLike`.  Normally the process-scoped
        :class:`~recommendation.catalog.MetadataIndex`, so the demo does not pay for a
        second copy of a 300 MB artifact.
    fields:
        Which catalogue fields to index.  Defaults to :data:`SEARCHABLE_FIELDS`.
    """

    def __init__(
        self,
        catalog: CatalogTextLike,
        *,
        fields: tuple[str, ...] = SEARCHABLE_FIELDS,
    ) -> None:
        if not isinstance(catalog, CatalogTextLike):
            raise TypeError("catalog must expose a 'records' mapping")
        self._catalog = catalog
        self._fields = tuple(fields)
        self._documents: list[tuple[str, dict[str, int], int, str | None]] = []
        self._document_frequency: dict[str, int] = {}
        self._average_length = 1.0
        self._build()

    # -- metadata ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Stable source name, recorded in the ledger as ``catalog_search``."""
        return "catalog_search"

    @property
    def catalogue_size(self) -> int:
        """How many catalogue records are indexed."""
        return len(self._documents)

    @property
    def fields(self) -> tuple[str, ...]:
        """The catalogue fields this source searches."""
        return self._fields

    # -- index ------------------------------------------------------------- #

    @staticmethod
    def _field_text(record: Any, field: str) -> str:
        """Return one searchable field's text, or an empty string when absent.

        Multi-valued fields are joined; nothing is paraphrased or truncated beyond the
        tokenizer's own behaviour.  A missing field contributes no text rather than a
        placeholder.
        """
        value = getattr(record, field, None)
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (tuple, list)):
            parts = [str(item) for item in value if isinstance(item, str) and item.strip()]
            return " ".join(parts)
        return ""

    def _build(self) -> None:
        """Tokenise every catalogue record once and precompute document frequencies."""
        frequency: dict[str, int] = {}
        for parent_asin, record in self._catalog.records.items():
            text = " ".join(
                part for part in (self._field_text(record, f) for f in self._fields) if part
            )
            tokens = tokenize(text)
            if not tokens:
                # A record with no searchable text is not indexed.  It stays discoverable
                # by identity lookup, which is the honest way to reach it.
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            title = getattr(record, "title", None)
            self._documents.append(
                (parent_asin, counts, len(tokens), title if isinstance(title, str) else None)
            )
            for term in counts:
                frequency[term] = frequency.get(term, 0) + 1
        self._document_frequency = frequency
        if self._documents:
            self._average_length = sum(doc[2] for doc in self._documents) / len(self._documents)

    # -- search ------------------------------------------------------------ #

    def search(self, query: str, *, limit: int = 10) -> CatalogSearchResult:
        """Return the top ``limit`` catalogue records by BM25 over the query.

        Deterministic: ties break on identity, so repeated searches cannot reorder.  A
        blank or unmatched query returns zero hits with ``matched=False`` rather than an
        arbitrary slice of the catalogue - inventing a result set for a query that matched
        nothing would be exactly the fabrication this plane exists to prevent.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        terms = tuple(dict.fromkeys(tokenize(query)))
        if not terms:
            return CatalogSearchResult(
                hits=(), query=query, matched=False, catalogue_size=self.catalogue_size
            )

        document_count = len(self._documents)
        if document_count == 0:
            return CatalogSearchResult(
                hits=(), query=query, matched=False, catalogue_size=0
            )

        scored: list[tuple[float, str, tuple[str, ...], str | None]] = []
        for parent_asin, counts, length, title in self._documents:
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = _idf(self._document_frequency.get(term, 0), document_count)
                if idf <= 0.0:
                    continue
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * length / self._average_length
                )
                score += idf * (frequency * (BM25_K1 + 1.0)) / denominator
            if score <= 0.0:
                continue
            scored.append((score, parent_asin, (), title))

        if not scored:
            # Terms existed but no document scored: report the honest miss.
            return CatalogSearchResult(
                hits=(), query=query, matched=False, catalogue_size=self.catalogue_size
            )

        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        hits = tuple(
            CatalogSearchHit(
                parent_asin=parent_asin,
                score=round(score, 6),
                matched_fields=fields,
                title=title,
            )
            for score, parent_asin, fields, title in scored[:limit]
        )
        return CatalogSearchResult(
            hits=hits, query=query, matched=True, catalogue_size=self.catalogue_size
        )

    def candidates(
        self, query: str, *, limit: int = 10
    ) -> list[tuple[str, int, float]]:
        """Return ``(parent_asin, rank, score)`` triples in the shape the ledger records.

        The rank is this source's own 1-based rank, preserved verbatim by the ledger.
        """
        result = self.search(query, limit=limit)
        return [
            (hit.parent_asin, position, hit.score)
            for position, hit in enumerate(result.hits, start=1)
        ]

    # -- introspection ----------------------------------------------------- #

    def status(self) -> dict[str, Any]:
        """Return a small JSON-serialisable description of the index."""
        return {
            "source": self.name,
            "indexed_records": self.catalogue_size,
            "fields": list(self._fields),
            "score_kind": CATALOG_SEARCH_SCORE_KIND,
        }
