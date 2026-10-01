"""Full-catalogue lexical retrieval as a serving candidate source.

This is the third source of Serving V1: the *catalogue lexical / metadata* retriever.  It answers
"which catalogue items does this text look like?" over the whole catalogue, and returns **ranked
identities** — never candidates it invented, never a fallback slice when nothing matched.

Layering
--------
The BM25 constants, the tokenizer and the IDF are imported from
:mod:`recommendation.rag.retrieval`, which is the same module the candidate-scoped enricher and the
research catalogue source use.  Text is therefore scored the same way everywhere in the repository;
what differs between this retriever and the enricher is the *document scope* (whole catalogue vs the
current candidates) and the output type (a ranking vs attributed evidence).

The scoring loop is deliberately small and mirrors the research catalogue source's rule exactly:
BM25 over the unique query terms, a non-positive score is not a hit, and ties break on identity so a
repeat search cannot reorder.  A behavioural equivalence test
(``tests/test_serving_fusion.py``) asserts this retriever and
``recommendation.control.catalog_search.CatalogSearchSource`` return the same order for the same
catalogue and query, so the two cannot drift apart silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from recommendation.rag.retrieval import BM25_B, BM25_K1, _idf, tokenize

__all__ = [
    "LEXICAL_SCORE_KIND",
    "SEARCHABLE_FIELDS",
    "CatalogueLexicalIndex",
    "field_text",
    "query_text_for",
]

#: What this source's score means, so a BM25 value is never read as a model score.
LEXICAL_SCORE_KIND = "bm25"

#: Catalogue fields searched, in a fixed order.  Every field is normalized catalogue text;
#: nothing here is generated, summarised or inferred.
SEARCHABLE_FIELDS: tuple[str, ...] = (
    "title",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
)


def field_text(record: Any, field: str) -> str:
    """Return one searchable field's text, or an empty string when absent.

    Multi-valued fields are joined; a missing field contributes nothing rather than a
    placeholder.  Identical in behaviour to the research catalogue source's helper.
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


#: How many of the most recent history items may contribute to the lexical query.
DEFAULT_QUERY_HISTORY_ITEMS = 5

#: Hard character bound on the query, applied after joining.
MAX_QUERY_CHARS = 512


def query_text_for(
    records: Sequence[Any],
    *,
    fields: Sequence[str] = (SEARCHABLE_FIELDS[0],),
    history_items: int = DEFAULT_QUERY_HISTORY_ITEMS,
    max_chars: int = MAX_QUERY_CHARS,
) -> str:
    """Build a bounded lexical query from the **most recent** catalogue records.

    The query is a function of trusted catalogue text only — never of the user's message — so the
    candidate set stays a function of trusted state.

    It is deliberately *bounded*, and that is a serving requirement rather than a refinement.
    Concatenating six searchable fields of a 20-item history produced thousands of distinct terms
    and measured ~8.4 s per request: the cost was the query, not the scoring.  The audited metadata
    arm likewise used small sampled text (5 seeds x ~100 characters) rather than whole histories.
    Titles are used by default because they are short and high-signal, and the bound is applied
    after joining so it holds whatever the caller passes.
    """
    if history_items < 1:
        raise ValueError("history_items must be >= 1")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    recent = [record for record in records if record is not None][-history_items:]
    parts: list[str] = []
    for record in recent:
        for field in fields:
            text = field_text(record, field)
            if text:
                parts.append(text)
    return " ".join(parts)[:max_chars]


@dataclass(frozen=True)
class CatalogueLexicalIndex:
    """A reusable BM25 index over the whole catalogue.

    Built once per process: tokenising 156k records per request would dominate the serving
    budget, and the catalogue is static for the lifetime of the process.
    """

    fields: tuple[str, ...]
    catalogue_size: int
    indexed_documents: int
    _documents: tuple[tuple[str, dict[str, int], int, str | None], ...]
    _document_frequency: dict[str, int]
    _average_length: float
    #: term -> document positions containing it.  Without this, one query scans the whole
    #: catalogue in Python: measured at ~10 s per request over 156k records, which is a serving
    #: blocker rather than a slow path.  Scores are unchanged — a document with no query term
    #: scores 0 and was already excluded.
    _postings: dict[str, tuple[int, ...]]

    @classmethod
    def build(
        cls,
        records: Mapping[str, Any],
        *,
        fields: Sequence[str] = SEARCHABLE_FIELDS,
    ) -> CatalogueLexicalIndex:
        """Tokenise every record once and precompute document frequencies."""
        resolved = tuple(fields)
        documents: list[tuple[str, dict[str, int], int, str | None]] = []
        frequency: dict[str, int] = {}
        for parent_asin, record in records.items():
            text = " ".join(
                part for part in (field_text(record, name) for name in resolved) if part
            )
            tokens = tokenize(text)
            if not tokens:
                # A record with no searchable text is not indexed.  It stays reachable by
                # identity lookup, which is the honest way to reach it.
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            title = getattr(record, "title", None)
            documents.append(
                (parent_asin, counts, len(tokens), title if isinstance(title, str) else None)
            )
            for term in counts:
                frequency[term] = frequency.get(term, 0) + 1
        postings: dict[str, list[int]] = {}
        for position, (_asin, counts, _length, _title) in enumerate(documents):
            for term in counts:
                postings.setdefault(term, []).append(position)
        average = (
            sum(entry[2] for entry in documents) / len(documents) if documents else 0.0
        )
        return cls(
            fields=resolved,
            catalogue_size=len(records),
            indexed_documents=len(documents),
            _documents=tuple(documents),
            _document_frequency=frequency,
            _average_length=average,
            _postings={term: tuple(positions) for term, positions in postings.items()},
        )

    def scored(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        """Return up to ``limit`` ``(parent_asin, bm25)`` pairs, best first.

        A blank query, or a query whose terms match nothing, returns an empty list — an honest
        "no lexical match" rather than an arbitrary slice of the catalogue.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        terms = tuple(dict.fromkeys(tokenize(query)))
        if not terms or not self._documents:
            return []
        document_count = len(self._documents)
        # Inverted-index accumulation: visit only (term, document) pairs that actually match.
        # The obvious nest -- for each matching document, loop over every query term -- is
        # |docs| x |terms| and measured ~10 s per request once a 20-item history contributed
        # thousands of distinct terms.  The arithmetic per pair is identical, so the ranking is
        # unchanged; only the work scales with the matches instead of the product.
        totals: dict[int, float] = {}
        for term in terms:
            positions = self._postings.get(term)
            if not positions:
                continue
            idf = _idf(self._document_frequency.get(term, 0), document_count)
            if idf <= 0.0:
                continue
            for position in positions:
                _asin, counts, length, _title = self._documents[position]
                frequency = counts[term]
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * length / self._average_length
                )
                contribution = idf * (frequency * (BM25_K1 + 1.0)) / denominator
                totals[position] = totals.get(position, 0.0) + contribution
        scored: list[tuple[float, str]] = [
            (score, self._documents[position][0])
            for position, score in totals.items()
            if score > 0.0
        ]
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [(parent_asin, round(score, 6)) for score, parent_asin in scored[:limit]]

    def ranking(self, query: str, *, limit: int) -> list[str]:
        """Return the ranked catalogue identities for ``query``."""
        return [parent_asin for parent_asin, _score in self.scored(query, limit=limit)]
