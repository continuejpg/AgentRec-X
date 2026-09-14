"""Candidate-scoped lexical evidence retrieval (Milestone 8-B).

The retrieval universe is **exactly** the metadata of the current SASRec candidates.
Given candidates ``A B C``, evidence can only ever come from ``A``, ``B`` or ``C`` --
never from another catalogue item, however well it might match the query text::

    RecommendationTool result (candidates A B C)
            |
            v
    candidate-local documents (metadata fields of A, B, C only)
            |
            v
    BM25 scoring over those documents
            |
            v
    attributed evidence fragments

Why BM25 and not embeddings
---------------------------
The candidate set is small (``k <= 100`` under the accepted Tool contract), so a
candidate-local lexical scorer is sufficient, fully offline, deterministic, needs no
model download, no API key and no vector database, and keeps the accepted dependency
closure untouched.  The public interface is deliberately narrow so a dense embedding
backend could replace the scorer later without changing the Agent contract.

Determinism and tie-breaking
----------------------------
Document scoring is deterministic.  Ordering is by descending retrieval score, then
by candidate position in the supplied candidate order (i.e. the SASRec rank), then by
the fixed field order, then by text.  No wall-clock, hash order or float instability
enters the result.

Not a reranker
--------------
Evidence is ranked only *within* the allowed candidate scope, and the candidate list
is never reordered.  Retrieval scores are never combined with SASRec scores, and this
module has no access to recommendation scores at all beyond the candidate order it is
handed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ..catalog.schemas import MissingMetadata, ProductMetadata
from .schemas import EVIDENCE_FIELDS, ProductEvidence

__all__ = [
    "BM25_B",
    "BM25_K1",
    "MAX_EVIDENCE_PER_CANDIDATE",
    "MAX_EVIDENCE_PER_FIELD",
    "TokenizedDocument",
    "build_documents",
    "retrieve_evidence",
    "tokenize",
]

#: Standard BM25 parameters.
BM25_K1 = 1.2
BM25_B = 0.75

#: At most this many fragments are returned per candidate; a small, bounded context
#: window is all the Agent needs for one candidate.
MAX_EVIDENCE_PER_CANDIDATE = 4

#: At most this many fragments per (candidate, field), so a long description cannot
#: crowd out a title or a feature bullet.
MAX_EVIDENCE_PER_FIELD = 2

#: Provenance prefix for every fragment, matching the catalog source label.
_PROVENANCE_ROOT = "amazon_reviews_2023:meta_categories"

#: Tokenisation: lowercase alphanumeric runs.  No stemming and no stop-word list, so
#: behaviour is transparent and reproducible; punctuation cannot create terms.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return lowercase alphanumeric tokens for ``text``.

    Deterministic and language-agnostic: no stemming, no stop words, no model.  A
    non-string argument yields no tokens rather than raising, because the query is
    untrusted text.
    """
    if not isinstance(text, str) or not text:
        return ()
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))


@dataclass(frozen=True)
class TokenizedDocument:
    """One searchable fragment: ``(candidate index, field, text)`` plus its tokens."""

    candidate_index: int
    field_rank: int
    field: str
    text: str
    provenance: str
    detail_key: str | None
    tokens: tuple[str, ...]
    term_frequencies: Counter[str]

    @property
    def length(self) -> int:
        """Number of tokens in the fragment."""
        return len(self.tokens)


def _provenance_for(field: str, detail_key: str | None) -> str:
    """Build a traceable provenance string for one fragment."""
    if detail_key is not None:
        return f"{_PROVENANCE_ROOT}#details:{detail_key}"
    return f"{_PROVENANCE_ROOT}#{field}"


def _field_values(metadata: ProductMetadata, field: str) -> list[tuple[str, str | None]]:
    """Return ``(text, detail_key)`` pairs contributed by ``field``, in source order."""
    if field == "details":
        return [(value, key) for key, value in metadata.details]
    value = getattr(metadata, field, None)
    if value is None:
        return []
    if isinstance(value, tuple):
        return [(entry, None) for entry in value]
    if isinstance(value, str):
        return [(value, None)]
    return []


def build_documents(
    records: Sequence[ProductMetadata | MissingMetadata],
) -> tuple[TokenizedDocument, ...]:
    """Build the candidate-local document collection, in a fixed deterministic order.

    Documents are produced per candidate in submission order, then per field in
    :data:`~recommendation.rag.schemas.EVIDENCE_FIELDS` order, then per source value.
    This ordering is the tie-breaker for equal retrieval scores.
    """
    documents: list[TokenizedDocument] = []
    for candidate_index, record in enumerate(records):
        if isinstance(record, MissingMetadata):
            continue
        for field_rank, field in enumerate(EVIDENCE_FIELDS):
            for text, detail_key in _field_values(record, field):
                tokens = tokenize(text)
                if not tokens:
                    # A fragment with no tokens is not searchable and is not
                    # evidence; it is skipped rather than padded.
                    continue
                documents.append(
                    TokenizedDocument(
                        candidate_index=candidate_index,
                        field_rank=field_rank,
                        field=field,
                        text=text,
                        provenance=_provenance_for(field, detail_key),
                        detail_key=detail_key,
                        tokens=tokens,
                        term_frequencies=Counter(tokens),
                    )
                )
    return tuple(documents)


def _idf(document_frequency: int, document_count: int) -> float:
    """BM25 probabilistic IDF, floored at zero so a ubiquitous term contributes nothing."""
    return max(
        0.0,
        math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)),
    )


def retrieve_evidence(
    query: str,
    records: Sequence[ProductMetadata | MissingMetadata],
    *,
    max_per_candidate: int = MAX_EVIDENCE_PER_CANDIDATE,
    max_per_field: int = MAX_EVIDENCE_PER_FIELD,
    fallback_per_candidate: int = 1,
    diagnostics: dict[str, object] | None = None,
) -> tuple[tuple[ProductEvidence, ...], ...]:
    """Retrieve attributed evidence for each candidate, in candidate submission order.

    Returns one tuple of :class:`ProductEvidence` per input record, positionally
    aligned with ``records`` so a candidate can never lose its place.

    Behaviour by case:

    * **known metadata with a matching fragment** -> fragments ranked by BM25 score,
      bounded by ``max_per_candidate`` and ``max_per_field``;
    * **known metadata, no query token matches** -> a documented deterministic
      fallback: the first ``fallback_per_candidate`` fragments per candidate in field
      order, scored ``0.0``.  The fallback depends only on the candidate's own
      metadata, never on the query, so it cannot fabricate relevance.  When
      ``diagnostics`` is supplied it is updated with ``{"fallback_used": True}`` so a
      caller can label the result honestly;
    * **known metadata with no searchable text at all** -> an empty tuple (the caller
      records ``no_searchable_text``);
    * **missing metadata** -> an empty tuple (the caller records ``no_metadata``).

    The query only ever selects among documents belonging to ``records``.  There is
    no code path that could reach a catalogue item outside the supplied candidates.
    """
    record_list = list(records)
    documents = build_documents(record_list)
    if diagnostics is not None:
        diagnostics["document_count"] = len(documents)
        diagnostics["query_terms"] = tokenize(query)
        diagnostics["fallback_used"] = False
    if not documents:
        return tuple(() for _ in record_list)

    query_terms = tuple(dict.fromkeys(tokenize(query)))
    per_candidate: list[list[ProductEvidence]] = [[] for _ in record_list]

    def apply_fallback() -> None:
        if diagnostics is not None:
            diagnostics["fallback_used"] = True
        for document in documents:
            bucket = per_candidate[document.candidate_index]
            if len(bucket) >= min(fallback_per_candidate, max_per_candidate):
                continue
            bucket.append(
                ProductEvidence(
                    parent_asin=record_list[document.candidate_index].parent_asin,  # type: ignore[union-attr]
                    field=document.field,
                    text=document.text,
                    retrieval_score=0.0,
                    provenance=document.provenance,
                    detail_key=document.detail_key,
                )
            )

    if not query_terms:
        # Blank / whitespace-only query: no relevance signal exists, so return the
        # documented factual fallback rather than inventing one.
        apply_fallback()
        return tuple(tuple(bucket) for bucket in per_candidate)

    document_count = len(documents)
    document_frequency = Counter[str]()
    for document in documents:
        for term in set(document.term_frequencies):
            document_frequency[term] += 1
    average_length = sum(document.length for document in documents) / document_count

    scored: list[tuple[float, int, int, str, TokenizedDocument]] = []
    # Whether a candidate has *any* fragment containing a query term.  This is
    # tracked separately from the score: BM25 IDF is legitimately zero when a term
    # appears in every candidate-local document (it then carries no information), so
    # a fragment that really matches the query must still count as retrieved rather
    # than being reported as "no lexical match".
    matched_candidates: set[int] = set()
    for document in documents:
        score = 0.0
        contains_query_term = False
        for term in query_terms:
            frequency = document.term_frequencies.get(term, 0)
            if not frequency:
                continue
            contains_query_term = True
            idf = _idf(document_frequency.get(term, 0), document_count)
            if idf <= 0.0:
                continue
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * document.length / average_length
            )
            score += idf * (frequency * (BM25_K1 + 1.0)) / denominator
        if not contains_query_term:
            continue
        matched_candidates.add(document.candidate_index)
        scored.append(
            (
                # negated for ascending sort on score
                -round(score, 6),
                document.candidate_index,
                document.field_rank,
                document.text,
                document,
            )
        )

    if not matched_candidates:
        # No fragment contains any query term.  Return the documented fallback
        # subset (still only from these candidates) and let the caller mark it.
        apply_fallback()
        return tuple(tuple(bucket) for bucket in per_candidate)

    scored.sort(key=lambda entry: (entry[0], entry[1], entry[2], entry[3]))
    field_counts: Counter[tuple[int, int]] = Counter()
    for negated_score, candidate_index, field_rank, _text, document in scored:
        bucket = per_candidate[candidate_index]
        if len(bucket) >= max_per_candidate:
            continue
        if field_counts[(candidate_index, field_rank)] >= max_per_field:
            continue
        field_counts[(candidate_index, field_rank)] += 1
        bucket.append(
            ProductEvidence(
                parent_asin=record_list[candidate_index].parent_asin,  # type: ignore[union-attr]
                field=document.field,
                text=document.text,
                retrieval_score=round(-negated_score, 6),
                provenance=document.provenance,
                detail_key=document.detail_key,
            )
        )

    return tuple(tuple(bucket) for bucket in per_candidate)
