"""Candidate enrichment: attach candidate-scoped evidence to a Tool result (Milestone 8-B).

This is the boundary the Agent calls.  It performs exactly four steps:

1. read the accepted M7A :class:`~recommendation.tools.schemas.RecommendationToolResult`;
2. look up metadata for **the candidate identities in that result only**;
3. retrieve evidence for each candidate, restricted to that candidate's metadata;
4. return an :class:`~recommendation.rag.schemas.EnrichmentResult` whose candidate
   sequence is byte-for-byte the Tool's sequence.

Hard invariants
---------------
* **Candidate universe isolation.**  The only identifiers that ever reach the
  metadata index or the retriever are the ones in the Tool result.  There is no
  code path that can add a catalogue item, and the retriever has no catalogue handle
  of its own.
* **Candidate order preservation.**  Each :class:`EnrichedRecommendation` embeds the
  original :class:`~recommendation.tools.schemas.ToolRecommendation` object
  unchanged, so rank, ``parent_asin``, ``item_id`` and raw SASRec ``score`` cannot
  drift.  No product is compared with another product and nothing is reordered.
* **No history leakage.**  Enrichment receives a Tool result, candidate identities
  and an untrusted query string.  It is never given the trusted interaction history
  and has no field in which to store it.
* **Missing metadata is explicit.**  A candidate without metadata stays in the list,
  unchanged, with ``metadata_status="missing"`` and a documented
  ``fallback_reason``.  It is never dropped, replaced or invented.
* **No network.**  Everything is in-process over a preloaded index.

This module deliberately does **not** render user-facing text; rendering is the
Agent's job (see :mod:`recommendation.agent.graph`), so the same structured evidence
can later be handed to a different renderer or an LLM without changing this contract.
"""

from __future__ import annotations

import time
from typing import Sequence

from ..catalog.schemas import MetadataLookup, MissingMetadata, ProductMetadata
from ..tools.schemas import RecommendationToolResult
from .retrieval import retrieve_evidence
from .schemas import (
    CandidateEvidence,
    EnrichedRecommendation,
    EnrichmentResult,
    ProductEvidence,
)

__all__ = [
    "ProductEnricher",
    "enrich_candidates",
]


def enrich_candidates(
    result: RecommendationToolResult,
    metadata: MetadataLookup,
    *,
    query: str = "",
) -> EnrichmentResult:
    """Attach candidate-scoped evidence to ``result`` for the untrusted ``query``.

    Parameters
    ----------
    result:
        The accepted Tool result.  Treated as read-only: it is never mutated, and no
        candidate is added, removed or reordered.
    metadata:
        Any object satisfying
        :class:`~recommendation.catalog.schemas.MetadataLookup`.  Injected, so the
        Agent never constructs a hidden global metadata store and tests can supply a
        small fake.
    query:
        Untrusted natural-language text used only to select evidence among the
        current candidates' metadata.  A blank query is allowed and yields the
        documented factual fallback.

    Returns
    -------
    EnrichmentResult
        Positionally aligned with ``result.recommendations``.
    """
    if not isinstance(result, RecommendationToolResult):
        raise TypeError(
            f"result must be a RecommendationToolResult, got {type(result).__name__}"
        )

    started = time.perf_counter()
    # Only the candidates the Tool produced may be looked up.
    candidate_asins: tuple[str, ...] = tuple(
        item.parent_asin for item in result.recommendations
    )
    records = metadata.lookup_many(candidate_asins)
    diagnostics: dict[str, object] = {}
    evidence_per_candidate = retrieve_evidence(query, records, diagnostics=diagnostics)
    fallback_used = bool(diagnostics.get("fallback_used"))

    items: list[EnrichedRecommendation] = []
    found = missing = evidence_count = 0
    for item, record, evidence in zip(
        result.recommendations, records, evidence_per_candidate
    ):
        is_missing = isinstance(record, MissingMetadata)
        if is_missing:
            missing += 1
        else:
            found += 1
        evidence_count += len(evidence)
        items.append(
            EnrichedRecommendation(
                # Copied through unchanged: rank / parent_asin / score cannot drift.
                recommendation=item,
                metadata_status="missing" if is_missing else "found",
                metadata=None if is_missing else record,
                evidence=evidence,
                fallback_reason=_fallback_reason(is_missing, record, evidence, fallback_used),
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return EnrichmentResult(
        items=tuple(items),
        requested_k=result.requested_k,
        returned_k=result.returned_k,
        metadata_found=found,
        metadata_missing=missing,
        evidence_count=evidence_count,
        query_used=query if query and query.strip() else None,
        timings_ms={"enrich": round(elapsed_ms, 4)},
    )


def _fallback_reason(
    is_missing: bool,
    record: ProductMetadata | MissingMetadata,
    evidence: Sequence[ProductEvidence],
    fallback_used: bool,
) -> str | None:
    """Explain why a candidate produced no relevance-scored evidence, or ``None``.

    ``fallback_used`` is reported by the retriever rather than inferred from the
    scores, because a genuinely matching fragment can legitimately score ``0.0`` when
    a term carries no information (BM25 IDF floors at zero).  Inferring would
    mislabel a real match as "irrelevant".
    """
    if is_missing:
        return "no_metadata"
    assert isinstance(record, ProductMetadata)  # narrowed by the caller's branch
    if not record.has_searchable_text:
        return "no_searchable_text"
    if fallback_used:
        return "no_lexical_match"
    return None


class ProductEnricher:
    """Reusable candidate enricher bound to a preloaded metadata index.

    The index is loaded once and reused for every call, so a graph invocation never
    reparses the metadata artifact.
    """

    def __init__(self, metadata: MetadataLookup) -> None:
        for method in ("lookup", "lookup_many"):
            if not callable(getattr(metadata, method, None)):
                raise TypeError(
                    f"metadata must provide a callable {method}() method"
                )
        self._metadata = metadata

    @property
    def metadata(self) -> MetadataLookup:
        """The injected metadata lookup (exposed for inspection/tests)."""
        return self._metadata

    def enrich(self, result: RecommendationToolResult, query: str = "") -> EnrichmentResult:
        """Attach candidate-scoped evidence to ``result``."""
        return enrich_candidates(result, self._metadata, query=query)

    def evidence_for(self, parent_asin: str, query: str = "") -> tuple[ProductEvidence, ...]:
        """Retrieve evidence for a single candidate identity.

        Convenience for diagnostics; it uses the same candidate-scoped retriever, so
        it can never reach beyond the supplied identity.
        """
        records = self._metadata.lookup_many((parent_asin,))
        return retrieve_evidence(query, records)[0]

    def candidate_evidence(
        self, result: RecommendationToolResult, query: str = ""
    ) -> tuple[CandidateEvidence, ...]:
        """Return per-candidate evidence records aligned with the Tool result."""
        enrichment = self.enrich(result, query)
        return tuple(
            CandidateEvidence(
                parent_asin=item.parent_asin,
                rank=item.rank,
                status="evidence" if item.evidence else "none",
                metadata_status=item.metadata_status,
                evidence=item.evidence,
                fallback_reason=item.fallback_reason,
            )
            for item in enrichment.items
        )
