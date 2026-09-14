"""Candidate-scoped product RAG (Milestone 8-B).

Turns the opaque SASRec candidates produced by the accepted Recommendation Tool into
grounded, attributable product context::

    trusted interaction history
            -> RecommendationTool (accepted M7A)
            -> ranked candidate parent_asins
            -> metadata lookup for THOSE candidates only (M8-A catalog layer)
            -> candidate-local lexical evidence retrieval
            -> grounded evidence for the Agent's final response

The direction that is explicitly **not** implemented::

    user query -> global retrieval -> new recommendation candidates

SASRec remains the only source of recommendation candidates.  This package cannot
add, remove or reorder them, and it never sees the trusted interaction history.

Scope rules:

* retrieval runs only over metadata of the current Tool candidates -- a catalogue
  item outside that set is unreachable by construction;
* evidence is verbatim, attributed to ``(parent_asin, field, provenance)``, and text
  from different products is never merged;
* missing metadata and missing searchable text are explicit outcomes, never
  fabricated;
* everything is offline, deterministic and dependency-free (a candidate-local BM25
  scorer), so the accepted M6 dependency closure is untouched;
* no product is ranked against another product and no retrieval score is combined
  with a SASRec score -- that would be reranking, which belongs to a later milestone.

See ``README.md`` for the retrieval algorithm, the query handling rules and the
deterministic tie-breaking.
"""

from __future__ import annotations

from .enrichment import ProductEnricher, enrich_candidates
from .retrieval import (
    BM25_B,
    BM25_K1,
    MAX_EVIDENCE_PER_CANDIDATE,
    MAX_EVIDENCE_PER_FIELD,
    build_documents,
    retrieve_evidence,
    tokenize,
)
from .schemas import (
    EVIDENCE_FIELDS,
    FALLBACK_REASONS,
    CandidateEvidence,
    EnrichedRecommendation,
    EnrichmentResult,
    ProductEvidence,
    RagQuery,
)

__all__ = [
    "BM25_B",
    "BM25_K1",
    "EVIDENCE_FIELDS",
    "FALLBACK_REASONS",
    "MAX_EVIDENCE_PER_CANDIDATE",
    "MAX_EVIDENCE_PER_FIELD",
    "CandidateEvidence",
    "EnrichedRecommendation",
    "EnrichmentResult",
    "ProductEnricher",
    "ProductEvidence",
    "RagQuery",
    "build_documents",
    "enrich_candidates",
    "retrieve_evidence",
    "tokenize",
]

__version__ = "0.1.0"
