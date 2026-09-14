"""Deterministic preference-aware reranking (Milestone 10B).

Reorders the candidates of an accepted M10A evidence report under an explicit,
auditable lexicographic policy: fewer explicit preference violations first, then more
explicit matches, then the original SASRec rank, then ``item_id`` as a final
deterministic fallback.

Guarantees:

* **the same candidates come out that went in** -- never added, removed, replaced or
  filtered, even when a candidate violates a preference;
* **the raw SASRec score is never modified** -- no normalisation, no penalty, no
  combined score, no weights or coefficients;
* **UNKNOWN evidence is neutral** -- counted, never rewarded or penalised, so metadata
  sparsity cannot change ranking;
* the M10A evidence records and the input report are passed through untouched;
* the reranker reads only the evidence report: no memory store, no retriever, no model,
  no RecommendationTool, no HTTP, no LLM.

See ``README.md`` for the policy rationale, the reason codes and the M10B/M10C boundary.
"""

from __future__ import annotations

from .reranker import (
    DEFAULT_DIAGNOSTIC_K,
    PreferenceReranker,
    rerank_candidates,
    sort_key_for,
)
from .schemas import (
    RERANK_SORT_KEY_DOC,
    CandidateEvaluation,
    RerankReason,
    RerankedCandidate,
    RerankingDiagnostics,
    RerankingError,
    RerankingReport,
    TopKAdherence,
)

__all__ = [
    "DEFAULT_DIAGNOSTIC_K",
    "RERANK_SORT_KEY_DOC",
    "CandidateEvaluation",
    "PreferenceReranker",
    "RerankReason",
    "RerankedCandidate",
    "RerankingDiagnostics",
    "RerankingError",
    "RerankingReport",
    "TopKAdherence",
    "rerank_candidates",
    "sort_key_for",
]

__version__ = "0.1.0"
