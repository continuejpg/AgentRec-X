"""SASRec inference and deterministic serving-time ranking (Milestone 6).

Framework independent: this package owns checkpoint loading, item mapping, history
encoding, the model forward pass and production ranking.  The HTTP layer lives in
:mod:`recommendation.api` and must call these components rather than reimplementing
any of their semantics.

Serving ranking is deliberately **not** the evaluator's ranking: there is no
held-out target in production, so the candidate set is simply the catalog minus the
items the caller has already interacted with (PAD excluded).  See
:mod:`recommendation.inference.ranking` for the frozen tie rule.
"""

from __future__ import annotations

from .ranking import (
    PAD_ID,
    RankedItem,
    RankingError,
    eligible_candidate_count,
    rank_top_k,
    reference_rank_top_k,
    validate_k,
    validate_score_vector,
)
from .sasrec import (
    SUPPORTED_DEVICES,
    InferenceConfig,
    InferenceError,
    Recommendation,
    RecommendationResult,
    RequestValidationError,
    SASRecInferenceEngine,
    UnknownItemError,
    load_item_mapping,
    resolve_device,
)

__all__ = [
    "PAD_ID",
    "SUPPORTED_DEVICES",
    "InferenceConfig",
    "InferenceError",
    "RankedItem",
    "RankingError",
    "Recommendation",
    "RecommendationResult",
    "RequestValidationError",
    "SASRecInferenceEngine",
    "UnknownItemError",
    "eligible_candidate_count",
    "load_item_mapping",
    "rank_top_k",
    "reference_rank_top_k",
    "resolve_device",
    "validate_k",
    "validate_score_vector",
]

__version__ = "0.1.0"
