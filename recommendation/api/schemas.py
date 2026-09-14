"""Pydantic request/response schemas for the recommendation HTTP API.

These schemas define the wire contract only.  They perform structural validation
(types, length bounds); catalog-aware validation such as "is this ``parent_asin`` in
the served item mapping?" belongs to the inference engine, which owns the mapping.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

#: Minimum / maximum number of recommendations a client may request.
MIN_K = 1
MAX_K = 100

#: A non-empty ``parent_asin`` string.
NonEmptyAsin = Annotated[str, Field(min_length=1)]


class RecommendRequest(BaseModel):
    """Body of ``POST /v1/recommend``."""

    model_config = ConfigDict(extra="forbid")

    history: Annotated[list[NonEmptyAsin], Field(min_length=1)] = Field(
        ...,
        description=(
            "Chronological sequence of Amazon parent_asin values the user has "
            "interacted with. Duplicates are allowed (repeated interactions are real "
            "interactions). Only the newest max_seq_len items reach the model, but "
            "every supplied item is excluded from the recommendations."
        ),
        examples=[["B00EXAMPLE1", "B00EXAMPLE2"]],
    )
    k: Annotated[int, Field(ge=MIN_K, le=MAX_K)] = Field(
        default=10,
        description=f"Number of recommendations to return, {MIN_K}..{MAX_K}.",
    )


class RecommendationItem(BaseModel):
    """One recommended item."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., description="1-based position in the ranking.")
    item_id: int = Field(
        ...,
        description="Internal integer model item id, in 1..num_items. PAD (0) is never returned.",
    )
    parent_asin: str = Field(..., description="External Amazon parent_asin identity.")
    score: float = Field(
        ...,
        description=(
            "Raw SASRec model score. This is NOT a probability, confidence, or "
            "purchase likelihood; it is only meaningful for ordering candidates."
        ),
    )


class RecommendResponse(BaseModel):
    """Body of a successful ``POST /v1/recommend``."""

    model_config = ConfigDict(extra="forbid")

    recommendations: list[RecommendationItem]
    requested_k: int
    returned_k: int = Field(
        ...,
        description=(
            "How many recommendations were returned; min(requested_k, eligible "
            "candidates). 0 when no unseen candidate remains."
        ),
    )
    history_length: int = Field(..., description="Number of items the caller supplied.")
    effective_history_length: int = Field(
        ..., description="How many of those items reached the model window."
    )
    history_truncated: bool = Field(
        ..., description="True when the supplied history exceeded max_seq_len."
    )
    eligible_candidates: int = Field(
        ..., description="Catalog size minus the supplied history items."
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict, description="Server-side scoring/ranking timings in milliseconds."
    )


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="'ok' only when the model is loaded and usable.")
    model_loaded: bool
    device: str | None = None
    detail: str | None = Field(
        default=None,
        description="Present when the process is running but the model is not ready.",
    )


class ModelInfoResponse(BaseModel):
    """Body of ``GET /v1/model``."""

    model_config = ConfigDict(extra="forbid")

    model_type: str
    num_items: int
    max_seq_len: int
    hidden_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    device: str
    checkpoint_sha256: str
    parameter_count: int = Field(
        ..., description="Total parameters in the served model (frozen for inference)."
    )
    model_parameters_frozen: bool = Field(
        ...,
        description=(
            "True when every served parameter has requires_grad=False; serving never "
            "creates gradients."
        ),
    )
    provenance: dict[str, object] = Field(
        ...,
        description=(
            "Separates the Git state recorded by the formal training run from the "
            "post-run accepted source checkpoint."
        ),
    )


class ErrorResponse(BaseModel):
    """Structured error body returned for client errors."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")


__all__ = [
    "MAX_K",
    "MIN_K",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "RecommendRequest",
    "RecommendResponse",
    "RecommendationItem",
]
