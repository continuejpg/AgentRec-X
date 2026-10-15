"""Pydantic request/response schemas for the recommendation HTTP API.

These schemas define the wire contract only.  They perform structural validation
(types, length bounds); catalog-aware validation such as "is this ``parent_asin`` in
the served item mapping?" belongs to the inference engine, which owns the mapping.
"""

from __future__ import annotations

from typing import Any, Annotated

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


class TigerHealthBlock(BaseModel):
    """TIGER's lifecycle, as the HTTP surface reports it.

    ``ready`` is derived from the engine's own handshake, so a deployment whose worker is still
    loading reports ``state="loading"``/``ready=false`` and can never look available.
    """

    # ``ignore`` rather than ``forbid``: the engine's ``health()`` is the source of truth and may
    # grow a field; that must never turn /health into a 500.
    model_config = ConfigDict(extra="ignore")

    state: str = Field(..., description="disabled | not_started | loading | ready | failed")
    ready: bool = Field(..., description="True only once the worker handshake completed.")
    started: bool | None = None
    device: str | None = None
    assets_missing: list[str] = Field(default_factory=list)
    checkpoint_sha256: str | None = None
    sid_artifact_sha256: str | None = None
    retrieval_mode: str | None = Field(
        default=None, description="The mode the engine runs: CERTIFIED or APPROXIMATE."
    )
    error: str | None = None


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
    score_kind: str | None = Field(
        default=None,
        description=(
            "What the scores mean.  A fused result reports 'rrf_rank_fusion'; a single-model "
            "result reports that model's own kind.  Never a probability."
        ),
    )
    provenance: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-identity provenance: the fused rank and score, which sources contributed, each "
            "source's own rank and score kind, and - for TIGER candidates - the model's score, "
            "Semantic ID and retrieval mode.  Carried rather than dropped: a fused answer that "
            "cannot say where it came from is not an auditable one."
        ),
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
    # -- Which engine is actually serving, visible over HTTP (never only in a log). -- #
    requested_engine: str | None = Field(
        default=None, description="What AGENTRECX_SERVING_FUSION asked for."
    )
    active_engine: str | None = Field(
        default=None, description="'fusion_rrf' or 'sasrec': what is actually answering."
    )
    fallback_used: bool = Field(
        default=False, description="True when fusion was requested but SASRec-only is serving."
    )
    tiger: TigerHealthBlock | None = Field(
        default=None,
        description=(
            "TIGER's lifecycle when the serving profile includes it; absent (or state "
            "'disabled') when it does not."
        ),
    )
    fallback_reason: str | None = Field(
        default=None, description="Why the fallback happened. Present only when it did."
    )
    fusion_sources: list[str] | None = None
    rrf_k: int | None = None
    source_head: int | None = None


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
    # -- Serving V1 (P2-A).  Absent when a single-source engine serves. -- #
    active_engine: str | None = Field(
        default=None,
        description=(
            "'fusion_rrf' when several sources are fused, absent for the accepted single-source "
            "engine. Makes the engine that actually answered discoverable over HTTP."
        ),
    )
    serving_fusion: str | None = Field(default=None, description="'rrf', or absent.")
    fusion_sources: list[str] | None = Field(
        default=None, description="The fixed source set, in the order they are consulted."
    )
    source_score_kinds: dict[str, str] | None = Field(
        default=None, description="What each source's own score means; never a fused score."
    )
    rrf_k: int | None = Field(default=None, description="The fusion constant (60). Not a knob.")
    source_head_depth: int | None = Field(
        default=None,
        description=(
            "How deep each source's ranking is read before fusion (100, the deepest the accepted "
            "serving engines permit)."
        ),
    )
    score_kind: str | None = Field(
        default=None, description="'rrf_rank_fusion' when scores are fused."
    )
    lexical_documents: int | None = None
    catalogue_size: int | None = None
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
