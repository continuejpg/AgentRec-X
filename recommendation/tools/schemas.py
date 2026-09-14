"""Schemas for the Agent-facing Recommendation Tool.

The central design decision lives here: the **model-facing request** and the
**trusted user history** are two different types that cannot be merged.

* :class:`RecommendationToolRequest` holds only what an agent/LLM may choose - ``k``.
  It deliberately has no history field, so an LLM cannot supply one.
* :class:`RecommendationContext` holds the trusted chronological interaction history,
  supplied by the host application's state, and is passed as a separate argument.

Keeping them apart is what enforces the anti-hallucination boundary structurally
rather than by convention.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

#: Accepted range for the number of recommendations.
MIN_K = 1
MAX_K = 100

#: Default number of recommendations.
DEFAULT_K = 10

#: A non-empty ``parent_asin`` string: whitespace is stripped and blanks are rejected.
NonEmptyAsin = Annotated[str, Field(min_length=1)]


def _normalise_asin(value: object) -> str:
    """Strip surrounding whitespace and reject blank/non-string values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty parent_asin string")
    return value.strip()


def _normalise_history(value: object, handler) -> object:
    """Apply :func:`_normalise_asin` to every entry of a history sequence.

    Applying the normaliser here (rather than inside the ``tuple[...]`` item type)
    means each entry is stripped individually instead of the whole tuple being passed
    to the item validator.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("history must be a sequence of parent_asin strings")
    return tuple(_normalise_asin(item) for item in value)


class RecommendationToolRequest(BaseModel):
    """Model-facing request arguments.

    Intentionally minimal: ``k`` is the only parameter SASRec can act on.  Fields such
    as ``brand``, ``color``, ``category``, ``budget``, ``query``, ``intent``,
    ``constraints`` or ``natural_language_request`` are **not** accepted, because the
    sequential recommender cannot consume them; they belong to later Agent/RAG/Critic
    milestones.

    There is no ``history`` field: user history is trusted application state and is
    supplied through :class:`RecommendationContext`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ``strict=True`` matters: without it pydantic coerces "10" -> 10 and True -> 1,
    # which would let malformed agent output silently become a valid request.
    k: Annotated[int, Field(ge=MIN_K, le=MAX_K, strict=True)] = Field(
        default=DEFAULT_K,
        description=f"Number of recommendations to request, {MIN_K}..{MAX_K}.",
    )


class RecommendationContext(BaseModel):
    """Trusted application context for one recommendation call.

    ``user_history`` must come from trusted application state (a memory/profile store,
    a request-scoped session, a batch job), **never** from model-generated text.

    Requirements: non-empty, all entries non-empty ``parent_asin`` strings, and in
    chronological order.  Duplicates are allowed and are **not** deduplicated, because
    repeated interactions are real interactions.  The list is never reordered or
    truncated here: the engine owns model-window truncation, and the full history must
    stay available to the engine for seen-item masking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_history: Annotated[
        tuple[NonEmptyAsin, ...], Field(min_length=1), BeforeValidator(_normalise_history)
    ] = Field(
        ...,
        description=(
            "Trusted chronological sequence of Amazon parent_asin values the user has "
            "interacted with. Supplied by the host application, not by a model."
        ),
        examples=[("B00EXAMPLE1", "B00EXAMPLE2", "B00EXAMPLE3")],
    )


class ToolRecommendation(BaseModel):
    """One recommended item, preserving the engine's output exactly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(..., description="1-based rank, contiguous from 1.")
    parent_asin: str = Field(..., description="External Amazon parent_asin identity.")
    item_id: int = Field(..., description="Internal integer model item id in 1..num_items.")
    score: float = Field(
        ...,
        description=(
            "Raw SASRec model score, preserved from the engine. This is NOT a "
            "probability, confidence, relevance score, CTR, or conversion likelihood; "
            "it is only meaningful for ordering candidates."
        ),
    )


class RecommendationToolResult(BaseModel):
    """Structured Tool result.

    Contains structured data only - never natural-language text, explanations or
    product metadata.  Future agent layers turn this into user-facing language.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendations: list[ToolRecommendation]
    requested_k: int
    returned_k: int = Field(
        ...,
        description=(
            "min(requested_k, eligible candidates). May be 0 when no unseen candidate "
            "remains; that is a valid result, not an error."
        ),
    )
    history_length: int = Field(..., description="Length of the trusted supplied history.")
    effective_history_length: int = Field(
        ..., description="How many history items reached the model window."
    )
    history_truncated: bool = Field(
        ..., description="True when the trusted history exceeded the model window."
    )
    eligible_candidates: int = Field(
        default=0, description="Catalog size minus the supplied history items."
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Engine-side scoring/ranking timings. Diagnostic only.",
    )


__all__ = [
    "DEFAULT_K",
    "MAX_K",
    "MIN_K",
    "RecommendationContext",
    "RecommendationToolRequest",
    "RecommendationToolResult",
    "ToolRecommendation",
]
