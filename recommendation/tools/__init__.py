"""Agent-facing tools (Milestone 7A).

Application/business orchestration layer above inference.  The Recommendation Tool
wraps the accepted :class:`~recommendation.inference.sasrec.SASRecInferenceEngine` and
exposes a small, framework-neutral contract that future agent orchestration can call.

This package deliberately contains:

* no LangGraph or LangChain,
* no LLM or prompt logic,
* no HTTP client and no FastAPI dependency,
* no memory, user database or planner.

The layer boundary is::

    Future Agent -> RecommendationTool -> SASRecInferenceEngine -> SASRec model
"""

from __future__ import annotations

from .errors import (
    InvalidRecommendationRequest,
    MissingUserHistory,
    RecommendationToolError,
    RecommendationUnavailable,
    UnknownHistoryItem,
)
from .recommendation import (
    TOOL_DESCRIPTION,
    TOOL_NAME,
    TOOL_VERSION,
    RecommendationEngine,
    RecommendationTool,
)
from .schemas import (
    DEFAULT_K,
    MAX_K,
    MIN_K,
    RecommendationContext,
    RecommendationToolRequest,
    RecommendationToolResult,
    ToolRecommendation,
)

__all__ = [
    "DEFAULT_K",
    "MAX_K",
    "MIN_K",
    "RecommendationContext",
    "RecommendationEngine",
    "RecommendationTool",
    "RecommendationToolError",
    "RecommendationToolRequest",
    "RecommendationToolResult",
    "ToolRecommendation",
    "InvalidRecommendationRequest",
    "MissingUserHistory",
    "RecommendationUnavailable",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "TOOL_VERSION",
    "UnknownHistoryItem",
]

__version__ = "1.0.0"
