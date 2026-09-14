"""Tool-level domain errors for the Agent-facing Recommendation Tool.

The Tool is the boundary between future agent orchestration and the SASRec inference
engine, so it must not leak lower-layer exceptions.  Each category below maps a
deliberate set of engine failures onto a stable, client-safe meaning.

Nothing here ever carries filesystem paths, checkpoint internals, stack traces or
environment details.  Messages are written to be safe to surface to a caller.
"""

from __future__ import annotations


class RecommendationToolError(Exception):
    """Base class for every Tool-level domain error.

    Catching this type is sufficient to handle any failure the Tool reports.
    """

    #: Stable machine-readable code for programmatic handling.
    code: str = "recommendation_tool_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serialisable, client-safe error body."""
        return {"error": self.code, "detail": self.message}


class InvalidRecommendationRequest(RecommendationToolError):
    """The model-facing request arguments are malformed (for example an invalid ``k``)."""

    code = "invalid_request"


class MissingUserHistory(RecommendationToolError):
    """No trusted user history was supplied.

    SASRec is a sequential recommender and requires interaction history; the Tool has
    no popularity or random fallback, and cold-start handling is out of scope for
    Milestone 7A.
    """

    code = "missing_user_history"


class UnknownHistoryItem(RecommendationToolError):
    """The trusted history references a ``parent_asin`` outside the served catalog.

    The Tool fails rather than dropping the item, substituting PAD, or continuing with
    a partial history.
    """

    code = "unknown_history_item"


class RecommendationUnavailable(RecommendationToolError):
    """The recommendation engine could not produce a result.

    Covers an engine that failed to load, a model/model-server failure, or any
    unexpected internal error.  The public message is intentionally generic.
    """

    code = "recommendation_unavailable"


__all__ = [
    "InvalidRecommendationRequest",
    "MissingUserHistory",
    "RecommendationToolError",
    "RecommendationUnavailable",
    "UnknownHistoryItem",
]
