"""Agent-facing Recommendation Tool (Milestone 7A).

A thin, framework-independent business layer above the accepted
:class:`~recommendation.inference.sasrec.SASRecInferenceEngine`::

    Future Agent -> RecommendationTool -> SASRecInferenceEngine -> SASRec model

The Tool is **not** a second recommender.  It owns only:

* the agent-facing input contract,
* trusted-context handling,
* business-level validation,
* calling the engine,
* normalizing the engine's output into a typed result,
* mapping engine failures onto Tool domain errors.

It never re-implements checkpoint loading, item mapping, history encoding, SASRec
scoring, seen-item masking, ranking or tie handling - all of that stays in the engine.

Trust boundary (mandatory)
--------------------------
The user history comes from *trusted application state*, passed as a separate
:class:`~recommendation.tools.schemas.RecommendationContext` argument::

    tool.run(request=RecommendationToolRequest(k=10),
             context=RecommendationContext(user_history=[...]))

:class:`RecommendationToolRequest` has no history field, so an LLM choosing the
request arguments cannot supply, invent or alter the history.  The Tool never fetches
history itself and never interprets natural language.

No network: the Tool calls the engine in-process.  It never talks to the FastAPI
service over HTTP, and this module imports neither FastAPI nor any agent framework.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from recommendation.inference import (
    InferenceError,
    RecommendationResult,
    RequestValidationError as EngineRequestValidationError,
    UnknownItemError,
)

from .errors import (
    InvalidRecommendationRequest,
    MissingUserHistory,
    RecommendationToolError,
    RecommendationUnavailable,
    UnknownHistoryItem,
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

#: Stable tool identity, framework-neutral.
TOOL_NAME = "recommend_products"
TOOL_DESCRIPTION = (
    "Generate SASRec recommendations from trusted chronological interaction history."
)
TOOL_VERSION = 1

#: Error text is deliberately generic so nothing internal can leak to a caller.
_UNAVAILABLE_MESSAGE = (
    "recommendations are temporarily unavailable; the recommendation engine could not "
    "complete the request"
)


@runtime_checkable
class RecommendationEngine(Protocol):
    """Minimal engine interface the Tool depends on.

    Structural typing keeps the Tool decoupled from the concrete engine class, so unit
    tests and future callers can supply any object that satisfies this shape.
    """

    def recommend(self, history_parent_asins: Any, k: int = DEFAULT_K) -> RecommendationResult:
        """Return recommendations for a chronological ``parent_asin`` history."""
        ...


def _validate_k(k: Any) -> int:
    """Validate ``k`` at the Tool boundary, raising a Tool domain error."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise InvalidRecommendationRequest(
            f"k must be an integer, got {type(k).__name__}"
        )
    if not MIN_K <= k <= MAX_K:
        raise InvalidRecommendationRequest(
            f"k must be between {MIN_K} and {MAX_K}, got {k}"
        )
    return k


class RecommendationTool:
    """Business-level recommendation tool wrapping a recommendation engine.

    Parameters
    ----------
    engine:
        Any object implementing :class:`RecommendationEngine` - normally the accepted
        :class:`~recommendation.inference.sasrec.SASRecInferenceEngine`.  The engine
        (and therefore the loaded model) is supplied once and reused for every call;
        the Tool never loads a checkpoint or mapping itself.

    Raises
    ------
    RecommendationUnavailable
        If ``engine`` does not satisfy the required interface.
    """

    def __init__(self, engine: RecommendationEngine) -> None:
        if engine is None or not isinstance(engine, RecommendationEngine):
            raise RecommendationUnavailable(
                "engine must provide a callable recommend(history, k=...) method"
            )
        self._engine = engine

    # -- metadata ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Stable tool name for future agent tool registration."""
        return TOOL_NAME

    @property
    def description(self) -> str:
        """Framework-neutral, human-readable purpose."""
        return TOOL_DESCRIPTION

    @property
    def version(self) -> int:
        """Tool contract version."""
        return TOOL_VERSION

    @property
    def engine(self) -> RecommendationEngine:
        """The reused engine instance (exposed for lifecycle inspection/tests)."""
        return self._engine

    def metadata(self) -> dict[str, Any]:
        """Return the tool's stable metadata block."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }

    # -- invocation -------------------------------------------------------- #

    def run(
        self,
        request: RecommendationToolRequest | None = None,
        context: RecommendationContext | None = None,
    ) -> RecommendationToolResult:
        """Produce structured recommendations.

        ``request`` carries model-facing arguments (only ``k``); ``context`` carries the
        **trusted** user history.  They are separate arguments by design so orchestration
        can inject history from application state rather than from model output.

        Raises
        ------
        InvalidRecommendationRequest
            Malformed request arguments (for example an out-of-range ``k``).
        MissingUserHistory
            No trusted history, or an empty one.  There is no fallback.
        UnknownHistoryItem
            The trusted history references an item outside the served catalog.
        RecommendationUnavailable
            The engine failed to produce a result.
        """
        request = request if request is not None else RecommendationToolRequest()
        if not isinstance(request, RecommendationToolRequest):
            raise InvalidRecommendationRequest(
                f"request must be a RecommendationToolRequest, got {type(request).__name__}"
            )
        k = _validate_k(request.k)
        history = self._trusted_history(context)

        try:
            result = self._engine.recommend(list(history), k=k)
        except UnknownItemError as exc:
            # Strict policy: never drop, coerce to PAD, or continue with partial history.
            raise UnknownHistoryItem(str(exc)) from exc
        except EngineRequestValidationError as exc:
            raise InvalidRecommendationRequest(str(exc)) from exc
        except InferenceError as exc:
            raise RecommendationUnavailable(_UNAVAILABLE_MESSAGE) from exc
        except RecommendationToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak framework internals
            raise RecommendationUnavailable(_UNAVAILABLE_MESSAGE) from exc

        return self._to_result(result)

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _trusted_history(context: RecommendationContext | None) -> tuple[str, ...]:
        """Extract and validate the trusted history from the caller's context."""
        if context is None:
            raise MissingUserHistory(
                "trusted user history is required; supply RecommendationContext("
                "user_history=[...]) from application state"
            )
        if not isinstance(context, RecommendationContext):
            raise MissingUserHistory(
                f"context must be a RecommendationContext, got {type(context).__name__}"
            )
        history = context.user_history
        if not history:
            raise MissingUserHistory(
                "trusted user history is empty; SASRec requires interaction history "
                "and no popularity or random fallback exists"
            )
        # The schema already enforces non-empty strings; keep a defensive check so the
        # invariant holds even if a fake/duck-typed context is passed.
        for entry in history:
            if not isinstance(entry, str) or not entry.strip():
                raise MissingUserHistory(
                    "trusted user history entries must be non-empty parent_asin strings"
                )
        return tuple(history)

    @staticmethod
    def _to_result(result: RecommendationResult) -> RecommendationToolResult:
        """Normalize the engine result into the Tool result, preserving values exactly.

        No sorting, filtering, re-ranking or score transformation happens here: ranks,
        item ids, ``parent_asin`` values and raw scores are copied through unchanged, as
        is candidate-exhaustion behaviour.
        """
        recommendations = [
            ToolRecommendation(
                rank=item.rank,
                parent_asin=item.parent_asin,
                item_id=item.item_id,
                score=item.score,
            )
            for item in result.recommendations
        ]
        return RecommendationToolResult(
            recommendations=recommendations,
            requested_k=result.requested_k,
            returned_k=len(recommendations),
            history_length=result.history_length,
            effective_history_length=result.effective_history_length,
            history_truncated=result.history_truncated,
            eligible_candidates=result.eligible_candidates,
            timings_ms=dict(result.timings_ms),
        )


__all__ = [
    "RecommendationEngine",
    "RecommendationTool",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "TOOL_VERSION",
]
