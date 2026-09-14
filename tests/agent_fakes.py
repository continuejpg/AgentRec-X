"""Reusable offline test doubles for the Milestone 7B agent-graph tests.

Everything here is dependency-free: no model checkpoint, no catalog artifact, no
GPU, no network and no LLM provider SDK.  The Recommendation Tool is driven through
its public constructor with a duck-typed engine (the same seam Milestone 7A already
exercises), and the decision model is a scripted stub, so the graph is tested as
orchestration rather than as a model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.tools import (  # noqa: E402
    RecommendationToolResult,
    ToolRecommendation,
)

#: Trusted histories used throughout the graph tests.
HISTORY: tuple[str, ...] = ("B000000001", "B000000002", "B000000003")

#: History containing a deliberate duplicate, to prove order/duplicates survive.
HISTORY_WITH_DUPLICATE: tuple[str, ...] = ("B000000001", "B000000002", "B000000001")


class RecordingEngine:
    """A fake recommendation engine that records exactly what the Tool passed down.

    Implements the structural ``recommend(history_parent_asins, k=...)`` interface
    the Tool requires, so no real engine, checkpoint or mapping is loaded.
    """

    def __init__(
        self,
        *,
        catalog_size: int = 32,
        available: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.catalog_size = catalog_size
        #: ``None`` means "return min(k, catalog minus seen)"; an int caps the
        #: result to model candidate exhaustion explicitly.
        self.available = available
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """How many times the engine was invoked."""
        return len(self.calls)

    @property
    def last_history(self) -> list[str] | None:
        """The exact history list the engine received on the last call."""
        return self.calls[-1]["history"] if self.calls else None

    @property
    def last_k(self) -> int | None:
        """The exact ``k`` the engine received on the last call."""
        return self.calls[-1]["k"] if self.calls else None

    def recommend(self, history_parent_asins: Any, k: int = 10) -> RecommendationToolResult:
        """Record the call and return a deterministic structured result."""
        history = list(history_parent_asins)
        self.calls.append({"history": history, "k": k})

        if self.error is not None:
            raise self.error

        eligible = max(self.catalog_size - len(history), 0)
        count = eligible if self.available is None else min(self.available, k, eligible)
        count = min(count, k)

        recommendations = [
            ToolRecommendation(
                rank=rank,
                # Deliberately distinct from every history value so leaks are visible.
                parent_asin=f"B9{rank:08d}",
                item_id=1000 + rank,
                score=round(1.0 / rank, 6),
            )
            for rank in range(1, count + 1)
        ]
        return RecommendationToolResult(
            recommendations=recommendations,
            requested_k=k,
            returned_k=count,
            history_length=len(history),
            effective_history_length=len(history),
            history_truncated=False,
            eligible_candidates=eligible,
            timings_ms={},
        )


class ScriptedDecisionModel:
    """A decision model that returns a fixed payload and records what it was shown.

    ``payload`` is returned verbatim (it may be an ``AgentDecision``, a dict or a
    JSON string) so the graph's validation layer is genuinely exercised.
    """

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[Any, ...]] = []

    @property
    def call_count(self) -> int:
        """How many times ``decide`` was called."""
        return len(self.calls)

    @property
    def last_messages(self) -> tuple[Any, ...] | None:
        """The exact message tuple handed to the decision model, if any."""
        return self.calls[-1] if self.calls else None

    @property
    def last_prompt_text(self) -> str:
        """All message content concatenated, for leak assertions."""
        messages = self.last_messages or ()
        return "\n".join(getattr(m, "content", "") for m in messages)

    def decide(self, messages: Sequence[Any]) -> Any:
        """Record the messages and return the scripted payload."""
        self.calls.append(tuple(messages))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


__all__ = [
    "HISTORY",
    "HISTORY_WITH_DUPLICATE",
    "RecordingEngine",
    "ScriptedDecisionModel",
]
