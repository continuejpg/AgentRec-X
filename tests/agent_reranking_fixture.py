"""Reusable offline fixtures for the Milestone 10D agent-reranking tests.

Everything here is deterministic and dependency-free: no checkpoint, no GPU, no
network, no LLM provider SDK and no database.  The candidate set, the metadata index
and the preference memory are all synthetic, but the objects flowing through the graph
are the **real** accepted types:

* candidates are real ``EnrichedRecommendation`` values produced by the real
  ``ProductEnricher`` over a real ``MetadataIndex``;
* evidence is produced by the real M10A ``PreferenceCandidateMatcher``;
* order comes from the real M10B ``PreferenceReranker``.

The counting wrappers exist only to observe call counts and identities, which is what
lets the tests assert "the matcher ran exactly once" and "the same reranker instance was
reused across turns" without inspecting private graph internals.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.metadata import normalize_product_record  # noqa: E402
from recommendation.catalog.schemas import ProductMetadata  # noqa: E402
from recommendation.memory import (  # noqa: E402
    InMemoryPreferenceStore,
    PreferenceMemoryService,
    RuleBasedPreferenceExtractor,
)
from recommendation.preference_matching import PreferenceCandidateMatcher  # noqa: E402
from recommendation.rag import ProductEnricher  # noqa: E402
from recommendation.reranking import PreferenceReranker  # noqa: E402
from recommendation.tools.schemas import (  # noqa: E402
    RecommendationToolResult,
    ToolRecommendation,
)

__all__ = [
    "CANDIDATE_ROWS",
    "INITIAL_ORDER",
    "QUERY",
    "CountingMatcher",
    "CountingReranker",
    "FixedEngine",
    "StaticMemory",
    "build_index",
    "build_metadata",
    "build_tool",
    "build_tool_result",
    "make_service",
    "rendered_order",
]


#: ``(parent_asin, item_id, sasrec_score, colour, title, store)``.
#:
#: The initial SASRec order is exactly this row order.  Each candidate carries a
#: distinctive title token so evidence-alignment assertions can prove that a candidate
#: is printed with **its own** metadata after the order changes.
CANDIDATE_ROWS: tuple[tuple[str, int, float, str, str, str | None], ...] = (
    ("cand-red", 101, 0.9000, "red", "RedWidget", None),
    ("cand-blue", 102, 0.8000, "blue", "BlueWidget", None),
    ("cand-black", 103, 0.7000, "black", "BlackWidget", "Acme"),
    ("cand-green", 104, 0.6000, "green", "GreenWidget", None),
)

#: The initial (upstream SASRec) candidate order.
INITIAL_ORDER: tuple[str, ...] = tuple(row[0] for row in CANDIDATE_ROWS)

#: Retrieval query used by the tests.  It contains each candidate's distinctive title
#: token so every candidate really has candidate-scoped evidence to print.
QUERY = "RedWidget BlueWidget BlackWidget GreenWidget"

#: Matches the ``<n>. <parent_asin> (`` line of a rendered candidate block.
_RENDERED_LINE = re.compile(r"^(\d+)\. (B[0-9A-Za-z]+|cand-[a-z]+) \(", re.MULTILINE)


def build_metadata(row: tuple[Any, ...]) -> ProductMetadata:
    """Normalise one candidate row into real M8 ``ProductMetadata``."""
    parent_asin, _item_id, _score, colour, title, store = row
    record: dict[str, Any] = {
        "parent_asin": parent_asin,
        "title": title,
        "details": {"Color": colour},
    }
    if store is not None:
        record["store"] = store
    return normalize_product_record(record)


def build_index(asins: Sequence[str] | None = None) -> Any:
    """Build an in-memory metadata index over the synthetic candidate rows."""
    from recommendation.catalog import MetadataIndex

    selected = set(asins) if asins is not None else None
    records = [
        build_metadata(row)
        for row in CANDIDATE_ROWS
        if selected is None or row[0] in selected
    ]
    return MetadataIndex.from_records(records)


def build_tool_result(rows: Sequence[tuple[Any, ...]] = CANDIDATE_ROWS) -> RecommendationToolResult:
    """Build the real Tool result for the candidate rows, in their given order."""
    return RecommendationToolResult(
        recommendations=tuple(
            ToolRecommendation(rank=position, parent_asin=row[0], item_id=row[1], score=row[2])
            for position, row in enumerate(rows, start=1)
        ),
        requested_k=len(rows),
        returned_k=len(rows),
        history_length=3,
        effective_history_length=3,
        history_truncated=False,
        eligible_candidates=1000,
        timings_ms={"scoring": 1.0, "ranking": 0.5},
    )


class FixedEngine:
    """A deterministic engine returning a fixed, caller-owned candidate list.

    Implements the structural ``recommend(history_parent_asins, k=...)`` interface the
    accepted Tool requires, so no checkpoint is loaded.  It records the exact history it
    was given, which lets a test prove the Tool's trusted-history boundary.
    """

    def __init__(
        self,
        rows: Sequence[tuple[Any, ...]] = CANDIDATE_ROWS,
        *,
        error: Exception | None = None,
    ) -> None:
        self._rows = tuple(rows)
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

    def recommend(self, history_parent_asins: Any, k: int = 10) -> RecommendationToolResult:
        """Record the call and return the fixed candidate list."""
        history = list(history_parent_asins)
        self.calls.append({"history": history, "k": k})
        if self.error is not None:
            raise self.error
        rows = self._rows[:k]
        return build_tool_result(rows)


def build_tool(engine: FixedEngine) -> Any:
    """Wrap an engine in the accepted, unmodified Recommendation Tool."""
    from recommendation.tools import RecommendationTool

    return RecommendationTool(engine)


class CountingMatcher:
    """The real M10A matcher plus call/identity observation.

    Delegates every call: the tests observe the accepted matcher's real output, never a
    reimplementation of matching.
    """

    def __init__(self, delegate: PreferenceCandidateMatcher | None = None) -> None:
        self._delegate = delegate or PreferenceCandidateMatcher()
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """How many times ``match`` was invoked."""
        return len(self.calls)

    @property
    def last_candidates(self) -> tuple[Any, ...] | None:
        """The exact candidate objects handed to the matcher on the last call."""
        return self.calls[-1]["candidates"] if self.calls else None

    @property
    def last_preferences(self) -> Any:
        """The exact preference argument handed to the matcher on the last call."""
        return self.calls[-1]["preferences"] if self.calls else None

    def match(self, *, candidates: Any, preferences: Any) -> Any:
        """Record the inputs and delegate to the accepted matcher."""
        self.calls.append({"candidates": tuple(candidates), "preferences": preferences})
        return self._delegate.match(candidates=candidates, preferences=preferences)


class CountingReranker:
    """The real M10B reranker plus call/identity observation."""

    def __init__(self, delegate: PreferenceReranker | None = None) -> None:
        self._delegate = delegate or PreferenceReranker()
        self.calls: list[Any] = []

    @property
    def call_count(self) -> int:
        """How many times ``rerank`` was invoked."""
        return len(self.calls)

    @property
    def last_report(self) -> Any:
        """The exact evidence report handed to the reranker on the last call."""
        return self.calls[-1] if self.calls else None

    def rerank(self, report: Any) -> Any:
        """Record the report and delegate to the accepted reranker."""
        self.calls.append(report)
        return self._delegate.rerank(report)


class StaticMemory:
    """A preference-memory service returning one fixed snapshot.

    Used where a test needs exact control over the active preferences.  It implements
    the accepted structural seam, so the graph cannot tell it apart from the real
    service; the real service is used for the ADD / REPLACE / REMOVE lifecycle tests.
    """

    def __init__(self, snapshot: Any) -> None:
        self._snapshot = snapshot
        self.load_calls: list[str] = []
        self.turn_calls: list[dict[str, Any]] = []

    @property
    def snapshot(self) -> Any:
        """The snapshot this service always returns."""
        return self._snapshot

    def get_active_preferences(self, user_key: str) -> Any:
        """Record the read and return the fixed snapshot."""
        self.load_calls.append(user_key)
        return self._snapshot

    def process_turn(self, **kwargs: Any) -> Any:
        """Record the write attempt; this double never persists anything."""
        self.turn_calls.append(kwargs)
        return None


def make_service() -> PreferenceMemoryService:
    """A real M9 service over a fresh in-memory store and the real rule-based extractor."""
    return PreferenceMemoryService(InMemoryPreferenceStore(), RuleBasedPreferenceExtractor())


def rendered_order(text: str) -> tuple[str, ...]:
    """Extract the rendered candidate sequence from a final response.

    Reads the ``<n>. <parent_asin> (`` lines in order, so an assertion compares exactly
    the sequence a human would read.
    """
    return tuple(match.group(2) for match in _RENDERED_LINE.finditer(text))
