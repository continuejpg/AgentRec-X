"""Tests for the Agent-facing Recommendation Tool (Milestone 7A).

Three layers are covered:

* the request/context contract (validation, trusted-history separation),
* the engine integration contract, using a **fake** engine that records exactly what
  it was called with, so the Tool's behaviour is provable without loading a model,
* lifecycle/architecture guarantees (engine reuse, no HTTP, no agent frameworks).

Real-checkpoint integration lives in ``tests/test_recommendation_tool_integration.py``.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pydantic = pytest.importorskip("pydantic", reason="tool schemas require pydantic")

from recommendation.inference import (  # noqa: E402
    InferenceError,
    Recommendation,
    RecommendationResult,
    RequestValidationError as EngineRequestValidationError,
    UnknownItemError,
)
from recommendation.tools import (  # noqa: E402
    DEFAULT_K,
    MAX_K,
    MIN_K,
    InvalidRecommendationRequest,
    MissingUserHistory,
    RecommendationContext,
    RecommendationTool,
    RecommendationToolError,
    RecommendationToolRequest,
    RecommendationToolResult,
    RecommendationUnavailable,
    UnknownHistoryItem,
)
from recommendation.tools.recommendation import (  # noqa: E402
    TOOL_DESCRIPTION,
    TOOL_NAME,
    TOOL_VERSION,
    RecommendationEngine,
)

KNOWN = ("B000000001", "B000000002", "B000000003")


# --------------------------------------------------------------------------- #
# Fake engine: records calls, never loads a model
# --------------------------------------------------------------------------- #


class FakeEngine:
    """Deterministic stand-in for :class:`SASRecInferenceEngine`.

    Records the exact arguments it received so the Tool's pass-through behaviour can be
    asserted, and can be told to raise a specific engine error.
    """

    def __init__(self, *, items: int = 5, raise_error: BaseException | None = None) -> None:
        self.calls: list[tuple[list[str], int]] = []
        self.items = items
        self.raise_error = raise_error

    def recommend(self, history_parent_asins, k=DEFAULT_K) -> RecommendationResult:
        """Record the call and return a synthetic result (or raise a queued error)."""
        history = list(history_parent_asins)
        self.calls.append((history, k))
        if self.raise_error is not None:
            raise self.raise_error
        seen = set(history)
        recommendations = [
            Recommendation(
                rank=position,
                item_id=item_id,
                parent_asin=f"B{item_id:09d}",
                score=round(10.0 - item_id, 4),
            )
            for position, item_id in enumerate(range(1, self.items + 1), start=1)
            if f"B{item_id:09d}" not in seen
        ][:k]
        return RecommendationResult(
            recommendations=recommendations,
            requested_k=k,
            history_length=len(history),
            effective_history_length=min(len(history), 4),
            history_truncated=len(history) > 4,
            eligible_candidates=self.items - len(seen),
            timings_ms={"scoring": 1.0, "ranking": 2.0},
        )


class DuckEngine:
    """A structurally-compatible engine that is not a SASRecInferenceEngine."""

    def __init__(self) -> None:
        self.seen_kwargs: dict[str, object] = {}

    def recommend(self, history_parent_asins, k=DEFAULT_K):  # noqa: ANN001
        """Return an empty result and remember how it was called."""
        self.seen_kwargs = {"history": list(history_parent_asins), "k": k}
        return RecommendationResult(
            recommendations=[], requested_k=k, history_length=len(list(history_parent_asins)),
            effective_history_length=1, history_truncated=False, eligible_candidates=0,
        )


@pytest.fixture()
def tool() -> RecommendationTool:
    """A Tool wrapping a fake engine."""
    return RecommendationTool(FakeEngine())


def context(history=KNOWN) -> RecommendationContext:
    """Build a trusted context."""
    return RecommendationContext(user_history=tuple(history))


# --------------------------------------------------------------------------- #
# 1-12. Request / context contract
# --------------------------------------------------------------------------- #


def test_default_k_is_ten() -> None:
    """The request defaults to k=10."""
    assert RecommendationToolRequest().k == DEFAULT_K == 10


def test_k_bounds_accepted() -> None:
    """k=1 and k=100 are accepted."""
    assert RecommendationToolRequest(k=MIN_K).k == 1
    assert RecommendationToolRequest(k=MAX_K).k == 100


@pytest.mark.parametrize("bad", [0, -1, 101, 1000])
def test_k_out_of_range_rejected(bad: int) -> None:
    """k outside 1..100 is a schema error."""
    with pytest.raises(Exception):
        RecommendationToolRequest(k=bad)


@pytest.mark.parametrize("bad", ["10", 1.5, None, True])
def test_non_integer_k_rejected(bad) -> None:
    """Non-integer k is refused by the schema."""
    with pytest.raises(Exception):
        RecommendationToolRequest(k=bad)


def test_request_has_no_history_field() -> None:
    """The model-facing request cannot carry history - the anti-hallucination boundary."""
    assert "history" not in RecommendationToolRequest.model_fields
    assert "user_history" not in RecommendationToolRequest.model_fields
    # and extra fields are refused outright
    with pytest.raises(Exception):
        RecommendationToolRequest(k=5, history=list(KNOWN))


def test_request_rejects_premature_agent_fields() -> None:
    """Fields SASRec cannot consume are not part of the contract."""
    for field in ("budget", "brand", "color", "category", "query", "intent",
                  "constraints", "location", "natural_language_request"):
        with pytest.raises(Exception):
            RecommendationToolRequest(k=5, **{field: "x"})


def test_missing_trusted_history_rejected(tool: RecommendationTool) -> None:
    """Omitting the context is an explicit failure, not an empty result."""
    with pytest.raises(MissingUserHistory):
        tool.run(RecommendationToolRequest(k=5), None)
    with pytest.raises(MissingUserHistory):
        tool.run(RecommendationToolRequest(k=5))


def test_empty_trusted_history_rejected(tool: RecommendationTool) -> None:
    """An empty history fails clearly; there is no fallback.

    Two independent guards cover this:

    * the context schema refuses to build an empty history at all, so invalid input
      cannot even be represented;
    * the Tool's own guard rejects a non-context object outright.

    Both are asserted so the fail-fast property does not depend on a single layer.
    """
    with pytest.raises(Exception, match="user_history"):
        RecommendationContext(user_history=())

    with pytest.raises(MissingUserHistory, match="RecommendationContext"):
        tool.run(RecommendationToolRequest(k=5), type("Empty", (), {"user_history": ()})())


def test_empty_history_rejected_at_schema_level() -> None:
    """The context schema itself forbids an empty history."""
    with pytest.raises(Exception):
        RecommendationContext(user_history=())


@pytest.mark.parametrize("bad", ["", "   "])
def test_non_string_or_blank_history_items_rejected(bad: str) -> None:
    """Blank strings are refused by the schema."""
    with pytest.raises(Exception):
        RecommendationContext(user_history=(bad,))


def test_non_string_history_items_rejected() -> None:
    """Non-string entries are refused by the schema."""
    for bad in (123, None, 1.5):
        with pytest.raises(Exception):
            RecommendationContext(user_history=(bad,))  # type: ignore[arg-type]


def test_duplicate_history_is_preserved(tool: RecommendationTool) -> None:
    """Duplicates reach the engine unchanged (repeated interactions are real)."""
    history = ("B000000001", "B000000001", "B000000002")
    tool.run(RecommendationToolRequest(k=3), context(history))
    engine = tool.engine
    assert isinstance(engine, FakeEngine)
    passed_history, _ = engine.calls[0]
    assert passed_history == list(history)
    assert passed_history.count("B000000001") == 2


def test_chronological_order_is_preserved(tool: RecommendationTool) -> None:
    """The Tool never reorders the trusted history."""
    history = ("B000000003", "B000000001", "B000000002")  # deliberately not sorted
    tool.run(RecommendationToolRequest(k=2), context(history))
    engine = tool.engine
    assert isinstance(engine, FakeEngine)
    assert engine.calls[0][0] == list(history)


def test_source_history_is_not_mutated(tool: RecommendationTool) -> None:
    """Caller-owned input lists are never modified."""
    raw_history = ["B000000001", "B000000002", "B000000003"]
    snapshot = list(raw_history)
    ctx = RecommendationContext(user_history=tuple(raw_history))
    tool.run(RecommendationToolRequest(k=3), ctx)

    assert raw_history == snapshot
    assert list(ctx.user_history) == snapshot
    engine = tool.engine
    assert isinstance(engine, FakeEngine)
    assert engine.calls[0][0] == snapshot


# --------------------------------------------------------------------------- #
# 13-24. Engine integration contract
# --------------------------------------------------------------------------- #


def test_tool_calls_engine_exactly_once(tool: RecommendationTool) -> None:
    """One Tool call is exactly one engine call."""
    tool.run(RecommendationToolRequest(k=3), context())
    engine = tool.engine
    assert isinstance(engine, FakeEngine)
    assert len(engine.calls) == 1


def test_tool_passes_exact_history_and_k(tool: RecommendationTool) -> None:
    """The engine receives the exact history and k, with no extra arguments."""
    tool.run(RecommendationToolRequest(k=7), context())
    engine = tool.engine
    assert isinstance(engine, FakeEngine)
    history, k = engine.calls[0]
    assert history == list(KNOWN)
    assert k == 7


def test_tool_does_not_truncate_history_itself(tool: RecommendationTool) -> None:
    """The Tool passes the full history; the engine owns model-window truncation."""
    long_history = [f"B{index:09d}" for index in range(1, 21)]
    engine = FakeEngine(items=30)
    RecommendationTool(engine).run(
        RecommendationToolRequest(k=3), RecommendationContext(user_history=tuple(long_history))
    )
    passed_history, _ = engine.calls[0]
    assert len(passed_history) == 20, "the Tool must not truncate the history"

    result = RecommendationTool(engine).run(
        RecommendationToolRequest(k=3), RecommendationContext(user_history=tuple(long_history))
    )
    assert result.history_length == 20
    assert result.history_truncated is True  # reported by the engine, not applied by the Tool


def test_tool_does_not_deduplicate_history(tool: RecommendationTool) -> None:
    """Duplicates are passed through, not collapsed."""
    history = ("B000000001", "B000000002", "B000000001", "B000000001")
    tool.run(RecommendationToolRequest(k=2), context(history))
    engine = tool.engine
    assert isinstance(engine, FakeEngine)
    assert engine.calls[0][0] == list(history)


def test_tool_preserves_engine_ranks_and_ids(tool: RecommendationTool) -> None:
    """Ranks, item ids, parent_asins and scores are copied through unchanged."""
    engine = FakeEngine(items=5)
    result = RecommendationTool(engine).run(RecommendationToolRequest(k=3), context())
    # rebuild what the engine itself returned for comparison
    reference = engine.recommend(list(KNOWN), k=3)
    assert [(r.rank, r.item_id, r.parent_asin, r.score) for r in result.recommendations] == [
        (r.rank, r.item_id, r.parent_asin, r.score) for r in reference.recommendations
    ]
    assert result.requested_k == reference.requested_k
    assert result.returned_k == len(reference.recommendations)


def test_tool_does_not_rerank_engine_results() -> None:
    """The Tool must not sort or re-score: a deliberately unsorted engine result stays."""

    class OutOfOrderEngine:
        def recommend(self, history_parent_asins, k=DEFAULT_K):  # noqa: ANN001
            return RecommendationResult(
                recommendations=[
                    Recommendation(rank=1, item_id=9, parent_asin="B9", score=1.0),
                    Recommendation(rank=2, item_id=2, parent_asin="B2", score=9.0),
                ],
                requested_k=k, history_length=1, effective_history_length=1,
                history_truncated=False, eligible_candidates=10,
            )

    result = RecommendationTool(OutOfOrderEngine()).run(None, context())
    assert [(r.rank, r.item_id, r.score) for r in result.recommendations] == [
        (1, 9, 1.0),
        (2, 2, 9.0),
    ], "the Tool must preserve the engine's order and ranks exactly"


def test_candidate_exhaustion_is_preserved() -> None:
    """returned_k < requested_k passes through without error."""
    engine = FakeEngine(items=3)
    result = RecommendationTool(engine).run(RecommendationToolRequest(k=50), context())
    assert result.requested_k == 50
    assert result.returned_k == len(result.recommendations) <= 3


def test_zero_candidate_result_is_preserved() -> None:
    """An empty engine result stays an empty, valid Tool result."""
    result = RecommendationTool(DuckEngine()).run(RecommendationToolRequest(k=5), context())
    assert result.recommendations == []
    assert result.returned_k == 0
    assert isinstance(result, RecommendationToolResult)


def test_engine_timings_are_retained() -> None:
    """Safe engine metadata is kept when available."""
    result = RecommendationTool(FakeEngine()).run(RecommendationToolRequest(k=2), context())
    assert result.timings_ms == {"scoring": 1.0, "ranking": 2.0}
    assert result.eligible_candidates >= 0


def test_duck_typed_engine_is_accepted() -> None:
    """The Tool depends on structure, not on a concrete engine class."""
    engine = DuckEngine()
    RecommendationTool(engine).run(RecommendationToolRequest(k=1), context())
    assert engine.seen_kwargs["k"] == 1
    assert RecommendationEngine  # runtime-checkable protocol exists


# --------------------------------------------------------------------------- #
# 25-30. Error contract
# --------------------------------------------------------------------------- #


def test_unknown_item_maps_to_tool_error() -> None:
    """An engine unknown-item failure becomes UnknownHistoryItem."""
    tool = RecommendationTool(FakeEngine(raise_error=UnknownItemError("unknown parent_asin: 'BX'")))
    with pytest.raises(UnknownHistoryItem):
        tool.run(RecommendationToolRequest(k=5), context())


def test_engine_unavailable_maps_to_tool_error() -> None:
    """A failed engine becomes RecommendationUnavailable."""
    tool = RecommendationTool(FakeEngine(raise_error=InferenceError("model not loaded")))
    with pytest.raises(RecommendationUnavailable):
        tool.run(RecommendationToolRequest(k=5), context())


def test_invalid_request_from_engine_maps_to_tool_error() -> None:
    """An engine request-validation failure becomes InvalidRecommendationRequest."""
    tool = RecommendationTool(FakeEngine(raise_error=EngineRequestValidationError("bad history")))
    with pytest.raises(InvalidRecommendationRequest):
        tool.run(RecommendationToolRequest(k=5), context())


def test_unexpected_internal_error_does_not_leak_paths() -> None:
    """A raw internal exception is wrapped and its details are not exposed."""
    secret = "/root/AgentRec-X/runs/sasrec_canonical_2026/best.pt"
    tool = RecommendationTool(FakeEngine(raise_error=RuntimeError(f"cannot read {secret}")))
    with pytest.raises(RecommendationUnavailable) as excinfo:
        tool.run(RecommendationToolRequest(k=5), context())
    message = str(excinfo.value)
    assert secret not in message
    assert "/root/" not in message
    assert "Traceback" not in message
    assert "best.pt" not in message


def test_error_bodies_are_client_safe() -> None:
    """Every Tool error exposes a stable code and a sanitised detail."""
    tool = RecommendationTool(FakeEngine(raise_error=InferenceError("/root/secret/best.pt gone")))
    with pytest.raises(RecommendationUnavailable) as excinfo:
        tool.run(RecommendationToolRequest(k=5), context())
    body = excinfo.value.as_dict()
    assert body["error"] == "recommendation_unavailable"
    assert "/root/" not in body["detail"]


def test_tool_does_not_silently_recover_unknown_item() -> None:
    """Strict policy: no dropping, no PAD substitution, no partial history."""
    engine = FakeEngine(raise_error=UnknownItemError("unknown parent_asin: 'BZZZ'"))
    tool = RecommendationTool(engine)
    with pytest.raises(UnknownHistoryItem):
        tool.run(RecommendationToolRequest(k=5), context())
    # the Tool made exactly one attempt and did not retry with a reduced history
    assert len(engine.calls) == 1
    assert engine.calls[0][0] == list(KNOWN)


def test_tool_has_no_fallback_recommendation_path() -> None:
    """No popularity/random fallback: the Tool calls the engine and nothing else."""
    tool = RecommendationTool(FakeEngine(raise_error=InferenceError("down")))
    with pytest.raises(RecommendationUnavailable):
        tool.run(RecommendationToolRequest(k=5), context())
    for forbidden in ("popular", "popularity", "random", "trending", "fallback"):
        assert not hasattr(tool, forbidden)


def test_missing_history_is_not_an_empty_result() -> None:
    """Missing history raises rather than returning an empty recommendation list."""
    with pytest.raises(RecommendationToolError):
        RecommendationTool(FakeEngine()).run(RecommendationToolRequest(k=5), None)


def test_invalid_tool_request_object_is_rejected() -> None:
    """Passing a non-request object is a Tool domain error."""
    with pytest.raises(InvalidRecommendationRequest):
        RecommendationTool(FakeEngine()).run({"k": 5}, context())  # type: ignore[arg-type]


def test_invalid_context_object_is_rejected() -> None:
    """Passing a non-context object is a Tool domain error."""
    with pytest.raises(MissingUserHistory):
        RecommendationTool(FakeEngine()).run(
            RecommendationToolRequest(k=5), {"user_history": list(KNOWN)}  # type: ignore[arg-type]
        )


def test_tool_rejects_engine_without_recommend() -> None:
    """Construction validates the engine interface."""
    with pytest.raises(RecommendationUnavailable):
        RecommendationTool(object())  # type: ignore[arg-type]
    with pytest.raises(RecommendationUnavailable):
        RecommendationTool(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 31-37. Lifecycle / architecture
# --------------------------------------------------------------------------- #


def test_repeated_calls_reuse_the_same_engine() -> None:
    """The engine instance is reused, never rebuilt per call."""
    engine = FakeEngine()
    tool = RecommendationTool(engine)
    for _ in range(10):
        tool.run(RecommendationToolRequest(k=2), context())
    assert tool.engine is engine
    assert len(engine.calls) == 10


def test_no_checkpoint_load_per_call() -> None:
    """The Tool performs no checkpoint or mapping I/O itself."""
    source = inspect.getsource(sys.modules["recommendation.tools.recommendation"])
    for forbidden in ("torch.load", "sha256_file", "load_item_mapping", "json.load",
                      "SASRecInferenceEngine(", "build_model"):
        assert forbidden not in source, f"the Tool must not do I/O: found {forbidden}"


def test_core_tool_module_has_no_http_or_framework_imports() -> None:
    """No HTTP client, no FastAPI, no agent framework in the core Tool."""
    module = sys.modules["recommendation.tools.recommendation"]
    source = inspect.getsource(module)
    forbidden_imports = (
        "import fastapi", "from fastapi", "import starlette", "from starlette",
        "import httpx", "import requests", "import urllib", "import http.client",
        "import socket", "import aiohttp", "import ssl", "uvicorn",
        "import langgraph", "from langgraph", "import langchain", "from langchain",
        "import openai", "import anthropic",
    )
    for forbidden in forbidden_imports:
        assert forbidden not in source, f"forbidden dependency in core Tool: {forbidden}"


def test_importing_tools_does_not_import_http_or_framework_modules() -> None:
    """Importing the Tool package must not pull in web/agent frameworks.

    Checked in a subprocess so the assertion is about a clean interpreter rather than
    about whatever earlier tests happened to import.
    """
    import subprocess

    # NOTE: ``socket`` is deliberately NOT in this list.  Importing the Tool imports
    # the inference engine, which imports torch, and torch imports socket for its
    # distributed/telemetry machinery.  That is model-runtime baggage, not a network
    # dependency of the Tool: the Tool itself opens no connection (asserted separately
    # by the stub-engine request test in test_recommendation_tool_integration.py).
    script = (
        "import sys; sys.path.insert(0, %r);"
        "import recommendation.tools as t;"
        "bad = [m for m in ('fastapi','starlette','httpx','requests','urllib3',"
        "'aiohttp','langgraph','langchain','openai','anthropic','uvicorn') "
        "if m in sys.modules];"
        "print('FORBIDDEN:' + ','.join(bad));"
        "print('TOOL:' + t.RecommendationTool.__name__)" % str(REPO_ROOT)
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "FORBIDDEN:" in completed.stdout
    forbidden = completed.stdout.split("FORBIDDEN:")[1].splitlines()[0].strip()
    assert forbidden == "", f"importing recommendation.tools pulled in: {forbidden}"


def test_tool_metadata_is_framework_neutral(tool: RecommendationTool) -> None:
    """Metadata exposes a stable name/description/version."""
    assert tool.name == TOOL_NAME == "recommend_products"
    assert tool.description == TOOL_DESCRIPTION
    assert tool.version == TOOL_VERSION == 1
    assert tool.metadata() == {
        "name": "recommend_products",
        "description": TOOL_DESCRIPTION,
        "version": 1,
    }


def test_tool_has_no_framework_base_classes() -> None:
    """The Tool is a plain Python class with no framework inheritance."""
    bases = RecommendationTool.__mro__[1:]
    assert bases == (object,), f"unexpected base classes: {bases}"


def test_deterministic_repeated_result() -> None:
    """Identical engine, history and k produce identical results (timings excluded)."""
    tool = RecommendationTool(FakeEngine())
    first = tool.run(RecommendationToolRequest(k=4), context())
    for _ in range(5):
        repeat = tool.run(RecommendationToolRequest(k=4), context())
        assert repeat.recommendations == first.recommendations
        assert repeat.requested_k == first.requested_k
        assert repeat.returned_k == first.returned_k
        assert repeat.history_length == first.history_length
        assert repeat.effective_history_length == first.effective_history_length
        assert repeat.history_truncated == first.history_truncated
        assert repeat.eligible_candidates == first.eligible_candidates


def test_result_is_serialisable(tool: RecommendationTool) -> None:
    """The result round-trips through JSON."""
    import json

    payload = json.loads(json.dumps(tool.run(RecommendationToolRequest(k=3), context()).model_dump()))
    assert payload["returned_k"] == len(payload["recommendations"])
    assert set(payload["recommendations"][0]) == {"rank", "parent_asin", "item_id", "score"}


def test_result_contains_no_natural_language_or_product_metadata(tool: RecommendationTool) -> None:
    """The Tool returns structured data only - no prose, no fabricated attributes."""
    result = tool.run(RecommendationToolRequest(k=3), context())
    payload = result.model_dump()
    for forbidden in ("title", "description", "price", "brand", "image",
                      "explanation", "reason", "summary", "message", "text"):
        assert forbidden not in payload
        for item in payload["recommendations"]:
            assert forbidden not in item


def test_score_field_is_not_named_as_probability(tool: RecommendationTool) -> None:
    """The score keeps its raw-model-score semantics in the schema."""
    from recommendation.tools.schemas import ToolRecommendation

    fields = set(ToolRecommendation.model_fields)
    assert fields == {"rank", "parent_asin", "item_id", "score"}
    description = ToolRecommendation.model_fields["score"].description or ""
    assert "NOT a probability" in description
