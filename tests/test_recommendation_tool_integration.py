"""Integration tests for the Recommendation Tool (Milestone 7A).

Two suites:

* a **network-disabled** proof that the Tool performs no I/O and no HTTP when given a
  stub engine (so the guarantee holds independently of torch's import baggage);
* a **real-checkpoint** smoke using the accepted Milestone 5 ``best.pt``, the full
  156,746-item mapping and ``run.json``.

CPU only.  No recommendation-quality metric is computed: this is integration
correctness.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("torch", reason="integration tests require PyTorch")

from recommendation.inference import (  # noqa: E402
    InferenceConfig,
    RecommendationResult,
    SASRecInferenceEngine,
)
from recommendation.tools import (  # noqa: E402
    MissingUserHistory,
    RecommendationContext,
    RecommendationTool,
    RecommendationToolRequest,
)

CHECKPOINT = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "best.pt"
MANIFEST = REPO_ROOT / "runs" / "sasrec_canonical_2026" / "run.json"
MAPPINGS = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO_ROOT / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

#: Accepted formal checkpoint digest.
ACCEPTED_SHA256 = "352bd3ae7ebc5e20adbaac0388ac20fa4b9cc547a579500191d940f77a105912"

artifacts_available = pytest.mark.skipif(
    not (CHECKPOINT.exists() and MAPPINGS.exists() and SEQUENCES.exists()),
    reason="accepted Milestone 5 artifacts not present (they are git-ignored)",
)

_CACHE: dict[str, object] = {}


def _engine() -> SASRecInferenceEngine:
    """Build (and cache) the real inference engine once for the module."""
    if "engine" not in _CACHE:
        _CACHE["engine"] = SASRecInferenceEngine(
            InferenceConfig(
                checkpoint_path=CHECKPOINT,
                mappings_path=MAPPINGS,
                manifest_path=MANIFEST if MANIFEST.exists() else None,
                device="cpu",
                expected_checkpoint_sha256=ACCEPTED_SHA256,
            )
        )
    return _CACHE["engine"]  # type: ignore[return-value]


def _real_histories(limit: int = 5) -> list[list[str]]:
    """Return bounded real test-style histories: train_history + validation_target."""
    if "histories" not in _CACHE:
        payload = json.loads(SEQUENCES.read_text(encoding="utf-8"))
        histories: list[list[str]] = []
        for record in payload["sequences"][:limit]:
            asins = record["parent_asins"]
            histories.append(list(asins[:-2]) + [asins[-2]])  # test target NOT appended
        _CACHE["histories"] = histories
    return _CACHE["histories"]  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# No-network / no-side-effect proof, independent of the real model
# --------------------------------------------------------------------------- #


class StubEngine:
    """Engine stub that fails the test if anything tries to touch the outside world."""

    def __init__(self) -> None:
        self.calls = 0

    def recommend(self, history_parent_asins, k=10) -> RecommendationResult:
        """Return a minimal result without any I/O."""
        self.calls += 1
        return RecommendationResult(
            recommendations=[],
            requested_k=k,
            history_length=len(list(history_parent_asins)),
            effective_history_length=1,
            history_truncated=False,
            eligible_candidates=0,
            timings_ms={},
        )


def test_tool_performs_no_network_io_with_stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocking socket creation and name resolution still lets the Tool succeed.

    This proves the Tool path is fully in-process: no HTTP client, no DNS, no sockets.
    """

    def _forbid(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("the Recommendation Tool must not open a network connection")

    monkeypatch.setattr(socket, "socket", _forbid)
    monkeypatch.setattr(socket, "create_connection", _forbid)
    monkeypatch.setattr(socket, "getaddrinfo", _forbid)
    monkeypatch.setattr(socket, "gethostbyname", _forbid)

    tool = RecommendationTool(StubEngine())
    result = tool.run(
        RecommendationToolRequest(k=5),
        RecommendationContext(user_history=("B000000001", "B000000002")),
    )
    assert result.returned_k == 0
    assert result.requested_k == 5


# --------------------------------------------------------------------------- #
# Real accepted-checkpoint smoke
# --------------------------------------------------------------------------- #


@artifacts_available
def test_real_checkpoint_tool_smoke() -> None:
    """The Tool works end to end against the accepted checkpoint and full mapping."""
    engine = _engine()
    tool = RecommendationTool(engine)

    checks: dict[str, bool] = {}
    checks["engine reused (same instance)"] = tool.engine is engine
    checks["checkpoint identity"] = engine.checkpoint_sha256 == ACCEPTED_SHA256
    checks["model frozen"] = all(not p.requires_grad for p in engine.model.parameters())

    history = _real_histories(1)[0]
    snapshot = list(history)
    result = tool.run(
        RecommendationToolRequest(k=5), RecommendationContext(user_history=tuple(history))
    )

    checks["structured result returned"] = result.requested_k == 5
    checks["source history unmodified"] = history == snapshot
    checks["no PAD returned"] = all(r.item_id != 0 for r in result.recommendations)
    checks["no seen item returned"] = not (
        {r.parent_asin for r in result.recommendations} & set(history)
    )
    checks["ranks contiguous from 1"] = [r.rank for r in result.recommendations] == list(
        range(1, result.returned_k + 1)
    )
    checks["asin/item_id mapping consistent"] = all(
        r.parent_asin == engine.item_id_to_parent_asin(r.item_id) for r in result.recommendations
    )
    checks["scores finite"] = all(
        r.score == r.score and abs(r.score) != float("inf") for r in result.recommendations
    )
    checks["returned_k <= requested_k"] = result.returned_k <= result.requested_k
    checks["history length reported"] = result.history_length == len(history)

    for name, ok in checks.items():
        assert ok, f"failed: {name}"


@artifacts_available
def test_real_checkpoint_repeated_tool_call_is_deterministic() -> None:
    """Identical trusted history and k give identical structured output."""
    tool = RecommendationTool(_engine())
    history = _real_histories(1)[0]
    context = RecommendationContext(user_history=tuple(history))

    first = tool.run(RecommendationToolRequest(k=5), context)
    for _ in range(3):
        repeat = tool.run(RecommendationToolRequest(k=5), context)
        assert repeat.recommendations == first.recommendations
        assert repeat.returned_k == first.returned_k
        assert repeat.history_length == first.history_length
        assert repeat.effective_history_length == first.effective_history_length
        assert repeat.history_truncated == first.history_truncated
        assert repeat.eligible_candidates == first.eligible_candidates


@artifacts_available
def test_real_checkpoint_engine_not_reloaded_per_tool_call() -> None:
    """Many Tool calls reuse one loaded engine and never mutate weights."""
    import torch

    engine = _engine()
    tool = RecommendationTool(engine)
    weights = engine.model.item_embedding.weight.detach().clone()
    loaded_at = engine.loaded_at
    history = _real_histories(1)[0]
    context = RecommendationContext(user_history=tuple(history))

    for _ in range(5):
        tool.run(RecommendationToolRequest(k=3), context)

    assert tool.engine is engine
    assert engine.loaded_at == loaded_at
    assert torch.equal(engine.model.item_embedding.weight.detach(), weights)


@artifacts_available
def test_real_checkpoint_unknown_history_item_fails_strictly() -> None:
    """An unknown parent_asin in trusted history is rejected, never dropped."""
    from recommendation.tools import UnknownHistoryItem

    tool = RecommendationTool(_engine())
    history = list(_real_histories(1)[0]) + ["B9999999999"]
    with pytest.raises(UnknownHistoryItem):
        tool.run(
            RecommendationToolRequest(k=5), RecommendationContext(user_history=tuple(history))
        )


@artifacts_available
def test_real_checkpoint_zero_candidate_history_is_not_an_error() -> None:
    """A history covering the whole catalog returns an empty, valid result."""
    engine = _engine()
    tool = RecommendationTool(engine)
    all_items = engine._id2item[1:]  # noqa: SLF001 - integration introspection
    result = tool.run(
        RecommendationToolRequest(k=5),
        RecommendationContext(user_history=tuple(all_items)),  # type: ignore[arg-type]
    )
    assert result.recommendations == []
    assert result.returned_k == 0
    assert result.eligible_candidates == 0
    assert result.history_truncated is True


@artifacts_available
def test_missing_history_never_becomes_empty_recommendations() -> None:
    """Absent trusted history raises rather than silently returning nothing."""
    tool = RecommendationTool(_engine())
    with pytest.raises(MissingUserHistory):
        tool.run(RecommendationToolRequest(k=5), None)
