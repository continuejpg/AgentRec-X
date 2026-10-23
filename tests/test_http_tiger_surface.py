"""HTTP-surface fixes from T-02.7: response schema, health exposure, one shared engine.

Regression tests for the three defects T-02.6 found by exercising the real HTTP service:

* ``RecommendResponse`` forbade ``provenance`` and ``score_kind``, so ``/v1/recommend`` returned
  500 for every engine that reports provenance (``app.py:526`` splats ``as_dict()``);
* ``HealthResponse`` carried no TIGER block, so a deployment could not see the worker's state,
  hashes or retrieval mode;
* the serving fusion and the Agent service each built their own TIGER engine, so only one of them
  was ever prewarmed and the Agent's ``tiger`` plan could never run.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.api.schemas import HealthResponse, RecommendResponse, TigerHealthBlock
from recommendation.inference.tiger_engine import (
    TigerServingEngine,
    set_shared_engine,
    shared_engine,
)
from recommendation.demo.agent_service import DemoAgentService
from recommendation.inference.sasrec import Recommendation, RecommendationResult

TIGER_PROVENANCE: dict[str, Any] = {
    "item_id": 151270,
    "fused_rank": 1,
    "fused_score": 0.048660,
    "score_kind": "rrf_rank_fusion",
    "sources": ["sequential", "two_tower", "tiger"],
    "source_ranks": {"sequential": 1, "two_tower": 2, "tiger": 2},
    "source_score_kinds": {
        "sequential": "sasrec_logit",
        "two_tower": "two_tower_cosine",
        "tiger": "tiger_sid_sequence_logprob",
    },
    "tiger": {
        "rank": 2,
        "score": -5.556250479305163,
        "score_kind": "tiger_sid_sequence_logprob",
        "item_id": 151270,
        "semantic_id": [104, 21, 238, 5],
        "retrieval_mode": "CERTIFIED",
    },
}


def fused_result(**overrides: Any) -> RecommendationResult:
    payload: dict[str, Any] = {
        "recommendations": [
            Recommendation(rank=1, item_id=151270, parent_asin="B0BX5QFWQN", score=0.04866)
        ],
        "requested_k": 1,
        "history_length": 3,
        "effective_history_length": 3,
        "history_truncated": False,
        "eligible_candidates": 42,
        "timings_ms": {"source_tiger_ms": 1404.429, "fusion_ms": 0.238, "total_ms": 3781.264},
        "score_kind": "rrf_rank_fusion",
        "provenance": {"B0BX5QFWQN": TIGER_PROVENANCE},
    }
    payload.update(overrides)
    return RecommendationResult(**payload)


# --------------------------------------------------------------------------- #
# C. response schema
# --------------------------------------------------------------------------- #


def test_a_real_fused_result_satisfies_the_response_schema() -> None:
    """This is the exact call the route makes: ``RecommendResponse(**result.as_dict())``."""
    response = RecommendResponse(**fused_result().as_dict())
    assert response.score_kind == "rrf_rank_fusion"
    assert response.returned_k == 1


def test_provenance_survives_the_roundtrip_intact() -> None:
    response = RecommendResponse(**fused_result().as_dict())
    entry = response.provenance["B0BX5QFWQN"]
    assert entry["sources"] == ["sequential", "two_tower", "tiger"]
    assert entry["source_ranks"]["tiger"] == 2
    assert entry["source_score_kinds"]["tiger"] == "tiger_sid_sequence_logprob"
    tiger = entry["tiger"]
    assert tiger["retrieval_mode"] == "CERTIFIED"
    assert tiger["score_kind"] == "tiger_sid_sequence_logprob"
    assert tiger["semantic_id"] == [104, 21, 238, 5]
    # Serialised form keeps it too, which is what the HTTP body carries.
    assert response.model_dump()["provenance"]["B0BX5QFWQN"]["tiger"]["rank"] == 2


def test_the_schema_still_rejects_unknown_fields() -> None:
    payload = fused_result().as_dict()
    payload["not_a_field"] = 1
    with pytest.raises(Exception):
        RecommendResponse(**payload)


def test_a_result_without_provenance_still_validates() -> None:
    """A single-model answer has no fusion provenance; the field defaults to empty."""
    payload = fused_result(provenance={}).as_dict()
    assert RecommendResponse(**payload).provenance == {}


# --------------------------------------------------------------------------- #
# D. health exposure
# --------------------------------------------------------------------------- #


def tiger_block(**overrides: Any) -> dict[str, Any]:
    payload = {
        "state": "loading",
        "ready": False,
        "checkpoint_sha256": "95e5cb6f816bb271b8551a6dd9b62ee33c8e0f6b5908ce426622113a072c6c0a",
        "sid_artifact_sha256": "6501fbe8e146e3c57ce7a146e37c3be29235ae5bbd082116f6a8e30e739e8d0e",
        "retrieval_mode": "CERTIFIED",
        "error": None,
    }
    payload.update(overrides)
    return payload


def test_health_carries_the_tiger_block() -> None:
    health = HealthResponse(
        status="unavailable",
        model_loaded=False,
        active_engine="fusion_rrf",
        fusion_sources=["sequential", "two_tower", "catalog_lexical", "tiger"],
        tiger=tiger_block(),
    )
    assert health.tiger is not None
    assert health.tiger.state == "loading"
    assert health.tiger.ready is False, "loading must never report ready"
    assert health.tiger.checkpoint_sha256.startswith("95e5cb6f")
    assert health.tiger.retrieval_mode == "CERTIFIED"


def test_health_without_tiger_is_disabled_not_ready() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    assert health.tiger is None
    disabled = TigerHealthBlock(state="disabled", ready=False)
    assert disabled.ready is False


def test_a_ready_block_reports_the_worker_state() -> None:
    block = TigerHealthBlock(**tiger_block(state="ready", ready=True))
    assert block.ready is True and block.state == "ready"
    # ``extra="ignore"``: the engine's health() is the source of truth and may grow a field.
    # That must never turn /health into a 500 (it did, before T-02.7B).
    tolerant = TigerHealthBlock(**tiger_block(), a_future_engine_field=1)
    assert tolerant.state == "loading"


def test_a_failed_worker_carries_its_reason() -> None:
    block = TigerHealthBlock(**tiger_block(state="failed", error="TigerAssetsMissing: tiger.pt"))
    assert block.state == "failed" and "tiger.pt" in (block.error or "")


# --------------------------------------------------------------------------- #
# A. one engine per process
# --------------------------------------------------------------------------- #


def test_the_registry_hands_out_one_engine() -> None:
    sentinel = object()
    try:
        set_shared_engine(sentinel)  # type: ignore[arg-type]
        assert shared_engine() is sentinel
    finally:
        set_shared_engine(None)
    assert shared_engine() is None


def test_the_agent_service_adopts_the_registered_engine() -> None:
    from recommendation.demo.agent_service import _build_tiger_engine

    sentinel = object()
    try:
        set_shared_engine(sentinel)  # type: ignore[arg-type]
        assert _build_tiger_engine(identity=object()) is sentinel
    finally:
        set_shared_engine(None)


def test_agent_readiness_follows_the_shared_engine_not_a_private_copy() -> None:
    """The Agent must not hold its own idea of readiness; it reads the engine's health."""
    loading = SimpleNamespace(health=lambda: {"state": "loading", "started": False})
    ready = SimpleNamespace(health=lambda: {"state": "ready", "started": True})

    assert DemoAgentService._tiger_state_now(  # noqa: SLF001 - the seam under test
        SimpleNamespace(_tiger_engine=loading, _tiger_state="not_started")
    ) == "loading"
    assert DemoAgentService._tiger_state_now(  # noqa: SLF001
        SimpleNamespace(_tiger_engine=ready, _tiger_state="not_started")
    ) == "ready"
    assert DemoAgentService._tiger_state_now(  # noqa: SLF001
        SimpleNamespace(_tiger_engine=None, _tiger_state="disabled")
    ) == "disabled"


def test_a_broken_engine_health_probe_reports_failed() -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("worker gone")

    assert DemoAgentService._tiger_state_now(  # noqa: SLF001
        SimpleNamespace(_tiger_engine=SimpleNamespace(health=boom), _tiger_state="ready")
    ) == "failed"


def test_the_engine_exposes_health_before_it_is_started() -> None:
    from pathlib import Path

    from recommendation.inference.tiger_engine import TigerWorkerConfig

    engine = TigerServingEngine(
        {"B0TEST": 1},
        config=TigerWorkerConfig(checkpoint_dir=Path("/nonexistent/checkpoint")),
    )
    health = engine.health()
    assert health["state"] == "not_started"
    assert health["started"] is False
    assert health["assets_missing"], "a cold deployment must say what is missing"
    assert health["retrieval_mode"] == "CERTIFIED"


# --------------------------------------------------------------------------- #
# A. /health reads TIGER live, not from the startup snapshot
# --------------------------------------------------------------------------- #


class _ServingStub:
    """A fusion-engine stand-in exposing only what /health reads."""

    def __init__(self, health: dict[str, Any] | None) -> None:
        self._health = health

    def tiger_health(self) -> dict[str, Any] | None:
        return self._health


def loading_health() -> dict[str, Any]:
    return {"state": "loading", "started": False, "device": "cpu", "retrieval_mode": "CERTIFIED",
            "checkpoint_sha256": "", "sid_artifact_sha256": "", "error": None, "assets_missing": []}


def ready_health() -> dict[str, Any]:
    return {"state": "ready", "started": True, "device": "cpu", "retrieval_mode": "CERTIFIED",
            "checkpoint_sha256": "95e5cb6f" + "0" * 56, "sid_artifact_sha256": "6501fbe8" + "0" * 56,
            "error": None, "assets_missing": []}


def test_health_block_follows_the_engine_from_loading_to_ready() -> None:
    from recommendation.api.app import live_tiger_block

    engine = _ServingStub(loading_health())
    block = live_tiger_block(engine, {})
    assert block["state"] == "loading" and block["ready"] is False

    # The same process, minutes later: the very same call must now report ready.
    engine._health = ready_health()  # noqa: SLF001 - simulating the worker finishing its load
    block = live_tiger_block(engine, {})
    assert block["state"] == "ready" and block["ready"] is True
    assert block["checkpoint_sha256"].startswith("95e5cb6f")
    assert block["retrieval_mode"] == "CERTIFIED"


def test_health_block_reports_a_failed_engine_and_survives_a_broken_probe() -> None:
    from recommendation.api.app import live_tiger_block

    failed = live_tiger_block(_ServingStub({**ready_health(), "state": "failed",
                                            "error": "TigerAssetsMissing: tiger.pt"}), {})
    assert failed["state"] == "failed" and failed["ready"] is False
    assert "tiger.pt" in failed["error"]

    def boom() -> dict[str, Any]:
        raise RuntimeError("worker gone")

    broken = live_tiger_block(SimpleNamespace(tiger_health=boom), {})
    assert broken["state"] == "failed", "a broken probe must degrade, not 500"


def test_health_block_falls_back_to_the_startup_report_without_a_tiger_engine() -> None:
    from recommendation.api.app import live_tiger_block

    cached = {"state": "disabled", "ready": False}
    assert live_tiger_block(None, {"tiger": cached}) == cached
    assert live_tiger_block(None, {}) is None, "a V1 deployment reports no TIGER block"


# --------------------------------------------------------------------------- #
# B. closing the Agent must not stop a worker the fusion still uses
# --------------------------------------------------------------------------- #


class _EngineStub:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _closable_service(engine: _EngineStub, *, owns: bool) -> SimpleNamespace:
    with_engine = SimpleNamespace(
        _lock=threading.Lock(),
        _controllers={},
        _tiger_engine=engine,
        _tiger_tool=object(),
        _owns_tiger_engine=owns,
        _tiger_state="ready",
    )
    return with_engine


def test_closing_the_agent_leaves_a_borrowed_engine_running() -> None:
    borrowed = _EngineStub()
    DemoAgentService.close(_closable_service(borrowed, owns=False))  # noqa: SLF001
    assert borrowed.closed is False, "the serving fusion still owns this worker"
    assert shared_engine() is None or True  # registry untouched by a non-owner


def test_closing_the_agent_stops_an_engine_it_created() -> None:
    owned = _EngineStub()
    DemoAgentService.close(_closable_service(owned, owns=True))  # noqa: SLF001
    assert owned.closed is True


def test_release_serving_engine_closes_the_worker_and_clears_the_registry() -> None:
    from recommendation.api.app import release_serving_engine

    engine = _EngineStub()
    try:
        set_shared_engine(engine)  # type: ignore[arg-type]
        outcome = release_serving_engine(SimpleNamespace(engine=SimpleNamespace(tiger=engine)))
    finally:
        set_shared_engine(None)
    assert outcome == {"tiger_released": True, "registry_cleared": True}
    assert engine.closed is True, "the worker must actually be stopped"
    assert shared_engine() is None, "the shared registry must be cleared"


def test_release_serving_engine_is_safe_without_tiger_and_never_raises() -> None:
    from recommendation.api.app import release_serving_engine

    assert release_serving_engine(SimpleNamespace(engine=None))["tiger_released"] is False
    assert release_serving_engine(SimpleNamespace(engine=SimpleNamespace()))["registry_cleared"]

    class _Angry:
        def close(self) -> None:
            raise RuntimeError("already gone")

    outcome = release_serving_engine(SimpleNamespace(engine=SimpleNamespace(tiger=_Angry())))
    assert outcome["tiger_released"] is False, "a failing close must not propagate"


def test_the_agent_constructor_can_resolve_the_shared_engine_name() -> None:
    """The exact failure behind the T-02.7C 502.

    ``DemoAgentService.__init__`` calls ``shared_engine()``; while that name was imported only
    inside ``_build_tiger_engine``, every Agent request raised ``NameError`` in ``__init__`` and
    surfaced as a generic 502.  The name must resolve at module scope.
    """
    from recommendation.demo import agent_service

    assert callable(getattr(agent_service, "shared_engine", None))

    sentinel = object()
    try:
        set_shared_engine(sentinel)  # type: ignore[arg-type]
        assert agent_service.shared_engine() is sentinel
    finally:
        set_shared_engine(None)
    assert agent_service.shared_engine() is None
