"""Real FastAPI + MCP + BM25 evidence for the two opt-in personalization features.

A: M9 long-term preference adds real auxiliary candidates to the formal recommendation pool.
B: a real Like event changes the formal scorer's behavior contribution on the next turn.

The tests use isolated SQLite, the real C4 catalogue and the real CPU/BM25 MCP child.  No
V2 service, Qwen, GPU or frontend is involved.
"""

from __future__ import annotations

import dataclasses
import sys
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.test_web_demo_unit as twu  # noqa: E402
from tests.test_web_cold_resume_e2e import (  # noqa: E402
    _create,
    _real_settings,
    _wait_ready,
)
from web_demo.app import create_app  # noqa: E402

PROFILE = "profile_0057"


def _chat_products(client: TestClient, session_id: str, message: str) -> dict:
    response = client.post(
        "/api/chat/stream", json={"session_id": session_id, "message": message}
    )
    assert response.status_code == 200, response.text
    events = twu.parse_sse(response.text)
    products = next(
        (event["data"] for event in events if event["event"] == "products"), None
    )
    assert products is not None, events
    return products


def test_memory_aware_retrieval_adds_real_candidates(tmp_path: Path) -> None:
    settings = dataclasses.replace(
        _real_settings(tmp_path),
        memory_aware_retrieval=True,
        behavior_aware_rerank=True,
    )
    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        session_id = _create(client)
        remembered = client.post(
            "/api/memory/remember",
            json={"session_id": session_id, "message": "I prefer USB."},
        )
        assert remembered.status_code == 200, remembered.text

        products = _chat_products(client, session_id, "studio headphones")
        retrieval = products["memory_retrieval"]
        assert retrieval["enabled"] is True
        assert retrieval["queries"] == ["studio headphones usb"]
        assert retrieval["base_candidate_count"] == 20
        assert retrieval["new_item_ids"], retrieval
        assert retrieval["merged_candidate_count"] > retrieval["base_candidate_count"]
        assert retrieval["error"] is None
        assert retrieval["extra_latency_ms"] >= 0
        # Base candidates are retained, not replaced by memory.
        assert retrieval["base_candidate_count"] <= retrieval["merged_candidate_count"]
        assert products["recommended_product_id"]
        # Both flags ON is an independent combination: memory adds candidates while the
        # behavior snapshot is empty and contributes no signals.
        assert products["behavior_rerank"]["enabled"] is True
        assert products["behavior_rerank"]["active_product_signals"] == 0


def test_behavior_aware_rerank_contributes_on_the_next_turn(tmp_path: Path) -> None:
    settings = dataclasses.replace(
        _real_settings(tmp_path),
        memory_aware_retrieval=False,
        behavior_aware_rerank=True,
    )
    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        session_id = _create(client)
        first = _chat_products(client, session_id, "wireless studio headphones")
        assert first["behavior_rerank"]["enabled"] is True
        assert first["behavior_rerank"]["active_product_signals"] == 0
        assert "memory_retrieval" not in first

        target = first["items"][2] if len(first["items"]) > 2 else first["items"][-1]
        event = {
            "event_id": "like-" + uuid.uuid4().hex[:12],
            "event_type": "explicit_item_feedback",
            "recommendation_id": first["recommendation_id"],
            "product_id": target["product_id"],
            "displayed_position": target["displayed_position"],
            "event_source": "LIVE_USER_INTERACTION",
            "metadata": {"feedback": "like"},
        }
        recorded = client.post(
            "/api/behavior/events",
            json={"session_id": session_id, "events": [event]},
        )
        assert recorded.status_code == 200, recorded.text

        second = _chat_products(client, session_id, "wireless studio headphones")
        rerank = second["behavior_rerank"]
        assert rerank["enabled"] is True
        assert rerank["active_product_signals"] >= 1
        assert rerank["active_attribute_signals"] >= 1
        assert rerank["candidates_with_behavior_signal"] >= 1
        assert rerank["top_behavior_score"] > 0
        assert rerank["top_behavior_candidate_id"]
        # The behavior is a soft signal: the real product remains a normal recommendation.
        assert second["recommended_product_id"]
        assert "memory_retrieval" not in second


def test_both_flags_off_keeps_the_personalization_metadata_absent(tmp_path: Path) -> None:
    settings = dataclasses.replace(
        _real_settings(tmp_path),
        memory_aware_retrieval=False,
        behavior_aware_rerank=False,
    )
    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        session_id = _create(client)
        client.post(
            "/api/memory/remember",
            json={"session_id": session_id, "message": "I prefer USB."},
        )
        products = _chat_products(client, session_id, "studio headphones")
        assert "memory_retrieval" not in products
        assert "behavior_rerank" not in products
        assert products["recommended_product_id"]
