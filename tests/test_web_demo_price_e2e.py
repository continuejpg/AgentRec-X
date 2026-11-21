"""M1-F: real cold resume with the demo historical FX price (real MCP + BM25).

The user asks for a <=5000 CNY laptop, then revises the budget to 8000 CNY and adds gaming.
The real C4 catalogue identity is matched to the real Raw Meta USD price, converted with
``Decimal`` at the fixed demo rate 7.00, and the budget Gate finalizes a real gaming laptop.
The durable result keeps the converted CNY price, original USD price and rate, so a replay
under the strict mode configuration cannot recompute it.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.test_web_demo_unit as twu  # noqa: E402
from tests.test_web_cold_resume_e2e import (  # noqa: E402
    _count_calls,
    _create,
    _mcp_child_pids,
    _real_settings,
    _reply,
    _turns,
    _wait_for_new_child,
    _wait_ready,
)
from web_demo.app import create_app  # noqa: E402
from web_demo.handle_lifecycle import HandleRef  # noqa: E402
from web_demo.state_store import SQLiteWebStateStore  # noqa: E402

PROFILE = "profile_0057"
ASK_QUERY = "推荐一台 5000 元的笔记本"
REPLY_TEXT = "游戏，预算改成 8000 元"
SIDECAR = REPO_ROOT / "data/enrichment/electronics_metadata_sidecar.jsonl"
DEMO_TOOLS = [
    "restore_clarification_handle",
    "submit_clarification_reply",
    "get_active_preferences",
    "recommend_products",
    "get_product_info",
]


def test_real_demo_fx_price_survives_cold_resume_and_replay(tmp_path: Path) -> None:
    settings = dataclasses.replace(
        _real_settings(tmp_path),
        price_mode="demo_historical_fx",
        demo_price_sidecar=SIDECAR,
    )
    db_path = settings.web_state_db_path
    assert db_path is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        session_id = _create(client)
        pids_a = _mcp_child_pids()

        chat = client.post(
            "/api/chat/stream", json={"session_id": session_id, "message": ASK_QUERY}
        )
        assert chat.status_code == 200, chat.text
        events = twu.parse_sse(chat.text)
        clarification = next(
            (event["data"] for event in events if event["event"] == "clarification"), None
        )
        assert clarification is not None, "the real Gate must ask for the missing use case"
        clarification_id = clarification["clarification_id"]

        state = SQLiteWebStateStore(db_path)
        try:
            pending = state.get_pending_for_clarification(session_id, clarification_id)
            assert pending is not None and pending.continuation is not None
            original = pending.continuation["mcp"]["task_requirements"]
            assert original["budget"]["max_amount"] == 5000.0
            assert original["category"] == "laptop"
            assert "5000" in pending.query, "the original request stays in history"
            reply_turn_id = pending.reply_turn_id
        finally:
            state.close()

        assert manager.check_handle(HandleRef(PROFILE, session_id, clarification_id)).ok
        restart = client.post("/api/demo/restart", json={"profile_id": PROFILE})
        assert restart.status_code == 200, restart.text
        _wait_ready(client)
        pids_b = _wait_for_new_child(pids_a)
        assert set(pids_b).isdisjoint(pids_a)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        ).code == "stale_process"

        calls = _count_calls(manager)
        response = _reply(client, session_id, clarification_id, message=REPLY_TEXT)
        assert response.status_code == 200, response.text
        body = response.json()
        tools = [tool for tool, _ in calls]
        assert tools == DEMO_TOOLS, tools

        recommend_args = next(arguments for tool, arguments in calls
                              if tool == "recommend_products")
        assert "5000" in recommend_args["query"], "MCP receives the original history text"
        merged = recommend_args["task_requirements"]
        assert merged["budget"]["max_amount"] == 8000.0
        assert merged["budget"]["currency"] == "CNY"
        assert merged["category"] == "laptop"
        assert "gaming" in merged["use_cases"]

        recommendation = body["recommendation"]
        assert body["error"] is None
        assert recommendation is not None
        assert recommendation["retrieval_query"] == "推荐一台 笔记本 laptop gaming"
        assert "5000" not in recommendation["retrieval_query"]
        assert "8000" not in recommendation["retrieval_query"]
        items = recommendation["items"]
        assert len(items) == 1
        item = items[0]
        assert item["product_id"] == "B0BY2Z27LC"
        assert "Gaming Laptop" in item["title"]
        price = item["price"]
        assert price["status"] == "historical_metadata"
        assert price["display"] == "¥6,293.00 CNY"
        assert price["value"] == 6293.0
        assert price["currency"] == "CNY"
        assert price["original_usd_price"] == 899.0
        assert price["fx_rate"] == "7.00"
        assert price["converted_cny_price"] == "6293.00"
        assert price["price_kind"] == "historical_demo_converted"
        assert price["source_price_kind"] == "historical_crawl_time"
        assert price["history_year"] == 2023
        assert price["reason"] == "演示参考价：历史美元价格按固定汇率换算，非当前市场售价。"
        assert "演示参考价" in price["reason"]
        assert "非当前市场售价" in price["reason"]
        assert price["source"] == "amazon_reviews_2023:raw_meta_Electronics"

        budget = body["budget_verification"]
        assert budget["budget"]["max_amount"] == 8000.0
        assert budget["selected_candidate_id"] == "B0BY2Z27LC"
        assert budget["selected_status"] == "within"
        assert budget["within"] == 1
        assert budget["unknown"] == 3
        assert budget["selected_price"]["amount"] == 6293.0
        assert budget["selected_price"]["currency"] == "CNY"
        assert budget["selected_price"]["price_kind"] == "historical_demo_converted"
        assert budget["selected_price"]["original_usd_price"] == 899.0
        assert budget["selected_price"]["fx_rate"] == "7.00"
        assert "Price note:" in body["message"]
        assert "演示参考价：历史美元价格按固定汇率换算，非当前市场售价。" in body["message"]

        state = SQLiteWebStateStore(db_path)
        try:
            consumed = state.get_pending_for_clarification(session_id, clarification_id)
            assert consumed is not None and consumed.status == "CONSUMED"
            assert "5000" in consumed.query
            request = state.get_request(session_id, reply_turn_id)
            assert request is not None
            assert (request.status, request.execution_stage) == ("COMPLETED", "COMPLETED")
            stored = request.result["payload"]
            assert stored["retrieval_query"] == "推荐一台 笔记本 laptop gaming"
            assert stored["budget_verification"]["selected_price"]["fx_rate"] == "7.00"
            assert stored["task_requirements"]["budget"]["max_amount"] == 8000.0
            stored_items = stored["reply"]["products"]["items"]
            assert len(stored_items) == 1
            stored_price = stored_items[0]["price"]
            assert stored_price["converted_cny_price"] == "6293.00"
            assert stored_price["original_usd_price"] == 899.0
            assert stored_price["fx_rate"] == "7.00"
            assert stored_price["reason"].startswith("演示参考价")
            assert stored["reply"]["recommendation"]["id"] is not None
            turns = [row.turn_id for row in state.list_turns(session_id)]
            assert turns == [f"{session_id}-t1", reply_turn_id]
        finally:
            state.close()

        calls_before_replay = len(calls)
        replay = _reply(client, session_id, clarification_id, message=REPLY_TEXT)
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
        assert len(calls) == calls_before_replay
        assert _turns(db_path, session_id) == [f"{session_id}-t1", reply_turn_id]

    # Restart the BFF in STRICT mode.  The durable reply must replay the already-committed
    # CNY price and rate unchanged, with zero MCP calls, proving it is not recomputed.
    strict = dataclasses.replace(settings, price_mode="strict", demo_price_sidecar=None)
    with TestClient(create_app(settings=strict)) as client2:
        _wait_ready(client2)
        calls2 = _count_calls(client2.app.state.manager)
        replayed = _reply(client2, session_id, clarification_id, message=REPLY_TEXT)
        assert replayed.status_code == 200, replayed.text
        replayed_body = replayed.json()
        assert calls2 == [], "a committed reply must replay without MCP or price recomputation"
        assert replayed_body == body
        replayed_price = replayed_body["recommendation"]["items"][0]["price"]
        assert replayed_price["converted_cny_price"] == "6293.00"
        assert replayed_price["original_usd_price"] == 899.0
        assert replayed_price["fx_rate"] == "7.00"
