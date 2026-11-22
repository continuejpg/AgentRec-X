"""M4: real two-round ASK across two real MCP child restarts (CPU/BM25).

The exact conversation is:

    "推荐一台笔记本" -> ASK1 (use case) -> "游戏" -> ASK2 (budget) -> "8000 元"
    -> real final recommendation with the demo historical USD -> CNY price and budget Gate.

Both ASKs are produced by the real MCP server.  The test kills the real stdio child after
ASK1 and again after ASK2, proving each pending resumes from its own durable continuation and
never reuses an old in-process handle.
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
ASK_QUERY = "推荐一台笔记本"
REPLY_ONE = "游戏"
REPLY_TWO = "8000 元"
SIDECAR = REPO_ROOT / "data/enrichment/electronics_metadata_sidecar.jsonl"


def _restart_and_wait(client: TestClient, old_pids: list[int]) -> list[int]:
    response = client.post("/api/demo/restart", json={"profile_id": PROFILE})
    assert response.status_code == 200, response.text
    _wait_ready(client)
    new_pids = _wait_for_new_child(old_pids)
    assert set(new_pids).isdisjoint(old_pids), (old_pids, new_pids)
    return new_pids


def test_two_round_ask_survives_two_mcp_restarts(tmp_path: Path) -> None:
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
        assert pids_a

        # ---- ASK1: the real MCP task policy asks for the missing laptop use case. ----
        chat = client.post(
            "/api/chat/stream", json={"session_id": session_id, "message": ASK_QUERY}
        )
        assert chat.status_code == 200, chat.text
        events = twu.parse_sse(chat.text)
        clarification_one = next(
            (event["data"] for event in events if event["event"] == "clarification"), None
        )
        assert clarification_one is not None, "the real MCP must issue ASK1"
        assert "办公" in clarification_one["question"] or "office" in clarification_one["question"].lower()
        ask_one = clarification_one["clarification_id"]

        state = SQLiteWebStateStore(db_path)
        try:
            pending_one = state.get_pending_for_clarification(session_id, ask_one)
            assert pending_one is not None and pending_one.continuation is not None
            assert pending_one.payload["reason"] == "missing_laptop_use_case"
            original = pending_one.continuation["mcp"]["task_requirements"]
            assert original["category"] == "laptop"
            assert original["budget"] is None
            reply_one_id = pending_one.reply_turn_id
            task_one = state.get_task(pending_one.task_id)
            assert task_one is not None and task_one.status == "WAITING_FOR_USER"
        finally:
            state.close()

        # ---- Restart #1: ASK1 is restored in a different child, then Reply1 creates ASK2. ----
        assert manager.check_handle(HandleRef(PROFILE, session_id, ask_one)).ok
        pids_b = _restart_and_wait(client, pids_a)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, ask_one)
        ).code == "stale_process"

        calls_one = _count_calls(manager)
        first = _reply(client, session_id, ask_one, message=REPLY_ONE)
        assert first.status_code == 200, first.text
        first_body = first.json()
        assert first_body["status"] == "awaiting_clarification", first_body
        assert first_body["recommendation"] is None
        assert first_body["clarification"] is not None
        ask_two = first_body["clarification"]["clarification_id"]
        assert ask_two and ask_two != ask_one
        assert "预算" in first_body["clarification"]["question"]
        assert "8000" not in first_body["message"]
        view_after_one = client.get(f"/api/sessions/{session_id}").json()
        assert view_after_one["pending_clarification"]["clarification_id"] == ask_two
        tools_one = [tool for tool, _ in calls_one]
        assert tools_one[0] == "restore_clarification_handle"
        assert "recommend_products" in tools_one
        assert "get_product_info" not in tools_one, "ASK2 must not pretend to recommend"

        state = SQLiteWebStateStore(db_path)
        try:
            consumed_one = state.get_pending_for_clarification(session_id, ask_one)
            assert consumed_one is not None and consumed_one.status == "CONSUMED"
            pending_two = state.get_pending_for_clarification(session_id, ask_two)
            assert pending_two is not None and pending_two.status == "OPEN"
            assert pending_two.pending_id != consumed_one.pending_id
            assert pending_two.reply_turn_id != reply_one_id
            assert pending_two.turn_id == reply_one_id, "ASK2 is carried by the Reply1 turn"
            assert pending_two.continuation is not None
            second_requirements = pending_two.continuation["mcp"]["task_requirements"]
            assert second_requirements["category"] == "laptop"
            assert "gaming" in second_requirements["use_cases"]
            assert second_requirements["budget"] is None
            task_two = state.get_task(pending_two.task_id)
            assert task_two is not None and task_two.status == "WAITING_FOR_USER"
            assert task_two.task_version > task_one.task_version
            request_one = state.get_request(session_id, reply_one_id)
            assert request_one is not None
            assert (request_one.status, request_one.execution_stage) == (
                "COMPLETED", "COMPLETED"
            )
            assert request_one.result["outcome"] == "awaiting_clarification"
            assert request_one.result["next_clarification_id"] == ask_two
            assert request_one.result["next_pending_id"] == pending_two.pending_id
            assert request_one.result["next_reply_turn_id"] == pending_two.reply_turn_id
            assert [row.turn_id for row in state.list_turns(session_id)] == [
                f"{session_id}-t1", reply_one_id,
            ]
            reply_two_id = pending_two.reply_turn_id
        finally:
            state.close()

        # Reply1 is idempotent: it replays the committed ASK2 and creates no ASK3.
        calls_before_replay = len(calls_one)
        replay_one = _reply(client, session_id, ask_one, message=REPLY_ONE)
        assert replay_one.status_code == 200, replay_one.text
        replay_body = replay_one.json()
        assert replay_body["clarification"]["clarification_id"] == ask_two
        assert replay_body["status"] == "awaiting_clarification"
        assert len(calls_one) == calls_before_replay, "replay must be zero MCP"
        state = SQLiteWebStateStore(db_path)
        try:
            stored_two = state.get_pending(pending_two.pending_id)
            assert stored_two is not None
            assert len(state.list_pending(stored_two.task_id)) == 2
        finally:
            state.close()

        # ---- Restart #2: ASK2 is restored in a third child, then Reply2 finalizes. ----
        assert manager.check_handle(HandleRef(PROFILE, session_id, ask_two)).ok
        pids_c = _restart_and_wait(client, pids_b)
        assert pids_c and set(pids_c).isdisjoint(pids_b)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, ask_two)
        ).code == "stale_process"

        calls_two = _count_calls(manager)
        second = _reply(client, session_id, ask_two, message=REPLY_TWO)
        assert second.status_code == 200, second.text
        body = second.json()
        assert "status" not in body
        assert body["clarification"] is None
        assert body["error"] is None
        assert body["recommendation"] is not None
        recommendation = body["recommendation"]
        items = recommendation["items"]
        assert len(items) == 1
        item = items[0]
        assert item["product_id"] == "B0BY2Z27LC", item
        assert "Gaming Laptop" in item["title"]
        price = item["price"]
        assert price["status"] == "historical_metadata"
        assert price["value"] == 6293.0
        assert price["currency"] == "CNY"
        assert price["original_usd_price"] == 899.0
        assert price["display"] == "¥6,293.00 CNY"
        assert recommendation["retrieval_query"] == "推荐一台笔记本 laptop gaming"
        assert "8000" not in recommendation["retrieval_query"]
        assert body["task_requirements"]["budget"]["max_amount"] == 8000.0
        assert body["task_requirements"]["budget"]["currency"] == "CNY"
        assert "gaming" in body["task_requirements"]["use_cases"]

        state = SQLiteWebStateStore(db_path)
        try:
            consumed_two = state.get_pending_for_clarification(session_id, ask_two)
            assert consumed_two is not None and consumed_two.status == "CONSUMED"
            assert state.get_open_pending_for_session(session_id) is None
            task_final = state.get_task(consumed_two.task_id)
            assert task_final is not None and task_final.status == "COMPLETED"
            request_two = state.get_request(session_id, reply_two_id)
            assert request_two is not None
            assert (request_two.status, request_two.execution_stage) == (
                "COMPLETED", "COMPLETED"
            )
            assert [row.turn_id for row in state.list_turns(session_id)] == [
                f"{session_id}-t1", reply_one_id, reply_two_id,
            ]
        finally:
            state.close()

        # Replay the final reply: identical HTTP body, zero MCP, no extra turn.
        calls_before_final_replay = len(calls_two)
        final_replay = _reply(client, session_id, ask_two, message=REPLY_TWO)
        assert final_replay.status_code == 200, final_replay.text
        assert final_replay.json() == body
        assert len(calls_two) == calls_before_final_replay
        assert _turns(db_path, session_id) == [
            f"{session_id}-t1", reply_one_id, reply_two_id,
        ]

        # A late duplicate of Reply1 still replays its committed ASK2; it cannot create ASK3
        # after the task has already finalized.
        stale_one = _reply(client, session_id, ask_one, message=REPLY_ONE)
        assert stale_one.status_code == 200, stale_one.text
        assert stale_one.json()["clarification"]["clarification_id"] == ask_two
        assert len(calls_two) == calls_before_final_replay
        state = SQLiteWebStateStore(db_path)
        try:
            task_final = state.get_active_task(session_id)
            assert task_final is None
            assert len(state.list_pending(
                state.get_pending(pending_one.pending_id).task_id
            )) == 2
        finally:
            state.close()
        assert _turns(db_path, session_id) == [
            f"{session_id}-t1", reply_one_id, reply_two_id,
        ]
