"""P0 real E2E: task-scoped hard constraints cross BFF → MCP → CPU Gate.

The scenarios use the real FastAPI BFF, real MCP stdio child, real C4/BM25 catalogue and
isolated SQLite.  They intentionally use English ``avoid wired`` wording because the M9
extractor is English-rule based; the Chinese limitation is documented separately.

Scenario A (single round):
    vague headphones → ASK → MCP restart → "Please avoid wired headphones."
    → finalized wireless product, never a wired product; replay is zero-MCP.

Scenario B (M4 two rounds):
    "推荐一台笔记本" → ASK1 → "游戏，avoid wired" → ASK2 → MCP restart
    → "8000 元".  The constraint must still be present in the post-restart recommendation
    call; if no candidate can verify the exclusion, the response must be a bounded refusal,
    never a violating product.
"""

from __future__ import annotations

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


def _restart(client: TestClient, old_pids: list[int]) -> list[int]:
    response = client.post("/api/demo/restart", json={"profile_id": PROFILE})
    assert response.status_code == 200, response.text
    _wait_ready(client)
    new_pids = _wait_for_new_child(old_pids)
    assert set(new_pids).isdisjoint(old_pids), (old_pids, new_pids)
    return new_pids


def _attributes(card: dict) -> dict[str, dict]:
    return {
        str(item.get("attribute")): dict(item)
        for item in card.get("attributes") or ()
        if isinstance(item, dict) and item.get("attribute")
    }


def test_real_avoid_wired_is_executed_across_cold_resume_and_replay(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    assert db_path is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        session_id = _create(client)
        pids_a = _mcp_child_pids()
        assert pids_a

        chat = client.post(
            "/api/chat/stream",
            json={
                "session_id": session_id,
                "message": "I need some new headphones. Not sure what matters yet.",
            },
        )
        assert chat.status_code == 200, chat.text
        events = twu.parse_sse(chat.text)
        clarification = next(
            (event["data"] for event in events if event["event"] == "clarification"), None
        )
        assert clarification is not None
        ask_one = clarification["clarification_id"]

        # Cold Resume before the reply: the handle must be rebuilt from the durable
        # continuation, then the reply's task constraint must reach recommend_products.
        assert manager.check_handle(HandleRef(PROFILE, session_id, ask_one)).ok
        pids_b = _restart(client, pids_a)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, ask_one)
        ).code == "stale_process"

        calls = _count_calls(manager)
        response = _reply(
            client, session_id, ask_one,
            message="Please avoid wired headphones.",
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["error"] is None
        assert body["recommendation"] is not None, body
        selected = body["recommendation"]["items"][0]
        assert selected["product_id"]
        assert "Wireless" in selected["title"], selected["title"]
        attributes = _attributes(selected)
        assert "wireless" in attributes
        assert "wired" not in attributes, "a wired product must never be presented"
        constraints = body["task_constraints"]
        assert constraints["hard_exclusions"][0]["attribute"] == "wired"
        assert constraints["hard_exclusions"][0]["polarity"] == "exclude"

        recommend_args = next(
            arguments for tool, arguments in calls
            if tool == "recommend_products"
        )
        assert recommend_args["task_constraints"] == constraints
        assert [tool for tool, _ in calls][0] == "restore_clarification_handle"

        state = SQLiteWebStateStore(db_path)
        try:
            pending = state.get_pending_for_clarification(session_id, ask_one)
            assert pending is not None and pending.status == "CONSUMED"
            request = state.get_request(session_id, pending.reply_turn_id)
            assert request is not None and request.status == "COMPLETED"
            task = state.get_task(request.task_id)
            assert task is not None and task.status == "COMPLETED"
        finally:
            state.close()

        # Same reply is a durable replay: identical body, zero MCP, no extra turn.
        calls_before = len(calls)
        replay = _reply(
            client, session_id, ask_one,
            message="Please avoid wired headphones.",
        )
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
        assert len(calls) == calls_before
        assert _turns(db_path, session_id) == [f"{session_id}-t1", body["turn_id"]]
        assert pids_b


def test_ask2_continuation_preserves_constraint_across_cold_resume(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    assert db_path is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        session_id = _create(client)
        pids_a = _mcp_child_pids()
        assert pids_a

        chat = client.post(
            "/api/chat/stream", json={"session_id": session_id, "message": "推荐一台笔记本"}
        )
        assert chat.status_code == 200, chat.text
        events = twu.parse_sse(chat.text)
        ask_one = next(
            event["data"] for event in events if event["event"] == "clarification"
        )["clarification_id"]

        calls_one = _count_calls(manager)
        first = _reply(client, session_id, ask_one, message="游戏，avoid wired")
        assert first.status_code == 200, first.text
        first_body = first.json()
        assert first_body["status"] == "awaiting_clarification", first_body
        ask_two = first_body["clarification"]["clarification_id"]
        assert first_body["task_constraints"]["hard_exclusions"][0]["attribute"] == "wired"

        state = SQLiteWebStateStore(db_path)
        try:
            pending_two = state.get_pending_for_clarification(session_id, ask_two)
            assert pending_two is not None and pending_two.continuation is not None
            stored = pending_two.continuation["mcp"]["task_constraints"]
            assert stored["hard_exclusions"][0]["attribute"] == "wired"
            reply_two_id = pending_two.reply_turn_id
        finally:
            state.close()
        assert any(tool == "recommend_products" for tool, _ in calls_one)

        # Kill the child that created ASK2; the new child must restore the continuation
        # (including the task constraints) before Reply2.
        assert manager.check_handle(HandleRef(PROFILE, session_id, ask_two)).ok
        _restart(client, pids_a)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, ask_two)
        ).code == "stale_process"

        calls_two = _count_calls(manager)
        second = _reply(client, session_id, ask_two, message="8000 元")
        assert second.status_code == 200, second.text
        body = second.json()
        assert body["task_constraints"]["hard_exclusions"][0]["attribute"] == "wired"
        recommend_args = next(
            arguments for tool, arguments in calls_two
            if tool == "recommend_products"
        )
        assert recommend_args["task_constraints"] == body["task_constraints"]
        if body["recommendation"] is not None:
            selected = body["recommendation"]["items"][0]
            assert "wired" not in _attributes(selected)
        else:
            assert body["refusal"]["code"] == "no_verified_candidate_in_scope", body
            assert body["message"]

        state = SQLiteWebStateStore(db_path)
        try:
            consumed = state.get_pending_for_clarification(session_id, ask_two)
            assert consumed is not None and consumed.status == "CONSUMED"
            request = state.get_request(session_id, reply_two_id)
            assert request is not None and request.status == "COMPLETED"
            task = state.get_task(request.task_id)
            assert task is not None and task.status == "COMPLETED"
        finally:
            state.close()
