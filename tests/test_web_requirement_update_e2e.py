"""M1-E: Cold Resume after a real budget/use-case update (real MCP + FastAPI).

Scenario: "推荐一台 5000 元的笔记本" -> real ASK -> MCP child restart -> "游戏，预算改成
8000 元" -> restore the original task, merge the latest requirements, and run the real
recommendation.

The C4 Electronics catalogue exposes **no price** in this deployment, and the enriched Raw
Meta sidecar prices are USD while the user's budget is CNY.  A budget-compliant product
therefore cannot be verified here.  The required outcome is an explicit, durable refusal with
the merged requirements -- never a fabricated product card.
"""

from __future__ import annotations

import json
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
UPDATE_TOOLS = [
    "restore_clarification_handle",
    "submit_clarification_reply",
    "get_active_preferences",
    "recommend_products",
]


def test_cold_resume_merges_budget_and_refuses_unverifiable_price(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    memory_db = settings.memory_db_path
    assert db_path is not None and memory_db is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        session_id = _create(client)
        pids_a = _mcp_child_pids()
        assert pids_a

        # 1-3. Real Chinese shopping request -> real ASK.
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

        # 4. Task/Pending/Continuation persisted with the ORIGINAL 5000 CNY / laptop task.
        state = SQLiteWebStateStore(db_path)
        try:
            pending = state.get_pending_for_clarification(session_id, clarification_id)
            assert pending is not None and pending.continuation is not None
            original = pending.continuation["mcp"]["task_requirements"]
            assert original["budget"]["max_amount"] == 5000.0
            assert original["budget"]["currency"] == "CNY"
            assert original["category"] == "laptop"
            reply_turn_id = pending.reply_turn_id
        finally:
            state.close()

        # 5-7. Kill child A, start child B, prove the old handle is dead.
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        ).ok
        restart = client.post("/api/demo/restart", json={"profile_id": PROFILE})
        assert restart.status_code == 200, restart.text
        _wait_ready(client)
        pids_b = _wait_for_new_child(pids_a)
        assert set(pids_b).isdisjoint(pids_a), (pids_a, pids_b)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        ).code == "stale_process"

        # 8-12. Original session + original pending + new Chinese reply -> cold resume.
        calls = _count_calls(manager)
        response = _reply(client, session_id, clarification_id, message=REPLY_TEXT)
        assert response.status_code == 200, response.text
        body = response.json()
        tools = [tool for tool, _ in calls]
        assert tools == UPDATE_TOOLS, tools

        # Restore carried the original 5000/laptop task...
        restore_requirements = calls[0][1]["continuation"]["task_requirements"]
        assert restore_requirements["budget"]["max_amount"] == 5000.0
        assert restore_requirements["category"] == "laptop"
        # ...and the recommender received the merged 8000/gaming/laptop requirements.
        recommend_requirements = calls[3][1]["task_requirements"]
        assert recommend_requirements["budget"]["max_amount"] == 8000.0
        assert recommend_requirements["budget"]["currency"] == "CNY"
        assert recommend_requirements["budget"]["source_text"] == REPLY_TEXT
        assert recommend_requirements["category"] == "laptop"
        assert "gaming" in recommend_requirements["use_cases"]

        # Not a fake success: no product, explicit machine-readable refusal.
        assert body["recommendation"] is None
        assert body["error"] is None
        assert body["turn_id"] == reply_turn_id
        refusal = body["refusal"]
        assert refusal["code"] == "budget_unverifiable"
        assert refusal["budget"]["budget"]["max_amount"] == 8000.0
        assert refusal["budget"]["unknown"] == refusal["budget"]["candidates_checked"]
        assert refusal["task_requirements"]["budget"]["max_amount"] == 8000.0
        assert refusal["task_requirements"]["category"] == "laptop"
        assert "gaming" in refusal["task_requirements"]["use_cases"]
        assert "No product recommendation" in body["message"]
        assert "8000 CNY" in body["message"]
        assert "price UNKNOWN" in body["message"]
        assert "not a recommendation" in body["message"]

        # 13. Durable state: request completed, pending consumed, ONE reply turn with no product.
        state = SQLiteWebStateStore(db_path)
        try:
            consumed = state.get_pending_for_clarification(session_id, clarification_id)
            assert consumed is not None and consumed.status == "CONSUMED"
            request = state.get_request(session_id, reply_turn_id)
            assert request is not None
            assert (request.status, request.execution_stage) == ("COMPLETED", "COMPLETED")
            stored = request.result["payload"]
            assert stored["refusal"]["code"] == "budget_unverifiable"
            assert stored["reply"]["products"]["items"] == []
            assert stored["reply"]["recommendation"]["id"] is None
            assert stored["task_requirements"]["budget"]["max_amount"] == 8000.0
            turns = [row.turn_id for row in state.list_turns(session_id)]
            assert turns == [f"{session_id}-t1", reply_turn_id]
            task = state.get_task(request.task_id)
            assert task is not None and task.status == "COMPLETED"
        finally:
            state.close()

        # The Chinese reply was NOT invented into permanent M9 preferences.
        preferences = client.get(
            "/api/preferences", params={"session_id": session_id}
        ).json()
        assert [row["value"] for row in preferences["active"]] == []

        # 14. Replay: identical refusal, zero MCP, no duplicate turn.
        calls_before_replay = len(calls)
        replay = _reply(client, session_id, clarification_id, message=REPLY_TEXT)
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
        assert len(calls) == calls_before_replay
        assert _turns(db_path, session_id) == [f"{session_id}-t1", reply_turn_id]

    # 15. BFF restart: the durable refusal turn is restored and replays with zero MCP.
    with TestClient(create_app(settings=settings)) as client2:
        _wait_ready(client2)
        calls2 = _count_calls(client2.app.state.manager)
        view = client2.get(f"/api/sessions/{session_id}").json()
        assert [turn["turn_id"] for turn in view["turns"]] == [
            f"{session_id}-t1", reply_turn_id,
        ]
        assert view["pending_clarification"] is None
        restored = view["turns"][-1]
        assert restored["products"]["items"] == []
        assert restored["assistant_text"] == body["message"]

        replayed = _reply(client2, session_id, clarification_id, message=REPLY_TEXT)
        assert replayed.status_code == 200, replayed.text
        replay_body = replayed.json()
        assert replay_body["recommendation"] is None
        assert replay_body["refusal"]["code"] == "budget_unverifiable"
        assert replay_body["turn_id"] == reply_turn_id
        assert calls2 == [], "a completed refusal replay must not touch MCP"
