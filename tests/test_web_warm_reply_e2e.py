"""M1-C Round C: durable Warm Reply end to end (real MCP + FastAPI + SQLite).

The real-MCP tests start the actual BM25 MCP servers through the BFF's ``MCPManager``, ask a
clarification with a real catalogue query, reply through the real HTTP route, and then verify
the committed web state, the M9 preference, the replay behaviour and recovery after a BFF
restart.  The fail-closed tests at the end use the existing fake manager so each failure mode
is deterministic and cheap to reproduce.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.test_web_demo_unit as twu  # noqa: E402
from web_demo.app import create_app  # noqa: E402
from web_demo.config import Settings  # noqa: E402
from web_demo.mcp_manager import MCPToolTimeout  # noqa: E402
from web_demo.state_store import SQLiteWebStateStore  # noqa: E402

PROFILE = "profile_0057"
ASK_QUERY = "I need some new headphones. Not sure what matters yet."
REPLY_TEXT = "I prefer noise cancelling."
REPLY_TOOLS = [
    "submit_clarification_reply",
    "get_active_preferences",
    "recommend_products",
    "get_product_info",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _real_settings(tmp_path: Path) -> Settings:
    return Settings(
        search_mode="bm25",
        allow_memory_writes=True,
        log_dir=tmp_path / "logs",
        repo=REPO_ROOT,
        python_executable=sys.executable,
        mcp_timeout_s=180.0,
        web_state_db_path=tmp_path / "state.sqlite3",
        memory_db_path=tmp_path / "m9.sqlite3",
        clarification_ttl_s=1800.0,
    )


def _fake_settings(tmp_path: Path) -> Settings:
    return Settings(
        search_mode="bm25",
        allow_memory_writes=True,
        log_dir=tmp_path / "logs",
        web_state_db_path=tmp_path / "state.sqlite3",
        python_executable=sys.executable,
        mcp_timeout_s=30.0,
    )


def _wait_ready(client: TestClient, timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    response = None
    while time.time() < deadline:
        response = client.get("/api/health")
        if response.status_code == 200 and response.json().get("ready"):
            return
        time.sleep(1.0)
    raise AssertionError(
        "the real MCP manager never became ready: "
        + (response.text if response is not None else "no health response")
    )


def _count_calls(manager) -> list[str]:
    """Record every real tool call made through this manager."""
    calls: list[str] = []
    real = manager.call_tool

    async def counting(profile_id, tool, arguments=None, **kwargs):
        calls.append(tool)
        return await real(profile_id, tool, arguments, **kwargs)

    manager.call_tool = counting
    return calls


def _create(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"profile_id": PROFILE})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def _ask(client: TestClient, session_id: str) -> dict:
    response = client.post(
        "/api/chat/stream", json={"session_id": session_id, "message": ASK_QUERY}
    )
    assert response.status_code == 200, response.text
    events = twu.parse_sse(response.text)
    clarification = next(
        (event["data"] for event in events if event["event"] == "clarification"), None
    )
    assert clarification is not None, "the real catalogue query must produce a clarification"
    assert clarification["clarification_id"].startswith("clar_")
    return clarification


def _reply(
    client: TestClient, session_id: str, clarification_id: str,
    message: str = REPLY_TEXT,
):
    return client.post(
        "/api/clarification/reply",
        json={
            "session_id": session_id,
            "clarification_id": clarification_id,
            "message": message,
        },
    )


def _turns(db_path: Path, session_id: str) -> list[str]:
    state = SQLiteWebStateStore(db_path)
    try:
        return [row.turn_id for row in state.list_turns(session_id)]
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# real MCP: commit, M9, replay, restart recovery
# --------------------------------------------------------------------------- #


def test_real_mcp_warm_reply_commits_replays_and_survives_restart(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    assert db_path is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        calls = _count_calls(manager)

        session_id = _create(client)
        clarification = _ask(client, session_id)
        clarification_id = clarification["clarification_id"]
        assert "request_preference_clarification" in calls
        assert "submit_clarification_reply" not in calls

        calls_before = len(calls)
        response = _reply(client, session_id, clarification_id)
        assert response.status_code == 200, response.text
        body = response.json()

        assert calls[calls_before:] == REPLY_TOOLS, (
            "the warm reply must perform exactly the real M9 write, preference read, "
            f"resume recommendation and product enrichment; saw {calls[calls_before:]}"
        )
        assert body["turn_id"] == f"{session_id}-r1"
        assert body["clarification_id"] == clarification_id
        assert body["clarification"] is None
        assert body["error"] is None
        assert body["message"].startswith(f"Memory update ({PROFILE})")
        assert body["memory_update"]["added"] == 1
        assert body["memory_update"]["already_processed"] is False
        assert body["preferences"]["profile_id"] == PROFILE
        assert "noise cancelling" in [row["value"] for row in body["preferences"]["active"]]

        recommendation = body["recommendation"]
        recommendation_id = recommendation["recommendation_id"]
        assert recommendation_id.startswith("rec-")
        assert recommendation["request_id"] == f"{session_id}-r1"
        assert recommendation["items"], "a committed reply must display a real product card"
        card = recommendation["items"][0]
        assert card["product_id"] == recommendation["recommended_product_id"]
        assert card["title"]
        assert card["source"] == "recommendation"
        assert card["match_status"] == "recommended"
        assert card["displayed_position"] == 1
        assert recommendation["gate"]["action"] == "ALLOW_FINALIZE"
        assert card["unknown_fields"], "real catalogue UNKNOWN fields must stay visible"

        preferences = client.get("/api/preferences", params={"session_id": session_id}).json()
        assert "noise cancelling" in [row["value"] for row in preferences["active"]]

        view = client.get(f"/api/sessions/{session_id}").json()
        assert view["pending_clarification"] is None
        assert [turn["turn_id"] for turn in view["turns"]] == [
            f"{session_id}-t1", f"{session_id}-r1",
        ]
        assert view["turns"][-1]["products"]["items"][0]["product_id"] == card["product_id"]
        assert view["turns"][-1]["memory_update"]["added"] == 1
        assert [row["tool"] for row in view["turns"][-1]["tool_activity"]] == REPLY_TOOLS

        # Replay: identical response, no second MCP call, no second turn.
        calls_before_replay = len(calls)
        replay = _reply(client, session_id, clarification_id)
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
        assert len(calls) == calls_before_replay, "a replay must not call MCP again"
        assert _turns(db_path, session_id) == [f"{session_id}-t1", f"{session_id}-r1"]

    # The durable rows (independent connection).
    state = SQLiteWebStateStore(db_path)
    try:
        pending = state.get_pending_for_clarification(session_id, clarification_id)
        assert pending is not None and pending.status == "CONSUMED"
        request = state.get_request(session_id, f"{session_id}-r1")
        assert request is not None
        assert (request.status, request.execution_stage) == ("COMPLETED", "COMPLETED")
        stored = request.result["payload"]
        assert stored["reply"]["recommendation"]["id"] == recommendation_id
        assert stored["reply"]["products"]["items"][0]["product_id"] == card["product_id"]
        assert stored["reply"]["assistant_text"] == body["message"]
        assert [step["tool"] for step in stored["steps"]] == REPLY_TOOLS
        assert state.get_open_pending_for_session(session_id) is None
        task = state.get_task(request.task_id)
        assert task is not None and task.status == "COMPLETED"
    finally:
        state.close()

    # The real M9 SQLite database holds the preference under the Web session key.
    memory_db = sqlite3.connect(str(settings.memory_db_path))
    try:
        rows = memory_db.execute(
            "SELECT user_key, value FROM preference_memory WHERE status = 'active'"
        ).fetchall()
    finally:
        memory_db.close()
    assert (session_id, "noise cancelling") in rows

    # BFF restart: a new manager (new child, new handle table) restores the reply from SQLite.
    with TestClient(create_app(settings=settings)) as client2:
        _wait_ready(client2)
        calls2 = _count_calls(client2.app.state.manager)
        view = client2.get(f"/api/sessions/{session_id}").json()
        assert [turn["turn_id"] for turn in view["turns"]] == [
            f"{session_id}-t1", f"{session_id}-r1",
        ]
        assert view["turns"][-1]["products"]["items"][0]["product_id"] == card["product_id"]
        assert view["turns"][-1]["memory_update"]["added"] == 1
        assert view["pending_clarification"] is None

        replayed = _reply(client2, session_id, clarification_id)
        assert replayed.status_code == 200, replayed.text
        replay_body = replayed.json()
        assert replay_body["turn_id"] == f"{session_id}-r1"
        assert replay_body["recommendation"]["recommendation_id"] == recommendation_id
        assert replay_body["recommendation"]["items"][0]["product_id"] == card["product_id"]
        assert calls2 == [], "a durable replay must not start a new MCP child call"


def test_real_mcp_legacy_pending_without_continuation_is_refused(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    assert db_path is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        session_id = _create(client)
        clarification = _ask(client, session_id)
        clarification_id = clarification["clarification_id"]

    # Simulate a pending written before Cold Resume existed: the row is intact but has no
    # restorable snapshot, so the reply must fail closed instead of guessing a task.
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("UPDATE web_pending SET continuation_json = NULL")
        raw.commit()
    finally:
        raw.close()

    # Everything that could prove the handle lived in the stopped child process.
    with TestClient(create_app(settings=settings)) as client2:
        _wait_ready(client2)
        calls = _count_calls(client2.app.state.manager)
        before = _turns(db_path, session_id)
        response = _reply(client2, session_id, clarification_id)

    assert response.status_code == 409, response.text
    error = response.json()
    assert error["code"] == "clarification_resume_not_ready"
    assert error["detail"]["handle"]["code"] == "continuation_missing"
    assert calls == [], "no MCP call may follow a pending without a continuation"
    assert _turns(db_path, session_id) == before == [f"{session_id}-t1"]

    state = SQLiteWebStateStore(db_path)
    try:
        pending = state.get_pending_for_clarification(session_id, clarification_id)
        assert pending is not None and pending.status == "OPEN"
        task = state.get_active_task(session_id)
        assert task is not None and task.status == "WAITING_FOR_USER"
        assert state.get_request(session_id, pending.reply_turn_id) is None
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# HTTP fail-closed paths (deterministic fake manager)
# --------------------------------------------------------------------------- #


def _warm_fake_app(tmp_path: Path):
    manager = twu.warm_reply_manager()
    return manager, create_app(settings=_fake_settings(tmp_path), manager=manager)


def test_concurrent_replies_have_exactly_one_executor(tmp_path: Path) -> None:
    manager, app = _warm_fake_app(tmp_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]

        barrier = threading.Barrier(2)
        responses: list = []

        def worker() -> None:
            barrier.wait(timeout=30)
            responses.append(_reply(client, session_id, clarification_id))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        turns = _turns(tmp_path / "state.sqlite3", session_id)

    assert len(responses) == 2
    statuses = sorted(response.status_code for response in responses)
    details = [response.json() for response in responses]
    assert 200 in statuses, details
    # The duplicate never executes: depending on the race it is either a replay (200), a
    # lease rejection (409) or the store's fail-closed EXTERNAL_STARTED answer (503
    # "ambiguous").  None of them may start a second external call.
    assert set(statuses) <= {200, 409, 503}, details
    if 503 in statuses:
        ambiguous = next(r for r in responses if r.status_code == 503).json()
        assert ambiguous["code"] == "clarification_reply_ambiguous"
    if statuses == [200, 200]:
        assert responses[0].json() == responses[1].json()
    submits = [call for call in manager.calls if call[1] == "submit_clarification_reply"]
    assert len(submits) == 1, "exactly one executor may perform the M9 write"
    assert turns == [f"{session_id}-t1", f"{session_id}-r1"]


def test_reply_timeout_is_ambiguous_and_is_never_retried(tmp_path: Path) -> None:
    manager, app = _warm_fake_app(tmp_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        manager.errors["submit_clarification_reply"] = MCPToolTimeout(
            "submit_clarification_reply timed out", tool="submit_clarification_reply"
        )
        first = _reply(client, session_id, clarification_id)
        calls_after_first = len(manager.calls)
        second = _reply(client, session_id, clarification_id)
        calls_after_second = len(manager.calls)
        turns = _turns(tmp_path / "state.sqlite3", session_id)

    assert (first.status_code, second.status_code) == (503, 503), first.text
    assert first.json()["code"] == "clarification_reply_ambiguous"
    assert second.json()["code"] == "clarification_reply_ambiguous"
    assert calls_after_first == calls_after_second, "ambiguous never re-executes MCP"
    assert turns == [f"{session_id}-t1"]

    state = SQLiteWebStateStore(tmp_path / "state.sqlite3")
    try:
        pending = state.get_pending_for_clarification(session_id, clarification_id)
        assert pending is not None and pending.status == "OPEN"
        request = state.get_request(session_id, pending.reply_turn_id)
        assert request is not None
        assert (request.status, request.execution_stage) == ("FAILED", "AMBIGUOUS")
    finally:
        state.close()


def test_reset_fences_replay_of_the_old_request(tmp_path: Path) -> None:
    manager, app = _warm_fake_app(tmp_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        committed = _reply(client, session_id, clarification_id)
        assert committed.status_code == 200, committed.text
        reset = client.post(f"/api/sessions/{session_id}/reset")
        assert reset.status_code == 200
        calls_before = len(manager.calls)
        replayed = _reply(client, session_id, clarification_id)

    assert replayed.status_code == 409, replayed.text
    assert replayed.json()["code"] == "clarification_reply_fenced"
    assert len(manager.calls) == calls_before


def test_reply_from_another_session_cannot_use_the_handle(tmp_path: Path) -> None:
    manager, app = _warm_fake_app(tmp_path)
    with TestClient(app) as client:
        session_a = _create(client)
        _ask(client, session_a)
        clarification_id = client.get(f"/api/sessions/{session_a}").json()[
            "pending_clarification"
        ]["clarification_id"]
        session_b = _create(client)
        calls_before = len(manager.calls)
        response = _reply(client, session_b, clarification_id)
        pending_a = client.get(f"/api/sessions/{session_a}").json()["pending_clarification"]

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "clarification_not_pending"
    assert len(manager.calls) == calls_before
    assert pending_a is not None and pending_a["clarification_id"] == clarification_id
