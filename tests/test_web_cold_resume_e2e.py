"""M1-D S6: CPU/BM25 Cold Resume across a real MCP process restart.

The primary scenarios use the real BM25 MCP child, the real FastAPI app and isolated temporary
SQLite databases; the child PIDs are read from ``/proc`` to prove that process A really died
and that a different process B answered the reply.  The deterministic fake-manager tests at
the end inject the failure modes (corrupt/expired snapshots, races, crashes, durability) that
would be unsafe or too slow to force by killing a live process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.test_web_demo_unit as twu  # noqa: E402
from tests.test_web_reply_executor import _durable_result  # noqa: E402
from web_demo.app import create_app  # noqa: E402
from web_demo.config import Settings  # noqa: E402
from web_demo.continuation import (  # noqa: E402
    CONTINUATION_SCHEMA,
    CONTINUATION_VERSION,
    MCP_CONTINUATION_SCHEMA,
    build_web_continuation,
)
from web_demo.handle_lifecycle import HandleRef  # noqa: E402
from web_demo.mcp_manager import MCPToolError  # noqa: E402
from web_demo.state_store import SQLiteWebStateStore  # noqa: E402

PROFILE = "profile_0057"
USER_KEY = "p14e2_lite_0057"
ASK_QUERY = "I need some new headphones. Not sure what matters yet."
REPLY_TEXT = "I prefer noise cancelling."
COLD_TOOLS = [
    "restore_clarification_handle",
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


def _ppid(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        return -1
    return -1


def _mcp_child_pids() -> list[int]:
    """PIDs of this test process's real MCP children for the demo profile."""
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if _ppid(pid) != os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"recommendation.mcp.server" in cmdline and USER_KEY.encode() in cmdline:
            found.append(pid)
    return sorted(found)


def _wait_for_new_child(old_pids: list[int], timeout_s: float = 90.0) -> list[int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        current = _mcp_child_pids()
        if current and set(current).isdisjoint(old_pids):
            return current
        time.sleep(0.5)
    raise AssertionError(f"no new MCP child replaced {old_pids}: {_mcp_child_pids()}")


def _count_calls(manager) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    real = manager.call_tool

    async def counting(profile_id, tool, arguments=None, **kwargs):
        calls.append((tool, dict(arguments or {})))
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
    return clarification


def _reply(client: TestClient, session_id: str, clarification_id: str, message=REPLY_TEXT):
    return client.post(
        "/api/clarification/reply",
        json={
            "session_id": session_id,
            "clarification_id": clarification_id,
            "message": message,
        },
    )


def _m9_rows(memory_db: Path) -> list[tuple[str, str]]:
    if not memory_db.exists():
        return []
    connection = sqlite3.connect(str(memory_db))
    try:
        return [
            (str(user), str(value)) for user, value in connection.execute(
                "SELECT user_key, value FROM preference_memory WHERE status = 'active'"
            ).fetchall()
        ]
    finally:
        connection.close()


def _m9_count(memory_db: Path) -> int:
    if not memory_db.exists():
        return 0
    connection = sqlite3.connect(str(memory_db))
    try:
        return int(connection.execute("SELECT COUNT(*) FROM preference_memory").fetchone()[0])
    finally:
        connection.close()


def _pending_snapshot(db_path: Path, session_id: str, clarification_id: str):
    state = SQLiteWebStateStore(db_path)
    try:
        return state.get_pending_for_clarification(session_id, clarification_id)
    finally:
        state.close()


def _turns(db_path: Path, session_id: str) -> list[str]:
    state = SQLiteWebStateStore(db_path)
    try:
        return [row.turn_id for row in state.list_turns(session_id)]
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# real MCP: child restart, cold reply, replay, BFF restart
# --------------------------------------------------------------------------- #


def test_real_cold_resume_after_mcp_child_restart(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    memory_db = settings.memory_db_path
    assert db_path is not None and memory_db is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        manager = client.app.state.manager
        session_id = _create(client)
        pids_a = _mcp_child_pids()
        assert pids_a, "the real MCP child must be running"
        clarification = _ask(client, session_id)
        clarification_id = clarification["clarification_id"]

        # 4. Task/Pending/Continuation are durable as one ASK commit.
        pending = _pending_snapshot(db_path, session_id, clarification_id)
        assert pending is not None
        continuation = pending.continuation
        assert continuation is not None
        assert continuation["schema"] == CONTINUATION_SCHEMA
        assert continuation["version"] == CONTINUATION_VERSION
        assert continuation["session_id"] == session_id
        assert continuation["web_task_id"] == pending.task_id
        assert continuation["pending_id"] == pending.pending_id
        assert continuation["clarification_id"] == clarification_id
        assert continuation["reset_generation"] == 0
        assert continuation["expires_at"] == pending.expires_at
        assert continuation["mcp"]["task"]["task_id"] == USER_KEY
        assert continuation["mcp"]["current_query"] == ASK_QUERY
        assert continuation["mcp"]["session_id"] == session_id
        reply_turn_id = pending.reply_turn_id

        assert manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        ).ok, "the handle must be live before the restart"

        # 5-7. Kill child A, start child B, prove the old handle is dead.
        restart = client.post("/api/demo/restart", json={"profile_id": PROFILE})
        assert restart.status_code == 200, restart.text
        _wait_ready(client)
        pids_b = _wait_for_new_child(pids_a)
        assert set(pids_b).isdisjoint(pids_a), (pids_a, pids_b)
        old_handle = manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        )
        assert old_handle.code == "stale_process"

        # 8-12. Cold reply through the original session and pending.
        calls = _count_calls(manager)
        response = _reply(client, session_id, clarification_id)
        assert response.status_code == 200, response.text
        body = response.json()
        tools = [tool for tool, _ in calls]
        assert tools == COLD_TOOLS, tools
        assert calls[0][1]["session_id"] == session_id
        assert calls[0][1]["continuation"]["task"]["task_id"] == USER_KEY
        new_handle_id = calls[1][1]["clarification_id"]
        assert new_handle_id != clarification_id, "the old handle id must not be reused"
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, new_handle_id)
        ).ok, "the reply must use a handle issued by the new child"

        assert body["turn_id"] == reply_turn_id
        assert body["clarification_id"] == clarification_id
        assert body["clarification"] is None
        assert body["message"].startswith(f"Memory update ({PROFILE})")
        assert body["memory_update"]["added"] == 1
        assert "noise cancelling" in [
            row["value"] for row in body["preferences"]["active"]
        ]
        recommendation = body["recommendation"]
        recommendation_id = recommendation["recommendation_id"]
        assert recommendation_id.startswith("rec-")
        assert recommendation["request_id"] == reply_turn_id
        card = recommendation["items"][0]
        assert card["product_id"] == recommendation["recommended_product_id"]
        assert card["title"]
        assert card["source"] == "recommendation"

        # 13. SQLite: Task/Pending/Request/Reply Turn.
        state = SQLiteWebStateStore(db_path)
        try:
            consumed = state.get_pending_for_clarification(session_id, clarification_id)
            assert consumed is not None and consumed.status == "CONSUMED"
            request = state.get_request(session_id, reply_turn_id)
            assert request is not None
            assert (request.status, request.execution_stage) == ("COMPLETED", "COMPLETED")
            stored = request.result["payload"]
            assert stored["reply"]["recommendation"]["id"] == recommendation_id
            assert stored["reply"]["products"]["items"][0]["product_id"] == card["product_id"]
            task = state.get_task(request.task_id)
            assert task is not None and task.status == "COMPLETED"
            assert [row.turn_id for row in state.list_turns(session_id)] == [
                f"{session_id}-t1", reply_turn_id,
            ]
        finally:
            state.close()

        # 9. M9 really holds the preference under the Web session key.
        assert (session_id, "noise cancelling") in _m9_rows(memory_db)
        m9_before_replay = _m9_count(memory_db)

        # 14. Replay: identical result, zero MCP, zero duplicate M9 event.
        calls_before_replay = len(calls)
        replay = _reply(client, session_id, clarification_id)
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
        assert len(calls) == calls_before_replay, "a replay must not touch MCP"
        assert _m9_count(memory_db) == m9_before_replay

    # 15. BFF restart: the same Reply Turn / product / recommendation id are restored.
    with TestClient(create_app(settings=settings)) as client2:
        _wait_ready(client2)
        calls2 = _count_calls(client2.app.state.manager)
        view = client2.get(f"/api/sessions/{session_id}").json()
        assert [turn["turn_id"] for turn in view["turns"]] == [
            f"{session_id}-t1", reply_turn_id,
        ]
        assert view["pending_clarification"] is None
        restored = view["turns"][-1]
        assert restored["products"]["recommendation_id"] == recommendation_id
        assert restored["products"]["items"][0]["product_id"] == card["product_id"]
        assert restored["memory_update"]["added"] == 1

        replayed = _reply(client2, session_id, clarification_id)
        assert replayed.status_code == 200, replayed.text
        assert replayed.json()["turn_id"] == reply_turn_id
        assert replayed.json()["recommendation"]["recommendation_id"] == recommendation_id
        assert calls2 == [], "a completed reply must replay without any MCP call"


def test_real_cold_resume_after_bff_and_mcp_restart(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    db_path = settings.web_state_db_path
    memory_db = settings.memory_db_path
    assert db_path is not None and memory_db is not None

    with TestClient(create_app(settings=settings)) as client:
        _wait_ready(client)
        session_id = _create(client)
        pids_a = _mcp_child_pids()
        assert pids_a
        clarification = _ask(client, session_id)
        clarification_id = clarification["clarification_id"]
        reply_turn_id = _pending_snapshot(
            db_path, session_id, clarification_id
        ).reply_turn_id
    # Both the BFF and its MCP children are gone now.
    assert _mcp_child_pids() == []

    with TestClient(create_app(settings=settings)) as client2:
        _wait_ready(client2)
        manager = client2.app.state.manager
        pids_b = _mcp_child_pids()
        assert pids_b and set(pids_b).isdisjoint(pids_a), (pids_a, pids_b)
        assert manager.check_handle(
            HandleRef(PROFILE, session_id, clarification_id)
        ).code == "handle_missing"

        calls = _count_calls(manager)
        response = _reply(client2, session_id, clarification_id)
        assert response.status_code == 200, response.text
        body = response.json()
        assert [tool for tool, _ in calls] == COLD_TOOLS
        assert body["turn_id"] == reply_turn_id
        assert body["recommendation"]["items"]
        assert (session_id, "noise cancelling") in _m9_rows(memory_db)

    state = SQLiteWebStateStore(db_path)
    try:
        request = state.get_request(session_id, reply_turn_id)
        assert request is not None and request.status == "COMPLETED"
        assert state.get_open_pending_for_session(session_id) is None
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# deterministic fail-closed / race / crash coverage (fake MCP transport)
# --------------------------------------------------------------------------- #


def _inject_continuation(
    db_path: Path,
    session_id: str,
    clarification_id: str,
    *,
    overrides: dict | None = None,
    expired: bool = False,
) -> None:
    """Write a valid Cold Resume snapshot for a fake-manager ASK into the real SQLite row."""
    state = SQLiteWebStateStore(db_path)
    try:
        pending = state.get_pending_for_clarification(session_id, clarification_id)
        record = state.get_session(session_id)
        assert pending is not None and record is not None
        expires_at = pending.expires_at
        if expired:
            expires_at = time.time() - 10.0
        mcp = {
            "schema": MCP_CONTINUATION_SCHEMA,
            "version": 1,
            "mode": "cpu_live",
            "user_key": USER_KEY,
            "session_id": session_id,
            "clarification_id": clarification_id,
            "question": "What matters most for this purchase?",
            "reason": "missing_key_need",
            "current_query": pending.query,
            "task": {"task_id": USER_KEY, "current_query": pending.query},
            "task_constraints": {"hard_exclusions": [], "soft_preferences": [],
                                 "source_turns": []},
            "created_at": expires_at - 900.0,
            "expires_at": expires_at,
        }
        snapshot = build_web_continuation(
            mcp_continuation=mcp, query=pending.query, profile_id=PROFILE
        )
        snapshot.update({
            "session_id": session_id,
            "web_task_id": pending.task_id,
            "pending_id": pending.pending_id,
            "clarification_id": clarification_id,
            "reset_generation": record.reset_generation,
            "expires_at": expires_at,
        })
        snapshot.update(overrides or {})
        pending_id = pending.pending_id
    finally:
        state.close()

    raw = sqlite3.connect(str(db_path))
    try:
        if expired:
            # created_at must move first: the row CHECK requires expires_at > created_at.
            raw.execute(
                "UPDATE web_pending SET created_at = ? WHERE pending_id = ?",
                (expires_at - 100.0, pending_id),
            )
        raw.execute(
            "UPDATE web_pending SET continuation_json = ?, expires_at = ? "
            "WHERE pending_id = ?",
            (json.dumps(snapshot, sort_keys=True), expires_at, pending_id),
        )
        raw.commit()
    finally:
        raw.close()


def _install_cold_restore(manager: twu.FakeManager, session_id: str) -> None:
    counter = {"n": 0}

    def restore(profile_id: str, arguments: dict) -> dict:  # noqa: ARG001
        counter["n"] += 1
        return {
            "schema": MCP_CONTINUATION_SCHEMA,
            "status": "restored",
            "clarification_id": f"clar_restored_{counter['n']}",
            "restored_from_clarification_id": arguments["continuation"]["clarification_id"],
            "question": "What matters most for this purchase?",
            "reason": "missing_key_need",
            "expires_in_seconds": 600.0,
            "authorized_user_key": USER_KEY,
            "session_id": session_id,
            "memory_write_enabled": True,
        }

    manager.responses["restore_clarification_handle"] = restore


def _cold_ready(client: TestClient, manager: twu.FakeManager, tmp_path: Path) -> tuple[str, str]:
    """Prepare a fake app whose pending has a valid continuation and a stale handle."""
    session_id = _create(client)
    _ask(client, session_id)
    clarification_id = client.get(f"/api/sessions/{session_id}").json()[
        "pending_clarification"
    ]["clarification_id"]
    _inject_continuation(tmp_path / "state.sqlite3", session_id, clarification_id)
    _install_cold_restore(manager, session_id)
    asyncio.run(manager.restart(PROFILE))  # invalidate the in-process handle
    return session_id, clarification_id


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"schema": "agentrecx.web.clarification_continuation.v0"},
         "continuation_unsupported_schema"),
        ({"version": 99}, "continuation_unsupported_version"),
        ({"session_id": "web-someone-else"}, "continuation_session_mismatch"),
        ({"reset_generation": 7}, "continuation_generation_mismatch"),
    ],
)
def test_corrupt_continuation_is_refused_before_any_mcp(
    tmp_path: Path, overrides: dict, code: str
) -> None:
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        _inject_continuation(
            tmp_path / "state.sqlite3", session_id, clarification_id,
            overrides=overrides,
        )
        _install_cold_restore(manager, session_id)
        asyncio.run(manager.restart(PROFILE))
        calls_before = len(manager.calls)
        response = _reply(client, session_id, clarification_id)
        pending = _pending_snapshot(
            tmp_path / "state.sqlite3", session_id, clarification_id
        )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "clarification_resume_not_ready"
    assert response.json()["detail"]["handle"]["code"] == code
    assert len(manager.calls) == calls_before, "a corrupt snapshot must not reach MCP"
    assert pending is not None and pending.status == "OPEN"


def test_expired_continuation_is_refused_before_any_mcp(tmp_path: Path) -> None:
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        _inject_continuation(
            tmp_path / "state.sqlite3", session_id, clarification_id, expired=True
        )
        _install_cold_restore(manager, session_id)
        asyncio.run(manager.restart(PROFILE))
        calls_before = len(manager.calls)
        response = _reply(client, session_id, clarification_id)

    assert response.status_code == 409
    assert response.json()["detail"]["handle"]["code"] == "continuation_expired"
    assert len(manager.calls) == calls_before


def test_cold_restore_failure_is_refused_without_consuming_the_pending(
    tmp_path: Path,
) -> None:
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        _inject_continuation(tmp_path / "state.sqlite3", session_id, clarification_id)
        manager.responses["restore_clarification_handle"] = MCPToolError(
            "restore failed", tool="restore_clarification_handle"
        )
        asyncio.run(manager.restart(PROFILE))
        response = _reply(client, session_id, clarification_id)
        pending = _pending_snapshot(
            tmp_path / "state.sqlite3", session_id, clarification_id
        )

    assert response.status_code == 409
    assert response.json()["detail"]["handle"]["code"] == "cold_resume_failed"
    assert pending is not None and pending.status == "OPEN"
    # The failed restore never wrote M9 or started a request.
    assert [call for call in manager.calls if call[1] == "submit_clarification_reply"] == []


def test_two_concurrent_cold_replies_have_exactly_one_executor(tmp_path: Path) -> None:
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id, clarification_id = _cold_ready(client, manager, tmp_path)
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

    state = SQLiteWebStateStore(tmp_path / "state.sqlite3")
    try:
        turns = [row.turn_id for row in state.list_turns(session_id)]
    finally:
        state.close()

    assert len(responses) == 2
    statuses = sorted(response.status_code for response in responses)
    assert 200 in statuses, [response.text for response in responses]
    assert set(statuses) <= {200, 409, 503}
    submits = [call for call in manager.calls if call[1] == "submit_clarification_reply"]
    assert len(submits) == 1, "only one executor may perform the M9 write"
    assert turns == [f"{session_id}-t1", f"{session_id}-r1"]


def test_crash_after_external_execution_stays_ambiguous_after_restart(
    tmp_path: Path,
) -> None:
    manager = twu.warm_reply_manager()
    calls = {"recommend": 0}

    def recommend(profile_id, arguments):  # noqa: ARG001
        calls["recommend"] += 1
        if calls["recommend"] == 1:
            return twu.recommend_payload(status="awaiting_clarification")
        raise MCPToolError("recommend_products crashed", tool="recommend_products")

    manager.responses["recommend_products"] = recommend
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        first = _reply(client, session_id, clarification_id)
        turns = _turns(tmp_path / "state.sqlite3", session_id)

    assert first.status_code == 503, first.text
    assert first.json()["code"] == "clarification_reply_ambiguous"
    assert turns == [f"{session_id}-t1"]

    # A restarted BFF must not re-execute the uncertain external call.
    manager2 = twu.warm_reply_manager()
    with TestClient(create_app(settings=_fake_settings(tmp_path), manager=manager2)) as client2:
        calls_before = len(manager2.calls)
        retry = _reply(client2, session_id, clarification_id)
    assert retry.status_code == 503
    assert retry.json()["code"] == "clarification_reply_ambiguous"
    assert len(manager2.calls) == calls_before, "AMBIGUOUS never re-executes MCP"
    state = SQLiteWebStateStore(tmp_path / "state.sqlite3")
    try:
        pending = state.get_pending_for_clarification(session_id, clarification_id)
        assert pending is not None and pending.status == "OPEN"
        request = state.get_request(session_id, pending.reply_turn_id)
        assert request is not None
        assert (request.status, request.execution_stage) == ("FAILED", "AMBIGUOUS")
    finally:
        state.close()


def test_result_durable_after_restart_finalizes_without_any_mcp(tmp_path: Path) -> None:
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_id = _create(client)
        _ask(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]

    store = SQLiteWebStateStore(tmp_path / "state.sqlite3")
    try:
        pending = store.get_pending_for_clarification(session_id, clarification_id)
        assert pending is not None
        claim = store.claim_reply_request(
            session_id, pending.pending_id,
            payload={"message": REPLY_TEXT, "confirmed_by_user": True,
                     "confirmation_source": "web_ui_explicit_user_reply"},
        )
        store.mark_request_external_started(
            session_id, pending.reply_turn_id, claim_token=claim.claim_token
        )
        store.checkpoint_reply_result(
            session_id, pending.reply_turn_id, claim_token=claim.claim_token,
            result=_durable_result(pending),
        )
        raw = sqlite3.connect(str(tmp_path / "state.sqlite3"))
        try:
            raw.execute("UPDATE web_request SET lease_expires_at = 0")
            raw.commit()
        finally:
            raw.close()
    finally:
        store.close()

    manager2 = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager2)
    with TestClient(create_app(settings=_fake_settings(tmp_path), manager=manager2)) as client2:
        calls = _count_calls(manager2)
        response = _reply(client2, session_id, clarification_id)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recommendation"]["recommendation_id"] == "rec-checkpoint"
    assert calls == [], "a durable checkpoint finalizes without MCP"
    assert _turns(tmp_path / "state.sqlite3", session_id) == [
        f"{session_id}-t1", f"{session_id}-r1",
    ]


def test_reset_and_cross_session_still_block_cold_replies(tmp_path: Path) -> None:
    manager = twu.warm_reply_manager()
    app = create_app(settings=_fake_settings(tmp_path), manager=manager)
    with TestClient(app) as client:
        session_a = _create(client)
        _ask(client, session_a)
        clarification_id = client.get(f"/api/sessions/{session_a}").json()[
            "pending_clarification"
        ]["clarification_id"]
        session_b = _create(client)
        calls_before = len(manager.calls)
        cross = _reply(client, session_b, clarification_id)
        reset = client.post(f"/api/sessions/{session_a}/reset")
        assert reset.status_code == 200
        after_reset = _reply(client, session_a, clarification_id)

    assert cross.status_code == 409
    assert cross.json()["code"] == "clarification_not_pending"
    assert after_reset.status_code == 409
    assert after_reset.json()["code"] == "clarification_not_pending"
    assert len(manager.calls) == calls_before, "no MCP call for a refused reply"
