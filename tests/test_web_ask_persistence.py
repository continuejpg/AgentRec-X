"""FastAPI-level ASK durability tests for M1-C S3b."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.test_web_demo_unit as twu  # noqa: E402
from web_demo.app import create_app  # noqa: E402
from web_demo.config import Settings  # noqa: E402
from web_demo.sessions import DEFAULT_TITLE  # noqa: E402
from web_demo.state_store import (  # noqa: E402
    _SCHEMA_V1_STATEMENTS,
    _SCHEMA_V2_STATEMENTS,
    WEB_STATE_SCHEMA_VERSION,
    SQLiteWebStateStore,
    StateMigrationError,
    TaskStateConflict,
    StaleTaskVersion,
)

CHILD = textwrap.dedent(
    """
    import json
    import os
    import sys
    import tempfile
    from pathlib import Path

    repo, action, db = sys.argv[1], sys.argv[2], sys.argv[3]
    session_id = sys.argv[4] if len(sys.argv) > 4 else ""
    clarification_id = sys.argv[5] if len(sys.argv) > 5 else ""
    sys.path.insert(0, repo)

    from fastapi.testclient import TestClient
    import tests.test_web_demo_unit as twu
    from web_demo.app import create_app
    from web_demo.config import Settings
    from web_demo.state_store import SQLiteWebStateStore

    manager = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager)
    manager.responses.update({
        "recommend_products": twu.recommend_payload(status="awaiting_clarification"),
        "submit_clarification_reply": twu.reply_payload(),
        "get_active_preferences": twu.preferences_payload(),
    })
    tmp = Path(tempfile.mkdtemp(prefix="ask_child_"))
    settings = Settings(search_mode="bm25", allow_memory_writes=True,
                        log_dir=tmp / "logs", web_state_db_path=Path(db),
                        python_executable=sys.executable, mcp_timeout_s=30.0)
    app = create_app(settings=settings, manager=manager)

    def introspect(state, sid):
        task = state.get_active_task(sid)
        pending = state.get_open_pending_for_session(sid) if task else None
        turns = state.list_turns(sid)
        return {
            "task_id": None if task is None else task.task_id,
            "task_status": None if task is None else task.status,
            "task_version": None if task is None else task.task_version,
            "pending_id": None if pending is None else pending.pending_id,
            "reply_turn_id": None if pending is None else pending.reply_turn_id,
            "clarification_id": None if pending is None else pending.clarification_id,
            "payload": None if pending is None else pending.payload,
            "turn_ids": [row.turn_id for row in turns],
        }

    with TestClient(app) as client:
        if action == "ask":
            sid = client.post("/api/sessions", json={"profile_id": "profile_0057"}).json()["session_id"]
            chat = client.post("/api/chat/stream", json={"session_id": sid,
                              "message": "I need some new headphones."})
            events = twu.parse_sse(chat.text)
            view = client.get(f"/api/sessions/{sid}").json()
            state = SQLiteWebStateStore(db)
            try:
                info = introspect(state, sid)
                info["ledger"] = [r[0] for r in state._connection.execute(
                    "SELECT value FROM web_identity WHERE session_id=? ORDER BY kind, seq", (sid,))]
            finally:
                state.close()
            payload = {"pid": os.getpid(), "session_id": sid,
                       "events": [e["event"] for e in events],
                       "view_turns": len(view["turns"]),
                       "view_pending": view["pending_clarification"],
                       **info}
        elif action == "read":
            view = client.get(f"/api/sessions/{session_id}").json()
            state = SQLiteWebStateStore(db)
            try:
                info = introspect(state, session_id)
                info["ledger"] = [r[0] for r in state._connection.execute(
                    "SELECT value FROM web_identity WHERE session_id=? ORDER BY kind, seq",
                    (session_id,))]
            finally:
                state.close()
            payload = {"pid": os.getpid(), "session_id": session_id,
                       "view_turns": len(view["turns"]),
                       "view_pending": view["pending_clarification"],
                       "manager_calls": len(manager.calls), **info}
        elif action == "reply":
            response = client.post("/api/clarification/reply", json={
                "session_id": session_id, "clarification_id": clarification_id,
                "message": "For gaming."})
            state = SQLiteWebStateStore(db)
            try:
                info = introspect(state, session_id)
                info["ledger"] = [r[0] for r in state._connection.execute(
                    "SELECT value FROM web_identity WHERE session_id=? ORDER BY kind, seq",
                    (session_id,))]
            finally:
                state.close()
            body = response.json()
            payload = {"pid": os.getpid(), "status": response.status_code,
                       "code": body.get("code"), "manager_calls": len(manager.calls), **info}
        else:
            raise SystemExit(f"unknown action {action!r}")
    print(json.dumps(payload))
    """
)


def _manager(finalize: bool = False) -> twu.FakeManager:
    manager = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager)
    manager.responses.update(
        {
            "recommend_products": twu.recommend_payload(
                status="finalized" if finalize else "awaiting_clarification"
            ),
            "submit_clarification_reply": twu.reply_payload(),
            "get_active_preferences": twu.preferences_payload(),
        }
    )
    return manager


def _app(tmp_path: Path, db_path: Path | None, *, finalize: bool = False):
    settings = Settings(
        search_mode="bm25", allow_memory_writes=True, log_dir=tmp_path / "logs",
        web_state_db_path=db_path, python_executable=sys.executable, mcp_timeout_s=30.0,
    )
    return create_app(settings=settings, manager=_manager(finalize))


def _create(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"profile_id": "profile_0057"})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def _chat(client: TestClient, session_id: str, message: str = "I need some new headphones."):
    response = client.post("/api/chat/stream", json={"session_id": session_id, "message": message})
    assert response.status_code == 200, response.text
    return twu.parse_sse(response.text)


def _run_child(tmp_path: Path, action: str, db_path: Path, *args: str) -> dict:
    script = tmp_path / "ask_child.py"
    script.write_text(CHILD, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(script), str(REPO_ROOT), action, str(db_path), *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240,
                            env=environment, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# A: real HTTP ASK is persisted
# --------------------------------------------------------------------------- #


def test_http_ask_persists_task_turn_pending_and_one_reply_id(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        session_id = _create(client)
        events = _chat(client, session_id)
        assert "clarification" in [event["event"] for event in events]
        view = client.get(f"/api/sessions/{session_id}").json()

    state = SQLiteWebStateStore(db_path)
    try:
        turns = state.list_turns(session_id)
        assert len(turns) == 1
        assert turns[0].turn_id == f"{session_id}-t1"
        task = state.get_active_task(session_id)
        assert task is not None and task.status == "WAITING_FOR_USER"
        assert task.task_id != session_id
        pending = state.get_open_pending_for_session(session_id)
        assert pending is not None and pending.status == "OPEN"
        assert pending.turn_id == turns[0].turn_id
        assert pending.clarification_id == "clar_fake_handle"
        assert pending.payload["question"].startswith("What matters most")
        session = state.get_session(session_id)
        assert session is not None and session.reply_counter == 1
    finally:
        state.close()

    # The BFF view and the durable pending are the same clarification and reply id.
    assert view["pending_clarification"]["clarification_id"] == "clar_fake_handle"
    assert view["turns"][0]["turn_id"] == f"{session_id}-t1"
    ledger = _ledger(db_path, session_id)
    assert ledger.count(f"{session_id}-r1") == 1
    assert ledger.count(f"{session_id}-r2") == 0


def _ledger(db_path: Path, session_id: str) -> list[str]:
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT value FROM web_identity WHERE session_id = ?", (session_id,)
        )
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# B: a failed ASK commit must not be reported to the browser
# --------------------------------------------------------------------------- #


def test_failed_ask_commit_emits_no_clarification_and_leaves_no_ghost(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute(
            "CREATE TRIGGER fail_pending BEFORE INSERT ON web_pending "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        raw.commit()
    finally:
        raw.close()

    with TestClient(app) as client:
        session_id = _create(client)
        events = _chat(client, session_id)
        names = [event["event"] for event in events]
        assert "clarification" not in names, "an uncommitted ASK must not reach the browser"
        assert "error" in names

    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("DROP TRIGGER fail_pending")
        raw.commit()
        pending_rows = raw.execute("SELECT COUNT(*) FROM web_pending").fetchone()[0]
    finally:
        raw.close()

    state = SQLiteWebStateStore(db_path)
    try:
        assert pending_rows == 0
        assert state.get_active_task(session_id) is None
        tasks = state.list_tasks(session_id)
        assert len(tasks) == 1 and tasks[0].status == "FAILED"
        session = state.get_session(session_id)
        assert session is not None and session.reply_counter == 0
        turns = state.list_turns(session_id)
        assert len(turns) == 1, "the failed turn is recorded as an error, not as an ASK"
        assert turns[0].payload["turn"]["error"]
        assert "clarification" not in (turns[0].payload["turn"].get("clarification") or {})
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# C: recovery in a new OS process
# --------------------------------------------------------------------------- #


def test_ask_recovers_in_new_os_process(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    written = _run_child(tmp_path, "ask", db_path)
    assert written["pid"] != os.getpid()
    assert written["task_status"] == "WAITING_FOR_USER"
    assert written["turn_ids"] == [f"{written['session_id']}-t1"]
    assert written["reply_turn_id"] == f"{written['session_id']}-r1"

    read = _run_child(tmp_path, "read", db_path, written["session_id"])
    assert read["pid"] != written["pid"]
    assert read["manager_calls"] == 0, "recovery must not call the planner or MCP"
    for field in ("task_id", "task_status", "task_version", "pending_id",
                  "reply_turn_id", "clarification_id", "payload", "turn_ids", "ledger"):
        assert read[field] == written[field], f"{field} changed across processes"
    assert read["view_turns"] == written["view_turns"] == 1
    assert read["view_pending"]["clarification_id"] == "clar_fake_handle"
    assert len(read["ledger"]) == 2  # exactly one turn id and one reply id


# --------------------------------------------------------------------------- #
# D: ordinary chat history is persisted
# --------------------------------------------------------------------------- #


def test_ordinary_chat_history_is_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path, finalize=True)
    with TestClient(app) as client:
        session_id = _create(client)
        events = _chat(client, session_id, "wireless earbuds")
        assert "clarification" not in [event["event"] for event in events]

    state = SQLiteWebStateStore(db_path)
    try:
        turns = state.list_turns(session_id)
        assert len(turns) == 1
        assert turns[0].payload["turn"]["products"]["items"]
        task = state.get_active_task(session_id)
        assert task is None, "a finalized chat must close its task"
    finally:
        state.close()

    app2 = _app(tmp_path, db_path)
    with TestClient(app2) as client2:
        view = client2.get(f"/api/sessions/{session_id}").json()
        assert len(view["turns"]) == 1
        assert view["turns"][0]["products"]["items"]
        assert view["pending_clarification"] is None


# --------------------------------------------------------------------------- #
# E: persistent reset clears history atomically
# --------------------------------------------------------------------------- #


def test_persistent_reset_clears_history_and_keeps_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _chat(client, session_id)
        before = client.get(f"/api/sessions/{session_id}").json()
        response = client.post(f"/api/sessions/{session_id}/reset")
        assert response.status_code == 200
        view = response.json()
        assert view["session_id"] == session_id
        assert view["turns"] == []
        assert view["pending_clarification"] is None
        assert view["title"] == DEFAULT_TITLE

    state = SQLiteWebStateStore(db_path)
    try:
        assert state.list_turns(session_id) == ()
        assert state.get_active_task(session_id) is None
        pending = state.get_open_pending_for_session(session_id)
        assert pending is None
        session = state.get_session(session_id)
        assert session is not None
        assert session.reset_generation == 1
        assert (session.turn_counter, session.reply_counter) == (1, 1)
    finally:
        state.close()
    assert sorted(_ledger(db_path, session_id)) == [
        f"{session_id}-r1", f"{session_id}-t1"
    ]

    app2 = _app(tmp_path, db_path)
    with TestClient(app2) as client2:
        restored = client2.get(f"/api/sessions/{session_id}").json()
        assert restored["turns"] == []
        assert restored["pending_clarification"] is None
        assert restored["title"] == DEFAULT_TITLE


# --------------------------------------------------------------------------- #
# F: a restored pending cannot be answered
# --------------------------------------------------------------------------- #


def test_restored_pending_reply_is_blocked_without_side_effects(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    written = _run_child(tmp_path, "ask", db_path)
    replied = _run_child(
        tmp_path, "reply", db_path, written["session_id"], written["clarification_id"]
    )

    assert replied["status"] == 409
    assert replied["code"] == "clarification_resume_not_ready"
    assert replied["manager_calls"] == 0, "no MCP handle may be contacted"
    assert replied["pending_id"] == written["pending_id"]
    assert replied["task_version"] == written["task_version"]
    assert replied["ledger"] == written["ledger"], "no identity may be issued on refusal"


# --------------------------------------------------------------------------- #
# G: TTL is absolute and does not strand the task
# --------------------------------------------------------------------------- #


def test_expired_pending_is_not_restored_and_task_is_released(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _chat(client, session_id)

    raw = sqlite3.connect(str(db_path))
    try:
        now = time.time()
        raw.execute(
            "UPDATE web_pending SET created_at = ?, expires_at = ?",
            (now - 120, now - 60),
        )
        raw.commit()
    finally:
        raw.close()

    app2 = _app(tmp_path, db_path)
    with TestClient(app2) as client2:
        view = client2.get(f"/api/sessions/{session_id}").json()
        assert view["pending_clarification"] is None

    state = SQLiteWebStateStore(db_path)
    try:
        assert state.get_open_pending_for_session(session_id) is None
        tasks = state.list_tasks(session_id)
        assert len(tasks) == 1 and tasks[0].status == "CANCELLED"
        assert tasks[0].close_reason == "clarification_expired"
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# H: stale instance cannot commit after another reset
# --------------------------------------------------------------------------- #


def test_stale_ask_cannot_reopen_after_reset(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    first = SQLiteWebStateStore(db_path)
    second = SQLiteWebStateStore(db_path)
    try:
        first.create_session(session_id="web-race", profile_id="profile_0057")
        task = first.create_task("web-race", initial_requirements={"raw_query": "x"})
        turn_id = first.allocate_turn_id("web-race")

        second.reset_session("web-race")  # another instance resets while A is mid-ask

        with pytest.raises((TaskStateConflict, StaleTaskVersion)):
            first.commit_ask(
                "web-race", task.task_id, expected_task_version=task.task_version,
                clarification_id="clar-race", source_turn_id=turn_id,
                query="q", payload={"question": "?"}, expires_at=time.time() + 600,
            )
        assert first.get_active_task("web-race") is None
    finally:
        first.close()
        second.close()


# --------------------------------------------------------------------------- #
# I: v2 -> v3 migration
# --------------------------------------------------------------------------- #


def _build_v2_database(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    try:
        for statement in _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS:
            connection.execute(statement)
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '2')")
        connection.execute(
            "INSERT INTO web_session(session_id, profile_id, title, created_at, updated_at, "
            "behavior_enabled, turn_counter, reply_counter, reset_generation, retired) "
            "VALUES ('web-v2', 'profile_0057', 'v2 session', 10.0, 20.0, 1, 1, 1, 0, 0)"
        )
        connection.execute(
            "INSERT INTO web_identity(session_id, kind, seq, value, created_at) "
            "VALUES ('web-v2', 'turn', 1, 'web-v2-t1', 10.0), "
            "('web-v2', 'reply', 1, 'web-v2-r1', 10.0)"
        )
        connection.execute(
            "INSERT INTO web_task(task_id, session_id, status, task_version, reset_generation, "
            "current_requirements_json, requirements_version, created_at, updated_at) "
            "VALUES ('task-v2', 'web-v2', 'WAITING_FOR_USER', 2, 0, '{\"raw_query\": \"q\"}', 0, 10.0, 11.0)"
        )
        connection.execute(
            "INSERT INTO web_pending(pending_id, task_id, clarification_id, query, turn_id, "
            "reply_turn_id, mode, payload_json, status, created_at, expires_at, consumed_at) "
            "VALUES ('pend-v2', 'task-v2', 'clar-v2', 'q', 'web-v2-t1', 'web-v2-r1', "
            "'cpu_live', '{\"question\": \"?\"}', 'OPEN', 10.0, 9999999999.0, NULL)"
        )
        connection.commit()
    finally:
        connection.close()


def test_v2_database_migrates_to_v3_preserving_state(tmp_path: Path) -> None:
    db_path = tmp_path / "v2.sqlite3"
    _build_v2_database(db_path)

    store = SQLiteWebStateStore(db_path)
    try:
        assert store.schema_version == WEB_STATE_SCHEMA_VERSION == 6
        sessions = store.list_sessions(10)
        assert [row.session_id for row in sessions] == ["web-v2"]
        task = store.get_task("task-v2")
        assert task is not None and task.status == "WAITING_FOR_USER"
        pending = store.get_open_pending("task-v2")
        assert pending is not None and pending.clarification_id == "clar-v2"
        assert store.list_turns("web-v2") == ()
    finally:
        store.close()

    raw = sqlite3.connect(str(db_path))
    try:
        assert raw.execute("SELECT COUNT(*) FROM web_identity").fetchone()[0] == 2
        assert raw.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == str(WEB_STATE_SCHEMA_VERSION)
    finally:
        raw.close()


def test_failed_v3_migration_rolls_back(tmp_path: Path) -> None:
    db_path = tmp_path / "v2_broken.sqlite3"
    _build_v2_database(db_path)
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("CREATE TABLE web_turn (x INTEGER)")  # incompatible placeholder
        raw.commit()
    finally:
        raw.close()

    with pytest.raises(StateMigrationError):
        SQLiteWebStateStore(db_path)

    raw = sqlite3.connect(str(db_path))
    try:
        assert raw.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "2"
        assert raw.execute("SELECT COUNT(*) FROM web_task").fetchone()[0] == 1
        assert raw.execute("SELECT COUNT(*) FROM web_pending").fetchone()[0] == 1
    finally:
        raw.close()


# --------------------------------------------------------------------------- #
# J: memory mode is unchanged
# --------------------------------------------------------------------------- #


def test_memory_mode_ask_and_reply_still_work(tmp_path: Path) -> None:
    app = _app(tmp_path, None)
    with TestClient(app) as client:
        session_id = _create(client)
        events = _chat(client, session_id)
        assert "clarification" in [event["event"] for event in events]
        reply = client.post(
            "/api/clarification/reply",
            json={"session_id": session_id, "clarification_id": "clar_fake_handle",
                  "message": "For gaming."},
        )
        assert reply.status_code == 200, reply.text
        view = client.get(f"/api/sessions/{session_id}").json()
        assert len(view["turns"]) == 2
        assert view["pending_clarification"] is None
