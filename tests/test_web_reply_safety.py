"""Fail-closed clarification replies and same-session concurrency (M1-C S3b.1)."""

from __future__ import annotations

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

from tests.test_web_ask_persistence import (  # noqa: E402
    _app,
    _chat,
    _create,
    _run_child,
)
from web_demo.persistent_sessions import PersistentSessionStore  # noqa: E402
from web_demo.state_store import (  # noqa: E402
    SQLiteWebStateStore,
    TaskStateConflict,
)

REPLY_CODE = "clarification_resume_not_ready"


def _reply(client: TestClient, session_id: str, clarification_id: str):
    return client.post(
        "/api/clarification/reply",
        json={
            "session_id": session_id,
            "clarification_id": clarification_id,
            "message": "For gaming.",
        },
    )


def _snapshot(db_path: Path, session_id: str) -> dict:
    state = SQLiteWebStateStore(db_path)
    try:
        task = state.get_active_task(session_id)
        pending = state.get_open_pending_for_session(session_id)
        session = state.get_session(session_id)
        return {
            "task_status": None if task is None else task.status,
            "task_version": None if task is None else task.task_version,
            "pending_status": None if pending is None else pending.status,
            "pending_id": None if pending is None else pending.pending_id,
            "reply_counter": None if session is None else session.reply_counter,
            "turn_counter": None if session is None else session.turn_counter,
            "turns": [row.turn_id for row in state.list_turns(session_id)],
        }
    finally:
        state.close()


def _ledger(db_path: Path) -> list[str]:
    connection = sqlite3.connect(str(db_path))
    try:
        return [str(row[0]) for row in connection.execute("SELECT value FROM web_identity")]
    finally:
        connection.close()


def _counts(db_path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(str(db_path))
    try:
        open_pending = connection.execute(
            "SELECT COUNT(*) FROM web_pending WHERE status='OPEN'"
        ).fetchone()[0]
        active = connection.execute(
            "SELECT COUNT(*) FROM web_task WHERE status IN ('RUNNING','WAITING_FOR_USER')"
        ).fetchone()[0]
        return int(open_pending), int(active)
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# Persistent warm reply is refused (the P0 inconsistency)
# --------------------------------------------------------------------------- #


def test_persistent_warm_reply_is_refused_without_side_effects(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        session_id = _create(client)
        events = _chat(client, session_id)
        assert "clarification" in [event["event"] for event in events]
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]

        before = _snapshot(db_path, session_id)
        ledger_before = _ledger(db_path)
        calls_before = len(app.state.manager.calls)

        response = _reply(client, session_id, clarification_id)

    assert response.status_code == 409
    assert response.json()["code"] == REPLY_CODE
    assert len(app.state.manager.calls) == calls_before, "no MCP call may be made"
    assert _snapshot(db_path, session_id) == before
    assert _ledger(db_path) == ledger_before
    assert before["pending_status"] == "OPEN"
    assert before["task_status"] == "WAITING_FOR_USER"


def test_persistent_restored_reply_is_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    written = _run_child(tmp_path, "ask", db_path)

    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        before = _snapshot(db_path, written["session_id"])
        response = _reply(client, written["session_id"], written["clarification_id"])

    assert response.status_code == 409
    assert response.json()["code"] == REPLY_CODE
    assert app.state.manager.calls == []
    assert _snapshot(db_path, written["session_id"]) == before


def test_duplicate_persistent_reply_stays_refused(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _chat(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        before = _snapshot(db_path, session_id)
        first = _reply(client, session_id, clarification_id)
        second = _reply(client, session_id, clarification_id)

    assert (first.status_code, second.status_code) == (409, 409)
    assert first.json()["code"] == second.json()["code"] == REPLY_CODE
    assert _snapshot(db_path, session_id) == before


# --------------------------------------------------------------------------- #
# Memory mode is untouched
# --------------------------------------------------------------------------- #


def test_memory_mode_reply_still_succeeds(tmp_path: Path) -> None:
    app = _app(tmp_path, None)
    with TestClient(app) as client:
        session_id = _create(client)
        _chat(client, session_id)
        response = _reply(client, session_id, "clar_fake_handle")
        view = client.get(f"/api/sessions/{session_id}").json()

    assert response.status_code == 200, response.text
    assert view["pending_clarification"] is None
    assert len(view["turns"]) == 2


# --------------------------------------------------------------------------- #
# Same-session concurrency (restored coverage)
# --------------------------------------------------------------------------- #


def test_same_session_concurrent_chats_keep_sqlite_consistent(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app_one = _app(tmp_path, db_path)
    app_two = _app(tmp_path, db_path)
    barrier = threading.Barrier(2)
    outcomes: dict[int, list[dict]] = {}
    errors: list[BaseException] = []

    with TestClient(app_one) as client_one, TestClient(app_two) as client_two:
        session_id = _create(client_one)

        def worker(index: int, client: TestClient) -> None:
            try:
                barrier.wait(timeout=30)
                outcomes[index] = _chat(client, session_id)
            except BaseException as exc:  # noqa: BLE001 - collected and asserted
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(0, client_one)),
            threading.Thread(target=worker, args=(1, client_two)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

    assert errors == []
    done_ids = [
        event["data"]["turn_id"]
        for events in outcomes.values()
        for event in events
        if event["event"] == "done"
    ]
    assert done_ids, "at least one concurrent chat must complete"

    open_pending, active_tasks = _counts(db_path)
    assert open_pending <= 1, "two concurrent chats must not leave two OPEN pendings"
    assert active_tasks <= 1, "at most one active task may remain"

    ledger = _ledger(db_path)
    assert len(ledger) == len(set(ledger)), "identity ledger must never contain duplicates"

    state = SQLiteWebStateStore(db_path)
    try:
        stored = {row.turn_id for row in state.list_turns(session_id)}
        assert set(done_ids) <= stored
        tasks = state.list_tasks(session_id)
        assert len([row for row in tasks if row.status in ("RUNNING", "WAITING_FOR_USER")]) <= 1
        for row in tasks:
            if row.status == "CANCELLED":
                assert row.close_reason, "a cancelled task keeps its reason"
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# Reset racing an in-flight ASK / late turn
# --------------------------------------------------------------------------- #


def test_reset_wins_over_inflight_ask(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)
    with TestClient(app) as client:
        session_id = _create(client)
        _chat(client, session_id)
        view = client.get(f"/api/sessions/{session_id}").json()
        clarification_id = view["pending_clarification"]["clarification_id"]

        state = SQLiteWebStateStore(db_path)
        try:
            task = state.get_active_task(session_id)
            pending = state.get_open_pending_for_session(session_id)
            turn_id = state.list_turns(session_id)[0].turn_id
        finally:
            state.close()
        assert task is not None and pending is not None

        reset = client.post(f"/api/sessions/{session_id}/reset")
        assert reset.status_code == 200

        late = state_late_commit(db_path, session_id, task, turn_id)
        assert late is False, "a late ASK must not reopen the cancelled task"

        stale_reply = _reply(client, session_id, clarification_id)
        assert stale_reply.status_code == 409
        assert stale_reply.json()["code"] == "clarification_not_pending"

    open_pending, active_tasks = _counts(db_path)
    assert (open_pending, active_tasks) == (0, 0)
    assert _snapshot(db_path, session_id)["turns"] == []


def state_late_commit(db_path: Path, session_id: str, task, turn_id: str) -> bool:
    state = SQLiteWebStateStore(db_path)
    try:
        state.commit_ask(
            session_id, task.task_id, expected_task_version=task.task_version,
            clarification_id="clar-late", source_turn_id=turn_id,
            query="late", payload={"question": "late"},
            expires_at=time.time() + 600,
        )
        return True
    except TaskStateConflict:
        return False
    finally:
        state.close()


def test_late_non_ask_turn_cannot_resurrect_history(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    state = SQLiteWebStateStore(db_path)
    try:
        state.create_session(session_id="web-late", profile_id="profile_0057")
        task = state.create_task("web-late", initial_requirements={"raw_query": "x"})
        turn_id = state.allocate_turn_id("web-late")
        state.reset_session("web-late")

        with pytest.raises(TaskStateConflict):
            state.append_turn(
                "web-late",
                turn={"turn_id": turn_id, "user_text": "late"},
                task_id=task.task_id,
                reset_generation=task.reset_generation,
            )
        assert state.list_turns("web-late") == ()
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# RUNNING-task crash window
# --------------------------------------------------------------------------- #


def test_running_task_crash_window_state_is_visible_and_reusable(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    state = SQLiteWebStateStore(db_path)
    try:
        state.create_session(session_id="web-crash", profile_id="profile_0057")
        task = state.create_task("web-crash", initial_requirements={"raw_query": "x"})
        state.allocate_turn_id("web-crash")
    finally:
        state.close()

    store = PersistentSessionStore(db_path)
    try:
        session = store.get("web-crash")
        assert session is not None
        # A crash between task creation and the turn/ASK commit leaves a RUNNING task with
        # no persisted turn and no pending question.  It is visible and reusable.
        assert session.turns == []
        assert session.pending_clarification is None
        active = store.state_store.get_active_task("web-crash")
        assert active is not None and active.task_id == task.task_id
        assert active.status == "RUNNING"
        reused = store.begin_chat_task(session, "continue")
        assert reused.task_id == task.task_id
    finally:
        store.close()
