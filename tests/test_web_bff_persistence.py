"""FastAPI-level persistence tests for M1-C S3a.

The BFF is exercised through real ``TestClient`` HTTP routes.  ``FakeManager`` replaces
only the MCP transport; session identity allocation, SQLite writes and the reset path are
the real production code.  Cross-process tests spawn real OS processes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
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
from web_demo.persistent_sessions import (  # noqa: E402
    PERSISTENT_SESSION_ID_BYTES,
    PersistentSessionStore,
)
from web_demo.sessions import DEFAULT_TITLE, SessionStore  # noqa: E402
from web_demo.state_store import SQLiteWebStateStore, StateStoreError  # noqa: E402

VIEW_KEYS = {
    "session_id", "profile_id", "title", "created_at", "updated_at", "turns",
    "pending_clarification", "behavior_enabled", "latest_recommendation_id",
}

BFF_CHILD = textwrap.dedent(
    """
    import json
    import os
    import sys
    import tempfile
    from pathlib import Path

    repo, action, db = sys.argv[1], sys.argv[2], sys.argv[3]
    session_id = sys.argv[4] if len(sys.argv) > 4 else ""
    sys.path.insert(0, repo)

    from fastapi.testclient import TestClient
    import tests.test_web_demo_unit as twu
    from web_demo.app import create_app
    from web_demo.config import Settings

    manager = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager)
    manager.responses.update({
        "recommend_products": twu.recommend_payload(status="awaiting_clarification"),
        "submit_clarification_reply": twu.reply_payload(),
        "get_active_preferences": twu.preferences_payload(),
    })
    tmp = Path(tempfile.mkdtemp(prefix="bff_child_"))
    settings = Settings(
        search_mode="bm25", allow_memory_writes=True, log_dir=tmp / "logs",
        web_state_db_path=Path(db), python_executable=sys.executable, mcp_timeout_s=30.0,
    )
    app = create_app(settings=settings, manager=manager)

    with TestClient(app) as client:
        if action == "write":
            sid = client.post(
                "/api/sessions", json={"profile_id": "profile_0057"}
            ).json()["session_id"]
            chat = client.post(
                "/api/chat/stream",
                json={"session_id": sid, "message": "I need wireless earbuds for work"},
            )
            _ = chat.text
            client.post(
                "/api/behavior/enabled", json={"session_id": sid, "enabled": False}
            )
            view = client.get(f"/api/sessions/{sid}").json()
            payload = {
                "pid": os.getpid(), "session_id": sid, "title": view["title"],
                "profile_id": view["profile_id"],
                "behavior_enabled": view["behavior_enabled"],
            }
        elif action == "read":
            response = client.get(f"/api/sessions/{session_id}")
            view = response.json()
            payload = {
                "pid": os.getpid(), "status": response.status_code,
                "session_id": view.get("session_id"), "title": view.get("title"),
                "profile_id": view.get("profile_id"),
                "behavior_enabled": view.get("behavior_enabled"),
            }
        elif action == "turn":
            sid = session_id or client.post(
                "/api/sessions", json={"profile_id": "profile_0057"}
            ).json()["session_id"]
            chat = client.post(
                "/api/chat/stream",
                json={"session_id": sid, "message": "I need some new headphones."},
            )
            events = twu.parse_sse(chat.text)
            done = next(event["data"] for event in events if event["event"] == "done")
            reply = client.post(
                "/api/clarification/reply",
                json={"session_id": sid, "clarification_id": "clar_fake_handle",
                      "message": "I prefer black."},
            )
            payload = {
                "pid": os.getpid(), "session_id": sid,
                "chat_status": chat.status_code, "chat_turn_id": done["turn_id"],
                "reply_status": reply.status_code,
                "reply_turn_id": reply.json().get("turn_id"),
            }
        else:
            raise SystemExit(f"unknown action {action!r}")
    print(json.dumps(payload))
    """
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _manager() -> twu.FakeManager:
    manager = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager)
    manager.responses.update(
        {
            "recommend_products": twu.recommend_payload(status="awaiting_clarification"),
            "submit_clarification_reply": twu.reply_payload(),
            "get_active_preferences": twu.preferences_payload(),
        }
    )
    return manager


def _settings(tmp_path: Path, db_path: Path | None) -> Settings:
    return Settings(
        search_mode="bm25",
        allow_memory_writes=True,
        log_dir=tmp_path / "logs",
        web_state_db_path=db_path,
        python_executable=sys.executable,
        mcp_timeout_s=30.0,
    )


def _app(tmp_path: Path, db_path: Path | None, manager=None):
    return create_app(settings=_settings(tmp_path, db_path), manager=manager or _manager())


def _create(client: TestClient, profile_id: str = "profile_0057") -> dict:
    response = client.post("/api/sessions", json={"profile_id": profile_id})
    assert response.status_code == 200, response.text
    return response.json()


def _chat(client: TestClient, session_id: str, message: str = "I need some new headphones."):
    response = client.post("/api/chat/stream", json={"session_id": session_id, "message": message})
    assert response.status_code == 200, response.text
    return twu.parse_sse(response.text)


def _turn_id(events: list[dict]) -> str:
    return next(event["data"]["turn_id"] for event in events if event["event"] == "done")


def _run_child(tmp_path: Path, action: str, db_path: Path, session_id: str = "") -> dict:
    script = tmp_path / "bff_child.py"
    script.write_text(BFF_CHILD, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(script), str(REPO_ROOT), action, str(db_path)]
    if session_id:
        command.append(session_id)
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=240,
        env=environment, cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"child {action} failed: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _ledger(db_path: Path, session_id: str) -> list[str]:
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT value FROM web_identity WHERE session_id = ? ORDER BY kind, seq",
            (session_id,),
        )
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# Test A: HTTP-created session is really persisted
# --------------------------------------------------------------------------- #


def test_http_created_session_is_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)

    with TestClient(app) as client:
        view = _create(client)
        session_id = view["session_id"]
        assert view["title"] == DEFAULT_TITLE
        assert len(session_id) == len("web-") + 2 * PERSISTENT_SESSION_ID_BYTES

        fetched = client.get(f"/api/sessions/{session_id}")
        assert fetched.status_code == 200
        assert fetched.json()["session_id"] == session_id

        listed = client.get("/api/sessions").json()["sessions"]
        assert any(item["session_id"] == session_id for item in listed)

    state = SQLiteWebStateStore(db_path)
    try:
        record = state.get_session(session_id)
        assert record is not None
        assert record.profile_id == "profile_0057"
        assert record.behavior_enabled is True
        assert record.title == DEFAULT_TITLE
        assert (record.turn_counter, record.reply_counter) == (0, 0)
    finally:
        state.close()

    # Legacy 8-hex ids created by earlier versions remain readable over HTTP.
    legacy = SQLiteWebStateStore(db_path)
    try:
        legacy.create_session(session_id="web-1a2b3c4d", profile_id="profile_0049")
    finally:
        legacy.close()
    app2 = _app(tmp_path, db_path)
    with TestClient(app2) as client2:
        response = client2.get("/api/sessions/web-1a2b3c4d")
        assert response.status_code == 200
        assert response.json()["session_id"] == "web-1a2b3c4d"


# --------------------------------------------------------------------------- #
# Test B: metadata across a real OS process restart
# --------------------------------------------------------------------------- #


def test_http_metadata_survives_os_process_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"

    written = _run_child(tmp_path, "write", db_path)
    read = _run_child(tmp_path, "read", db_path, written["session_id"])

    assert written["pid"] != read["pid"]
    assert os.getpid() not in {written["pid"], read["pid"]}
    assert read["status"] == 200
    assert read["session_id"] == written["session_id"]
    assert read["profile_id"] == written["profile_id"] == "profile_0057"
    assert read["title"] == written["title"]
    assert written["title"] == "I need wireless earbuds for work"
    assert read["behavior_enabled"] is False


# --------------------------------------------------------------------------- #
# Test C: identity allocation across a real OS process restart
# --------------------------------------------------------------------------- #


def test_http_identities_survive_os_process_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"

    first = _run_child(tmp_path, "turn", db_path)
    assert first["chat_status"] == 200 and first["reply_status"] == 200
    session_id = first["session_id"]
    assert first["chat_turn_id"] == f"{session_id}-t1"
    assert first["reply_turn_id"] == f"{session_id}-r1"

    second = _run_child(tmp_path, "turn", db_path, session_id)
    assert second["pid"] != first["pid"]
    assert second["chat_turn_id"] == f"{session_id}-t2"
    assert second["reply_turn_id"] == f"{session_id}-r2"
    assert second["chat_turn_id"] != first["chat_turn_id"]
    assert second["reply_turn_id"] != first["reply_turn_id"]

    state = SQLiteWebStateStore(db_path)
    try:
        record = state.get_session(session_id)
        assert record is not None
        assert (record.turn_counter, record.reply_counter) == (2, 2)
    finally:
        state.close()
    assert sorted(_ledger(db_path, session_id)) == sorted(
        [f"{session_id}-r1", f"{session_id}-r2", f"{session_id}-t1", f"{session_id}-t2"]
    )


# --------------------------------------------------------------------------- #
# Test D: HTTP reset delegates to the SQLite reset
# --------------------------------------------------------------------------- #


def test_http_reset_delegates_to_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)

    with TestClient(app) as client:
        session_id = _create(client)["session_id"]
        first_turn = _turn_id(_chat(client, session_id))
        assert first_turn == f"{session_id}-t1"

        # Seed a real active task + OPEN pending; the BFF does not create tasks yet.
        state = SQLiteWebStateStore(db_path)
        try:
            seeded_turn = state.allocate_turn_id(session_id)
            task = state.create_task(session_id, initial_requirements={"budget_max": 5000})
            pending = state.commit_ask(
                session_id, task.task_id, expected_task_version=1,
                clarification_id="clar-seed", source_turn_id=seeded_turn,
                query="Recommend a laptop under 5000 for me.",
                payload={"question": "Office or gaming?"},
                expires_at=time.time() + 600,
            )
            before = state.get_session(session_id)
        finally:
            state.close()

        response = client.post(f"/api/sessions/{session_id}/reset")
        assert response.status_code == 200, response.text
        view = response.json()
        assert set(view) == VIEW_KEYS
        assert view["session_id"] == session_id
        assert view["turns"] == []
        assert view["pending_clarification"] is None
        assert view["title"] == DEFAULT_TITLE

        after_state = SQLiteWebStateStore(db_path)
        try:
            after = after_state.get_session(session_id)
            assert after is not None
            assert after.reset_generation == before.reset_generation + 1
            assert (after.turn_counter, after.reply_counter) == (
                before.turn_counter, before.reply_counter
            )
            assert after_state.get_task(task.task_id).status == "CANCELLED"
            assert after_state.get_pending(pending.pending_id).status == "CANCELLED"
        finally:
            after_state.close()

        second_turn = _turn_id(_chat(client, session_id))
        assert second_turn not in {first_turn, seeded_turn}
        assert second_turn == f"{session_id}-t3"

    # Each ASK allocates the pending's reply identity too (demo_flow.py:349), so the
    # ledger holds one turn + one reply id per chat, plus the seeded task's pair.
    assert _ledger(db_path, session_id) == sorted(
        [
            f"{session_id}-t1", f"{session_id}-r1",
            f"{session_id}-t2", f"{session_id}-r2",
            f"{session_id}-t3", f"{session_id}-r3",
        ]
    )


# --------------------------------------------------------------------------- #
# Test E: two app instances, concurrent identity allocation
# --------------------------------------------------------------------------- #


def test_two_apps_allocate_identities_concurrently(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app_one = _app(tmp_path, db_path)
    app_two = _app(tmp_path, db_path)

    results: dict[int, str] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    with TestClient(app_one) as client_one, TestClient(app_two) as client_two:
        session_id = _create(client_one)["session_id"]

        def worker(index: int, client: TestClient) -> None:
            try:
                barrier.wait(timeout=30)
                results[index] = _turn_id(_chat(client, session_id))
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
        assert set(results.values()) == {f"{session_id}-t1", f"{session_id}-t2"}

    state = SQLiteWebStateStore(db_path)
    try:
        record = state.get_session(session_id)
        assert record is not None
        assert record.turn_counter == 2
    finally:
        state.close()
    # One turn id plus one pending reply id per concurrent chat.
    assert sorted(_ledger(db_path, session_id)) == [
        f"{session_id}-r1", f"{session_id}-r2",
        f"{session_id}-t1", f"{session_id}-t2",
    ]


# --------------------------------------------------------------------------- #
# Test F: fail closed when SQLite is unusable
# --------------------------------------------------------------------------- #


def test_unusable_database_refuses_to_start(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"this is not a sqlite database")

    with pytest.raises(StateStoreError):
        create_app(settings=_settings(tmp_path, corrupt), manager=_manager())

    assert corrupt.read_bytes() == b"this is not a sqlite database"


def test_runtime_store_failure_returns_503_not_fake_success(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)

    with TestClient(app) as client:
        session_id = _create(client)["session_id"]
        app.state.sessions.close()

        created = client.post("/api/sessions", json={"profile_id": "profile_0057"})
        assert created.status_code == 503
        assert created.json()["code"] == "state_store_unavailable"

        listed = client.get("/api/sessions")
        assert listed.status_code == 503

        missing = client.get(f"/api/sessions/{session_id}")
        assert missing.status_code == 503

        # The SSE route resolves the session first, so it fails closed with 503 rather
        # than streaming a successful-looking turn.
        chat = client.post(
            "/api/chat/stream", json={"session_id": session_id, "message": "hello"}
        )
        assert chat.status_code == 503
        assert chat.json()["code"] == "state_store_unavailable"

    state = SQLiteWebStateStore(db_path)
    try:
        assert len(state.list_sessions(50)) == 1, "no phantom session may be created"
    finally:
        state.close()


# --------------------------------------------------------------------------- #
# Test G: the in-memory mode is still the default
# --------------------------------------------------------------------------- #


def test_memory_mode_is_still_the_default(tmp_path: Path) -> None:
    app = create_app(settings=_settings(tmp_path, None), manager=_manager())
    assert isinstance(app.state.sessions, SessionStore)
    assert not isinstance(app.state.sessions, PersistentSessionStore)

    with TestClient(app) as client:
        view = _create(client)
        assert view["session_id"].startswith("web-")
        assert client.get(f"/api/sessions/{view['session_id']}").status_code == 200


# --------------------------------------------------------------------------- #
# Test H: S3a capability boundary
# --------------------------------------------------------------------------- #


def test_s3a_boundary_identity_persists_history_does_not(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite3"
    app = _app(tmp_path, db_path)

    with TestClient(app) as client:
        session_id = _create(client)["session_id"]
        chat_turn = _turn_id(_chat(client, session_id, "Recommend a laptop under 5000 for me."))
        in_process = client.get(f"/api/sessions/{session_id}").json()
        # In-process the ASK is visible; it lives only in memory (S3b will persist it).
        assert in_process["pending_clarification"] is not None
        assert len(in_process["turns"]) == 1
        assert in_process["title"] == "Recommend a laptop under 5000 for me."

    # A fresh app on the same database stands in for a restarted BFF process.
    app2 = _app(tmp_path, db_path)
    with TestClient(app2) as client2:
        restored = client2.get(f"/api/sessions/{session_id}").json()
        assert restored["session_id"] == session_id
        assert restored["profile_id"] == "profile_0057"
        assert restored["title"] == "Recommend a laptop under 5000 for me."  # metadata durable
        assert restored["turns"] == []                    # history: NOT durable in S3a
        assert restored["pending_clarification"] is None  # pending: NOT durable in S3a

    state = SQLiteWebStateStore(db_path)
    try:
        record = state.get_session(session_id)
        assert record is not None
        assert record.turn_counter == 1
        assert state.list_tasks(session_id) == (), "the BFF created no Task in S3a"
    finally:
        state.close()

    raw = sqlite3.connect(str(db_path))
    try:
        assert raw.execute("SELECT COUNT(*) FROM web_pending").fetchone()[0] == 0
        assert raw.execute("SELECT COUNT(*) FROM web_task").fetchone()[0] == 0
    finally:
        raw.close()

    # The identity ledger is durable even though the conversation is not: the chat
    # allocated a turn id and, for the pending ASK, its reply id.
    assert _ledger(db_path, session_id) == sorted([chat_turn, f"{session_id}-r1"])
