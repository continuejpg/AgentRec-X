"""M9 SQLite durability through the real MCP AppContext (M1-C S4)."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.mcp.server import AppContext  # noqa: E402
from recommendation.memory.store import (  # noqa: E402
    MEMORY_DB_ENV_VAR,
    SQLitePreferenceStore,
)
from recommendation.shopping_companion.memory import ShoppingCompanionMemory  # noqa: E402
from tests.test_web_ask_persistence import _chat, _create  # noqa: E402
from web_demo.app import create_app  # noqa: E402
from web_demo.config import Settings  # noqa: E402
from web_demo.mcp_manager import MCPManager  # noqa: E402

PREF = "I prefer black."
DROP = "I don't care about color anymore."

CHILD = textwrap.dedent(
    """
    import json
    import os
    import sys

    repo, db, sid, turn_id, message = sys.argv[1:6]
    sys.path.insert(0, repo)
    from recommendation.mcp.server import AppContext
    from recommendation.memory.store import SQLitePreferenceStore
    from recommendation.shopping_companion.memory import ShoppingCompanionMemory

    store = SQLitePreferenceStore(db)
    ctx = AppContext(user_key="fixture-user", catalogue=None, memory=ShoppingCompanionMemory(),
                     task={}, search_mode="bm25", writes_enabled=True, memory_store=store)
    try:
        service = ctx.memory_for(sid).preference_service
        result = service.process_turn(user_key=sid, user_message=message,
                                      turn_id=turn_id, now=1000.0)
        active = [[e.value, e.polarity.value, e.status.value]
                  for e in service.get_active_preferences(sid).entries]
        payload = {
            "pid": os.getpid(),
            "added": len(result.update.added),
            "removed": len(result.update.removed),
            "skipped": result.update.skipped_duplicates,
            "already_processed": result.update.already_processed,
            "added_ids": [e.memory_id for e in result.update.added],
            "active": active,
            "history": len(service.get_memory_history(sid).entries),
        }
    finally:
        store.close()
    print(json.dumps(payload))
    """
)


def _ctx(db_path: Path, *, user_key: str = "fixture-user"):
    store = SQLitePreferenceStore(db_path)
    memory = ShoppingCompanionMemory()  # fixture memory: deliberately in-process
    context = AppContext(
        user_key=user_key, catalogue=None, memory=memory, task={"current_query": "q"},
        search_mode="bm25", writes_enabled=True, memory_store=store,
    )
    return context, store


def _active(service, user_key: str) -> list[tuple[str, str, str]]:
    return [
        (entry.value, entry.polarity.value, entry.status.value)
        for entry in service.get_active_preferences(user_key).entries
    ]


def _run_child(db_path: Path, sid: str, turn_id: str, message: str) -> dict:
    script = db_path.parent / "m9_child.py"
    script.write_text(CHILD, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(REPO_ROOT), str(db_path), sid, turn_id, message],
        capture_output=True, text=True, timeout=240, env=environment, cwd=str(db_path.parent),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# A: the real AppContext path uses SQLite
# --------------------------------------------------------------------------- #


def test_appcontext_session_memory_writes_through_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "m9.sqlite3"

    context, store = _ctx(db_path)
    try:
        service = context.memory_for("web-s4-a").preference_service
        result = service.process_turn(user_key="web-s4-a", user_message=PREF, turn_id="r1",
                                      now=1000.0)
        assert len(result.update.added) == 1
        assert ("black", "prefer", "active") in _active(service, "web-s4-a")
    finally:
        store.close()

    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute("SELECT COUNT(*) FROM preference_memory").fetchone()[0]
    finally:
        connection.close()
    assert rows == 1, "the session write must be a real SQLite row"

    reopened, store2 = _ctx(db_path)
    try:
        service2 = reopened.memory_for("web-s4-a").preference_service
        assert ("black", "prefer", "active") in _active(service2, "web-s4-a")
    finally:
        store2.close()


# --------------------------------------------------------------------------- #
# C + D: cross-process withdraw / re-add / retry idempotency
# --------------------------------------------------------------------------- #


def test_cross_process_withdraw_readd_and_retry(tmp_path: Path) -> None:
    db_path = tmp_path / "m9.sqlite3"
    sid = "web-s4-c"

    added = _run_child(db_path, sid, "r1", PREF)
    assert added["added"] == 1
    assert ["black", "prefer", "active"] in added["active"]
    first_memory_id = added["added_ids"][0]
    assert added["pid"] != os.getpid()

    withdrawn = _run_child(db_path, sid, "r2", DROP)
    assert withdrawn["removed"] == 1
    assert ["black", "prefer", "active"] not in withdrawn["active"]
    assert withdrawn["pid"] != added["pid"]

    confirmed = _run_child(db_path, sid, "r3", PREF)  # retry-like new turn
    assert confirmed["added"] == 1
    assert ["black", "prefer", "active"] in confirmed["active"]
    assert confirmed["added_ids"][0] != first_memory_id
    assert confirmed["history"] >= 2

    # Same turn id + same message after a full process restart: no duplicate event.
    retried = _run_child(db_path, sid, "r3", PREF)
    assert retried["already_processed"] is True
    assert retried["added"] == 0
    assert retried["history"] == confirmed["history"]


# --------------------------------------------------------------------------- #
# E: session isolation survives restarts
# --------------------------------------------------------------------------- #


def test_session_isolation_across_processes(tmp_path: Path) -> None:
    db_path = tmp_path / "m9.sqlite3"
    _run_child(db_path, "web-s4-one", "r1", PREF)

    context, store = _ctx(db_path)
    try:
        one = context.memory_for("web-s4-one").preference_service
        two = context.memory_for("web-s4-two").preference_service
        assert ("black", "prefer", "active") in _active(one, "web-s4-one")
        assert _active(two, "web-s4-two") == []
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# F: fixture memory never touches the durable database
# --------------------------------------------------------------------------- #


def test_fixture_memory_does_not_pollute_durable_database(tmp_path: Path) -> None:
    from recommendation.mcp.server import DEFAULT_FIXTURE, _memory_for

    if not DEFAULT_FIXTURE.exists():
        pytest.skip("MCP fixture is not present in this checkout")

    db_path = tmp_path / "m9.sqlite3"
    fixture_memory, _task = _memory_for(DEFAULT_FIXTURE, "p14e2_lite_0049")
    try:
        fixture_memory.preference_service.get_active_preferences("p14e2_lite_0049")
    finally:
        fixture_memory.close()

    if db_path.exists():
        connection = sqlite3.connect(str(db_path))
        try:
            assert connection.execute(
                "SELECT COUNT(*) FROM preference_memory"
            ).fetchone()[0] == 0
        finally:
            connection.close()


# --------------------------------------------------------------------------- #
# G: BFF -> MCP child configuration
# --------------------------------------------------------------------------- #


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        search_mode="bm25", allow_memory_writes=True, log_dir=tmp_path / "logs",
        repo=REPO_ROOT, python_executable=sys.executable, mcp_timeout_s=30.0,
    )
    values.update(overrides)
    return Settings(**values)


def test_mcp_child_receives_absolute_memory_db(tmp_path: Path) -> None:
    m9_path = tmp_path / "m9.sqlite3"
    manager = MCPManager(_settings(tmp_path, memory_db_path=m9_path))
    runtime = SimpleNamespace(profile_id="profile_0057", user_key="p14e2_lite_0057",
                              writes_enabled=True)
    parameters = manager._server_parameters(runtime)
    assert parameters.env["AGENTRECX_MEMORY_DB"] == str(m9_path.resolve())
    assert parameters.env["PYTHONPATH"] == str(REPO_ROOT)
    assert parameters.env["OMP_NUM_THREADS"] == "1"


def test_memory_db_is_derived_from_web_state_and_absent_without_either(
    tmp_path: Path,
) -> None:
    derived_manager = MCPManager(
        _settings(tmp_path, web_state_db_path=tmp_path / "web_state.sqlite3")
    )
    assert derived_manager.memory_database_path() == (
        tmp_path / "web_state.memory.sqlite3"
    ).resolve()

    memory_manager = MCPManager(_settings(tmp_path))
    assert memory_manager.memory_database_path() is None
    runtime = SimpleNamespace(profile_id="profile_0057", user_key="u", writes_enabled=True)
    assert "AGENTRECX_MEMORY_DB" not in memory_manager._server_parameters(runtime).env


def test_env_var_name_matches_memory_store_constant() -> None:
    assert MEMORY_DB_ENV_VAR == "AGENTRECX_MEMORY_DB"


# --------------------------------------------------------------------------- #
# H: close / reopen / corruption
# --------------------------------------------------------------------------- #


def test_store_close_reopen_and_corruption(tmp_path: Path) -> None:
    db_path = tmp_path / "m9.sqlite3"
    store = SQLitePreferenceStore(db_path)
    store.close()
    with pytest.raises(sqlite3.Error):
        store.get_entries("web-h")

    reopened = SQLitePreferenceStore(db_path)
    reopened.close()

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        SQLitePreferenceStore(corrupt)
    assert corrupt.read_bytes() == b"not a sqlite database"


# --------------------------------------------------------------------------- #
# I: memory mode stays compatible
# --------------------------------------------------------------------------- #


def test_memory_mode_session_memory_is_in_process() -> None:
    context = AppContext(
        user_key="fixture-user", catalogue=None, memory=ShoppingCompanionMemory(),
        task={}, search_mode="bm25", writes_enabled=True,
    )
    service = context.memory_for("web-s4-mem").preference_service
    service.process_turn(user_key="web-s4-mem", user_message=PREF, turn_id="r1", now=1.0)
    assert ("black", "prefer", "active") in _active(service, "web-s4-mem")


# --------------------------------------------------------------------------- #
# J: persistent reply gate is unchanged
# --------------------------------------------------------------------------- #


def test_persistent_reply_stays_blocked_with_both_databases(tmp_path: Path) -> None:
    web_state = tmp_path / "web_state.sqlite3"
    m9_path = tmp_path / "m9.sqlite3"
    settings = _settings(tmp_path, web_state_db_path=web_state, memory_db_path=m9_path)
    manager = _manager()
    app = create_app(settings=settings, manager=manager)

    with TestClient(app) as client:
        session_id = _create(client)
        _chat(client, session_id)
        clarification_id = client.get(f"/api/sessions/{session_id}").json()[
            "pending_clarification"
        ]["clarification_id"]
        calls_before = len(manager.calls)
        response = client.post(
            "/api/clarification/reply",
            json={"session_id": session_id, "clarification_id": clarification_id,
                  "message": "For gaming."},
        )

    assert response.status_code == 409
    assert response.json()["code"] == "clarification_resume_not_ready"
    assert len(manager.calls) == calls_before
    assert not m9_path.exists() or sqlite3.connect(str(m9_path)).execute(
        "SELECT COUNT(*) FROM preference_memory"
    ).fetchone()[0] == 0


def _manager():
    import tests.test_web_demo_unit as twu

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
