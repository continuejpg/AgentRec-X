"""Tests for Task and multi-round Pending persistence (M1-C S2b).

Everything runs against real SQLite files under pytest's ``tmp_path``.  The migration
tests build a genuine v1 (S2a) database with rows, then upgrade it; no test fakes a
schema version by writing a number into an empty database.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_demo.state_store import (  # noqa: E402
    WEB_STATE_SCHEMA_VERSION,
    ActiveTaskExists,
    AskCommitError,
    PendingExpired,
    PendingStateConflict,
    SchemaVersionError,
    SessionRetired,
    StaleTaskVersion,
    StateDataCorruption,
    StateMigrationError,
    StateStoreError,
    SQLiteWebStateStore,
    TaskStateConflict,
)

# The exact S2a schema, used to build a real v1 database for the migration tests.
V1_DDL = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE web_session (
    session_id        TEXT PRIMARY KEY,
    profile_id        TEXT NOT NULL,
    title             TEXT NOT NULL,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    behavior_enabled  INTEGER NOT NULL CHECK (behavior_enabled IN (0, 1)),
    turn_counter      INTEGER NOT NULL DEFAULT 0 CHECK (turn_counter >= 0),
    reply_counter     INTEGER NOT NULL DEFAULT 0 CHECK (reply_counter >= 0),
    reset_generation  INTEGER NOT NULL DEFAULT 0 CHECK (reset_generation >= 0),
    retired           INTEGER NOT NULL DEFAULT 0 CHECK (retired IN (0, 1))
);
CREATE TABLE web_identity (
    session_id TEXT NOT NULL REFERENCES web_session(session_id),
    kind       TEXT NOT NULL CHECK (kind IN ('turn', 'reply')),
    seq        INTEGER NOT NULL CHECK (seq >= 1),
    value      TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, kind, value),
    UNIQUE (session_id, kind, seq)
);
"""

CHILD_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import sys
    import time

    repo, action, db = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = sys.argv[4] if len(sys.argv) > 4 else None
    sys.path.insert(0, repo)
    from web_demo.state_store import SQLiteWebStateStore

    store = SQLiteWebStateStore(db)
    try:
        if action == "task_write":
            store.create_session(session_id="web-os", profile_id="profile_0057")
            task = store.create_task("web-os", initial_requirements={"budget_max": 5000})
            turn = store.allocate_turn_id("web-os")
            pending = store.commit_ask(
                "web-os",
                task.task_id,
                expected_task_version=task.task_version,
                clarification_id="clar-os",
                source_turn_id=turn,
                query="Recommend a laptop under 5000 for me.",
                payload={"question": "Office or gaming?", "clarification_id": "clar-os"},
                expires_at=time.time() + 3600,
            )
            current = store.get_task(task.task_id)
            payload = {
                "pid": os.getpid(),
                "session_id": "web-os",
                "task_id": task.task_id,
                "task_status": current.status,
                "task_version": current.task_version,
                "pending_id": pending.pending_id,
                "reply_turn_id": pending.reply_turn_id,
                "expires_at": pending.expires_at,
                "query": pending.query,
                "question": pending.payload["question"],
            }
        elif action == "task_inspect":
            task = store.get_task(extra)
            pending = store.get_open_pending(extra)
            payload = {
                "pid": os.getpid(),
                "session_id": None if task is None else task.session_id,
                "task_id": None if task is None else task.task_id,
                "task_status": None if task is None else task.status,
                "task_version": None if task is None else task.task_version,
                "pending_id": None if pending is None else pending.pending_id,
                "reply_turn_id": None if pending is None else pending.reply_turn_id,
                "expires_at": None if pending is None else pending.expires_at,
                "query": None if pending is None else pending.query,
                "question": None if pending is None else pending.payload["question"],
            }
        else:
            raise SystemExit(f"unknown action {action!r}")
    finally:
        store.close()
    print(json.dumps(payload))
    """
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture()
def store(tmp_path: Path):
    instance = SQLiteWebStateStore(tmp_path / "web_state.sqlite3")
    try:
        yield instance
    finally:
        instance.close()


def _create_v1_database(path: Path) -> None:
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(V1_DDL)
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")
        connection.execute(
            "INSERT INTO web_session(session_id, profile_id, title, created_at, updated_at, "
            "behavior_enabled, turn_counter, reply_counter, reset_generation, retired) "
            "VALUES ('web-v1', 'profile_0057', 'Legacy session', 100.0, 150.0, 0, 2, 3, 4, 0)"
        )
        rows = [
            ("turn", 1, "web-v1-t1"),
            ("turn", 2, "web-v1-t2"),
            ("reply", 1, "web-v1-r1"),
            ("reply", 2, "web-v1-r2"),
            ("reply", 3, "web-v1-r3"),
        ]
        connection.executemany(
            "INSERT INTO web_identity(session_id, kind, seq, value, created_at) "
            "VALUES ('web-v1', ?, ?, ?, 200.0)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def _new_session(store: SQLiteWebStateStore, session_id: str = "web-s2b"):
    store.create_session(session_id=session_id, profile_id="profile_0057")
    return session_id


def _new_task(store: SQLiteWebStateStore, session_id: str = "web-s2b", requirements=None):
    if store.get_session(session_id) is None:
        _new_session(store, session_id)
    return store.create_task(
        session_id, initial_requirements=requirements or {"budget_max": 5000}
    )


def _ask(
    store: SQLiteWebStateStore,
    session_id: str,
    task,
    *,
    version: int | None = None,
    turn_id: str | None = None,
    expires_in: float = 3600.0,
    clarification_id: str = "clar-1",
    payload=None,
):
    source_turn = turn_id or store.allocate_turn_id(session_id)
    return store.commit_ask(
        session_id,
        task.task_id,
        expected_task_version=task.task_version if version is None else version,
        clarification_id=clarification_id,
        source_turn_id=source_turn,
        query="Recommend a laptop under 5000 for me.",
        payload=payload or {"question": "Office or gaming?", "clarification_id": clarification_id},
        expires_at=time.time() + expires_in,
    )


def _ledger(session_id: str, path: Path, kind: str | None = None) -> list[str]:
    connection = sqlite3.connect(str(path))
    try:
        if kind is None:
            rows = connection.execute(
                "SELECT value FROM web_identity WHERE session_id = ? ORDER BY kind, seq",
                (session_id,),
            )
        else:
            rows = connection.execute(
                "SELECT value FROM web_identity WHERE session_id = ? AND kind = ? ORDER BY seq",
                (session_id, kind),
            )
        return [str(row[0]) for row in rows]
    finally:
        connection.close()


def _run_child(tmp_path: Path, db_path: Path, action: str, extra: str = "") -> dict:
    script = tmp_path / "task_child.py"
    script.write_text(CHILD_SCRIPT, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(script), str(REPO_ROOT), action, str(db_path)]
    if extra:
        command.append(extra)
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120,
        env=environment, cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"child {action} failed: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Identity and lifecycle
# --------------------------------------------------------------------------- #


def test_task_identity_is_distinct_from_session(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)

    assert task.task_id != session_id
    assert len(task.task_id) == 36  # UUID4 textual form
    assert task.session_id == session_id
    assert task.status == "RUNNING"
    assert task.task_version == 1
    assert task.reset_generation == 0
    assert task.current_requirements == {"budget_max": 5000}
    assert task.requirements_version == 0

    assert store.get_task(task.task_id) == task
    assert store.get_active_task(session_id) == task
    assert store.list_tasks(session_id) == (task,)


def test_only_one_active_task_per_session(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    first = _new_task(store, session_id)

    with pytest.raises(ActiveTaskExists):
        store.create_task(session_id, initial_requirements={"budget_max": 8000})

    assert store.get_active_task(session_id) == first
    assert len(store.list_tasks(session_id)) == 1

    other = _new_task(store, "web-s2b-other")
    assert other.task_id != first.task_id


def test_session_can_have_sequential_tasks(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    first = _new_task(store, session_id)
    cancelled = store.cancel_task(
        session_id, first.task_id, expected_task_version=first.task_version, reason="done"
    )
    assert cancelled.status == "CANCELLED"
    assert cancelled.close_reason == "done"

    second = _new_task(store, session_id, {"budget_max": 8000})
    assert second.task_id != first.task_id
    assert second.status == "RUNNING"
    assert store.get_active_task(session_id) == second

    history = store.list_tasks(session_id)
    assert [task.task_id for task in history] == [first.task_id, second.task_id]
    assert [task.status for task in history] == ["CANCELLED", "RUNNING"]


def test_retired_session_cannot_create_task(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    store.retire_session(session_id)

    with pytest.raises(SessionRetired):
        store.create_task(session_id, initial_requirements={})
    with pytest.raises(SessionRetired):
        store.reset_session(session_id)


# --------------------------------------------------------------------------- #
# Multi-round ASK
# --------------------------------------------------------------------------- #


def test_three_continuous_asks_on_one_task(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    versions = [task.task_version]

    for round_index in range(3):
        pending = _ask(
            store, session_id, task,
            version=task.task_version,
            clarification_id=f"clar-{round_index + 1}",
        )
        task = store.get_task(task.task_id)
        versions.append(task.task_version)
        assert task.status == "WAITING_FOR_USER"
        assert pending.status == "OPEN"
        assert pending.reply_turn_id == f"{session_id}-r{round_index + 1}"

        if round_index < 2:
            task, consumed = store.consume_pending(
                session_id, task.task_id, pending.pending_id,
                expected_task_version=task.task_version,
                requirements={"budget_max": 5000 + round_index * 1000},
            )
            versions.append(task.task_version)
            assert consumed.status == "CONSUMED"
            assert consumed.consumed_at is not None
            assert task.status == "RUNNING"
            assert task.requirements_version == round_index + 1

    assert versions == [1, 2, 3, 4, 5, 6]
    assert task.requirements_version == 2

    history = store.list_pending(task.task_id)
    assert len(history) == 3
    # Rounds 1 and 2 were answered; round 3 is still waiting for the user.
    assert [row.status for row in history] == ["CONSUMED", "CONSUMED", "OPEN"]
    assert [row.reply_turn_id for row in history] == [
        f"{session_id}-r1", f"{session_id}-r2", f"{session_id}-r3"
    ]
    assert task.status == "WAITING_FOR_USER"
    assert store.get_active_task(session_id).task_id == task.task_id


def test_pending_history_is_retained(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)

    first = _ask(store, session_id, task, clarification_id="clar-1")
    task, _ = store.consume_pending(
        session_id, task.task_id, first.pending_id, expected_task_version=2
    )
    second = _ask(store, session_id, task, clarification_id="clar-2")

    history = store.list_pending(task.task_id)
    assert len(history) == 2
    assert {row.pending_id for row in history} == {first.pending_id, second.pending_id}
    assert history[0].status == "CONSUMED"
    assert history[1].status == "OPEN"
    assert history[0].pending_id != history[1].pending_id
    assert history[0].reply_turn_id != history[1].reply_turn_id


def test_only_one_open_pending_per_task(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    _ask(store, session_id, task, clarification_id="clar-1")

    # Force the task back to RUNNING without consuming, so the OPEN-pending guard (not
    # the status guard) is the rule under test.
    raw = sqlite3.connect(store.path)
    try:
        raw.execute(
            "UPDATE web_task SET status = 'RUNNING' WHERE task_id = ?", (task.task_id,)
        )
        raw.commit()

        with pytest.raises(PendingStateConflict):
            store.commit_ask(
                session_id,
                task.task_id,
                expected_task_version=2,
                clarification_id="clar-2",
                source_turn_id=store.allocate_turn_id(session_id),
                query="Recommend a laptop under 5000 for me.",
                payload={"question": "again?"},
                expires_at=time.time() + 60,
            )

        # The partial unique index is the database-level guarantee.
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                "INSERT INTO web_pending(pending_id, task_id, clarification_id, query, "
                "turn_id, reply_turn_id, mode, payload_json, status, created_at, "
                "expires_at, consumed_at) VALUES ('pend-dup', ?, 'clar-x', 'q', 't', "
                "'web-s2b-r999', 'cpu_live', '{}', 'OPEN', ?, ?, NULL)",
                (task.task_id, time.time(), time.time() + 60),
            )
    finally:
        raw.close()


def test_ask_allocates_reply_identity_in_same_transaction(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    turn_id = store.allocate_turn_id(session_id)
    payload = {"question": "Office or gaming?", "clarification_id": "clar-1", "extra": [1, 2]}

    pending = store.commit_ask(
        session_id,
        task.task_id,
        expected_task_version=1,
        clarification_id="clar-1",
        source_turn_id=turn_id,
        query="Recommend a laptop under 5000 for me.",
        payload=payload,
        expires_at=time.time() + 600,
    )

    session = store.get_session(session_id)
    assert session is not None
    assert session.reply_counter == 1
    assert pending.reply_turn_id == f"{session_id}-r1"
    assert pending.turn_id == turn_id
    assert pending.payload == payload
    assert pending.mode == "cpu_live"
    assert pending.status == "OPEN"
    assert _ledger(session_id, Path(store.path), "reply") == [f"{session_id}-r1"]

    updated = store.get_task(task.task_id)
    assert updated is not None
    assert updated.status == "WAITING_FOR_USER"
    assert updated.task_version == 2


def test_failed_ask_rolls_back_counter_and_ledger(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    turn_id = store.allocate_turn_id(session_id)

    raw = sqlite3.connect(store.path)
    try:
        raw.execute(
            "CREATE TRIGGER fail_pending BEFORE INSERT ON web_pending "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        raw.commit()

        with pytest.raises(AskCommitError):
            store.commit_ask(
                session_id, task.task_id, expected_task_version=1,
                clarification_id="clar-1", source_turn_id=turn_id,
                query="Recommend a laptop under 5000 for me.",
                payload={"question": "Office or gaming?"},
                expires_at=time.time() + 600,
            )

        session = store.get_session(session_id)
        assert session is not None
        assert session.reply_counter == 0, "reply counter must roll back with the pending row"
        assert _ledger(session_id, Path(store.path), "reply") == []
        assert store.list_pending(task.task_id) == ()
        unchanged = store.get_task(task.task_id)
        assert unchanged is not None
        assert unchanged.status == "RUNNING"
        assert unchanged.task_version == 1

        raw.execute("DROP TRIGGER fail_pending")
        raw.commit()
    finally:
        raw.close()

    retried = store.commit_ask(
        session_id, task.task_id, expected_task_version=1,
        clarification_id="clar-1", source_turn_id=turn_id,
        query="Recommend a laptop under 5000 for me.",
        payload={"question": "Office or gaming?"},
        expires_at=time.time() + 600,
    )
    assert retried.reply_turn_id == f"{session_id}-r1"  # no counter gap
    session = store.get_session(session_id)
    assert session is not None
    assert session.reply_counter == 1


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #


def test_reset_cancels_task_and_pending_but_keeps_identity(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    pending = _ask(store, session_id, task, clarification_id="clar-1")

    before = store.get_session(session_id)
    assert before is not None
    ledger_before = _ledger(session_id, Path(store.path))

    reset = store.reset_session(session_id)

    assert reset.session_id == session_id
    assert reset.reset_generation == before.reset_generation + 1
    assert reset.turn_counter == before.turn_counter
    assert reset.reply_counter == before.reply_counter
    assert reset.behavior_enabled == before.behavior_enabled
    assert _ledger(session_id, Path(store.path)) == ledger_before

    cancelled_task = store.get_task(task.task_id)
    assert cancelled_task is not None
    assert cancelled_task.status == "CANCELLED"
    assert cancelled_task.close_reason == "user_reset"
    assert cancelled_task.closed_at is not None
    assert cancelled_task.task_version == 3

    cancelled_pending = store.get_pending(pending.pending_id)
    assert cancelled_pending is not None
    assert cancelled_pending.status == "CANCELLED"
    assert store.get_active_task(session_id) is None
    assert store.get_open_pending(task.task_id) is None

    next_reply = store.allocate_reply_id(session_id)
    assert next_reply == f"{session_id}-r2"

    fresh = store.create_task(session_id, initial_requirements={"budget_max": 8000})
    assert fresh.task_id != task.task_id
    assert fresh.reset_generation == reset.reset_generation


def test_cancelled_task_cannot_be_updated(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    pending = _ask(store, session_id, task, clarification_id="clar-1")
    store.reset_session(session_id)

    with pytest.raises(TaskStateConflict):
        store.commit_ask(
            session_id, task.task_id, expected_task_version=3,
            clarification_id="clar-2",
            source_turn_id=store.allocate_turn_id(session_id),
            query="Recommend a laptop under 5000 for me.",
            payload={"question": "again?"},
            expires_at=time.time() + 60,
        )
    with pytest.raises(TaskStateConflict):
        store.consume_pending(
            session_id, task.task_id, pending.pending_id, expected_task_version=3
        )
    with pytest.raises(TaskStateConflict):
        store.complete_task(session_id, task.task_id, expected_task_version=3)


def test_advance_reset_generation_has_full_reset_semantics(store: SQLiteWebStateStore) -> None:
    """The S2a-named method must not be a generation-only shortcut."""
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    pending = _ask(store, session_id, task, clarification_id="clar-1")

    advanced = store.advance_reset_generation(session_id)

    assert advanced.reset_generation == 1
    cancelled = store.get_task(task.task_id)
    assert cancelled is not None and cancelled.status == "CANCELLED"
    expired = store.get_pending(pending.pending_id)
    assert expired is not None and expired.status == "CANCELLED"
    assert store.get_active_task(session_id) is None


# --------------------------------------------------------------------------- #
# OCC / terminal states
# --------------------------------------------------------------------------- #


def test_stale_version_cannot_overwrite_newer_state(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    pending = _ask(store, session_id, task, clarification_id="clar-1")
    task, _ = store.consume_pending(
        session_id, task.task_id, pending.pending_id,
        expected_task_version=2, requirements={"budget_max": 6000},
    )

    raw = sqlite3.connect(store.path)
    try:
        raw.execute(
            "UPDATE web_task SET task_version = 5, current_requirements_json = "
            "'{\"budget_max\": 9999}' WHERE task_id = ?",
            (task.task_id,),
        )
        raw.commit()
    finally:
        raw.close()

    with pytest.raises(StaleTaskVersion):
        store.complete_task(session_id, task.task_id, expected_task_version=4)

    unchanged = store.get_task(task.task_id)
    assert unchanged is not None
    assert unchanged.status == "RUNNING"
    assert unchanged.task_version == 5
    assert unchanged.current_requirements == {"budget_max": 9999}

    completed = store.complete_task(
        session_id, task.task_id, expected_task_version=5,
        requirements={"budget_max": 6000, "use_case": "gaming"},
    )
    assert completed.status == "COMPLETED"
    assert completed.task_version == 6
    assert completed.current_requirements == {"budget_max": 6000, "use_case": "gaming"}
    assert completed.requirements_version == 2


def test_terminal_tasks_stay_terminal(store: SQLiteWebStateStore) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    completed = store.complete_task(
        session_id, task.task_id, expected_task_version=1, requirements={"done": True}
    )
    assert completed.status == "COMPLETED"

    with pytest.raises(TaskStateConflict):
        store.complete_task(session_id, task.task_id, expected_task_version=2)
    with pytest.raises(TaskStateConflict):
        store.cancel_task(
            session_id, task.task_id, expected_task_version=2, reason="too late"
        )
    with pytest.raises(TaskStateConflict):
        store.commit_ask(
            session_id, task.task_id, expected_task_version=2,
            clarification_id="clar-1",
            source_turn_id=store.allocate_turn_id(session_id),
            query="q", payload={"question": "?"}, expires_at=time.time() + 60,
        )

    failing = _new_task(store, session_id, {"budget_max": 1})
    failed = store.fail_task(
        session_id, failing.task_id, expected_task_version=1, reason="boom"
    )
    assert failed.status == "FAILED"
    with pytest.raises(TaskStateConflict):
        store.complete_task(session_id, failing.task_id, expected_task_version=2)


# --------------------------------------------------------------------------- #
# TTL
# --------------------------------------------------------------------------- #


def test_pending_ttl_is_absolute_across_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "ttl.sqlite3"
    first = SQLiteWebStateStore(db_path)
    try:
        session_id = _new_session(first)
        task = _new_task(first, session_id)
        pending = _ask(first, session_id, task, expires_in=60.0)
        expires_at = pending.expires_at
    finally:
        first.close()

    second = SQLiteWebStateStore(db_path)
    try:
        reopened = second.get_pending(pending.pending_id)
        assert reopened is not None
        assert reopened.expires_at == expires_at, "TTL must not be recomputed on restart"

        assert second.get_open_pending(task.task_id, now=expires_at - 1) is not None
        assert second.get_open_pending(task.task_id, now=expires_at + 1) is None

        with pytest.raises(PendingExpired):
            second.consume_pending(
                session_id, task.task_id, pending.pending_id,
                expected_task_version=2, now=expires_at + 1,
            )
        still_open = second.get_pending(pending.pending_id)
        assert still_open is not None and still_open.status == "OPEN"

        expired = second.expire_overdue_pending(now=expires_at + 1)
        assert [row.pending_id for row in expired] == [pending.pending_id]
        assert expired[0].status == "EXPIRED"

        cancelled = second.get_task(task.task_id)
        assert cancelled is not None
        assert cancelled.status == "CANCELLED"
        assert cancelled.close_reason == "clarification_expired"

        # A cancelled task frees the session for a new goal.
        fresh = second.create_task(session_id, initial_requirements={"budget_max": 8000})
        assert fresh.status == "RUNNING"
        assert fresh.task_id != task.task_id
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# Corruption
# --------------------------------------------------------------------------- #


def test_corrupt_task_and_pending_json_fail_explicitly(
    store: SQLiteWebStateStore,
) -> None:
    session_id = _new_session(store)
    task = _new_task(store, session_id)
    pending = _ask(store, session_id, task, clarification_id="clar-1")

    raw = sqlite3.connect(store.path)
    try:
        raw.execute(
            "UPDATE web_task SET current_requirements_json = '{not-json' WHERE task_id = ?",
            (task.task_id,),
        )
        raw.commit()
        with pytest.raises(StateDataCorruption):
            store.get_task(task.task_id)
        with pytest.raises(StateDataCorruption):
            store.get_active_task(session_id)

        raw.execute(
            "UPDATE web_pending SET payload_json = 'not-json' WHERE pending_id = ?",
            (pending.pending_id,),
        )
        raw.commit()
        with pytest.raises(StateDataCorruption):
            store.get_pending(pending.pending_id)
    finally:
        raw.close()


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #


def test_v1_database_migrates_to_v2_preserving_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _create_v1_database(db_path)

    migrated = SQLiteWebStateStore(db_path)
    try:
        assert migrated.schema_version == WEB_STATE_SCHEMA_VERSION == 2

        session = migrated.get_session("web-v1")
        assert session is not None
        assert (session.turn_counter, session.reply_counter) == (2, 3)
        assert session.reset_generation == 4
        assert session.behavior_enabled is False
        assert session.title == "Legacy session"

        assert _ledger("web-v1", db_path) == [
            "web-v1-r1", "web-v1-r2", "web-v1-r3", "web-v1-t1", "web-v1-t2"
        ]

        # New tables exist and the migrated session can use them.
        assert migrated.list_tasks("web-v1") == ()
        task = migrated.create_task("web-v1", initial_requirements={"legacy": True})
        assert task.reset_generation == 4
        assert migrated.allocate_reply_id("web-v1") == "web-v1-r4"
    finally:
        migrated.close()

    raw = sqlite3.connect(str(db_path))
    try:
        tables = {
            str(row[0])
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"web_task", "web_pending"} <= tables
        assert raw.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0] == "2"
    finally:
        raw.close()


def test_failed_migration_rolls_back_and_keeps_v1(tmp_path: Path) -> None:
    db_path = tmp_path / "broken_migration.sqlite3"
    _create_v1_database(db_path)

    # An incompatible table with the name the migration needs makes the migration fail.
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("CREATE TABLE web_pending (x INTEGER)")
        raw.commit()
    finally:
        raw.close()

    with pytest.raises(StateMigrationError):
        SQLiteWebStateStore(db_path)

    raw = sqlite3.connect(str(db_path))
    try:
        version = raw.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "1", "a failed migration must not bump the version"
        tables = {
            str(row[0])
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "web_task" not in tables, "the migration must roll back atomically"
        assert raw.execute("SELECT COUNT(*) FROM web_session").fetchone()[0] == 1
        assert raw.execute("SELECT COUNT(*) FROM web_identity").fetchone()[0] == 5
    finally:
        raw.close()


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_two_connections_create_task_only_one_wins(tmp_path: Path) -> None:
    db_path = tmp_path / "task_race.sqlite3"
    setup = SQLiteWebStateStore(db_path)
    try:
        _new_session(setup)
    finally:
        setup.close()

    barrier = threading.Barrier(2)
    created: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        connection = SQLiteWebStateStore(db_path)
        try:
            barrier.wait(timeout=30)
            task = connection.create_task("web-s2b", initial_requirements={"x": 1})
            created.append(task.task_id)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted
            errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert len(created) == 1, f"expected exactly one task, got {created}"
    assert len(errors) == 1
    assert isinstance(errors[0], ActiveTaskExists)

    final = SQLiteWebStateStore(db_path)
    try:
        tasks = final.list_tasks("web-s2b")
        assert len(tasks) == 1
        assert tasks[0].task_id == created[0]
    finally:
        final.close()


def test_two_connections_ask_only_one_open_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "ask_race.sqlite3"
    setup = SQLiteWebStateStore(db_path)
    try:
        session_id = _new_session(setup)
        task = _new_task(setup, session_id)
        turn_ids = [setup.allocate_turn_id(session_id), setup.allocate_turn_id(session_id)]
    finally:
        setup.close()

    barrier = threading.Barrier(2)
    created: list[str] = []
    errors: list[BaseException] = []
    expires_at = time.time() + 600

    def worker(turn_id: str) -> None:
        connection = SQLiteWebStateStore(db_path)
        try:
            barrier.wait(timeout=30)
            pending = connection.commit_ask(
                session_id, task.task_id, expected_task_version=1,
                clarification_id=f"clar-{turn_id}", source_turn_id=turn_id,
                query="Recommend a laptop under 5000 for me.",
                payload={"question": "?"}, expires_at=expires_at,
            )
            created.append(pending.pending_id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=worker, args=(turn_id,)) for turn_id in turn_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert len(created) == 1, f"expected exactly one ASK, got {created}"
    assert len(errors) == 1
    assert isinstance(errors[0], (StaleTaskVersion, TaskStateConflict, PendingStateConflict))

    final = SQLiteWebStateStore(db_path)
    try:
        rows = final.list_pending(task.task_id)
        assert len(rows) == 1
        assert rows[0].status == "OPEN"
        reopened = final.get_task(task.task_id)
        assert reopened is not None
        assert reopened.status == "WAITING_FOR_USER"
        assert reopened.task_version == 2
        assert final.get_session(session_id).reply_counter == 1
    finally:
        final.close()


# --------------------------------------------------------------------------- #
# Real OS process restart
# --------------------------------------------------------------------------- #


def test_os_process_restart_recovers_task_and_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "process_task.sqlite3"

    written = _run_child(tmp_path, db_path, "task_write")
    inspected = _run_child(tmp_path, db_path, "task_inspect", written["task_id"])

    assert written["pid"] != inspected["pid"]
    assert os.getpid() not in {written["pid"], inspected["pid"]}

    for field in (
        "session_id", "task_id", "task_status", "task_version",
        "pending_id", "reply_turn_id", "expires_at", "query", "question",
    ):
        assert inspected[field] == written[field], f"{field} changed across processes"

    assert written["task_status"] == "WAITING_FOR_USER"
    assert written["task_version"] == 2
    assert written["question"] == "Office or gaming?"
    assert written["session_id"] == "web-os"
