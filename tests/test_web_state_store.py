"""Tests for the durable web-session identity store (M1-C S2a).

Covers the required A-I scenarios plus session retirement and a second,
process-level concurrency check.  Every test uses a real SQLite file under pytest's
``tmp_path``; nothing here touches the production M9, Behavior or Web Demo databases.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_demo.sessions import DEFAULT_TITLE, Session  # noqa: E402
from web_demo.state_store import (  # noqa: E402
    DEFAULT_SESSION_TITLE,
    SQLiteWebStateStore,
    IdentityAllocationError,
    SchemaVersionError,
    SessionAlreadyExists,
    SessionRetired,
    StateStoreClosedError,
    StateStoreError,
    UnknownSession,
    WEB_STATE_SCHEMA_VERSION,
)

CHILD_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import sys

    repo, action, db = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, repo)
    from web_demo.state_store import SQLiteWebStateStore

    store = SQLiteWebStateStore(db)
    try:
        if action == "write":
            record = store.create_session(
                session_id="web-proc", profile_id="profile_0057"
            )
            ids = [store.allocate_reply_id("web-proc"), store.allocate_reply_id("web-proc")]
            payload = {"pid": os.getpid(), "ids": ids, "reply_counter": record.reply_counter}
        elif action == "append":
            record = store.get_session("web-proc")
            value = store.allocate_reply_id("web-proc")
            payload = {
                "pid": os.getpid(),
                "found": record is not None,
                "reply_counter_before": None if record is None else record.reply_counter,
                "id": value,
            }
        elif action == "concurrent":
            payload = {
                "pid": os.getpid(),
                "ids": [store.allocate_reply_id("web-proc") for _ in range(5)],
            }
        else:
            record = store.get_session("web-proc")
            payload = {
                "pid": os.getpid(),
                "reply_counter": None if record is None else record.reply_counter,
                "turn_counter": None if record is None else record.turn_counter,
                "reset_generation": None if record is None else record.reset_generation,
                "retired": None if record is None else record.retired,
            }
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


def _ledger_rows(path: str | Path) -> list[tuple[str, int, str]]:
    connection = sqlite3.connect(str(path))
    try:
        return [
            (str(row[0]), int(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT kind, seq, value FROM web_identity ORDER BY kind, seq"
            )
        ]
    finally:
        connection.close()


def _child_script(tmp_path: Path) -> Path:
    path = tmp_path / "identity_child.py"
    path.write_text(CHILD_SCRIPT, encoding="utf-8")
    return path


def _run_child(tmp_path: Path, db_path: Path, action: str) -> dict:
    script = _child_script(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(REPO_ROOT), action, str(db_path)],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"child {action} failed: {result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def _spawn_child(tmp_path: Path, db_path: Path, action: str) -> subprocess.Popen:
    script = _child_script(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.Popen(
        [sys.executable, "-B", str(script), str(REPO_ROOT), action, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        cwd=str(tmp_path),
    )


# --------------------------------------------------------------------------- #
# Test A: create and read
# --------------------------------------------------------------------------- #


def test_create_and_read_session(store: SQLiteWebStateStore) -> None:
    created = store.create_session(
        session_id="web-a", profile_id="profile_0057", behavior_enabled=False
    )

    assert created.session_id == "web-a"
    assert created.profile_id == "profile_0057"
    assert created.behavior_enabled is False
    assert created.title == DEFAULT_SESSION_TITLE == DEFAULT_TITLE
    assert (created.turn_counter, created.reply_counter) == (0, 0)
    assert created.reset_generation == 0
    assert created.retired is False
    assert created.created_at > 0
    assert created.updated_at >= created.created_at

    fetched = store.get_session("web-a")
    assert fetched == created
    assert store.get_session("web-missing") is None

    with pytest.raises(SessionAlreadyExists):
        store.create_session(session_id="web-a", profile_id="profile_0049")
    with pytest.raises(ValueError):
        store.create_session(session_id="   ", profile_id="profile_0057")


def test_retired_session_cannot_allocate(store: SQLiteWebStateStore) -> None:
    store.create_session(session_id="web-retire", profile_id="profile_0057")

    retired = store.retire_session("web-retire")
    assert retired.retired is True
    assert store.retire_session("web-retire").retired is True  # idempotent

    with pytest.raises(SessionRetired):
        store.allocate_reply_id("web-retire")
    with pytest.raises(SessionRetired):
        store.advance_reset_generation("web-retire")
    with pytest.raises(UnknownSession):
        store.retire_session("web-missing")


# --------------------------------------------------------------------------- #
# Test B: sequential allocation
# --------------------------------------------------------------------------- #


def test_sequential_allocation_is_unique(store: SQLiteWebStateStore) -> None:
    store.create_session(session_id="web-b", profile_id="profile_0057")

    turn_ids = [store.allocate_turn_id("web-b") for _ in range(3)]
    reply_ids = [store.allocate_reply_id("web-b") for _ in range(3)]

    assert turn_ids == ["web-b-t1", "web-b-t2", "web-b-t3"]
    assert reply_ids == ["web-b-r1", "web-b-r2", "web-b-r3"]
    assert len(set(turn_ids + reply_ids)) == 6

    rows = _ledger_rows(store.path)
    assert len(rows) == 6
    assert [row for row in rows if row[0] == "turn"] == [
        ("turn", 1, "web-b-t1"), ("turn", 2, "web-b-t2"), ("turn", 3, "web-b-t3")
    ]
    assert [row for row in rows if row[0] == "reply"] == [
        ("reply", 1, "web-b-r1"), ("reply", 2, "web-b-r2"), ("reply", 3, "web-b-r3")
    ]

    record = store.get_session("web-b")
    assert record is not None
    assert (record.turn_counter, record.reply_counter) == (3, 3)


# --------------------------------------------------------------------------- #
# Test C: reset advances the generation but never rewinds identity
# --------------------------------------------------------------------------- #


def test_reset_generation_keeps_identities(store: SQLiteWebStateStore) -> None:
    store.create_session(
        session_id="web-c", profile_id="profile_0057", behavior_enabled=False
    )
    turn_ids = [store.allocate_turn_id("web-c") for _ in range(2)]
    reply_ids = [store.allocate_reply_id("web-c") for _ in range(2)]
    ledger_before = _ledger_rows(store.path)

    advanced = store.advance_reset_generation("web-c")

    assert advanced.session_id == "web-c"
    assert advanced.reset_generation == 1
    assert (advanced.turn_counter, advanced.reply_counter) == (2, 2)
    assert advanced.behavior_enabled is False

    new_turn = store.allocate_turn_id("web-c")
    new_reply = store.allocate_reply_id("web-c")

    assert new_turn not in set(turn_ids)
    assert new_reply not in set(reply_ids)
    assert (new_turn, new_reply) == ("web-c-t3", "web-c-r3")

    ledger_after = _ledger_rows(store.path)
    assert set(ledger_before).issubset(set(ledger_after))
    assert len(ledger_after) == 6

    final = store.get_session("web-c")
    assert final is not None
    assert final.reset_generation == 1
    assert (final.turn_counter, final.reply_counter) == (3, 3)


# --------------------------------------------------------------------------- #
# Test D: close and reopen in the same process
# --------------------------------------------------------------------------- #


def test_reopen_same_process_continues_sequence(tmp_path: Path) -> None:
    db_path = tmp_path / "reopen.sqlite3"

    first = SQLiteWebStateStore(db_path)
    try:
        first.create_session(session_id="web-d", profile_id="profile_0057")
        allocated = [first.allocate_reply_id("web-d"), first.allocate_reply_id("web-d")]
    finally:
        first.close()

    second = SQLiteWebStateStore(db_path)
    try:
        record = second.get_session("web-d")
        assert record is not None
        assert record.reply_counter == 2
        reused = second.allocate_reply_id("web-d")
        assert reused == "web-d-r3"
        assert reused not in set(allocated)
    finally:
        second.close()

    assert [row[2] for row in _ledger_rows(db_path)] == ["web-d-r1", "web-d-r2", "web-d-r3"]


# --------------------------------------------------------------------------- #
# Test E: real OS process restart
# --------------------------------------------------------------------------- #


def test_os_process_restart_keeps_counters(tmp_path: Path) -> None:
    db_path = tmp_path / "process.sqlite3"

    written = _run_child(tmp_path, db_path, "write")
    appended = _run_child(tmp_path, db_path, "append")
    inspected = _run_child(tmp_path, db_path, "inspect")

    pids = {written["pid"], appended["pid"], inspected["pid"]}
    assert len(pids) == 3, f"expected three distinct OS processes, got {pids}"
    assert os.getpid() not in pids

    assert written["ids"] == ["web-proc-r1", "web-proc-r2"]

    assert appended["found"] is True
    assert appended["reply_counter_before"] == 2
    assert appended["id"] == "web-proc-r3"

    assert inspected["reply_counter"] == 3
    assert inspected["turn_counter"] == 0
    assert inspected["reset_generation"] == 0
    assert inspected["retired"] is False

    assert [row[2] for row in _ledger_rows(db_path)] == [
        "web-proc-r1", "web-proc-r2", "web-proc-r3"
    ]


def test_os_processes_allocate_concurrently(tmp_path: Path) -> None:
    db_path = tmp_path / "process-concurrent.sqlite3"
    setup = SQLiteWebStateStore(db_path)
    try:
        setup.create_session(session_id="web-proc", profile_id="profile_0057")
    finally:
        setup.close()

    children = [_spawn_child(tmp_path, db_path, "concurrent") for _ in range(2)]
    outputs = []
    for child in children:
        stdout, stderr = child.communicate(timeout=120)
        assert child.returncode == 0, f"child failed: {stderr}"
        outputs.append(json.loads(stdout.strip().splitlines()[-1]))

    allocated = outputs[0]["ids"] + outputs[1]["ids"]
    assert len(allocated) == 10
    assert len(set(allocated)) == 10, "two OS processes received a duplicate identity"

    final = SQLiteWebStateStore(db_path)
    try:
        record = final.get_session("web-proc")
        assert record is not None
        assert record.reply_counter == 10
    finally:
        final.close()
    assert len(_ledger_rows(db_path)) == 10


# --------------------------------------------------------------------------- #
# Test F: two independent connections, controlled concurrency
# --------------------------------------------------------------------------- #


def test_two_connections_allocate_concurrently(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.sqlite3"
    setup = SQLiteWebStateStore(db_path)
    try:
        setup.create_session(session_id="web-conc", profile_id="profile_0057")
    finally:
        setup.close()

    per_worker = 25
    barrier = threading.Barrier(2)
    results: dict[int, list[str]] = {0: [], 1: []}
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        connection = SQLiteWebStateStore(db_path)
        try:
            barrier.wait(timeout=30)
            results[index] = [
                connection.allocate_reply_id("web-conc") for _ in range(per_worker)
            ]
        except BaseException as exc:  # noqa: BLE001 - collected and asserted
            errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=worker, args=(index,)) for index in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == [], f"concurrent allocation raised: {errors!r}"

    allocated = results[0] + results[1]
    assert len(allocated) == 2 * per_worker
    assert len(set(allocated)) == 2 * per_worker, "duplicate identity under concurrency"

    rows = _ledger_rows(db_path)
    assert len(rows) == 2 * per_worker
    assert sorted(row[1] for row in rows) == list(range(1, 2 * per_worker + 1))

    final = SQLiteWebStateStore(db_path)
    try:
        record = final.get_session("web-conc")
        assert record is not None
        assert record.reply_counter == 2 * per_worker
    finally:
        final.close()


# --------------------------------------------------------------------------- #
# Test G: a failed allocation rolls the whole unit back
# --------------------------------------------------------------------------- #


def test_failed_allocation_rolls_back(store: SQLiteWebStateStore) -> None:
    store.create_session(session_id="web-g", profile_id="profile_0057")

    raw = sqlite3.connect(store.path)
    try:
        raw.execute(
            "CREATE TRIGGER fail_identity BEFORE INSERT ON web_identity "
            f"WHEN NEW.value = 'web-g-t1' "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        raw.commit()

        with pytest.raises(IdentityAllocationError):
            store.allocate_turn_id("web-g")

        after_failure = store.get_session("web-g")
        assert after_failure is not None
        assert after_failure.turn_counter == 0, "the counter must roll back with the ledger"
        assert raw.execute("SELECT COUNT(*) FROM web_identity").fetchone()[0] == 0

        raw.execute("DROP TRIGGER fail_identity")
        raw.commit()
    finally:
        raw.close()

    assert store.allocate_turn_id("web-g") == "web-g-t1"
    after_retry = store.get_session("web-g")
    assert after_retry is not None
    assert after_retry.turn_counter == 1
    assert len(_ledger_rows(store.path)) == 1


# --------------------------------------------------------------------------- #
# Test H: corruption and incompatible schema
# --------------------------------------------------------------------------- #


def test_incompatible_schema_version_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "version.sqlite3"
    created = SQLiteWebStateStore(db_path)
    try:
        assert created.schema_version == WEB_STATE_SCHEMA_VERSION
        created.create_session(session_id="web-v", profile_id="profile_0057")
    finally:
        created.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
        raw.commit()
    finally:
        raw.close()

    with pytest.raises(SchemaVersionError):
        SQLiteWebStateStore(db_path)

    # The original database must not be rebuilt or overwritten.
    raw = sqlite3.connect(db_path)
    try:
        value = raw.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert value == "999"
        assert raw.execute("SELECT COUNT(*) FROM web_session").fetchone()[0] == 1
    finally:
        raw.close()


def test_non_sqlite_file_is_not_overwritten(tmp_path: Path) -> None:
    db_path = tmp_path / "junk.sqlite3"
    payload = b"this file is definitely not a sqlite database"
    db_path.write_bytes(payload)

    with pytest.raises(StateStoreError):
        SQLiteWebStateStore(db_path)

    assert db_path.read_bytes() == payload


def test_foreign_sqlite_database_is_not_adopted(tmp_path: Path) -> None:
    db_path = tmp_path / "foreign.sqlite3"
    raw = sqlite3.connect(db_path)
    try:
        raw.execute("CREATE TABLE other_app(id INTEGER PRIMARY KEY, note TEXT)")
        raw.execute("INSERT INTO other_app(note) VALUES ('keep me')")
        raw.commit()
    finally:
        raw.close()

    with pytest.raises(StateStoreError):
        SQLiteWebStateStore(db_path)

    raw = sqlite3.connect(db_path)
    try:
        tables = {
            str(row[0])
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "other_app" in tables
        assert "web_session" not in tables
        assert raw.execute("SELECT note FROM other_app").fetchone()[0] == "keep me"
    finally:
        raw.close()


def test_closed_store_raises_instead_of_returning_none(tmp_path: Path) -> None:
    instance = SQLiteWebStateStore(tmp_path / "closed.sqlite3")
    instance.close()
    instance.close()  # idempotent

    with pytest.raises(StateStoreClosedError):
        instance.get_session("web-any")
    with pytest.raises(StateStoreClosedError):
        instance.allocate_turn_id("web-any")


def test_unknown_session_is_explicit(store: SQLiteWebStateStore) -> None:
    assert store.get_session("web-unknown") is None
    with pytest.raises(UnknownSession):
        store.allocate_turn_id("web-unknown")
    with pytest.raises(UnknownSession):
        store.allocate_reply_id("web-unknown")
    with pytest.raises(UnknownSession):
        store.advance_reset_generation("web-unknown")


# --------------------------------------------------------------------------- #
# Test I: compatibility with the existing Session dataclass
# --------------------------------------------------------------------------- #


def test_session_dataclass_uses_persisted_counters(store: SQLiteWebStateStore) -> None:
    store.create_session(session_id="web-i", profile_id="profile_0057")
    turn_ids = [store.allocate_turn_id("web-i") for _ in range(2)]
    reply_ids = [store.allocate_reply_id("web-i") for _ in range(2)]

    record = store.get_session("web-i")
    assert record is not None

    session = Session(
        session_id=record.session_id,
        profile_id=record.profile_id,
        title=record.title,
        behavior_enabled=record.behavior_enabled,
        _turn_counter=record.turn_counter,
        _reply_counter=record.reply_counter,
    )

    next_turn = session.next_turn_id()
    next_reply = session.next_reply_turn_id()

    assert next_turn == "web-i-t3"
    assert next_reply == "web-i-r3"
    assert next_turn not in set(turn_ids)
    assert next_reply not in set(reply_ids)
