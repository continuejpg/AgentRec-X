"""S5a durable request ledger, claim leases and atomic finalize (real SQLite)."""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_demo.state_store import (  # noqa: E402
    WEB_STATE_SCHEMA_VERSION,
    SQLiteWebStateStore,
    RequestBusy,
    RequestConflict,
    RequestFenced,
    StaleTaskVersion,
    StateMigrationError,
    StateStoreError,)

REPLY = {"message": "For gaming.", "confirmed_by_user": True,
         "confirmation_source": "web_ui_explicit_user_reply"}


def _setup(store: SQLiteWebStateStore, sid: str = "web-r"):
    store.create_session(session_id=sid, profile_id="profile_0057")
    task = store.create_task(sid, initial_requirements={"raw_query": "q"})
    turn_id = store.allocate_turn_id(sid)
    pending = store.commit_ask(
        sid, task.task_id, expected_task_version=task.task_version,
        clarification_id="clar-1", source_turn_id=turn_id, query="q",
        payload={"question": "?"}, expires_at=time.time() + 600,
    )
    return task, pending


def _new_store(tmp_path: Path) -> SQLiteWebStateStore:
    return SQLiteWebStateStore(tmp_path / "state.sqlite3")


# --------------------------------------------------------------------------- #
# identity / conflict / busy
# --------------------------------------------------------------------------- #


def test_claim_uses_pending_reply_id_as_stable_request_id(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        assert claim.outcome == "claimed"
        assert claim.request.request_id == pending.reply_turn_id
        assert claim.request.status == "PROCESSING"
        assert claim.request.claim_attempt == 1
        assert store.get_request("web-r", pending.reply_turn_id) == claim.request
    finally:
        store.close()


def test_same_payload_retry_is_busy_and_different_payload_conflicts(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        with pytest.raises(RequestBusy):
            store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        with pytest.raises(RequestConflict):
            store.claim_reply_request(
                "web-r", pending.pending_id, payload={**REPLY, "message": "something else"}
            )
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# lease takeover + fencing
# --------------------------------------------------------------------------- #


def test_expired_lease_takeover_fences_the_old_token(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        first = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        second = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=60, now=now + 10
        )
        assert (first.outcome, second.outcome) == ("claimed", "reclaimed")
        assert second.request.claim_attempt == 2
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=first.claim_token,
                outcome="resumed", now=now + 11,
            )
        task, consumed, request = store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=second.claim_token,
            outcome="resumed", now=now + 12,
        )
        assert task.status == "RUNNING"
        assert consumed.status == "CONSUMED"
        assert request.status == "COMPLETED"
        assert request.result["outcome"] == "resumed"
    finally:
        store.close()


def test_two_connections_claim_only_one_wins(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    setup = SQLiteWebStateStore(db)
    try:
        _, pending = _setup(setup)
    finally:
        setup.close()

    outcomes: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        store = SQLiteWebStateStore(db)
        try:
            barrier.wait(timeout=30)
            claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
            outcomes.append(claim.outcome)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            store.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert outcomes == ["claimed"]
    assert len(errors) == 1 and isinstance(errors[0], RequestBusy)


# --------------------------------------------------------------------------- #
# finalize atomicity / reset fencing / replay / restart
# --------------------------------------------------------------------------- #


def test_finalize_sql_failure_rolls_back_everything(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        raw = sqlite3.connect(store.path)
        try:
            raw.execute(
                "CREATE TRIGGER fail_task BEFORE UPDATE ON web_task "
                "BEGIN SELECT RAISE(ABORT, 'injected'); END"
            )
            raw.commit()
        finally:
            raw.close()
        with pytest.raises((StateStoreError, sqlite3.Error)):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token, outcome="resumed"
            )
        raw = sqlite3.connect(store.path)
        try:
            raw.execute("DROP TRIGGER fail_task")
            raw.commit()
        finally:
            raw.close()
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
        task = store.get_active_task("web-r")
        assert task is not None and task.status == "WAITING_FOR_USER" and task.task_version == 2
    finally:
        store.close()


def test_reset_fences_a_claimed_request(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.reset_session("web-r")
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token, outcome="resumed"
            )
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
        assert store.get_open_pending_for_session("web-r") is None
    finally:
        store.close()


def test_completed_request_replays_without_re_execution_and_survives_restart(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.sqlite3"
    store = SQLiteWebStateStore(db)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, outcome="completed"
        )
    finally:
        store.close()

    reopened = SQLiteWebStateStore(db)
    try:
        replay = reopened.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        assert replay.outcome == "replay"
        assert replay.claim_token is None
        assert replay.request.status == "COMPLETED"
        assert replay.request.result["outcome"] == "completed"
    finally:
        reopened.close()


# --------------------------------------------------------------------------- #
# reset generation / stale task
# --------------------------------------------------------------------------- #


def test_stale_task_version_cannot_be_finalized(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        raw = sqlite3.connect(store.path)
        try:
            raw.execute(
                "UPDATE web_task SET task_version = task_version + 5 WHERE session_id = 'web-r'"
            )
            raw.commit()
        finally:
            raw.close()
        with pytest.raises(StaleTaskVersion):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token, outcome="resumed"
            )
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# migration v3 -> v4
# --------------------------------------------------------------------------- #


def test_v3_database_migrates_to_v4_preserving_state(tmp_path: Path) -> None:
    from web_demo.state_store import (
        _SCHEMA_V1_STATEMENTS,
        _SCHEMA_V2_STATEMENTS,
        _SCHEMA_V3_STATEMENTS,
    )

    db = tmp_path / "v3.sqlite3"
    connection = sqlite3.connect(str(db))
    try:
        for statement in (
            _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS + _SCHEMA_V3_STATEMENTS
        ):
            connection.execute(statement)
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '3')")
        connection.execute(
            "INSERT INTO web_session(session_id, profile_id, title, created_at, updated_at, "
            "behavior_enabled, turn_counter, reply_counter, reset_generation, retired) "
            "VALUES ('web-v3', 'profile_0057', 'v3', 1.0, 2.0, 1, 1, 1, 0, 0)"
        )
        connection.execute(
            "INSERT INTO web_task(task_id, session_id, status, task_version, reset_generation, "
            "current_requirements_json, requirements_version, created_at, updated_at) "
            "VALUES ('task-v3', 'web-v3', 'WAITING_FOR_USER', 2, 0, '{}', 0, 1.0, 2.0)"
        )
        connection.execute(
            "INSERT INTO web_pending(pending_id, task_id, clarification_id, query, turn_id, "
            "reply_turn_id, mode, payload_json, status, created_at, expires_at, consumed_at) "
            "VALUES ('pend-v3', 'task-v3', 'clar-v3', 'q', 'web-v3-t1', 'web-v3-r1', 'cpu_live', "
            "'{\"question\": \"?\"}', 'OPEN', 1.0, 9999999999.0, NULL)"
        )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteWebStateStore(db)
    try:
        assert store.schema_version == WEB_STATE_SCHEMA_VERSION == 6
        pending = store.get_open_pending("task-v3")
        assert pending is not None and pending.pending_id == "pend-v3"
        assert store.get_request("web-v3", "web-v3-r1") is None
        assert store.list_turns("web-v3") == ()
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# Phase A gate: generation-aware replay, lease renewal, finalize expiry
# --------------------------------------------------------------------------- #


def test_completed_replay_after_reset_is_fenced(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, outcome="completed"
        )
        store.reset_session("web-r")
        with pytest.raises(RequestFenced):
            store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        # Audit row survives; it is simply not replayable in the new generation.
        record = store.get_request("web-r", pending.reply_turn_id)
        assert record is not None and record.status == "COMPLETED"
    finally:
        store.close()


def test_renew_lease_extends_and_fences_stale_token(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        first = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        renewed = store.renew_request_lease(
            "web-r", pending.reply_turn_id, claim_token=first.claim_token,
            lease_duration=60, now=now + 4,
        )
        assert renewed.lease_expires_at == pytest.approx(now + 64)
        with pytest.raises(RequestBusy):
            store.claim_reply_request(
                "web-r", pending.pending_id, payload=REPLY, lease_duration=60, now=now + 10
            )
        second = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=60, now=now + 70
        )
        assert second.outcome == "reclaimed"
        with pytest.raises(RequestFenced):
            store.renew_request_lease(
                "web-r", pending.reply_turn_id, claim_token=first.claim_token, now=now + 71
            )
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=first.claim_token,
                outcome="resumed", now=now + 72,
            )
        task, consumed, request = store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=second.claim_token,
            outcome="resumed", now=now + 73,
        )
        assert (task.status, consumed.status, request.status) == (
            "RUNNING", "CONSUMED", "COMPLETED"
        )
    finally:
        store.close()


def test_finalize_after_lease_expiry_is_fenced(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        now = time.time()
        claim = store.claim_reply_request(
            "web-r", pending.pending_id, payload=REPLY, lease_duration=5, now=now
        )
        with pytest.raises(RequestFenced):
            store.finalize_reply_request(
                "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
                outcome="resumed", now=now + 10,
            )
        task = store.get_active_task("web-r")
        assert task is not None and task.status == "WAITING_FOR_USER" and task.task_version == 2
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.get_request("web-r", pending.reply_turn_id).status == "PROCESSING"
    finally:
        store.close()


def test_v4_migration_failure_rolls_back(tmp_path: Path) -> None:
    from web_demo.state_store import (
        _SCHEMA_V1_STATEMENTS,
        _SCHEMA_V2_STATEMENTS,
        _SCHEMA_V3_STATEMENTS,
    )

    db = tmp_path / "v3_broken.sqlite3"
    connection = sqlite3.connect(str(db))
    try:
        for statement in (
            _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS + _SCHEMA_V3_STATEMENTS
        ):
            connection.execute(statement)
        connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '3')")
        connection.execute(
            "INSERT INTO web_session(session_id, profile_id, title, created_at, updated_at, "
            "behavior_enabled, turn_counter, reply_counter, reset_generation, retired) "
            "VALUES ('web-b', 'profile_0057', 'b', 1.0, 2.0, 1, 1, 1, 0, 0)"
        )
        connection.execute(
            "INSERT INTO web_task(task_id, session_id, status, task_version, reset_generation, "
            "current_requirements_json, requirements_version, created_at, updated_at) "
            "VALUES ('task-b', 'web-b', 'WAITING_FOR_USER', 2, 0, '{}', 0, 1.0, 2.0)"
        )
        connection.execute("CREATE TABLE web_request (x INTEGER)")  # incompatible
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(StateMigrationError):
        SQLiteWebStateStore(db)

    raw = sqlite3.connect(str(db))
    try:
        assert raw.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "3"
        assert raw.execute("SELECT COUNT(*) FROM web_task").fetchone()[0] == 1
    finally:
        raw.close()


# --------------------------------------------------------------------------- #
# Reply lookup by clarification id (used by the HTTP reply route)
# --------------------------------------------------------------------------- #


def test_get_pending_for_clarification_covers_open_consumed_and_sessions(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        open_row = store.get_pending_for_clarification("web-r", pending.clarification_id)
        assert open_row is not None
        assert (open_row.pending_id, open_row.status) == (pending.pending_id, "OPEN")

        store.create_session(session_id="web-other", profile_id="profile_0057")
        assert store.get_pending_for_clarification(
            "web-other", pending.clarification_id
        ) is None
        assert store.get_pending_for_clarification("web-r", "clar-unknown") is None

        claim = store.claim_reply_request("web-r", pending.pending_id, payload=REPLY)
        store.mark_request_external_started(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token
        )
        result = {"reply": {"schema": "test"}}
        store.checkpoint_reply_result(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token, result=result
        )
        store.finalize_reply_request(
            "web-r", pending.reply_turn_id, claim_token=claim.claim_token,
            outcome="completed",
            turn={
                "turn_id": pending.reply_turn_id,
                "user_text": "For gaming.",
                "assistant_text": "Here is one match.",
                "products": {"items": []},
            },
            result=result,
        )
        consumed = store.get_pending_for_clarification("web-r", pending.clarification_id)
        assert consumed is not None and consumed.status == "CONSUMED"
        assert consumed.reply_turn_id == pending.reply_turn_id
        # The OPEN-only reader no longer sees it; the clarification reader still does.
        assert store.get_open_pending_for_session("web-r") is None
    finally:
        store.close()
