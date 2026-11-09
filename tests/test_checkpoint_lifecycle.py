"""Phase 0 regression tests: checkpoint lifecycle, privacy and robustness.

These tests target the failures found in the incremental audit.  They are intentionally
small and use the existing synthetic demo harness; no model, training or full evaluation is
invoked.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.agent_service import CLARIFICATION_TTL_SECONDS  # noqa: E402
from recommendation.demo.checkpoint import DurableCheckpointStore  # noqa: E402
from tests.demo_fixture import PROFILE_A, build_harness, close_harness  # noqa: E402


ENDPOINT = "/v1/demo/sessions/{session}/agent/shopping"


def _session_kwargs(session_id: str, *, index: int = 1) -> dict:
    now = time.time()
    return {
        "session_id": session_id,
        "user_key": f"demo-session:{session_id}",
        "profile_id": "demo-profile-1",
        "trusted_user_history": ("B000000001",),
        "created_at": now,
        "creation_index": index,
        "next_turn_sequence": 0,
        "decision_mode": "deterministic",
        "last_presentation": (),
        "last_presentation_turn": None,
    }


def _preference_request() -> dict:
    return {
        "message": "recommend gear under a tight budget",
        "k": 3,
        "sources": ["history"],
        "preference_discovery": True,
    }


def _db_rows(database: Path, query: str, parameters: tuple = ()) -> list[sqlite3.Row]:
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, parameters).fetchall()
    finally:
        connection.close()


def _pending_payload(database: Path, clarification_id: str) -> dict:
    rows = _db_rows(
        database,
        "SELECT payload_json FROM pending WHERE clarification_id = ?",
        (clarification_id,),
    )
    assert rows, "pending row is missing"
    return json.loads(rows[0]["payload_json"])


def test_retired_session_cannot_be_resurrected_by_a_stale_save(tmp_path: Path) -> None:
    store = DurableCheckpointStore(tmp_path / "checkpoint.sqlite3")
    try:
        store.save_session(**_session_kwargs("session-retired"))
        store.mark_session_retired("session-retired")
        assert store.load_sessions() == ()

        # This is the "late old process snapshot" case: a normal upsert must not revive
        # a row whose durable retired flag is already set.
        store.save_session(**_session_kwargs("session-retired", index=2))
        assert store.load_sessions() == ()
    finally:
        store.close()


def test_retire_cleans_pending_and_task_state_idempotently(tmp_path: Path) -> None:
    store = DurableCheckpointStore(tmp_path / "checkpoint.sqlite3")
    try:
        store.save_session(**_session_kwargs("session-cleanup"))
        store.save_pending(
            "clarification-cleanup",
            "session-cleanup",
            {"session_id": "session-cleanup", "created_at": time.time()},
            created_at=time.time(),
        )
        store.save_task_state(
            "session-cleanup",
            preferences=[],
            rounds=1,
            second_enabled=False,
            clarified=True,
        )

        store.mark_session_retired("session-cleanup")
        store.mark_session_retired("session-cleanup")  # idempotent

        # A stale process must not recreate task-scoped rows after retirement.
        store.save_pending(
            "late-clarification",
            "session-cleanup",
            {"session_id": "session-cleanup", "created_at": time.time()},
            created_at=time.time(),
        )
        store.save_task_state(
            "session-cleanup",
            preferences=[],
            rounds=2,
            second_enabled=True,
            clarified=True,
        )

        assert store.load_sessions() == ()
        assert store.load_pending() == ()
        assert store.load_task_states() == ()
    finally:
        store.close()


def test_reset_removes_pending_and_task_rows_after_restart(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    first = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = first.create_session(PROFILE_A)["session_id"]
        question = first.client.post(
            ENDPOINT.format(session=session), json=_preference_request()
        ).json()
        assert question["clarification"] is not None, question
        assert _db_rows(checkpoint, "SELECT COUNT(*) AS n FROM pending")[0]["n"] == 1
        assert _db_rows(checkpoint, "SELECT COUNT(*) AS n FROM task_state")[0]["n"] >= 1

        reset = first.client.delete(f"/v1/demo/sessions/{session}")
        assert reset.status_code == 200, reset.text
        service = first.runtime.agent_service()
        assert not service._pending_clarifications
        assert session not in service._task_preferences
    finally:
        close_harness(first)

    assert _db_rows(checkpoint, "SELECT COUNT(*) AS n FROM pending")[0]["n"] == 0
    assert _db_rows(checkpoint, "SELECT COUNT(*) AS n FROM task_state")[0]["n"] == 0

    second = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        assert second.client.get(f"/v1/demo/sessions/{session}").status_code == 404
    finally:
        close_harness(second)


def test_expired_pending_is_purged_and_not_restored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    first = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = first.create_session(PROFILE_A)["session_id"]
    finally:
        close_harness(first)

    store = DurableCheckpointStore(checkpoint)
    try:
        store.save_pending(
            "expired-clarification",
            session,
            {
                "session_id": session,
                "kind": "preference",
                "created_at": time.time() - CLARIFICATION_TTL_SECONDS - 60,
            },
            created_at=time.time() - CLARIFICATION_TTL_SECONDS - 60,
        )
        assert store.purge_expired_pending(CLARIFICATION_TTL_SECONDS) == 1
        assert store.load_pending() == ()

        with sqlite3.connect(str(checkpoint)) as connection:
            connection.execute(
                "INSERT INTO pending_quarantine("
                "clarification_id, task_id, payload_json, created_at, error, quarantined_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "old-quarantine",
                    session,
                    "{not-json",
                    time.time() - CLARIFICATION_TTL_SECONDS - 60,
                    "JSONDecodeError: test",
                    time.time(),
                ),
            )
            connection.commit()
        store.purge_expired_pending(CLARIFICATION_TTL_SECONDS)
        assert store.load_quarantined_pending() == ()
    finally:
        store.close()

    second = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        service = second.runtime.agent_service()
        assert "expired-clarification" not in service._pending_clarifications
        assert _db_rows(checkpoint, "SELECT COUNT(*) AS n FROM pending")[0]["n"] == 0
    finally:
        close_harness(second)


def test_corrupt_pending_is_quarantined_without_losing_other_rows(tmp_path: Path) -> None:
    database = tmp_path / "checkpoint.sqlite3"
    store = DurableCheckpointStore(database)
    try:
        store.save_pending(
            "valid-clarification",
            "task-valid",
            {"session_id": "task-valid", "created_at": time.time()},
            created_at=time.time(),
        )
        with sqlite3.connect(str(database)) as connection:
            connection.execute(
                "INSERT INTO pending(clarification_id, task_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("broken-clarification", "task-broken", "{not-json", time.time()),
            )
            connection.commit()

        pending = store.load_pending()
        assert [clarification_id for clarification_id, _ in pending] == [
            "valid-clarification"
        ]
        quarantined = store.load_quarantined_pending()
        assert [row["clarification_id"] for row in quarantined] == [
            "broken-clarification"
        ]
        assert "json" in quarantined[0]["error"].lower()
    finally:
        store.close()


def test_corrupt_pending_does_not_block_valid_restart_answer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    first = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = first.create_session(PROFILE_A)["session_id"]
        question = first.client.post(
            ENDPOINT.format(session=session), json=_preference_request()
        ).json()
        assert question["clarification"] is not None, question
        clarification_id = question["clarification"]["clarification_id"]
        option = next(
            item for item in question["preference"]["options"]
            if item["action"] in {"lower", "higher", "value"}
        )
    finally:
        close_harness(first)

    with sqlite3.connect(str(checkpoint)) as connection:
        connection.execute(
            "INSERT INTO pending(clarification_id, task_id, payload_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("broken-pending", "broken-task", "{not-json", time.time()),
        )
        connection.commit()

    second = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        answer = second.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": option["label"],
                "k": 3,
                "sources": ["history"],
                "clarification_id": clarification_id,
                "clarification_option": option["option_id"],
            },
        )
        assert answer.status_code == 200, answer.text
    finally:
        close_harness(second)


def test_new_pending_payload_does_not_duplicate_the_question(tmp_path: Path) -> None:
    checkpoint = tmp_path / "durable.sqlite3"
    harness = build_harness(tmp_path, checkpoint_db=checkpoint)
    try:
        session = harness.create_session(PROFILE_A)["session_id"]
        question = harness.client.post(
            ENDPOINT.format(session=session), json=_preference_request()
        ).json()
        clarification_id = question["clarification"]["clarification_id"]
        assert clarification_id
        payload = _pending_payload(checkpoint, clarification_id)
        # The question is already returned to the caller; it is not needed to resume.
        assert "question" not in payload
        # The resume path does need the original request message and a bounded base response.
        assert "message" in payload["request"]
        assert "base_response" in payload
    finally:
        close_harness(harness)


def test_busy_timeout_is_configured(tmp_path: Path) -> None:
    store = DurableCheckpointStore(tmp_path / "checkpoint.sqlite3")
    try:
        row = store._connection.execute("PRAGMA busy_timeout").fetchone()
        assert int(row[0]) >= 5000
    finally:
        store.close()


def test_old_schema_checkpoint_still_opens_and_loads(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(str(database)) as connection:
        connection.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta(key, value) VALUES ('schema_version', '1');
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_key TEXT NOT NULL UNIQUE,
                profile_id TEXT NOT NULL,
                history_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                creation_index INTEGER NOT NULL,
                next_turn_sequence INTEGER NOT NULL,
                decision_mode TEXT NOT NULL,
                last_presentation_json TEXT NOT NULL,
                last_presentation_turn TEXT,
                retired INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE pending (
                clarification_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE task_state (
                task_id TEXT PRIMARY KEY,
                preferences_json TEXT NOT NULL,
                rounds INTEGER NOT NULL,
                second_enabled INTEGER NOT NULL,
                clarified INTEGER NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, "
            "strftime('%s','now'))",
            (
                "legacy-session",
                "demo-session:legacy-session",
                "demo-profile-1",
                json.dumps(["B000000001"]),
                time.time(),
                1,
                0,
                "deterministic",
                "[]",
                None,
            ),
        )
        connection.execute(
            "INSERT INTO pending(clarification_id, task_id, payload_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                "legacy-pending",
                "legacy-session",
                json.dumps({"session_id": "legacy-session", "created_at": time.time()}),
                time.time(),
            ),
        )
        connection.commit()

    store = DurableCheckpointStore(database)
    try:
        sessions = store.load_sessions()
        assert [row["session_id"] for row in sessions] == ["legacy-session"]
        assert [row[0] for row in store.load_pending()] == ["legacy-pending"]
    finally:
        store.close()
