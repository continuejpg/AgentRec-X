"""SQLite-backed durable checkpoints for the demo session/task layer.

The accepted preference memory already persists explicit preferences.  It deliberately does
**not** persist the live demo session, pending questions or task-scoped preferences.  This
module adds that missing layer behind an explicit store:

* one row per session (profile, trusted history, turn sequence, decision mode, last shown cards);
* one row per pending clarification (opaque id, task key, JSON payload, creation time);
* one row per task (task preferences, round counters, clarified marker).

SQLite is used because the repository already ships it and a local demo must not require a
new service.  The interface is deliberately narrow so a deployment can replace this store
with Redis/Postgres or a durable workflow engine without changing the session/agent code.

Persistence contract (what is actually on disk)
-----------------------------------------------
Session rows contain the trusted history and the last-shown cards.  Pending rows are bounded
JSON, but they are **not free of user text**: a pending preference clarification needs its
original ``request.message`` and a bounded base response (including the rendered response
``message`` and product identities) to resume deterministically; a research pending row
carries its structured clarification payload and observation.  The duplicate ``question``
copy and unused diagnostic terms are deliberately not persisted.  Pending and quarantined
rows are deleted when consumed, when their task/session is reset, or when they
exceed the Agent's clarification TTL.  Session and task rows have no automatic TTL: a reset
marks the session retired (so it is never restored) and deletes its task/pending rows; the
retired session row remains as local audit state.  Corrupt pending rows are moved to
``pending_quarantine`` with the raw payload and a diagnostic error instead of being silently
discarded.  There is no encryption; the database is a local runtime artifact and should be
protected like the preference-memory database next to it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_BUSY_TIMEOUT_MS",
    "DurableCheckpointStore",
    "DurableStateError",
]


CHECKPOINT_SCHEMA_VERSION = 1

#: SQLite waits for a competing writer instead of failing immediately with "database is
#: locked".  This is a local-demo safety valve, not a cross-process lease.
DEFAULT_BUSY_TIMEOUT_MS = 5_000


class DurableStateError(RuntimeError):
    """The checkpoint database is unavailable or has an unsupported schema."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot checkpoint value of type {type(value).__name__}")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


class DurableCheckpointStore:
    """Transactional SQLite checkpoint store.

    The store is safe to share between threads in one process (one connection + RLock,
    ``check_same_thread=False``).  A multi-process deployment should use one database path
    shared by workers; the demo still keeps per-session turn serialization in process, so
    multi-worker correctness requires a shared lease in a later phase.
    """

    def __init__(self, database: Path | str) -> None:
        self._database = str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                self._database, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise DurableStateError(f"cannot open checkpoint database: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._closed = False
        self._ensure_schema()

    @property
    def database(self) -> str:
        return self._database

    def _ensure_schema(self) -> None:
        with self._lock:
            try:
                self._connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
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
                    CREATE TABLE IF NOT EXISTS pending (
                        clarification_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS pending_task_idx ON pending(task_id);
                    -- Additive table: schema_version stays 1 so checkpoint databases written
                    -- by the previous implementation remain readable.  No column is changed.
                    CREATE TABLE IF NOT EXISTS pending_quarantine (
                        clarification_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        error TEXT NOT NULL,
                        quarantined_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS task_state (
                        task_id TEXT PRIMARY KEY,
                        preferences_json TEXT NOT NULL,
                        rounds INTEGER NOT NULL,
                        second_enabled INTEGER NOT NULL,
                        clarified INTEGER NOT NULL,
                        updated_at REAL NOT NULL,
                        r1_anchor_json TEXT NOT NULL DEFAULT '[]'
                    );
                    """
                )
                # Additive R1 query-anchor persistence.  Existing checkpoint databases
                # (schema version 1) are migrated by adding only this defaulted column;
                # no task preference or pending row is rewritten.
                task_columns = {
                    str(row["name"])
                    for row in self._connection.execute("PRAGMA table_info(task_state)")
                }
                if "r1_anchor_json" not in task_columns:
                    self._connection.execute(
                        "ALTER TABLE task_state ADD COLUMN r1_anchor_json TEXT NOT NULL DEFAULT '[]'"
                    )
                row = self._connection.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                        (str(CHECKPOINT_SCHEMA_VERSION),),
                    )
                elif int(row["value"]) != CHECKPOINT_SCHEMA_VERSION:
                    raise DurableStateError(
                        f"checkpoint database {self._database!r} has schema version "
                        f"{row['value']}, expected {CHECKPOINT_SCHEMA_VERSION}"
                    )
            except sqlite3.Error as exc:
                raise DurableStateError(f"cannot initialise checkpoint schema: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    # -- sessions --------------------------------------------------------- #

    def save_session(
        self,
        *,
        session_id: str,
        user_key: str,
        profile_id: str,
        trusted_user_history: Sequence[str],
        created_at: float,
        creation_index: int,
        next_turn_sequence: int,
        decision_mode: str,
        last_presentation: Sequence[Mapping[str, Any]] = (),
        last_presentation_turn: str | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, user_key, profile_id, history_json, created_at,
                        creation_index, next_turn_sequence, decision_mode,
                        last_presentation_json, last_presentation_turn, retired, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, strftime('%s','now'))
                    ON CONFLICT(session_id) DO UPDATE SET
                        user_key=excluded.user_key,
                        profile_id=excluded.profile_id,
                        history_json=excluded.history_json,
                        created_at=excluded.created_at,
                        creation_index=excluded.creation_index,
                        next_turn_sequence=excluded.next_turn_sequence,
                        decision_mode=excluded.decision_mode,
                        last_presentation_json=excluded.last_presentation_json,
                        last_presentation_turn=excluded.last_presentation_turn,
                        updated_at=excluded.updated_at
                    WHERE sessions.retired = 0
                    """,
                    (
                        session_id,
                        user_key,
                        profile_id,
                        _dumps(list(trusted_user_history)),
                        float(created_at),
                        int(creation_index),
                        int(next_turn_sequence),
                        decision_mode,
                        _dumps(list(last_presentation)),
                        last_presentation_turn,
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def mark_session_retired(self, session_id: str) -> None:
        """Retire a session and delete its task-scoped pending/task rows atomically.

        The durable retired flag is the authority: an old process that writes a normal
        session snapshot afterwards must not be able to clear it (see the conditional upsert
        in :meth:`save_session`).  This method is idempotent.
        """
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "UPDATE sessions SET retired=1, updated_at=strftime('%s','now') "
                    "WHERE session_id=?",
                    (session_id,),
                )
                self._connection.execute("DELETE FROM pending WHERE task_id=?", (session_id,))
                self._connection.execute(
                    "DELETE FROM task_state WHERE task_id=?", (session_id,)
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def load_sessions(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM sessions WHERE retired=0 ORDER BY creation_index, session_id"
            ).fetchall()
        return tuple(
            {
                "session_id": row["session_id"],
                "user_key": row["user_key"],
                "profile_id": row["profile_id"],
                "trusted_user_history": tuple(json.loads(row["history_json"])),
                "created_at": float(row["created_at"]),
                "creation_index": int(row["creation_index"]),
                "next_turn_sequence": int(row["next_turn_sequence"]),
                "decision_mode": row["decision_mode"],
                "last_presentation": tuple(json.loads(row["last_presentation_json"])),
                "last_presentation_turn": row["last_presentation_turn"],
            }
            for row in rows
        )

    # -- pending questions ------------------------------------------------ #

    def _task_is_retired_locked(self, task_id: str) -> bool:
        """True when the task key names a durably retired session.

        The current demo uses the session id as the task key; a checkpoint written under a
        different task key cannot be lifecycle-routed by this check and is cleaned by the
        session-level sweep only when its task key matches.
        """
        row = self._connection.execute(
            "SELECT retired FROM sessions WHERE session_id=?", (task_id,)
        ).fetchone()
        return bool(row is not None and int(row["retired"]) != 0)

    def save_pending(
        self,
        clarification_id: str,
        task_id: str,
        payload: Mapping[str, Any],
        *,
        created_at: float,
    ) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                # A late write from an old process must not re-create a pending row for a
                # session that has already been retired/reset.  BEGIN IMMEDIATE serializes
                # this check with mark_session_retired() in the same database.
                if self._task_is_retired_locked(task_id):
                    self._connection.execute("COMMIT")
                    return
                self._connection.execute(
                    """
                    INSERT INTO pending(clarification_id, task_id, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(clarification_id) DO UPDATE SET
                        task_id=excluded.task_id,
                        payload_json=excluded.payload_json,
                        created_at=excluded.created_at
                    """,
                    (clarification_id, task_id, _dumps(dict(payload)), float(created_at)),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def delete_pending(self, clarification_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM pending WHERE clarification_id=?", (clarification_id,)
            )

    def clear_pending_for_task(self, task_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM pending WHERE task_id=?", (task_id,))

    def clear_task_state(self, task_id: str) -> None:
        """Delete one task's preference state.  Used when its session is reset."""
        with self._lock:
            self._connection.execute("DELETE FROM task_state WHERE task_id=?", (task_id,))

    def purge_expired_pending(self, ttl_seconds: float, *, now: float | None = None) -> int:
        """Delete pending questions older than ``ttl_seconds`` and return the row count.

        The Agent still performs a TTL check when answering; this method is the lifecycle
        sweep so expired rows do not accumulate and are not restored after a restart.
        """
        if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        cutoff = float(now if now is not None else time.time()) - float(ttl_seconds)
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM pending WHERE created_at < ?", (cutoff,)
            )
            # Quarantined rows contain the same raw payload and are diagnostics, not a
            # replay queue.  Apply the same retention cutoff so they do not grow forever.
            self._connection.execute(
                "DELETE FROM pending_quarantine WHERE created_at < ?", (cutoff,)
            )
            return int(cursor.rowcount)

    def load_pending(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Load valid pending rows, quarantining corrupt rows instead of aborting.

        A malformed pending row belongs to one task and must not prevent unrelated sessions
        from restoring.  The malformed row is moved verbatim into ``pending_quarantine`` with
        its diagnostic error and is never silently deleted.
        """
        with self._lock:
            rows = self._connection.execute(
                "SELECT clarification_id, task_id, payload_json, created_at "
                "FROM pending ORDER BY created_at, clarification_id"
            ).fetchall()
            valid: list[tuple[str, dict[str, Any]]] = []
            corrupt: list[tuple[sqlite3.Row, str]] = []
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"])
                    if not isinstance(payload, dict):
                        raise ValueError("pending payload must be a JSON object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    corrupt.append((row, f"{type(exc).__name__}: {exc}"))
                else:
                    valid.append((row["clarification_id"], payload))
            if corrupt:
                self._quarantine_rows_locked(corrupt)
        return tuple(valid)

    def _quarantine_rows_locked(
        self, corrupt: Sequence[tuple[sqlite3.Row, str]]
    ) -> None:
        """Move corrupt pending rows to the quarantine table in one transaction."""
        quarantined_at = time.time()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for row, error in corrupt:
                self._connection.execute(
                    """
                    INSERT INTO pending_quarantine(
                        clarification_id, task_id, payload_json, created_at,
                        error, quarantined_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(clarification_id) DO UPDATE SET
                        task_id=excluded.task_id,
                        payload_json=excluded.payload_json,
                        created_at=excluded.created_at,
                        error=excluded.error,
                        quarantined_at=excluded.quarantined_at
                    """,
                    (
                        row["clarification_id"],
                        row["task_id"],
                        row["payload_json"],
                        row["created_at"],
                        error,
                        quarantined_at,
                    ),
                )
                # Delete only the exact row we read.  A concurrent replacement is left in
                # pending and will be evaluated on the next load.
                self._connection.execute(
                    "DELETE FROM pending WHERE clarification_id=? AND payload_json=?",
                    (row["clarification_id"], row["payload_json"]),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def load_quarantined_pending(self) -> tuple[dict[str, Any], ...]:
        """Return quarantined corrupt rows for diagnostics; this is not a queue to replay."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT clarification_id, task_id, payload_json, created_at, error, "
                "quarantined_at FROM pending_quarantine "
                "ORDER BY quarantined_at, clarification_id"
            ).fetchall()
        return tuple(dict(row) for row in rows)

    # -- task state ------------------------------------------------------- #

    def save_task_state(
        self,
        task_id: str,
        *,
        preferences: Sequence[Mapping[str, Any]],
        rounds: int,
        second_enabled: bool,
        clarified: bool,
        r1_anchor_terms: Sequence[str] = (),
    ) -> None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                # Same lifecycle guard as save_pending(): a stale task checkpoint must not
                # survive a session reset.
                if self._task_is_retired_locked(task_id):
                    self._connection.execute("COMMIT")
                    return
                self._connection.execute(
                    """
                    INSERT INTO task_state(
                        task_id, preferences_json, rounds, second_enabled, clarified,
                        updated_at, r1_anchor_json
                    ) VALUES (?, ?, ?, ?, ?, strftime('%s','now'), ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        preferences_json=excluded.preferences_json,
                        rounds=excluded.rounds,
                        second_enabled=excluded.second_enabled,
                        clarified=excluded.clarified,
                        r1_anchor_json=excluded.r1_anchor_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        task_id,
                        _dumps(list(preferences)),
                        int(rounds),
                        int(bool(second_enabled)),
                        int(bool(clarified)),
                        _dumps(list(r1_anchor_terms)),
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def load_task_states(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM task_state ORDER BY task_id"
            ).fetchall()
        return tuple(
            (
                row["task_id"],
                {
                    "preferences": json.loads(row["preferences_json"]),
                    "rounds": int(row["rounds"]),
                    "second_enabled": bool(row["second_enabled"]),
                    "clarified": bool(row["clarified"]),
                    "r1_anchor_terms": list(json.loads(row["r1_anchor_json"] or "[]")),
                },
            )
            for row in rows
        )
