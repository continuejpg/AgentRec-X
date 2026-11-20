"""Durable state for the Web Demo: session identity, tasks and clarifications.

Scope (M1-C S2a + S2b)
-----------------------
* **Session identity** -- one row per session, monotonic turn/reply counters and an
  append-only ledger of every identity ever issued.
* **Task** -- one stable ``task_id`` per shopping goal, at most one active task per
  session, an explicit status machine, an optimistic ``task_version`` and the
  requirements snapshot supplied by the caller.
* **Pending clarification** -- one row per ASK, so a task can accumulate multiple ask
  rounds; at most one ``OPEN`` row per task, and a reply identity that is unique for the
  lifetime of the database.

This module deliberately does **not** store chat turns, product snapshots, MCP handles,
request idempotency records or cold-resume execution state, and it is not a
serialisation format for ``Session.view()``.  It also does not extract requirements from
natural language: it persists exactly the mapping the caller provides.

Mechanism
---------
``sqlite3`` from the standard library, WAL, ``foreign_keys=ON``, ``busy_timeout`` and
``BEGIN IMMEDIATE`` around every mutation.  Cross-process uniqueness is provided by
SQLite (transactions, primary keys, partial unique indexes), never by an in-process
lock; the ``RLock`` only stops two threads from interleaving statements on the *same*
connection.

Schema versions
---------------
v1 (S2a) had ``meta``/``web_session``/``web_identity``.  v2 (S2b) adds ``web_task`` and
``web_pending``.  A real v1 database is migrated in one transaction; a failed migration
rolls back and leaves v1 untouched.  Unknown versions are refused, and an existing
database is never rebuilt or overwritten.

Timestamps are Unix epoch seconds (``REAL``), matching the other SQLite stores here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

__all__ = [
    "DEFAULT_SESSION_TITLE",
    "PendingRecord",
    "SQLiteWebStateStore",
    "SessionRecord",
    "TaskRecord",
    "TURN_PAYLOAD_SCHEMA",
    "TurnRecord",
    "WEB_STATE_SCHEMA_VERSION",
    "ActiveTaskExists",
    "AskCommitError",
    "IdentityAllocationError",
    "PendingExpired",
    "PendingStateConflict",
    "SchemaVersionError",
    "SessionAlreadyExists",
    "SessionRetired",
    "StaleTaskVersion",
    "StateDataCorruption",
    "StateMigrationError",
    "StateStoreClosedError",
    "StateStoreError",
    "TaskStateConflict",
    "UnknownSession",
    "UnknownTask",
    "UnknownTurn",
]

#: Current schema version. v1 = S2a (identity), v2 = S2b (tasks + clarifications),
#: v3 = S3b (persisted chat turns).
WEB_STATE_SCHEMA_VERSION = 3

#: Mirrors ``web_demo.sessions.DEFAULT_TITLE``; a test asserts the two stay equal.
DEFAULT_SESSION_TITLE = "New shopping session"

#: SQLite waits for a competing writer instead of failing with "database is locked".
DEFAULT_BUSY_TIMEOUT_MS = 5_000

#: kind -> (counter column, identifier prefix).  Keeps ids compatible with
#: ``Session.next_turn_id()`` / ``Session.next_reply_turn_id()``.
_ID_KINDS: dict[str, tuple[str, str]] = {
    "turn": ("turn_counter", "t"),
    "reply": ("reply_counter", "r"),
}

TASK_STATUSES = ("RUNNING", "WAITING_FOR_USER", "COMPLETED", "CANCELLED", "FAILED")
ACTIVE_TASK_STATUSES = ("RUNNING", "WAITING_FOR_USER")
TERMINAL_TASK_STATUSES = ("COMPLETED", "CANCELLED", "FAILED")

#: Only transitions in this map are allowed; terminal states have none, which is what
#: makes "a completed/cancelled task can never be reopened" a Store-level guarantee.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "RUNNING": frozenset({"WAITING_FOR_USER", "COMPLETED", "CANCELLED", "FAILED"}),
    "WAITING_FOR_USER": frozenset({"RUNNING", "CANCELLED", "FAILED"}),
    "COMPLETED": frozenset(),
    "CANCELLED": frozenset(),
    "FAILED": frozenset(),
}

PENDING_STATUSES = ("OPEN", "CONSUMED", "EXPIRED", "CANCELLED")
PENDING_MODES = frozenset({"cpu_live", "qwen_live"})

#: Versioned envelope written to ``web_turn.payload_json``.
TURN_PAYLOAD_SCHEMA = "agentrecx.web.turn.v1"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_SCHEMA_V1_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS web_session (
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS web_identity (
        session_id TEXT NOT NULL REFERENCES web_session(session_id),
        kind       TEXT NOT NULL CHECK (kind IN ('turn', 'reply')),
        seq        INTEGER NOT NULL CHECK (seq >= 1),
        value      TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (session_id, kind, value),
        UNIQUE (session_id, kind, seq)
    )
    """,
)

_SCHEMA_V2_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS web_task (
        task_id                   TEXT PRIMARY KEY,
        session_id                TEXT NOT NULL REFERENCES web_session(session_id),
        status                    TEXT NOT NULL CHECK (status IN
                                    ('RUNNING','WAITING_FOR_USER','COMPLETED','CANCELLED','FAILED')),
        task_version              INTEGER NOT NULL CHECK (task_version >= 1),
        reset_generation          INTEGER NOT NULL CHECK (reset_generation >= 0),
        current_requirements_json TEXT NOT NULL,
        requirements_version      INTEGER NOT NULL DEFAULT 0 CHECK (requirements_version >= 0),
        created_at                REAL NOT NULL,
        updated_at                REAL NOT NULL,
        closed_at                 REAL,
        close_reason              TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS web_pending (
        pending_id       TEXT PRIMARY KEY,
        task_id          TEXT NOT NULL REFERENCES web_task(task_id),
        clarification_id TEXT NOT NULL,
        query            TEXT NOT NULL,
        turn_id          TEXT NOT NULL,
        reply_turn_id    TEXT NOT NULL UNIQUE,
        mode             TEXT NOT NULL CHECK (mode IN ('cpu_live','qwen_live')),
        payload_json     TEXT NOT NULL,
        status           TEXT NOT NULL CHECK (status IN
                           ('OPEN','CONSUMED','EXPIRED','CANCELLED')),
        created_at       REAL NOT NULL,
        expires_at       REAL NOT NULL,
        consumed_at      REAL,
        CHECK (expires_at > created_at),
        CHECK ((status = 'CONSUMED') = (consumed_at IS NOT NULL))
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_web_task_one_active
        ON web_task(session_id) WHERE status IN ('RUNNING','WAITING_FOR_USER')
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_web_pending_one_open
        ON web_pending(task_id) WHERE status = 'OPEN'
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_web_pending_task
        ON web_pending(task_id, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_web_pending_expiry
        ON web_pending(expires_at) WHERE status = 'OPEN'
    """,
)

_SCHEMA_V3_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS web_turn (
        session_id   TEXT NOT NULL REFERENCES web_session(session_id),
        turn_seq     INTEGER NOT NULL CHECK (turn_seq >= 1),
        turn_id      TEXT NOT NULL,
        task_id      TEXT REFERENCES web_task(task_id),
        payload_json TEXT NOT NULL,
        created_at   REAL NOT NULL,
        PRIMARY KEY (session_id, turn_seq),
        UNIQUE (session_id, turn_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_web_turn_task ON web_turn(task_id)
    """,
)

_SESSION_COLUMNS = (
    "session_id, profile_id, title, created_at, updated_at, behavior_enabled, "
    "turn_counter, reply_counter, reset_generation, retired"
)
_TASK_COLUMNS = (
    "task_id, session_id, status, task_version, reset_generation, "
    "current_requirements_json, requirements_version, created_at, updated_at, "
    "closed_at, close_reason"
)
_PENDING_COLUMNS = (
    "pending_id, task_id, clarification_id, query, turn_id, reply_turn_id, mode, "
    "payload_json, status, created_at, expires_at, consumed_at"
)
_TURN_COLUMNS = "session_id, turn_seq, turn_id, task_id, payload_json, created_at"
_PENDING_ALIASED_COLUMNS = (
    "p.pending_id, p.task_id, p.clarification_id, p.query, p.turn_id, p.reply_turn_id, "
    "p.mode, p.payload_json, p.status, p.created_at, p.expires_at, p.consumed_at"
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class StateStoreError(RuntimeError):
    """Base class for every web-state-store failure."""


class SchemaVersionError(StateStoreError):
    """The database was written by an incompatible schema version."""


class StateMigrationError(StateStoreError):
    """A schema migration failed and was rolled back."""


class StateDataCorruption(StateStoreError):
    """A critical JSON column cannot be parsed; the row must not be used."""


class UnknownSession(StateStoreError):
    """No session row exists for the requested ``session_id``."""


class SessionAlreadyExists(StateStoreError):
    """A session with this ``session_id`` has already been created."""


class SessionRetired(StateStoreError):
    """The session is retired and can no longer mutate or allocate identities."""


class UnknownTask(StateStoreError):
    """No task row exists for the requested ``task_id`` (or it belongs elsewhere)."""


class ActiveTaskExists(StateStoreError):
    """The session already has an active task."""


class TaskStateConflict(StateStoreError):
    """The requested transition is not allowed from the task's current state."""


class StaleTaskVersion(StateStoreError):
    """The caller's ``expected_task_version`` is not the stored version."""


class UnknownTurn(StateStoreError):
    """The source turn identity is not recorded for this session."""


class PendingStateConflict(StateStoreError):
    """The pending row is not OPEN, does not belong here, or one is already open."""


class PendingExpired(StateStoreError):
    """The pending row exists but its absolute TTL has passed."""


class IdentityAllocationError(StateStoreError):
    """An identity could not be allocated; the transaction was rolled back."""


class AskCommitError(StateStoreError):
    """The ASK transaction could not persist its pending row; it was rolled back."""


class StateStoreClosedError(StateStoreError):
    """The store has been closed and can no longer be used."""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SessionRecord:
    """Persisted identity state for one Web Demo session.

    ``turn_counter`` / ``reply_counter`` are the highest sequence number already issued,
    so the next turn is ``turn_counter + 1``.  ``reset_generation`` counts identity resets
    and never rewinds the counters.
    """

    session_id: str
    profile_id: str
    title: str
    created_at: float
    updated_at: float
    behavior_enabled: bool
    turn_counter: int
    reply_counter: int
    reset_generation: int
    retired: bool


@dataclass(frozen=True)
class TaskRecord:
    """One shopping goal with a stable identity.

    ``status`` is one of :data:`TASK_STATUSES`; ``task_version`` is the optimistic
    concurrency token; ``reset_generation`` records the session generation the task was
    created in, so a reset can fence it.
    """

    task_id: str
    session_id: str
    status: str
    task_version: int
    reset_generation: int
    current_requirements: dict[str, Any]
    requirements_version: int
    created_at: float
    updated_at: float
    closed_at: float | None
    close_reason: str | None


@dataclass(frozen=True)
class PendingRecord:
    """One ASK round.  ``reply_turn_id`` is the M9 idempotency identity for the reply."""

    pending_id: str
    task_id: str
    clarification_id: str
    query: str
    turn_id: str
    reply_turn_id: str
    mode: str
    payload: dict[str, Any]
    status: str
    created_at: float
    expires_at: float
    consumed_at: float | None


@dataclass(frozen=True)
class TurnRecord:
    """One persisted chat turn.

    ``payload`` is the validated envelope produced by :func:`pack_turn_payload`
    (``{"schema": ..., "turn": <TurnView>, "ranking_source": ...}``); it is never a
    substitute for the session view and never contains identity counters.
    """

    session_id: str
    turn_seq: int
    turn_id: str
    task_id: str | None
    payload: dict[str, Any]
    created_at: float


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class SQLiteWebStateStore:
    """SQLite-backed, process-shared session/task/clarification state store."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._connection = sqlite3.connect(
                self._path, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise StateStoreError(f"cannot open web state database {self._path!r}: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        try:
            try:
                # Only switch journal modes when needed: the pragma takes a brief
                # exclusive lock, which would make concurrent opens contend needlessly.
                current_mode = self._connection.execute("PRAGMA journal_mode").fetchone()[0]
                if str(current_mode or "").lower() != "wal":
                    self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA foreign_keys = ON")
                self._connection.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS}")
                self._initialise_schema()
            except sqlite3.DatabaseError as exc:
                # Also covers a file that is not SQLite at all: PRAGMA fails before the
                # schema probe.  Never fall back to an in-memory store.
                raise StateStoreError(
                    f"{self._path!r} is not a usable SQLite database: {exc}"
                ) from exc
        except BaseException:
            _close_quietly(self._connection)
            raise

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def path(self) -> str:
        return self._path

    @property
    def schema_version(self) -> int:
        with self._lock:
            self._ensure_open()
            return self._read_schema_version(self._connection)

    def close(self) -> None:
        """Close the connection.  Idempotent; the store is unusable afterwards."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._connection.close()
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                raise StateStoreError(f"cannot close web state database: {exc}") from exc

    def __enter__(self) -> "SQLiteWebStateStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema ------------------------------------------------------------- #

    def _initialise_schema(self) -> None:
        with self._lock:
            tables = self._table_names(self._connection)
            if not tables:
                self._create_latest_schema_locked()
                tables = self._table_names(self._connection)
            if "meta" not in tables or "web_session" not in tables:
                # Never adopt or overwrite a database that belongs to something else.
                raise StateStoreError(
                    f"{self._path!r} is a SQLite database without the web state schema; "
                    "refusing to modify it"
                )

            version = self._read_schema_version(self._connection)
            if version < WEB_STATE_SCHEMA_VERSION:
                self._migrate_to_latest_locked()
                version = self._read_schema_version(self._connection)
            if version != WEB_STATE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"web state database {self._path!r} has schema version {version}, "
                    f"expected {WEB_STATE_SCHEMA_VERSION}"
                )
            self._require_tables_locked(
                ("web_session", "web_identity", "web_task", "web_pending", "web_turn")
            )

    def _create_latest_schema_locked(self) -> None:
        with self._transaction() as connection:
            if self._table_names(connection):
                # Another process created it between the probe and this transaction.
                return
            for statement in _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS + _SCHEMA_V3_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(WEB_STATE_SCHEMA_VERSION),),
            )

    def _migrate_to_latest_locked(self) -> None:
        """Upgrade a real v1/v2 database in one transaction; never partially applies."""
        with self._transaction() as connection:
            version = self._read_schema_version(connection)
            if version == WEB_STATE_SCHEMA_VERSION:
                return  # another process migrated it while we waited for the write lock
            if version not in (1, 2):
                raise SchemaVersionError(
                    f"cannot migrate schema version {version} to {WEB_STATE_SCHEMA_VERSION}"
                )
            try:
                if version == 1:
                    for statement in _SCHEMA_V2_STATEMENTS:
                        connection.execute(statement)
                for statement in _SCHEMA_V3_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    "UPDATE meta SET value = ? WHERE key = 'schema_version'",
                    (str(WEB_STATE_SCHEMA_VERSION),),
                )
            except sqlite3.Error as exc:
                raise StateMigrationError(
                    f"migration to v{WEB_STATE_SCHEMA_VERSION} of {self._path!r} failed "
                    f"and was rolled back: {exc}"
                ) from exc

    def _read_schema_version(self, connection: sqlite3.Connection) -> int:
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot read schema version: {exc}") from exc
        if row is None:
            raise SchemaVersionError(
                f"{self._path!r} has no schema_version row; refusing to continue"
            )
        try:
            return int(row["value"])
        except (TypeError, ValueError) as exc:
            raise SchemaVersionError(
                f"{self._path!r} has a non-integer schema_version {row['value']!r}"
            ) from exc

    @staticmethod
    def _table_names(connection: sqlite3.Connection) -> set[str]:
        try:
            return {
                str(row["name"])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        except sqlite3.DatabaseError as exc:
            raise StateStoreError(f"cannot inspect database schema: {exc}") from exc

    def _require_tables_locked(self, required: tuple[str, ...]) -> None:
        tables = self._table_names(self._connection)
        missing = [name for name in required if name not in tables]
        if missing:
            raise StateStoreError(
                f"web state database {self._path!r} is missing tables {missing}; "
                f"schema version {WEB_STATE_SCHEMA_VERSION} is incomplete"
            )

    # -- session records ---------------------------------------------------- #

    def create_session(
        self,
        *,
        session_id: str,
        profile_id: str,
        behavior_enabled: bool = True,
        title: str | None = None,
    ) -> SessionRecord:
        """Create one session row with zero counters.

        ``title`` defaults to :data:`DEFAULT_SESSION_TITLE`, matching the in-memory
        ``Session`` default.  A duplicate ``session_id`` is an explicit error, not an
        upsert: silently adopting an existing session would hand out its identities.
        """
        session_id = _require_text(session_id, "session_id")
        profile_id = _require_text(profile_id, "profile_id")
        resolved_title = DEFAULT_SESSION_TITLE if title is None else _require_text(title, "title")
        timestamp = time.time()
        with self._transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO web_session(session_id, profile_id, title, created_at, "
                    "updated_at, behavior_enabled, turn_counter, reply_counter, "
                    "reset_generation, retired) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0)",
                    (session_id, profile_id, resolved_title, timestamp, timestamp,
                     1 if behavior_enabled else 0),
                )
            except sqlite3.IntegrityError as exc:
                raise SessionAlreadyExists(f"session {session_id!r} already exists") from exc
            row = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _to_session_record(row)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Return the persisted session, or ``None`` when it does not exist.

        Database failures raise :class:`StateStoreError`; they are never reported as
        "unknown session".
        """
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read session {session_id!r}: {exc}") from exc
        return None if row is None else _to_session_record(row)

    def update_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        behavior_enabled: bool | None = None,
    ) -> SessionRecord:
        """Persist the supported mutable session metadata in one transaction.

        Only ``title`` and ``behavior_enabled`` are mutable here; ``profile_id`` and
        ``created_at`` are immutable, and counters/identity are owned by
        :meth:`allocate_turn_id` / :meth:`allocate_reply_id`.  At least one field is
        required so a no-op call cannot silently "succeed".
        """
        if title is None and behavior_enabled is None:
            raise ValueError("at least one of title/behavior_enabled is required")
        resolved_title = None if title is None else _require_text(title, "title")
        timestamp = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT retired FROM web_session WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(row["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")

            assignments: list[str] = []
            parameters: list[Any] = []
            if resolved_title is not None:
                assignments.append("title = ?")
                parameters.append(resolved_title)
            if behavior_enabled is not None:
                assignments.append("behavior_enabled = ?")
                parameters.append(1 if behavior_enabled else 0)
            assignments.append("updated_at = ?")
            parameters.append(timestamp)
            parameters.append(session_id)
            connection.execute(
                f"UPDATE web_session SET {', '.join(assignments)} WHERE session_id = ?",
                tuple(parameters),
            )
            row = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _to_session_record(row)

    def list_sessions(self, limit: int = 20) -> tuple[SessionRecord, ...]:
        """Return the most recently updated sessions, newest first."""
        bounded = max(1, int(limit))
        with self._lock:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    f"SELECT {_SESSION_COLUMNS} FROM web_session "
                    "ORDER BY updated_at DESC, session_id ASC LIMIT ?",
                    (bounded,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot list sessions: {exc}") from exc
        return tuple(_to_session_record(row) for row in rows)

    def retire_session(self, session_id: str) -> SessionRecord:
        """Mark a session permanently unusable.  Terminal and idempotent.

        Retirement is *not* the Web Demo reset: reset keeps the session alive, cancels
        its active task and only advances ``reset_generation``.  A retired session can no
        longer create tasks, allocate identities or be reset.
        """
        with self._transaction() as connection:
            row = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise UnknownSession(f"no session {session_id!r}")
            if not int(row["retired"]):
                connection.execute(
                    "UPDATE web_session SET retired = 1, updated_at = ? WHERE session_id = ?",
                    (time.time(), session_id),
                )
                row = connection.execute(
                    f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        return _to_session_record(row)

    # -- identity allocation ------------------------------------------------ #

    def allocate_turn_id(self, session_id: str) -> str:
        """Allocate the next ``{session_id}-t{n}`` identity durably."""
        return self._allocate(session_id, "turn")

    def allocate_reply_id(self, session_id: str) -> str:
        """Allocate the next ``{session_id}-r{n}`` identity durably."""
        return self._allocate(session_id, "reply")

    def _allocate(self, session_id: str, kind: Literal["turn", "reply"]) -> str:
        with self._transaction() as connection:
            return self._allocate_in_transaction(connection, session_id, kind)

    def _allocate_in_transaction(
        self, connection: sqlite3.Connection, session_id: str, kind: Literal["turn", "reply"]
    ) -> str:
        """Allocate inside an existing transaction; the caller owns commit/rollback."""
        column, prefix = _ID_KINDS[kind]
        timestamp = time.time()
        row = connection.execute(
            "SELECT retired FROM web_session WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise UnknownSession(f"no session {session_id!r}")
        if int(row["retired"]):
            raise SessionRetired(f"session {session_id!r} is retired")
        connection.execute(
            f"UPDATE web_session SET {column} = {column} + 1, updated_at = ? "
            "WHERE session_id = ?",
            (timestamp, session_id),
        )
        seq = int(
            connection.execute(
                f"SELECT {column} FROM web_session WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        )
        value = f"{session_id}-{prefix}{seq}"
        try:
            connection.execute(
                "INSERT INTO web_identity(session_id, kind, seq, value, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, kind, seq, value, timestamp),
            )
        except sqlite3.IntegrityError as exc:
            # The counter is rolled back with this transaction, so no gap is left behind
            # and the same identity can be retried once the conflict is gone.
            raise IdentityAllocationError(
                f"identity {value!r} was already allocated"
            ) from exc
        return value

    # -- tasks -------------------------------------------------------------- #

    def create_task(
        self, session_id: str, *, initial_requirements: Mapping[str, Any] | None = None
    ) -> TaskRecord:
        """Create a new task with a fresh UUID4 identity.

        At most one task per session may be active (the partial unique index enforces it
        even across processes).  The task records the session's current
        ``reset_generation`` so a later reset can fence it.
        """
        session_id = _require_text(session_id, "session_id")
        requirements_json = _dumps_json(initial_requirements or {}, "initial_requirements")
        task_id = str(uuid.uuid4())
        timestamp = time.time()
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(session["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")
            try:
                connection.execute(
                    "INSERT INTO web_task(task_id, session_id, status, task_version, "
                    "reset_generation, current_requirements_json, requirements_version, "
                    "created_at, updated_at) VALUES (?, ?, 'RUNNING', 1, ?, ?, 0, ?, ?)",
                    (task_id, session_id, int(session["reset_generation"]),
                     requirements_json, timestamp, timestamp),
                )
            except sqlite3.IntegrityError as exc:
                raise ActiveTaskExists(
                    f"session {session_id!r} already has an active task"
                ) from exc
            row = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
            ).fetchone()
        return _to_task_record(row)

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read task {task_id!r}: {exc}") from exc
        return None if row is None else _to_task_record(row)

    def get_active_task(self, session_id: str) -> TaskRecord | None:
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_TASK_COLUMNS} FROM web_task WHERE session_id = ? "
                    "AND status IN ('RUNNING','WAITING_FOR_USER')",
                    (session_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read active task: {exc}") from exc
        return None if row is None else _to_task_record(row)

    def list_tasks(self, session_id: str) -> tuple[TaskRecord, ...]:
        with self._lock:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    f"SELECT {_TASK_COLUMNS} FROM web_task WHERE session_id = ? "
                    "ORDER BY created_at, task_id",
                    (session_id,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot list tasks: {exc}") from exc
        return tuple(_to_task_record(row) for row in rows)

    # -- pending / ASK ------------------------------------------------------ #

    def commit_ask(
        self,
        session_id: str,
        task_id: str,
        *,
        expected_task_version: int,
        clarification_id: str,
        source_turn_id: str,
        query: str,
        payload: Mapping[str, Any],
        expires_at: float,
        mode: str = "cpu_live",
        turn: Mapping[str, Any] | None = None,
        ranking_source: str | None = None,
    ) -> PendingRecord:
        """Persist one ASK round atomically.

        In a single ``BEGIN IMMEDIATE`` transaction: validate the session (not retired),
        the session/task generation pair, ownership, task status, expected version, the
        absence of another OPEN pending, and the source turn identity; then allocate the
        reply identity, insert the pending row, move the task to ``WAITING_FOR_USER`` and
        bump ``task_version``.  Any failure rolls the whole unit back, so there is no
        state where a reply identity was issued without its pending row.
        """
        session_id = _require_text(session_id, "session_id")
        task_id = _require_text(task_id, "task_id")
        clarification_id = _require_text(clarification_id, "clarification_id")
        source_turn_id = _require_text(source_turn_id, "source_turn_id")
        query = _require_text(query, "query")
        if mode not in PENDING_MODES:
            raise ValueError(f"mode must be one of {sorted(PENDING_MODES)}, got {mode!r}")
        payload_json = _dumps_json(payload, "payload")
        timestamp = time.time()
        expires_at = float(expires_at)
        if expires_at <= timestamp:
            raise ValueError("expires_at must be in the future")
        pending_id = f"pend-{uuid.uuid4().hex}"

        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(session["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")

            task = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None or str(task["session_id"]) != session_id:
                raise UnknownTask(f"no task {task_id!r} for session {session_id!r}")
            if int(task["reset_generation"]) != int(session["reset_generation"]):
                raise TaskStateConflict(
                    "task belongs to a previous reset generation"
                )
            if str(task["status"]) != "RUNNING":
                raise TaskStateConflict(
                    f"task {task_id!r} is {task['status']!r}; only RUNNING may ask"
                )
            if int(task["task_version"]) != int(expected_task_version):
                raise StaleTaskVersion(
                    f"task {task_id!r} is at version {task['task_version']}, "
                    f"expected {expected_task_version}"
                )

            open_pending = connection.execute(
                "SELECT pending_id FROM web_pending WHERE task_id = ? AND status = 'OPEN'",
                (task_id,),
            ).fetchone()
            if open_pending is not None:
                raise PendingStateConflict(
                    f"task {task_id!r} already has an open pending clarification"
                )

            source_turn = connection.execute(
                "SELECT 1 FROM web_identity WHERE session_id = ? AND kind = 'turn' "
                "AND value = ?",
                (session_id, source_turn_id),
            ).fetchone()
            if source_turn is None:
                raise UnknownTurn(
                    f"turn {source_turn_id!r} is not recorded for session {session_id!r}"
                )

            reply_turn_id = self._allocate_in_transaction(connection, session_id, "reply")

            try:
                connection.execute(
                    "INSERT INTO web_pending(pending_id, task_id, clarification_id, query, "
                    "turn_id, reply_turn_id, mode, payload_json, status, created_at, "
                    "expires_at, consumed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, NULL)",
                    (pending_id, task_id, clarification_id, query, source_turn_id,
                     reply_turn_id, mode, payload_json, timestamp, expires_at),
                )
            except sqlite3.IntegrityError as exc:
                raise AskCommitError(
                    f"could not persist pending clarification for task {task_id!r}"
                ) from exc

            updated = connection.execute(
                "UPDATE web_task SET status = 'WAITING_FOR_USER', "
                "task_version = task_version + 1, updated_at = ? "
                "WHERE task_id = ? AND status = 'RUNNING' AND task_version = ?",
                (timestamp, task_id, expected_task_version),
            )
            if updated.rowcount != 1:  # pragma: no cover - guarded by the checks above
                raise StaleTaskVersion(f"task {task_id!r} changed during ASK commit")

            if turn is not None:
                self._insert_turn_in_transaction(
                    connection, session_id, task_id, turn, ranking_source, timestamp
                )

            row = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                (pending_id,),
            ).fetchone()
        return _to_pending_record(row)

    def get_pending(self, pending_id: str) -> PendingRecord | None:
        """Return the stored pending row, whatever its status (including overdue OPEN)."""
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                    (pending_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read pending {pending_id!r}: {exc}") from exc
        return None if row is None else _to_pending_record(row)

    def get_open_pending(self, task_id: str, *, now: float | None = None) -> PendingRecord | None:
        """Return the OPEN pending only when its absolute TTL has not elapsed.

        An overdue row stays ``OPEN`` in storage until it is explicitly expired (see
        :meth:`expire_overdue_pending`); this method never resurrects it as valid.
        """
        reference = time.time() if now is None else float(now)
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_PENDING_COLUMNS} FROM web_pending "
                    "WHERE task_id = ? AND status = 'OPEN' AND expires_at > ?",
                    (task_id, reference),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read open pending: {exc}") from exc
        return None if row is None else _to_pending_record(row)

    def list_pending(self, task_id: str) -> tuple[PendingRecord, ...]:
        with self._lock:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE task_id = ? "
                    "ORDER BY created_at, pending_id",
                    (task_id,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot list pending rows: {exc}") from exc
        return tuple(_to_pending_record(row) for row in rows)

    def get_open_pending_for_session(
        self, session_id: str, *, now: float | None = None
    ) -> PendingRecord | None:
        """Return the session's current unexpired OPEN pending, if any."""
        reference = time.time() if now is None else float(now)
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_PENDING_ALIASED_COLUMNS} FROM web_pending p "
                    "JOIN web_task t ON p.task_id = t.task_id "
                    "WHERE t.session_id = ? AND p.status = 'OPEN' AND p.expires_at > ? "
                    "ORDER BY p.created_at DESC LIMIT 1",
                    (session_id, reference),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read session pending: {exc}") from exc
        return None if row is None else _to_pending_record(row)

    # -- turns -------------------------------------------------------------- #

    def append_turn(
        self,
        session_id: str,
        *,
        turn: Mapping[str, Any],
        task_id: str | None = None,
        ranking_source: str | None = None,
        reset_generation: int | None = None,
    ) -> TurnRecord:
        """Persist one finished chat turn.  Used for non-ASK turns.

        The turn id must have been allocated through :meth:`allocate_turn_id`; its
        sequence number is derived from that id so the ledger and the stored history can
        never disagree.  When ``reset_generation`` is supplied, a turn produced in a
        previous reset generation is refused, so a late result can never resurrect
        history the user has already cleared.
        """
        timestamp = time.time()
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if (
                reset_generation is not None
                and int(reset_generation) != int(session["reset_generation"])
            ):
                raise TaskStateConflict(
                    "turn belongs to a previous reset generation and must not be persisted"
                )
            turn_id = self._insert_turn_in_transaction(
                connection, session_id, task_id, turn, ranking_source, timestamp
            )
            row = connection.execute(
                f"SELECT {_TURN_COLUMNS} FROM web_turn WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            ).fetchone()
        return _to_turn_record(row)

    def list_turns(self, session_id: str) -> tuple[TurnRecord, ...]:
        with self._lock:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    f"SELECT {_TURN_COLUMNS} FROM web_turn WHERE session_id = ? "
                    "ORDER BY turn_seq",
                    (session_id,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot list turns: {exc}") from exc
        return tuple(_to_turn_record(row) for row in rows)

    def _insert_turn_in_transaction(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        task_id: str | None,
        turn: Mapping[str, Any],
        ranking_source: str | None,
        timestamp: float,
    ) -> str:
        envelope = pack_turn_payload(turn, ranking_source=ranking_source)
        turn_id = str(envelope["turn"]["turn_id"])
        seq = _turn_seq_from_id(turn_id, session_id)
        try:
            connection.execute(
                "INSERT INTO web_turn(session_id, turn_seq, turn_id, task_id, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, seq, turn_id, task_id,
                 json.dumps(envelope, sort_keys=True, ensure_ascii=False), timestamp),
            )
        except sqlite3.IntegrityError as exc:
            raise StateStoreError(
                f"turn {turn_id!r} is already persisted for session {session_id!r}"
            ) from exc
        return turn_id

    def consume_pending(
        self,
        session_id: str,
        task_id: str,
        pending_id: str,
        *,
        expected_task_version: int,
        requirements: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> tuple[TaskRecord, PendingRecord]:
        """Consume one OPEN pending: WAITING_FOR_USER -> RUNNING.

        This is a Store-level transition only.  It performs no M9 write and has no
        request-claim protection, so it must not be wired to the HTTP reply path before
        S5 adds request idempotency.
        """
        reference = time.time() if now is None else float(now)
        requirements_json = (
            None if requirements is None else _dumps_json(requirements, "requirements")
        )
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(session["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")

            task = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None or str(task["session_id"]) != session_id:
                raise UnknownTask(f"no task {task_id!r} for session {session_id!r}")
            if int(task["reset_generation"]) != int(session["reset_generation"]):
                raise TaskStateConflict("task belongs to a previous reset generation")
            if str(task["status"]) != "WAITING_FOR_USER":
                raise TaskStateConflict(
                    f"task {task_id!r} is {task['status']!r}; no reply is expected"
                )
            if int(task["task_version"]) != int(expected_task_version):
                raise StaleTaskVersion(
                    f"task {task_id!r} is at version {task['task_version']}, "
                    f"expected {expected_task_version}"
                )

            pending = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                (pending_id,),
            ).fetchone()
            if pending is None or str(pending["task_id"]) != task_id:
                raise UnknownTask(f"no pending {pending_id!r} for task {task_id!r}")
            if str(pending["status"]) != "OPEN":
                raise PendingStateConflict(
                    f"pending {pending_id!r} is {pending['status']!r}, not OPEN"
                )
            if float(pending["expires_at"]) <= reference:
                raise PendingExpired(
                    f"pending {pending_id!r} expired at {pending['expires_at']}"
                )

            connection.execute(
                "UPDATE web_pending SET status = 'CONSUMED', consumed_at = ? "
                "WHERE pending_id = ? AND status = 'OPEN'",
                (reference, pending_id),
            )
            if requirements_json is None:
                updated = connection.execute(
                    "UPDATE web_task SET status = 'RUNNING', "
                    "task_version = task_version + 1, updated_at = ? "
                    "WHERE task_id = ? AND status = 'WAITING_FOR_USER' AND task_version = ?",
                    (reference, task_id, expected_task_version),
                )
            else:
                updated = connection.execute(
                    "UPDATE web_task SET status = 'RUNNING', "
                    "task_version = task_version + 1, current_requirements_json = ?, "
                    "requirements_version = requirements_version + 1, updated_at = ? "
                    "WHERE task_id = ? AND status = 'WAITING_FOR_USER' AND task_version = ?",
                    (requirements_json, reference, task_id, expected_task_version),
                )
            if updated.rowcount != 1:  # pragma: no cover - guarded above
                raise StaleTaskVersion(f"task {task_id!r} changed during reply commit")

            task_row = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
            ).fetchone()
            pending_row = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                (pending_id,),
            ).fetchone()
        return _to_task_record(task_row), _to_pending_record(pending_row)

    def expire_overdue_pending(self, *, now: float | None = None) -> tuple[PendingRecord, ...]:
        """Expire every overdue OPEN pending and cancel the task waiting on it.

        A ``WAITING_FOR_USER`` task must never keep looking valid without a usable
        pending, so the task moves to terminal ``CANCELLED`` with reason
        ``clarification_expired``.  The user then starts a new task; the cancelled one
        keeps its identity and history and can never be reopened.
        """
        reference = time.time() if now is None else float(now)
        expired: list[PendingRecord] = []
        with self._transaction() as connection:
            rows = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending "
                "WHERE status = 'OPEN' AND expires_at <= ? ORDER BY created_at, pending_id",
                (reference,),
            ).fetchall()
            for row in rows:
                pending_id = str(row["pending_id"])
                task_id = str(row["task_id"])
                connection.execute(
                    "UPDATE web_pending SET status = 'EXPIRED' "
                    "WHERE pending_id = ? AND status = 'OPEN'",
                    (pending_id,),
                )
                connection.execute(
                    "UPDATE web_task SET status = 'CANCELLED', "
                    "close_reason = 'clarification_expired', closed_at = ?, "
                    "task_version = task_version + 1, updated_at = ? "
                    "WHERE task_id = ? AND status = 'WAITING_FOR_USER'",
                    (reference, reference, task_id),
                )
                updated = connection.execute(
                    f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                    (pending_id,),
                ).fetchone()
                expired.append(_to_pending_record(updated))
        return tuple(expired)

    # -- task transitions --------------------------------------------------- #

    def complete_task(
        self,
        session_id: str,
        task_id: str,
        *,
        expected_task_version: int,
        requirements: Mapping[str, Any] | None = None,
    ) -> TaskRecord:
        """RUNNING -> COMPLETED (terminal).  Only a current version may finish a task."""
        return self._finish_task(
            session_id, task_id, target="COMPLETED", expected=expected_task_version,
            requirements=requirements, reason=None,
        )

    def cancel_task(
        self, session_id: str, task_id: str, *, expected_task_version: int, reason: str
    ) -> TaskRecord:
        """RUNNING/WAITING -> CANCELLED (terminal); its OPEN pendings are cancelled too."""
        return self._finish_task(
            session_id, task_id, target="CANCELLED", expected=expected_task_version,
            requirements=None, reason=_require_text(reason, "reason"),
        )

    def fail_task(
        self, session_id: str, task_id: str, *, expected_task_version: int, reason: str
    ) -> TaskRecord:
        """RUNNING/WAITING -> FAILED (terminal)."""
        return self._finish_task(
            session_id, task_id, target="FAILED", expected=expected_task_version,
            requirements=None, reason=_require_text(reason, "reason"),
        )

    def _finish_task(
        self,
        session_id: str,
        task_id: str,
        *,
        target: str,
        expected: int,
        requirements: Mapping[str, Any] | None,
        reason: str | None,
    ) -> TaskRecord:
        requirements_json = (
            None if requirements is None else _dumps_json(requirements, "requirements")
        )
        timestamp = time.time()
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(session["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")
            task = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None or str(task["session_id"]) != session_id:
                raise UnknownTask(f"no task {task_id!r} for session {session_id!r}")
            current = str(task["status"])
            _assert_transition(current, target)
            if int(task["task_version"]) != int(expected):
                raise StaleTaskVersion(
                    f"task {task_id!r} is at version {task['task_version']}, expected {expected}"
                )
            if int(task["reset_generation"]) != int(session["reset_generation"]):
                raise TaskStateConflict("task belongs to a previous reset generation")

            if requirements_json is None:
                updated = connection.execute(
                    "UPDATE web_task SET status = ?, task_version = task_version + 1, "
                    "updated_at = ?, closed_at = ?, close_reason = ? "
                    "WHERE task_id = ? AND task_version = ? AND status = ?",
                    (target, timestamp, timestamp, reason, task_id, expected, current),
                )
            else:
                updated = connection.execute(
                    "UPDATE web_task SET status = ?, task_version = task_version + 1, "
                    "current_requirements_json = ?, "
                    "requirements_version = requirements_version + 1, updated_at = ?, "
                    "closed_at = ?, close_reason = ? "
                    "WHERE task_id = ? AND task_version = ? AND status = ?",
                    (target, requirements_json, timestamp, timestamp, reason,
                     task_id, expected, current),
                )
            if updated.rowcount != 1:  # pragma: no cover - guarded above
                raise StaleTaskVersion(f"task {task_id!r} changed during finalisation")

            if target in {"CANCELLED", "FAILED"}:
                connection.execute(
                    "UPDATE web_pending SET status = 'CANCELLED' "
                    "WHERE task_id = ? AND status = 'OPEN'",
                    (task_id,),
                )

            row = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task_id,)
            ).fetchone()
        return _to_task_record(row)

    # -- reset -------------------------------------------------------------- #

    def reset_session(self, session_id: str, *, title: str | None = None) -> SessionRecord:
        """Atomically reset the visible conversation of one session.

        Keeps ``session_id``, both counters and the identity ledger; advances
        ``reset_generation``; cancels the active task and its OPEN pending.  Completed
        history is untouched and M9 is in a different store, so it is not modified.
        ``title`` is applied in the same transaction so a reset cannot half-succeed.
        """
        resolved_title = None if title is None else _require_text(title, "title")
        timestamp = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(row["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")

            connection.execute(
                "DELETE FROM web_turn WHERE session_id = ?", (session_id,)
            )
            connection.execute(
                "UPDATE web_pending SET status = 'CANCELLED' WHERE status = 'OPEN' "
                "AND task_id IN (SELECT task_id FROM web_task WHERE session_id = ? "
                "AND status IN ('RUNNING','WAITING_FOR_USER'))",
                (session_id,),
            )
            connection.execute(
                "UPDATE web_task SET status = 'CANCELLED', close_reason = 'user_reset', "
                "closed_at = ?, task_version = task_version + 1, updated_at = ? "
                "WHERE session_id = ? AND status IN ('RUNNING','WAITING_FOR_USER')",
                (timestamp, timestamp, session_id),
            )
            if resolved_title is None:
                connection.execute(
                    "UPDATE web_session SET reset_generation = reset_generation + 1, "
                    "updated_at = ? WHERE session_id = ?",
                    (timestamp, session_id),
                )
            else:
                connection.execute(
                    "UPDATE web_session SET reset_generation = reset_generation + 1, "
                    "title = ?, updated_at = ? WHERE session_id = ?",
                    (resolved_title, timestamp, session_id),
                )
            row = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _to_session_record(row)

    def advance_reset_generation(self, session_id: str) -> SessionRecord:
        """Compatibility alias for :meth:`reset_session` (S2a name).

        It now performs the full atomic reset, so there is no public path that advances
        the generation while leaving an active task behind.
        """
        return self.reset_session(session_id)

    # -- internals ---------------------------------------------------------- #

    def _ensure_open(self) -> None:
        if self._closed:
            raise StateStoreClosedError("web state store is closed")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one ``BEGIN IMMEDIATE`` unit of work, committing or rolling back.

        A caller exception (including the typed errors above) rolls back, so a failed
        allocation or ASK can never leave an incremented counter or a half-written row.
        """
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot begin transaction: {exc}") from exc
            try:
                yield self._connection
            except BaseException:
                _rollback_quietly(self._connection)
                raise
            try:
                self._connection.execute("COMMIT")
            except sqlite3.Error as exc:
                _rollback_quietly(self._connection)
                raise StateStoreError(f"cannot commit transaction: {exc}") from exc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _dumps_json(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    try:
        return json.dumps(dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serialisable: {exc}") from exc


def _loads_json(raw: object, *, table: str, row_key: str, field: str) -> dict[str, Any]:
    try:
        value = json.loads(raw if isinstance(raw, str) else str(raw))
    except (TypeError, ValueError) as exc:
        raise StateDataCorruption(
            f"{table}.{field} for {row_key!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise StateDataCorruption(
            f"{table}.{field} for {row_key!r} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _assert_transition(current: str, target: str) -> None:
    if target not in TASK_STATUSES:
        raise ValueError(f"unknown task status {target!r}")
    if target not in _ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise TaskStateConflict(
            f"task transition {current!r} -> {target!r} is not allowed"
        )


def _to_session_record(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=str(row["session_id"]),
        profile_id=str(row["profile_id"]),
        title=str(row["title"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        behavior_enabled=bool(row["behavior_enabled"]),
        turn_counter=int(row["turn_counter"]),
        reply_counter=int(row["reply_counter"]),
        reset_generation=int(row["reset_generation"]),
        retired=bool(row["retired"]),
    )


def _to_task_record(row: sqlite3.Row) -> TaskRecord:
    task_id = str(row["task_id"])
    return TaskRecord(
        task_id=task_id,
        session_id=str(row["session_id"]),
        status=str(row["status"]),
        task_version=int(row["task_version"]),
        reset_generation=int(row["reset_generation"]),
        current_requirements=_loads_json(
            row["current_requirements_json"],
            table="web_task", row_key=task_id, field="current_requirements_json",
        ),
        requirements_version=int(row["requirements_version"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        closed_at=None if row["closed_at"] is None else float(row["closed_at"]),
        close_reason=None if row["close_reason"] is None else str(row["close_reason"]),
    )


def _to_pending_record(row: sqlite3.Row) -> PendingRecord:
    pending_id = str(row["pending_id"])
    return PendingRecord(
        pending_id=pending_id,
        task_id=str(row["task_id"]),
        clarification_id=str(row["clarification_id"]),
        query=str(row["query"]),
        turn_id=str(row["turn_id"]),
        reply_turn_id=str(row["reply_turn_id"]),
        mode=str(row["mode"]),
        payload=_loads_json(
            row["payload_json"],
            table="web_pending", row_key=pending_id, field="payload_json",
        ),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        expires_at=float(row["expires_at"]),
        consumed_at=None if row["consumed_at"] is None else float(row["consumed_at"]),
    )


def pack_turn_payload(
    turn: Mapping[str, Any], *, ranking_source: str | None = None
) -> dict[str, Any]:
    """Build the versioned envelope stored for one turn; validates the minimum shape."""
    if not isinstance(turn, Mapping):
        raise ValueError("turn must be a mapping")
    turn_dict = dict(turn)
    turn_id = str(turn_dict.get("turn_id") or "").strip()
    if not turn_id:
        raise ValueError("turn payload must contain a non-empty turn_id")
    if not isinstance(turn_dict.get("user_text"), str):
        raise ValueError("turn payload must contain user_text")
    return {
        "schema": TURN_PAYLOAD_SCHEMA,
        "turn": turn_dict,
        "ranking_source": None if ranking_source is None else str(ranking_source),
    }


def _turn_seq_from_id(turn_id: str, session_id: str) -> int:
    prefix = f"{session_id}-t"
    if not turn_id.startswith(prefix):
        raise ValueError(
            f"turn id {turn_id!r} does not belong to session {session_id!r}"
        )
    suffix = turn_id[len(prefix):]
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError(f"turn id {turn_id!r} has no valid sequence number")
    return int(suffix)


def _to_turn_record(row: sqlite3.Row) -> TurnRecord:
    turn_id = str(row["turn_id"])
    payload = _loads_json(
        row["payload_json"], table="web_turn", row_key=turn_id, field="payload_json"
    )
    if payload.get("schema") != TURN_PAYLOAD_SCHEMA:
        raise StateDataCorruption(
            f"web_turn.payload_json for {turn_id!r} has unexpected schema "
            f"{payload.get('schema')!r}"
        )
    turn = payload.get("turn")
    if not isinstance(turn, dict) or str(turn.get("turn_id")) != turn_id:
        raise StateDataCorruption(
            f"web_turn.payload_json for {turn_id!r} does not contain its own turn"
        )
    return TurnRecord(
        session_id=str(row["session_id"]),
        turn_seq=int(row["turn_seq"]),
        turn_id=turn_id,
        task_id=None if row["task_id"] is None else str(row["task_id"]),
        payload=payload,
        created_at=float(row["created_at"]),
    )


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:  # pragma: no cover - nothing left to roll back
        pass


def _close_quietly(connection: sqlite3.Connection) -> None:
    """Close a connection during failed construction without masking the original error."""
    try:
        connection.close()
    except sqlite3.Error:  # pragma: no cover - defensive
        pass
