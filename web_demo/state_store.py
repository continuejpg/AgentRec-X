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

import hashlib
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
    "RequestClaim",
    "RequestRecord",
    "SQLiteWebStateStore",
    "SessionRecord",
    "TaskRecord",
    "TURN_PAYLOAD_SCHEMA",
    "TurnRecord",
    "reply_payload_fingerprint",
    "WEB_STATE_SCHEMA_VERSION",
    "ActiveTaskExists",
    "AskCommitError",
    "IdentityAllocationError",
    "PendingExpired",
    "PendingStateConflict",
    "RequestAmbiguous",
    "RequestBusy",
    "RequestConflict",
    "RequestFenced",
    "RequestNotFound",
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
#: v3 = S3b (persisted chat turns), v4 = S5a (durable reply request ledger).
WEB_STATE_SCHEMA_VERSION = 6

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
#: Versioned envelope written to ``web_request.result_json`` for completed replies.
REPLY_RESULT_SCHEMA = "agentrecx.web.reply_result.v1"

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

_SCHEMA_V4_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS web_request (
        session_id                TEXT NOT NULL REFERENCES web_session(session_id),
        request_id                TEXT NOT NULL,
        pending_id                TEXT NOT NULL,
        task_id                   TEXT NOT NULL,
        request_kind              TEXT NOT NULL,
        payload_fingerprint       TEXT NOT NULL,
        status                    TEXT NOT NULL CHECK (status IN
                                    ('PROCESSING','COMPLETED','FAILED')),
        claim_token               TEXT,
        claim_attempt             INTEGER NOT NULL DEFAULT 0 CHECK (claim_attempt >= 0),
        lease_expires_at          REAL,
        task_version_at_claim     INTEGER NOT NULL,
        reset_generation_at_claim INTEGER NOT NULL,
        result_json               TEXT,
        created_at                REAL NOT NULL,
        updated_at                REAL NOT NULL,
        PRIMARY KEY (session_id, request_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_web_request_pending ON web_request(pending_id)
    """,
)

_SCHEMA_V5_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE web_request ADD COLUMN execution_stage TEXT NOT NULL DEFAULT 'CLAIMED'",
    "ALTER TABLE web_request ADD COLUMN payload_json TEXT",
    """
    UPDATE web_request SET execution_stage = CASE
        WHEN status = 'COMPLETED' THEN 'COMPLETED' ELSE 'AMBIGUOUS' END
    """,
)

#: v6 adds the Cold Resume snapshot.  Existing rows stay NULL (``PendingRecord.continuation``
#: is then ``None``) and such pendings are refused instead of being guessed at.
_SCHEMA_V6_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE web_pending ADD COLUMN continuation_json TEXT",
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
    "payload_json, status, created_at, expires_at, consumed_at, continuation_json"
)
_TURN_COLUMNS = "session_id, turn_seq, turn_id, task_id, payload_json, created_at"
_REQUEST_COLUMNS = (
    "execution_stage, payload_json, "
    "session_id, request_id, pending_id, task_id, request_kind, payload_fingerprint, "
    "status, claim_token, claim_attempt, lease_expires_at, task_version_at_claim, "
    "reset_generation_at_claim, result_json, created_at, updated_at"
)
_PENDING_ALIASED_COLUMNS = (
    "p.pending_id, p.task_id, p.clarification_id, p.query, p.turn_id, p.reply_turn_id, "
    "p.mode, p.payload_json, p.status, p.created_at, p.expires_at, p.consumed_at, "
    "p.continuation_json"
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


class RequestNotFound(StateStoreError):
    """No request ledger row exists for this session/request id."""


class RequestConflict(StateStoreError):
    """The same request id was reused with different request content."""


class RequestBusy(StateStoreError):
    """Another executor holds a live claim lease for this request."""


class RequestFenced(StateStoreError):
    """The caller's claim token/state is stale and cannot commit."""


class RequestAmbiguous(StateStoreError):
    """External execution may have happened but its result is unrecoverable."""


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
    #: Versioned Cold Resume snapshot written by :meth:`commit_ask`; ``None`` for every
    #: pending created before v6 or without a restorable MCP context.
    continuation: dict[str, Any] | None = None


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


@dataclass(frozen=True)
class RequestRecord:
    """One durable clarification-reply request and its claim/fencing state."""

    session_id: str
    request_id: str
    pending_id: str
    task_id: str
    request_kind: str
    payload_fingerprint: str
    status: str
    claim_token: str | None
    claim_attempt: int
    lease_expires_at: float | None
    task_version_at_claim: int
    reset_generation_at_claim: int
    result: dict[str, Any] | None
    created_at: float
    updated_at: float
    execution_stage: str = "CLAIMED"
    reply_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class RequestClaim:
    """Outcome of :meth:`SQLiteWebStateStore.claim_reply_request`."""

    outcome: str  # "claimed" | "reclaimed" | "replay"
    request: RequestRecord
    claim_token: str | None


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
                ("web_session", "web_identity", "web_task", "web_pending", "web_turn",
                 "web_request")
            )

    def _create_latest_schema_locked(self) -> None:
        with self._transaction() as connection:
            if self._table_names(connection):
                # Another process created it between the probe and this transaction.
                return
            for statement in (
                _SCHEMA_V1_STATEMENTS + _SCHEMA_V2_STATEMENTS
                + _SCHEMA_V3_STATEMENTS + _SCHEMA_V4_STATEMENTS
                + _SCHEMA_V5_STATEMENTS + _SCHEMA_V6_STATEMENTS
            ):
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
            if version not in (1, 2, 3, 4, 5):
                raise SchemaVersionError(
                    f"cannot migrate schema version {version} to {WEB_STATE_SCHEMA_VERSION}"
                )
            try:
                if version == 1:
                    for statement in _SCHEMA_V2_STATEMENTS:
                        connection.execute(statement)
                if version <= 2:
                    for statement in _SCHEMA_V3_STATEMENTS:
                        connection.execute(statement)
                for statement in _SCHEMA_V4_STATEMENTS:
                    connection.execute(statement)
                if version <= 4:
                    for statement in _SCHEMA_V5_STATEMENTS:
                        connection.execute(statement)
                if version <= 5:
                    for statement in _SCHEMA_V6_STATEMENTS:
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
        continuation: Mapping[str, Any] | None = None,
    ) -> PendingRecord:
        """Persist one ASK round atomically.

        In a single ``BEGIN IMMEDIATE`` transaction: validate the session (not retired),
        the session/task generation pair, ownership, task status, expected version, the
        absence of another OPEN pending, and the source turn identity; then allocate the
        reply identity, insert the pending row, move the task to ``WAITING_FOR_USER`` and
        bump ``task_version``.  Any failure rolls the whole unit back, so there is no
        state where a reply identity was issued without its pending row.

        ``continuation`` is the versioned Cold Resume snapshot
        (:mod:`web_demo.continuation`).  Its Web binding fields are overwritten with the
        values committed here (session/profile/task/pending/clarification/mode/generation/
        expiry), so the snapshot can never disagree with the row it belongs to.
        """
        session_id = _require_text(session_id, "session_id")
        task_id = _require_text(task_id, "task_id")
        clarification_id = _require_text(clarification_id, "clarification_id")
        source_turn_id = _require_text(source_turn_id, "source_turn_id")
        query = _require_text(query, "query")
        if mode not in PENDING_MODES:
            raise ValueError(f"mode must be one of {sorted(PENDING_MODES)}, got {mode!r}")
        payload_json = _dumps_json(payload, "payload")
        if continuation is not None and not isinstance(continuation, Mapping):
            raise ValueError("continuation must be a mapping")
        timestamp = time.time()
        expires_at = float(expires_at)
        if expires_at <= timestamp:
            raise ValueError("expires_at must be in the future")
        pending_id = f"pend-{uuid.uuid4().hex}"

        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired, profile_id FROM web_session "
                "WHERE session_id = ?",
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

            continuation_json: str | None = None
            if continuation is not None:
                # The row is the authority for the Web binding: a caller cannot store a
                # snapshot that points at another session/task/pending/generation/expiry.
                stamped = dict(continuation)
                stamped.update({
                    "session_id": session_id,
                    "profile_id": str(session["profile_id"]),
                    "web_task_id": task_id,
                    "pending_id": pending_id,
                    "clarification_id": clarification_id,
                    "mode": mode,
                    "reset_generation": int(task["reset_generation"]),
                    "expires_at": expires_at,
                })
                continuation_json = _dumps_json(stamped, "continuation")

            try:
                connection.execute(
                    "INSERT INTO web_pending(pending_id, task_id, clarification_id, query, "
                    "turn_id, reply_turn_id, mode, payload_json, status, created_at, "
                    "expires_at, consumed_at, continuation_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, NULL, ?)",
                    (pending_id, task_id, clarification_id, query, source_turn_id,
                     reply_turn_id, mode, payload_json, timestamp, expires_at,
                     continuation_json),
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

    def get_pending_for_clarification(
        self, session_id: str, clarification_id: str
    ) -> PendingRecord | None:
        """Return the session's most recent ASK for this clarification id (any status).

        The reply path needs the durable ``pending_id``/``reply_turn_id`` both for a fresh
        OPEN pending and for an idempotent replay after the pending was CONSUMED.  This is a
        read only; every state transition is still enforced by the claim/finalize methods.
        """
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_PENDING_ALIASED_COLUMNS} FROM web_pending p "
                    "JOIN web_task t ON p.task_id = t.task_id "
                    "WHERE t.session_id = ? AND p.clarification_id = ? "
                    "ORDER BY p.created_at DESC, p.pending_id DESC LIMIT 1",
                    (session_id, str(clarification_id)),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read session clarification: {exc}") from exc
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
        turn_seq: int | None = None,
    ) -> str:
        envelope = pack_turn_payload(turn, ranking_source=ranking_source)
        turn_id = str(envelope["turn"]["turn_id"])
        _validate_turn_id(turn_id, session_id)
        if turn_seq is None:
            # Insertion order across chat turns (-tN) and reply turns (-rN): a reply turn
            # must follow its source turn and must never collide with a later chat turn.
            row = connection.execute(
                "SELECT COALESCE(MAX(turn_seq), 0) AS current FROM web_turn "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn_seq = int(row["current"]) + 1
        try:
            connection.execute(
                "INSERT INTO web_turn(session_id, turn_seq, turn_id, task_id, "
                "payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_seq, turn_id, task_id,
                 json.dumps(envelope, sort_keys=True, ensure_ascii=False), timestamp),
            )
        except sqlite3.IntegrityError as exc:
            raise StateStoreError(
                f"turn {turn_id!r} is already persisted for session {session_id!r}"
            ) from exc
        return turn_id

    # -- reply request ledger (S5a) ----------------------------------------- #

    def get_request(self, session_id: str, request_id: str) -> RequestRecord | None:
        with self._lock:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                    "WHERE session_id = ? AND request_id = ?",
                    (session_id, request_id),
                ).fetchone()
            except sqlite3.Error as exc:
                raise StateStoreError(f"cannot read request: {exc}") from exc
        return None if row is None else _to_request_record(row)

    def claim_reply_request(
        self,
        session_id: str,
        pending_id: str,
        *,
        payload: Mapping[str, Any],
        lease_duration: float = 60.0,
        now: float | None = None,
    ) -> RequestClaim:
        """Claim the durable reply request identified by the pending's reply id.

        ``request_id`` is the pending's ``reply_turn_id``: it is already persisted, unique
        per pending and stable across HTTP retries and process restarts.  A different
        payload under the same id is a conflict, never a silent reuse.
        """
        if lease_duration <= 0:
            raise ValueError("lease_duration must be positive")
        reference = time.time() if now is None else float(now)
        fingerprint = reply_payload_fingerprint(session_id, pending_id, payload)
        token = f"claim-{uuid.uuid4().hex}"
        stored_payload = json.dumps({
            "message": str(payload.get("message") or ""),
            "confirmed_by_user": bool(payload.get("confirmed_by_user", True)),
            "confirmation_source": str(payload.get("confirmation_source")
                                     or "web_ui_explicit_user_reply"),
        }, sort_keys=True, ensure_ascii=False)
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(session["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")

            pending = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                (pending_id,),
            ).fetchone()
            if pending is None:
                raise PendingStateConflict(f"no pending {pending_id!r}")
            task = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?",
                (pending["task_id"],),
            ).fetchone()
            if task is None or str(task["session_id"]) != session_id:
                raise UnknownTask(f"pending {pending_id!r} does not belong to {session_id!r}")
            request_id = str(pending["reply_turn_id"])
            existing = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_fingerprint"]) != fingerprint:
                    raise RequestConflict(
                        "the same request id was reused with different request content"
                    )
                stage = str(existing["execution_stage"])
                if stage in {"EXTERNAL_STARTED", "AMBIGUOUS"}:
                    raise RequestAmbiguous(
                        "external execution started but no durable result exists; "
                        "automatic retry is refused"
                    )
                if stage == "RESULT_DURABLE":
                    if (
                        existing["lease_expires_at"] is not None
                        and float(existing["lease_expires_at"]) > reference
                    ):
                        raise RequestBusy(
                            f"request {request_id!r} result is leased by another executor"
                        )
                    updated = connection.execute(
                        "UPDATE web_request SET claim_token = ?, "
                        "claim_attempt = claim_attempt + 1, lease_expires_at = ?, "
                        "updated_at = ? WHERE session_id = ? AND request_id = ? "
                        "AND status = 'PROCESSING' AND execution_stage = 'RESULT_DURABLE' "
                        "AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
                        (token, reference + float(lease_duration), reference, session_id,
                         request_id, reference),
                    )
                    if updated.rowcount != 1:  # pragma: no cover
                        raise RequestBusy(f"request {request_id!r} changed during takeover")
                    row = connection.execute(
                        f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                        "WHERE session_id = ? AND request_id = ?",
                        (session_id, request_id),
                    ).fetchone()
                    return RequestClaim("result_ready", _to_request_record(row), token)
                if str(existing["status"]) == "COMPLETED":
                    # Replay before task/pending state checks: a completed reply legitimately
                    # leaves the task COMPLETED and the pending CONSUMED.  It must still be
                    # generation-aware: a reset makes the old success ineligible.
                    if int(existing["reset_generation_at_claim"]) != int(
                        session["reset_generation"]
                    ):
                        raise RequestFenced(
                            "completed request belongs to a previous reset generation"
                        )
                    return RequestClaim("replay", _to_request_record(existing), None)
            if str(task["status"]) != "WAITING_FOR_USER":
                raise TaskStateConflict(f"task {task['task_id']!r} is not waiting for a reply")
            if int(task["reset_generation"]) != int(session["reset_generation"]):
                raise TaskStateConflict("task belongs to a previous reset generation")
            if str(pending["status"]) != "OPEN":
                raise PendingStateConflict(
                    f"pending {pending_id!r} is {pending['status']!r}, not OPEN"
                )
            if float(pending["expires_at"]) <= reference:
                raise PendingExpired(f"pending {pending_id!r} has expired")

            request_id = str(pending["reply_turn_id"])
            existing = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()

            if existing is None:
                connection.execute(
                    "INSERT INTO web_request(session_id, request_id, pending_id, task_id, "
                    "request_kind, payload_fingerprint, status, claim_token, claim_attempt, "
                    "lease_expires_at, task_version_at_claim, reset_generation_at_claim, "
                    "result_json, created_at, updated_at, execution_stage, payload_json) "
                    "VALUES (?, ?, ?, ?, 'clarification_reply', ?, 'PROCESSING', ?, 1, ?, ?, ?, "
                    "NULL, ?, ?, 'CLAIMED', ?)",
                    (session_id, request_id, pending_id, task["task_id"], fingerprint, token,
                     reference + float(lease_duration), int(task["task_version"]),
                     int(task["reset_generation"]), reference, reference, stored_payload),
                )
                outcome = "claimed"
            else:
                if str(existing["payload_fingerprint"]) != fingerprint:
                    raise RequestConflict(
                        "the same request id was reused with different request content"
                    )
                if str(existing["status"]) == "COMPLETED":
                    row = connection.execute(
                        f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                        "WHERE session_id = ? AND request_id = ?",
                        (session_id, request_id),
                    ).fetchone()
                    return RequestClaim("replay", _to_request_record(row), None)
                if str(existing["status"]) == "PROCESSING" and (
                    existing["lease_expires_at"] is not None
                    and float(existing["lease_expires_at"]) > reference
                ):
                    raise RequestBusy(
                        f"request {request_id!r} is leased by another executor"
                    )
                updated = connection.execute(
                    "UPDATE web_request SET status = 'PROCESSING', claim_token = ?, "
                    "claim_attempt = claim_attempt + 1, lease_expires_at = ?, "
                    "task_version_at_claim = ?, reset_generation_at_claim = ?, "
                    "updated_at = ? WHERE session_id = ? AND request_id = ? "
                    "AND status IN ('PROCESSING','FAILED') "
                    "AND (lease_expires_at IS NULL OR lease_expires_at <= ?)",
                    (token, reference + float(lease_duration), int(task["task_version"]),
                     int(task["reset_generation"]), reference, session_id, request_id,
                     reference),
                )
                if updated.rowcount != 1:  # pragma: no cover - guarded by the checks above
                    raise RequestBusy(f"request {request_id!r} changed during takeover")
                outcome = "reclaimed"

            row = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
            return RequestClaim(outcome, _to_request_record(row), token)

    def renew_request_lease(
        self,
        session_id: str,
        request_id: str,
        *,
        claim_token: str,
        lease_duration: float = 60.0,
        now: float | None = None,
    ) -> RequestRecord:
        """Extend a live claim lease; a stale token or expiry is fenced, never renewed."""
        if lease_duration <= 0:
            raise ValueError("lease_duration must be positive")
        reference = time.time() if now is None else float(now)
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            updated = connection.execute(
                "UPDATE web_request SET lease_expires_at = ?, updated_at = ? "
                "WHERE session_id = ? AND request_id = ? AND status = 'PROCESSING' "
                "AND claim_token = ? AND reset_generation_at_claim = ? "
                "AND (lease_expires_at IS NULL OR lease_expires_at > ?)",
                (reference + float(lease_duration), reference, session_id, request_id,
                 claim_token, int(session["reset_generation"]), reference),
            )
            if updated.rowcount != 1:
                raise RequestFenced("lease cannot be renewed: stale token, expiry or reset")
            row = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return _to_request_record(row)

    def mark_request_external_started(
        self,
        session_id: str,
        request_id: str,
        *,
        claim_token: str,
        now: float | None = None,
    ) -> RequestRecord:
        """Persist EXTERNAL_STARTED *before* the MCP tool is invoked."""
        reference = time.time() if now is None else float(now)
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            updated = connection.execute(
                "UPDATE web_request SET execution_stage = 'EXTERNAL_STARTED', "
                "updated_at = ? WHERE session_id = ? AND request_id = ? "
                "AND status = 'PROCESSING' AND execution_stage = 'CLAIMED' "
                "AND claim_token = ? AND reset_generation_at_claim = ? "
                "AND (lease_expires_at IS NULL OR lease_expires_at > ?)",
                (reference, session_id, request_id, claim_token,
                 int(session["reset_generation"]), reference),
            )
            if updated.rowcount != 1:
                raise RequestFenced(
                    "external execution cannot start: stale token, lease or reset"
                )
            row = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return _to_request_record(row)

    def checkpoint_reply_result(
        self,
        session_id: str,
        request_id: str,
        *,
        claim_token: str,
        result: Mapping[str, Any],
        now: float | None = None,
    ) -> RequestRecord:
        """Persist the complete tool result so Finalize never has to call MCP again."""
        if not isinstance(result, Mapping):
            raise ValueError("result must be a mapping")
        reference = time.time() if now is None else float(now)
        payload = json.dumps(
            {"schema": REPLY_RESULT_SCHEMA, "payload": dict(result)},
            sort_keys=True, ensure_ascii=False,
        )
        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            existing = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
            if existing is None:
                raise RequestNotFound(f"no request {request_id!r}")
            if (
                str(existing["status"]) != "PROCESSING"
                or existing["claim_token"] != claim_token
            ):
                raise RequestFenced("claim token is stale or the request is not processing")
            if str(existing["execution_stage"]) not in {"EXTERNAL_STARTED", "RESULT_DURABLE"}:
                raise RequestFenced("external execution has not started for this request")
            if (
                existing["lease_expires_at"] is not None
                and float(existing["lease_expires_at"]) <= reference
            ):
                raise RequestFenced("claim lease expired before result checkpoint")
            if int(existing["reset_generation_at_claim"]) != int(session["reset_generation"]):
                raise RequestFenced("session was reset after this request was claimed")
            if existing["result_json"] is not None:
                stored = _loads_json(existing["result_json"], table="web_request",
                                     row_key=request_id, field="result_json")
                if stored.get("payload") != dict(result):
                    raise RequestConflict(
                        "a different result was already checkpointed for this request"
                    )
                return _to_request_record(existing)
            updated = connection.execute(
                "UPDATE web_request SET result_json = ?, execution_stage = 'RESULT_DURABLE', "
                "updated_at = ? WHERE session_id = ? AND request_id = ? "
                "AND status = 'PROCESSING' AND claim_token = ? "
                "AND execution_stage = 'EXTERNAL_STARTED'",
                (payload, reference, session_id, request_id, claim_token),
            )
            if updated.rowcount != 1:
                raise RequestFenced("result checkpoint lost a race with another executor")
            row = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return _to_request_record(row)

    def mark_request_ambiguous(
        self,
        session_id: str,
        request_id: str,
        *,
        claim_token: str,
        reason: str = "external_result_lost",
        now: float | None = None,
    ) -> RequestRecord:
        """Fail closed: side effects may have happened, result is not recoverable."""
        reference = time.time() if now is None else float(now)
        with self._transaction() as connection:
            updated = connection.execute(
                "UPDATE web_request SET status = 'FAILED', execution_stage = 'AMBIGUOUS', "
                "result_json = ?, updated_at = ? WHERE session_id = ? AND request_id = ? "
                "AND status = 'PROCESSING' AND claim_token = ? "
                "AND execution_stage = 'EXTERNAL_STARTED'",
                (json.dumps({"schema": REPLY_RESULT_SCHEMA, "error": str(reason)}),
                 reference, session_id, request_id, claim_token),
            )
            if updated.rowcount != 1:
                raise RequestFenced("only an EXTERNAL_STARTED claim can become AMBIGUOUS")
            row = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return _to_request_record(row)

    def finalize_reply_request(
        self,
        session_id: str,
        request_id: str,
        *,
        claim_token: str,
        outcome: str,
        requirements: Mapping[str, Any] | None = None,
        turn: Mapping[str, Any] | None = None,
        ranking_source: str | None = None,
        result: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> tuple[TaskRecord, PendingRecord, RequestRecord]:
        """Atomically commit a claimed reply: Task, Pending and Request in one transaction.

        ``outcome`` is ``"resumed"`` (task back to RUNNING) or ``"completed"`` (task
        COMPLETED).  The task version is the one captured at claim time, and reset
        generations are re-checked, so a reset or a newer executor fences this commit.
        """
        if outcome not in {"resumed", "completed"}:
            raise ValueError(f"unsupported finalize outcome {outcome!r}")
        reference = time.time() if now is None else float(now)
        requirements_json = (
            None if requirements is None else _dumps_json(requirements, "requirements")
        )
        if result is not None and not isinstance(result, Mapping):
            raise ValueError("result must be a mapping")

        with self._transaction() as connection:
            session = connection.execute(
                "SELECT reset_generation, retired FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise UnknownSession(f"no session {session_id!r}")
            if int(session["retired"]):
                raise SessionRetired(f"session {session_id!r} is retired")

            request = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
            if request is None:
                raise RequestNotFound(f"no request {request_id!r} for {session_id!r}")
            if str(request["status"]) != "PROCESSING" or request["claim_token"] != claim_token:
                raise RequestFenced("claim token is stale or the request is not processing")
            if (
                request["lease_expires_at"] is not None
                and float(request["lease_expires_at"]) <= reference
            ):
                raise RequestFenced("claim lease expired before finalize; the result was lost")
            stage = str(request["execution_stage"])
            if stage not in {"CLAIMED", "EXTERNAL_STARTED", "RESULT_DURABLE"}:
                raise RequestFenced(
                    f"request in stage {stage!r} cannot be finalized"
                )
            if int(request["reset_generation_at_claim"]) != int(session["reset_generation"]):
                raise RequestFenced("session was reset after this request was claimed")

            task = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?",
                (request["task_id"],),
            ).fetchone()
            if task is None or str(task["session_id"]) != session_id:
                raise RequestFenced("request task no longer belongs to this session")
            if int(task["reset_generation"]) != int(session["reset_generation"]):
                raise RequestFenced("task belongs to a previous reset generation")

            pending = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                (request["pending_id"],),
            ).fetchone()
            if pending is None or str(pending["task_id"]) != str(task["task_id"]):
                raise PendingStateConflict("pending no longer belongs to the request task")
            if str(pending["status"]) != "OPEN":
                raise PendingStateConflict(
                    f"pending is {pending['status']!r}, not OPEN"
                )
            if turn is not None:
                turn_id = str(turn.get("turn_id") or "")
                if turn_id != str(pending["reply_turn_id"]):
                    raise RequestFenced(
                        "reply turn id must equal the reply_turn_id issued at ASK time"
                    )
            if stage == "RESULT_DURABLE" and request["result_json"] is not None:
                checkpoint = _loads_json(
                    request["result_json"], table="web_request",
                    row_key=request_id, field="result_json",
                )
                checkpoint_payload = checkpoint.get("payload")
                if result is None:
                    result = checkpoint_payload
                elif dict(result) != checkpoint_payload:
                    raise RequestConflict(
                        "finalize result differs from the durable checkpoint"
                    )

            target_status = "RUNNING" if outcome == "resumed" else "COMPLETED"
            if outcome == "resumed":
                updated_task = connection.execute(
                    "UPDATE web_task SET status = 'RUNNING', task_version = task_version + 1, "
                    "updated_at = ? WHERE task_id = ? AND status = 'WAITING_FOR_USER' "
                    "AND task_version = ?",
                    (reference, task["task_id"], int(request["task_version_at_claim"])),
                )
            elif requirements_json is None:
                updated_task = connection.execute(
                    "UPDATE web_task SET status = 'COMPLETED', task_version = task_version + 1, "
                    "updated_at = ?, closed_at = ? WHERE task_id = ? "
                    "AND status = 'WAITING_FOR_USER' AND task_version = ?",
                    (reference, reference, task["task_id"],
                     int(request["task_version_at_claim"])),
                )
            else:
                updated_task = connection.execute(
                    "UPDATE web_task SET status = 'COMPLETED', task_version = task_version + 1, "
                    "current_requirements_json = ?, requirements_version = "
                    "requirements_version + 1, updated_at = ?, closed_at = ? "
                    "WHERE task_id = ? AND status = 'WAITING_FOR_USER' AND task_version = ?",
                    (requirements_json, reference, reference, task["task_id"],
                     int(request["task_version_at_claim"])),
                )
            if updated_task.rowcount != 1:
                raise StaleTaskVersion("task changed since this request was claimed")

            if turn is not None:
                # Same transaction: the reply turn is visible if and only if the reply
                # completed.  It reuses the reply identity issued at ASK time.
                self._insert_turn_in_transaction(
                    connection, session_id, str(task["task_id"]), turn,
                    ranking_source, reference,
                )

            consumed = connection.execute(
                "UPDATE web_pending SET status = 'CONSUMED', consumed_at = ? "
                "WHERE pending_id = ? AND task_id = ? AND status = 'OPEN'",
                (reference, request["pending_id"], task["task_id"]),
            )
            if consumed.rowcount != 1:
                raise PendingStateConflict("pending is no longer consumable by this request")

            result_payload = {
                "schema": REPLY_RESULT_SCHEMA,
                "outcome": outcome,
                "request_id": request_id,
                "pending_id": str(request["pending_id"]),
                "task_id": str(task["task_id"]),
                "turn_id": None if turn is None else str(turn.get("turn_id")),
                "payload": None if result is None else dict(result),
            }
            finalized = connection.execute(
                "UPDATE web_request SET status = 'COMPLETED', execution_stage = 'COMPLETED', "
                "result_json = ?, updated_at = ? "
                "WHERE session_id = ? AND request_id = ? AND status = 'PROCESSING' "
                "AND claim_token = ?",
                (json.dumps(result_payload, sort_keys=True, ensure_ascii=False), reference,
                 session_id, request_id, claim_token),
            )
            if finalized.rowcount != 1:  # pragma: no cover - guarded above
                raise RequestFenced("request was completed by another executor")

            task_row = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM web_task WHERE task_id = ?", (task["task_id"],)
            ).fetchone()
            pending_row = connection.execute(
                f"SELECT {_PENDING_COLUMNS} FROM web_pending WHERE pending_id = ?",
                (request["pending_id"],),
            ).fetchone()
            request_row = connection.execute(
                f"SELECT {_REQUEST_COLUMNS} FROM web_request "
                "WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return (
            _to_task_record(task_row),
            _to_pending_record(pending_row),
            _to_request_record(request_row),
        )

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
        continuation=(
            None if row["continuation_json"] is None
            else _loads_json(
                row["continuation_json"],
                table="web_pending", row_key=pending_id, field="continuation_json",
            )
        ),
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


def _validate_turn_id(turn_id: str, session_id: str) -> None:
    """A stored turn must belong to the session and carry a real sequence suffix."""
    prefix = f"{session_id}-"
    if not turn_id.startswith(prefix):
        raise ValueError(f"turn id {turn_id!r} does not belong to session {session_id!r}")
    suffix = turn_id[len(prefix):]
    if len(suffix) < 2 or suffix[0] not in {"t", "r"} or not suffix[1:].isdigit():
        raise ValueError(f"turn id {turn_id!r} has no valid sequence number")
    if int(suffix[1:]) < 1:
        raise ValueError(f"turn id {turn_id!r} has no valid sequence number")


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


def reply_payload_fingerprint(
    session_id: str, pending_id: str, payload: Mapping[str, Any]
) -> str:
    """Deterministic SHA-256 over the fields that decide a reply's logical identity."""
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    canonical = {
        "session_id": str(session_id),
        "pending_id": str(pending_id),
        "message": str(payload.get("message") or ""),
        "confirmed_by_user": bool(payload.get("confirmed_by_user", True)),
        "confirmation_source": str(
            payload.get("confirmation_source") or "web_ui_explicit_user_reply"
        ),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _to_request_record(row: sqlite3.Row) -> RequestRecord:
    request_id = str(row["request_id"])
    result = (
        None
        if row["result_json"] is None
        else _loads_json(row["result_json"], table="web_request", row_key=request_id,
                         field="result_json")
    )
    if result is not None and result.get("schema") != REPLY_RESULT_SCHEMA:
        raise StateDataCorruption(
            f"web_request.result_json for {request_id!r} has unexpected schema "
            f"{result.get('schema')!r}"
        )
    return RequestRecord(
        session_id=str(row["session_id"]),
        request_id=request_id,
        pending_id=str(row["pending_id"]),
        task_id=str(row["task_id"]),
        request_kind=str(row["request_kind"]),
        payload_fingerprint=str(row["payload_fingerprint"]),
        status=str(row["status"]),
        claim_token=None if row["claim_token"] is None else str(row["claim_token"]),
        claim_attempt=int(row["claim_attempt"]),
        lease_expires_at=(
            None if row["lease_expires_at"] is None else float(row["lease_expires_at"])
        ),
        task_version_at_claim=int(row["task_version_at_claim"]),
        reset_generation_at_claim=int(row["reset_generation_at_claim"]),
        result=result,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        execution_stage=str(row["execution_stage"]),
        reply_payload=(
            None if row["payload_json"] is None
            else _loads_json(row["payload_json"], table="web_request",
                             row_key=request_id, field="payload_json")
        ),
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
