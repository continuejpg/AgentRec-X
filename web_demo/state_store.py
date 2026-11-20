"""Durable identity state for the Web Demo (M1-C S2a).

Scope is deliberately narrow: **session identity and monotonically allocated turn /
reply identities only**.  This module does not store chat turns, product snapshots,
tasks, pending clarifications or request idempotency records (later slices), and it is
*not* a serialisation format for ``Session.view()``.

Why the counters live here
--------------------------
``web_demo.sessions.Session`` keeps ``_turn_counter`` / ``_reply_counter`` in memory, so
a process restart reissues identifiers M9 has already recorded (D-1).  The identifiers
are business identities, not conversation history: once handed to M9 they become part of
that turn's idempotency key and must never be issued twice.  This store therefore owns
the authoritative sequence and an append-only ledger of every identity it has issued.

Mechanism
---------
``sqlite3`` from the standard library, WAL, ``foreign_keys=ON``, ``busy_timeout`` and
``BEGIN IMMEDIATE`` around every mutation.  Cross-process uniqueness comes from SQLite
(``BEGIN IMMEDIATE`` + primary keys), never from an in-process lock; the ``RLock`` below
only prevents two threads from interleaving statements on the *same* connection.

Schema migration
----------------
``meta.schema_version`` is written on creation and verified on every open.  An unknown
version, a non-SQLite file, or a SQLite database that is not this schema is reported as
an error: the store never silently rebuilds or overwrites an existing database.

Timestamps are Unix epoch seconds (``REAL``), matching the other SQLite stores in this
repository.  A later slice that rebuilds a ``web_demo.sessions.Session`` converts them to
the ISO-8601 strings that ``Session.created_at`` expects.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

__all__ = [
    "DEFAULT_SESSION_TITLE",
    "SQLiteWebStateStore",
    "SessionRecord",
    "IdentityAllocationError",
    "SchemaVersionError",
    "SessionAlreadyExists",
    "SessionRetired",
    "StateStoreClosedError",
    "StateStoreError",
    "UnknownSession",
    "WEB_STATE_SCHEMA_VERSION",
]

#: Schema version written to ``meta`` and verified on every open.
WEB_STATE_SCHEMA_VERSION = 1

#: Mirrors ``web_demo.sessions.DEFAULT_TITLE``; a test asserts the two stay equal.
DEFAULT_SESSION_TITLE = "New shopping session"

#: SQLite waits for a competing writer instead of failing with "database is locked".
DEFAULT_BUSY_TIMEOUT_MS = 5_000

#: kind -> (counter column, identifier prefix).  The prefix keeps ids compatible with
#: ``Session.next_turn_id()`` / ``Session.next_reply_turn_id()``.
_ID_KINDS: dict[str, tuple[str, str]] = {
    "turn": ("turn_counter", "t"),
    "reply": ("reply_counter", "r"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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
);

CREATE TABLE IF NOT EXISTS web_identity (
    session_id TEXT NOT NULL REFERENCES web_session(session_id),
    kind       TEXT NOT NULL CHECK (kind IN ('turn', 'reply')),
    seq        INTEGER NOT NULL CHECK (seq >= 1),
    value      TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, kind, value),
    UNIQUE (session_id, kind, seq)
);
"""

_SESSION_COLUMNS = (
    "session_id, profile_id, title, created_at, updated_at, behavior_enabled, "
    "turn_counter, reply_counter, reset_generation, retired"
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class StateStoreError(RuntimeError):
    """Base class for every web-state-store failure."""


class SchemaVersionError(StateStoreError):
    """The database was written by an incompatible schema version."""


class UnknownSession(StateStoreError):
    """No session row exists for the requested ``session_id``."""


class SessionAlreadyExists(StateStoreError):
    """A session with this ``session_id`` has already been created."""


class SessionRetired(StateStoreError):
    """The session is retired and can no longer mutate or allocate identities."""


class IdentityAllocationError(StateStoreError):
    """An identity could not be allocated; the transaction was rolled back."""


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


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class SQLiteWebStateStore:
    """SQLite-backed, process-shared session identity store."""

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
            return self._read_schema_version_locked()

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
            try:
                tables = {
                    str(row["name"])
                    for row in self._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            except sqlite3.DatabaseError as exc:
                raise StateStoreError(
                    f"{self._path!r} is not a readable SQLite database: {exc}"
                ) from exc

            if not tables:
                # Brand-new (or empty) database: create this schema.
                try:
                    self._connection.executescript(_SCHEMA)
                    self._connection.execute(
                        "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                        (str(WEB_STATE_SCHEMA_VERSION),),
                    )
                except sqlite3.IntegrityError:
                    # Another process created it concurrently; fall through to verification.
                    pass
                except sqlite3.Error as exc:
                    raise StateStoreError(
                        f"cannot initialise web state schema in {self._path!r}: {exc}"
                    ) from exc
                self._verify_schema_version_locked()
                return

            if "meta" not in tables or "web_session" not in tables:
                # Never adopt or overwrite a database that belongs to something else.
                raise StateStoreError(
                    f"{self._path!r} is a SQLite database without the web state schema; "
                    "refusing to modify it"
                )
            self._verify_schema_version_locked()

    def _read_schema_version_locked(self) -> int:
        try:
            row = self._connection.execute(
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

    def _verify_schema_version_locked(self) -> None:
        version = self._read_schema_version_locked()
        if version != WEB_STATE_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"web state database {self._path!r} has schema version {version}, "
                f"expected {WEB_STATE_SCHEMA_VERSION}"
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
        return _to_record(row)

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
        return None if row is None else _to_record(row)

    def retire_session(self, session_id: str) -> SessionRecord:
        """Mark a session permanently unusable.  Terminal and idempotent.

        Retirement is *not* the Web Demo reset: reset keeps the session alive and only
        advances ``reset_generation``.  A retired session can no longer allocate
        identities or be reset.
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
        return _to_record(row)

    # -- identity allocation ------------------------------------------------ #

    def allocate_turn_id(self, session_id: str) -> str:
        """Allocate the next ``{session_id}-t{n}`` identity durably."""
        return self._allocate(session_id, "turn")

    def allocate_reply_id(self, session_id: str) -> str:
        """Allocate the next ``{session_id}-r{n}`` identity durably."""
        return self._allocate(session_id, "reply")

    def _allocate(self, session_id: str, kind: Literal["turn", "reply"]) -> str:
        column, prefix = _ID_KINDS[kind]
        timestamp = time.time()
        with self._transaction() as connection:
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
            seq_row = connection.execute(
                f"SELECT {column} FROM web_session WHERE session_id = ?", (session_id,)
            ).fetchone()
            seq = int(seq_row[0])
            value = f"{session_id}-{prefix}{seq}"
            try:
                connection.execute(
                    "INSERT INTO web_identity(session_id, kind, seq, value, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, kind, seq, value, timestamp),
                )
            except sqlite3.IntegrityError as exc:
                # The counter is rolled back with this transaction, so no gap is left
                # behind and the same identity can be retried once the conflict is gone.
                raise IdentityAllocationError(
                    f"identity {value!r} was already allocated"
                ) from exc
        return value

    def advance_reset_generation(self, session_id: str) -> SessionRecord:
        """Advance the reset generation without touching identities or counters.

        This is the identity-layer half of a Web Demo reset only: it records that the
        visible conversation was cleared and keeps every counter and ledger row so the
        next identity can never repeat.  Wiring it into ``/api/sessions/{id}/reset`` is a
        later slice.
        """
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
                "UPDATE web_session SET reset_generation = reset_generation + 1, "
                "updated_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            row = connection.execute(
                f"SELECT {_SESSION_COLUMNS} FROM web_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _to_record(row)

    # -- internals ---------------------------------------------------------- #

    def _ensure_open(self) -> None:
        if self._closed:
            raise StateStoreClosedError("web state store is closed")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one ``BEGIN IMMEDIATE`` unit of work, committing or rolling back.

        A caller exception (including the typed errors above) rolls back, so a failed
        allocation can never leave an incremented counter or a half-written ledger row.
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


def _to_record(row: sqlite3.Row) -> SessionRecord:
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
