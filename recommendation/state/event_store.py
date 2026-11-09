"""Append-only SQLite event store for the M5-MVP unified decision state.

The store is the durable boundary of the state engine: it persists validated
:class:`~recommendation.state.events.StateEvent` envelopes and assigns the deterministic
per-user ordering key (``sequence``) that replay depends on.  It uses the Python standard
library (``sqlite3``) only -- no framework and no new dependency, mirroring the accepted
M9 preference store.

Storage contract
----------------
Two tables:

* ``meta`` -- key/value metadata.  The ``event_store_schema_version`` row is written when
  the schema is created and verified on every open; a database written by a different
  store contract is refused with
  :class:`~recommendation.state.errors.EventStoreError` instead of being read with
  today's assumptions.
* ``events`` -- one row per appended event, with the canonical JSON payload and its
  SHA-256 hash, the event's scope (``long_term`` included, ``NULL`` otherwise) so a
  long-term event can be replayed for every task of its user, and
  ``UNIQUE(user_key, sequence)``.

``timestamp`` is metadata only.  The ordering key is ``sequence``, assigned inside the
same ``BEGIN IMMEDIATE`` transaction as the duplicate check and the insert:
``1 + MAX(sequence)`` for the event's ``user_key``.  Reads order by ``sequence`` ASC, so a
stream can never be reordered by clock skew or retry timing.

Idempotency and conflicts
-------------------------
The caller supplies ``event_id``; it is the global identity of a submission.  Appending
an id that is already stored is idempotent **only** when the stored event's semantic
fields match the submission (``timestamp``/``sequence`` are metadata, so a retry with a
fresh timestamp is still the same event): the stored event is returned unchanged and no
new row or sequence is created.  A same-id submission with different semantics --
including the same id submitted under another ``user_key`` -- raises
:class:`~recommendation.state.errors.EventConflictError` and writes nothing; the message
names the id, never the stored payload.

Concurrency
-----------
One connection with ``check_same_thread=False`` and a re-entrant lock guards every access,
matching the accepted M9 store.  Every access is serialised by the lock and each append is
one ``BEGIN IMMEDIATE`` transaction, so the connection is safe to share across threads.
SQLite's default per-thread affinity would otherwise make the store unusable from a
multi-threaded server that creates the store during startup and touches it from workers.

What this store does not do
---------------------------
It does not reduce state.  :meth:`SQLiteEventStore.replay` returns the ordered stream a
:class:`~recommendation.state.belief_updater.BeliefUpdater` folds; the fold -- and the
notion of a rejected transition -- belongs to the reducer/manager layer.  There is also no
table, column or method for the trusted chronological purchase history SASRec consumes:
this is a conversational decision store only.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from pydantic import ValidationError

from .errors import EventConflictError, EventStoreError, StateEngineError
from .events import EVENT_SCHEMA_VERSION, StateEvent
from .schemas import PreferenceScope, canonical_json

__all__ = [
    "EVENT_STORE_SCHEMA_VERSION",
    "STATE_DB_ENV_VAR",
    "EventStore",
    "SQLiteEventStore",
]

#: Version of the event-store contract (tables, columns, ordering semantics).  Stored in
#: the ``meta`` table when the schema is created and verified on every open, so a database
#: written for another contract is refused explicitly rather than misread.
EVENT_STORE_SCHEMA_VERSION = 1

#: Environment variable naming the decision-event database, so runtime composition code
#: can point a manager at a store without hard-coding a machine-specific path.
STATE_DB_ENV_VAR = "AGENTRECX_STATE_DB"

#: Key of the schema-version row in ``meta``.
_SCHEMA_VERSION_KEY = "event_store_schema_version"

#: DDL executed once when the store is opened; every statement is idempotent.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id       TEXT PRIMARY KEY,
        user_key       TEXT NOT NULL,
        session_id     TEXT,
        task_id        TEXT NOT NULL,
        event_type     TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        payload        TEXT NOT NULL,
        payload_hash   TEXT NOT NULL,
        actor          TEXT NOT NULL,
        source         TEXT NOT NULL,
        timestamp      REAL NOT NULL,
        sequence       INTEGER NOT NULL,
        scope          TEXT,
        UNIQUE (user_key, sequence)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_user_task_sequence
        ON events (user_key, task_id, sequence)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_user_scope_sequence
        ON events (user_key, scope, sequence)
    """,
)

#: Column list shared by every select and by the insert, kept in one place so the row
#: mapping cannot drift from the DDL.
_EVENT_COLUMNS = (
    "event_id, user_key, session_id, task_id, event_type, schema_version, payload, "
    "payload_hash, actor, source, timestamp, sequence, scope"
)

_INSERT_EVENT = (
    f"INSERT INTO events ({_EVENT_COLUMNS}) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_SELECT_EVENT_BY_ID = f"SELECT {_EVENT_COLUMNS} FROM events WHERE event_id = ?"

_SELECT_EVENT_BY_ID_FOR_USER = (
    f"SELECT {_EVENT_COLUMNS} FROM events WHERE event_id = ? AND user_key = ?"
)

_SELECT_EVENTS_FOR_USER = (
    f"SELECT {_EVENT_COLUMNS} FROM events WHERE user_key = ? ORDER BY sequence ASC"
)

_SELECT_EVENTS_FOR_TASK = (
    f"SELECT {_EVENT_COLUMNS} FROM events "
    "WHERE user_key = ? AND task_id = ? ORDER BY sequence ASC"
)

_SELECT_EVENTS_FOR_TASK_WITH_LONG_TERM = (
    f"SELECT {_EVENT_COLUMNS} FROM events "
    "WHERE user_key = ? AND (task_id = ? OR scope = ?) ORDER BY sequence ASC"
)

_SELECT_MAX_SEQUENCE = "SELECT MAX(sequence) AS max_sequence FROM events WHERE user_key = ?"

_SELECT_EVENT_COUNT = "SELECT COUNT(*) AS event_count FROM events WHERE user_key = ?"


def _require_identifier(value: object, *, field: str) -> str:
    """Return a stripped, non-blank identifier argument or raise ``ValueError``.

    Read entry points take caller-supplied identities that may not have gone through a
    validating model, so a blank value is refused here instead of silently matching
    nothing.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must not be blank")
    return stripped


@runtime_checkable
class EventStore(Protocol):
    """Structural read/append contract the state manager is written against.

    The manager depends on this protocol rather than on
    :class:`SQLiteEventStore`, so a test double or a future backend only has to provide
    these operations.  Implementations must be user-scoped (every read filters
    ``user_key``), deterministic in read order (by store-assigned sequence) and
    idempotent for duplicate ``event_id`` submissions with matching semantics.
    """

    @property
    def schema_version(self) -> int:
        """Version of the store contract recorded in the database."""
        ...

    def append(self, event: StateEvent) -> StateEvent:
        """Persist ``event`` and return the stored copy with its assigned sequence."""
        ...

    def get_events(
        self, task_id: str, user_key: str, *, include_long_term: bool = True
    ) -> tuple[StateEvent, ...]:
        """Return the task stream, including the user's long-term events."""
        ...

    def replay(self, task_id: str, user_key: str) -> tuple[StateEvent, ...]:
        """Return the ordered event stream to fold for one task."""
        ...

    def get_event(
        self, event_id: str, *, user_key: str | None = None
    ) -> StateEvent | None:
        """Return one stored event, optionally restricted to its owning user."""
        ...

    def get_all_events(self, user_key: str) -> tuple[StateEvent, ...]:
        """Return every event of ``user_key`` in sequence order."""
        ...

    def count(self, user_key: str) -> int:
        """Number of stored events owned by ``user_key``."""
        ...

    def close(self) -> None:
        """Release the database connection."""
        ...


class SQLiteEventStore:
    """SQLite-backed append-only event store (standard library only).

    See the module docstring for the schema, ordering, idempotency and concurrency
    guarantees.  The instance is a context manager and is safe to share across threads:
    every access is serialised by an ``RLock`` and every append is one
    ``BEGIN IMMEDIATE`` transaction.
    """

    def __init__(self, database: str | Path) -> None:
        self._database = str(database)
        self._lock = threading.RLock()
        # ``check_same_thread=False`` is required, not cosmetic: every access below is
        # serialised by ``self._lock`` and multi-statement writes run inside one
        # ``BEGIN IMMEDIATE`` transaction, so the connection is safe to share.  SQLite's
        # default same-thread check would otherwise break any server that creates the
        # store on one thread and serves requests on others (the accepted M9 store hit
        # exactly this failure).
        self._connection = sqlite3.connect(
            self._database, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            # Durability/consistency settings that are safe for a single-writer local
            # store.  WAL is set before any transaction; it is a no-op for ``:memory:``.
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._ensure_schema()
        except BaseException:
            self._connection.close()
            raise

    # -- lifecycle ---------------------------------------------------------- #

    @property
    def database(self) -> str:
        """Configured database path (``:memory:`` for an ephemeral store)."""
        return self._database

    @property
    def schema_version(self) -> int:
        """Stored event-store schema version."""
        with self._lock:
            version = self._read_schema_version()
        if version is None:
            raise EventStoreError(
                f"event store {self._database!r} records no schema version"
            )
        return version

    def close(self) -> None:
        """Close the database connection.  The store is unusable afterwards."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteEventStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        """Create the schema and record/verify its version in one transaction."""
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for statement in _SCHEMA_STATEMENTS:
                    self._connection.execute(statement)
                version = self._read_schema_version()
                if version is None:
                    self._connection.execute(
                        "INSERT INTO meta (key, value) VALUES (?, ?)",
                        (_SCHEMA_VERSION_KEY, str(EVENT_STORE_SCHEMA_VERSION)),
                    )
                elif version != EVENT_STORE_SCHEMA_VERSION:
                    raise EventStoreError(
                        f"event store {self._database!r} has schema version {version}, "
                        f"expected {EVENT_STORE_SCHEMA_VERSION}"
                    )
                self._connection.execute("COMMIT")
            except EventStoreError:
                self._rollback_quietly()
                raise
            except sqlite3.Error as exc:
                self._rollback_quietly()
                raise EventStoreError(
                    f"could not initialise event store {self._database!r}: {exc}"
                ) from exc

    def _read_schema_version(self) -> int | None:
        """Read the stored schema version row, if the database has one."""
        row = self._fetchone(
            "SELECT value FROM meta WHERE key = ?", (_SCHEMA_VERSION_KEY,)
        )
        if row is None:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError) as exc:
            raise EventStoreError(
                f"event store {self._database!r} records a non-integer schema version"
            ) from exc

    def _rollback_quietly(self) -> None:
        """Roll back the current transaction, ignoring "no transaction" errors."""
        with contextlib.suppress(sqlite3.Error):
            self._connection.execute("ROLLBACK")

    # -- transaction helpers ------------------------------------------------ #

    @contextlib.contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        """One ``BEGIN IMMEDIATE`` unit of work: commit on success, rollback on error."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._rollback_quietly()
            raise
        else:
            try:
                self._connection.execute("COMMIT")
            except sqlite3.Error:
                self._rollback_quietly()
                raise

    def _fetchone(
        self, query: str, params: tuple[object, ...] = ()
    ) -> sqlite3.Row | None:
        """Run one read statement under the lock, converting SQL errors to typed ones."""
        with self._lock:
            try:
                return self._connection.execute(query, params).fetchone()
            except sqlite3.Error as exc:
                raise EventStoreError(f"event store read failed: {exc}") from exc

    def _fetchall(
        self, query: str, params: tuple[object, ...] = ()
    ) -> list[sqlite3.Row]:
        """Run one read statement under the lock, converting SQL errors to typed ones."""
        with self._lock:
            try:
                return self._connection.execute(query, params).fetchall()
            except sqlite3.Error as exc:
                raise EventStoreError(f"event store read failed: {exc}") from exc

    # -- row mapping -------------------------------------------------------- #

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StateEvent:
        """Reconstruct a validated :class:`StateEvent` from one database row.

        An unsupported stored envelope version, unreadable payload JSON or a payload that
        no longer satisfies the frozen contract is a typed store error, never a partially
        trusted event.
        """
        event_id = row["event_id"]
        stored_version = row["schema_version"]
        if stored_version != EVENT_SCHEMA_VERSION:
            raise EventStoreError(
                f"stored event {event_id!r} has event schema version {stored_version}, "
                f"expected {EVENT_SCHEMA_VERSION}"
            )
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError) as exc:
            raise EventStoreError(
                f"stored event {event_id!r} has an unreadable payload"
            ) from exc
        try:
            return StateEvent.model_validate(
                {
                    "event_id": event_id,
                    "user_key": row["user_key"],
                    "session_id": row["session_id"],
                    "task_id": row["task_id"],
                    "event_type": row["event_type"],
                    "schema_version": stored_version,
                    "payload": payload,
                    "actor": row["actor"],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "sequence": row["sequence"],
                }
            )
        except ValidationError as exc:
            raise EventStoreError(
                f"stored event {event_id!r} does not satisfy the current event contract"
            ) from exc

    # -- writes ------------------------------------------------------------- #

    def append(self, event: StateEvent) -> StateEvent:
        """Persist ``event`` and return the stored copy carrying its assigned sequence.

        The duplicate check, the per-user ``MAX(sequence) + 1`` assignment and the insert
        run in one ``BEGIN IMMEDIATE`` transaction, so concurrent writers cannot
        interleave and a failure leaves no partial row or sequence gap.  A duplicate id
        with identical semantic content returns the stored event unchanged (idempotent);
        a duplicate id with different semantics raises ``EventConflictError``.
        """
        if not isinstance(event, StateEvent):
            raise TypeError("append requires a StateEvent instance")
        with self._lock:
            try:
                with self._immediate_transaction():
                    existing = self._connection.execute(
                        _SELECT_EVENT_BY_ID, (event.event_id,)
                    ).fetchone()
                    if existing is not None:
                        stored = self._row_to_event(existing)
                        if stored.is_same_submission(event):
                            return stored
                        raise EventConflictError(
                            f"event id {event.event_id!r} is already stored with "
                            "different semantics"
                        )
                    row = self._connection.execute(
                        _SELECT_MAX_SEQUENCE, (event.user_key,)
                    ).fetchone()
                    highest = row["max_sequence"] if row is not None else None
                    sequence = int(highest) + 1 if highest is not None else 1
                    stored_event = event.stored(sequence)
                    scope = stored_event.scope
                    self._connection.execute(
                        _INSERT_EVENT,
                        (
                            stored_event.event_id,
                            stored_event.user_key,
                            stored_event.session_id,
                            stored_event.task_id,
                            stored_event.event_type.value,
                            stored_event.schema_version,
                            canonical_json(stored_event.canonical_payload()),
                            stored_event.payload_hash(),
                            stored_event.actor.value,
                            stored_event.source,
                            stored_event.timestamp,
                            sequence,
                            scope.value if scope is not None else None,
                        ),
                    )
                    return stored_event
            except StateEngineError:
                raise
            except sqlite3.Error as exc:
                raise EventStoreError(
                    f"could not append event {event.event_id!r}: {exc}"
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive typed boundary
                raise EventStoreError(
                    f"could not append event {event.event_id!r}"
                ) from exc

    # -- reads -------------------------------------------------------------- #

    def get_events(
        self, task_id: str, user_key: str, *, include_long_term: bool = True
    ) -> tuple[StateEvent, ...]:
        """Return ``user_key``'s stream for ``task_id`` in sequence order.

        The stream is the task's own events plus, when ``include_long_term`` is true,
        every event of the same user whose payload scope is ``long_term`` -- including
        long-term events emitted in another task.  Task-scoped events from other tasks
        are never included, and no other user's events can match the query.
        """
        resolved_task = _require_identifier(task_id, field="task_id")
        resolved_user = _require_identifier(user_key, field="user_key")
        if include_long_term:
            rows = self._fetchall(
                _SELECT_EVENTS_FOR_TASK_WITH_LONG_TERM,
                (resolved_user, resolved_task, PreferenceScope.LONG_TERM.value),
            )
        else:
            rows = self._fetchall(
                _SELECT_EVENTS_FOR_TASK, (resolved_user, resolved_task)
            )
        return tuple(self._row_to_event(row) for row in rows)

    def replay(self, task_id: str, user_key: str) -> tuple[StateEvent, ...]:
        """Return the ordered stream a reducer folds for one task.

        Identical to :meth:`get_events` with its default arguments; named separately so
        callers state their intent.  The store does not reduce state: replaying means
        folding this sequence through the belief updater.
        """
        return self.get_events(task_id, user_key)

    def get_event(
        self, event_id: str, *, user_key: str | None = None
    ) -> StateEvent | None:
        """Return one stored event by id, or ``None``.

        With ``user_key`` supplied the lookup is user-scoped, so an event id belonging to
        another user is reported as not found rather than disclosed.
        """
        resolved_event_id = _require_identifier(event_id, field="event_id")
        if user_key is None:
            row = self._fetchone(_SELECT_EVENT_BY_ID, (resolved_event_id,))
        else:
            resolved_user = _require_identifier(user_key, field="user_key")
            row = self._fetchone(
                _SELECT_EVENT_BY_ID_FOR_USER, (resolved_event_id, resolved_user)
            )
        return None if row is None else self._row_to_event(row)

    def get_all_events(self, user_key: str) -> tuple[StateEvent, ...]:
        """Return every event of ``user_key`` across tasks, in sequence order."""
        resolved_user = _require_identifier(user_key, field="user_key")
        rows = self._fetchall(_SELECT_EVENTS_FOR_USER, (resolved_user,))
        return tuple(self._row_to_event(row) for row in rows)

    def count(self, user_key: str) -> int:
        """Number of stored events owned by ``user_key`` (0 for an unknown user)."""
        resolved_user = _require_identifier(user_key, field="user_key")
        row = self._fetchone(_SELECT_EVENT_COUNT, (resolved_user,))
        return int(row["event_count"]) if row is not None else 0
