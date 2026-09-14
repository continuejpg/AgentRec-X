"""Preference-memory stores (Milestone 9).

Two interchangeable backends behind one narrow contract:

* :class:`InMemoryPreferenceStore` -- for unit tests and ephemeral runtimes;
* :class:`SQLitePreferenceStore` -- stdlib :mod:`sqlite3` persistence, so memory
  survives process restarts without adding a database dependency.

What these stores do **not** hold
---------------------------------
Only conversational explicit preferences.  There is deliberately no method that
accepts an interaction event, an ``item_id`` or an Amazon ``parent_asin``.  Trusted
interaction history stays in application/graph input state and reaches SASRec through
the accepted :class:`~recommendation.tools.RecommendationTool` path; it is never
persisted here.  A behavioural event therefore has no route into this database.

Determinism
-----------
Reads return entries ordered by ``(logical_seq, memory_id)``, which is stable and
independent of dict or row iteration order.  ``logical_seq`` is assigned by the
service, not by the store, so ordering never depends on wall-clock or hash order.
Stores never mutate a stored record in place: supersession and removal are recorded
by writing an updated row, and the audit trail is preserved.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from .schemas import (
    MEMORY_SCHEMA_VERSION,
    PreferenceMemoryEntry,
    PreferenceStatus,
)

__all__ = [
    "MEMORY_DB_ENV_VAR",
    "PreferenceStore",
    "InMemoryPreferenceStore",
    "SQLitePreferenceStore",
    "deterministic_memory_id",
]

#: Environment variable used to point the runtime at a memory database, so no
#: machine-specific absolute path is hard-coded anywhere in the code.
MEMORY_DB_ENV_VAR = "AGENTRECX_MEMORY_DB"


def deterministic_memory_id(
    user_key: str, source_turn_id: str, kind: str, value: str, polarity: str
) -> str:
    """Return a stable memory id derived from the entry's identity.

    Identity is the *origin* of the preference -- user, source turn, kind, value and
    polarity -- not the wall-clock time or an auto-increment counter.  Two runs that
    process the same turn therefore produce the same id, which is what makes
    re-processing a turn idempotent at the identity level rather than only by
    comparing text.
    """
    payload = "\x1f".join(
        (user_key, source_turn_id, kind, value, polarity)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


@runtime_checkable
class PreferenceStore(Protocol):
    """Storage contract for preference memory.

    Implementations must be deterministic, must never mutate a stored entry in place,
    and must never mix users.
    """

    def add_entry(self, entry: PreferenceMemoryEntry) -> PreferenceMemoryEntry:
        """Persist a new entry; raise ``ValueError`` if its id already exists."""
        ...

    def get_entry(self, user_key: str, memory_id: str) -> PreferenceMemoryEntry | None:
        """Return one entry belonging to ``user_key``, or ``None``."""
        ...

    def get_entries(self, user_key: str, *, active_only: bool = False) -> tuple[PreferenceMemoryEntry, ...]:
        """Return ``user_key``'s entries in deterministic order."""
        ...

    def update_entry(self, entry: PreferenceMemoryEntry) -> PreferenceMemoryEntry:
        """Replace an existing entry (same id, same user) with a new version."""
        ...

    def max_logical_seq(self, user_key: str) -> int:
        """Highest ``logical_seq`` used for ``user_key`` (0 when none)."""
        ...

    def count(self, user_key: str, *, active_only: bool = False) -> int:
        """Number of entries for ``user_key``."""
        ...


def _sort_key(entry: PreferenceMemoryEntry) -> tuple[int, str]:
    """Deterministic ordering key: logical sequence, then memory id."""
    return (entry.logical_seq, entry.memory_id)


class InMemoryPreferenceStore:
    """Process-local preference store, used by unit tests and ephemeral runtimes."""

    def __init__(self) -> None:
        #: ``user_key -> memory_id -> entry``.  No process-global state: each instance
        #: is independent, and instances are injected rather than imported.
        self._entries: dict[str, dict[str, PreferenceMemoryEntry]] = {}

    def add_entry(self, entry: PreferenceMemoryEntry) -> PreferenceMemoryEntry:
        """Persist a new entry."""
        table = self._entries.setdefault(entry.user_key, {})
        if entry.memory_id in table:
            raise ValueError(
                f"memory id {entry.memory_id!r} already exists for user {entry.user_key!r}"
            )
        table[entry.memory_id] = entry
        return entry

    def get_entry(self, user_key: str, memory_id: str) -> PreferenceMemoryEntry | None:
        """Return one entry belonging to ``user_key``, or ``None``."""
        return self._entries.get(user_key, {}).get(memory_id)

    def get_entries(
        self, user_key: str, *, active_only: bool = False
    ) -> tuple[PreferenceMemoryEntry, ...]:
        """Return ``user_key``'s entries in deterministic order."""
        entries = list(self._entries.get(user_key, {}).values())
        if active_only:
            entries = [entry for entry in entries if entry.status is PreferenceStatus.ACTIVE]
        return tuple(sorted(entries, key=_sort_key))

    def update_entry(self, entry: PreferenceMemoryEntry) -> PreferenceMemoryEntry:
        """Replace an existing entry, preserving its id and user."""
        table = self._entries.get(entry.user_key)
        if table is None or entry.memory_id not in table:
            raise KeyError(
                f"memory id {entry.memory_id!r} does not exist for user {entry.user_key!r}"
            )
        table[entry.memory_id] = entry
        return entry

    def max_logical_seq(self, user_key: str) -> int:
        """Highest ``logical_seq`` used for ``user_key`` (0 when none)."""
        entries = self._entries.get(user_key, {})
        return max((entry.logical_seq for entry in entries.values()), default=0)

    def count(self, user_key: str, *, active_only: bool = False) -> int:
        """Number of entries for ``user_key``."""
        return len(self.get_entries(user_key, active_only=active_only))

    def user_keys(self) -> tuple[str, ...]:
        """Return every user key known to this store (diagnostics/tests only)."""
        return tuple(sorted(self._entries))


#: SQLite schema.  Kept as one statement string so the version and DDL live together.
_SQLITE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preference_memory (
    memory_id       TEXT NOT NULL,
    user_key        TEXT NOT NULL,
    kind            TEXT NOT NULL,
    value           TEXT NOT NULL,
    polarity        TEXT NOT NULL,
    source_text     TEXT NOT NULL,
    source_turn_id  TEXT NOT NULL,
    extractor       TEXT NOT NULL,
    status          TEXT NOT NULL,
    logical_seq     INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    supersedes      TEXT,
    superseded_by   TEXT,
    PRIMARY KEY (user_key, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_pref_user_status_seq
    ON preference_memory (user_key, status, logical_seq);

CREATE INDEX IF NOT EXISTS idx_pref_user_turn
    ON preference_memory (user_key, source_turn_id);
"""

_COLUMNS = (
    "memory_id, user_key, kind, value, polarity, source_text, source_turn_id, "
    "extractor, status, logical_seq, created_at, supersedes, superseded_by"
)

#: Assignment list mirroring ``_COLUMNS`` order, excluding the key columns.
_COLUMNS_ASSIGNMENTS = (
    "kind = ?, value = ?, polarity = ?, source_text = ?, source_turn_id = ?, "
    "extractor = ?, status = ?, logical_seq = ?, created_at = ?, "
    "supersedes = ?, superseded_by = ?"
)

#: INSERT statement for one preference entry.
_INSERT = (
    "INSERT INTO preference_memory (" + _COLUMNS + ") "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class SQLitePreferenceStore:
    """SQLite-backed preference store using only the Python standard library.

    Schema
    ------
    Two tables: ``meta`` (holding ``schema_version``) and ``preference_memory`` with
    primary key ``(user_key, memory_id)``, so a memory id can never be reused across
    users and rows can never leak between them.  The schema version is written on
    creation and verified on open, so a future migration is explicit rather than
    silent.

    Transactions
    ------------
    The connection runs in SQLite's default transaction mode and every mutating call
    commits explicitly after a single statement, so a partial multi-row update cannot
    be observed.  Higher-level multi-step updates (supersede + add) are performed by
    the service inside one ``transaction()`` block.
    """

    def __init__(self, database: str | Path) -> None:
        self._database = str(database)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self._database, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        # Durability/consistency settings that are safe for a single-writer local store.
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA foreign_keys = ON")
        #: True while an outer `_atomic_write` block owns a transaction.
        self._in_transaction = False
        #: Set by an inner block that failed so the outer block rolls back.
        self._rollback_only = False
        self._ensure_schema()

    # -- lifecycle --------------------------------------------------------- #

    @property
    def database(self) -> str:
        """Configured database path (``:memory:`` for an ephemeral store)."""
        return self._database

    def _ensure_schema(self) -> None:
        """Create the schema and record/verify its version."""
        with self._lock:
            self._connection.executescript(_SQLITE_SCHEMA)
            row = self._connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(MEMORY_SCHEMA_VERSION),),
                )
            elif int(row["value"]) != MEMORY_SCHEMA_VERSION:
                raise RuntimeError(
                    f"memory database {self._database!r} has schema version "
                    f"{row['value']}, expected {MEMORY_SCHEMA_VERSION}"
                )

    @property
    def schema_version(self) -> int:
        """Schema version recorded in the database."""
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row["value"])

    def close(self) -> None:
        """Close the connection.  The store is unusable afterwards."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLitePreferenceStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def transaction(self) -> _Transaction:
        """Context manager grouping several writes into one transaction.

        SQLite refuses to nest transactions, so this must not be used inside
        :meth:`_atomic_write`; the memory service performs its multi-step updates
        through that reentrant helper instead.
        """
        return _Transaction(self._connection, self._lock)

    @contextmanager
    def _atomic_write(self) -> Iterator[None]:
        """Run a multi-statement update atomically (``BEGIN IMMEDIATE``).

        A check-then-insert or supersede-then-add sequence must not be observable half
        applied, so it is wrapped in one immediate transaction: rollback on any error,
        commit on success.

        Writes nest.  An inner block runs inside the outer transaction and, on failure,
        signals an active transaction so the outer block rolls back: SQLite has no
        savepoints here, and the fallback path is required for correctness anyway
        because the service must not crash after a ``supersede`` has already succeeded.
        """
        with self._lock:
            if self._in_transaction:
                try:
                    yield
                except BaseException:
                    self._rollback_only = True
                    raise
                return
            self._connection.execute("BEGIN IMMEDIATE")
            self._in_transaction = True
            self._rollback_only = False
            try:
                yield
            except BaseException:
                self._in_transaction = False
                self._connection.execute("ROLLBACK")
                raise
            self._in_transaction = False
            if self._rollback_only:
                # An inner block failed but was caught by the caller; the whole unit of
                # work is discarded rather than half-applied.
                self._rollback_only = False
                self._connection.execute("ROLLBACK")
            else:
                self._connection.execute("COMMIT")

    @contextmanager
    def _atomic_read(self) -> Iterator[None]:
        """Run a multi-statement read against a single consistent snapshot.

        The transaction is always rolled back (nothing was written), which keeps the
        connection out of a lingering open transaction.
        """
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                yield
            finally:
                self._connection.execute("ROLLBACK")

    # -- reads -------------------------------------------------------------- #

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> PreferenceMemoryEntry:
        """Convert a database row into a validated entry."""
        return PreferenceMemoryEntry(
            memory_id=row["memory_id"],
            user_key=row["user_key"],
            kind=row["kind"],
            value=row["value"],
            polarity=row["polarity"],
            source_text=row["source_text"],
            source_turn_id=row["source_turn_id"],
            extractor=row["extractor"],
            status=row["status"],
            logical_seq=row["logical_seq"],
            created_at=row["created_at"],
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
        )

    def get_entry(self, user_key: str, memory_id: str) -> PreferenceMemoryEntry | None:
        """Return one entry belonging to ``user_key``, or ``None``."""
        with self._lock:
            row = self._connection.execute(
                f"SELECT {_COLUMNS} FROM preference_memory "
                "WHERE user_key = ? AND memory_id = ?",
                (user_key, memory_id),
            ).fetchone()
        return None if row is None else self._row_to_entry(row)

    def get_entries(
        self, user_key: str, *, active_only: bool = False
    ) -> tuple[PreferenceMemoryEntry, ...]:
        """Return ``user_key``'s entries in deterministic order."""
        query = f"SELECT {_COLUMNS} FROM preference_memory WHERE user_key = ?"
        params: list[object] = [user_key]
        if active_only:
            query += " AND status = ?"
            params.append(PreferenceStatus.ACTIVE.value)
        query += " ORDER BY logical_seq ASC, memory_id ASC"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(self._row_to_entry(row) for row in rows)

    def get_entries_for_turn(
        self, user_key: str, source_turn_id: str
    ) -> tuple[PreferenceMemoryEntry, ...]:
        """Return every entry recorded for one source turn (idempotency checks)."""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT {_COLUMNS} FROM preference_memory "
                "WHERE user_key = ? AND source_turn_id = ? "
                "ORDER BY logical_seq ASC, memory_id ASC",
                (user_key, source_turn_id),
            ).fetchall()
        return tuple(self._row_to_entry(row) for row in rows)

    def max_logical_seq(self, user_key: str) -> int:
        """Highest ``logical_seq`` used for ``user_key`` (0 when none)."""
        with self._lock:
            row = self._connection.execute(
                "SELECT MAX(logical_seq) AS seq FROM preference_memory WHERE user_key = ?",
                (user_key,),
            ).fetchone()
        return int(row["seq"]) if row is not None and row["seq"] is not None else 0

    def count(self, user_key: str, *, active_only: bool = False) -> int:
        """Number of entries for ``user_key``."""
        query = "SELECT COUNT(*) AS n FROM preference_memory WHERE user_key = ?"
        params: list[object] = [user_key]
        if active_only:
            query += " AND status = ?"
            params.append(PreferenceStatus.ACTIVE.value)
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        return int(row["n"])

    # -- writes ------------------------------------------------------------- #

    def add_entry(self, entry: PreferenceMemoryEntry) -> PreferenceMemoryEntry:
        """Persist a new entry; reject a duplicate ``(user_key, memory_id)``.

        The duplicate check and the insert share one immediate transaction, so the
        check-then-insert sequence cannot interleave with another writer.
        """
        with self._atomic_write():
            existing = self.get_entry(entry.user_key, entry.memory_id)
            if existing is not None:
                raise ValueError(
                    f"memory id {entry.memory_id!r} already exists for user "
                    f"{entry.user_key!r}"
                )
            self._connection.execute(
                _INSERT,
                (
                    entry.memory_id,
                    entry.user_key,
                    entry.kind.value,
                    entry.value,
                    entry.polarity.value,
                    entry.source_text,
                    entry.source_turn_id,
                    entry.extractor,
                    entry.status.value,
                    entry.logical_seq,
                    entry.created_at,
                    entry.supersedes,
                    entry.superseded_by,
                ),
            )
        return entry

    def update_entry(self, entry: PreferenceMemoryEntry) -> PreferenceMemoryEntry:
        """Replace an existing entry with a new version, keeping its id and user."""
        with self._lock:
            existing = self.get_entry(entry.user_key, entry.memory_id)
            if existing is None:
                raise KeyError(
                    f"memory id {entry.memory_id!r} does not exist for user "
                    f"{entry.user_key!r}"
                )
            self._connection.execute(
                f"UPDATE preference_memory SET {_COLUMNS_ASSIGNMENTS} "
                "WHERE user_key = ? AND memory_id = ?",
                (
                    entry.kind.value,
                    entry.value,
                    entry.polarity.value,
                    entry.source_text,
                    entry.source_turn_id,
                    entry.extractor,
                    entry.status.value,
                    entry.logical_seq,
                    entry.created_at,
                    entry.supersedes,
                    entry.superseded_by,
                    entry.user_key,
                    entry.memory_id,
                ),
            )
        return entry


class _Transaction:
    """Explicit transaction scope for multi-statement memory updates."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self._connection = connection
        self._lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        self._connection.execute("BEGIN")
        return self._connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        try:
            if exc_type is None:
                self._connection.execute("COMMIT")
            else:
                self._connection.execute("ROLLBACK")
        finally:
            self._lock.release()
        return False
