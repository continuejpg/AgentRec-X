"""Persistent ``SessionStore`` adapter for the Web Demo BFF (M1-C S3a).

``SessionStore`` in :mod:`web_demo.sessions` is a process-local registry: it is the
right default for the existing tests, but a BFF restart loses every session and reissues
turn/reply identities that M9 has already recorded.  This adapter keeps the same public
surface (``create`` / ``get`` / ``list``) while delegating the authoritative state to
:class:`web_demo.state_store.SQLiteWebStateStore`.

Scope of S3a
------------
Only session identity, counters and the supported metadata (title, ``behavior_enabled``)
are durable.  Chat turns, recommendation snapshots, pending clarifications and the MCP
executor remain in memory: a session loaded in a new process is a fresh *view* with the
right identity, not a restored conversation.  A persisted OPEN pending therefore cannot
be answered through the current BFF yet; that is S3b work.

Why a live cache
----------------
The BFF resolves the session on every request (``get_session`` in ``app.py``).  If this
adapter rebuilt a ``Session`` from SQLite each time, the pending clarification created by
the ASK would disappear before the user could reply.  Loaded sessions are therefore
cached per process; metadata and identities that *are* durable still go to SQLite on
every write, and a new process starts with a fresh cache built from the database.

The session id is opaque.  It is 64 bits of ``secrets`` entropy rather than the previous
32, because the id is now a capability that survives restarts; nothing parses its length,
and ids created by earlier versions are read back unchanged.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path

from web_demo.sessions import Session
from web_demo.state_store import SessionRecord, SQLiteWebStateStore

__all__ = [
    "PERSISTENT_SESSION_ID_BYTES",
    "PERSISTENT_SESSION_ID_PREFIX",
    "PersistentSessionStore",
    "iso_from_epoch",
]

#: Prefix kept identical to the in-memory store so the id stays an opaque ``web-…`` token.
PERSISTENT_SESSION_ID_PREFIX = "web-"
#: 8 bytes = 64 bits of entropy (the in-memory store still uses 4 bytes).
PERSISTENT_SESSION_ID_BYTES = 8


def iso_from_epoch(timestamp: float) -> str:
    """Convert a SQLite epoch timestamp to the ISO-8601 ``Z`` form ``Session`` uses."""
    return (
        datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


class PersistentSessionStore:
    """Duck-typed replacement for :class:`web_demo.sessions.SessionStore`.

    The three public methods match the in-memory store, so ``create_app`` and the HTTP
    routes need no branching: passing this object as ``sessions=`` (or configuring
    ``Settings.web_state_db_path``) switches the BFF to durable identity.
    """

    def __init__(self, database: str | Path) -> None:
        self._state = SQLiteWebStateStore(database)
        self._live: dict[str, Session] = {}

    # -- introspection ------------------------------------------------------ #

    @property
    def state_store(self) -> SQLiteWebStateStore:
        """The underlying SQLite store (inspection/tests; not exposed over HTTP)."""
        return self._state

    @property
    def database(self) -> str:
        return self._state.path

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        """Drop the live cache and close the SQLite connection.  Idempotent enough."""
        self._live.clear()
        self._state.close()

    # -- SessionStore surface ----------------------------------------------- #

    def create(self, profile_id: str, *, behavior_enabled: bool = True) -> Session:
        session_id = self._new_session_id()
        record = self._state.create_session(
            session_id=session_id,
            profile_id=profile_id,
            behavior_enabled=behavior_enabled,
        )
        return self._remember(record)

    def get(self, session_id: str) -> Session | None:
        cached = self._live.get(session_id)
        if cached is not None:
            return cached
        record = self._state.get_session(session_id)
        if record is None:
            return None
        return self._remember(record)

    def list(self, limit: int = 20) -> list[Session]:
        return [self._remember(record) for record in self._state.list_sessions(limit)]

    # -- internals ---------------------------------------------------------- #

    def _new_session_id(self) -> str:
        return PERSISTENT_SESSION_ID_PREFIX + secrets.token_hex(PERSISTENT_SESSION_ID_BYTES)

    def _remember(self, record: SessionRecord) -> Session:
        session = self._live.get(record.session_id)
        if session is None:
            session = self._to_session(record)
            self._live[record.session_id] = session
        return session

    def _to_session(self, record: SessionRecord) -> Session:
        return Session(
            session_id=record.session_id,
            profile_id=record.profile_id,
            title=record.title,
            created_at=iso_from_epoch(record.created_at),
            updated_at=iso_from_epoch(record.updated_at),
            behavior_enabled=record.behavior_enabled,
            # Snapshots only: with ``_store`` attached, allocation always goes to SQLite.
            _turn_counter=record.turn_counter,
            _reply_counter=record.reply_counter,
            _store=self._state,
        )
