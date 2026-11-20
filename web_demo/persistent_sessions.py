"""Persistent ``SessionStore`` adapter for the Web Demo BFF (M1-C S3a + S3b).

``SessionStore`` in :mod:`web_demo.sessions` is a process-local registry.  This adapter
keeps the same public surface (``create`` / ``get`` / ``list``) while delegating the
authoritative state to :class:`web_demo.state_store.SQLiteWebStateStore`.

S3a made session identity, counters and metadata durable.  S3b adds the parts needed to
recover a *view*: persisted chat turns, the active task, and the current OPEN pending
clarification.  A restored pending is display-only: the MCP handle that created it died
with the old process, so replies are blocked until S5 adds request idempotency.

The loaded ``Session`` is cached per process because the BFF resolves it on every
request and keeps in-request state (tool activity, pending) in it.  SQLite remains the
authority: identity allocation, metadata, ASK commits and resets always go to the
database, and a cache entry is dropped when this process resets the session.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from web_demo.sessions import PendingClarification, Session, Turn
from web_demo.state_store import (
    SessionRecord,
    SQLiteWebStateStore,
    StaleTaskVersion,
    TaskRecord,
    TaskStateConflict,
)

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
    """Duck-typed replacement for :class:`web_demo.sessions.SessionStore`."""

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
        self._live.clear()
        self._state.close()

    # -- identity / metadata proxies (``Session._store`` points here) ------- #

    def allocate_turn_id(self, session_id: str) -> str:
        return self._state.allocate_turn_id(session_id)

    def allocate_reply_id(self, session_id: str) -> str:
        return self._state.allocate_reply_id(session_id)

    def update_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        behavior_enabled: bool | None = None,
    ) -> SessionRecord:
        return self._state.update_session_metadata(
            session_id, title=title, behavior_enabled=behavior_enabled
        )

    def reset_session(self, session_id: str, *, title: str | None = None) -> SessionRecord:
        record = self._state.reset_session(session_id, title=title)
        # The cached object may still show the pre-reset ASK; drop it so the next read
        # hydrates the persisted (cleared) state instead of an expired local view.
        self._live.pop(session_id, None)
        return record

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

    # -- chat task lifecycle (persistent mode only) ------------------------- #

    def begin_chat_task(self, session: Session, message: str) -> TaskRecord:
        """Return the task this chat belongs to, creating or superseding as needed.

        Minimal deterministic rule (documented limitation: one active task per session):

        * no active task -> create one for this turn;
        * an active ``RUNNING`` task -> continue it (multi-round conversation);
        * an active ``WAITING_FOR_USER`` task -> the user typed a new request instead of
          answering, so the old goal is cancelled (its OPEN pending is cancelled too)
          and a fresh task is created.  Requirements are never overwritten in place.
        """
        active = self._state.get_active_task(session.session_id)
        if active is not None and active.status == "WAITING_FOR_USER":
            try:
                self._state.cancel_task(
                    session.session_id,
                    active.task_id,
                    expected_task_version=active.task_version,
                    reason="superseded_by_new_chat",
                )
            except (TaskStateConflict, StaleTaskVersion):
                # Another instance moved it first; fall through to a fresh task.
                pass
            active = None
        if active is None:
            active = self._state.create_task(
                session.session_id, initial_requirements={"raw_query": message}
            )
        return active

    def persist_ask(
        self,
        session: Session,
        task: TaskRecord,
        turn: Turn,
        clarification: Mapping[str, Any],
        *,
        query: str,
        clarification_id: str,
        mode: str,
        expires_at: float,
        ranking_source: str | None,
    ) -> PendingClarification:
        """Commit Task + Turn + Pending + Reply ID in one SQLite transaction.

        The reply identity is allocated *inside* that transaction (``commit_ask``), so the
        caller must not allocate one first.  Nothing is yielded to the browser until this
        returns.
        """
        record = self._state.commit_ask(
            session.session_id,
            task.task_id,
            expected_task_version=task.task_version,
            clarification_id=clarification_id,
            source_turn_id=turn.turn_id,
            query=query,
            payload=clarification,
            expires_at=expires_at,
            mode=mode,
            turn=turn.view(),
            ranking_source=ranking_source,
        )
        pending = PendingClarification(
            payload=record.payload,
            query=record.query,
            turn_id=record.turn_id,
            reply_turn_id=record.reply_turn_id,
            mode=record.mode,
            created_at=iso_from_epoch(record.created_at),
        )
        session.pending_clarification = pending
        return pending

    def persist_finished_turn(
        self, session: Session, task: TaskRecord | None, turn: Turn, *, stop_reason: str
    ) -> None:
        """Persist a chat that did not end in an ASK, and close its task.

        An ASK turn was already written by :meth:`persist_ask` and its task must stay
        ``WAITING_FOR_USER``, so that case is a no-op here.
        """
        if stop_reason == "awaiting_clarification":
            return
        try:
            self._state.append_turn(
                session.session_id,
                turn=turn.view(),
                task_id=None if task is None else task.task_id,
                reset_generation=None if task is None else task.reset_generation,
            )
        except TaskStateConflict:
            # A reset happened while this turn was in flight: the late result must not
            # resurrect history the user has already cleared.
            return
        if task is None:
            return
        try:
            if stop_reason == "error":
                self._state.fail_task(
                    session.session_id,
                    task.task_id,
                    expected_task_version=task.task_version,
                    reason="chat_failed",
                )
            else:
                self._state.complete_task(
                    session.session_id,
                    task.task_id,
                    expected_task_version=task.task_version,
                )
        except (TaskStateConflict, StaleTaskVersion):
            # A concurrent instance already closed the task; that terminal state wins.
            pass

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
        session_id = record.session_id
        # Expire overdue pendings first: a WAITING task must not survive without a usable
        # question, and the absolute TTL must never be recomputed.
        self._state.expire_overdue_pending()

        stored_turns = self._state.list_turns(session_id)
        turns = [Turn(**turn_record.payload["turn"]) for turn_record in stored_turns]

        session = Session(
            session_id=session_id,
            profile_id=record.profile_id,
            title=record.title,
            created_at=iso_from_epoch(record.created_at),
            updated_at=iso_from_epoch(record.updated_at),
            turns=turns,
            behavior_enabled=record.behavior_enabled,
            last_query=turns[-1].user_text if turns else "",
            # Snapshots only: with ``_store`` attached, allocation always goes to SQLite.
            _turn_counter=record.turn_counter,
            _reply_counter=record.reply_counter,
            _store=self,
        )

        # Rebuild behaviour contexts from the stored turn content; never re-run the
        # recommender to recreate history.
        for stored, turn in zip(stored_turns, turns):
            products = turn.products
            if not isinstance(products, Mapping) or not products.get("recommendation_id"):
                continue
            try:
                session.record_recommendation(
                    dict(products),
                    turn_id=turn.turn_id,
                    ranking_source=str(stored.payload.get("ranking_source") or "restored"),
                )
            except Exception:  # noqa: BLE001 - history stays usable without the context
                continue

        pending = self._state.get_open_pending_for_session(session_id)
        if pending is not None:
            session.pending_clarification = PendingClarification(
                payload=pending.payload,
                query=pending.query,
                turn_id=pending.turn_id,
                reply_turn_id=pending.reply_turn_id,
                mode=pending.mode,
                created_at=iso_from_epoch(pending.created_at),
                restored=True,
            )
        return session
