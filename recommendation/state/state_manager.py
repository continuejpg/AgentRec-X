"""Small orchestration API over the append-only event store and the belief reducer.

The manager is the only place that combines the two halves of the M5-MVP state
infrastructure: the store owns durable, ordered events, and
:class:`~recommendation.state.belief_updater.BeliefUpdater` owns the deterministic
event-to-state reduction.  The manager adds no policy of its own -- no ranking, no memory
write, no model or LLM call.  Its four operations are the ones
``docs/M5_UNIFIED_DECISION_STATE.md`` section 7 specifies: ``load_or_create``,
``apply_event``, ``get_state`` and ``replay_state``.

Derived state, not cached state
-------------------------------
Every read reconstructs the state by replaying the stored event stream from an empty
state.  Nothing is snapshotted or persisted separately, so the returned state can never
diverge from the events that produced it.  The cost -- one full fold per call -- is the
documented, accepted limitation of this milestone for conversational task lengths.

Validation before append (why it is not append-then-replay)
-----------------------------------------------------------
The store is append-only and nothing is ever deleted, so an invalid transition must never
reach it: an append-then-replay implementation would persist the rejected event, and every
later replay of that task would raise forever on the poisoned row.  :meth:`apply_event`
therefore runs the **same single reducer** as a pure dry run on the current replay state
*before* calling ``store.append``:

1. if the event id is already stored for its user, the store's own idempotent/conflict
   semantics decide -- a matching re-submission returns the stored event and replays
   unchanged, a differing one raises ``EventConflictError``;
2. otherwise the reducer validates the transition on the current replay state (with the
   caller's placeholder sequence cleared, because the store assigns the real sequence).  A
   rejected transition (e.g. confirming a hypothesis that does not exist) raises before
   anything is appended, so the store and the previous state stay byte-identical;
3. only then is the event appended (the store's transaction assigns its per-user
   sequence), and the task is replayed from the stored stream.

The reducer is imported, never duplicated: the dry run and the replay use the same
:class:`~recommendation.state.belief_updater.BeliefUpdater` instance.

Concurrency
-----------
One in-process ``RLock`` serialises a manager's whole validate -> append -> replay
sequence, so two calls on the same manager cannot interleave between the dry run and the
append.  The underlying store is itself thread-safe (one lock plus ``BEGIN IMMEDIATE``
appends), but the manager lock closes the cross-call validation window.  Concurrent
writers to the same task from separate processes or separate managers are out of scope for
this MVP: the store is a single-writer local SQLite store (known limitation).
"""

from __future__ import annotations

import threading

from .belief_updater import BeliefUpdater
from .decision_state import UnifiedDecisionState, empty_state
from .event_store import EventStore
from .events import StateEvent
from .schemas import MAX_SESSION_ID_LENGTH, bounded_identifier

__all__ = ["StateManager"]


class StateManager:
    """Replay-based orchestration over an :class:`EventStore` and a ``BeliefUpdater``.

    The store is injected (any object satisfying the
    :class:`~recommendation.state.event_store.EventStore` protocol), so tests and future
    backends do not depend on the SQLite implementation.  The updater is injected for the
    same reason and defaults to the deterministic reducer.
    """

    def __init__(
        self, store: EventStore, *, updater: BeliefUpdater | None = None
    ) -> None:
        self._store = store
        self._updater = updater if updater is not None else BeliefUpdater()
        # Serialises the whole apply sequence (duplicate check -> dry run -> append ->
        # replay) for this manager.  The store serialises single operations; this lock
        # stops another call on the same manager from changing the state the dry run
        # validated against before the append lands.
        self._lock = threading.RLock()

    # -- reads -------------------------------------------------------------- #

    def load_or_create(
        self, user_key: str, session_id: str | None, task_id: str
    ) -> UnifiedDecisionState:
        """Return the replayed state for a task, stamped with the caller's session.

        The state is derived from the stored event stream (or is the empty state when the
        task has no events yet).  The returned ``session_id`` is the caller's current
        session when one is supplied, because the manager is being entered for that
        session; passing ``None`` keeps the session recorded by the last task-scoped event.
        Nothing is persisted by this call.
        """
        state = self.replay_state(user_key, task_id)
        if session_id is None:
            return state
        # ``model_copy`` does not validate, so the caller-supplied session is validated
        # with the same frozen identifier rule the state model applies.
        resolved_session = bounded_identifier(
            session_id, field="session_id", max_length=MAX_SESSION_ID_LENGTH
        )
        return state.model_copy(update={"session_id": resolved_session})

    def get_state(self, user_key: str, task_id: str) -> UnifiedDecisionState:
        """Return the replayed state for a task (empty when it has no events)."""
        return self.replay_state(user_key, task_id)

    def replay_state(self, user_key: str, task_id: str) -> UnifiedDecisionState:
        """Explicit deterministic reconstruction: empty state folded with the stream.

        The stream is the store-ordered task stream, including the user's ``long_term``
        events from other tasks.  The fold is the updater's ``apply_all``, so replay is
        the same function that applies new events; no second reduction path exists.
        """
        state = empty_state(user_key=user_key, task_id=task_id)
        return self._updater.apply_all(state, self._store.get_events(task_id, user_key))

    # -- writes ------------------------------------------------------------- #

    def apply_event(self, event: StateEvent) -> UnifiedDecisionState:
        """Validate, append and replay one event, returning the caller's task state.

        A rejected append or a rejected transition raises before any replay is returned,
        so the store and the previously observable state are unchanged.  Re-submitting an
        event id already stored for its user is idempotent (the stored event is returned
        by the store and the same state is replayed); the same id with different
        semantics raises ``EventConflictError``.
        """
        if not isinstance(event, StateEvent):
            raise TypeError("apply_event requires a StateEvent instance")
        with self._lock:
            # (1) A known id for this user: let the store decide idempotent vs conflict.
            # Do not pre-validate a re-submission against the already-applied state.
            if self._store.get_event(event.event_id, user_key=event.user_key) is not None:
                stored = self._store.append(event)
                return self.replay_state(stored.user_key, stored.task_id)

            # (2) Dry-run the transition on the current replay state.  The reducer is
            # pure, so the result is discarded; only the rejection matters.  The
            # caller's placeholder sequence is cleared because the store assigns the
            # real one inside the append transaction.
            current = self.replay_state(event.user_key, event.task_id)
            validation_event = event.model_copy(update={"sequence": None})
            self._updater.apply(current, validation_event)

            # (3) Accepted: append (sequence assignment, conflict detection, durable
            # row) and replay the stored stream into the returned derived state.
            stored = self._store.append(event)
            return self.replay_state(stored.user_key, stored.task_id)
