"""Typed errors for the M5-MVP unified decision-state engine.

Every rejection in this package is one of these types, so a caller can tell *why* a
state transition or event append was refused without parsing a message.  The reducer
is pure: raising one of these must leave the caller's state untouched.
"""

from __future__ import annotations

__all__ = [
    "CrossUserStateError",
    "EventConflictError",
    "EventStoreError",
    "StateEngineError",
    "UnsupportedTransitionError",
]


class StateEngineError(Exception):
    """Base class for every deterministic state-engine rejection."""


class EventStoreError(StateEngineError):
    """The append-only event store could not perform the requested operation."""


class EventConflictError(EventStoreError):
    """An event id was reused with a different semantic payload."""


class CrossUserStateError(StateEngineError):
    """An operation would let one user's events reach another user's state."""


class UnsupportedTransitionError(StateEngineError):
    """An event is well-formed but not legal for the current state.

    Examples: confirming a hypothesis that does not exist, retracting nothing, an
    inference that would require overriding an explicit statement, or a task-scoped
    event arriving for a different task.
    """
