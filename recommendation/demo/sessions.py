"""Application-level demo sessions for the Milestone 11 web demo.

A :class:`DemoSession` is the application's unit of isolation.  It owns exactly three
things and conflates nothing::

    session_id            opaque capability token (UUID4), the only client-visible handle
    user_key              the Milestone 9 preference-memory namespace for THIS session
    trusted_user_history  the read-only behavioural history taken from a demo profile

It is explicitly **not** conversational memory, and it is **not** writable from user
text.  The browser transcript is a fourth, purely visual concept that lives only in the
page (Milestone 11 design note).

Isolation rules, all enforced here
----------------------------------
* ``user_key`` is derived from the session id, never from the profile id, so two
  sessions that share a demo profile still have completely separate preference memory;
* turn identifiers are allocated by the **server** as ``<session_id>:<sequence>`` and
  are never accepted from a client;
* one :class:`threading.Lock` per session serialises that session's turns, so two
  simultaneous messages cannot share a turn id or interleave state.  There is no global
  lock: distinct sessions proceed independently.

Persistence split (stated explicitly, because it is easy to overclaim)
---------------------------------------------------------------------
* **preference memory** is persisted by the accepted Milestone 9 SQLite store and
  therefore survives an HTTP request, and also a server restart;
* the **live session registry is process-local** and does **not** survive a server
  restart.  A browser session id is therefore a demo capability, not a durable
  identity, and the demo makes no production-authentication claim.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping

from .profiles import DemoProfile

__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "DemoError",
    "DemoSession",
    "DemoSessionManager",
    "SessionCapacityExceeded",
    "TurnAllocation",
    "UnknownProfile",
    "UnknownSession",
]

#: Bounded live-session registry: a local demo never needs unbounded sessions.
DEFAULT_MAX_SESSIONS = 64

#: Prefix of the server-derived preference-memory namespace.
_USER_KEY_PREFIX = "demo-session"


class DemoError(Exception):
    """Base class for demo-layer failures."""

    code = "demo_error"


class UnknownProfile(DemoError):
    """The requested demo profile does not exist."""

    code = "unknown_profile"


class UnknownSession(DemoError):
    """The session id is unknown, expired or already reset.

    Deliberately an error rather than an implicit new session: silently creating one
    would hand a caller a fresh, empty preference namespace and quietly break the
    isolation the session id is supposed to represent.
    """

    code = "session_not_found"


class SessionCapacityExceeded(DemoError):
    """The bounded live-session registry is full."""

    code = "session_capacity_exceeded"


@dataclass
class DemoSession:
    """One isolated demo session."""

    session_id: str
    user_key: str
    profile_id: str
    trusted_user_history: tuple[str, ...]
    created_at: float
    creation_index: int
    next_turn_sequence: int = 0
    #: Serialises this session's turns only.  Repr-excluded: it is not state to display.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def turns_completed(self) -> int:
        """How many turn identifiers have been allocated."""
        return self.next_turn_sequence

    def turn_id_for(self, sequence: int) -> str:
        """Return the server-owned turn identifier for a sequence number."""
        return f"{self.session_id}:{sequence:06d}"


@dataclass(frozen=True)
class TurnAllocation:
    """One allocated turn: the session, its 1-based number and its server-owned id."""

    session: DemoSession
    turn_number: int
    turn_id: str


class DemoSessionManager:
    """Bounded, process-local registry of demo sessions.

    Parameters
    ----------
    profiles:
        The server-owned demo profiles, keyed by ``profile_id``.
    max_sessions:
        Capacity of the live registry.  Reaching it raises
        :class:`SessionCapacityExceeded` rather than evicting a live session.
    ttl_seconds:
        Optional idle timeout.  ``None`` disables expiry; expired sessions are swept
        when a new session is created.
    clock:
        Injectable time source, so expiry is testable without sleeping.
    """

    def __init__(
        self,
        profiles: Mapping[str, DemoProfile],
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when set")
        self._profiles = dict(profiles)
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, DemoSession] = {}
        self._retired_user_keys: set[str] = set()
        self._creation_counter = 0
        self._registry_lock = threading.Lock()

    # -- introspection ----------------------------------------------------- #

    @property
    def profiles(self) -> Mapping[str, DemoProfile]:
        """The server-owned demo profile registry."""
        return dict(self._profiles)

    @property
    def max_sessions(self) -> int:
        """Live-session capacity."""
        return self._max_sessions

    @property
    def ttl_seconds(self) -> float | None:
        """Idle timeout, or ``None`` when expiry is disabled."""
        return self._ttl_seconds

    def session_ids(self) -> tuple[str, ...]:
        """Live session ids in logical creation order."""
        with self._registry_lock:
            return tuple(
                session.session_id
                for session in sorted(
                    self._sessions.values(), key=lambda item: item.creation_index
                )
            )

    def __len__(self) -> int:
        with self._registry_lock:
            return len(self._sessions)

    def retired_user_keys(self) -> tuple[str, ...]:
        """Preference-memory namespaces belonging to reset sessions, in creation order.

        Exposed so a test can prove a reset session's namespace is never reissued and is
        never shared with a live session.
        """
        with self._registry_lock:
            return tuple(sorted(self._retired_user_keys))

    # -- lifecycle --------------------------------------------------------- #

    def create(self, profile_id: str) -> DemoSession:
        """Create a fresh, isolated session for ``profile_id``.

        Raises
        ------
        UnknownProfile
            No such demo profile exists.
        SessionCapacityExceeded
            The live registry is full.
        """
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise UnknownProfile(f"no demo profile {profile_id!r} is configured")

        with self._registry_lock:
            self._sweep_expired_locked()
            if len(self._sessions) >= self._max_sessions:
                raise SessionCapacityExceeded(
                    f"the demo supports at most {self._max_sessions} concurrent sessions"
                )
            session_id = str(uuid.uuid4())
            # The namespace is derived from the SESSION, never from the profile, so two
            # sessions on the same profile can never share preference memory.
            user_key = f"{_USER_KEY_PREFIX}:{session_id}"
            if user_key in self._retired_user_keys:  # pragma: no cover - uuid4 collision
                raise DemoError("generated a preference-memory namespace that was retired")
            self._creation_counter += 1
            session = DemoSession(
                session_id=session_id,
                user_key=user_key,
                profile_id=profile.profile_id,
                trusted_user_history=tuple(profile.trusted_user_history),
                created_at=self._clock(),
                creation_index=self._creation_counter,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> DemoSession:
        """Return a live session, or raise :class:`UnknownSession`.

        An expired session is removed and reported exactly like an unknown one, so a
        client can never tell "never existed" from "timed out" and never silently gets a
        new namespace.
        """
        with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSession(f"no live demo session {session_id!r}")
            if self._is_expired(session):
                self._retire_locked(session)
                raise UnknownSession(f"demo session {session_id!r} has expired")
            return session

    def delete(self, session_id: str) -> DemoSession:
        """Reset a session: remove it from the live registry and retire its namespace.

        Semantics (explicit, because "reset" is otherwise ambiguous):

        * the session id stops working immediately -- this is verified by taking the
          session's own turn lock first, so a delete waits for an in-flight turn and no
          later turn can slip in between;
        * its preference-memory ``user_key`` is **retired and never reissued**, so the
          namespace is unreachable through the API for the rest of the process lifetime;
        * no other session is touched in any way.

        What it deliberately does **not** do: erase Milestone 9 rows.  Accepted M9
        semantics retain provenance for superseded and removed entries, the store's
        public interface exposes no user-scoped delete, and the Milestone 11 web layer is
        forbidden from carrying store-mutation logic.  The session's rows therefore
        remain as unreachable audit provenance.  Callers must create a new session
        explicitly; the server never creates one silently.
        """
        with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSession(f"no live demo session {session_id!r}")

        # Wait for any in-flight turn on this session before retiring it.
        with session.lock:
            with self._registry_lock:
                if self._sessions.pop(session_id, None) is None:
                    raise UnknownSession(f"no live demo session {session_id!r}")
                self._retired_user_keys.add(session.user_key)
        return session

    def reset(self, session_id: str) -> DemoSession:
        """Alias for :meth:`delete` under the name the API uses."""
        return self.delete(session_id)

    # -- turn allocation --------------------------------------------------- #

    @contextmanager
    def turn(self, session_id: str) -> Iterator[TurnAllocation]:
        """Allocate the next turn for a session and hold that session's lock.

        The lock is held for the whole ``with`` body, which is what makes two
        simultaneous messages to one session sequential instead of interleaved.  The
        sequence number is consumed even if the body raises, so a turn identifier is
        never reused for a retried request.
        """
        session = self.get(session_id)
        with session.lock:
            # The session may have been reset while this thread waited for the lock.
            with self._registry_lock:
                if session_id not in self._sessions:
                    raise UnknownSession(f"no live demo session {session_id!r}")
                if self._is_expired(session):
                    self._retire_locked(session)
                    raise UnknownSession(f"demo session {session_id!r} has expired")
            session.next_turn_sequence += 1
            yield TurnAllocation(
                session=session,
                turn_number=session.next_turn_sequence,
                turn_id=session.turn_id_for(session.next_turn_sequence),
            )

    # -- internals --------------------------------------------------------- #

    def _is_expired(self, session: DemoSession) -> bool:
        if self._ttl_seconds is None:
            return False
        return (self._clock() - session.created_at) > self._ttl_seconds

    def _retire_locked(self, session: DemoSession) -> None:
        """Remove a session and retire its namespace.  Caller holds the registry lock."""
        self._sessions.pop(session.session_id, None)
        self._retired_user_keys.add(session.user_key)

    def _sweep_expired_locked(self) -> tuple[str, ...]:
        """Drop expired sessions.  Caller holds the registry lock."""
        if self._ttl_seconds is None:
            return ()
        expired = [
            session
            for session in sorted(self._sessions.values(), key=lambda item: item.creation_index)
            if self._is_expired(session)
        ]
        for session in expired:
            self._retire_locked(session)
        return tuple(session.session_id for session in expired)
