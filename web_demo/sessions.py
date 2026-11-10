"""In-memory session / turn store for the Web Demo BFF.

The store is intentionally process-local: this is a local demo and no database exists.
A session is bound to exactly one fixed demo profile; the browser never sends a user id.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

DEFAULT_TITLE = "New shopping session"
TITLE_MAX_CHARS = 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime | None = None) -> str:
    return (ts or utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class Turn:
    """One persisted user turn with its real tool-derived view models."""

    turn_id: str
    user_text: str
    assistant_text: str = ""
    tool_activity: list[dict[str, Any]] = field(default_factory=list)
    products: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    clarification: dict[str, Any] | None = None
    memory_update: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    # Actual planner used for this turn (never inferred or fabricated by the UI).
    planner: str | None = None
    planner_model: str | None = None
    planner_calls: int | None = None
    planner_latency_ms: float | None = None
    created_at: str = field(default_factory=iso)

    def view(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "tool_activity": [dict(item) for item in self.tool_activity],
            "products": self.products,
            "comparison": self.comparison,
            "clarification": self.clarification,
            "memory_update": self.memory_update,
            "error": self.error,
            "planner": self.planner,
            "planner_model": self.planner_model,
            "planner_calls": self.planner_calls,
            "planner_latency_ms": self.planner_latency_ms,
            "created_at": self.created_at,
        }


@dataclass
class PendingClarification:
    """A clarification handle issued to this session, plus its stable reply turn id.

    ``reply_turn_id`` is generated when the clarification is issued and reused for every
    retry of the same reply, which is what makes the MCP write idempotent.
    """

    payload: dict[str, Any]
    query: str
    turn_id: str
    reply_turn_id: str
    mode: str = "cpu_live"
    created_at: str = field(default_factory=iso)
    reply_attempts: int = 0

    def view(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass
class Session:
    session_id: str
    profile_id: str
    title: str = DEFAULT_TITLE
    created_at: str = field(default_factory=iso)
    updated_at: str = field(default_factory=iso)
    turns: list[Turn] = field(default_factory=list)
    pending_clarification: PendingClarification | None = None
    _turn_counter: int = 0
    _reply_counter: int = 0

    # -- ids -----------------------------------------------------------------
    def next_turn_id(self) -> str:
        self._turn_counter += 1
        return f"{self.session_id}-t{self._turn_counter}"

    def next_reply_turn_id(self) -> str:
        self._reply_counter += 1
        return f"{self.session_id}-r{self._reply_counter}"

    # -- mutation ------------------------------------------------------------
    def touch(self) -> None:
        self.updated_at = iso()

    def set_title_from(self, text: str) -> None:
        if self.title != DEFAULT_TITLE:
            return
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return
        self.title = cleaned[:TITLE_MAX_CHARS] + ("…" if len(cleaned) > TITLE_MAX_CHARS else "")

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.touch()

    def reset_turns(self) -> None:
        self.turns = []
        self.pending_clarification = None
        self._turn_counter = 0
        self._reply_counter = 0
        self.touch()

    # -- views ---------------------------------------------------------------
    def view(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [turn.view() for turn in self.turns],
            "pending_clarification": (
                self.pending_clarification.view() if self.pending_clarification else None
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": len(self.turns),
        }


class SessionStore:
    """Thread-safe process-local session registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, profile_id: str) -> Session:
        with self._lock:
            for _ in range(32):
                session_id = "web-" + secrets.token_hex(4)
                if session_id not in self._sessions:
                    break
            else:  # pragma: no cover - collision loop is practically unreachable
                raise RuntimeError("could not allocate a unique session id")
            session = Session(session_id=session_id, profile_id=profile_id)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self, limit: int = 20) -> list[Session]:
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)
        return sessions[: max(1, limit)]
