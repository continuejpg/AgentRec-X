"""Warm clarification-handle lifetime proof (M1-C Round A).

Why this module exists
----------------------
A durable ASK persists a ``clarification_id`` in SQLite, but the *authority* for that id is
a dictionary inside one MCP child process (``recommendation/mcp/server.py::AppContext.handles``).
The earlier Web gate took a caller boolean (``warm_available=True``), which proves nothing: the
same value survived a child restart, a manager restart, or an attempt to resume a handle that
was issued to a different browser session.  This module replaces that boolean with a checkable
fact.

How the proof works
-------------------
1. The BFF records every clarification handle it *observes issued* by a child, together with
   the identity of the process that issued it: ``MCPManager.instance_id`` (one per manager
   object) and ``ServerRuntime.process_instance_id`` (one per child process, regenerated on
   every ``start``/``restart``).
2. A reply is accepted only when the recorded identity still equals the live identity of the
   child serving that profile, that child is ready, the handle belongs to the same Web session,
   and the handle has not reached its absolute expiry.
3. Every lookup miss is a rejection.  There is no fallback, no "assume warm" branch, and no
   state that could be resurrected from SQLite after a restart.

The mirror is exact while the identities match because the child never evicts a handle before
its TTL (``ctx.handles`` entries are only added, and only their ``status`` changes), so
"recorded by the current process" and "still present in the child" coincide.  The child keeps
its own validation as defence in depth: an unknown or expired id is still a tool error there.
A dedicated read-only "probe" tool was deliberately *not* added: it would cost a subprocess
round-trip without adding a fact, and the MCP tool surface stays unchanged.

The expiry deadline is measured from the *request start* of the ASK call, never from its
response, so the BFF can only be stricter than the child.  A handle whose zero/negative
lifetime cannot be established is not recorded at all: an untrackable handle must never become
warm.

Everything here is pure (no MCP import, no I/O) so the decision is testable without a
subprocess.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "HANDLE_EXPIRED",
    "HANDLE_MISSING",
    "HANDLE_NOT_READY",
    "HANDLE_OK",
    "HANDLE_SESSION_MISMATCH",
    "HANDLE_STALE_PROCESS",
    "HANDLE_UNKNOWN_PROCESS",
    "HandleCheck",
    "HandleRef",
    "HandleTicket",
    "ProcessIdentity",
    "issued_handles_in_response",
    "verify_handle",
]

#: The recorded handle was issued by the current child process and is still resumable.
HANDLE_OK = "ok"
#: Nothing was ever recorded for this clarification id (or the record is unusable).
HANDLE_MISSING = "handle_missing"
#: The handle was issued to a different Web session.
HANDLE_SESSION_MISMATCH = "session_mismatch"
#: No live child process identity is known for this profile.
HANDLE_UNKNOWN_PROCESS = "process_unavailable"
#: The child process (or the manager) that issued the handle is gone.
HANDLE_STALE_PROCESS = "stale_process"
#: The child process is not ready to serve calls (starting, stopped, or after a crash).
HANDLE_NOT_READY = "process_not_ready"
#: The handle reached the absolute expiry reported by the child.
HANDLE_EXPIRED = "expired"

#: Rejection codes, in the order :func:`verify_handle` reports them.
REJECTION_CODES = (
    HANDLE_MISSING,
    HANDLE_SESSION_MISMATCH,
    HANDLE_UNKNOWN_PROCESS,
    HANDLE_STALE_PROCESS,
    HANDLE_NOT_READY,
    HANDLE_EXPIRED,
)


@dataclass(frozen=True)
class HandleRef:
    """The handle a reply is asking to resume, named by its durable coordinates."""

    profile_id: str
    session_id: str
    clarification_id: str


@dataclass(frozen=True)
class HandleTicket:
    """Proof that *this* child process issued one clarification handle.

    ``issued_at``/``expires_at`` are epoch seconds derived from the ASK request start, so
    ``expires_at`` is never later than the child's own deadline for the same handle.
    """

    clarification_id: str
    profile_id: str
    session_id: str | None
    manager_instance_id: str
    process_instance_id: str
    issued_at: float
    expires_at: float
    tool: str = ""

    def view(self) -> dict[str, Any]:
        """Diagnostics payload; contains no catalogue identity and no user data."""
        return {
            "clarification_id": self.clarification_id,
            "profile_id": self.profile_id,
            "tool": self.tool,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class ProcessIdentity:
    """The live identity of the child process that would serve a profile right now."""

    profile_id: str
    manager_instance_id: str
    process_instance_id: str
    ready: bool
    generation: int = 0


@dataclass(frozen=True)
class HandleCheck:
    """The decision about one handle; ``ok`` is the only thing callers may branch on."""

    ok: bool
    code: str = HANDLE_OK
    detail: str = ""
    ticket: HandleTicket | None = None

    def as_payload(self) -> dict[str, Any]:
        """A stable, opaque diagnostic shape for HTTP error details and logs."""
        return {"ok": self.ok, "code": self.code, "detail": self.detail}


def verify_handle(
    ref: HandleRef,
    ticket: HandleTicket | None,
    identity: ProcessIdentity | None,
    *,
    now: float | None = None,
) -> HandleCheck:
    """Decide whether ``ref`` may be resumed by the process described by ``identity``.

    The facts are checked in a fixed order so a rejection code always names the first failing
    fact: recorded handle -> session binding -> process availability -> process staleness ->
    readiness -> absolute expiry.  A restart therefore reports ``stale_process`` even when the
    handle would also have expired meanwhile.

    A ticket without a positive lifetime, or one recorded for a different handle/profile, is
    treated as *no* proof: the registry is evidence only if it is about this exact handle.
    """
    if ticket is None:
        return HandleCheck(False, HANDLE_MISSING,
                           "no handle was recorded for this clarification id")
    if ticket.clarification_id != ref.clarification_id or ticket.profile_id != ref.profile_id:
        return HandleCheck(
            False, HANDLE_MISSING,
            "the recorded handle belongs to another clarification id or profile",
        )
    if not (ticket.expires_at > ticket.issued_at):
        return HandleCheck(False, HANDLE_MISSING,
                           "the recorded handle has no positive lifetime")
    if not ref.session_id or ticket.session_id != ref.session_id:
        return HandleCheck(
            False, HANDLE_SESSION_MISMATCH,
            "the handle was issued in another Web session",
        )
    if identity is None or identity.profile_id != ref.profile_id:
        return HandleCheck(False, HANDLE_UNKNOWN_PROCESS,
                           "no live MCP child process serves this profile")
    if (identity.manager_instance_id != ticket.manager_instance_id
            or identity.process_instance_id != ticket.process_instance_id):
        return HandleCheck(
            False, HANDLE_STALE_PROCESS,
            "the MCP child process that issued this handle is no longer running",
        )
    if not identity.ready:
        return HandleCheck(False, HANDLE_NOT_READY,
                           "the MCP child process is not ready")
    reference = time.time() if now is None else float(now)
    if reference > ticket.expires_at:
        return HandleCheck(
            False, HANDLE_EXPIRED,
            f"the handle expired at {ticket.expires_at:.3f}",
        )
    return HandleCheck(True, HANDLE_OK, "issued by the current MCP child process", ticket)


def issued_handles_in_response(structured: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Extract ``(clarification_id, lifetime_seconds)`` from a real MCP tool result.

    Two documented shapes carry a handle: the top-level fields of
    ``request_preference_clarification`` and the nested ``clarification`` object of
    ``run_agent_turn``.  A missing id, a missing lifetime, or a non-positive lifetime yields
    nothing, so an untrackable handle never becomes warm.
    """
    found: list[tuple[str, float]] = []
    candidates: list[Mapping[str, Any]] = [structured]
    nested = structured.get("clarification")
    if isinstance(nested, Mapping):
        candidates.append(nested)
    for payload in candidates:
        clarification_id = payload.get("clarification_id")
        lifetime = payload.get("expires_in_seconds")
        if not isinstance(clarification_id, str) or not clarification_id.strip():
            continue
        if isinstance(lifetime, bool) or not isinstance(lifetime, (int, float)):
            continue
        seconds = float(lifetime)
        if not seconds > 0.0:
            continue
        if (clarification_id, seconds) not in found:
            found.append((clarification_id, seconds))
    return found
