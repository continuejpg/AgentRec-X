"""Versioned, JSON-serialisable Cold Resume snapshot for a pending clarification.

Why this exists
---------------
A clarification handle lives only inside one MCP child process (``AppContext.handles``).
When that child restarts, the durable Web state still knows *that* a question was asked, but
not enough to rebuild the MCP-side task: the original task dict, the question, the query and
the absolute deadline all lived in the dead process.  This module defines the snapshot that
closes that gap and the validation applied before anything is resumed.

The snapshot has two layers:

* the outer ``agentrecx.web.clarification_continuation.v1`` binds the Web side
  (session, task, pending, clarification, reset generation, absolute expiry);
* the inner ``...mcp.clarification_continuation.v1`` block is produced by the MCP child at
  ASK time and carries the *exact* task dict that request used, plus question/query/expiry.

Rules:

* the snapshot contains no handle identity that the new process may reuse, no memory address,
  no process object and no closure — only JSON data;
* the ``web_pending`` row is the authority for the outer binding: ``commit_ask`` overwrites
  those fields with the values it just committed, so a caller cannot smuggle a different
  session/task/pending into the snapshot;
* a missing, malformed, expired, version-incompatible or differently-bound snapshot is a
  rejection, never a guess made from the user's new reply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "CONTINUATION_SCHEMA",
    "CONTINUATION_VERSION",
    "MCP_CONTINUATION_SCHEMA",
    "ContinuationCheck",
    "build_web_continuation",
    "validate_continuation",
]

CONTINUATION_SCHEMA = "agentrecx.web.clarification_continuation.v1"
CONTINUATION_VERSION = 1
MCP_CONTINUATION_SCHEMA = "agentrecx.mcp.clarification_continuation.v1"

#: Rejection codes, all mapped to HTTP 409 ``clarification_resume_not_ready``.
CONTINUATION_MISSING = "continuation_missing"
CONTINUATION_UNREADABLE = "continuation_unreadable"
CONTINUATION_SCHEMA_MISMATCH = "continuation_unsupported_schema"
CONTINUATION_VERSION_MISMATCH = "continuation_unsupported_version"
CONTINUATION_MODE_UNAVAILABLE = "continuation_mode_unavailable"
CONTINUATION_PROFILE_MISMATCH = "continuation_profile_mismatch"
CONTINUATION_SESSION_MISMATCH = "continuation_session_mismatch"
CONTINUATION_TASK_MISMATCH = "continuation_task_mismatch"
CONTINUATION_PENDING_MISMATCH = "continuation_pending_mismatch"
CONTINUATION_CLARIFICATION_MISMATCH = "continuation_clarification_mismatch"
CONTINUATION_GENERATION_MISMATCH = "continuation_generation_mismatch"
CONTINUATION_EXPIRED = "continuation_expired"
CONTINUATION_MALFORMED = "continuation_malformed"

_EXPIRY_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class ContinuationCheck:
    """The decision about one snapshot; ``ok`` is the only branchable field."""

    ok: bool
    code: str = "ok"
    detail: str = ""
    #: The MCP-side block that may be sent to a new child for restoration.
    payload: dict[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "detail": self.detail}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_web_continuation(
    *,
    mcp_continuation: Any,
    query: str,
    profile_id: str,
) -> dict[str, Any] | None:
    """Wrap the child's ASK continuation in the Web snapshot, or return ``None``.

    ``None`` means "this ASK did not provide a restorable MCP context" (for example a Qwen
    handle or a legacy server).  The pending is then stored without a continuation and a
    later cold resume is refused instead of guessed.  The Web binding fields written here are
    placeholders; :meth:`SQLiteWebStateStore.commit_ask` overwrites them with the committed
    identity values.
    """
    if not isinstance(mcp_continuation, Mapping):
        return None
    if mcp_continuation.get("schema") != MCP_CONTINUATION_SCHEMA:
        return None
    if mcp_continuation.get("mode") != "cpu_live":
        return None
    session_id = _text(mcp_continuation.get("session_id"))
    if not session_id:
        return None
    expires_at = _number(mcp_continuation.get("expires_at"))
    created_at = _number(mcp_continuation.get("created_at"))
    question = _text(mcp_continuation.get("question"))
    current_query = _text(mcp_continuation.get("current_query"))
    if expires_at is None or created_at is None or not question or not current_query:
        return None
    task = mcp_continuation.get("task")
    if not isinstance(task, Mapping) or not _text(task.get("task_id")):
        return None
    return {
        "schema": CONTINUATION_SCHEMA,
        "version": CONTINUATION_VERSION,
        "mode": "cpu_live",
        "profile_id": str(profile_id),
        "session_id": session_id,
        "web_task_id": "",
        "pending_id": "",
        "clarification_id": _text(mcp_continuation.get("clarification_id")),
        "reset_generation": 0,
        "query": str(query),
        "question": question,
        "reason": str(mcp_continuation.get("reason") or ""),
        "expires_at": expires_at,
        "mcp": dict(mcp_continuation),
    }


def validate_continuation(
    continuation: Any,
    *,
    session_id: str,
    profile_id: str,
    task_id: str,
    pending_id: str,
    clarification_id: str,
    reset_generation: int,
    expires_at: float,
    query: str,
    expected_user_key: str,
    now: float | None = None,
) -> ContinuationCheck:
    """Validate a stored snapshot against the current durable row and this process config.

    Every fact is checked against the *current* row, so a reset, a different pending or a
    different session cannot reuse an old snapshot.  The returned ``payload`` is only the
    MCP-side block; the outer binding stays in the Web layer.
    """
    if continuation is None:
        return ContinuationCheck(
            False, CONTINUATION_MISSING,
            "this pending was stored before Cold Resume existed; it cannot be restored",
        )
    if not isinstance(continuation, Mapping):
        return ContinuationCheck(False, CONTINUATION_UNREADABLE,
                                 "the stored continuation is not an object")
    if continuation.get("schema") != CONTINUATION_SCHEMA:
        return ContinuationCheck(
            False, CONTINUATION_SCHEMA_MISMATCH,
            f"unsupported continuation schema {continuation.get('schema')!r}",
        )
    if continuation.get("version") != CONTINUATION_VERSION:
        return ContinuationCheck(
            False, CONTINUATION_VERSION_MISMATCH,
            f"unsupported continuation version {continuation.get('version')!r}",
        )
    if continuation.get("mode") != "cpu_live":
        return ContinuationCheck(
            False, CONTINUATION_MODE_UNAVAILABLE,
            f"cold resume is only implemented for cpu_live, not {continuation.get('mode')!r}",
        )
    if str(continuation.get("profile_id")) != str(profile_id):
        return ContinuationCheck(False, CONTINUATION_PROFILE_MISMATCH,
                                 "the continuation belongs to another MCP profile")
    if str(continuation.get("session_id")) != str(session_id):
        return ContinuationCheck(False, CONTINUATION_SESSION_MISMATCH,
                                 "the continuation belongs to another Web session")
    if str(continuation.get("web_task_id")) != str(task_id):
        return ContinuationCheck(False, CONTINUATION_TASK_MISMATCH,
                                 "the continuation belongs to another task")
    if str(continuation.get("pending_id")) != str(pending_id):
        return ContinuationCheck(False, CONTINUATION_PENDING_MISMATCH,
                                 "the continuation belongs to another pending question")
    if str(continuation.get("clarification_id")) != str(clarification_id):
        return ContinuationCheck(
            False, CONTINUATION_CLARIFICATION_MISMATCH,
            "the continuation belongs to another clarification id",
        )
    try:
        stored_generation = int(continuation.get("reset_generation"))
    except (TypeError, ValueError):
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the continuation has no integer reset_generation")
    if stored_generation != int(reset_generation):
        return ContinuationCheck(
            False, CONTINUATION_GENERATION_MISMATCH,
            "the session was reset after this question was asked",
        )
    stored_expiry = _number(continuation.get("expires_at"))
    if stored_expiry is None or abs(stored_expiry - float(expires_at)) > _EXPIRY_TOLERANCE_S:
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the continuation expiry disagrees with the pending row")
    if str(continuation.get("query")).strip() != str(query).strip():
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the continuation query disagrees with the pending row")
    reference = time.time() if now is None else float(now)
    if reference > stored_expiry:
        return ContinuationCheck(False, CONTINUATION_EXPIRED,
                                 f"the continuation expired at {stored_expiry:.3f}")

    mcp = continuation.get("mcp")
    if not isinstance(mcp, Mapping):
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the continuation has no MCP context block")
    if mcp.get("schema") != MCP_CONTINUATION_SCHEMA:
        return ContinuationCheck(False, CONTINUATION_SCHEMA_MISMATCH,
                                 "the MCP context block has an unsupported schema")
    if mcp.get("mode") != "cpu_live":
        return ContinuationCheck(False, CONTINUATION_MODE_UNAVAILABLE,
                                 "the MCP context block is not a cpu_live handle")
    if str(mcp.get("user_key")) != str(expected_user_key):
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the MCP context was issued to another trusted user")
    if str(mcp.get("session_id")) != str(session_id):
        return ContinuationCheck(False, CONTINUATION_SESSION_MISMATCH,
                                 "the MCP context was issued in another session")
    if str(mcp.get("clarification_id")) != str(clarification_id):
        return ContinuationCheck(False, CONTINUATION_CLARIFICATION_MISMATCH,
                                 "the MCP context carries another clarification id")
    if _text(mcp.get("current_query")) != str(query).strip():
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the MCP context query disagrees with the pending row")
    if not _text(mcp.get("question")):
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the MCP context has no question")
    task = mcp.get("task")
    if not isinstance(task, Mapping) or not _text(task.get("task_id")):
        return ContinuationCheck(False, CONTINUATION_MALFORMED,
                                 "the MCP context has no task snapshot")
    return ContinuationCheck(True, "ok", "durable continuation is restorable", dict(mcp))
