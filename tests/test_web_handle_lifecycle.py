"""Round A: warm clarification-handle lifetime proof (unit + real MCP child).

The point of these tests is the *replacement of a caller boolean by a checkable fact*: a
reply may only resume a handle that the **current** MCP child process issued, in the same Web
session, before its absolute deadline.  Every rejection must happen before any claim, any MCP
tool call and any M9 write.

The fast tests drive the pure decision and a manager whose client is a controlled fake; the
last test uses a real stdio MCP child and a real restart, because the whole claim ("the
process that issued it is gone") is only meaningful against a real process.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_web_request_store import _new_store, _setup  # noqa: E402
from web_demo.config import Settings  # noqa: E402
from web_demo.handle_lifecycle import (  # noqa: E402
    HANDLE_EXPIRED,
    HANDLE_MISSING,
    HANDLE_NOT_READY,
    HANDLE_OK,
    HANDLE_SESSION_MISMATCH,
    HANDLE_STALE_PROCESS,
    HANDLE_UNKNOWN_PROCESS,
    REJECTION_CODES,
    HandleCheck,
    HandleRef,
    HandleTicket,
    ProcessIdentity,
    issued_handles_in_response,
    verify_handle,
)
from web_demo.mcp_manager import MCPManager, MCPManagerError  # noqa: E402
from web_demo.reply_executor import WarmReplyExecutor  # noqa: E402

PROFILE = "profile_0057"
NOW = 1_700_000_000.0
#: Every rejection class a reply must survive without touching the outside world.
EXECUTOR_REJECTIONS = [
    HANDLE_MISSING,
    HANDLE_SESSION_MISMATCH,
    HANDLE_UNKNOWN_PROCESS,
    HANDLE_STALE_PROCESS,
    HANDLE_NOT_READY,
    HANDLE_EXPIRED,
]


# --------------------------------------------------------------------------- #
# pure decision
# --------------------------------------------------------------------------- #


def _ref(**overrides) -> HandleRef:
    values = {"profile_id": PROFILE, "session_id": "web-a", "clarification_id": "clar_a"}
    values.update(overrides)
    return HandleRef(**values)


def _ticket(**overrides) -> HandleTicket:
    values = {
        "clarification_id": "clar_a",
        "profile_id": PROFILE,
        "session_id": "web-a",
        "manager_instance_id": "mgr-1",
        "process_instance_id": "proc-1",
        "issued_at": NOW,
        "expires_at": NOW + 60.0,
        "tool": "request_preference_clarification",
    }
    values.update(overrides)
    return HandleTicket(**values)


def _identity(**overrides) -> ProcessIdentity:
    values = {
        "profile_id": PROFILE,
        "manager_instance_id": "mgr-1",
        "process_instance_id": "proc-1",
        "ready": True,
        "generation": 1,
    }
    values.update(overrides)
    return ProcessIdentity(**values)


def test_handle_of_the_current_process_is_warm() -> None:
    check = verify_handle(_ref(), _ticket(), _identity(), now=NOW + 1.0)
    assert check.ok
    assert check.code == HANDLE_OK
    assert check.ticket is not None and check.ticket.clarification_id == "clar_a"


def test_unknown_handle_is_rejected() -> None:
    check = verify_handle(_ref(), None, _identity(), now=NOW)
    assert not check.ok
    assert check.code == HANDLE_MISSING
    assert check.ticket is None


def test_ticket_for_another_handle_or_profile_is_not_evidence() -> None:
    other_handle = verify_handle(_ref(), _ticket(clarification_id="clar_b"), _identity(), now=NOW)
    assert other_handle.code == HANDLE_MISSING
    other_profile = verify_handle(_ref(), _ticket(profile_id="profile_0049"), _identity(), now=NOW)
    assert other_profile.code == HANDLE_MISSING


def test_ticket_without_a_positive_lifetime_is_not_evidence() -> None:
    broken = verify_handle(
        _ref(), _ticket(issued_at=NOW, expires_at=NOW), _identity(), now=NOW
    )
    assert broken.code == HANDLE_MISSING


def test_handle_from_another_session_is_rejected() -> None:
    foreign = verify_handle(
        _ref(session_id="web-b"), _ticket(), _identity(), now=NOW
    )
    assert foreign.code == HANDLE_SESSION_MISMATCH
    # A handle issued without a session namespace is never resumable from the Web layer.
    sessionless = verify_handle(
        _ref(), _ticket(session_id=None), _identity(), now=NOW
    )
    assert sessionless.code == HANDLE_SESSION_MISMATCH


def test_child_restart_invalidates_the_handle() -> None:
    restarted = verify_handle(
        _ref(), _ticket(), _identity(process_instance_id="proc-2"), now=NOW + 1.0
    )
    assert restarted.code == HANDLE_STALE_PROCESS


def test_manager_restart_invalidates_the_handle() -> None:
    restarted = verify_handle(
        _ref(),
        _ticket(),
        _identity(manager_instance_id="mgr-2", process_instance_id="proc-2"),
        now=NOW + 1.0,
    )
    assert restarted.code == HANDLE_STALE_PROCESS


def test_unavailable_or_unready_process_is_rejected() -> None:
    unknown = verify_handle(_ref(), _ticket(), None, now=NOW + 1.0)
    assert unknown.code == HANDLE_UNKNOWN_PROCESS
    other_profile = verify_handle(
        _ref(), _ticket(), _identity(profile_id="profile_0049"), now=NOW + 1.0
    )
    assert other_profile.code == HANDLE_UNKNOWN_PROCESS
    unready = verify_handle(_ref(), _ticket(), _identity(ready=False), now=NOW + 1.0)
    assert unready.code == HANDLE_NOT_READY


def test_expiry_is_inclusive_of_the_child_deadline() -> None:
    assert verify_handle(_ref(), _ticket(), _identity(), now=NOW + 60.0).ok
    expired = verify_handle(_ref(), _ticket(), _identity(), now=NOW + 60.001)
    assert expired.code == HANDLE_EXPIRED


def test_every_rejection_code_is_distinct_and_reachable() -> None:
    assert len(set(REJECTION_CODES)) == len(REJECTION_CODES)
    observed = {
        verify_handle(_ref(), None, _identity(), now=NOW).code,
        verify_handle(_ref(session_id="web-b"), _ticket(), _identity(), now=NOW).code,
        verify_handle(_ref(), _ticket(), None, now=NOW).code,
        verify_handle(_ref(), _ticket(), _identity(process_instance_id="p2"), now=NOW).code,
        verify_handle(_ref(), _ticket(), _identity(ready=False), now=NOW).code,
        verify_handle(_ref(), _ticket(), _identity(), now=NOW + 999).code,
    }
    assert observed == set(REJECTION_CODES)


def test_response_handle_extraction_requires_a_live_handle() -> None:
    assert issued_handles_in_response(
        {"clarification_id": "clar_a", "expires_in_seconds": 30}
    ) == [("clar_a", 30.0)]
    assert issued_handles_in_response({
        "status": "awaiting_clarification",
        "clarification": {"clarification_id": "qclar_b", "expires_in_seconds": 30.0},
    }) == [("qclar_b", 30.0)]
    # No id, no lifetime, a zero/negative lifetime or a boolean is never evidence.
    assert issued_handles_in_response({"clarification_id": "clar_a"}) == []
    assert issued_handles_in_response({"clarification_id": "clar_a", "expires_in_seconds": 0}) == []
    assert issued_handles_in_response({"clarification_id": "clar_a", "expires_in_seconds": -1}) == []
    assert issued_handles_in_response(
        {"clarification_id": "clar_a", "expires_in_seconds": True}
    ) == []
    assert issued_handles_in_response(
        {"clarification_id": "  ", "expires_in_seconds": 10}
    ) == []
    assert issued_handles_in_response({"status": "finalized"}) == []


# --------------------------------------------------------------------------- #
# manager mirror (fake client, real manager code path)
# --------------------------------------------------------------------------- #


class _FakeClient:
    """Stands in for one MCP client so the manager's proof path runs without a child."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def call_tool(self, tool: str, arguments: dict, progress_callback=None):  # noqa: ARG002
        self.calls.append(tool)
        return SimpleNamespace(
            is_error=False, structured_content=dict(self.responses[tool]), content=[]
        )


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "search_mode": "bm25",
        "allow_memory_writes": True,
        "log_dir": tmp_path / "logs",
        "repo": REPO_ROOT,
        "python_executable": sys.executable,
        "mcp_timeout_s": 120.0,
        "memory_db_path": tmp_path / "m9.sqlite3",
    }
    values.update(overrides)
    return Settings(**values)


def test_manager_records_issued_handles_and_never_calls_mcp_to_check(tmp_path: Path) -> None:
    manager = MCPManager(_settings(tmp_path))
    runtime = manager.runtimes[PROFILE]
    client = _FakeClient({
        "request_preference_clarification": {
            "clarification_id": "clar_a", "expires_in_seconds": 60.0,
        },
        "run_agent_turn": {
            "status": "awaiting_clarification",
            "clarification": {"clarification_id": "qclar_b", "expires_in_seconds": 30.0},
        },
    })
    runtime.client = client
    runtime.status = "ready"
    runtime.process_instance_id = "proc-a"

    async def calls() -> None:
        await manager.call_tool(
            PROFILE, "request_preference_clarification",
            {"query": "wireless headphones", "session_id": "web-a"},
        )
        await manager.call_tool(
            PROFILE, "run_agent_turn", {"query": "wireless headphones", "session_id": "web-a"}
        )

    asyncio.run(calls())
    assert client.calls == ["request_preference_clarification", "run_agent_turn"]

    first = manager.check_handle(_ref())
    assert first.ok and first.ticket is not None
    assert first.ticket.process_instance_id == "proc-a"
    assert first.ticket.session_id == "web-a"
    # The deadline is measured from the request start, never the response.
    assert first.ticket.issued_at <= time.time()
    assert first.ticket.expires_at - first.ticket.issued_at == pytest.approx(60.0)

    nested = manager.check_handle(_ref(clarification_id="qclar_b"))
    assert nested.ok and nested.ticket is not None
    assert nested.ticket.tool == "run_agent_turn"
    assert nested.ticket.expires_at - nested.ticket.issued_at == pytest.approx(30.0)

    assert manager.check_handle(_ref(session_id="web-b")).code == HANDLE_SESSION_MISMATCH
    assert manager.check_handle(_ref(clarification_id="clar_unknown")).code == HANDLE_MISSING
    assert client.calls == ["request_preference_clarification", "run_agent_turn"]

    # A child that is still starting cannot serve the reply even with a matching id.
    runtime.status = "starting"
    assert manager.check_handle(_ref()).code == HANDLE_NOT_READY

    # Same manager, new child process: the recorded handle is gone with the old process.
    runtime.status = "ready"
    runtime.process_instance_id = "proc-b"
    assert manager.check_handle(_ref()).code == HANDLE_STALE_PROCESS
    assert manager.check_handle(_ref(clarification_id="qclar_b")).code == HANDLE_STALE_PROCESS
    assert client.calls == ["request_preference_clarification", "run_agent_turn"]


def test_manager_without_a_running_child_rejects_every_handle(tmp_path: Path) -> None:
    manager = MCPManager(_settings(tmp_path))
    assert manager.process_identity(PROFILE) is not None
    assert manager.process_identity(PROFILE).process_instance_id == ""
    assert not manager.process_identity(PROFILE).ready
    assert manager.process_identity("profile_unknown") is None
    assert manager.check_handle(_ref()).code == HANDLE_MISSING


# --------------------------------------------------------------------------- #
# coordinator: rejected handle means zero claim, zero MCP, zero M9
# --------------------------------------------------------------------------- #


def _session() -> SimpleNamespace:
    return SimpleNamespace(session_id="web-r", profile_id=PROFILE)


def _tool_spy() -> tuple[list[str], object]:
    calls: list[str] = []

    async def call_tool(profile: str, tool: str, arguments: dict) -> dict:  # noqa: ARG001
        calls.append(tool)
        raise AssertionError(f"no MCP call may run for this case (saw {tool!r})")

    return calls, call_tool


def _executor(store, *, check, call_tool) -> WarmReplyExecutor:
    def assemble(request):  # noqa: ARG001 - no handle-proof test may reach assembly
        raise AssertionError("assembly must not run when the handle is rejected")

    return WarmReplyExecutor(
        store, call_tool=call_tool, profile_id=PROFILE, assemble=assemble,
        handle_check=check, heartbeat_interval=3600.0, lease_duration=30.0,
    )


@pytest.mark.parametrize("code", EXECUTOR_REJECTIONS)
def test_rejected_handle_never_claims_calls_mcp_or_writes_m9(tmp_path: Path, code: str) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        calls, spy = _tool_spy()
        executor = _executor(
            store, check=lambda ref, code=code: HandleCheck(False, code, "rejected"),
            call_tool=spy,
        )
        outcome = asyncio.run(executor.execute(
            _session(), pending, message="For gaming.",
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "warm_unavailable"
        assert outcome.handle is not None and outcome.handle.code == code
        assert outcome.detail.startswith(f"{code}:")
        assert calls == []
        assert store.get_request("web-r", pending.reply_turn_id) is None
        assert store.get_open_pending_for_session("web-r").status == "OPEN"
        assert store.list_turns("web-r") == ()
    finally:
        store.close()


def test_checker_is_asked_with_the_durable_pending_coordinates(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        seen: list[HandleRef] = []

        def check(ref: HandleRef) -> HandleCheck:
            seen.append(ref)
            return HandleCheck(True, HANDLE_OK)

        calls, spy = _tool_spy()
        executor = _executor(store, check=check, call_tool=spy)
        outcome = asyncio.run(executor.execute(
            _session(), pending, message="For gaming.",
            clarification_id="clar_someone_else",
        ))
        # The live handle is proven before the caller's echoed id is compared, and the
        # probe is built from the durable pending, not from the request body.
        assert outcome.outcome == "conflict"
        assert calls == []
        assert [(ref.profile_id, ref.session_id, ref.clarification_id) for ref in seen] == [
            (PROFILE, "web-r", pending.clarification_id)
        ]
    finally:
        store.close()


def test_executor_without_a_checker_fails_closed(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    try:
        _, pending = _setup(store)
        calls, spy = _tool_spy()

        def assemble(request):  # noqa: ARG001 - fail-closed test never assembles
            raise AssertionError("assembly must not run without a live handle")

        executor = WarmReplyExecutor(
            store, call_tool=spy, profile_id=PROFILE, assemble=assemble,
        )
        outcome = asyncio.run(executor.execute(
            _session(), pending, message="For gaming.",
            clarification_id=pending.clarification_id,
        ))
        assert outcome.outcome == "warm_unavailable"
        assert outcome.handle is not None and outcome.handle.code == HANDLE_MISSING
        assert calls == []
        assert store.get_request("web-r", pending.reply_turn_id) is None
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# real stdio MCP child: restart really invalidates the handle
# --------------------------------------------------------------------------- #


async def _ready(manager: MCPManager) -> None:
    deadline = time.time() + 180
    while True:
        try:
            manager.ensure_ready(PROFILE)
            return
        except MCPManagerError:
            if time.time() > deadline:
                raise
            await asyncio.sleep(1.0)


def test_real_child_restart_and_expiry_invalidate_issued_handle(tmp_path: Path) -> None:
    settings = _settings(tmp_path, clarification_ttl_s=3.0)

    async def flow() -> None:
        manager = MCPManager(settings)
        manager.start()
        try:
            await _ready(manager)
            runtime = manager.runtimes[PROFILE]

            ask = await manager.call_tool(
                PROFILE, "request_preference_clarification",
                {"query": "wireless headphones", "session_id": "web-handle-a"},
            )
            cid = str(ask["clarification_id"])
            ticket = manager.issued_handle(cid)
            assert ticket is not None
            assert ticket.process_instance_id == runtime.process_instance_id
            assert ticket.session_id == "web-handle-a"
            assert manager.check_handle(_ref(session_id="web-handle-a", clarification_id=cid)).ok
            assert manager.check_handle(
                _ref(session_id="web-handle-b", clarification_id=cid)
            ).code == HANDLE_SESSION_MISMATCH
            assert manager.check_handle(
                _ref(session_id="web-handle-a", clarification_id="clar_unknown")
            ).code == HANDLE_MISSING

            calls_before = dict(runtime.last_call or {})
            manager.check_handle(_ref(session_id="web-handle-a", clarification_id=cid))
            assert dict(runtime.last_call or {}) == calls_before  # checks made no tool call

            first_process = runtime.process_instance_id
            await manager.restart(PROFILE)
            await _ready(manager)
            assert runtime.process_instance_id != first_process
            stale = manager.check_handle(
                _ref(session_id="web-handle-a", clarification_id=cid)
            )
            assert stale.code == HANDLE_STALE_PROCESS

            # A real handle from the new child is still accepted: this is a proof, not a
            # blanket denial.  Then the absolute deadline rejects it without any call.
            again = await manager.call_tool(
                PROFILE, "request_preference_clarification",
                {"query": "wireless headphones", "session_id": "web-handle-a"},
            )
            cid2 = str(again["clarification_id"])
            assert manager.check_handle(
                _ref(session_id="web-handle-a", clarification_id=cid2)
            ).ok
            await asyncio.sleep(float(settings.clarification_ttl_s) + 0.5)
            expired = manager.check_handle(
                _ref(session_id="web-handle-a", clarification_id=cid2)
            )
            assert expired.code == HANDLE_EXPIRED
        finally:
            await manager.stop()

    asyncio.run(flow())
