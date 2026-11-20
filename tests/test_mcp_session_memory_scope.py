"""Real MCP tests for Web Session M9 scope (S5b-P2b-B1a)."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_demo.config import Settings  # noqa: E402
from web_demo.mcp_manager import MCPManager, MCPManagerError  # noqa: E402


def _settings(tmp_path: Path, db: Path) -> Settings:
    return Settings(search_mode="bm25", allow_memory_writes=True,
                    log_dir=tmp_path / "logs", repo=REPO_ROOT,
                    python_executable=sys.executable, mcp_timeout_s=120.0,
                    memory_db_path=db)


async def _ready(manager: MCPManager) -> None:
    deadline = time.time() + 180
    while True:
        try:
            manager.ensure_ready("profile_0057")
            return
        except MCPManagerError:
            if time.time() > deadline:
                raise
            await asyncio.sleep(2)


async def _ask(manager: MCPManager, sid: str) -> str:
    ask = await manager.call_tool("profile_0057", "request_preference_clarification",
                                  {"query": "wireless headphones", "session_id": sid})
    return ask["clarification_id"]


async def _reply(manager: MCPManager, sid: str, cid: str, message: str, turn_id: str):
    return await manager.call_tool("profile_0057", "submit_clarification_reply", {
        "clarification_id": cid, "user_message": message, "turn_id": turn_id,
        "confirmed_by_user": True, "confirmation_source": "web_ui_explicit_user_reply",
        "session_id": sid,
    })


async def _active(manager: MCPManager, sid: str) -> list[str]:
    prefs = await manager.call_tool("profile_0057", "get_active_preferences",
                                    {"session_id": sid})
    return [str(p.get("value")) for p in (prefs.get("active_preferences") or [])]


def test_web_reply_scope_is_session_and_isolated(tmp_path: Path) -> None:
    db = tmp_path / "m9.sqlite3"
    settings = _settings(tmp_path, db)

    async def flow() -> None:
        manager = MCPManager(settings)
        manager.start()
        try:
            await _ready(manager)
            cid = await _ask(manager, "web-scope-a")
            reply = await _reply(manager, "web-scope-a", cid, "I prefer black.", "r1")
            assert reply["update"]["added"] == 1
            assert reply["user_key"] == "web-scope-a"
            assert await _active(manager, "web-scope-a") == ["black"]
            assert await _active(manager, "web-scope-b") == []

            # Same turn id + same message: M9 deduplicates within the session.
            retry = await _reply(manager, "web-scope-a", cid, "I prefer black.", "r1")
            assert retry["update"]["already_processed"] is True
            assert retry["update"]["added"] == 0

            # A second session shares the fixture task but never shares the preference.
            cid_b = await _ask(manager, "web-scope-b")
            await _reply(manager, "web-scope-b", cid_b, "I prefer black.", "r1")
            assert await _active(manager, "web-scope-a") == ["black"]
            assert await _active(manager, "web-scope-b") == ["black"]

            # Withdrawal and re-add through fresh real handles, new turn ids.
            cid2 = await _ask(manager, "web-scope-a")
            await _reply(manager, "web-scope-a", cid2, "I don't care about color anymore.", "r2")
            assert await _active(manager, "web-scope-a") == []
            assert await _active(manager, "web-scope-b") == ["black"]
            cid3 = await _ask(manager, "web-scope-a")
            await _reply(manager, "web-scope-a", cid3, "I prefer black.", "r3")
            assert await _active(manager, "web-scope-a") == ["black"]
        finally:
            await manager.stop()

    asyncio.run(flow())

    connection = sqlite3.connect(str(db))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT user_key, value, status FROM preference_memory ORDER BY user_key"
        ).fetchall()
    finally:
        connection.close()
    keys = {row["user_key"] for row in rows}
    assert keys == {"web-scope-a", "web-scope-b"}
    assert all(row["user_key"] != "p14e2_lite_0057" for row in rows)


def test_session_preference_survives_mcp_restart(tmp_path: Path) -> None:
    db = tmp_path / "m9.sqlite3"
    settings = _settings(tmp_path, db)

    async def write() -> None:
        manager = MCPManager(settings)
        manager.start()
        try:
            await _ready(manager)
            cid = await _ask(manager, "web-restart")
            await _reply(manager, "web-restart", cid, "I prefer black.", "r1")
        finally:
            await manager.stop()

    asyncio.run(write())

    async def read() -> None:
        manager = MCPManager(settings)
        manager.start()
        try:
            await _ready(manager)
            assert await _active(manager, "web-restart") == ["black"]
            assert await _active(manager, "web-other") == []
        finally:
            await manager.stop()

    asyncio.run(read())


def test_fixture_path_keeps_task_id_identity() -> None:
    from recommendation.shopping_companion.memory import ShoppingCompanionMemory
    from recommendation.skills.clarification import resume_with_user_reply

    memory = ShoppingCompanionMemory()
    try:
        result = resume_with_user_reply(
            memory=memory, task={"task_id": "fixture-user-1"}, user_message="I prefer black.",
            turn_id="fx-r1",
        )
        assert result["user_key"] == "fixture-user-1"
        assert result["update"]["added"] == 1
    finally:
        memory.close()
