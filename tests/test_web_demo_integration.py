"""Live integration test: real BM25 MCP servers + the BFF's long-lived MCP manager.

This test starts TWO real official-MCP stdio servers (p14e2_lite_0049 read-only and
p14e2_lite_0057 with memory writes enabled) through :class:`web_demo.mcp_manager.MCPManager`.
It is slow (catalogue load) and therefore gated:

    AGENTRECX_RUN_MCP_INTEGRATION=1 \
        .venv/bin/python -m pytest -q tests/test_web_demo_integration.py

It verifies connect + discovery, real search (catalogue_size 76263), product evidence, the
Rule Planner/Gate recommendation, P16 comparison, the real P16 error path, and the full
clarification -> M9 write -> preference refresh sequence using the additive
``web_ui_explicit_user_reply`` confirmation channel.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from web_demo.config import REPO, Settings
from web_demo.mcp_manager import MCPManager, MCPManagerError

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("AGENTRECX_RUN_MCP_INTEGRATION") != "1",
        reason="live MCP integration test; set AGENTRECX_RUN_MCP_INTEGRATION=1 to run",
    ),
]

RUNS = REPO / "runs/web_demo"
TRANSCRIPT_PATH = RUNS / "mcp_integration_transcript.json"
EXPECTED_TOOLS = {
    "search_products",
    "get_product_info",
    "recommend_products",
    "get_active_preferences",
    "compare_products",
    "request_preference_clarification",
    "submit_clarification_reply",
}


def _save(transcript: dict[str, Any]) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_PATH.write_text(json.dumps(transcript, indent=1, ensure_ascii=False) + "\n",
                               encoding="utf-8")


async def _wait_ready(manager: MCPManager, transcript: dict[str, Any], timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = manager.status()
        if manager.ready():
            return
        if status == "error":
            raise AssertionError(
                "MCP servers failed to start: "
                + json.dumps([server for server in manager.health_servers()], ensure_ascii=False)
            )
        await asyncio.sleep(0.5)
    raise AssertionError(f"MCP servers not ready after {timeout_s}s: {manager.health_servers()}")


async def _call(
    manager: MCPManager,
    transcript: dict[str, Any],
    profile_id: str,
    tool: str,
    arguments: dict[str, Any],
    *,
    expect_error: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await manager.call_tool(profile_id, tool, arguments)
    except MCPManagerError as exc:
        latency = round((time.perf_counter() - started) * 1000.0, 2)
        transcript["steps"].append(
            {
                "profile_id": profile_id,
                "tool": tool,
                "arguments": arguments,
                "ok": False,
                "error_code": exc.code,
                "error_message": exc.message,
                "latency_ms": latency,
            }
        )
        if expect_error:
            return {"__error__": exc.as_payload()}
        raise
    latency = round((time.perf_counter() - started) * 1000.0, 2)
    transcript["steps"].append(
        {
            "profile_id": profile_id,
            "tool": tool,
            "arguments": arguments,
            "ok": True,
            "latency_ms": latency,
            "result_keys": sorted(result.keys()),
        }
    )
    return result


async def _scenario(tmp_path: Path) -> dict[str, Any]:
    settings = Settings(
        search_mode="bm25",
        allow_memory_writes=True,
        log_dir=tmp_path / "logs",
        mcp_timeout_s=300.0,
    )
    manager = MCPManager(settings)
    transcript: dict[str, Any] = {
        "schema": "agentrecx.web_demo.mcp_integration.v1",
        "search_mode": "bm25",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }
    try:
        started = time.perf_counter()
        manager.start()
        await _wait_ready(manager, transcript)
        transcript["ready_after_s"] = round(time.perf_counter() - started, 2)
        servers = manager.health_servers()
        transcript["servers"] = servers
        assert {server["profile_id"] for server in servers} == {"profile_0049", "profile_0057"}
        for server in servers:
            assert server["status"] == "ready", server
            assert server["startup_ms"] and server["startup_ms"] > 0
            assert set(server["tools"] or []) == EXPECTED_TOOLS, server["tools"]
        assert manager.ready() is True
        assert manager.aggregate_status()[2] is True  # at least one server ready -> cpu_live
        writes = {server["profile_id"]: server["writes_enabled"] for server in servers}
        assert writes == {"profile_0049": False, "profile_0057": True}, writes
        assert all(
            "writes_disabled" not in (server["error"] or "") for server in servers
        )

        # --- real catalogue search / evidence / Gate recommendation (read-only profile) ---
        search = await _call(
            manager, transcript, "profile_0049", "search_products",
            {"query": "wireless earbuds for working out", "top_k": 3},
        )
        assert search["status"] == "ok"
        assert search["catalogue_size"] == 76263
        assert search["search_mode"] == "bm25"
        assert search["hits"], "real BM25 search returned no hits"
        hit_ids = [hit["item_id"] for hit in search["hits"]]
        assert all(hit["namespaced_id"].startswith("amazon_electronics_c4:") for hit in search["hits"])
        assert manager.observed_catalogue_size == 76263

        info = await _call(
            manager, transcript, "profile_0049", "get_product_info",
            {"product_ref": "B004K6R43G"},
        )
        assert info["status"] == "ok"
        assert info["product_info"]["title"]
        assert info["product_evidence"], "expected real P15 evidence spans"
        span = info["product_evidence"][0]
        assert span["span_text"] and span["verification_status"] == "lexical_span_exact"
        assert {item["attribute"] for item in info["unknown_fields"]} >= {"price", "brand"}
        assert info["selected_product"]["source_record"].endswith("B004K6R43G")

        recommendation = await _call(
            manager, transcript, "profile_0049", "recommend_products",
            {"query": "I need earbuds I can use while working out."},
        )
        assert recommendation["status"] == "finalized", recommendation.get("status")
        assert recommendation["gate"]["action"] == "ALLOW_FINALIZE"
        assert recommendation["selected_product_id"]
        assert recommendation["selected_product"]["item_id"] == recommendation["selected_product_id"]

        # --- real P16 comparison ---
        comparison = await _call(
            manager, transcript, "profile_0049", "compare_products",
            {"product_refs": ["B004K6R43G", "B00008VF63"]},
        )
        assert comparison["status"] == "comparison_ready", comparison.get("status")
        assert [product["item_id"] for product in comparison["products"]] == [
            "B004K6R43G", "B00008VF63",
        ]
        assert comparison["claims_valid"] > 0 and comparison["claims_rejected"] == 0
        assert comparison["qwen_used"] is False
        assert comparison["final_answer"]

        # --- real error payload path (no mocking, no fake fallback) ---
        # The P16 comparison skill reports invalid references as a real result payload
        # (status=invalid_product_reference + errors), not as an MCP transport error; the BFF maps
        # that status to HTTP 400.  An MCP-level error is exercised with a read-only memory write.
        bad_compare = await _call(
            manager, transcript, "profile_0049", "compare_products",
            {"product_refs": ["B004K6R43G", "B000NOTREAL"]},
        )
        assert bad_compare["status"] == "invalid_product_reference", bad_compare.get("status")
        assert bad_compare["errors"], "expected the real P16 reference errors"
        transcript["comparison_error"] = {
            "status": bad_compare["status"],
            "errors": bad_compare["errors"],
            "final_answer": bad_compare["final_answer"],
        }

        # --- clarification -> M9 memory write -> preference refresh (write-enabled profile) ---
        # The chat controller relies on the real Gate asking for clarification on a vague request.
        vague = await _call(
            manager, transcript, "profile_0057", "recommend_products",
            {"query": "I need some new headphones. Not sure what matters yet."},
        )
        transcript["vague_recommendation"] = {
            "status": vague["status"],
            "gate_action": (vague.get("gate") or {}).get("action"),
            "gate_reason_code": (vague.get("gate") or {}).get("reason_code"),
        }
        assert vague["status"] == "awaiting_clarification", (
            f"vague query did not ask for clarification: {transcript['vague_recommendation']}"
        )

        preferences_before = await _call(
            manager, transcript, "profile_0057", "get_active_preferences", {}
        )
        assert preferences_before["authorized_user_key"] == "p14e2_lite_0057"
        active_before = list(preferences_before["active_preferences"])
        clarification = await _call(
            manager, transcript, "profile_0057", "request_preference_clarification",
            {"query": "I need some new headphones. Not sure what matters yet."},
        )
        assert clarification["status"] == "awaiting_clarification"
        assert clarification["requires_user_reply"] is True
        assert clarification["memory_write_enabled"] is True
        clarification_id = clarification["clarification_id"]
        assert clarification_id.startswith("clar_")

        reply_turn_id = f"web-integration-{uuid.uuid4().hex}"
        reply = await _call(
            manager, transcript, "profile_0057", "submit_clarification_reply",
            {
                "clarification_id": clarification_id,
                "user_message": "I prefer noise cancelling and wireless.",
                "turn_id": reply_turn_id,
                "confirmed_by_user": True,
                "confirmation_source": "web_ui_explicit_user_reply",
            },
        )
        assert reply["status"] == "memory_updated"
        assert reply["confirmation_source"] == "web_ui_explicit_user_reply"
        assert reply["already_processed"] is False
        update = reply["update"]
        transcript["memory_update"] = {
            "added": update["added"],
            "superseded": update["superseded"],
            "removed": update["removed"],
            "skipped_duplicates": update["skipped_duplicates"],
            "already_processed": update["already_processed"],
            "confirmation_source": reply["confirmation_source"],
        }
        assert update["added"] == 2, update
        assert update["removed"] == 0

        preferences_after = await _call(
            manager, transcript, "profile_0057", "get_active_preferences", {}
        )
        active_after = list(preferences_after["active_preferences"])
        values_after = {str(entry.get("value")) for entry in active_after}
        assert {"noise cancelling", "wireless"} <= values_after, values_after
        assert len(active_after) == len(active_before) + 2

        # Same reply turn id is idempotent: no duplicate memory entry.
        retry = await _call(
            manager, transcript, "profile_0057", "submit_clarification_reply",
            {
                "clarification_id": clarification_id,
                "user_message": "I prefer noise cancelling and wireless.",
                "turn_id": reply_turn_id,
                "confirmed_by_user": True,
                "confirmation_source": "web_ui_explicit_user_reply",
            },
        )
        assert retry["already_processed"] is True
        assert retry["update"]["added"] == 0

        # Read-only profile refuses the write with the real MCP error.
        ro_clarification = await _call(
            manager, transcript, "profile_0049", "request_preference_clarification",
            {"query": "I need earbuds."},
        )
        denied = await _call(
            manager, transcript, "profile_0049", "submit_clarification_reply",
            {
                "clarification_id": ro_clarification["clarification_id"],
                "user_message": "I prefer wireless.",
                "turn_id": f"web-integration-{uuid.uuid4().hex}",
                "confirmed_by_user": True,
                "confirmation_source": "web_ui_explicit_user_reply",
            },
            expect_error=True,
        )
        assert denied["__error__"]["code"] == "memory_writes_disabled", denied["__error__"]

        transcript["summary"] = {
            "servers_ready": True,
            "ready_after_s": transcript["ready_after_s"],
            "catalogue_size": search["catalogue_size"],
            "hit_ids": hit_ids,
            "recommended": recommendation["selected_product_id"],
            "comparison_status": comparison["status"],
            "memory_added": update["added"],
            "active_preferences_after": len(active_after),
            "tools": sorted(EXPECTED_TOOLS),
        }
        return transcript
    finally:
        await manager.stop()


def test_live_bm25_mcp_tools_memory_and_errors(tmp_path: Path) -> None:
    transcript = asyncio.run(_scenario(tmp_path))
    _save(transcript)
    assert TRANSCRIPT_PATH.is_file()
    # The MCP server stderr logs are real files created by the manager.
    log_dir = tmp_path / "logs"
    for profile_id in ("profile_0049", "profile_0057"):
        log_path = log_dir / f"mcp-{profile_id}.log"
        assert log_path.is_file(), log_path
        assert "agentrecx-mcp manifest" in log_path.read_text(encoding="utf-8", errors="replace")
