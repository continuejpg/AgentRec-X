"""P17 independent MCP stdio client E2E.

Run on the local CPU host from the project environment:
  .venv/bin/python -m experiments.p17.mcp_client_e2e
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p17"
PYTHON = str(REPO / ".venv/bin/python")
FIXTURE = REPO / "data/p14e2/benchmark/dev.agent.jsonl"


def params(user_key: str, *, search_mode: str, writes: bool, ttl: float = 1800.0) -> StdioServerParameters:
    args = ["-m", "recommendation.mcp.server", "--user-key", user_key,
            "--memory-fixture", str(FIXTURE), "--search-mode", search_mode,
            "--clarification-ttl-seconds", str(ttl)]
    if writes:
        args.append("--allow-memory-writes")
    return StdioServerParameters(
        command=PYTHON, args=args, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )


async def call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await client.call_tool(name, arguments)
        latency = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "tool": name, "arguments": arguments, "is_error": bool(result.is_error),
            "content": [getattr(block, "text", None) for block in result.content],
            "structured_content": result.structured_content,
            "client_latency_ms": latency,
        }
    except Exception as exc:
        return {
            "tool": name, "arguments": arguments, "is_error": True,
            "content": [], "structured_content": None,
            "client_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "exception": f"{type(exc).__name__}: {exc}",
        }


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transcript: dict[str, Any] = {"schema": "agentrecx.p17.mcp_client_e2e.v1",
                                  "sdk": "mcp==2.3.0", "transport": "stdio",
                                  "tool_calls": [], "servers": []}
    # Server A: trusted local user 0049, BGE, writes explicitly enabled for this test.
    started = time.perf_counter()
    transport_a = params("p14e2_lite_0049", search_mode="bge", writes=True)
    async with Client(transport_a) as client:
        start_ms = round((time.perf_counter() - started) * 1000.0, 2)
        discovery_started = time.perf_counter()
        listing = await client.list_tools()
        discovery_ms = round((time.perf_counter() - discovery_started) * 1000.0, 2)
        transcript["servers"].append({
            "label": "A_user_0049_bge_writes",
            "server_info": client.server_info.model_dump(mode="json") if client.server_info else None,
            "protocol_version": client.protocol_version,
            "startup_and_connect_ms": start_ms,
            "tool_discovery_ms": discovery_ms,
            "tools": [{"name": tool.name, "description": tool.description,
                       "input_schema": tool.input_schema} for tool in listing.tools],
        })
        transcript["tool_calls"].append(await call(client, "search_products",
                                                  {"query": "wireless earbuds for working out", "top_k": 3}))
        transcript["tool_calls"].append(await call(client, "get_product_info",
                                                  {"product_ref": "B004K6R43G"}))
        transcript["tool_calls"].append(await call(client, "recommend_products",
                                                  {"query": "I need earbuds I can use while working out."}))
        transcript["tool_calls"].append(await call(client, "get_active_preferences", {}))
        transcript["tool_calls"].append(await call(client, "compare_products",
                                                  {"product_refs": ["B004K6R43G", "B00008VF63"]}))
        clarification = await call(client, "request_preference_clarification",
                                   {"query": "I need earbuds I can use while working out."})
        transcript["tool_calls"].append(clarification)
        handle = (clarification.get("structured_content") or {}).get("clarification_id")
        transcript["clarification_id_for_test"] = handle
        if handle:
            transcript["tool_calls"].append(await call(
                client, "submit_clarification_reply",
                {"clarification_id": handle, "user_message": "I prefer noise cancelling.",
                 "turn_id": "p17-test-turn-1", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
            transcript["tool_calls"].append(await call(client, "get_active_preferences", {}))
            transcript["tool_calls"].append(await call(
                client, "submit_clarification_reply",
                {"clarification_id": handle, "user_message": "I prefer noise cancelling.",
                 "turn_id": "p17-test-turn-1", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
            transcript["tool_calls"].append(await call(
                client, "submit_clarification_reply",
                {"clarification_id": "clar_invalid_handle", "user_message": "I prefer wireless.",
                 "turn_id": "p17-bad-handle", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
        # A new handle submitted without explicit confirmation must be refused (read-only by default
        # for model-generated text; only the labelled test confirmation is accepted here).
        unconfirmed = await call(client, "request_preference_clarification",
                                 {"query": "I need earbuds I can use while working out."})
        transcript["tool_calls"].append(unconfirmed)
        unconfirmed_id = (unconfirmed.get("structured_content") or {}).get("clarification_id")
        if unconfirmed_id:
            transcript["tool_calls"].append(await call(
                client, "submit_clarification_reply",
                {"clarification_id": unconfirmed_id, "user_message": "I prefer wireless.",
                 "turn_id": "p17-unconfirmed", "confirmed_by_user": False,
                 "confirmation_source": None}))
        transcript["tool_calls"].append(await call(client, "unknown_tool", {}))
        # Failure/boundary cases on the same real server.
        transcript["tool_calls"].append(await call(client, "get_product_info", {"product_ref": "B000NOTREAL"}))
        transcript["tool_calls"].append(await call(client, "get_product_info", {"product_ref": "sports:123"}))
        transcript["tool_calls"].append(await call(client, "compare_products",
                                                  {"product_refs": ["B004K6R43G", "B00008VF63", "B004CRRRB2"]}))
        transcript["tool_calls"].append(await call(client, "compare_products",
                                                  {"product_refs": ["B004K6R43G", "B004K6R43G"]}))
        transcript["tool_calls"].append(await call(client, "compare_products",
                                                  {"product_refs": ["B004K6R43G", "B00008VF63",
                                                                    "B004CRRRB2", "B08GFW1ZNG", "B0BLSQ5T75", "B0BQ38B3NC"]}))
        transcript["tool_calls"].append(await call(client, "search_products",
                                                  {"query": "mouse", "top_k": 999}))
    # Server B: different trusted user, BM25, writes disabled. Shows server-side binding and default safe mode.
    started = time.perf_counter()
    transport_b = params("p14e2_lite_0056", search_mode="bm25", writes=False, ttl=2.0)
    async with Client(transport_b) as client_b:
        transcript["servers"].append({
            "label": "B_user_0056_bm25_writes_disabled",
            "server_info": client_b.server_info.model_dump(mode="json") if client_b.server_info else None,
            "protocol_version": client_b.protocol_version,
            "startup_and_connect_ms": round((time.perf_counter() - started) * 1000.0, 2),
        })
        transcript["tool_calls"].append(await call(client_b, "get_active_preferences", {}))
        transcript["tool_calls"].append(await call(client_b, "search_products", {"query": "wireless keyboard", "top_k": 2}))
        transcript["tool_calls"].append(await call(client_b, "compare_products",
                                                  {"product_refs": ["B0EWHGDVC", "B09828X5WY"]}))
        handle_b = await call(client_b, "request_preference_clarification", {"query": "I need a keyboard."})
        transcript["tool_calls"].append(handle_b)
        hb = (handle_b.get("structured_content") or {}).get("clarification_id")
        if hb:
            transcript["tool_calls"].append(await call(
                client_b, "submit_clarification_reply",
                {"clarification_id": hb, "user_message": "I prefer wireless.",
                 "turn_id": "p17-b-write-test", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
            await anyio.sleep(2.2)
            transcript["tool_calls"].append(await call(
                client_b, "submit_clarification_reply",
                {"clarification_id": hb, "user_message": "I prefer wireless.",
                 "turn_id": "p17-b-expired-test", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
        if handle:
            transcript["tool_calls"].append(await call(
                client_b, "submit_clarification_reply",
                {"clarification_id": handle, "user_message": "cross-user handle attempt.",
                 "turn_id": "p17-cross-user", "confirmed_by_user": True,
                 "confirmation_source": "explicit_test_user_reply"}))
    (OUT / "client_transcript.json").write_text(json.dumps(transcript, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps({
        "schema": transcript["schema"],
        "servers": len(transcript["servers"]),
        "tool_calls": len(transcript["tool_calls"]),
        "tools": [t["name"] for t in transcript["servers"][0]["tools"]],
        "errors": [call["tool"] for call in transcript["tool_calls"] if call.get("is_error")],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    anyio.run(main)
