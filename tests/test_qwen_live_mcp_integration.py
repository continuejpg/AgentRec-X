"""Contract integration test for the optional P20 run_agent_turn MCP tool.

A local HTTP double (NOT Qwen) returns deterministic PlannerActions.  The test proves that the
real MCP server + real executor/catalogue/BM25/Gate path executes those actions, exposes the
conditional 8th tool and forwards real progress notifications.  It is gated because it loads the
real catalogue:

    AGENTRECX_RUN_QWEN_MCP_INTEGRATION=1 .venv/bin/python -m pytest -q \
        tests/test_qwen_live_mcp_integration.py
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from web_demo.config import REPO, Settings
from web_demo.mcp_manager import MCPManager, MCPManagerError

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("AGENTRECX_RUN_QWEN_MCP_INTEGRATION") != "1",
        reason="real-catalogue MCP contract test; set AGENTRECX_RUN_QWEN_MCP_INTEGRATION=1",
    ),
]
RUNS = REPO / "runs/web_demo/qwen_live"
QUERY = "I need earbuds I can use while working out."


class _FakePlanner(BaseHTTPRequestHandler):
    """Deterministic PlannerAction test double; never claimed to be Qwen."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        if self.path == "/explain":
            self.send_error(404)
            return
        if self.path != "/plan":
            self.send_error(404)
            return
        step = int(payload.get("step") or 0)
        query = str(payload.get("query") or QUERY)
        candidate_count = int(payload.get("candidate_count") or 0)
        reranked = bool(payload.get("reranked"))
        if candidate_count <= 0:
            action = "RETRIEVE"
            arguments = {"query": query}
        elif not reranked:
            action = "RERANK"
            arguments = {}
        else:
            action = "FINALIZE"
            arguments = {}
        response = {
            "payload": {"action": action, "arguments": arguments, "rationale": "contract test double"},
            "text": json.dumps({"action": action, "step": step}),
            "errors": [],
            "latency_ms": 7.5,
            "input_tokens": 100,
            "output_tokens": 10,
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


async def _wait_ready(manager: MCPManager, timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if manager.ready():
            return
        if manager.status() == "error":
            raise AssertionError(f"MCP startup failed: {manager.health_servers()}")
        await asyncio.sleep(0.5)
    raise AssertionError(f"MCP not ready after {timeout_s}s: {manager.health_servers()}")


async def _scenario(server: ThreadingHTTPServer, endpoint: str) -> dict[str, Any]:
    settings = Settings(
        search_mode="bm25",
        allow_memory_writes=False,
        qwen_live_enabled=True,
        qwen_model_endpoint=endpoint,
        qwen_explain=False,
        qwen_turn_timeout_s=300.0,
        mcp_timeout_s=300.0,
    )
    manager = MCPManager(settings)
    progress: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def on_progress(current: float, total: float | None, text: str | None) -> None:
        del current, total
        try:
            event = json.loads(text or "")
        except (TypeError, ValueError):
            return
        if isinstance(event, dict):
            progress.append(event)

    try:
        manager.start()
        await _wait_ready(manager)
        runtime = manager.runtimes["profile_0049"]
        tool_names = [str(tool["name"]) for tool in (runtime.tools or [])]
        assert len(tool_names) == 8, tool_names
        assert "run_agent_turn" in tool_names
        result = await manager.call_tool(
            "profile_0049", "run_agent_turn", {"query": QUERY},
            timeout_s=300.0, progress_callback=on_progress,
        )
        transcript = {
            "schema": "agentrecx.p20.qwen_mcp_contract.v1",
            "model": "HTTP test double (not Qwen)",
            "query": QUERY,
            "tools": tool_names,
            "tool_count": len(tool_names),
            "runtime_profile": runtime.profile_id,
            "terminal": {
                "status": result.get("status"),
                "selected_product_id": result.get("selected_product_id"),
                "candidate_ids": result.get("candidate_ids"),
            },
            "planner": result.get("planner"),
            "gate": result.get("gate"),
            "evidence_bundle_present": bool(result.get("evidence_bundle")),
            "progress_events": progress,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "mcp_contract_transcript.json").write_text(
            json.dumps(transcript, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return transcript
    finally:
        await manager.stop()


def test_run_agent_turn_contract_with_real_catalogue(tmp_path: Path) -> None:
    del tmp_path
    handler = type("Handler", (_FakePlanner,), {"requests": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        transcript = asyncio.run(_scenario(server, endpoint))
    finally:
        server.shutdown()
        server.server_close()

    planner = transcript["planner"] or {}
    actions = [item.get("action") for item in planner.get("actions") or []]
    assert planner.get("kind") == "qwen_base"
    assert planner.get("calls", 0) >= 2
    assert "RETRIEVE" in actions
    assert transcript["terminal"]["status"] in {
        "finalized", "awaiting_clarification", "no_verified_candidate_in_scope",
        "budget_exhausted",
    }
    assert transcript["progress_events"], "MCP progress notifications were not delivered"
    progress_tools = {str(item.get("tool")) for item in transcript["progress_events"]}
    assert "qwen_planner" in progress_tools
    qwen_rows = [item for item in transcript["progress_events"] if item.get("tool") == "qwen_planner"]
    assert any(item.get("status") == "running" for item in qwen_rows)
    assert any(item.get("status") == "ok" for item in qwen_rows)
    assert not any(item.get("status") == "running" for item in qwen_rows[-1:]), qwen_rows[-1:]
    assert any(item.startswith(("retrieve_", "rerank_", "finalize_")) for item in progress_tools)
    # No hidden Rule fallback: the model's action sequence is what the executor consumed.
    assert transcript["planner"]["calls"] == len(planner.get("actions") or [])
