"""Real Qwen Live MCP run for the metadata-enriched catalogue.

Starts the existing MCP server with the optional 8th ``run_agent_turn`` tool, points it
at the local Qwen3.5-9B Base GGUF /plan+/explain adapter, and records the real model
PlannerActions, candidate IDs, Gate, Evidence, and final answer.  This is an experiment
driver; the production Agent code is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

REPO = Path(__file__).resolve().parents[2]
PYTHON = str(REPO / ".venv/bin/python")
FIXTURE = REPO / "data/p14e2/benchmark/dev.agent.jsonl"
SIDECAR = REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"
OUT = REPO / "runs/p20_metadata"
QWEN_URL = "http://127.0.0.1:18029"
QWEN_MODEL = "Qwen3.5-9B-Base"


def _params(mode: str) -> StdioServerParameters:
    args = [
        "-m", "recommendation.mcp.server",
        "--user-key", "p14e2_lite_0049",
        "--memory-fixture", str(FIXTURE),
        "--search-mode", "bge",
        "--metadata-mode", mode,
    ]
    env = {
        "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "AGENTRECX_METADATA_SIDECAR": str(SIDECAR),
        "AGENTRECX_QWEN_PLANNER": "1",
        "AGENTRECX_QWEN_PLANNER_URL": QWEN_URL,
        "AGENTRECX_QWEN_PLANNER_MODEL": QWEN_MODEL,
        "AGENTRECX_QWEN_PLANNER_TIMEOUT_S": "300",
        "AGENTRECX_QWEN_EXPLAIN": "1",
    }
    return StdioServerParameters(command=PYTHON, args=args, cwd=str(REPO), env=env)


async def _call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await client.call_tool(name, arguments)
        return {
            "tool": name, "arguments": arguments, "is_error": bool(result.is_error),
            "content": [getattr(block, "text", None) for block in result.content],
            "structured_content": result.structured_content,
            "client_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"tool": name, "arguments": arguments, "is_error": True,
                "content": [], "structured_content": None,
                "client_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "exception": f"{type(exc).__name__}: {exc}"}


async def run(mode: str, query: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "agentrecx.p20.qwen_live.v1",
        "mode": mode,
        "qwen_url": QWEN_URL,
        "query": query,
        "qwen_used": True,
        "calls": [],
    }
    started = time.perf_counter()
    async with Client(_params(mode)) as client:
        payload["startup_and_connect_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        payload["tools"] = [tool.name for tool in (await client.list_tools()).tools]
        payload["calls"].append(await _call(client, "run_agent_turn", {"query": query}))
        # Always read back the selected product metadata/evidence via the ordinary tool.
        result = payload["calls"][0].get("structured_content") or {}
        selected = result.get("selected_product_id")
        if selected:
            payload["calls"].append(await _call(client, "get_product_info", {"product_ref": selected}))
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "enriched"), required=True)
    parser.add_argument("--query", default="I need earbuds I can use while working out.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = args.out or OUT / f"qwen_live_{args.mode}.json"
    payload = anyio.run(run, args.mode, args.query)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    first = payload["calls"][0]
    sc = first.get("structured_content") or {}
    print(json.dumps({
        "mode": args.mode, "out": str(out),
        "is_error": first.get("is_error"),
        "status": sc.get("status"),
        "planner": sc.get("planner"),
        "selected": sc.get("selected_product_id"),
        "gate": (sc.get("gate") or {}).get("action"),
        "final_answer": (sc.get("rag") or {}).get("final_answer"),
        "elapsed_ms": payload["elapsed_ms"],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
