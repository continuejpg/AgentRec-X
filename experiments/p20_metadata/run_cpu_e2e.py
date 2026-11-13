"""CPU MCP Before/After runner for the optional metadata sidecar.

This is an experiment driver, not a new Agent runtime.  It starts the existing MCP server
with either ``--metadata-mode original`` or ``--metadata-mode enriched`` and calls the
existing tools.  Qwen Live is intentionally not enabled here; it needs the separate
Qwen3.5-9B /plan service.
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

TASKS = [
    ("t1_wireless_earbuds", "I need earbuds I can use while working out."),
    ("t2_wired_headphones", "I need wired headphones for office calls."),
    ("t3_wireless_keyboard", "I need a wireless keyboard for my desktop."),
    ("t4_noise_cancelling_commute", "I need noise cancelling headphones for commuting."),
]


def _params(mode: str) -> StdioServerParameters:
    args = [
        "-m", "recommendation.mcp.server",
        "--user-key", "p14e2_lite_0049",
        "--memory-fixture", str(FIXTURE),
        "--search-mode", "bge",
        "--metadata-mode", mode,
    ]
    env = {"PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "AGENTRECX_METADATA_SIDECAR": str(SIDECAR)}
    return StdioServerParameters(command=PYTHON, args=args, cwd=str(REPO), env=env)


async def _call(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await client.call_tool(name, arguments)
        return {
            "tool": name,
            "arguments": arguments,
            "is_error": bool(result.is_error),
            "content": [getattr(block, "text", None) for block in result.content],
            "structured_content": result.structured_content,
            "client_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    except Exception as exc:  # noqa: BLE001 - the driver records real failures
        return {
            "tool": name, "arguments": arguments, "is_error": True,
            "content": [], "structured_content": None,
            "client_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "exception": f"{type(exc).__name__}: {exc}",
        }


async def run(mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema": "agentrecx.p20.metadata_cpu_before_after.v1",
        "mode": mode,
        "planner": "rule",
        "qwen_used": False,
        "metadata_sidecar": str(SIDECAR),
        "server_startup_and_connect_ms": None,
        "calls": [],
    }
    async with Client(_params(mode)) as client:
        payload["server_startup_and_connect_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        payload["protocol_version"] = client.protocol_version
        payload["tools"] = [tool.name for tool in (await client.list_tools()).tools]
        # A known JVC earbud: C4 has text; enriched adds brand/wired and P15 metadata spans.
        payload["calls"].append(await _call(client, "get_product_info", {"product_ref": "B08DKG3P4R"}))
        # A known keyboard: C4 has "Mechanical Feeling"; upstream details has more fields.
        payload["calls"].append(await _call(client, "get_product_info", {"product_ref": "B09H2JQD48"}))
        for task_id, query in TASKS:
            payload["calls"].append(await _call(
                client, "recommend_products", {"query": query}))
            payload["calls"][-1]["task_id"] = task_id
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "enriched"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = args.out or OUT / f"cpu_{args.mode}.json"
    payload = anyio.run(run, args.mode)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "out": str(out),
        "startup_ms": payload.get("server_startup_and_connect_ms"),
        "calls": len(payload["calls"]),
        "errors": [call["tool"] for call in payload["calls"] if call.get("is_error")],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
