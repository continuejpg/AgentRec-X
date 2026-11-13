"""Real Qwen Agent E2E with the independent verifier in shadow mode.

The existing MCP server, Planner, Executor, Gate, corrector and P15 answer remain the
authority.  The verifier only adds a read-only `shadow_evidence_verifier` field to the
result; it cannot change the final answer, Gate decision or Memory.

Run:
  AGENTRECX_QWEN_PLANNER=1 ...
  .venv/bin/python experiments/p20_metadata/run_shadow_verifier_e2e.py
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
TASKS = [
    "p14e2_lite_0041",
    "p14e2_lite_0042",
    "p14e2_lite_0043",
    "p14e2_lite_0044",
    "p14e2_lite_0049",
    "p14e2_lite_0055",
]


def _params() -> StdioServerParameters:
    args = [
        "-m", "recommendation.mcp.server",
        "--user-key", "p14e2_lite_0049",
        "--memory-fixture", str(FIXTURE),
        "--search-mode", "bge",
        "--metadata-mode", "enriched",
        "--evidence-verifier",
    ]
    env = {
        "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "AGENTRECX_METADATA_SIDECAR": str(SIDECAR),
        "AGENTRECX_QWEN_PLANNER": "1",
        "AGENTRECX_QWEN_PLANNER_URL": QWEN_URL,
        "AGENTRECX_QWEN_PLANNER_MODEL": QWEN_MODEL,
        "AGENTRECX_QWEN_PLANNER_TIMEOUT_S": "300",
        "AGENTRECX_QWEN_EXPLAIN": "1",
        "AGENTRECX_EVIDENCE_VERIFIER": "1",
    }
    return StdioServerParameters(command=PYTHON, args=args, cwd=str(REPO), env=env)


def _load_queries() -> dict[str, str]:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["task_id"]: row["current_query"] for row in rows}


async def main(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = _load_queries()
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema": "agentrecx.p20.shadow_verifier_e2e.v1",
        "model": QWEN_MODEL, "gpu": "AutoDL RTX 4090 24GB",
        "mode": "shadow_only", "tasks": [],
        "startup_and_connect_ms": None,
    }
    async with Client(_params()) as client:
        payload["startup_and_connect_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        tools = [tool.name for tool in (await client.list_tools()).tools]
        payload["tools"] = tools
        for task_id in TASKS:
            call_started = time.perf_counter()
            try:
                result = await client.call_tool("run_agent_turn",
                                                {"query": queries[task_id]})
                sc = result.structured_content or {}
                item = {
                    "task_id": task_id,
                    "query": queries[task_id],
                    "is_error": bool(result.is_error),
                    "client_latency_ms": round((time.perf_counter() - call_started) * 1000.0, 2),
                    "status": sc.get("status"),
                    "selected_product_id": sc.get("selected_product_id"),
                    "planner_actions": [a.get("action") for a in (sc.get("planner") or {}).get("actions") or []],
                    "gate": (sc.get("gate") or {}).get("action"),
                    "parsed_hard_constraints": sc.get("parsed_hard_constraints"),
                    "corrective_retrieval": sc.get("corrective_retrieval"),
                    "claim_validation": (sc.get("rag") or {}).get("claim_validation"),
                    "final_answer": (sc.get("rag") or {}).get("final_answer"),
                    "shadow_evidence_verifier": sc.get("shadow_evidence_verifier"),
                    "latency_ms": sc.get("latency_ms"),
                }
            except Exception as exc:  # noqa: BLE001
                item = {"task_id": task_id, "query": queries[task_id], "is_error": True,
                        "exception": f"{type(exc).__name__}: {exc}"}
            payload["tasks"].append(item)
            print(json.dumps({
                "task": task_id, "status": item.get("status"),
                "selected": item.get("selected_product_id"),
                "gate": item.get("gate"),
                "shadow_status": (item.get("shadow_evidence_verifier") or {}).get("status"),
                "verdicts": [(v.get("claim_id"), v.get("verdict")) for v in
                              ((item.get("shadow_evidence_verifier") or {}).get("verdicts") or [])],
                "error": item.get("is_error"),
            }, ensure_ascii=False), flush=True)
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    payload["verdict_counts"] = {}
    for item in payload["tasks"]:
        for verdict in (item.get("shadow_evidence_verifier") or {}).get("verdicts") or []:
            key = str(verdict.get("verdict"))
            payload["verdict_counts"][key] = payload["verdict_counts"].get(key, 0) + 1
    payload["notes"] = [
        "shadow only: verifier never changes Gate, final answer or Memory",
        "same Qwen checkpoint, independent verifier prompt/context",
        "P15 deterministic validator is retained and reported separately",
    ]
    (OUT / "shadow_verifier_e2e.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT / "shadow_verifier_e2e.json"),
                      "tasks": len(payload["tasks"]), "verdict_counts": payload["verdict_counts"],
                      "elapsed_ms": payload["elapsed_ms"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main, argparse.Namespace()))
