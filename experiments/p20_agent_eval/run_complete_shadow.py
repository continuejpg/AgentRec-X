"""Run the Complete Shadow Agent through the existing MCP server on a representative subset.

One server process per task is used so the correct task-scoped Memory fixture and user key are
loaded.  The Agent, Gate, corrective retrieval and P15 validator are unchanged; the verifier
runs only in shadow mode.
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

from experiments.p14c0.run_dev import _evaluate
from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p20_metadata.corrective_eval import _is_success
from experiments.p20_agent_eval.task_suite import OUT as RUNS

REPO = Path(__file__).resolve().parents[2]
PYTHON = str(REPO / ".venv/bin/python")
SIDECAR = REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"
QWEN_URL = "http://127.0.0.1:18029"
QWEN_MODEL = "Qwen3.5-9B-Base"

# Representative C subset: recommend, memory conflict, ambiguous, no-match, insufficient.
C_IDS = [
    "p14e2_lite_0041",
    "p14e2_lite_0042",
    "p14e2_lite_0049",
    "p14e2_lite_0056",
    "p14e2_lite_0060",
    "p14b3_eval100_0001",
    "p14b3_eval100_0073",
    "p14b3_eval100_0089",
    "p14b3_eval100_0093",
    "p14b3_eval100_0109",
]


def _params(task: dict[str, Any]) -> StdioServerParameters:
    args = [
        "-m", "recommendation.mcp.server",
        "--user-key", task["user_id"],
        "--memory-fixture", task["memory_fixture"],
        "--search-mode", "bge",
        "--metadata-mode", "enriched",
        "--corrective-rag",
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


def _hard_violation(gate: dict[str, Any]) -> bool:
    return ((gate or {}).get("hard_constraint_status") or {}).get("hard_constraints") == "violated"


def _claim_metrics(record: dict[str, Any]) -> dict[str, Any]:
    rag = record.get("rag") or {}
    validation = rag.get("claim_validation") or {}
    valid = validation.get("valid_claims") or []
    rejected = validation.get("rejected_claims") or []
    shadow = record.get("shadow_evidence_verifier") or {}
    verdicts = {item.get("claim_id"): item.get("verdict") for item in shadow.get("verdicts") or []}
    p15_ids = {item.get("claim_id") for item in valid}
    rejected_ids = {item.get("claim_id") for item in rejected}
    supported = sum(1 for claim_id in p15_ids if verdicts.get(claim_id) == "SUPPORTED")
    unsupported = sum(1 for claim_id in p15_ids if verdicts.get(claim_id) not in (None, "SUPPORTED"))
    disagreements = [
        claim_id for claim_id in p15_ids
        if verdicts.get(claim_id) not in (None, "SUPPORTED")
    ]
    rejected_with_verdict = [
        claim_id for claim_id in rejected_ids if verdicts.get(claim_id) not in (None, "SUPPORTED")
    ]
    return {
        "claims_total": len(valid) + len(rejected),
        "p15_valid": len(valid),
        "p15_rejected": len(rejected),
        "verifier_supported": supported,
        "verifier_unsupported": unsupported,
        "p15_valid_but_verifier_not_supported": disagreements,
        "p15_rejected_and_verifier_not_supported": rejected_with_verdict,
        "final_answer_evidence_support": bool(valid) and unsupported == 0,
        "verifier_verdict_counts": {
            verdict: sum(1 for value in verdicts.values() if value == verdict)
            for verdict in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
        },
        "verifier_errors": shadow.get("errors") or [],
    }


async def one_task(client: Client, task: dict[str, Any], catalogue: ElectronicsCatalogue) -> dict[str, Any]:
    started = time.perf_counter()
    result = await client.call_tool("run_agent_turn", {"query": task["current_query"]})
    sc = result.structured_content or {}
    status = sc.get("status")
    selected = sc.get("selected_product_id")
    success = _is_success(task, status, selected, catalogue, task.get("scored_evaluator"))
    error = None
    if result.is_error or not sc:
        error = "MCP run_agent_turn returned is_error without structured content"
        blocks = [getattr(block, "text", "") for block in result.content]
        if blocks:
            error = f"{error}: {' '.join(blocks)[:800]}"
    record = {
        "task_id": task["task_id"], "query": task["current_query"],
        "expected": task["expected"], "category": task["category"],
        "label_source": task["label_source"], "status": status,
        "selected_product_id": selected, "success": success,
        "hard_constraint_violation": _hard_violation(sc.get("gate") or {}),
        "planner": sc.get("planner"),
        "planner_actions": [item.get("action") for item in (sc.get("planner") or {}).get("actions") or []],
        "planner_calls": (sc.get("planner") or {}).get("calls"),
        "search_calls": ((sc.get("search") or {}).get("retrievals")
                         if isinstance(sc.get("search"), dict) else None),
        "gate": sc.get("gate"),
        "corrective_retrieval": sc.get("corrective_retrieval"),
        "claim_validation": (sc.get("rag") or {}).get("claim_validation"),
        "final_answer": (sc.get("rag") or {}).get("final_answer"),
        "evidence_bundle": sc.get("evidence_bundle"),
        "shadow_evidence_verifier": sc.get("shadow_evidence_verifier"),
        "rag": sc.get("rag"),
        "client_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "agent_latency_ms": sc.get("latency_ms"),
        "is_error": bool(result.is_error),
        "error": error,
    }
    record["claim_metrics"] = _claim_metrics(record)
    return record


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    suite = json.loads((RUNS / "task_suite.json").read_text(encoding="utf-8"))
    by_id = {task["task_id"]: task for task in suite["tasks"]}
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    records: list[dict[str, Any]] = []
    for task_id in C_IDS:
        task = by_id[task_id]
        started = time.perf_counter()
        try:
            async with Client(_params(task)) as client:
                record = await one_task(client, task, catalogue)
            record["server_startup_and_call_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        except Exception as exc:  # noqa: BLE001
            record = {"task_id": task_id, "query": task["current_query"],
                      "expected": task["expected"], "status": "error", "selected_product_id": None,
                      "success": False, "error": f"{type(exc).__name__}: {exc}",
                      "server_startup_and_call_ms": round((time.perf_counter() - started) * 1000.0, 2)}
        records.append(record)
        print(json.dumps({
            "task": task_id, "status": record.get("status"),
            "selected": record.get("selected_product_id"), "success": record.get("success"),
            "claims": (record.get("claim_metrics") or {}).get("claims_total"),
            "verifier": (record.get("claim_metrics") or {}).get("verifier_verdict_counts"),
            "shadow_status": (record.get("shadow_evidence_verifier") or {}).get("status"),
            "error": record.get("error"),
        }, ensure_ascii=False), flush=True)

    task_metrics = {
        "tasks": len(records),
        "success": sum(1 for r in records if r.get("success")),
        "recommend_tasks": sum(1 for r in records if r.get("expected") == "recommend"),
        "recommend_success": sum(1 for r in records if r.get("expected") == "recommend" and r.get("success")),
        "nonrecommend_tasks": sum(1 for r in records if r.get("expected") != "recommend"),
        "nonrecommend_success": sum(1 for r in records if r.get("expected") != "recommend" and r.get("success")),
        "hard_constraint_violations": sum(1 for r in records if r.get("hard_constraint_violation")),
        "corrective_attempts": sum(((r.get("corrective_retrieval") or {}).get("attempts") or 0) for r in records),
        "corrective_recovered": sum(
            1 for r in records if ((r.get("corrective_retrieval") or {}).get("attempts") or 0)
            and r.get("status") == "finalized" and r.get("selected_product_id")),
        "claims_total": sum((r.get("claim_metrics") or {}).get("claims_total", 0) for r in records),
        "p15_valid": sum((r.get("claim_metrics") or {}).get("p15_valid", 0) for r in records),
        "p15_rejected": sum((r.get("claim_metrics") or {}).get("p15_rejected", 0) for r in records),
        "verifier_supported": sum((r.get("claim_metrics") or {}).get("verifier_supported", 0) for r in records),
        "verifier_unsupported": sum((r.get("claim_metrics") or {}).get("verifier_unsupported", 0) for r in records),
        "final_answer_evidence_support": sum(
            1 for r in records if (r.get("claim_metrics") or {}).get("final_answer_evidence_support")),
        "shadow_only_no_output_change": True,
        "planner_input_tokens": sum(int((r.get("planner") or {}).get("input_tokens") or 0) for r in records),
        "planner_output_tokens": sum(int((r.get("planner") or {}).get("output_tokens") or 0) for r in records),
        "planner_calls": sum(int((r.get("planner") or {}).get("calls") or 0) for r in records),
        "search_calls": sum(int(r.get("search_calls") or 0) for r in records),
        "explain_input_tokens": sum(int(((r.get("rag") or {}).get("model_stats") or {}).get("input_tokens") or 0)
                                    for r in records),
        "explain_output_tokens": sum(int(((r.get("rag") or {}).get("model_stats") or {}).get("output_tokens") or 0)
                                     for r in records),
        "verifier_input_tokens": sum(int(((r.get("shadow_evidence_verifier") or {}).get("model_stats") or {}).get("input_tokens") or 0)
                                      for r in records),
        "verifier_output_tokens": sum(int(((r.get("shadow_evidence_verifier") or {}).get("model_stats") or {}).get("output_tokens") or 0)
                                       for r in records),
        "verifier_latency_ms": round(sum(float(((r.get("shadow_evidence_verifier") or {}).get("model_stats") or {}).get("latency_ms") or 0)
                                          for r in records), 2),
        "total_agent_latency_ms": sum(float(r.get("agent_latency_ms") or 0) for r in records),
        "total_client_latency_ms": sum(float(r.get("server_startup_and_call_ms") or 0) for r in records),
    }
    payload = {
        "schema": "agentrecx.p20.complete_shadow_agent.v1",
        "task_subset": C_IDS,
        "label_status": suite["label_status"],
        "config": {"metadata_mode": "enriched", "corrective_rag": True,
                   "verifier": "shadow", "planner": "qwen"},
        "tasks": records, "metrics": task_metrics,
    }
    (RUNS / "complete_shadow.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(task_metrics, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
