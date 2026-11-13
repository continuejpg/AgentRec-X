"""One real-Qwen positive corrective recovery with P15 final answer.

Controlled but real-catalogue task:
  user query: "I need a keyboard for my desk."
  task constraints: mechanical + wireless (the controlled task's explicit constraints)
  initial retrieval: the Planner's broad query "keyboard for desk" misses the verified
  candidates; the Executor performs one evidence-driven corrective search; the next
  Qwen FINALIZE passes the existing Gate and P15 builds the final answer.

Run:
  AGENTRECX_QWEN_PLANNER=1 \
  AGENTRECX_QWEN_PLANNER_URL=http://127.0.0.1:18029 \
  .venv/bin/python experiments/p20_metadata/corrective_recovery_live.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p15.explainer import EvidenceExplainer, QwenExplanationClient
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p20_metadata"
QWEN_URL = "http://127.0.0.1:18029"
QUERY = "I need a keyboard for my desk."
HARD = [
    {"kind": "attribute", "value": "mechanical", "source_text": "controlled task constraint"},
    {"kind": "attribute", "value": "wireless", "source_text": "controlled task constraint"},
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENTRECX_QWEN_PLANNER", "1")
    os.environ.setdefault("AGENTRECX_QWEN_PLANNER_URL", QWEN_URL)
    os.environ.setdefault("AGENTRECX_QWEN_PLANNER_MODEL", "Qwen3.5-9B-Base")
    os.environ.setdefault("AGENTRECX_QWEN_PLANNER_TIMEOUT_S", "300")
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    search = configure_catalogue(catalogue)
    task = {
        "task_id": "controlled_mechanical_wireless_keyboard",
        "current_query": QUERY,
        "catalog_namespace": "amazon_electronics_c4",
    }
    history = {"active_preferences": []}
    explainer = EvidenceExplainer(QwenExplanationClient(QWEN_URL, timeout=300.0))
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=QUERY, preferences=[], hard=HARD,
        task_context=task, memory_history=history, explainer=explainer,
        corrective_rag=True,
    )
    planner = HttpPlannerProposer.from_env()
    observation = PlannerObservation(
        step=0, query=QUERY,
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": QUERY, "preferences": []},
    )
    started = time.perf_counter()
    error = None
    try:
        run = run_planner(planner=planner, executor=executor,
                          initial_observation=observation, max_steps=MAX_STEPS)
        status = executor.terminal_status
        selected = executor.selected_candidate_id
        steps = run.as_dict()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        status, selected, steps = "error", None, {"steps": []}
    payload: dict[str, Any] = {
        "schema": "agentrecx.p20.corrective_recovery_live.v1",
        "query": QUERY,
        "controlled_constraints": HARD,
        "search": search,
        "status": status,
        "selected_product_id": selected,
        "actions": [step.get("action") for step in steps.get("steps") or []],
        "planner_steps": steps,
        "corrective_retrieval": {
            "enabled": True,
            "attempts": executor.corrective_attempts,
            "trace": list(executor.corrective_trace),
        },
        "gate": executor.last_decision,
        "final_answer": (executor.last_explanation or {}).get("final_answer"),
        "claim_validation": (executor.last_explanation or {}).get("validation"),
        "explanation_error": executor.explanation_error,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "error": error,
        "notes": [
            "real Qwen Base /plan and /explain; no Rule fallback",
            "controlled task constraints are explicit and task-scoped",
            "corrective retrieval is an Executor recovery after a Gate evidence gap, not a Qwen PlannerAction",
        ],
    }
    out = OUT / "corrective_positive_live.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out), "status": status, "selected": selected,
        "actions": payload["actions"],
        "corrective_attempts": executor.corrective_attempts,
        "corrective_trace": executor.corrective_trace,
        "gate": (executor.last_decision or {}).get("action"),
        "final_answer": payload["final_answer"],
        "latency_ms": payload["latency_ms"], "error": error,
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
