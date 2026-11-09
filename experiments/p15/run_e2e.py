"""P15 E2E: existing Agent (Rule planner + shared tools/BGE) + Evidence Bundle + Qwen Base explanation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor, prepare_hard
from experiments.p14c0.run_dev import _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p15.explainer import EvidenceExplainer, QwenExplanationClient
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p15"
TASKS = (
    "p14e2_lite_0042", "p14e2_lite_0043", "p14e2_lite_0044", "p14e2_lite_0045",
    "p14e2_lite_0046", "p14e2_lite_0049", "p14e2_lite_0050", "p14e2_lite_0053",
    "p14e2_lite_0054", "p14e2_lite_0055", "p14e2_lite_0056", "p14e2_lite_0057",
    "p14e2_lite_0060",
)


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def non_final_answer(status: str, question: str | None, gate: dict | None) -> str:
    if status == "awaiting_clarification":
        return ("I need one clarification before recommending a product: " + (question or "could you clarify your needs?")
                + "\nI have not produced a product recommendation because the existing Decision Gate requires clarification.")
    reason = (gate or {}).get("reason") or "no verified candidate satisfied the current constraints"
    return ("I can't make a verified product recommendation for this request. Reason: " + str(reason)
            + "\nI have not invented a product or product fact to fill the gap.")


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    search_config = configure_catalogue(catalogue)
    if search_config["status"] != "active":
        raise RuntimeError(f"BGE must be active for P15: {search_config}")
    tasks = {row["task_id"]: row for row in load(DATA / "dev.agent.jsonl")}
    evaluator = {row["task_id"]: row for row in load(DATA / "dev.evaluator.jsonl")}
    explainer = EvidenceExplainer(QwenExplanationClient("http://127.0.0.1:18009"))
    (OUT / "e2e").mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in TASKS:
        task = tasks[task_id]
        preferences, history = _memory_read(catalogue, task)
        executor = ElectronicsPlannerExecutor(
            catalogue=catalogue, current_query=task["current_query"], preferences=preferences,
            hard=prepare_hard(task["current_query"]), explainer=explainer,
            task_context=task, memory_history=history,
        )
        observation = PlannerObservation(
            step=0, query=task["current_query"],
            m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                      "preferences": preferences},
        )
        started = time.perf_counter()
        error = None
        try:
            planner_run = run_planner(planner=P14ARulePlanner(), executor=executor,
                                      initial_observation=observation, max_steps=MAX_STEPS)
            steps = planner_run.as_dict()
            status = executor.terminal_status
            if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
                status = "budget_exhausted"
                executor.terminal_status = status
        except Exception as exc:  # record honestly; no RAG output on a failed run
            error = f"{type(exc).__name__}: {exc}"
            steps = {"steps": [], "final_ids": [], "termination": "error"}
            status = "error"
        explanation = executor.last_explanation
        if explanation is None:
            explanation = {
                "status": "not_applicable",
                "reason": status,
                "final_answer": non_final_answer(status, executor.ask_question, executor.last_decision),
                "bundle": None,
                "validation": {"status": "not_applicable", "valid_claims": [], "rejected_claims": []},
            }
            raw_answer = None
        else:
            raw_answer = explanation.get("raw_output")
        record = {
            "schema": "agentrecx.p15.e2e_trajectory.v1",
            "task_id": task_id,
            "split": "dev",
            "task_type": evaluator[task_id].get("task_type"),
            "expected_decision": evaluator[task_id].get("expected_decision"),
            "current_query": task["current_query"],
            "search_backend": search_config,
            "search_mode": executor.last_search.get("score_kind") if executor.last_search else None,
            "memory_preferences": preferences,
            "memory_history": history,
            "terminal": {"status": status, "selected_candidate_id": executor.selected_candidate_id,
                         "ask_question": executor.ask_question},
            "gate": executor.last_decision,
            "planner_run": steps,
            "tool_calls": executor.tool_calls,
            "rag_explanation": explanation,
            "qwen_raw_answer": raw_answer,
            "final_user_answer": explanation.get("final_answer"),
            "error": error,
            "explanation_error": executor.explanation_error,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "model_stats": (explanation.get("model_stats") if explanation else None),
        }
        (OUT / "e2e" / f"{task_id}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
        records.append(record)
        print(task_id, status, "final_answer" if explanation.get("bundle") else "no_bundle",
              (explanation.get("model_stats") or {}).get("latency_ms") if explanation else None, flush=True)
    (OUT / "e2e_trajectories.json").write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    print(f"wrote {len(records)} trajectories to {OUT / 'e2e'}")


if __name__ == "__main__":
    main()
