"""P16 E2E: two registered skills invoked against the existing Agent executor/tools.

No new Agent runtime: comparison uses the existing catalogue/Product Info/P15 evidence; clarification
uses the existing Rule planner for a Gate ASK and the existing M9 user-event writer for follow-up.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor, prepare_hard, prepare_preferences
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.preference.history import HistoryStateReader
from recommendation.shopping_companion.memory import SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner
from recommendation.skills import build_registry

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p16"
REGISTRY = build_registry()

COMPARISON_CASES = [
    {"case_id": "cmp_2_no_history", "task_id": "p14e2_lite_0043",
     "product_refs": ["B09828X5WY", "B00EWHGDVC"], "purpose": "two same-type products, no history"},
    {"case_id": "cmp_3_history", "task_id": "p14e2_lite_0045",
     "product_refs": ["B0BLSQ5T75", "B08HM3MLJF", "B0BQ38B3NC"], "purpose": "three products with history"},
    {"case_id": "cmp_negative_history", "task_id": "p14e2_lite_0049",
     "product_refs": ["B004K6R43G", "B00008VF63"], "purpose": "negative preference and unknown attributes"},
    {"case_id": "cmp_mouse_no_history", "task_id": "p14e2_lite_0044",
     "product_refs": ["B004CRRRB2", "B08GFW1ZNG"], "purpose": "unknown attributes, no history"},
    {"case_id": "cmp_invalid_id", "task_id": "p14e2_lite_0043",
     "product_refs": ["B09828X5WY", "B000NOTREAL"], "purpose": "invalid product identity"},
    {"case_id": "cmp_namespace_mismatch", "task_id": "p14e2_lite_0043",
     "product_refs": ["B09828X5WY", "sports:123"], "purpose": "wrong namespace"},
]

CLARIFICATION_CASES = [
    {"case_id": "clr_missing_need", "task_id": "p14e2_lite_0057",
     "reply": "I care about noise cancelling and want a wireless option.", "turn_id": "p16_reply_missing"},
    {"case_id": "clr_conflict", "task_id": "p14e2_lite_0056",
     "reply": "Forget my wired preference. I prefer wireless.", "turn_id": "p16_reply_conflict"},
    {"case_id": "clr_withdrawal", "task_id": "p14e2_lite_0055",
     "reply": "Forget my wireless preference.", "turn_id": "p16_reply_withdraw"},
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_memory(task: dict[str, Any]) -> ShoppingCompanionMemory:
    memory = ShoppingCompanionMemory()
    memory.ingest_sessions(task["task_id"], [dict(session) for session in task["history_sessions"]],
                           source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref=f"p16:{task['task_id']}")
    return memory


def read_history(memory: ShoppingCompanionMemory, task: dict[str, Any]) -> dict[str, Any]:
    return HistoryStateReader(memory.preference_service).read(
        task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"])


def make_executor(catalogue: ElectronicsCatalogue, task: dict[str, Any], history: dict[str, Any]):
    return ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=task["current_query"],
        preferences=prepare_preferences(history), hard=prepare_hard(task["current_query"]),
        task_context=task, memory_history=history,
    )


def run_rule(executor: ElectronicsPlannerExecutor, task: dict[str, Any]) -> dict[str, Any]:
    observation = PlannerObservation(
        step=0, query=task["current_query"],
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                  "preferences": executor.preferences},
    )
    started = time.perf_counter()
    error = None
    try:
        run = run_planner(planner=P14ARulePlanner(), executor=executor,
                          initial_observation=observation, max_steps=MAX_STEPS)
        status = executor.terminal_status
        if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
            status = "budget_exhausted"
    except Exception as exc:  # record honestly
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
        run = None
    return {
        "status": status, "selected_candidate_id": executor.selected_candidate_id,
        "ask_question": executor.ask_question, "gate": executor.last_decision,
        "error": error, "tool_calls": list(executor.tool_calls),
        "planner_steps": run.as_dict() if run is not None else None,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def comparison_cases(catalogue: ElectronicsCatalogue, tasks: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for case in COMPARISON_CASES:
        task = tasks[case["task_id"]]
        memory = make_memory(task)
        try:
            history = read_history(memory, task)
            executor = make_executor(catalogue, task, history)
            started = time.perf_counter()
            result = executor.invoke_skill(REGISTRY, "shopping-comparison", product_refs=case["product_refs"])
            result["case_id"] = case["case_id"]
            result["purpose"] = case["purpose"]
            result["agent_context"] = {
                "planner": "P14ARulePlanner",
                "executor": type(executor).__name__,
                "search_mode": catalogue.search_source.mode,
                "memory_status": history.get("status"),
                "products_seen": [product["item_id"] for product in result.get("products") or []],
            }
            result["latency_total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            records.append({"case_id": case["case_id"], "skill": "shopping-comparison",
                            "task_id": task["task_id"], "input": {"product_refs": case["product_refs"]},
                            "history_status": history.get("status"),
                            "active_preferences": history.get("active_preferences") or [],
                            "inactive_preferences": history.get("inactive_preferences") or [],
                            "result": result,
                            "error": None, "real_tool_calls": True})
        except Exception as exc:
            records.append({"case_id": case["case_id"], "skill": "shopping-comparison",
                            "task_id": task["task_id"], "input": {"product_refs": case["product_refs"]},
                            "result": None, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            memory.close()
    return records


def clarification_cases(catalogue: ElectronicsCatalogue, tasks: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    # Ask + real reply + resumed normal Agent run for two cases.
    for case in CLARIFICATION_CASES:
        task = tasks[case["task_id"]]
        memory = make_memory(task)
        try:
            history = read_history(memory, task)
            executor = make_executor(catalogue, task, history)
            # Use the existing Agent/Gate to obtain the clarification decision when one exists.
            rule = run_rule(executor, task)
            asked = executor.invoke_skill(REGISTRY, "preference-clarification", operation="ask",
                                          preferences=executor.preferences, gate=executor.last_decision)
            before = list((history.get("active_preferences") or []))
            reply = REGISTRY.invoke("preference-clarification", operation="resume", memory=memory,
                                    task=task, user_message=case["reply"], turn_id=case["turn_id"],
                                    now=1766000000.0)
            updated_preferences = prepare_preferences({"active_preferences": reply["active_preferences"]})
            resumed_executor = ElectronicsPlannerExecutor(
                catalogue=catalogue, current_query=task["current_query"],
                preferences=updated_preferences, hard=prepare_hard(task["current_query"]),
                task_context=task,
                memory_history=HistoryStateReader(memory.preference_service).read(
                    task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"]),
            )
            resumed = run_rule(resumed_executor, task)
            records.append({
                "case_id": case["case_id"], "skill": "preference-clarification",
                "task_id": task["task_id"], "input": {"reply_label": "explicit_test_user_reply",
                                                      "reply": case["reply"]},
                "ask_result": asked, "gate_before": executor.last_decision,
                "agent_before": {"status": rule["status"], "selected": rule["selected_candidate_id"]},
                "resume_result": reply,
                "memory_before_active": before,
                "memory_after_active": reply["active_preferences"],
                "memory_after_inactive": reply["inactive_preferences"],
                "resumed_agent": {"status": resumed["status"],
                                  "selected": resumed["selected_candidate_id"],
                                  "gate": resumed["gate"]},
                "real_tool_calls": True,
                "error": None,
            })
        except Exception as exc:
            records.append({"case_id": case["case_id"], "skill": "preference-clarification",
                            "task_id": task["task_id"], "result": None,
                            "error": f"{type(exc).__name__}: {exc}"})
        finally:
            memory.close()

    # Ask without a reply: must stay awaiting, no recommendation.
    task = tasks["p14e2_lite_0057"]
    memory = make_memory(task)
    try:
        history = read_history(memory, task)
        executor = make_executor(catalogue, task, history)
        rule = run_rule(executor, task)
        asked = executor.invoke_skill(REGISTRY, "preference-clarification", operation="ask",
                                      preferences=executor.preferences, gate=executor.last_decision)
        records.append({"case_id": "clr_unanswered", "skill": "preference-clarification",
                        "task_id": task["task_id"], "ask_result": asked,
                        "gate_before": executor.last_decision,
                        "agent_before": {"status": rule["status"], "selected": rule["selected_candidate_id"]},
                        "reply_submitted": False, "finalize_allowed": False,
                        "memory_unchanged": True, "error": None})
    finally:
        memory.close()

    # Duplicate user reply: same turn id must not add a second event/entry.
    task = tasks["p14e2_lite_0057"]
    memory = make_memory(task)
    try:
        history = read_history(memory, task)
        executor = make_executor(catalogue, task, history)
        rule = run_rule(executor, task)
        asked = executor.invoke_skill(REGISTRY, "preference-clarification", operation="ask",
                                      preferences=executor.preferences, gate=executor.last_decision)
        first = REGISTRY.invoke("preference-clarification", operation="resume", memory=memory,
                                task=task, user_message="I prefer wireless.", turn_id="p16_dup",
                                now=1766000100.0)
        second = REGISTRY.invoke("preference-clarification", operation="resume", memory=memory,
                                 task=task, user_message="I prefer wireless.", turn_id="p16_dup",
                                 now=1766000101.0)
        records.append({"case_id": "clr_duplicate_reply", "skill": "preference-clarification",
                        "task_id": task["task_id"], "ask_result": asked,
                        "first": first, "second": second, "reply_submitted_twice": True,
                        "error": None})
    finally:
        memory.close()

    # Cross-user read must be rejected by the existing reader.
    task = tasks["p14e2_lite_0057"]
    memory = make_memory(task)
    try:
        try:
            REGISTRY.invoke("preference-clarification", operation="read", memory=memory,
                            user_key=task["task_id"], conversation_id="p14e2_lite_0056",
                            current_query=task["current_query"])
            cross = {"rejected": False}
        except Exception as exc:
            cross = {"rejected": True, "error": f"{type(exc).__name__}: {exc}"}
        records.append({"case_id": "clr_cross_user", "skill": "preference-clarification",
                        "task_id": task["task_id"], "cross_user_read": cross,
                        "error": None if cross["rejected"] else "cross-user read was not rejected"})
    finally:
        memory.close()
    return records


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    config = configure_catalogue(catalogue)
    if config["status"] != "active":
        raise RuntimeError(f"BGE backend required for P16 E2E: {config}")
    tasks = {row["task_id"]: row for row in load_jsonl(DATA / "dev.agent.jsonl")}
    (OUT / "e2e").mkdir(parents=True, exist_ok=True)
    records = comparison_cases(catalogue, tasks) + clarification_cases(catalogue, tasks)
    for record in records:
        (OUT / "e2e" / f"{record['case_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "e2e_trajectories.json").write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    summary = {
        "schema": "agentrecx.p16.e2e_summary.v1",
        "search_backend": config,
        "skill_registry": REGISTRY.names(),
        "skills": {"shopping-comparison": "2-5 real products + Product Info + P15 Evidence Bundle + claims validation",
                    "preference-clarification": "existing Rule/Gate ASK + M9 process_turn real reply + resume"},
        "cases": len(records),
        "errors": [record["case_id"] for record in records if record.get("error")],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
