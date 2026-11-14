import json
from pathlib import Path

from recommendation.preference.history import HistoryStateReader
from recommendation.shopping_companion.memory import SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory

DATA = Path("data/p100_extended_benchmark")
RUN = Path("runs/p100_extended_evaluation")
tasks = {r["base_query_id"]: r for r in [json.loads(l) for l in (DATA / "p100_agent_tasks_memory_on.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
rows = [json.loads(l) for l in (RUN / "p100_agent_e2e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
existing = {r["base_query_id"]: r for r in rows if r["arm"] == "existing"}
out = []
for bid, task in tasks.items():
    scenario = task["scenario_type"]
    if scenario not in {"unknown_evidence", "multiple_preferences"}:
        continue
    with ShoppingCompanionMemory() as memory:
        memory.ingest_sessions(task["task_id"], [dict(s) for s in task["history_sessions"]],
                               source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref=f"p52:{task['task_id']}")
        history = HistoryStateReader(memory.preference_service).read(task["task_id"], conversation_id=task["task_id"], current_query=task["current_query"])
    raw = history["active_preferences"]
    prepared = existing[bid].get("active_preferences") or []
    if scenario == "unknown_evidence":
        classification = ("writer_produced_unknown_preference_but_executor_canonicalisation_dropped_it"
                          if raw and not prepared else "writer_produced_nothing")
    else:
        classification = ("extractor_only_one_preference" if len(raw) < 2 and len(prepared) < 2
                          else "canonicalisation_reduced_preferences")
    out.append({
        "base_query_id": bid, "scenario_type": scenario,
        "history_sessions": task["history_sessions"],
        "writer_active_preferences": raw,
        "executor_prepared_preferences": prepared,
        "classification": classification,
        "raw_count": len(raw), "prepared_count": len(prepared),
    })
Path("runs/agentrec_x_phase52").mkdir(parents=True, exist_ok=True)
Path("runs/agentrec_x_phase52/p100_memory_writer_anomalies.json").write_text(json.dumps({
    "schema": "agentrecx.p52.memory_writer_anomalies.v1",
    "note": "Raw Writer state is compared with executor-prepared active preferences; no bypass injection was used.",
    "cases": out,
}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for row in out:
    print(row["base_query_id"], row["scenario_type"], "raw", [(p["value"], p.get("polarity")) for p in row["writer_active_preferences"]], "prepared", [(p["value"], p.get("polarity")) for p in row["executor_prepared_preferences"]], row["classification"])
