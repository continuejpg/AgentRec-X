import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"
OUT = REPO / "runs/agentrec_x_phase52"
OUT.mkdir(parents=True, exist_ok=True)

tasks = {r["base_query_id"]: r for r in [json.loads(l) for l in (DATA / "p100_agent_tasks_memory_on.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
pools = {r["base_query_id"]: r for r in [json.loads(l) for l in (DATA / "p100_bge_pools.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
rows = [json.loads(l) for l in (RUN / "p100_agent_e2e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

def category(query):
    q = query.lower()
    for name, words in {
        "audio": ("headphone", "earbud", "speaker", "audio", "microphone"),
        "phone_tablet": ("phone", "tablet", "case"),
        "computer": ("laptop", "computer", "monitor", "keyboard", "mouse", "usb", "hard drive", "ssd"),
        "tv_video": ("tv", "television", "projector", "camera"),
        "gaming": ("gaming", "game", "console"),
        "home": ("vacuum", "coffee", "blender", "kitchen", "home", "light"),
        "accessory": ("charger", "cable", "adapter", "power", "battery"),
    }.items():
        if any(word in q for word in words):
            return name
    return "other"

analysis = []
for row in rows:
    task = tasks.get(row["base_query_id"]) or {}
    pool = pools.get(row["base_query_id"]) or {}
    eligible = sum(1 for e in pool.get("eligibility") or [] if e.get("presentable"))
    analysis.append({
        "base_query_id": row["base_query_id"], "task_id": row["task_id"], "arm": row["arm"],
        "query": row["query"], "query_category": category(row["query"]),
        "scenario_type": row["scenario_type"], "status": row["status"],
        "gate_action": row.get("gate_action"), "gate_reason": row.get("gate_reason"),
        "eligible_candidates": eligible, "valid_topk": row["valid_topk"],
        "topk_evidence_supported": row["evidence_supported"],
        "active_preferences": row.get("active_preferences") or [],
        "error": row.get("error"),
    })

summary = {
    "schema": "agentrecx.p52.ask_analysis.v1",
    "source": "existing 300 P100 Agent E2E trajectories; no rerun",
    "status_counts": dict(Counter(row["status"] for row in rows)),
    "status_by_arm": {arm: dict(Counter(row["status"] for row in rows if row["arm"] == arm)) for arm in ("existing", "natural", "hybrid")},
    "gate_reason_counts": dict(Counter(str(row.get("gate_reason")) for row in rows)),
    "ask_reason_counts": dict(Counter(str(row.get("gate_reason")) for row in rows if row["status"] == "awaiting_clarification")),
    "ask_by_scenario": {scenario: {arm: sum(1 for row in rows if row["arm"] == arm and row["scenario_type"] == scenario and row["status"] == "awaiting_clarification") for arm in ("existing", "natural", "hybrid")} for scenario in sorted({row["scenario_type"] for row in rows})},
    "ask_by_category": {},
    "no_verified_candidates": [row for row in analysis if row["status"] == "no_verified_candidate_in_scope"],
    "planner_errors": [row for row in analysis if row["status"] == "error"],
    "note": "ASK is not automatically a failure. No independent human memory preference Gold exists; reasons are classified from Gate reason and scenario/eligible/evidence inputs, not from a gold label.",
}
for category_name in sorted({row["query_category"] for row in analysis}):
    subset = [row for row in analysis if row["query_category"] == category_name]
    summary["ask_by_category"][category_name] = {
        "trajectories": len(subset),
        "asks": sum(1 for row in subset if row["status"] == "awaiting_clarification"),
        "insufficient_criteria": sum(1 for row in subset if row["gate_reason"] == "insufficient_decision_criteria"),
        "conflicts": sum(1 for row in subset if row["gate_reason"] == "candidate_conflicts_with_stated_preference"),
        "errors": sum(1 for row in subset if row["status"] == "error"),
    }
summary["classification"] = {
    "insufficient_decision_criteria": "Gate says user criteria + candidate evidence were insufficient; often UNKNOWN metadata / unjudged pool, not a Ranker bug",
    "candidate_conflicts_with_stated_preference": "Gate detected a stated preference conflict; ASK is appropriate and preserved",
    "no_verified_candidate_in_scope": "search/retrieval or hard/evidence coverage produced no verified candidate; not a success",
    "planner_error": "p100_008 produced an invalid planner action; recorded as error, no forced terminal",
}
(OUT / "p100_ask_analysis.json").write_text(json.dumps({"summary": summary, "cases": analysis}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=1)[:7000])
