"""Compute P100 Agent E2E metrics from actual trajectories."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"
TRAJ = RUN / "p100_agent_e2e.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(values, p):
    values = sorted(values)
    return round(values[int(p * (len(values) - 1))], 3) if values else None


def main():
    rows = read_jsonl(TRAJ) if TRAJ.exists() else []
    tasks = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_agent_tasks_memory_on.jsonl")}
    pools = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_bge_pools.jsonl")}
    meta = {row["item_id"]: row for row in read_jsonl(DATA / "p100_candidate_metadata.jsonl")}
    out = {"schema": "agentrecx.p100.agent_e2e_metrics.v1", "trajectories": len(rows),
           "expected_trajectories": 300, "complete": len(rows) == 300, "arms": {},
           "memory_state_consistency": {}, "gate_reason_counts": {}, "notes": []}
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["base_query_id"]][row["arm"]] = row
    for arm in ("existing", "natural", "hybrid"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        status = Counter(row["status"] for row in arm_rows)
        lat = [row["e2e_latency_ms"] for row in arm_rows]
        planner_lat = [call["latency_ms"] for row in arm_rows for call in row.get("planner_calls") or [] if call.get("latency_ms")]
        ranker_lat = [row["ranker_latency_ms"] for row in arm_rows if row.get("ranker_latency_ms") is not None]
        out["arms"][arm] = {
            "trajectories": len(arm_rows),
            "status": dict(status),
            "finalize": status.get("finalized", 0),
            "ask": status.get("awaiting_clarification", 0),
            "errors": status.get("error", 0),
            "budget_exhausted": status.get("budget_exhausted", 0),
            "hard_constraint_violations": sum(row["hard_violations"] for row in arm_rows),
            "selected_in_eligible": sum(1 for row in arm_rows if row.get("selected_in_eligible") is True),
            "valid_topk_items": sum(row["valid_topk"] for row in arm_rows),
            "evidence_supported_items": sum(row["evidence_supported"] for row in arm_rows),
            "ask_without_topk": sum(1 for row in arm_rows if row["ask"] and row["valid_topk"] == 0),
            "top1_change_vs_existing": None,
            "e2e_p50_ms": pct(lat, 0.5), "e2e_p95_ms": pct(lat, 0.95),
            "planner_p50_ms": pct(planner_lat, 0.5), "planner_p95_ms": pct(planner_lat, 0.95),
            "ranker_p50_ms": pct(ranker_lat, 0.5) if ranker_lat else None,
            "invalid_planner_actions": sum(1 for row in arm_rows for call in row.get("planner_calls") or [] if call.get("errors")),
        }
    # Cross-arm top1 differences.
    for arm in ("natural", "hybrid"):
        changes = 0
        comparable = 0
        for bid, arms in by_task.items():
            if "existing" in arms and arm in arms:
                comparable += 1
                changes += int(arms["existing"].get("top1") != arms[arm].get("top1"))
        if arm in out["arms"]:
            out["arms"][arm]["top1_change_vs_existing"] = f"{changes}/{comparable}"
    # Memory state consistency.
    scenarios = Counter()
    consistent = 0
    for row in rows:
        if row["arm"] != "existing":
            continue
        task = tasks.get(row["base_query_id"]) or {}
        prefs = row.get("active_preferences") or []
        positives = [p["value"] for p in prefs if str(p.get("polarity")) in {"prefer", "positive"}]
        negatives = [p["value"] for p in prefs if str(p.get("polarity")) in {"avoid", "negative"}]
        scenario = task.get("scenario_type")
        scenarios[scenario] += 1
        if scenario in {"no_memory", "preference_withdrawal"}:
            ok = not positives and not negatives
        elif scenario == "positive_memory":
            ok = bool(positives)
        elif scenario == "negative_memory":
            ok = bool(negatives)
        elif scenario == "multiple_preferences":
            ok = len(positives) >= 2
        elif scenario == "preference_update":
            ok = bool(positives) and not negatives
        elif scenario == "query_memory_competition":
            ok = bool(positives)
        elif scenario == "unknown_evidence":
            values = [p["value"] for p in prefs]
            candidate_attrs = {attr for item_id in (pools.get(row["base_query_id"]) or {}).get("candidate_pool_ids", [])
                               for attr in (meta.get(item_id) or {}).get("attributes") or []}
            ok = bool(values) and all(value not in candidate_attrs for value in values)
        else:
            ok = True
        consistent += int(ok)
    out["memory_state_consistency"] = {"scenario_counts": dict(scenarios), "consistent_existing": consistent,
                                        "total_existing": scenarios.get(None, 0) or sum(scenarios.values())}
    for row in rows:
        gate = row.get("gate_reason") or row.get("gate_action")
        if gate:
            out["gate_reason_counts"][str(gate)] = out["gate_reason_counts"].get(str(gate), 0) + 1
    out["notes"] = ["FINALIZE is a legal terminal state, not a recommendation-quality label.",
                    "ASK is not scored as failure; correctness has no independent memory Gold.",
                    "Top-K items are validated per candidate; unjudged products are not negatives."]
    (RUN / "p100_agent_e2e_metrics.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1)[:7000])


if __name__ == "__main__":
    main()
