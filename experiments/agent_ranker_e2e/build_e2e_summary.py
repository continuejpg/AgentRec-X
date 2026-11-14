import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

R = Path("runs/agent_ranker_phase43")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    e2e = json.loads((R / "e2e_p20.json").read_text(encoding="utf-8"))
    records = e2e["records"]
    by_arm = defaultdict(list)
    for row in records:
        by_arm[row["arm"]].append(row)
    arms = {}
    for arm, rows in by_arm.items():
        lat = sorted(row["e2e_latency_ms"] for row in rows)
        planner_lat = [value for row in rows for value in row["planner_latencies_ms"] if value is not None]
        bge_lat = [value for row in rows for value in row["bge_retrieval_latencies_ms"]]
        arms[arm] = {
            "tasks": len(rows),
            "success": sum(1 for row in rows if row["success"]),
            "finalized": sum(1 for row in rows if row["status"] == "finalized"),
            "awaiting_clarification": sum(1 for row in rows if row["status"] == "awaiting_clarification"),
            "hard_constraint_violations": sum(1 for row in rows if row["hard_constraint_violation"]),
            "final_answer_evidence_supported": sum(1 for row in rows if row["final_answer"]["evidence_supported"]),
            "errors": sum(1 for row in rows if row["error"]),
            "e2e_p50_ms": round(statistics.median(lat), 2),
            "e2e_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 2),
            "planner_p50_ms": round(statistics.median(planner_lat), 2) if planner_lat else None,
            "bge_p50_ms": round(statistics.median(bge_lat), 2) if bge_lat else None,
            "ranker_latency_ms": [row["ranker_latency_ms"] for row in rows],
            "selected": {row["task_id"]: row["selected_product_id"] for row in rows},
            "0051": next((row for row in rows if row["task_id"] == "p14e2_lite_0051"), None),
        }
    cf = json.loads((R / "e2e_counterfactual.json").read_text(encoding="utf-8"))
    cf_status = Counter((row["state"], row["arm"], row["status"]) for row in cf["records"])
    cf_summary = {
        "records": len(cf["records"]),
        "cases": len({row["case_id"] for row in cf["records"]}),
        "status_by_arm": {f"{state}/{arm}": dict(Counter(row["status"] for row in cf["records"] if row["state"] == state and row["arm"] == arm)) for state in ("baseline_memory_state", "mutated_memory_state") for arm in ("existing", "structured", "hybrid") if any(row["state"] == state and row["arm"] == arm for row in cf["records"])},
        "hard_constraint_violations": sum(1 for row in cf["records"] if row["hard_constraint_violation"]),
        "requires_clarification_awaiting": sum(1 for row in cf["records"] if row["requires_clarification"] and row["status"] == "awaiting_clarification"),
        "hybrid_budget_exhausted": sum(1 for row in cf["records"] if row["arm"] == "hybrid" and row["status"] == "budget_exhausted"),
    }
    constraint = json.loads((R / "memory_constraint_tests.json").read_text(encoding="utf-8"))
    reproduction = json.loads((R / "reproduction_differences.json").read_text(encoding="utf-8"))
    blind = read_jsonl(R / "blind_review_tasks.jsonl")
    payload = {
        "schema": "agentrecx.agent_ranker_phase43.summary.v1",
        "p20_e2e": {
            "tasks": e2e["tasks"],
            "arms": arms,
            "retrieval": e2e["retrieval"],
            "bge_warmup_ms": e2e["bge_warmup_ms"],
            "model_lifecycle": e2e["model_lifecycle"],
        },
        "counterfactual_e2e": cf_summary,
        "memory_constraints": constraint,
        "reproduction": {
            "pair_changes": reproduction["pair_changes"]["pair_prediction_changed"],
            "candidate_texts_changed": reproduction["product_serialization"]["candidate_texts_changed"],
            "candidate_texts_total": reproduction["product_serialization"]["candidate_texts_compared"],
            "format_control_no_memory": reproduction["format_control_no_memory"],
        },
        "blind_review": {"tasks": len(blind), "human_gold": 0, "human_single_review": 0},
        "provenance": {
            "p20": "provisional observed-acceptable; task-level labels automated/pending human",
            "benchmark_v2": "Silver rule-derived; 145 pairs; no Human Gold",
            "counterfactual": "mechanistic; no direction gold",
            "human_gold": 0,
        },
    }
    (R / "e2e_summary.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1)[:9000])


if __name__ == "__main__":
    main()
