import json
from collections import defaultdict
from pathlib import Path

R = Path("runs/agent_ranker_phase44")
OUT = Path("runs/agent_ranker_phase45")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {row["case_id"]: row for row in read_jsonl(Path("runs/memory_contribution_phase1/counterfactual_cases.jsonl"))}
    ranking = json.loads((R / "cf_ranking_metrics.json").read_text(encoding="utf-8"))
    rank_rows = {(row["case_id"], row["state"]): row for row in ranking["rows"]}
    output = {"schema": "agentrecx.agent_ranker_phase45.ask_quality.v1",
              "source": "Phase 4.4 Rule/Qwen E2E trajectories; no new 192-run replay",
              "gate_reason_availability": "Phase 4.4 matrix stored gate action/status, not full GateContext; reason classified conservatively, no fabricated reason code.",
              "cases": []}
    for planner in ("rule", "qwen"):
        matrix_path = R / ("cf_matrix_qwen_full.json" if planner == "qwen" and (R / "cf_matrix_qwen_full.json").exists() else f"cf_matrix_{planner}.json")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        by_case_state = defaultdict(dict)
        for row in matrix["records"]:
            by_case_state[(row["case_id"], row["state"])][row["arm"]] = row
        for (case_id, state), arms in sorted(by_case_state.items()):
            statuses = {arm: row["status"] for arm, row in arms.items()}
            if len(set(statuses.values())) <= 1:
                continue
            case = cases[case_id]
            rank = rank_rows.get((case_id, state)) or {}
            rankings = rank.get("rankings") or {}
            if case.get("requires_clarification"):
                classification = "CLEAR_CONFLICT_ASK_REQUIRED"
            elif any(value == "error" for value in statuses.values()):
                classification = "PLANNER_ERROR_MIXED"
            elif any(value == "awaiting_clarification" for value in statuses.values()):
                classification = "UNJUDGED_POSSIBLE_UNNECESSARY_ASK"
            else:
                classification = "MIXED_TERMINAL"
            output["cases"].append({
                "planner": planner,
                "case_id": case_id,
                "state": state,
                "category": case["category"],
                "query": case["query"],
                "requires_clarification": bool(case.get("requires_clarification")),
                "active_preferences_baseline": (case.get("baseline_memory_state") or {}).get("active_preferences") or [],
                "active_preferences_mutated": (case.get("mutated_memory_state") or {}).get("active_preferences") or [],
                "candidate_pool_size": len(case.get("eligible_candidate_ids") or []),
                "status_by_arm": statuses,
                "gate_action_by_arm": {arm: row.get("gate_action") for arm, row in arms.items()},
                "selected_by_arm": {arm: row.get("selected_product_id") for arm, row in arms.items()},
                "ranker_top3": {arm: (rankings.get(arm) or [])[:3] for arm in rankings},
                "classification": classification,
                "truth_status": "UNJUDGED" if not case.get("requires_clarification") else "ASK_REQUIRED",
                "note": "No independent user preference Gold; ASK correctness cannot be proven from these cases alone. Natural asking more is not automatically wrong.",
            })
    output["summary"] = {
        "divergent_case_state_pairs": len(output["cases"]),
        "rule": sum(1 for case in output["cases"] if case["planner"] == "rule"),
        "qwen": sum(1 for case in output["cases"] if case["planner"] == "qwen"),
        "clear_conflict": sum(1 for case in output["cases"] if case["classification"] == "CLEAR_CONFLICT_ASK_REQUIRED"),
        "unjudged_possible_over_ask": sum(1 for case in output["cases"] if case["classification"] == "UNJUDGED_POSSIBLE_UNNECESSARY_ASK"),
        "planner_error_mixed": sum(1 for case in output["cases"] if case["classification"] == "PLANNER_ERROR_MIXED"),
    }
    (OUT / "ask_quality_diagnostics.json").write_text(json.dumps(output, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=1))


if __name__ == "__main__":
    main()
