import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path("runs/agent_ranker_phase44")
P43 = Path("runs/agent_ranker_phase43")


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def matrix_summary(path):
    data = load(path) if Path(path).suffix else None
    return data


def summarize(data):
    records = data["records"]
    status = Counter((row["arm"], row["state"], row["status"]) for row in records)
    by_arm = {}
    for arm in ("existing", "natural", "hybrid"):
        rows = [row for row in records if row["arm"] == arm]
        by_arm[arm] = {
            "trajectories": len(rows),
            "finalized": sum(1 for row in rows if row["status"] == "finalized"),
            "awaiting_clarification": sum(1 for row in rows if row["status"] == "awaiting_clarification"),
            "budget_exhausted": sum(1 for row in rows if row["status"] == "budget_exhausted"),
            "errors": sum(1 for row in rows if row["status"] == "error"),
            "hard_constraint_violations": sum(1 for row in rows if row["hard_constraint_violation"]),
            "selected_in_eligible": sum(1 for row in rows if row["selected_in_eligible"] is True),
        }
    conflict = [row for row in records if row["requires_clarification"]]
    return {
        "trajectories": len(records),
        "status": {f"{arm}/{state}/{status_name}": count for (arm, state, status_name), count in status.items()},
        "by_arm": by_arm,
        "conflict_requires_clarification": len(conflict),
        "conflict_awaiting": sum(1 for row in conflict if row["status"] == "awaiting_clarification"),
        "hard_constraint_violations": sum(1 for row in records if row["hard_constraint_violation"]),
    }


def main():
    rule = load("cf_matrix_rule.json")
    qwen = load("cf_matrix_qwen_full.json") if (R / "cf_matrix_qwen_full.json").exists() else (load("cf_matrix_qwen.json") if (R / "cf_matrix_qwen.json").exists() else None)
    ranking = load("cf_ranking_metrics.json")
    fix = load("hybrid_budget_fix.json")
    p20 = json.loads((P43 / "e2e_summary.json").read_text(encoding="utf-8"))
    blind = (P43 / "blind_review_tasks.jsonl").read_text(encoding="utf-8").splitlines()
    payload = {
        "schema": "agentrecx.agent_ranker_phase44.summary.v1",
        "hybrid_fix": {
            "root_cause": fix["root_cause"],
            "before_status": fix["before_status"],
            "after_status": fix["after_status"],
            "changed_trajectories": sum(1 for row in fix["before_after"] if row["changed"]),
            "formula_unchanged": fix["formula_unchanged"],
        },
        "rule_planner": summarize(rule),
        "qwen_planner": summarize(qwen) if qwen else None,
        "cf_ranking": ranking["by_arm"],
        "p20_e2e": {
            arm: {
                "success": values["success"],
                "finalized": values["finalized"],
                "hard_constraint_violations": values["hard_constraint_violations"],
                "answer_evidence_supported": values["final_answer_evidence_supported"],
                "e2e_p50_ms": values["e2e_p50_ms"],
                "e2e_p95_ms": values["e2e_p95_ms"],
            }
            for arm, values in p20["p20_e2e"]["arms"].items()
        },
        "blind_review": {"tasks": len([line for line in blind if line.strip()]), "human_gold": 0, "human_single_review": 0},
        "provenance": {"p20": "provisional", "benchmark_v2": "Silver rule-derived", "counterfactual": "mechanistic", "human_gold": 0},
    }
    (R / "phase44_summary.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1)[:7000])


if __name__ == "__main__":
    main()
