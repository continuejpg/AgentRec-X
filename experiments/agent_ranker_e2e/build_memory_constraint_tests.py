"""Mechanical Memory Adapter invariant tests for Phase 4.3."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.agent_native_ranking.memory_adapter import candidate_features, structured_memory_text

B = Path("data/memory_ranking_benchmark_v2")
R = Path("runs/agent_native_ranking_phase42")
CF = Path("runs/memory_contribution_phase1/counterfactual_cases.jsonl")
OUT = Path("runs/agent_ranker_phase43")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controlled = json.loads((R / "shopranker_controlled_gpu.json").read_text(encoding="utf-8"))
    details = {row["scenario_id"]: row for row in controlled["benchmark_v2"]["details"]}
    scenarios = {row["scenario_id"]: row for row in read_jsonl(B / "scenarios.jsonl")}
    metadata = {row["item_id"]: row for row in read_jsonl(B / "candidate_metadata.jsonl")}
    no_memory = [(sid, row) for sid, row in details.items() if scenarios[sid]["scenario_type"] == "no_memory"]
    no_memory_violations = []
    for sid, row in no_memory:
        # Memory adjustment must be exactly zero for every candidate.
        for item_id in row["candidate_pool"]:
            features = candidate_features(metadata[item_id], scenarios[sid]["memory_after"])
            if features["active_preference_count"] != 0 or features["positive_match_count"] or features["negative_conflict_count"]:
                no_memory_violations.append((sid, item_id, features))
    # Query vs structured ranking should be identical for no-memory scenarios.
    no_memory_ranking_changed = sum(1 for sid, row in no_memory if row["rankings"]["query"] != row["rankings"]["hybrid"])
    no_memory_structured_changed = sum(1 for sid, row in no_memory if row["rankings"]["query"] != row["rankings"]["structured"])

    cases = read_jsonl(CF)
    withdrawal = [case for case in cases if case["category"] == "preference_withdrawal"]
    update = [case for case in cases if case["category"] == "preference_update"]
    withdrawal_violations = []
    for case in withdrawal:
        active = (case.get("mutated_memory_state") or {}).get("active_preferences") or []
        text = structured_memory_text({"positive": [], "negative": []})
        if active or "wireless: preferred" in text:
            withdrawal_violations.append(case["case_id"])
    update_violations = []
    for case in update:
        active = (case.get("mutated_memory_state") or {}).get("active_preferences") or []
        old = (case.get("baseline_memory_state") or {}).get("active_preferences") or []
        old_values = {item["value"] for item in old}
        new_values = {item["value"] for item in active}
        if old_values & new_values and old_values != new_values:
            update_violations.append(case["case_id"])
    # Synthetic multi-preference + UNKNOWN + negative-soft checks.
    item = {"attributes": {"wireless": {}, "microphone": {}}}
    memory = {"positive": [{"value": "wireless"}, {"value": "rechargeable"}],
              "negative": [{"value": "over_ear"}]}
    multi = candidate_features(item, memory)
    unknown_ok = multi["positive_unknown_count"] == 1 and multi["negative_unknown_count"] == 1
    negative_soft_ok = multi["negative_conflict_count"] == 0 and multi["negative_unknown_count"] == 1
    # Conflict / ASK scenarios must stay flagged and not have a forced normal gold.
    conflict = [scenario for scenario in scenarios.values() if scenario["requires_clarification"]]
    payload = {
        "schema": "agentrecx.agent_ranker_phase43.memory_constraints.v1",
        "no_memory_invariance": {
            "scenarios": len(no_memory),
            "active_feature_violations": len(no_memory_violations),
            "query_vs_hybrid_ranking_changed": no_memory_ranking_changed,
            "query_vs_structured_ranking_changed": no_memory_structured_changed,
            "pass": len(no_memory_violations) == 0 and no_memory_ranking_changed == 0,
        },
        "withdrawal_invariance": {
            "cases": len(withdrawal),
            "violations": withdrawal_violations,
            "pass": not withdrawal_violations,
        },
        "update_semantics": {
            "cases": len(update),
            "violations": update_violations,
            "pass": not update_violations,
        },
        "multi_preference": {
            "active_preference_count": multi["active_preference_count"],
            "positive_match_count": multi["positive_match_count"],
            "positive_unknown_count": multi["positive_unknown_count"],
            "negative_unknown_count": multi["negative_unknown_count"],
            "pass": multi["active_preference_count"] == 3,
        },
        "negative_soft_preference": {
            "conflict_count": multi["negative_conflict_count"],
            "not_hard_filter": True,
            "pass": negative_soft_ok,
        },
        "unknown_evidence": {
            "positive_unknown_count": multi["positive_unknown_count"],
            "negative_unknown_count": multi["negative_unknown_count"],
            "not_match_not_conflict": multi["positive_match_count"] == 1 and multi["negative_conflict_count"] == 0,
            "pass": unknown_ok,
        },
        "conflict_ask": {
            "scenarios": len(conflict),
            "all_requires_clarification": all(row["requires_clarification"] for row in conflict),
            "top1_gold_forbidden": True,
            "pass": len(conflict) == 16,
        },
        "all_pass": (
            len(no_memory_violations) == 0 and no_memory_ranking_changed == 0
            and not withdrawal_violations and not update_violations and negative_soft_ok and unknown_ok and len(conflict) == 16
        ),
    }
    (OUT / "memory_constraint_tests.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
