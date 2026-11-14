import json
from pathlib import Path

R = Path("runs/agent_ranker_phase44")
OLD = Path("runs/agent_ranker_phase43/e2e_counterfactual.json")
NEW = R / "cf_matrix_rule.json"
OUT = R / "hybrid_budget_fix.json"

old = json.loads(OLD.read_text(encoding="utf-8"))
new = json.loads(NEW.read_text(encoding="utf-8"))
old_map = {(row["case_id"], row["state"], row["arm"]): row for row in old["records"]}
new_map = {(row["case_id"], row["state"], row["arm"]): row for row in new["records"]}
matched = []
for key, before in old_map.items():
    if key[2] != "hybrid":
        continue
    after = new_map.get(key)
    if after is None:
        continue
    matched.append({
        "case_id": key[0], "state": key[1],
        "before": {"status": before["status"], "actions": before["actions"], "gate": before["gate_action"], "selected": before["selected_product_id"]},
        "after": {"status": after["status"], "actions": after["actions"], "gate": after["gate_action"], "selected": after["selected_product_id"]},
        "changed": before["status"] != after["status"] or before["selected_product_id"] != after["selected_product_id"],
    })
payload = {
    "schema": "agentrecx.agent_ranker_phase44.hybrid_budget_fix.v1",
    "root_cause": "The Phase 4.3 counterfactual runner reused one ElectronicsPlannerExecutor for the existing, structured and hybrid arms. After the first arm finalized, executor.terminal_status='finalized', selected_candidate_id and last_rows/ledger state leaked into the next arm. The next run therefore stopped after RETRIEVE and was recorded as budget_exhausted. This was an experiment integration/state-reuse bug, not a Gate or Hybrid score bug.",
    "fix": "Instantiate a fresh ElectronicsPlannerExecutor per (case, state, arm), attach a fresh FixedPoolSource and a fresh CachedRanker. Do not increase MAX_STEPS, bypass Gate, or force FINALIZE.",
    "formula_unchanged": "pool_z(shopranker_query_only) + 0.5*positive_match_rate - 0.75*negative_conflict_rate",
    "before_status": {"budget_exhausted": 16, "finalized": 0, "awaiting_clarification": 0},
    "after_status": {
        "finalized": sum(1 for row in new["records"] if row["arm"] == "hybrid" and row["status"] == "finalized"),
        "awaiting_clarification": sum(1 for row in new["records"] if row["arm"] == "hybrid" and row["status"] == "awaiting_clarification"),
        "budget_exhausted": sum(1 for row in new["records"] if row["arm"] == "hybrid" and row["status"] == "budget_exhausted"),
    },
    "before_after": matched,
    "note": "The before artifacts are preserved in runs/agent_ranker_phase43; the fixed run is written to runs/agent_ranker_phase44.",
}
OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"root_cause": payload["root_cause"], "after_status": payload["after_status"], "records": len(matched)}, indent=1))
