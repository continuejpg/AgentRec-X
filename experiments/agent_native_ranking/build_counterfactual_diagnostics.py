import json
from pathlib import Path

R = Path("runs/agent_native_ranking_phase42")
controlled = json.load(open(R / "shopranker_controlled_gpu.json", encoding="utf-8"))
diagnostics = json.load(open(R / "dcn_alignment_diagnostics.json", encoding="utf-8"))
shadow = json.load(open(R / "shadow_comparison.json", encoding="utf-8"))
rows = []
for case in controlled["counterfactual"]["details"]:
    dcn = next((row for row in diagnostics["counterfactual_B_one_pref"]["details"] if row["case_id"] == case["case_id"]), None)
    rows.append({
        "case_id": case["case_id"],
        "category": case["category"],
        "requires_clarification": case["requires_clarification"],
        "shop_query_top1_changed": case["baseline_memory_state"]["query_ranking"][:1] != case["mutated_memory_state"]["query_ranking"][:1],
        "shop_structured_top1_changed": case["baseline_memory_state"]["structured_ranking"][:1] != case["mutated_memory_state"]["structured_ranking"][:1],
        "shop_query_full_changed": case["baseline_memory_state"]["query_ranking"] != case["mutated_memory_state"]["query_ranking"],
        "shop_structured_full_changed": case["baseline_memory_state"]["structured_ranking"] != case["mutated_memory_state"]["structured_ranking"],
        "dcn_b_top1_changed": dcn["top1_changed"] if dcn else None,
        "dcn_b_full_changed": dcn["full_changed"] if dcn else None,
        "dcn_b_before_top1": dcn["before_top1"] if dcn else None,
        "dcn_b_after_top1": dcn["after_top1"] if dcn else None,
    })
payload = {
    "schema": "agentrecx.agent_native_ranking_phase42.counterfactual.v1",
    "provenance": "mechanistic; no independent direction labels; ASK cases must not be scored as normal Top-K gold",
    "cases": len(rows),
    "requires_clarification": sum(1 for row in rows if row["requires_clarification"]),
    "summary": shadow["counterfactual"],
    "by_category": {
        category: {
            "cases": len([row for row in rows if row["category"] == category]),
            "shop_structured_top1_changed": sum(1 for row in rows if row["category"] == category and row["shop_structured_top1_changed"]),
            "shop_structured_full_changed": sum(1 for row in rows if row["category"] == category and row["shop_structured_full_changed"]),
        }
        for category in sorted({row["category"] for row in rows})
    },
    "details": rows,
}
(R / "counterfactual_diagnostics.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({key: payload[key] for key in ("cases", "requires_clarification", "summary", "by_category")}, indent=1))
