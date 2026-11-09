"""P16 metrics for the two registered shopping skills."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/p16"


def main() -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((RUNS / "e2e").glob("*.json"))]
    comparison = [row for row in records if row["skill"] == "shopping-comparison"]
    clarification = [row for row in records if row["skill"] == "preference-clarification"]
    successful = [row for row in comparison if (row.get("result") or {}).get("status") == "comparison_ready"]
    invalid = [row for row in comparison if (row.get("result") or {}).get("status") in
               {"invalid_product_reference", "invalid_product_count", "role_mismatch"}]
    products = [product for row in successful for product in (row["result"].get("products") or [])]
    claims_valid = sum(int((row["result"] or {}).get("claims_valid") or 0) for row in successful)
    claims_rejected = sum(int((row["result"] or {}).get("claims_rejected") or 0) for row in successful)
    unknown_count = sum(int((row["result"] or {}).get("unknown_count") or 0) for row in successful)
    qwen_comparison = sum(1 for row in successful if (row["result"] or {}).get("qwen_used"))
    comparison_latencies = [float(row["result"]["latency_ms"]) for row in successful]
    preference_observations = [
        relation for row in successful for product in (row["result"].get("products") or [])
        for relation in (product.get("preference_relations") or [])
        if relation.get("observed_in_catalogue_text")
    ]
    asked = [row for row in clarification if (row.get("ask_result") or {}).get("status") == "awaiting_clarification"]
    updated = [row for row in clarification if (row.get("resume_result") or {}).get("update")]
    duplicate = next((row for row in clarification if row["case_id"] == "clr_duplicate_reply"), None)
    cross = next((row for row in clarification if row["case_id"] == "clr_cross_user"), None)
    unanswered = next((row for row in clarification if row["case_id"] == "clr_unanswered"), None)
    withdrawal = next((row for row in clarification if row["case_id"] == "clr_withdrawal"), None)
    conflict = next((row for row in clarification if row["case_id"] == "clr_conflict"), None)
    metrics = {
        "schema": "agentrecx.p16.metrics.v1",
        "cases": len(records),
        "comparison": {
            "cases": len(comparison), "successful_comparisons": len(successful),
            "expected_invalid_reference_cases": len(invalid),
            "products_compared": len(products), "unique_products": len({product["item_id"] for product in products}),
            "products_with_unknown_fields": sum(1 for product in products if product.get("unknown_fields")),
            "claims_valid": claims_valid, "claims_rejected": claims_rejected,
            "unknown_count": unknown_count, "qwen_calls": qwen_comparison,
            "preference_observations": len(preference_observations),
            "mean_skill_latency_ms": round(statistics.mean(comparison_latencies), 2) if comparison_latencies else None,
            "tool_calls": sorted({call for row in successful for call in (row["result"].get("tool_calls") or [])}),
        },
        "clarification": {
            "cases": len(clarification), "awaiting_questions": len(asked),
            "memory_updates": len(updated),
            "conflict_resolved": bool(conflict and any(
                item["value"] == "wired" and item["status"] == "removed"
                for item in conflict["resume_result"]["inactive_preferences"])),
            "withdrawal_removed": bool(withdrawal and withdrawal["resume_result"]["update"]["removed"] >= 1),
            "duplicate_idempotent": bool(duplicate and duplicate["second"]["already_processed"]),
            "duplicate_active_count_stable": bool(
                duplicate and len(duplicate["second"]["active_preferences"]) == len(duplicate["first"]["active_preferences"])),
            "unanswered_not_finalized": bool(unanswered and unanswered["ask_result"]["finalize_allowed"] is False
                                             and unanswered["agent_before"]["selected"] is None),
            "cross_user_rejected": bool(cross and cross["cross_user_read"]["rejected"]),
            "reply_label": "explicit_test_user_reply (not evaluator, not real online user)",
        },
        "skill_registry": ["preference-clarification", "shopping-comparison"],
        "search_backend": ((comparison[0].get("result") or {}).get("agent_context") or {}).get("search_mode") if comparison else None,
        "evidence": "P15 Evidence Bundle + automated Claim-Evidence Validation; semantic truth not human-verified",
        "qwen_used_in_comparison": qwen_comparison > 0,
        "notes": ["comparison latency is same-run wall-clock on these P16 tasks; it is not a direct optimization claim over P15's different explanation tasks",
                  "all reported user replies are explicitly labelled test inputs, not real online user behaviour"],
    }
    (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (RUNS / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            result = row.get("result")
            handle.write(json.dumps({
                "case_id": row["case_id"], "skill": row["skill"], "task_id": row.get("task_id"),
                "status": (result or {}).get("status") if row["skill"] == "shopping-comparison"
                else (row.get("ask_result") or {}).get("status"),
                "final_answer": (result or {}).get("final_answer") if row["skill"] == "shopping-comparison"
                else (row.get("ask_result") or {}).get("final_answer"),
                "error": row.get("error"),
            }, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
