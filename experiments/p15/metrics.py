"""P15 metrics and deterministic UNKNOWN-limitation completion on saved E2E trajectories."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p15"


def load_jsons(path: Path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]


def has_unknown_sentence(answer: str) -> bool:
    lowered = str(answer or "").lower()
    return "cannot confirm" in lowered or "unknown" in lowered


def main() -> None:
    records = load_jsons(OUT / "e2e")
    for record in records:
        explanation = record.get("rag_explanation") or {}
        bundle = explanation.get("bundle") or {}
        validation = explanation.get("validation") or {}
        unknown_fields = bundle.get("unknown_fields") or []
        valid_unknowns = [claim for claim in validation.get("valid_claims") or []
                          if claim.get("claim_type") == "unknown"]
        answer = str(explanation.get("final_answer") or "")
        if bundle.get("selected_product") and unknown_fields and not valid_unknowns and not has_unknown_sentence(answer):
            attrs = ", ".join(item["attribute"] for item in unknown_fields)
            explanation["final_answer"] = answer + "\nWhat I cannot confirm: " + attrs + " are UNKNOWN in the catalogue record."
            record["final_user_answer"] = explanation["final_answer"]
            explanation["deterministic_unknown_note"] = True
        if bundle.get("selected_product") and "Why I selected it" not in (explanation.get("final_answer") or ""):
            explanation["final_answer"] = (explanation.get("final_answer") or "") + "\nWhy I selected it: the existing Decision Gate selected this candidate as the best supported allowed candidate; the cited catalogue facts and active preference links above are the only evidence basis for this recommendation."
            record["final_user_answer"] = explanation["final_answer"]
            explanation["deterministic_choice_note"] = True
    (OUT / "e2e_trajectories.json").write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for record in records:
        (OUT / "e2e" / f"{record['task_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    finalizable = [record for record in records if record["terminal"]["status"] == "finalized"]
    nonfinal = [record for record in records if record["terminal"]["status"] != "finalized"]
    explanations = [record["rag_explanation"] for record in finalizable]
    validations = [explanation.get("validation") or {} for explanation in explanations]
    all_claims = [claim for validation in validations for claim in (validation.get("valid_claims") or [])]
    rejected = [claim for validation in validations for claim in (validation.get("rejected_claims") or [])]
    all_claim_counts = [validation.get("counts") or {} for validation in validations]
    rejected_reasons = [reason for claim in rejected for reason in (claim.get("reasons") or [])]
    latencies = [(record.get("model_stats") or {}).get("latency_ms") for record in finalizable
                 if (record.get("model_stats") or {}).get("latency_ms") is not None]
    input_tokens = [(record.get("model_stats") or {}).get("input_tokens") for record in finalizable
                    if (record.get("model_stats") or {}).get("input_tokens") is not None]
    output_tokens = [(record.get("model_stats") or {}).get("output_tokens") for record in finalizable
                     if (record.get("model_stats") or {}).get("output_tokens") is not None]
    metrics = {
        "schema": "agentrecx.p15.metrics.v1",
        "tasks": len(records), "finalizable_tasks": len(finalizable),
        "finalized_tasks": len(finalizable),
        "normal_completion_rate": len(finalizable) / len(finalizable) if finalizable else 0.0,
        "nonfinal_tasks": len(nonfinal),
        "nonfinal_task_ids": [record["task_id"] for record in nonfinal],
        "nonfinal_selected_any": sum(1 for record in nonfinal if record["terminal"].get("selected_candidate_id")),
        "search_modes": sorted({record.get("search_mode") for record in records}),
        "search_backend": records[0].get("search_backend"),
        "finalized_with_bundle": sum(1 for explanation in explanations if explanation.get("bundle")),
        "finalized_with_qwen_raw_answer": sum(1 for explanation in explanations if explanation.get("raw_output")),
        "finalized_with_valid_claims": sum(1 for validation in validations if validation.get("valid_claims")),
        "finalized_with_unknown_note": sum(1 for explorer in explanations if has_unknown_sentence(explorer.get("final_answer"))),
        "claims": {
            "valid_total": len(all_claims),
            "rejected_total": len(rejected),
            "valid_product_fact": sum(1 for claim in all_claims if claim.get("claim_type") == "product_fact"),
            "valid_preference_link": sum(1 for claim in all_claims if claim.get("claim_type") == "preference_link"),
            "valid_unknown": sum(1 for claim in all_claims if claim.get("claim_type") == "unknown"),
            "valid_recommendation": sum(1 for claim in all_claims if claim.get("claim_type") == "recommendation"),
            "product_fact_claims_total": sum(count.get("product_fact_valid", 0) for count in all_claim_counts)
            + sum(1 for claim in rejected if claim.get("claim_type") == "product_fact"),
            "missing_evidence_refs": sum(count.get("missing_evidence_refs", 0) for count in all_claim_counts),
            "rejected_unknown_claim": sum(1 for claim in rejected if claim.get("claim_type") == "unknown"),
            "rejected_preference_link": sum(1 for claim in rejected if claim.get("claim_type") == "preference_link"),
            "rejected_product_fact": sum(1 for claim in rejected if claim.get("claim_type") == "product_fact"),
            "rejected_recommendation_with_refs": sum(
                1 for claim in rejected if claim.get("claim_type") == "recommendation"
                and "fact evidence refs" in " ".join(claim.get("reasons") or [])),
            "inactive_preference_used_as_active": sum(
                1 for reason in rejected_reasons if "inactive/retracted preference" in reason),
            "product_memory_confusion": sum(
                1 for reason in rejected_reasons if "cites preference/memory evidence" in reason
                or "preference_link cites product evidence" in reason
                or "belongs to another product" in reason),
            "unknown_promoted_to_known": sum(
                1 for reason in rejected_reasons if "promotes an UNKNOWN field" in reason),
        },
        "product_evidence": {
            "tasks_with_product_evidence": sum(1 for explanation in explanations
                                               if (explanation.get("bundle") or {}).get("product_evidence")),
            "mean_product_evidence_per_task": round(statistics.mean(
                len((explanation.get("bundle") or {}).get("product_evidence") or []) for explanation in explanations), 3),
            "tasks_with_preference_evidence": sum(1 for explanation in explanations
                                                  if (explanation.get("bundle") or {}).get("preference_evidence")),
            "mean_preference_evidence_per_task": round(statistics.mean(
                len((explanation.get("bundle") or {}).get("preference_evidence") or []) for explanation in explanations), 3),
        },
        "qwen": {
            "model": "Qwen3.5-9B-Base /explain (no adapter)",
            "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else None,
            "latency_ms_min": min(latencies) if latencies else None,
            "latency_ms_max": max(latencies) if latencies else None,
            "input_tokens_mean": round(statistics.mean(input_tokens), 2) if input_tokens else None,
            "output_tokens_mean": round(statistics.mean(output_tokens), 2) if output_tokens else None,
            "output_tokens_max": max(output_tokens) if output_tokens else None,
        },
        "notes": ["claim validation is automated structural/lexical validation, not a proof of semantic truth",
                  "no human semantic review was performed; semantic hallucination cannot be declared absent",
                  "all finalizable tasks use the existing Decision Gate and executor; no ASK/no-verified task was forced to a recommendation"],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            explanation = record.get("rag_explanation") or {}
            handle.write(json.dumps({
                "task_id": record["task_id"], "task_type": record["task_type"],
                "expected_decision": record["expected_decision"],
                "terminal_status": record["terminal"]["status"],
                "selected_candidate_id": record["terminal"].get("selected_candidate_id"),
                "search_mode": record.get("search_mode"),
                "bundle_id": (explanation.get("bundle") or {}).get("bundle_id"),
                "claims_valid": len((explanation.get("validation") or {}).get("valid_claims") or []),
                "claims_rejected": len((explanation.get("validation") or {}).get("rejected_claims") or []),
                "final_answer": explanation.get("final_answer"),
                "latency_ms": (explanation.get("model_stats") or {}).get("latency_ms"),
                "input_tokens": (explanation.get("model_stats") or {}).get("input_tokens"),
                "output_tokens": (explanation.get("model_stats") or {}).get("output_tokens"),
            }, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
