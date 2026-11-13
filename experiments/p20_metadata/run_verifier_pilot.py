"""GPU pilot: deterministic P15 validator vs existing-Qwen Independent Evidence Verifier.

The verifier receives bundles produced by P15, but only product/claim/evidence fields;
it never receives the P15 validation result, the Planner rationale, or the Gate decision.

Outputs (git-ignored):
  runs/p20_metadata/verifier_pilot.json
  runs/p20_metadata/verifier_batches.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from recommendation.control.evidence_verifier import QwenVerifierClient, verify_claims

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p20_metadata"
QWEN_URL = "http://127.0.0.1:18029"
BATCH_SIZE = 1


def _load_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundles = json.loads((OUT / "verifier_bundles.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in
             (OUT / "claim_evidence_cases.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    return bundles, cases


def _evidence(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for item in bundle.get("product_evidence") or []:
        evidence.append(dict(item))
    for item in bundle.get("preference_evidence") or []:
        evidence.append({
            "evidence_id": item.get("evidence_id"),
            "product_id": None,
            "raw_field": "preference",
            "span_text": item.get("source_text") or item.get("value"),
            "raw_excerpt": item.get("source_text") or item.get("value"),
            "attribute": "preference",
            "verification_status": item.get("verification_status"),
            "value": item.get("value"),
            "polarity": item.get("polarity"),
            "active": item.get("active"),
            "user_id": item.get("user_id"),
        })
    return evidence


def _verifier_claim(case: Mapping[str, Any]) -> dict[str, Any]:
    claim = dict(case["claim"])
    claim["claim_id"] = case["case_id"]
    return claim


def _metrics(cases: list[dict[str, Any]], verdicts: dict[str, str], baseline: dict[str, bool]) -> dict[str, Any]:
    supported = [c for c in cases if c["label"] == "SUPPORTED"]
    unsupported = [c for c in cases if c["label"] != "SUPPORTED"]
    flagged = [c for c in cases if verdicts.get(c["case_id"]) not in (None, "SUPPORTED")]
    true_flagged = [c for c in flagged if c["label"] != "SUPPORTED"]
    baseline_flagged = [c for c in cases if baseline.get(c["case_id"]) is False]
    baseline_true = [c for c in baseline_flagged if c["label"] != "SUPPORTED"]
    return {
        "cases": len(cases),
        "supported_cases": len(supported),
        "unsupported_cases": len(unsupported),
        "verifier_supported_accuracy": round(
            sum(1 for c in supported if verdicts.get(c["case_id"]) == "SUPPORTED") / max(1, len(supported)), 3),
        "verifier_unsupported_recall": round(len(true_flagged) / max(1, len(unsupported)), 3),
        "verifier_unsupported_precision": round(len(true_flagged) / max(1, len(flagged)), 3),
        "verifier_false_rejection_rate": round(
            sum(1 for c in supported if verdicts.get(c["case_id"]) not in (None, "SUPPORTED"))
            / max(1, len(supported)), 3),
        "verifier_insufficient_correct": sum(
            1 for c in cases if c["label"] == "INSUFFICIENT_EVIDENCE"
            and verdicts.get(c["case_id"]) == "INSUFFICIENT_EVIDENCE"),
        "verifier_contradicted_correct": sum(
            1 for c in cases if c["label"] == "CONTRADICTED"
            and verdicts.get(c["case_id"]) == "CONTRADICTED"),
        "baseline_accept_rate": round(sum(1 for c in cases if baseline.get(c["case_id"])) / max(1, len(cases)), 3),
        "baseline_structural_recall_on_unsupported": round(
            len(baseline_true) / max(1, len(unsupported)), 3),
        "baseline_false_rejection_rate": round(
            sum(1 for c in supported if baseline.get(c["case_id"]) is False) / max(1, len(supported)), 3),
        "invalid_verdict_count": sum(1 for c in cases if c["case_id"] not in verdicts),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bundles, cases = _load_cases()
    client = QwenVerifierClient(QWEN_URL, timeout=360.0)
    all_verdicts: dict[str, str] = {}
    batch_records: list[dict[str, Any]] = []
    token_in = token_out = latency = 0.0
    errors: list[str] = []
    for bundle_id, bundle in bundles.items():
        bundle_cases = [c for c in cases if c["bundle_id"] == bundle_id]
        for start in range(0, len(bundle_cases), BATCH_SIZE):
            chunk = bundle_cases[start:start + BATCH_SIZE]
            claims = [_verifier_claim(case) for case in chunk]
            started = time.perf_counter()
            result = verify_claims(
                client=client,
                product_id=bundle.get("selected_product", {}).get("item_id"),
                claims=claims,
                evidence=_evidence(bundle),
                unknown_fields=[item.get("attribute") for item in bundle.get("unknown_fields") or []],
                model="Qwen3.5-9B-Base",
            )
            wall = round(time.perf_counter() - started, 3)
            stats = result.model_stats or {}
            latency += float(stats.get("latency_ms") or wall * 1000.0)
            token_in += int(stats.get("input_tokens") or 0)
            token_out += int(stats.get("output_tokens") or 0)
            errors.extend(result.errors)
            by_id = {item.claim_id: item.verdict for item in result.verdicts}
            all_verdicts.update(by_id)
            batch_records.append({
                "bundle_id": bundle_id,
                "case_ids": [case["case_id"] for case in chunk],
                "status": result.status,
                "id_alignment": result.id_alignment,
                "verdicts": by_id,
                "errors": result.errors,
                "latency_ms": stats.get("latency_ms"),
                "wall_seconds": wall,
                "input_tokens": stats.get("input_tokens"),
                "output_tokens": stats.get("output_tokens"),
                "raw_output": result.raw_output,
            })
            print(json.dumps({
                "bundle": bundle_id, "cases": [case["case_id"] for case in chunk],
                "verifier_status": result.status, "verdicts": by_id,
                "input_tokens": stats.get("input_tokens"), "output_tokens": stats.get("output_tokens"),
                "latency_ms": stats.get("latency_ms"),
            }, ensure_ascii=False), flush=True)
    baseline = {case["case_id"]: bool(case.get("baseline", {}).get("accepted")) for case in cases}
    all_metrics = _metrics(cases, all_verdicts, baseline)
    combined_verdicts = {
        case["case_id"]: (all_verdicts.get(case["case_id"]) if baseline[case["case_id"]]
                          else "INSUFFICIENT_EVIDENCE")
        for case in cases
    }
    all_combined = _metrics(cases, combined_verdicts, baseline)
    provenance_categories = {"preference_wrong_source", "preference_missing_ref",
                             "metadata_source_conflict", "contradicted_form_factor"}
    provenance_cases = [c for c in cases if c["category"] in provenance_categories]
    combined_detected = sum(1 for c in provenance_cases
                            if combined_verdicts[c["case_id"]] != "SUPPORTED")
    dev_cases = [c for c in cases if c["split"] == "dev"]
    holdout_cases = [c for c in cases if c["split"] == "holdout"]
    verdict_detail = []
    for case in cases:
        verdict_detail.append({
            **case,
            "verifier_verdict": all_verdicts.get(case["case_id"]),
            "combined_verdict": combined_verdicts.get(case["case_id"]),
            "baseline_accepted": baseline.get(case["case_id"]),
        })
    payload = {
        "schema": "agentrecx.p20.verifier_pilot.v1",
        "model": "Qwen3.5-9B-Base /explain (same checkpoint, independent verifier role)",
        "gpu": "AutoDL RTX 4090 24GB",
        "batch_size": BATCH_SIZE,
        "batches": len(batch_records),
        "cases": cases,
        "verdicts": verdict_detail,
        "batch_records": batch_records,
        "metrics": {
            "all": all_metrics,
            "dev": _metrics(dev_cases, all_verdicts, baseline),
            "holdout": _metrics(holdout_cases, all_verdicts, baseline),
            "combined": all_combined,
            "combined_dev": _metrics(dev_cases, combined_verdicts, baseline),
            "combined_holdout": _metrics(holdout_cases, combined_verdicts, baseline),
            "citation_or_conflict_cases": len(provenance_cases),
            "citation_or_conflict_detected_by_combined": combined_detected,
            "id_alignment_counts": {
                alignment: sum(1 for batch in batch_records if batch.get("id_alignment") == alignment)
                for alignment in {batch.get("id_alignment") for batch in batch_records}
            },
            "total_input_tokens": token_in,
            "total_output_tokens": token_out,
            "total_latency_ms": round(latency, 2),
            "errors": len(errors),
        },
        "label_status": "provisional_rule_authored_not_human_verified",
        "failure_errors": errors,
    }
    (OUT / "verifier_pilot.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "verifier_batches.jsonl").open("w", encoding="utf-8") as handle:
        for record in batch_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(payload["metrics"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
