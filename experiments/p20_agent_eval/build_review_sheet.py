"""Build a human-reviewable Claim–Evidence sheet from the Complete Shadow run.

No label is promoted to human_verified.  The sheet is deliberately pending review/blank.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/p20_agent_eval"
SHADOW = RUNS / "complete_shadow.json"


def _evidence_map(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence = {}
    for item in bundle.get("product_evidence") or []:
        evidence[str(item.get("evidence_id"))] = item
    for item in bundle.get("preference_evidence") or []:
        evidence[str(item.get("evidence_id"))] = {
            **item, "raw_field": "preference", "span_text": item.get("source_text"),
            "raw_excerpt": item.get("source_text"),
        }
    return evidence


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in payload["tasks"]:
        validation = task.get("claim_validation") or {}
        bundle = task.get("evidence_bundle") or {}
        evidence = _evidence_map(bundle)
        verdicts = {v.get("claim_id"): v for v in
                    ((task.get("shadow_evidence_verifier") or {}).get("verdicts") or [])}
        valid = {v.get("claim_id"): v for v in validation.get("valid_claims") or []}
        rejected = {v.get("claim_id"): v for v in validation.get("rejected_claims") or []}
        for claim_id in sorted(set(valid) | set(rejected)):
            item = valid.get(claim_id) or rejected.get(claim_id) or {}
            claim = item.get("claim") or {}
            refs = list(claim.get("evidence_refs") or [])
            cited = [evidence.get(ref) for ref in refs]
            cited = [c for c in cited if c]
            rows.append({
                "task_id": task.get("task_id"),
                "product_id": task.get("selected_product_id"),
                "claim_id": claim_id,
                "claim_type": claim.get("claim_type") or item.get("claim_type"),
                "claim_text": claim.get("claim_text") or item.get("claim_text"),
                "cited_refs": refs,
                "cited_evidence": [
                    {"evidence_id": c.get("evidence_id"), "raw_field": c.get("raw_field"),
                     "span_text": c.get("span_text"), "raw_excerpt": c.get("raw_excerpt"),
                     "source_record": c.get("source_record")}
                    for c in cited
                ],
                "cited_evidence_text": " | ".join(
                    f"{c.get('evidence_id')}:{c.get('span_text')}({c.get('raw_field')})"
                    for c in cited),
                "p15_status": item.get("status") or task.get("claim_validation", {}).get("status"),
                "p15_is_accepted": claim_id in valid,
                "p15_reasons": item.get("reasons") or [],
                "verifier_verdict": (verdicts.get(claim_id) or {}).get("verdict"),
                "verifier_reason": (verdicts.get(claim_id) or {}).get("reason"),
                "verifier_refs": (verdicts.get(claim_id) or {}).get("evidence_refs") or [],
                "human_review_status": "pending_independent_human_review",
                "human_verdict": "",
                "human_notes": "",
                "provisional_note": "P15 automated_validation is not semantic truth; no human gold",
            })
    return rows


def _priority(row: dict[str, Any]) -> int:
    if row["p15_is_accepted"] and row["verifier_verdict"] not in (None, "SUPPORTED"):
        return 0
    if not row["p15_is_accepted"]:
        return 1
    text = (row["claim_text"] or "").lower()
    if any(word in text for word in ("ipx", "sweat", "mechanical", "over-ear", "on-ear")):
        return 2
    if row["claim_type"] == "preference_link":
        return 3
    return 4


def main() -> None:
    payload = json.loads(SHADOW.read_text(encoding="utf-8"))
    all_rows = _rows(payload)
    selected = sorted(all_rows, key=lambda row: (_priority(row), row["task_id"], row["claim_id"]))[:20]
    doc = {
        "schema": "agentrecx.p20.claim_review_sheet.v1",
        "source": "runs/p20_agent_eval/complete_shadow.json",
        "label_status": "provisional_pending_independent_human_review",
        "review_instructions": [
            "Judge only whether the cited evidence supports the claim.",
            "Do not use world knowledge or assume catalogue facts not present in the evidence.",
            "Fill human_verdict with SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE.",
            "Mark human_review_status=independently_reviewed only after a real independent review.",
        ],
        "claims": selected,
    }
    (RUNS / "claim_review_sheet.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["task_id", "product_id", "claim_id", "claim_type", "claim_text", "cited_refs",
              "cited_evidence_text", "p15_status", "p15_is_accepted", "p15_reasons",
              "verifier_verdict", "verifier_reason", "human_verdict",
              "human_review_status", "human_notes"]
    with (RUNS / "claim_review_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({
                **row, "cited_refs": "|".join(row["cited_refs"]),
                "p15_reasons": "|".join(row["p15_reasons"]),
            })
    lines = [
        "# Claim–Evidence Review Sheet (PROVISIONAL)",
        "",
        "No label here is human-verified. P15 `automated_validated` is not semantic truth.",
        "Review only against the cited evidence; do not use world knowledge.",
        "",
        "| Task | Product | Claim | Cited evidence | P15 | Qwen | Human verdict | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        claim = (row["claim_text"] or "").replace("|", "/")[:180]
        lines.append(
            f"| {row['task_id']} | {row['product_id']} | {claim} | "
            f"{(row['cited_evidence_text'] or '')[:220].replace('|', '/')} | {row['p15_status']} | "
            f"{row['verifier_verdict']} |  |  |")
    (RUNS / "claim_review_sheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"claims_total": len(all_rows), "review_rows": len(selected),
                      "out": str(RUNS / "claim_review_sheet.json")}, indent=1))


if __name__ == "__main__":
    main()
