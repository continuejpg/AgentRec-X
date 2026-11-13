"""Qwen-only (no deterministic prechecks) comparison for the evidence verifier."""

from __future__ import annotations

import json
import time
from pathlib import Path

from recommendation.control.evidence_verifier import QwenVerifierClient, verify_claims

from experiments.p20_metadata.run_verifier_pilot import (
    OUT, QWEN_URL, _evidence, _load_cases, _metrics, _verifier_claim,
)


def main() -> int:
    bundles, cases = _load_cases()
    client = QwenVerifierClient(QWEN_URL, timeout=360.0)
    verdicts: dict[str, str] = {}
    records: list[dict] = []
    errors: list[str] = []
    token_in = token_out = latency = 0.0
    for case in cases:
        bundle = bundles[case["bundle_id"]]
        claim = _verifier_claim(case)
        started = time.perf_counter()
        result = verify_claims(
            client=client,
            product_id=str((bundle.get("selected_product") or {}).get("item_id")),
            claims=[claim], evidence=_evidence(bundle),
            unknown_fields=[item.get("attribute") for item in bundle.get("unknown_fields") or []],
            model="Qwen3.5-9B-Base", use_prechecks=False,
        )
        wall = round(time.perf_counter() - started, 3)
        stats = result.model_stats or {}
        latency += float(stats.get("latency_ms") or wall * 1000.0)
        token_in += int(stats.get("input_tokens") or 0)
        token_out += int(stats.get("output_tokens") or 0)
        errors.extend(result.errors)
        by_id = {item.claim_id: item.verdict for item in result.verdicts}
        verdicts.update(by_id)
        records.append({"case_id": case["case_id"], "status": result.status,
                        "verdict": by_id.get(case["case_id"]), "errors": result.errors,
                        "id_alignment": result.id_alignment,
                        "latency_ms": stats.get("latency_ms"), "wall_seconds": wall,
                        "input_tokens": stats.get("input_tokens"),
                        "output_tokens": stats.get("output_tokens"),
                        "raw_output": result.raw_output})
        print(json.dumps({"case": case["case_id"], "status": result.status,
                          "verdict": by_id.get(case["case_id"])}, ensure_ascii=False), flush=True)
    baseline = {case["case_id"]: bool(case.get("baseline", {}).get("accepted")) for case in cases}
    payload = {
        "schema": "agentrecx.p20.verifier_qwen_only.v1",
        "model": "Qwen3.5-9B-Base /explain",
        "mode": "qwen_only_no_prechecks",
        "batch_size": 1,
        "cases": cases,
        "records": records,
        "metrics": {
            **_metrics(cases, verdicts, baseline),
            "total_input_tokens": token_in, "total_output_tokens": token_out,
            "total_latency_ms": round(latency, 2), "errors": len(errors),
        },
        "label_status": "provisional_rule_authored_not_human_verified",
    }
    (OUT / "verifier_qwen_only.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
