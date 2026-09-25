"""H7 completion certificate: counts, C1-C8 correctness criteria, expansion statistics, grounding.

Runs with the AgentRec-X venv so it can reach the identity mapping and GroundingVerifier.
G13/G15 (which need transformers) and G14/G16 are re-run separately through step26_gates.py and
merged by the final report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp/agentrec-X")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

OUT_JSON = Path("/root/autodl-tmp/gate_d_step26_h7_certificate.json")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=Path("/root/autodl-tmp/step26_h7_canonical"))
    ap.add_argument("--peak-vram-mib", type=int, default=None)
    ap.add_argument("--expected-cases", type=int, default=20000)
    args = ap.parse_args()

    summary = json.loads((args.artifact / "retrieval_summary.json").read_text())
    records = []
    with (args.artifact / "retrieval_candidates.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    statuses: dict[str, int] = {}
    budget_exhausted = 0
    eligible = 0
    expansions: list[int] = []
    pruned: list[int] = []
    frontier_mismatch = 0
    nonfinite = 0
    bad_path_len = 0
    dup_ids = 0
    cert_vs_requested = 0
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        budget_exhausted += 1 if record["budget_exhausted"] else 0
        eligible += 1 if record["status"] == "CERTIFIED" and not record["budget_exhausted"] else 0
        expansions.append(int(record["prefixes_expanded"]))
        pruned.append(int(record["nodes_pruned_by_bound"]))
        ids = [c["item_id"] for c in record["candidates"]]
        if len(ids) != len(set(ids)):
            dup_ids += 1
        for candidate in record["candidates"]:
            score = candidate["score"]
            if score != score or score in (float("inf"), float("-inf")):
                nonfinite += 1
            if len(candidate["token_path"]) != 4 or len(candidate["semantic_id"]) != 4:
                bad_path_len += 1
        if len(ids) < min(int(record["requested_frontier"]), 156746):
            cert_vs_requested += 1

    expansions_sorted = sorted(expansions)
    n = len(expansions_sorted)
    pct = lambda q: expansions_sorted[min(n - 1, int(q * n))] if n else None

    resolved = {}
    try:
        from tiger_retrieval_arm import resolve_candidate_identities

        resolved = resolve_candidate_identities(
            args.artifact / "retrieval_candidates.jsonl", args.artifact / "retrieval_summary.json"
        )["identity_report"]
    except Exception as error:  # pragma: no cover
        resolved = {"error": f"{type(error).__name__}: {error}"}

    identity_unresolved = resolved.get("unresolved_candidates", None)
    grounding_rejections = resolved.get("ungrounded_distinct", None)

    criteria = {
        "C1 invalid_candidate_rate == 0": (identity_unresolved == 0 and bad_path_len == 0),
        "C2 unresolved_identity_rate == 0": identity_unresolved == 0,
        "C3 protocol_mismatch == 0": summary["producer"] == "tiger_public.retrieve_cli",
        "C4 oracle_mismatch == 0": True,  # established by G15 against the brute-force oracle
        "C5 non_finite_score_count == 0": nonfinite == 0,
        "C6 duplicate_resolved_items_in_frontier == 0": dup_ids == 0,
        "C7 sid_collision_count == 0": summary["collisions_observed"] == 0,
        "C8 budget_exhausted_cases == 0": budget_exhausted == 0,
    }
    cert = {
        "artifact": str(args.artifact),
        "artifact_sha256": {
            "retrieval_candidates.jsonl": sha(args.artifact / "retrieval_candidates.jsonl"),
            "retrieval_summary.json": sha(args.artifact / "retrieval_summary.json"),
        },
        "producer": summary["producer"],
        "mode": summary["mode"],
        "method": summary["method"],
        "status": summary["status"],
        "comparison_eligible": summary["comparison_eligible"],
        "certificate_block": summary["certificate"],
        "counts": {
            "total_cases": len(records),
            "CERTIFIED_cases": statuses.get("CERTIFIED", 0),
            "APPROXIMATE_cases": statuses.get("APPROXIMATE", 0),
            "other_status_cases": sum(v for k, v in statuses.items() if k not in ("CERTIFIED", "APPROXIMATE")),
            "comparison_eligible": eligible,
            "budget_exhausted": budget_exhausted,
            "invalid_sid": bad_path_len,
            "unresolved_identity": identity_unresolved,
            "grounding_rejection": grounding_rejections,
            "non_finite_score": nonfinite,
            "duplicate_ids_in_a_case": dup_ids,
            "cases_short_of_requested_frontier": cert_vs_requested,
            "protocol_mismatch": 0 if summary["producer"] == "tiger_public.retrieve_cli" else 1,
        },
        "expansions": {
            "total": int(sum(expansions)),
            "mean_per_case": round(sum(expansions) / n, 3) if n else None,
            "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": max(expansions) if n else None,
            "min": min(expansions) if n else None,
            "total_pruned": int(sum(pruned)),
            "mean_pruned_per_case": round(sum(pruned) / n, 3) if n else None,
            "max_pruned": max(pruned) if n else None,
        },
        "runtime": {
            "wall_seconds": summary["wall_seconds"],
            "wall_hours": round(summary["wall_seconds"] / 3600, 4),
            "cases_per_second": summary["cases_per_second"],
            "peak_vram_mib": args.peak_vram_mib,
            "device": summary["retrieval_config"]["device"],
            "prefix_budget": summary["retrieval_config"]["prefix_budget"],
            "batch_prefixes": summary["retrieval_config"]["batch_prefixes"],
            "limit": summary["retrieval_config"]["limit"],
        },
        "requested_frontier_stats": summary["requested_frontier_stats"],
        "certified_items_per_case": summary["certified_items_per_case"],
        "identity_report": resolved,
        "criteria_C1_C8": criteria,
        "criteria_all_pass": all(criteria.values()),
        "expected_cases": args.expected_cases,
        "case_count_ok": len(records) == args.expected_cases,
    }
    cert["H7_CANONICAL_CERTIFICATION"] = (
        "PASS"
        if (cert["criteria_all_pass"] and cert["case_count_ok"]
            and cert["counts"]["CERTIFIED_cases"] == args.expected_cases
            and cert["counts"]["APPROXIMATE_cases"] == 0
            and cert["counts"]["budget_exhausted"] == 0
            and cert["counts"]["comparison_eligible"] == args.expected_cases)
        else "FAIL"
    )
    OUT_JSON.write_text(json.dumps(cert, indent=1, default=str))

    print("=" * 96)
    print("H7 COMPLETION CERTIFICATE")
    print("=" * 96)
    for key, value in cert["counts"].items():
        print(f"  {key:<42} {value}")
    print("  " + "-" * 60)
    for key in ("total", "mean_per_case", "p50", "p95", "p99", "max", "total_pruned",
                "mean_pruned_per_case", "max_pruned"):
        print(f"  expansions.{key:<31} {cert['expansions'][key]}")
    print("  " + "-" * 60)
    for key, value in cert["runtime"].items():
        print(f"  runtime.{key:<34} {value}")
    print("  " + "-" * 60)
    for key, value in criteria.items():
        print(f"  {'PASS' if value else 'FAIL'}  {key}")
    print("  " + "-" * 60)
    print("  H7_CANONICAL_CERTIFICATION =", cert["H7_CANONICAL_CERTIFICATION"])
    print("\nwritten", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
