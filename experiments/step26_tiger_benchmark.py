"""Step 2.6 H8: the canonical shared-evaluator report for the ``tiger_fp32`` arm.

NOT RUN YET.  H8 is gated behind H7 (the full 20 000-case CERTIFIED retrieval) and an explicit
authorization.  This module exists so that H7/H8 are executable the moment they are authorized,
and so that the wiring it depends on is auditable now.

It computes nothing that could be mistaken for a result until it is invoked.  It reuses the
frozen protocol end to end: the same cohort selection, the same ``EvaluationCase`` objects, the
same full catalogue, the same shared evaluator (``benchmark_public.evaluate_arm``) and the same
K values.  No TIGER-specific metric implementation exists anywhere in this file.

Frozen reference values are copied in as *references only* and are never recomputed:
    SASRec    Recall@10 = 0.01350
    Two-Tower Recall@10 = 0.01435
    GenRec-v0 Recall@10 = 0.00185
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments.benchmark_public as B  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

ARM_NAME = "tiger_fp32"
FROZEN_REFERENCES = {
    "SASRec_Recall@10": 0.01350,
    "TwoTower_Recall@10": 0.01435,
    "GenRecv0_Recall@10": 0.00185,
}
DEFAULT_OUT = REPO / "runs" / "step26_tiger_benchmark" / "run.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True,
                        help="producer output directory (retrieval_candidates.jsonl + summary)")
    parser.add_argument("--size", type=int, default=B.DEFAULT_COHORT_SIZE)
    parser.add_argument("--batch-size", type=int, default=B.DEFAULT_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sequences = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
    mappings = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
    # Exactly the accepted runner's extraction (experiments/phase5_benchmark.py): SplitReport is a
    # dataclass, and the catalogue size is `catalog_size`.
    cases, split_report = load_cohort_from_artifacts(sequences, mappings)
    selection = B.cohort_from_cases(cases, size=args.size)
    cohort = list(selection["cases"])
    num_items = int(split_report.catalog_size)

    summary = json.loads((args.artifact / "retrieval_summary.json").read_text())
    if summary["producer"] != "tiger_public.retrieve_cli":
        raise SystemExit("refusing: the artifact was not produced by the Step 2.6 retrieval module")
    if summary["status"] != "CERTIFIED" or not summary["comparison_eligible"]:
        raise SystemExit(
            "refusing: a canonical report requires CERTIFIED, comparison_eligible retrieval; "
            f"got status={summary['status']!r} comparison_eligible={summary['comparison_eligible']!r}"
        )

    sys.path.insert(0, str(REPO / "experiments"))
    from tiger_retrieval_arm import arm_tiger_fp32  # noqa: E402

    batches = arm_tiger_fp32(
        candidates_path=args.artifact / "retrieval_candidates.jsonl",
        summary_path=args.artifact / "retrieval_summary.json",
        num_items=num_items,
    )

    started = time.perf_counter()
    metrics = B.evaluate_arm(cases=cohort, num_items=num_items, batches=batches,
                             batch_size=args.batch_size)
    wall = time.perf_counter() - started

    payload = {
        "arm": ARM_NAME,
        "retrieval_status": summary["status"],
        "comparison_eligible": summary["comparison_eligible"],
        "certificate": summary["certificate"],
        "metrics": metrics,
        "evaluated_users": len(cohort),
        "num_items": num_items,
        "k_values": list(B.K_VALUES),
        "cohort_selection": {k: v for k, v in selection.items() if k != "cases"},
        "wall_seconds": round(wall, 3),
        "frozen_references_not_recomputed": FROZEN_REFERENCES,
        "hash_chain": {
            "checkpoint_tiger_pt": summary["hashes"]["checkpoint_tiger_pt"],
            "sid_semantic_ids": summary["hashes"]["sid_semantic_ids"],
            "sid_layout": summary["hashes"]["sid_layout"],
            "generator_layout": summary["hashes"]["generator_layout"],
            "catalogue_json": summary["hashes"]["catalogue_json"],
            "retrieval_candidates_jsonl": summary["hashes"]["candidates_jsonl"],
            "retrieval_summary_json": sha256_file(args.artifact / "retrieval_summary.json"),
            "benchmark_protocol_module": sha256_file(REPO / "experiments" / "benchmark_public.py"),
            "shared_evaluator_module": sha256_file(
                REPO / "recommendation" / "evaluation" / "batched.py"),
            "arm_module": sha256_file(REPO / "experiments" / "tiger_retrieval_arm.py"),
        },
        "note": (
            "Metrics come from the unchanged shared evaluator. TIGER must not be characterised as "
            "better or worse than SASRec / Two-Tower / GenRec-v0 on the basis of anything except "
            "these CERTIFIED numbers."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({"arm": ARM_NAME, "retrieval_status": payload["retrieval_status"],
                      "evaluated_users": payload["evaluated_users"],
                      "wall_seconds": payload["wall_seconds"],
                      "out": str(args.out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
