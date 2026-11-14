"""Write the machine-readable manifest for the Phase 1 Memory Contribution run.

The manifest pins the inputs, the benchmark configuration, the exact artifacts and
their hashes.  It intentionally records that no model, retrieval, P18 asset or GPU
was used.

Run from the repository root::

    .venv/bin/python -m experiments.memory_contribution.build_manifest
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from typing import Any

from experiments.memory_contribution import common

OUT = common.OUT_DIR
SOURCE_FILES = [
    common.REPO / "experiments/memory_contribution/__init__.py",
    common.REPO / "experiments/memory_contribution/common.py",
    common.REPO / "experiments/memory_contribution/run_fixed_pool.py",
    common.REPO / "experiments/memory_contribution/run_counterfactual.py",
    common.REPO / "experiments/memory_contribution/build_manifest.py",
    common.REPO / "tests/test_memory_contribution.py",
]
ARTIFACTS = [
    "fixed_pool_comparison.json",
    "per_task_results.jsonl",
    "arm_results.jsonl",
    "0051_diagnostic.json",
    "counterfactual_cases.jsonl",
    "counterfactual_summary.json",
]


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=common.REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - manifest must still be writable
        return None


def main() -> int:
    comparison = common.read_json(OUT / "fixed_pool_comparison.json")
    counterfactual = common.read_json(OUT / "counterfactual_summary.json")
    artifact_hashes = {}
    for name in ARTIFACTS:
        path = OUT / name
        if path.is_file():
            artifact_hashes[name] = {
                "bytes": path.stat().st_size,
                "sha256": common.sha256_file(path),
            }
    manifest: dict[str, Any] = {
        "schema": common.SCHEMA_MANIFEST,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "phase": "memory_contribution_phase1",
        "scope": "local_cpu_benchmark_only",
        "git_head_at_run": _git_head(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cwd": os.getcwd(),
            "gpu_used": False,
            "training_used": False,
            "model_called": False,
            "retrieval_rebuilt": False,
        },
        "inputs": {
            "ranking_dataset": {
                "path": "runs/preference_ranking_v1/ranking_dataset.jsonl",
                "sha256": common.sha256_file(common.RANKING_DATASET),
                "tasks": 30,
                "recommend_tasks": 19,
            },
            "ranking_manifest": {
                "path": "runs/preference_ranking_v1/manifest.json",
                "sha256": common.sha256_file(common.RANKING_MANIFEST),
            },
            "p20_suite": {
                "path": "runs/p20_agent_eval/task_suite.json",
                "sha256": common.sha256_file(common.P20_SUITE),
                "tasks": 30,
            },
            "original_bge_trace": {
                "path": "runs/advanced_retrieval_v2/task_arm_original_bge.json",
                "sha256": common.sha256_file(common.RETRIEVAL_TRACE),
                "role": "provenance only; the fixed pools are read from the Ranking v1 dataset",
            },
        },
        "configuration": {
            "arms": list(common.ARMS),
            "candidate_protocol": "fixed Original-BGE Top-20 from the stored Ranking v1 dataset; no new retrieval",
            "eligibility": "hard_tier == 0 and role_status in {MATCH, NOT_REQUIRED}; shared by A/B/C",
            "scorer": comparison["scorer_version"],
            "scorer_config": comparison["scorer_config"],
            "gate": "recommendation.preference.decision_gate.DecisionGate v0.1 with no remaining retrieval/evidence budget",
            "label_status": common.LABEL_STATUS,
            "unjudged_policy": "candidates not in the provisional acceptable set are unjudged, not negatives",
            "memory_off_semantics": "historical preference channel disabled; current-query hard requirements and any explicit current-query soft preference retained",
            "memory_on_semantics": "legal M9 HistoryStateReader active preferences added",
            "counterfactual_cases": counterfactual["cases"],
            "counterfactual_categories": sorted(counterfactual["by_category"]),
        },
        "source_files": {
            str(path.relative_to(common.REPO)).replace("\\", "/"): common.sha256_file(path)
            for path in SOURCE_FILES
            if path.is_file()
        },
        "artifacts": artifact_hashes,
        "results": {
            "A_hit1": comparison["arm_metrics"]["A_relevance_only"]["hit@1"],
            "B_hit1": comparison["arm_metrics"]["B_scorer_memory_off"]["hit@1"],
            "C_hit1": comparison["arm_metrics"]["C_scorer_memory_on"]["hit@1"],
            "C_minus_B_hit1": comparison["headline"]["Q2_C_minus_B_hit1"],
            "B_vs_C_full_ranking_changed_tasks": comparison["b_vs_c"]["full_ranking_changed_tasks"],
            "B_vs_C_top1_changed_tasks": comparison["b_vs_c"]["top1_changed_tasks"],
            "active_memory_tasks": comparison["b_vs_c"]["active_memory_tasks"],
            "hit1_benefit_tasks": comparison["b_vs_c"]["hit1_benefit_tasks"],
            "hit1_regression_tasks": comparison["b_vs_c"]["hit1_regression_tasks"],
            "counterfactual_pass_rate": counterfactual["memory_consistency_pass_rate"],
            "counterfactual_checks_passed": counterfactual["checks_passed"],
            "counterfactual_checks_total": counterfactual["checks_total"],
            "case_0051_rank_B": comparison["case_0051"]["candidates"]["B08D32JM5F"]["rank_B"],
            "case_0051_rank_C": comparison["case_0051"]["candidates"]["B08D32JM5F"]["rank_C"],
        },
        "protection": {
            "p18_frozen_manifest_modified": False,
            "p18_reserved_used": False,
            "planner_modified": False,
            "retrieval_modified": False,
            "decision_gate_modified": False,
            "scorer_modified": False,
            "production_ranking_default_changed": False,
            "git_push": False,
        },
        "claims": {
            "memory_correctly_changes_ranking": "established on the constructed fixed pool and memory-state tests",
            "memory_improves_observed_acceptable_metrics": (
                "established on the provisional observed-acceptable labels for this 19-task suite"
            ),
            "memory_improves_real_user_satisfaction": "not established",
        },
        "notes": [
            "A and B rank identically because the production Scorer has no query-soft-preference channel; current query conditions are hard requirements retained in every arm.",
            "The labels are provisional automated/verified labels, not full human gold; unlabelled candidates are unjudged.",
            "No model was trained, no GPU was used, and no retrieval pool was rebuilt.",
        ],
    }
    common.write_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
