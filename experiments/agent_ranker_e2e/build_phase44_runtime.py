import hashlib
import json
from pathlib import Path

R = Path("runs/agent_ranker_phase44")
summary = json.loads((R / "phase44_summary.json").read_text(encoding="utf-8"))
runtime = {
    "schema": "agentrecx.agent_ranker_phase44.gpu_runtime.v1",
    "gpu": "NVIDIA RTX 4090 D 24GB",
    "qwen_planner_used_mb": 17551,
    "shopranker_cf_scoring": {
        "wall_seconds": 7.763,
        "peak_allocated_mb": 3572.42,
        "scope": "16 cases x 2 states x 18-20 fixed candidates; Query + Natural scoring",
    },
    "rule_planner_cf_matrix": {"trajectories": 96, "note": "CPU Rule Planner; no GPU model generation"},
    "qwen_planner_cf_matrix": {
        "trajectories": 96,
        "wall_seconds": 1012.0,
        "note": "wall time includes 3-rankers x 2-states x 16 cases, Qwen /plan generation and CPU executor; one planner parse failure on cf010 (3 records)",
    },
    "concurrent_residency": {
        "qwen_planner_mb": 17551,
        "shopranker_20_candidate_peak_allocated_mb": 1917.29,
        "shopranker_300_scenario_peak_allocated_mb": 4583.37,
        "no_oom": True,
        "observed_nvidia_used_mb": 19067,
        "scope": "Qwen planner resident; ShopRanker 20-candidate batch concurrently loaded/scored; full 300-scenario Shop run separately",
    },
    "p20_e2e_from_phase43": {
        arm: {"p50_ms": values["e2e_p50_ms"], "p95_ms": values["e2e_p95_ms"]}
        for arm, values in summary["p20_e2e"].items()
    },
    "ranker_only": {"shopranker_p50_ms": 109.978, "hybrid_p50_ms": 110.09},
    "cpu_memory_adapter": {"20_candidates_p50_ms": 0.0704, "20_candidates_mean_ms": 0.0709},
    "gate_latency": "included in E2E; not separately instrumented",
    "oom_count": 0,
    "execution_errors": {"p20_e2e": 0, "qwen_cf_planner_parse_errors": 3, "rule_cf_errors": 0},
}
(R / "gpu_runtime.json").write_text(json.dumps(runtime, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

files = [
    "runs/agent_ranker_phase44/hybrid_budget_fix.json",
    "runs/agent_ranker_phase44/cf_matrix_rule.json",
    "runs/agent_ranker_phase44/cf_matrix_qwen.json",
    "runs/agent_ranker_phase44/cf_ranking_metrics.json",
    "runs/agent_ranker_phase44/cf_shop_scores.json",
    "runs/agent_ranker_phase44/phase44_summary.json",
    "runs/agent_ranker_phase44/blind_review_tool.html",
]
manifest = {
    "schema": "agentrecx.agent_ranker_phase44.manifest.v1",
    "base_commit": "b8fef57",
    "seed": 42,
    "gpu": "NVIDIA RTX 4090 D 24GB",
    "model_versions": {
        "qwen_planner": "Qwen3.5-9B-Base local checkpoint",
        "shopranker": "srpone/zoowork-shopranker-0.6b frozen",
        "existing_scorer": "preference_scorer_v0.1",
    },
    "configs": {
        "hybrid_formula": "pool_z(shopranker_query_only) + 0.5*positive_match_rate - 0.75*negative_conflict_rate",
        "hybrid_weight_change": "none",
        "cf_candidate_pool": "fixed BGE counterfactual pool",
        "planner_modes": ["rule", "qwen"],
        "action_budget": "unchanged MAX_STEPS=8; no budget increase",
    },
    "provenance": {"human_gold": 0, "human_single_review": 0, "benchmark_v2": "Silver rule-derived", "p20": "provisional", "counterfactual": "mechanistic"},
    "files": {},
}
for name in files:
    path = Path(name)
    if path.exists():
        data = path.read_bytes()
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    else:
        manifest["files"][name] = {"missing": True}
(R / "experiment_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"runtime": runtime["qwen_planner_cf_matrix"], "files": len(manifest["files"])}, indent=1))
