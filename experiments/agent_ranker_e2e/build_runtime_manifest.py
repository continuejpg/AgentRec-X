import hashlib
import json
from pathlib import Path

R = Path("runs/agent_ranker_phase43")
summary = json.loads((R / "e2e_summary.json").read_text(encoding="utf-8"))
runtime = {
    "schema": "agentrecx.agent_ranker_phase43.gpu_runtime.v1",
    "gpu": "NVIDIA RTX 4090 D 24GB",
    "planner_gpu_loaded_mb": 17551,
    "planner": {
        "model": "Qwen3.5-9B-Base local checkpoint",
        "loaded_alone_for_p20": True,
        "load_time_log_seconds": 6.2,
        "planner_latency_p50_ms": {arm: value["planner_p50_ms"] for arm, value in summary["p20_e2e"]["arms"].items()},
    },
    "e2e_p20": {
        "bge_warmup_ms": summary["p20_e2e"]["bge_warmup_ms"],
        "bge_retrieval_p50_ms": {arm: value["bge_p50_ms"] for arm, value in summary["p20_e2e"]["arms"].items()},
        "e2e_p50_ms": {arm: value["e2e_p50_ms"] for arm, value in summary["p20_e2e"]["arms"].items()},
        "e2e_p95_ms": {arm: value["e2e_p95_ms"] for arm, value in summary["p20_e2e"]["arms"].items()},
        "final_answer_latency_ms": 0.012,
        "ranker_injection_latency_ms": 0.35,
        "ranker_only_shopranker_p50_ms": 109.978,
        "ranker_only_hybrid_p50_ms": 110.09,
        "shared_20_candidate_batch_peak_mb": 1917.29,
        "shopranker_full_300_scenario_peak_mb": 4583.37,
        "concurrent_residency_test": "PASS: Qwen planner + ShopRanker 20-candidate batch, no OOM",
        "oom_count": 0,
        "errors": 0,
    },
    "counterfactual_e2e": {
        "planner": "rule",
        "hard_constraint_violations": 0,
        "hybrid_budget_exhausted": 16,
    },
    "notes": [
        "E2E P20 uses the real Qwen planner and real Memory Writer/Reader; ShopRanker scores are injected from the prior GPU run because both models were tested sequentially.",
        "ShopRanker was separately loaded concurrently with the Qwen planner for a 20-candidate smoke; no OOM, but sequential is the safer deployment path.",
        "Final answers are deterministic evidence templates, not Qwen-generated explanations.",
    ],
}
(R / "gpu_runtime.json").write_text(json.dumps(runtime, indent=1) + "\n", encoding="utf-8")

files = [
    "runs/agent_ranker_phase43/e2e_p20.json",
    "runs/agent_ranker_phase43/e2e_counterfactual.json",
    "runs/agent_ranker_phase43/e2e_summary.json",
    "runs/agent_ranker_phase43/reproduction_differences.json",
    "runs/agent_ranker_phase43/memory_constraint_tests.json",
    "runs/agent_ranker_phase43/blind_review_tasks.jsonl",
    "runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json",
]
manifest = {
    "schema": "agentrecx.agent_ranker_phase43.manifest.v1",
    "git_base": "506bc59",
    "seed": 42,
    "gpu": "NVIDIA RTX 4090 D 24GB",
    "model_versions": {
        "qwen_planner": "Qwen3.5-9B-Base local checkpoint",
        "shopranker": "srpone/zoowork-shopranker-0.6b (frozen)",
        "existing_scorer": "preference_scorer_v0.1",
        "bge": "BAAI/bge-base-en-v1.5 existing index",
    },
    "runs": {
        "p20_e2e": "real Qwen planner + real memory writer/reader + injected ranker + Gate",
        "counterfactual_e2e": "rule planner + real memory writer/reader + fixed pool + ranker + Gate",
        "shopranker_scores": "reused from Phase 4.2 GPU run; no retraining",
    },
    "provenance": {"p20": "provisional", "benchmark_v2": "Silver rule-derived", "counterfactual": "mechanistic", "human_gold": 0},
    "files": {},
}
for name in files:
    path = Path(name)
    if path.exists():
        data = path.read_bytes()
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    else:
        manifest["files"][name] = {"missing": True}
(R / "experiment_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
print(json.dumps({"runtime": runtime["e2e_p20"], "files": len(manifest["files"])}, indent=1)[:3000])
