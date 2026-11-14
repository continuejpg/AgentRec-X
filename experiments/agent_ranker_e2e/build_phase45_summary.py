import hashlib
import json
from pathlib import Path

R = Path("runs/agent_ranker_phase45")
P43 = Path("runs/agent_ranker_phase43")
topk = json.loads((R / "topk_metrics.json").read_text(encoding="utf-8"))
expl = json.loads((R / "explanation_metrics.json").read_text(encoding="utf-8"))
ask = json.loads((R / "ask_quality_diagnostics.json").read_text(encoding="utf-8"))
cf010 = json.loads((R / "cf010_parse_diagnosis.json").read_text(encoding="utf-8"))
human = json.loads((R / "human_preference_eval.json").read_text(encoding="utf-8"))
p20 = json.loads((P43 / "e2e_summary.json").read_text(encoding="utf-8"))
summary = {
    "schema": "agentrecx.agent_ranker_phase45.summary.v1",
    "topk": topk,
    "explanations": {key: expl[key] for key in ("candidate_id_consistency", "ranking_order_consistency", "attribute_claims", "unsupported_attributes", "missing_products", "mean_latency_ms", "max_latency_ms", "mismatches")},
    "ask_quality": ask["summary"],
    "cf010_parse": {"trace_records": cf010["trace_records"], "invalid_payloads": cf010["invalid_payloads"], "attempts_per_invalid": cf010["attempts_per_invalid"], "raw_lengths": cf010["raw_lengths"], "cause": cf010["cause"]},
    "human_review": {"tasks": human["tasks_total"], "human_review_count": human["human_review_count"], "provenance": human["review_provenance"], "agreement": human["model_agreement"]},
    "p20_e2e": {arm: {"success": values["success"], "finalized": values["finalized"], "hard_constraints": values["hard_constraint_violations"], "answer_evidence": values["final_answer_evidence_supported"], "p50_ms": values["e2e_p50_ms"]} for arm, values in p20["p20_e2e"]["arms"].items()},
    "provenance": {"human_gold": 0, "human_single_review": 0, "p20": "provisional", "benchmark_v2": "Silver", "counterfactual": "mechanistic"},
}
(R / "phase45_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
runtime = {
    "schema": "agentrecx.agent_ranker_phase45.gpu_runtime.v1",
    "gpu": "NVIDIA RTX 4090 D 24GB",
    "qwen_planner_loaded_mb": 17551,
    "topk_e2e": {
        "records": 57,
        "e2e_p50_ms": {arm: values["e2e_p50_ms"] for arm, values in topk["arms"].items()},
        "e2e_p95_ms": {arm: values["e2e_p95_ms"] for arm, values in topk["arms"].items()},
        "topk_presenter_p50_ms": {arm: values["topk_p50_ms"] for arm, values in topk["arms"].items()},
        "oom": 0,
    },
    "qwen_explanations": {"tasks": expl["tasks"], "mean_latency_ms": expl["mean_latency_ms"], "max_latency_ms": expl["max_latency_ms"], "errors": 0},
    "counterfactual_costs_from_phase44": {"qwen_matrix_wall_seconds": 1012.0, "shop_cf_scoring_wall_seconds": 7.763, "shop_cf_peak_mb": 3572.42},
    "concurrent": {"qwen_planner_mb": 17551, "shop_20_candidate_extra_peak_mb": 1917.29, "nvidia_total_used_mb": 19067, "oom": 0},
    "cpu_memory_adapter_20_candidates_p50_ms": 0.0704,
    "ranker_only_shop_p50_ms": 109.978,
    "ranker_only_hybrid_p50_ms": 110.09,
    "notes": ["No new model training", "TopK uses cached arm scores but actual Qwen planner + actual TopK presentation/validation", "Qwen explanation is shadow-only and does not choose products"],
}
(R / "gpu_runtime.json").write_text(json.dumps(runtime, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
files = [
    "runs/agent_ranker_phase45/topk_e2e.json",
    "runs/agent_ranker_phase45/topk_metrics.json",
    "runs/agent_ranker_phase45/topk_explanations.json",
    "runs/agent_ranker_phase45/explanation_metrics.json",
    "runs/agent_ranker_phase45/ask_quality_diagnostics.json",
    "runs/agent_ranker_phase45/cf010_parse_diagnosis.json",
    "runs/agent_ranker_phase45/cf010_trace.jsonl",
    "runs/agent_ranker_phase45/human_preference_eval.json",
    "runs/agent_ranker_phase45/blind_review_results.csv",
]
manifest = {
    "schema": "agentrecx.agent_ranker_phase45.manifest.v1",
    "base_commit": "9461245",
    "seed": 42,
    "models": {"qwen": "Qwen3.5-9B-Base", "shopranker": "srpone/zoowork-shopranker-0.6b frozen", "scorer": "preference_scorer_v0.1"},
    "configs": {"hybrid_formula": "unchanged", "topk": "1/3/5 per-candidate validation", "qwen_explanation": "shadow only; no selection/reorder", "human_gold": 0},
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
print(json.dumps({"topk": summary["topk"]["arms"], "explanation": summary["explanations"], "ask": summary["ask_quality"], "human": summary["human_review"]}, indent=1)[:5000])
