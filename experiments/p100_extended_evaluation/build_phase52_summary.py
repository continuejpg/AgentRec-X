import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/agentrec_x_phase52"
P100 = REPO / "runs/p100_extended_evaluation"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    waterfall = load(OUT / "p100_retrieval_judgment_waterfall.json")
    evidence = [json.loads(l) for l in (OUT / "p100_evidence_unsupported_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    memory = load(OUT / "p100_memory_writer_anomalies.json")
    ask = load(OUT / "p100_ask_analysis.json")
    bge = load(OUT / "p100_judged_bge_metrics.json")
    vs = load(OUT / "p100_judged_bge_vs_shopranker.json")
    bge_runtime = load(OUT / "p100_judged_bge_runtime.json")
    e2e = load(P100 / "p100_agent_e2e_metrics.json")
    summary = {
        "schema": "agentrecx.p52.summary.v1",
        "retrieval_waterfall": waterfall["totals"],
        "conditional_recall_at20_indexed_positive": waterfall["totals"].get("conditional_recall_at20_indexed_positive"),
        "judged_queries_with_catalog_overlap": 49,
        "judged_queries_with_catalog_positive_overlap": 44,
        "agent_unjudged_fraction": 0.993,
        "evidence_unsupported": len(evidence),
        "evidence_by_arm": {arm: sum(1 for row in evidence if row["arm"] == arm) for arm in ("existing", "natural", "hybrid")},
        "memory_anomalies": sum(1 for row in memory["cases"] if row.get("anomaly")),
        "memory_anomaly_classification": {},
        "ask": ask["summary"],
        "bge_judged": {key: bge[key] for key in ("ndcg@3", "ndcg@5", "ndcg@10", "mrr_positive_threshold_E_or_S", "pairwise_accuracy", "pairwise_pairs", "valid_queries")},
        "shopranker_judged": load(P100 / "p100_judged_metrics.json")["arms"]["shopranker_query_only"],
        "bge_vs_shopranker": vs,
        "gpu": {"bge_judged_wall_seconds": bge_runtime["wall_seconds"], "bge_peak_mb": bge_runtime["gpu_peak_mb"],
                "shopranker_gpu_scoring_wall_seconds": load(P100 / "p100_ranker_gpu_runtime.json")["wall_seconds"],
                "shopranker_peak_mb": load(P100 / "p100_ranker_gpu_runtime.json")["peak_memory_mb"],
                "e2e_ranker_injection_p50_ms": {"natural": e2e["arms"]["natural"]["ranker_p50_ms"], "hybrid": e2e["arms"]["hybrid"]["ranker_p50_ms"]},
                "qwen_planner_p50_ms": {arm: e2e["arms"][arm]["planner_p50_ms"] for arm in e2e["arms"]},
                "full_e2e_p50_ms": {arm: e2e["arms"][arm]["e2e_p50_ms"] for arm in e2e["arms"]},
                "oom": 0},
        "human": {"human_gold": 0, "human_single_review": 0},
    }
    for row in memory["cases"]:
        if not row.get("anomaly"):
            continue
        summary["memory_anomaly_classification"][row["classification"]] = summary["memory_anomaly_classification"].get(row["classification"], 0) + 1
    (OUT / "phase52_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    files = [OUT / name for name in ("p100_retrieval_judgment_waterfall.json", "p100_retrieval_coverage_per_query.csv",
                                      "p100_evidence_unsupported_cases.jsonl", "p100_memory_writer_anomalies.json",
                                      "p100_ask_analysis.json", "p100_judged_bge_predictions.jsonl",
                                      "p100_judged_bge_metrics.json", "p100_judged_bge_vs_shopranker.json",
                                      "p100_judged_bge_runtime.json", "phase52_summary.json")]
    manifest = {
        "schema": "agentrecx.p52.manifest.v1", "base_commit": "a79494c", "seed": 20261114,
        "model": {"bge": "BAAI/bge-base-en-v1.5 frozen", "shopranker": "srpone/zoowork-shopranker-0.6b frozen",
                   "qwen": "Qwen3.5-9B-Base", "existing": "preference_scorer_v0.1"},
        "configs": {"p100_queries": 100, "agent_candidate_rows": 2000, "judged_products": 1971,
                    "no_retraining": True, "no_p100_expansion": True, "no_production_gate_change": True,
                    "no_human_gold_fabricated": True},
        "files": {},
    }
    for path in files:
        if path.exists():
            manifest["files"][str(path.relative_to(REPO))] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "phase52_experiment_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1)[:7000])


if __name__ == "__main__":
    main()
