import hashlib
import json
from pathlib import Path

ROOT = Path(".")
files = [
    "data/memory_ranking_benchmark_v2/scenarios.jsonl",
    "data/memory_ranking_benchmark_v2/pair_tasks.jsonl",
    "data/memory_ranking_benchmark_v2/candidate_metadata.jsonl",
    "runs/agent_native_ranking_phase42/memory_adapter_features.jsonl",
    "runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json",
    "runs/agent_native_ranking_phase42/dcn_alignment_diagnostics.json",
    "runs/agent_native_ranking_phase42/shadow_comparison.json",
]
manifest = {
    "schema": "agentrecx.agent_native_ranking_phase42.manifest.v1",
    "seed": 42,
    "gpu": "NVIDIA RTX 4090 D 24GB",
    "model_versions": {
        "shopranker": "srpone/zoowork-shopranker-0.6b (frozen, Apache-2.0)",
        "existing_scorer": "preference_scorer_v0.1",
        "dcn_A": "A_base_query_only.pt",
        "dcn_B": "B_preference_on.pt",
        "dcn_C": "C_full_query.pt",
    },
    "configs": {
        "memory_adapter": "experiments/agent_native_ranking/memory_adapter.py; active-only, multi-preference, UNKNOWN!=absence",
        "hybrid_formula": "pool_z(shop_query_only) + 0.5*positive_match_rate - 0.75*negative_conflict_rate",
        "hybrid_tuning": "none; weights fixed before result inspection",
        "dcn_inference": "reuse ESCI checkpoints; 64 hidden, 3 cross layers; no retraining",
        "shopranker_inference": "official prompt, yes/no logit; no fine-tuning; max_length=1536",
    },
    "provenance": {
        "p20": "provisional observed-acceptable; 19 recommend tasks",
        "benchmark_v2": "300 synthetic-memory scenarios; 145 Silver pairs; 16 ASK",
        "counterfactual": "16 mechanistic cases; no independent direction Gold",
        "human_gold": 0,
    },
    "files": {},
}
for name in files:
    path = ROOT / name
    if path.exists():
        data = path.read_bytes()
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    else:
        manifest["files"][name] = {"missing": True}
Path("runs/agent_native_ranking_phase42/experiment_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=1, ensure_ascii=False)[:3000])
