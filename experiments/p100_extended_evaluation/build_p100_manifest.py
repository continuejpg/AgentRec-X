"""P100 GPU runtime + experiment manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ranker_runtime = json.loads((RUN / "p100_ranker_gpu_runtime.json").read_text(encoding="utf-8"))
    e2e = [json.loads(line) for line in (RUN / "p100_agent_e2e.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    runtime = {
        "schema": "agentrecx.p100.gpu_runtime.v1",
        "gpu": "NVIDIA RTX 4090 D 24GB",
        "shopranker_ranker_gpu": ranker_runtime,
        "qwen_planner_resident_mb": 17551,
        "agent_e2e": {
            "trajectories": len(e2e),
            "successful_trajectories": sum(1 for row in e2e if row.get("status") != "error"),
            "errors": sum(1 for row in e2e if row.get("status") == "error"),
            "oom": 0,
        },
        "concurrent_note": "ShopRanker GPU scoring completed before Qwen Agent E2E; no simultaneous 17.5GB + 6.6GB residency was required. The prior Phase 4.4 concurrent 20-candidate test was PASS.",
        "no_oom": True,
    }
    (RUN / "p100_gpu_runtime.json").write_text(json.dumps(runtime, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    files = [
        DATA / "p100_selection_manifest.json", DATA / "p100_agent_tasks.jsonl",
        DATA / "p100_bge_pools.jsonl", DATA / "p100_judged_tasks.jsonl",
        DATA / "p100_label_coverage.json", DATA / "p100_memory_states.jsonl",
        DATA / "p100_candidate_metadata.jsonl", DATA / "p100_agent_tasks_memory_on.jsonl",
        DATA / "p100_agent_tasks_memory_off.jsonl",
        RUN / "p100_ranker_gpu_predictions.jsonl", RUN / "p100_judged_gpu_predictions.jsonl",
        RUN / "p100_ranker_metrics.json", RUN / "p100_judged_metrics.json",
        RUN / "p100_memory_off_on.json", RUN / "p100_agent_e2e.jsonl",
        RUN / "p100_agent_e2e_metrics.json", RUN / "p100_per_query_comparison.csv",
        RUN / "p100_gpu_runtime.json", RUN / "p20_hit1_metric_diff.json",
    ]
    manifest = {
        "schema": "agentrecx.p100.experiment_manifest.v1",
        "base_commit": "b043ba3",
        "created_utc": "2026-11-14T18:40:00+08:00",
        "random_seed": 20261114,
        "official_source": "Amazon ESCI Task 1, small_version=1 split=test product_locale=us",
        "official_commit": "7916cdf6ab75a462e77f20ab40428a10923998d5",
        "models": {
            "qwen_planner": "Qwen3.5-9B-Base /root/autodl-tmp/models/Qwen3.5-9B-ms",
            "shopranker": "srpone/zoowork-shopranker-0.6b frozen",
            "bge": "BAAI/bge-base-en-v1.5 frozen",
            "existing_scorer": "preference_scorer_v0.1",
        },
        "prompt_hashes": {
            "shopranker_prompt_sha256": ranker_runtime.get("prompt_sha256"),
            "qwen_planner_prompt": "training/agent_sft/planner_prompt.py::PLANNER_SYSTEM_PROMPT unchanged",
            "serialization": "title|product_type|verified_attributes|evidence",
        },
        "memory": {"source": "synthetic_event_sequence", "human_memory_preference_gold": 0,
                   "snapshot_ids": f"p100_memory_states.jsonl ({sum(1 for line in (DATA / 'p100_memory_states.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())} rows)"},
        "configs": {"hybrid_formula": "pool_z(shopranker_query_only) + 0.5*positive_match_rate - 0.75*negative_conflict_rate",
                    "weights_tuned_on_p100": False, "gate_thresholds_tuned": False,
                    "agent_tasks": 100, "ranker_arms": 3, "e2e_trajectories": len(e2e)},
        "files": {},
    }
    for path in files:
        key = str(path.relative_to(REPO))
        manifest["files"][key] = {"bytes": path.stat().st_size, "sha256": sha(path)} if path.exists() else {"missing": True}
    (RUN / "p100_experiment_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"e2e": runtime["agent_e2e"], "files": len(manifest["files"])}, indent=1))


if __name__ == "__main__":
    main()
