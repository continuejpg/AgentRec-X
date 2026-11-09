"""P14-D2 evaluation: dense retrieval repair and fixed-pool history ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.p14d1r.evaluate import evaluate, rank_scores
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
RUNS = REPO / "runs/p14d2"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(rows: list[dict[str, Any]], method: str, conditional: bool = False) -> dict[str, Any]:
    values = [row["methods"][method] for row in rows if not conditional or row["methods"][method]["target_in_pool"]]
    if not values:
        return {"n": 0}
    return {"n": len(values), "hit_at_1": sum(1 for value in values if value["hit_at_1"]),
            "hit_at_5": sum(1 for value in values if value["hit_at_5"]),
            "hit_at_10": sum(1 for value in values if value["hit_at_10"]),
            "mrr_at_10": round(sum(value["mrr_at_10"] for value in values) / len(values), 6)}


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    retrieval = json.loads((RUNS / "natural_retrieval.json").read_text(encoding="utf-8"))
    tasks = {row["task_id"]: row for row in load_jsonl(DATA / "dev_tasks.jsonl")}
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    pools = {row["task_id"]: row for row in load_jsonl(RUNS / "pools/improved_natural.jsonl")}
    scores = json.loads((RUNS / "scores_history_ablation.json").read_text(encoding="utf-8"))
    rows = []
    for tid, task in tasks.items():
        target_id = targets[tid]["target_parent_asin"]
        pool = pools[tid]
        bge_rank = [candidate["item_id"] for candidate in sorted(pool["candidates"],
                                                                 key=lambda candidate: (candidate["rank"], candidate["item_id"]))]
        row = {"task_id": tid, "bge_target_rank": (bge_rank.index(target_id) + 1) if target_id in bge_rank else None,
               "target_in_pool": target_id in bge_rank, "methods": {}}
        row["methods"]["bge_top20"] = evaluate(bge_rank, target_id)
        row["methods"]["qwen3_query_only"] = evaluate(rank_scores(pool, scores["scores"]["query_only"][tid]), target_id)
        row["methods"]["qwen3_history_recovered"] = evaluate(rank_scores(pool, scores["scores"]["history_real"][tid]), target_id)
        row["methods"]["qwen3_history_shuffled"] = evaluate(rank_scores(pool, scores["scores"]["history_shuffled"][tid]), target_id)
        rows.append(row)
    methods = ("bge_top20", "qwen3_query_only", "qwen3_history_recovered", "qwen3_history_shuffled")
    summary = {method: {"all": aggregate(rows, method), "conditional_target_in_pool": aggregate(rows, method, True)}
               for method in methods}
    effects = {}
    for method in ("qwen3_history_recovered", "qwen3_history_shuffled"):
        base = {row["task_id"]: row["methods"]["qwen3_query_only"]["target_rank"] for row in rows}
        other = {row["task_id"]: row["methods"][method]["target_rank"] for row in rows}
        effects[f"query_only->{method}"] = {
            "n": len(rows),
            "hit1_query_only": sum(1 for row in rows if row["methods"]["qwen3_query_only"]["hit_at_1"]),
            "hit1_variant": sum(1 for row in rows if row["methods"][method]["hit_at_1"]),
            "improved": [tid for tid in base if other[tid] is not None and (base[tid] is None or other[tid] < base[tid])],
            "worsened": [tid for tid in base if base[tid] is not None and (other[tid] is None or other[tid] > base[tid])],
        }
    index_manifest = {
        "schema": "agentrecx.p14d2.dense_index_manifest.v1",
        "model": retrieval["bge_index_manifest"]["model"],
        "catalogue": retrieval["bge_index_manifest"]["catalogue"],
        "resources": retrieval["bge_index_manifest"]["resources"],
        "files": {
            "index_npz": "runs/p14d2/bge_index.npz",
            "query_vectors_npz": "runs/p14d2/query_vectors.npz",
            "retrieval_json": "runs/p14d2/bge_retrieval.json",
            "catalogue_docs": "runs/p14d2/catalogue_docs.jsonl",
        },
        "identity_check": "identities exactly equal to ElectronicsCatalogue accepted item_ids; 76,263 vectors; no duplicates",
        "old_p8_index": retrieval["old_p8_bge_index"],
    }
    comparison = {
        "schema": "agentrecx.p14d2.comparison.v1",
        "dense_retrieval": retrieval["variants"],
        "selected_fixed_pool": retrieval["selected_fixed_pool"],
        "target_filter_loss": retrieval["target_filter_loss"],
        "history_ablation_on_bge_pool": {"tasks": len(rows), "methods": summary, "effects": effects},
        "history_pool_target_reach": sum(1 for row in rows if row["target_in_pool"]),
        "cost": {
            "bge": retrieval["bge_index_manifest"]["resources"],
            "qwen_history_ablation": scores["resources"],
        },
        "model_manifest": index_manifest,
        "per_task": rows,
        "notes": ["BGE index rebuilt against the exact P13-R ElectronicsCatalogue identities; old P8 index not reused",
                  "BM25/BGE/Hybrid original and rewritten queries are separate; no mixing",
                  "fixed-pool history ablation uses BGE original-query Top-20, target not injected",
                  "Qwen history prompt and model are unchanged from P14-D1-R"],
    }
    (RUNS / "dense_index_manifest.json").write_text(json.dumps(index_manifest, indent=1, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    (RUNS / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    with (RUNS / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (RUNS / "model_manifest.json").write_text(json.dumps({
        "schema": "agentrecx.p14d2.model_manifest.v1",
        "dense": retrieval["bge_index_manifest"],
        "qwen3_history_ablation": {"model": scores["model"], "resources": scores["resources"]},
        "old_p8_bge": retrieval["old_p8_bge_index"],
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dense": retrieval["variants"], "selected": retrieval["selected_fixed_pool"],
                      "history": summary, "effects": effects, "target_reach": comparison["history_pool_target_reach"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
