"""P14-D2 natural retrieval comparison: BM25, aligned BGE, fixed RRF hybrid.

No target injection. Evaluator target IDs are used only after candidate ranking.
Selection of a fixed improved pool follows a rule fixed before looking at history results:
use the method with the highest Recall@20 (ties: BGE then Hybrid), if it beats BM25.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.product_role import assess_candidate_role, infer_product_target

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
RUNS = REPO / "runs/p14d2"
RRF_K = 60


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rrf_rank(lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in lists:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item_id: (-scores[item_id], item_id))


def recall_at(ranked_by_task: dict[str, list[str]], targets: dict[str, str], k: int) -> int:
    return sum(1 for task_id, ranked in ranked_by_task.items()
               if targets[task_id] in ranked[:k])


def target_in_catalogue(target_id: str, catalogue: ElectronicsCatalogue) -> bool:
    return catalogue.get(target_id) is not None


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    tasks = {row["task_id"]: row for row in load_jsonl(DATA / "dev_tasks.jsonl")}
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    bge = json.loads((RUNS / "bge_retrieval.json").read_text(encoding="utf-8"))
    bm25 = {"original": {}, "rewritten": {}}
    for tid, task in tasks.items():
        for variant, query in (("original", task["query_original"]), ("rewritten", task["query_rewritten"])):
            result = catalogue.search_source.search(query, limit=100)
            bm25[variant][tid] = [hit.parent_asin.split(":", 1)[-1] for hit in result.hits]
    variants = ("original", "rewritten")
    rankings: dict[str, dict[str, list[str]]] = {}
    for variant in variants:
        bge_rank = {tid: [row["item_id"] for row in bge["top100"][variant][tid]] for tid in tasks}
        hybrid_rank = {tid: rrf_rank([bm25[variant][tid], bge_rank[tid]]) for tid in tasks}
        rankings[variant] = {"bm25": bm25[variant], "bge": bge_rank, "hybrid_rrf": hybrid_rank}
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for method in ("bm25", "bge", "hybrid_rrf"):
            ranks = rankings[variant][method]
            summary[variant][method] = {
                "recall_at20": recall_at(ranks, {tid: targets[tid]["target_parent_asin"] for tid in tasks}, 20),
                "recall_at100": recall_at(ranks, {tid: targets[tid]["target_parent_asin"] for tid in tasks}, 100),
                "tasks": len(tasks),
            }

    # Target not in catalogue and optional Product Type/Role filter loss (query type inferred from original query).
    target_filter = {"target_not_in_catalogue": 0, "target_would_be_filtered_by_role": 0,
                     "target_role_match_or_not_required": 0, "per_task": []}
    for tid, task in tasks.items():
        target_id = targets[tid]["target_parent_asin"]
        product = catalogue.get(target_id)
        if product is None:
            target_filter["target_not_in_catalogue"] += 1
            target_filter["per_task"].append({"task_id": tid, "reason": "not_in_catalogue"})
            continue
        target = infer_product_target(task["query_original"])
        assessment = assess_candidate_role(product_type=product.product_type, title=product.title, target=target)
        allowed = assessment.status.value in {"MATCH", "NOT_REQUIRED"}
        if allowed:
            target_filter["target_role_match_or_not_required"] += 1
        else:
            target_filter["target_would_be_filtered_by_role"] += 1
        target_filter["per_task"].append({"task_id": tid, "status": assessment.status.value,
                                          "product_type": product.product_type,
                                          "target_product_type": target.product_type})

    # Pre-registered pool choice rule: highest recall@20, tie BGE then Hybrid; only if > BM25.
    candidates = []
    for method in ("bge", "hybrid_rrf"):
        for variant in variants:
            score = summary[variant][method]["recall_at20"]
            baseline = summary[variant]["bm25"]["recall_at20"]
            if score > baseline:
                candidates.append((score, variant, method))
    candidates.sort(key=lambda item: (-item[0], 0 if item[2] == "bge" else 1))
    selected = None
    if candidates:
        _score, variant, method = candidates[0]
        selected = {"method": method, "query_variant": variant,
                    "recall_at20": _score,
                    "rule": "highest Recall@20 among BGE/Hybrid that beats BM25; tie prefers BGE"}
    result = {
        "schema": "agentrecx.p14d2.natural_retrieval.v1",
        "variants": {variant: summary[variant] for variant in variants},
        "selected_fixed_pool": selected,
        "target_filter_loss": target_filter,
        "bge_index_manifest": {key: bge[key] for key in ("model", "catalogue", "resources")},
        "old_p8_bge_index": {
            "path": "runs/p8_bge/bge_base_dense.npz", "item_count": 156746,
            "intersection_with_p13r_item_ids": 0,
            "reason": "wrong catalogue identity; not reused",
        },
        "notes": ["evaluator target IDs are used only for scoring after candidate ranking",
                  "BM25/BGE/Hybrid all see the same task queries; original and rewritten are separate",
                  "Hybrid is fixed RRF k=60 equal weights; no tuning"],
        "per_task": {variant: {method: rankings[variant][method] for method in rankings[variant]}
                     for variant in variants},
    }
    (RUNS / "natural_retrieval.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    if selected is not None:
        pool = []
        history_pools = {row["task_id"]: row for row in load_jsonl(REPO / "runs/p14d1r/pools/natural.jsonl")}
        ranks = rankings[selected["query_variant"]][selected["method"]]
        for tid, task in tasks.items():
            candidates = []
            for rank, item_id in enumerate(ranks[tid][:20], start=1):
                product = catalogue.get(item_id)
                if product is None:
                    continue
                candidates.append({"item_id": item_id, "rank": rank, "title": product.title, "text": product.text[:700]})
            pool.append({**history_pools[tid], "candidates": candidates,
                         "retrieval_method": selected["method"], "retrieval_query": selected["query_variant"]})
        (RUNS / "pools").mkdir(parents=True, exist_ok=True)
        (RUNS / "pools/improved_natural.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pool), encoding="utf-8")
        selected["fixed_pool_tasks"] = len(pool)
        selected["fixed_pool_note"] = "top-20 from pre-registered method; target not injected; same pool for all history methods"
    (RUNS / "natural_retrieval.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"variants": result["variants"], "selected": selected,
                      "target_filter": {key: value for key, value in target_filter.items() if key != "per_task"}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
