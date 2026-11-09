"""P14-D1 local evaluation: real-history reranking under two protocols.

Protocol A: natural BM25 top-20 (target may be unreachable); natural metrics include misses.
Protocol B: target-included 1-in-5 diagnostic (target is one of five; conditional reranking only).
Methods: BM25, Qwen3-Reranker-4B Query Only, Qwen3-Reranker-4B + Real History, V0 history proxy,
and a rewritten-query Query-Only diagnostic (reported separately).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
RUNS = REPO / "runs/p14d1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_from_scores(pool: dict[str, Any], score_map: dict[str, float]) -> list[str]:
    return [candidate["item_id"] for candidate in sorted(
        pool["candidates"],
        key=lambda candidate: (-float(score_map.get(candidate["item_id"], float("-inf"))),
                               candidate["bm25_rank"], candidate["item_id"]),
    )]


def rank_bm25(pool: dict[str, Any]) -> list[str]:
    return [candidate["item_id"] for candidate in sorted(
        pool["candidates"], key=lambda candidate: (candidate["bm25_rank"], candidate["item_id"]))]


def v0_rank(pool: dict[str, Any], catalogue: ElectronicsCatalogue) -> list[str]:
    rows = []
    for candidate in pool["candidates"]:
        product = catalogue.get(candidate["item_id"])
        if product is None:
            continue
        rows.append({
            "product_ref": {"namespace": ELECTRONICS_NAMESPACE, "item_id": candidate["item_id"]},
            "query_rank": candidate["bm25_rank"], "query_score": candidate.get("bm25_score", 0.0),
            "attributes": {name: {"evidence": "catalog_meta:text"} for name in (product.attributes or {})},
            "form_factor": product.form_factor,
        })
    if not rows:
        return []
    scored = PreferenceAwareScorer().score(
        preferences=pool.get("history_proxy_preferences") or [],
        current_query=pool["query_original"], candidates=rows,
        hard_requirements=[], namespace=ELECTRONICS_NAMESPACE,
    )
    by_id = {row["product_ref"]["item_id"]: row for row in scored}
    order = sorted(by_id, key=lambda item_id: (by_id[item_id].get("rank") or 10**9,
                                               by_id[item_id].get("query_rank") or 10**9, item_id))
    return order


def evaluate_order(ranked: list[str], target_id: str) -> dict[str, Any]:
    rank = None
    if target_id in ranked:
        rank = ranked.index(target_id) + 1
    return {
        "target_rank": rank,
        "hit_at_1": bool(rank == 1),
        "hit_at_5": bool(rank is not None and rank <= 5),
        "hit_at_10": bool(rank is not None and rank <= 10),
        "mrr_at_10": round(1.0 / rank, 6) if rank is not None and rank <= 10 else 0.0,
        "target_in_pool": rank is not None,
    }


def aggregate(rows: list[dict[str, Any]], method: str, *, conditional: bool = False) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row["methods"][method]
        if conditional and not value["target_in_pool"]:
            continue
        values.append(value)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "hit_at_1": sum(1 for value in values if value["hit_at_1"]),
        "hit_at_5": sum(1 for value in values if value["hit_at_5"]),
        "hit_at_10": sum(1 for value in values if value["hit_at_10"]),
        "mrr_at_10": round(sum(value["mrr_at_10"] for value in values) / len(values), 6),
    }


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    tasks = load_jsonl(DATA / "dev_tasks.jsonl")
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    natural = {row["task_id"]: row for row in load_jsonl(RUNS / "pools/natural.jsonl")}
    target5 = {row["task_id"]: row for row in load_jsonl(RUNS / "pools/target5.jsonl")}
    natural_scores = json.loads((RUNS / "scores_natural.json").read_text(encoding="utf-8"))
    target5_scores = json.loads((RUNS / "scores_target5.json").read_text(encoding="utf-8"))

    natural_rows = []
    for task in tasks:
        tid = task["task_id"]
        target = targets[tid]
        pool = natural[tid]
        row = {"task_id": tid, "task_type": "natural", "target_category": target["target_category"],
               "target_in_catalogue": target["target_in_catalogue"],
               "target_bm25_rank_top100": None, "history_length": task["history_length"],
               "history_in_catalogue": task["history_in_catalogue"],
               "history_verified": task["history_verified"],
               "methods": {}, "query_original": task["query_original"],
               "query_rewritten": task["query_rewritten"]}
        for method, scores in (("qwen3_query_only", natural_scores["scores"]["query_only"]),
                               ("qwen3_history", natural_scores["scores"]["history"]),
                               ("qwen3_query_only_rewritten", natural_scores["scores"].get("query_only_rewritten"))):
            if scores is None:
                continue
            row["methods"][method] = evaluate_order(rank_from_scores(pool, scores[tid]), target["target_parent_asin"])
        row["methods"]["bm25"] = evaluate_order(rank_bm25(pool), target["target_parent_asin"])
        row["methods"]["v0_proxy"] = evaluate_order(v0_rank(pool, catalogue), target["target_parent_asin"])
        natural_rows.append(row)

    # Top-100 reachability from the candidate manifest (target rank is evaluator-side only).
    candidate_manifest = json.loads((RUNS / "candidate_manifest.json").read_text(encoding="utf-8"))
    for row, per_task in zip(natural_rows, candidate_manifest["per_task"]):
        row["target_bm25_rank_top100"] = per_task["target_rank_bm25_top100"]

    target_rows = []
    for tid, pool in target5.items():
        target = targets[tid]
        row = {"task_id": tid, "target_category": target["target_category"],
               "history_length": pool["history_length"],
               "history_in_catalogue": pool["history_in_catalogue_count"],
               "history_verified": pool["history_verified_count"],
               "methods": {}, "query_original": pool["query_original"]}
        for method, scores in (("qwen3_query_only", target5_scores["scores"]["query_only"]),
                               ("qwen3_history", target5_scores["scores"]["history"]),
                               ("qwen3_query_only_rewritten", target5_scores["scores"].get("query_only_rewritten"))):
            if scores is None:
                continue
            row["methods"][method] = evaluate_order(rank_from_scores(pool, scores[tid]), target["target_parent_asin"])
        row["methods"]["bm25"] = evaluate_order(rank_bm25(pool), target["target_parent_asin"])
        row["methods"]["v0_proxy"] = evaluate_order(v0_rank(pool, catalogue), target["target_parent_asin"])
        target_rows.append(row)

    methods = ("bm25", "v0_proxy", "qwen3_query_only", "qwen3_history", "qwen3_query_only_rewritten")
    summary = {}
    for method in methods:
        if not any(method in row["methods"] for row in natural_rows):
            continue
        summary[method] = {
            "natural_all_135": aggregate(natural_rows, method),
            "natural_target_in_pool": aggregate(natural_rows, method, conditional=True),
        }
    target_summary = {method: aggregate(target_rows, method) for method in methods
                      if any(method in row["methods"] for row in target_rows)}

    # History increment for the same Qwen model/query/candidate pool.
    increments = {}
    for protocol, rows in (("natural", natural_rows), ("target5", target_rows)):
        q = [row for row in rows if "qwen3_history" in row["methods"] and "qwen3_query_only" in row["methods"]]
        increments[protocol] = {
            "n": len(q),
            "history_hit1": sum(1 for row in q if row["methods"]["qwen3_history"]["hit_at_1"]),
            "query_only_hit1": sum(1 for row in q if row["methods"]["qwen3_query_only"]["hit_at_1"]),
            "history_mrr": round(sum(row["methods"]["qwen3_history"]["mrr_at_10"] for row in q) / len(q), 6),
            "query_only_mrr": round(sum(row["methods"]["qwen3_query_only"]["mrr_at_10"] for row in q) / len(q), 6),
            "history_improved": [row["task_id"] for row in q
                                 if row["methods"]["qwen3_history"]["target_rank"] and
                                 (row["methods"]["qwen3_query_only"]["target_rank"] is None or
                                  row["methods"]["qwen3_history"]["target_rank"] < row["methods"]["qwen3_query_only"]["target_rank"])],
            "history_worsened": [row["task_id"] for row in q
                                 if row["methods"]["qwen3_query_only"]["target_rank"] and
                                 (row["methods"]["qwen3_history"]["target_rank"] is None or
                                  row["methods"]["qwen3_history"]["target_rank"] > row["methods"]["qwen3_query_only"]["target_rank"])],
        }
        # By history length and metadata coverage.
        for label, predicate in (("history_1_9", lambda row: 1 <= row["history_length"] <= 9),
                                 ("history_10_49", lambda row: 10 <= row["history_length"] <= 49),
                                 ("history_50_plus", lambda row: row["history_length"] >= 50),
                                 ("history_with_catalogue_metadata", lambda row: row["history_in_catalogue"] > 0),
                                 ("history_without_catalogue_metadata", lambda row: row["history_in_catalogue"] == 0)):
            subset = [row for row in q if predicate(row)]
            if subset:
                increments[protocol][label] = {
                    "n": len(subset),
                    "query_only_hit1": sum(1 for row in subset if row["methods"]["qwen3_query_only"]["hit_at_1"]),
                    "history_hit1": sum(1 for row in subset if row["methods"]["qwen3_history"]["hit_at_1"]),
                    "query_only_mrr": round(sum(row["methods"]["qwen3_query_only"]["mrr_at_10"] for row in subset) / len(subset), 6),
                    "history_mrr": round(sum(row["methods"]["qwen3_history"]["mrr_at_10"] for row in subset) / len(subset), 6),
                }

    # Same-query user distinction (if the dev split has repeated queries).
    query_groups = Counter(task["query_original"] for task in tasks)
    repeated = {query: count for query, count in query_groups.items() if count > 1}

    # Blind review: 25 cases with history, anonymous candidates, no model ranks/target IDs.
    review = []
    for task in tasks[:25]:
        tid = task["task_id"]
        pool = target5.get(tid, natural[tid])
        candidates = []
        for index, candidate in enumerate(pool["candidates"][:5]):
            candidates.append({"label": chr(65 + index), "title": candidate["title"],
                               "evidence": candidate["text"][:300]})
        review.append({
            "case_id": f"p14d1_blind_{len(review) + 1:02d}",
            "query_original": task["query_original"],
            "query_rewritten": task["query_rewritten"],
            "history": task["history"][-10:],
            "candidate_count": len(candidates), "candidates": candidates,
            "review_status": "pending_human_review", "human_verified": False,
            "notes": "Candidates anonymized; target ID and model rankings intentionally omitted.",
        })
    (RUNS / "blind_review.json").write_text(json.dumps(
        {"schema": "agentrecx.p14d1.blind_review.v1", "human_verified": False,
         "review_status": "pending_human_review", "cases": review}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")

    model_manifest = {
        "schema": "agentrecx.p14d1.model_manifest.v1",
        "qwen3_reranker": natural_scores["model"],
        "qwen3_resources_natural": natural_scores["resources"],
        "qwen3_resources_target5": target5_scores["resources"],
        "v0_proxy": {"name": "Preference Scorer V0 + weak history attribute proxy",
                     "definition": "positive attributes/form factors observed in pre-target catalogue-mapped history items; no negatives; no target-derived features"},
        "bm25": "P13-R Electronics BM25 (catalog_search)",
    }
    comparison = {
        "schema": "agentrecx.p14d1.comparison.v1",
        "alignment": json.loads((DATA / "alignment_manifest.json").read_text(encoding="utf-8")),
        "candidate_manifest": candidate_manifest,
        "protocol_a_natural": {
            "definition": "BM25 top-20; target not injected; misses count as unreachable",
            "tasks": len(natural_rows),
            "methods": summary,
        },
        "protocol_b_target5": {
            "definition": "target-included 1-in-5 diagnostic; four deterministic BM25 distractors; conditional reranking only",
            "tasks": len(target_rows),
            "methods": target_summary,
        },
        "history_increment": increments,
        "same_query_repeated": repeated,
        "target_in_catalogue": candidate_manifest["target_in_catalogue"],
        "target_in_bm25_top20": candidate_manifest["target_in_bm25_top20"],
        "target_in_bm25_top100": candidate_manifest["target_in_bm25_top100"],
        "history_metadata_coverage": {
            "tasks_with_any_history_metadata": candidate_manifest["history_task_metadata_coverage"],
            "history_items_in_catalogue": candidate_manifest["history_items_in_catalogue"],
            "total_history_items": sum(task["history_length"] for task in tasks),
            "tasks_with_proxy_preferences": candidate_manifest["tasks_with_proxy_preferences"],
        },
        "temporal_validation": {
            "target_in_history": 0, "history_at_or_after_target": 0,
            "history_chronological_violations": 0,
            "target_timestamp_source": "P13-R raw dev timestamps derived from Amazon-C4 valid/test; history release is cutoff at target",
        },
        "query_mapping": {"original_available": 135, "rewritten_available": sum(1 for task in tasks if task["query_rewritten"]),
                          "primary": "original Amazon-C4", "rewritten": "diagnostic only, never merged"},
        "model_manifest": model_manifest,
        "blind_review": "runs/p14d1/blind_review.json",
        "per_task_natural": natural_rows,
        "per_task_target5": target_rows,
        "notes": [
            "Only one observed positive per query; all other products are unjudged, not true negatives.",
            "Protocol B is target-included conditional reranking and is not an end-to-end retrieval metric.",
            "The independent catalogue has limited target retrieval and sparse history metadata; these are reported as limitations.",
            "No training/fine-tuning; Qwen3 and V0 frozen; old simulation benchmarks are not mixed with these numbers.",
        ],
    }
    (RUNS / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    with (RUNS / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in natural_rows:
            handle.write(json.dumps({"protocol": "natural", **row}, ensure_ascii=False, sort_keys=True) + "\n")
        for row in target_rows:
            handle.write(json.dumps({"protocol": "target5", **row}, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"protocol_a": summary, "protocol_b": target_summary,
                      "history_increment": {k: {kk: vv for kk, vv in v.items() if kk != "history_improved" and kk != "history_worsened"}
                                            for k, v in increments.items()},
                      "counts": {"target_catalogue": candidate_manifest["target_in_catalogue"],
                                 "top20": candidate_manifest["target_in_bm25_top20"],
                                 "top100": candidate_manifest["target_in_bm25_top100"]}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
