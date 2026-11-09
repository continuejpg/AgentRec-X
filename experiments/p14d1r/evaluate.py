"""P14-D1-R evaluation: recovered-history ablation, shuffled control, retrieval diagnosis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
DATA_R = REPO / "data/p14d1r"
RUNS_D1 = REPO / "runs/p14d1"
RUNS = REPO / "runs/p14d1r"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_scores(pool: dict[str, Any], score_map: dict[str, float]) -> list[str]:
    return [candidate["item_id"] for candidate in sorted(
        pool["candidates"],
        key=lambda candidate: (-float(score_map.get(candidate["item_id"], float("-inf"))),
                               candidate["bm25_rank"], candidate["item_id"]))]


def rank_bm25(pool: dict[str, Any]) -> list[str]:
    return [candidate["item_id"] for candidate in sorted(
        pool["candidates"], key=lambda candidate: (candidate["bm25_rank"], candidate["item_id"]))]


def rank_v0(pool: dict[str, Any], catalogue: ElectronicsCatalogue) -> list[str]:
    rows = []
    for candidate in pool["candidates"]:
        product = catalogue.get(candidate["item_id"])
        if product is None:
            continue
        rows.append({"product_ref": {"namespace": ELECTRONICS_NAMESPACE, "item_id": candidate["item_id"]},
                     "query_rank": candidate["bm25_rank"], "query_score": candidate.get("bm25_score", 0.0),
                     "attributes": {name: {"evidence": "catalog_meta:text"} for name in (product.attributes or {})},
                     "form_factor": product.form_factor})
    if not rows:
        return []
    scored = PreferenceAwareScorer().score(preferences=pool.get("history_proxy_preferences") or [],
                                           current_query=pool["query_original"], candidates=rows,
                                           hard_requirements=[], namespace=ELECTRONICS_NAMESPACE)
    by_id = {row["product_ref"]["item_id"]: row for row in scored}
    return sorted(by_id, key=lambda item_id: (by_id[item_id].get("rank") or 10**9,
                                              by_id[item_id].get("query_rank") or 10**9, item_id))


def evaluate(ranked: list[str], target_id: str) -> dict[str, Any]:
    rank = ranked.index(target_id) + 1 if target_id in ranked else None
    return {"target_rank": rank, "hit_at_1": rank == 1,
            "hit_at_5": bool(rank and rank <= 5), "hit_at_10": bool(rank and rank <= 10),
            "mrr_at_10": round(1.0 / rank, 6) if rank and rank <= 10 else 0.0,
            "target_in_pool": rank is not None}


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
    tasks = load_jsonl(DATA / "dev_tasks.jsonl")
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    natural = {row["task_id"]: row for row in load_jsonl(RUNS / "pools/natural.jsonl")}
    target5 = {row["task_id"]: row for row in load_jsonl(RUNS / "pools/target5.jsonl")}
    qo_natural = json.loads((RUNS_D1 / "scores_natural.json").read_text(encoding="utf-8"))
    qo_target5 = json.loads((RUNS_D1 / "scores_target5.json").read_text(encoding="utf-8"))
    hist_natural = json.loads((RUNS / "scores_natural.json").read_text(encoding="utf-8"))
    hist_target5 = json.loads((RUNS / "scores_target5.json").read_text(encoding="utf-8"))
    before_natural = json.loads((RUNS_D1 / "scores_natural.json").read_text(encoding="utf-8"))
    before_target5 = json.loads((RUNS_D1 / "scores_target5.json").read_text(encoding="utf-8"))
    coverage = {row["task_id"]: row for row in json.loads(
        (DATA_R / "history_metadata_manifest.json").read_text(encoding="utf-8"))["per_task"]}

    natural_rows = []
    for task in tasks:
        tid = task["task_id"]
        target_id = targets[tid]["target_parent_asin"]
        pool = natural[tid]
        row = {"task_id": tid, "history_length": coverage[tid]["history_items"],
               "history_with_metadata": coverage[tid]["history_with_metadata"],
               "history_verified": coverage[tid]["history_verified"],
               "history_proxy_preferences": coverage[tid]["history_proxy_preferences"],
               "methods": {}}
        row["methods"]["bm25"] = evaluate(rank_bm25(pool), target_id)
        row["methods"]["v0_proxy_recovered"] = evaluate(rank_v0(pool, catalogue), target_id)
        row["methods"]["qwen3_query_only"] = evaluate(rank_scores(pool, qo_natural["scores"]["query_only"][tid]), target_id)
        row["methods"]["qwen3_history_recovered"] = evaluate(rank_scores(pool, hist_natural["scores"]["history_real"][tid]), target_id)
        row["methods"]["qwen3_history_shuffled"] = evaluate(rank_scores(pool, hist_natural["scores"]["history_shuffled"][tid]), target_id)
        row["methods"]["qwen3_history_before_recovery"] = evaluate(rank_scores(pool, before_natural["scores"]["history"][tid]), target_id)
        row["target_in_natural_pool"] = row["methods"]["bm25"]["target_in_pool"]
        natural_rows.append(row)

    target_rows = []
    for tid, pool in target5.items():
        target_id = targets[tid]["target_parent_asin"]
        row = {"task_id": tid, "history_length": coverage[tid]["history_items"],
               "history_with_metadata": coverage[tid]["history_with_metadata"],
               "history_verified": coverage[tid]["history_verified"],
               "methods": {}}
        row["methods"]["bm25"] = evaluate(rank_bm25(pool), target_id)
        row["methods"]["v0_proxy_recovered"] = evaluate(rank_v0(pool, catalogue), target_id)
        row["methods"]["qwen3_query_only"] = evaluate(rank_scores(pool, qo_target5["scores"]["query_only"][tid]), target_id)
        row["methods"]["qwen3_history_recovered"] = evaluate(rank_scores(pool, hist_target5["scores"]["history_real"][tid]), target_id)
        row["methods"]["qwen3_history_shuffled"] = evaluate(rank_scores(pool, hist_target5["scores"]["history_shuffled"][tid]), target_id)
        row["methods"]["qwen3_history_before_recovery"] = evaluate(rank_scores(pool, before_target5["scores"]["history"][tid]), target_id)
        target_rows.append(row)

    methods = ("bm25", "v0_proxy_recovered", "qwen3_query_only", "qwen3_history_before_recovery",
               "qwen3_history_recovered", "qwen3_history_shuffled")
    natural_summary = {method: {"all": aggregate(natural_rows, method),
                                "conditional_target_in_pool": aggregate(natural_rows, method, True)}
                       for method in methods}
    target_summary = {method: aggregate(target_rows, method) for method in methods}

    # Same-model same-pool history effects; exclude target-unreachable natural tasks for conditional view.
    effects = {}
    for protocol, rows in (("natural", natural_rows), ("target5", target_rows)):
        source = [row for row in rows if row["methods"]["qwen3_history_recovered"]["target_in_pool"]
                  or protocol == "target5"]
        def ranks(method):
            return {row["task_id"]: row["methods"][method]["target_rank"] for row in source}
        real = ranks("qwen3_history_recovered")
        shuffled = ranks("qwen3_history_shuffled")
        query = ranks("qwen3_query_only")
        before = ranks("qwen3_history_before_recovery")
        def delta(base, other):
            improved = [tid for tid in base if other[tid] is not None and (base[tid] is None or other[tid] < base[tid])]
            worsened = [tid for tid in base if base[tid] is not None and (other[tid] is None or other[tid] > base[tid])]
            return {"improved": improved, "worsened": worsened,
                    "n_improved": len(improved), "n_worsened": len(worsened)}
        effects[protocol] = {
            "n": len(source),
            "real_vs_query_only": delta(query, real),
            "shuffled_vs_query_only": delta(query, shuffled),
            "real_vs_shuffled": delta(shuffled, real),
            "before_vs_after_metadata": delta(before, real),
            "qwen3_history_before_recovery_hit1": sum(1 for row in source if row["methods"]["qwen3_history_before_recovery"]["hit_at_1"]),
            "qwen3_history_recovered_hit1": sum(1 for row in source if row["methods"]["qwen3_history_recovered"]["hit_at_1"]),
            "qwen3_history_shuffled_hit1": sum(1 for row in source if row["methods"]["qwen3_history_shuffled"]["hit_at_1"]),
            "qwen3_query_only_hit1": sum(1 for row in source if row["methods"]["qwen3_query_only"]["hit_at_1"]),
        }

    # Coverage buckets, using only target-reachable natural tasks for ranking; report counts.
    buckets = {"0": [], "1-2": [], "3-7": [], "8+": []}
    for row in target_rows:
        count = row["history_with_metadata"]
        key = "0" if count == 0 else "1-2" if count <= 2 else "3-7" if count <= 7 else "8+"
        buckets[key].append(row)
    bucket_summary = {}
    for key, rows in buckets.items():
        if rows:
            bucket_summary[key] = {
                "n": len(rows),
                "query_only_hit1": sum(1 for row in rows if row["methods"]["qwen3_query_only"]["hit_at_1"]),
                "recovered_hit1": sum(1 for row in rows if row["methods"]["qwen3_history_recovered"]["hit_at_1"]),
                "shuffled_hit1": sum(1 for row in rows if row["methods"]["qwen3_history_shuffled"]["hit_at_1"]),
                "query_only_mrr": round(sum(row["methods"]["qwen3_query_only"]["mrr_at_10"] for row in rows) / len(rows), 6),
                "recovered_mrr": round(sum(row["methods"]["qwen3_history_recovered"]["mrr_at_10"] for row in rows) / len(rows), 6),
            }

    retrieval = json.loads((RUNS / "retrieval_diagnosis.json").read_text(encoding="utf-8")) if (RUNS / "retrieval_diagnosis.json").is_file() else None

    # Blind review with recovered history; no target ID, no model ranks.
    blind = []
    for task in tasks[:25]:
        tid = task["task_id"]
        pool = target5.get(tid, natural[tid])
        candidates = [{"label": chr(65 + index), "title": candidate["title"], "evidence": candidate["text"][:300]}
                      for index, candidate in enumerate(pool["candidates"][:5])]
        blind.append({"case_id": f"p14d1r_blind_{len(blind) + 1:02d}",
                      "query_original": task["query_original"],
                      "query_rewritten": task["query_rewritten"],
                      "history": task["history"][-10:],
                      "candidates": candidates,
                      "review_status": "pending_human_review", "human_verified": False})
    (RUNS / "blind_review.json").write_text(json.dumps(
        {"schema": "agentrecx.p14d1r.blind_review.v1", "human_verified": False,
         "review_status": "pending_human_review", "cases": blind}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")

    manifest = json.loads((DATA_R / "history_metadata_manifest.json").read_text(encoding="utf-8"))
    comparison = {
        "schema": "agentrecx.p14d1r.comparison.v1",
        "exposed_split": True,
        "metadata_recovery": manifest,
        "retrieval_diagnosis": retrieval,
        "protocol_a_natural": {"tasks": len(natural_rows), "methods": natural_summary},
        "protocol_b_target5": {"tasks": len(target_rows), "methods": target_summary},
        "history_effects": effects,
        "coverage_buckets_target5": bucket_summary,
        "cost": {"natural_history_scores": hist_natural["resources"],
                 "target5_history_scores": hist_target5["resources"],
                 "query_only_scores_reused_from_d1": True},
        "blind_review": "runs/p14d1r/blind_review.json",
        "per_task_natural": natural_rows,
        "per_task_target5": target_rows,
        "notes": ["same Qwen3 model/query/candidate pool as P14-D1; only history metadata changed",
                  "target-included target5 is conditional and is not natural retrieval performance",
                  "shuffled history is from a different user, length/category matched, recipient target removed",
                  "no learning/training; labels are observed positives only"],
    }
    (RUNS / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (RUNS / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in natural_rows:
            handle.write(json.dumps({"protocol": "natural", **row}, ensure_ascii=False, sort_keys=True) + "\n")
        for row in target_rows:
            handle.write(json.dumps({"protocol": "target5", **row}, ensure_ascii=False, sort_keys=True) + "\n")
    (RUNS / "model_manifest.json").write_text(json.dumps({
        "schema": "agentrecx.p14d1r.model_manifest.v1",
        "qwen3_reranker": hist_natural["model"],
        "neutral_instruction": hist_natural["model"]["instruction"],
        "resources_natural": hist_natural["resources"],
        "resources_target5": hist_target5["resources"],
        "v0_proxy": "Preference Scorer V0 + weak observed-history attribute proxy (positive only; no target)",
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"natural": natural_summary, "target5": target_summary,
                      "effects": {k: {kk: vv for kk, vv in v.items() if kk not in ("real_vs_query_only", "shuffled_vs_query_only", "real_vs_shuffled", "before_vs_after_metadata")} for k, v in effects.items()},
                      "coverage": {k: manifest["after_metadata_coverage"] for k in ("after_metadata_coverage",)},
                      "retrieval": retrieval["bm25_query_variants"] if retrieval else None}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
