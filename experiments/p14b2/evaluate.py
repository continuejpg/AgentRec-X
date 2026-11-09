"""P14-B2 local evaluation: BM25 vs V0 vs Neural Query-Only vs Query+Preferences.

All four methods rank the SAME fixed, Product Type/Role filtered candidate pool (built by
build_pools.py).  Evaluator labels are used only here, after neural scoring, never as model input.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.p14b1.diagnose import verified_ids
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p14b2"
POOLS = OUT / "pools/dev.jsonl"
SCORES = OUT / "scores_dev.json"
TRAIN_POOLS = OUT / "pools/train_smoke.jsonl"
TRAIN_SCORES = OUT / "scores_train_smoke.json"
METHODS = ("bm25", "v0", "neural_query_only", "neural_query_prefs")
METHOD_LABELS = {
    "bm25": "BM25",
    "v0": "Preference Scorer V0",
    "neural_query_only": "Neural Query Only",
    "neural_query_prefs": "Neural Query + Active Preferences",
}
FORM_FACTORS = {"over_ear", "on_ear", "in_ear"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank(pool: dict[str, Any], method: str, scores: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates = list(pool["candidates"])
    if method == "bm25":
        return sorted(candidates, key=lambda c: (c["bm25_rank"], c["item_id"]))
    if method == "v0":
        return sorted(candidates, key=lambda c: (c.get("v0_rank") if c.get("v0_rank") is not None else 10**9,
                                                 c["bm25_rank"], c["item_id"]))
    key = "query_only" if method == "neural_query_only" else "query_prefs"
    assert scores is not None
    score_map = scores["scores"][pool["task_id"]][key]
    return sorted(candidates, key=lambda c: (-float(score_map.get(c["item_id"], float("-inf"))),
                                             c["bm25_rank"], c["item_id"]))


def _preference_stats(product: Any, preferences: list[dict[str, Any]]) -> dict[str, int]:
    matched = violated = unknown = 0
    form = getattr(product, "form_factor", None) if product else None
    attributes = set((product.attributes or {})) if product else set()
    for preference in preferences:
        value = str(preference.get("value"))
        is_negative = preference.get("polarity") == "negative"
        if value in FORM_FACTORS:
            observed = form == value
            contradicts = form is not None and form != value
        else:
            observed = value in attributes
            contradicts = False
        if observed and is_negative:
            violated += 1
        elif observed:
            matched += 1
        elif contradicts and not is_negative:
            violated += 1
        elif not contradicts:
            unknown += 1
    return {"matched": matched, "violated": violated, "unknown": unknown}


def _hard_violations(product: Any, hard: list[dict[str, Any]]) -> int:
    if product is None:
        return 0
    form = product.form_factor
    attributes = set(product.attributes or {})
    violations = 0
    for requirement in hard:
        value = str(requirement.get("value"))
        if value in FORM_FACTORS:
            if form is not None and form != value:
                violations += 1
        elif value not in attributes:
            # missing evidence is "unknown", not a known violation
            continue
    return violations


def evaluate_pool(pool: dict[str, Any], evaluator: dict[str, Any], catalogue: ElectronicsCatalogue,
                  scores: dict[str, Any] | None, history: dict[str, Any]) -> dict[str, Any]:
    verified = verified_ids(evaluator, catalogue) if evaluator["expected_decision"] == "recommend" else set()
    policy = pool["target"]["product_type"]
    task = {"task_id": pool["task_id"], "task_type": evaluator["task_type"],
            "expected_decision": evaluator["expected_decision"], "pool_size": len(pool["candidates"]),
            "role_rejected": pool["role_rejected"], "verified_total": len(verified),
            "candidate_unreachable": bool(verified) and not (verified & {c["item_id"] for c in pool["candidates"]}),
            "methods": {}}
    for method in METHODS:
        ranked = _rank(pool, method, scores)
        ids = [c["item_id"] for c in ranked]
        valid_ranks = [rank for rank, item_id in enumerate(ids, start=1) if item_id in verified]
        top = ranked[0] if ranked else None
        product = catalogue.get(top["item_id"]) if top else None
        stats = _preference_stats(product, pool["active_preferences"])
        task["methods"][method] = {
            "ranked_ids": ids,
            "top1": ids[0] if ids else None,
            "top5": ids[:5],
            "top1_valid": bool(ids) and ids[0] in verified,
            "top5_valid": bool(valid_ranks and valid_ranks[0] <= 5),
            "mrr_at_10": round(1.0 / valid_ranks[0], 6) if valid_ranks and valid_ranks[0] <= 10 else 0.0,
            "first_valid_rank": valid_ranks[0] if valid_ranks else None,
            "preference_stats": stats,
            "hard_constraint_violations": _hard_violations(product, pool["hard_requirements"]),
            "product_type_violation": bool(product and policy and product.product_type != policy),
            "top1_title": product.title if product else None,
            "top1_product_type": product.product_type if product else None,
        }
    task["history"] = history
    return task


def _aggregate(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    recommend = [task for task in tasks if task["expected_decision"] == "recommend"]
    summary = {}
    for method in METHODS:
        rows = [task["methods"][method] for task in recommend]
        summary[method] = {
            "recommend_tasks": len(rows),
            "top1_valid": sum(1 for row in rows if row["top1_valid"]),
            "top5_valid": sum(1 for row in rows if row["top5_valid"]),
            "mrr_at_10": round(sum(row["mrr_at_10"] for row in rows) / len(rows), 6) if rows else 0.0,
            "preference_matched_top1": sum(row["preference_stats"]["matched"] for row in rows),
            "preference_violated_top1": sum(row["preference_stats"]["violated"] for row in rows),
            "preference_unknown_top1": sum(row["preference_stats"]["unknown"] for row in rows),
            "hard_constraint_violations": sum(row["hard_constraint_violations"] for row in rows),
            "product_type_violations": sum(1 for row in rows if row["product_type_violation"]),
            "candidate_unreachable": sum(1 for task in recommend if task["candidate_unreachable"]),
        }
    return summary


def _ablation(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    recommend = [task for task in tasks if task["expected_decision"] == "recommend"]
    with_prefs = [task for task in recommend if task["history"]["prefs"]]
    no_prefs = [task for task in recommend if not task["history"]["prefs"]]
    changed = []
    for task in with_prefs:
        q = task["methods"]["neural_query_only"]["ranked_ids"]
        p = task["methods"]["neural_query_prefs"]["ranked_ids"]
        if q != p:
            changed.append({
                "task_id": task["task_id"], "top1_query_only": q[0], "top1_query_prefs": p[0],
                "top1_changed": q[0] != p[0],
                "query_only_valid": task["methods"]["neural_query_only"]["top1_valid"],
                "query_prefs_valid": task["methods"]["neural_query_prefs"]["top1_valid"],
                "pref_summary": task["history"]["prefs"],
            })
    no_pref_diff = [task["task_id"] for task in no_prefs
                    if task["methods"]["neural_query_only"]["ranked_ids"] !=
                    task["methods"]["neural_query_prefs"]["ranked_ids"]]
    return {
        "history_tasks": len(with_prefs), "no_history_tasks": len(no_prefs),
        "history_tasks_order_changed": len(changed),
        "history_top1_changed": sum(1 for row in changed if row["top1_changed"]),
        "history_changed_tasks": changed,
        "no_history_order_changed": no_pref_diff,
        "neural_prefs_top1_valid": sum(1 for task in with_prefs
                                       if task["methods"]["neural_query_prefs"]["top1_valid"]),
        "neural_query_top1_valid": sum(1 for task in with_prefs
                                       if task["methods"]["neural_query_only"]["top1_valid"]),
    }


def _failure_cases(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        if task["expected_decision"] != "recommend":
            continue
        if task["task_id"] not in {"p14e2_lite_0046", "p14e2_lite_0053"} and \
                all(task["methods"][method]["top1_valid"] for method in METHODS):
            continue
        rows.append({
            "task_id": task["task_id"], "task_type": task["task_type"],
            "history": task["history"],
            "methods": {
                method: {
                    "top1": task["methods"][method]["top1"],
                    "top1_title": task["methods"][method]["top1_title"],
                    "valid": task["methods"][method]["top1_valid"],
                    "first_valid_rank": task["methods"][method]["first_valid_rank"],
                    "preference_stats": task["methods"][method]["preference_stats"],
                    "top5": task["methods"][method]["top5"],
                }
                for method in METHODS
            },
        })
    return rows


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    evaluators = {line["task_id"]: line for line in _load_jsonl(REPO / "data/p14e2/benchmark/dev.evaluator.jsonl")}
    pools = _load_jsonl(POOLS)
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    train_pools = _load_jsonl(TRAIN_POOLS)
    train_scores = json.loads(TRAIN_SCORES.read_text(encoding="utf-8"))
    train_evaluators = {line["task_id"]: line for line in _load_jsonl(REPO / "data/p14e2/benchmark/train.evaluator.jsonl")}

    # Per-task history metadata (legal active preferences only).
    history = {pool["task_id"]: {"prefs": pool["preference_summary"] is not None,
                                 "summary": pool["preference_summary"],
                                 "active_count": len(pool["active_preferences"])} for pool in pools}
    tasks = []
    for pool in pools:
        tasks.append(evaluate_pool(pool, evaluators[pool["task_id"]], catalogue, scores, history[pool["task_id"]]))
    train_tasks = [evaluate_pool(pool, train_evaluators[pool["task_id"]], catalogue, train_scores,
                                 {"prefs": pool["preference_summary"] is not None,
                                  "summary": pool["preference_summary"],
                                  "active_count": len(pool["active_preferences"])})
                   for pool in train_pools]
    summary = _aggregate(tasks)
    train_summary = _aggregate(train_tasks)
    ablation = _ablation(tasks)
    failures = _failure_cases(tasks)

    # Ranking change vs V0 / BM25 for the report.
    changed = {}
    for method in METHODS:
        changed[method] = {
            "vs_bm25_top1_changed": sum(1 for task in tasks if task["expected_decision"] == "recommend"
                                       and task["methods"][method]["top1"] != task["methods"]["bm25"]["top1"]),
            "vs_v0_top1_changed": sum(1 for task in tasks if task["expected_decision"] == "recommend"
                                      and task["methods"][method]["top1"] != task["methods"]["v0"]["top1"]),
        }
    improved = {}
    for method in METHODS:
        improved[method] = {
            "tasks_top1_correct_where_v0_wrong": [
                task["task_id"] for task in tasks if task["expected_decision"] == "recommend"
                and task["methods"][method]["top1_valid"] and not task["methods"]["v0"]["top1_valid"]],
            "tasks_top1_wrong_where_v0_correct": [
                task["task_id"] for task in tasks if task["expected_decision"] == "recommend"
                and not task["methods"][method]["top1_valid"] and task["methods"]["v0"]["top1_valid"]],
        }
    model = scores["model"]
    resources = scores["resources"]
    total_seconds = round(sum(scores["task_seconds"].values()), 3)
    model_manifest = {
        "schema": "agentrecx.p14b2.model_manifest.v1",
        "repo_id": model["repo_id"],
        "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
        "source": "Hugging Face via hf-mirror.com snapshot_download",
        "local_path_remote": model["path"],
        "license": "MIT (per model card README)",
        "architecture": model["architectures"],
        "num_parameters": model["num_parameters"],
        "dtype": model["dtype"],
        "max_length": model["max_length"],
        "batch_size": model["batch_size"],
        "transformers": model["transformers"],
        "torch": model["torch"],
        "gpu": "NVIDIA GeForce RTX 4090 D 24GB (shared with running Qwen service)",
        "resources": {
            **resources,
            "total_scoring_seconds": total_seconds,
            "tasks": scores["tasks"],
            "pairs_scored": scores["pairs_scored"],
            "candidates_scored": scores["candidates_scored"],
            "mean_task_seconds": round(total_seconds / max(1, scores["tasks"]), 3),
        },
    }
    comparison = {
        "schema": "agentrecx.p14b2.comparison.v1",
        "pool_definition": {
            "source": "BM25 top-20; Product Type/Role MATCH/NOT_REQUIRED only",
            "same_pool_for_all_methods": True,
            "top_k": 20,
            "tie_break": "bm25_rank then item_id",
        },
        "methods": {method: METHOD_LABELS[method] for method in METHODS},
        "summary": summary,
        "train_smoke": train_summary,
        "ablation": ablation,
        "changed": changed,
        "improved": improved,
        "failures": failures,
        "model_manifest": model_manifest,
        "per_task": tasks,
        "notes": [
            "Dev 20 was already used for P14-C0-R development; this is a development-set diagnostic.",
            "Labels used only for evaluation after ranking; no evaluator field is in the reranker input.",
            "All methods share the same role-filtered candidates; type-mismatched candidates cannot return via neural score.",
            "No SFT/LTR training; pretrained BAAI/bge-reranker-base inference only.",
            "Qwen service was not stopped; reranker ran one-shot and released GPU memory.",
        ],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=1, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    with (OUT / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "ablation": {k: v for k, v in ablation.items()
                                                        if k not in {"history_changed_tasks"}},
                      "failures": [row["task_id"] for row in failures],
                      "resources": model_manifest["resources"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
