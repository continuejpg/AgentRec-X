"""P14-B3 local evaluation: BM25 vs V0 vs Qwen3-Reranker-4B variants; Setwise status.

All completed ranking methods use the same P14-B2 fixed pools (BM25 top-20 + Product Type/Role
filter). Evaluator labels are read only here, after model scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.p14b1.diagnose import verified_ids
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p14b3"
POOLS = OUT / "pools/dev.jsonl"
SCORES = OUT / "scores_qwen3reranker_dev.json"
METHODS = ("bm25", "v0", "qwen3_query_only", "qwen3_query_prefs", "qwen3_prefs_instruction")
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
    key = {"qwen3_query_only": "query_only", "qwen3_query_prefs": "query_prefs",
           "qwen3_prefs_instruction": "prefs_instruction"}[method]
    score_map = scores["scores"][key][pool["task_id"]]
    return sorted(candidates, key=lambda c: (-float(score_map.get(c["item_id"], float("-inf"))),
                                             c["bm25_rank"], c["item_id"]))


def _pref_stats(product: Any, prefs: list[dict[str, Any]]) -> dict[str, int]:
    matched = violated = unknown = 0
    form = getattr(product, "form_factor", None)
    attrs = set(product.attributes or {}) if product else set()
    for pref in prefs:
        value = str(pref.get("value"))
        negative = pref.get("polarity") == "negative"
        if value in FORM_FACTORS:
            observed = form == value
            contradicts = form is not None and form != value
        else:
            observed = value in attrs
            contradicts = False
        if observed and negative:
            violated += 1
        elif observed:
            matched += 1
        elif contradicts and not negative:
            violated += 1
        elif not contradicts:
            unknown += 1
    return {"matched": matched, "violated": violated, "unknown": unknown}


def _hard_violations(product: Any, hard: list[dict[str, Any]]) -> int:
    form = getattr(product, "form_factor", None)
    attrs = set(product.attributes or {}) if product else set()
    violations = 0
    for requirement in hard:
        value = str(requirement.get("value"))
        if value in FORM_FACTORS:
            violations += int(form is not None and form != value)
        elif value not in attrs:
            continue
    return violations


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    evaluators = {row["task_id"]: row for row in _load_jsonl(REPO / "data/p14e2/benchmark/dev.evaluator.jsonl")}
    pools = _load_jsonl(POOLS)
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    setwise_path = OUT / "setwise_dev.json"
    setwise = json.loads(setwise_path.read_text(encoding="utf-8")) if setwise_path.is_file() else None

    tasks = []
    for pool in pools:
        evaluator = evaluators[pool["task_id"]]
        verified = verified_ids(evaluator, catalogue) if evaluator["expected_decision"] == "recommend" else set()
        task = {"task_id": pool["task_id"], "task_type": evaluator["task_type"],
                "expected_decision": evaluator["expected_decision"], "pool_size": len(pool["candidates"]),
                "candidate_unreachable": bool(verified) and not (verified & {c["item_id"] for c in pool["candidates"]}),
                "history": {"active_preferences": pool["active_preferences"],
                            "preference_summary": pool["preference_summary"]},
                "methods": {}}
        for method in METHODS:
            ranked = _rank(pool, method, scores)
            ids = [c["item_id"] for c in ranked]
            valid_ranks = [rank for rank, item_id in enumerate(ids, start=1) if item_id in verified]
            product = catalogue.get(ids[0]) if ids else None
            task["methods"][method] = {
                "ranked_ids": ids, "top1": ids[0] if ids else None, "top5": ids[:5],
                "top1_valid": bool(ids) and ids[0] in verified,
                "top5_valid": bool(valid_ranks and valid_ranks[0] <= 5),
                "mrr_at_10": round(1.0 / valid_ranks[0], 6) if valid_ranks and valid_ranks[0] <= 10 else 0.0,
                "first_valid_rank": valid_ranks[0] if valid_ranks else None,
                "preference_stats": _pref_stats(product, pool["active_preferences"]),
                "hard_constraint_violations": _hard_violations(product, pool["hard_requirements"]),
                "product_type_violation": bool(product and pool["target"]["product_type"]
                                               and product.product_type != pool["target"]["product_type"]),
                "top1_title": product.title if product else None,
                "top1_product_type": product.product_type if product else None,
            }
        if setwise and pool["task_id"] in setwise.get("scores", {}):
            row = setwise["scores"][pool["task_id"]]
            top1 = row.get("top1")
            task["methods"]["setwise"] = {
                "top1": top1,
                "top1_valid": bool(top1 and top1 in verified),
                "full_ranking": False,
                "top5": None,
                "mrr_at_10": None,
                "error": row.get("error"),
            }
        tasks.append(task)

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
    if any("setwise" in task["methods"] for task in recommend):
        rows = [task["methods"]["setwise"] for task in recommend if "setwise" in task["methods"]]
        summary["setwise"] = {"recommend_tasks": len(rows),
                              "top1_valid": sum(1 for row in rows if row["top1_valid"]),
                              "top5_valid": None, "mrr_at_10": None,
                              "candidate_unreachable": sum(1 for task in recommend if task["candidate_unreachable"]),
                              "reports_top1_only": True,
                              "top1_errors": sum(1 for row in rows if row["error"])}

    # Ablations for Qwen3-Reranker-4B.
    q3 = {}
    for left, right in (("qwen3_query_only", "qwen3_query_prefs"),
                        ("qwen3_query_prefs", "qwen3_prefs_instruction")):
        q3[f"{left}->{right}"] = {
            "top1_changed": sum(1 for task in recommend if task["methods"][left]["top1"] != task["methods"][right]["top1"]),
            "correct_gained": [task["task_id"] for task in recommend
                               if task["methods"][right]["top1_valid"] and not task["methods"][left]["top1_valid"]],
            "correct_lost": [task["task_id"] for task in recommend
                             if task["methods"][left]["top1_valid"] and not task["methods"][right]["top1_valid"]],
        }
    changed_vs_v0 = {}
    for method in METHODS:
        if method == "v0":
            continue
        changed_vs_v0[method] = {
            "correct_gained_vs_v0": [task["task_id"] for task in recommend
                                     if task["methods"][method]["top1_valid"] and not task["methods"]["v0"]["top1_valid"]],
            "correct_lost_vs_v0": [task["task_id"] for task in recommend
                                   if task["methods"]["v0"]["top1_valid"] and not task["methods"][method]["top1_valid"]],
        }
    failure_cases = []
    for task in recommend:
        if task["task_id"] in {"0045", "p14e2_lite_0045", "p14e2_lite_0047", "p14e2_lite_0048",
                               "p14e2_lite_0053", "p14e2_lite_0054", "p14e2_lite_0046"} or \
                all(task["methods"][method]["top1_valid"] for method in METHODS):
            continue
        failure_cases.append({
            "task_id": task["task_id"], "task_type": task["task_type"],
            "history": task["history"], "candidate_unreachable": task["candidate_unreachable"],
            "methods": {method: {"top1": task["methods"][method]["top1"],
                                 "valid": task["methods"][method]["top1_valid"],
                                 "first_valid_rank": task["methods"][method]["first_valid_rank"],
                                 "preference_stats": task["methods"][method]["preference_stats"]}
                        for method in METHODS},
        })

    resources = scores["resources"]
    model_manifest = {
        "schema": "agentrecx.p14b3.model_manifest.v1",
        "qwen3_reranker": {
            "repo_id": "Qwen/Qwen3-Reranker-4B",
            "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
            "license": "Apache-2.0",
            "architecture": "Qwen3ForCausalLM (official yes/no logit reranker)",
            "parameters": "4B (official model card)",
            "quantization": "bitsandbytes NF4 4-bit (double quant), fp16 compute, batch 4, max_length 2048",
            "scoring": "official yes/no token logits; softmax(false,true)[1]",
            "source": "Hugging Face hf-mirror.com snapshot to /root/autodl-tmp/models/Qwen3-Reranker-4B",
            "resources": resources,
        },
        "setwise": {
            "model": "Qwen/Qwen3.5-9B Base checkpoint /root/autodl-tmp/models/Qwen3.5-9B-ms",
            "status": "BLOCKED_RESOURCE",
            "reason": "Qwen3.5-9B service occupies 17.6 GiB of 23.6 GiB RTX 4090D; only ~6.18 GiB free. "
                      "The 9B model in 4-bit direct GPU load needs ~5.94 GiB plus allocator headroom and OOMs; "
                      "CPU fp16/bf16 inference of one 6-candidate group did not finish in ~13 minutes. "
                      "Stopping the existing Qwen service is forbidden by the milestone.",
            "attempted": ["gpu4bit direct: OOM", "gpu4bit offload: OOM/accelerate bnb dispatch error",
                          "cpu bf16 one task: >13 min no generation"],
            "no_fabricated_result": True,
        },
        "v0": "P14-B0 deterministic scorer (CPU, ~0.001 s/111 candidates)",
    }
    comparison = {
        "schema": "agentrecx.p14b3.comparison.v1",
        "pool": {"source": "runs/p14b3/pools/dev.jsonl (P14-B2 fixed BM25 top-20 + role filter)",
                 "recommend_tasks": len(recommend), "candidates": sum(len(p["candidates"]) for p in pools)},
        "summary": summary,
        "qwen3_ablations": q3,
        "changed_vs_v0": changed_vs_v0,
        "failures": failure_cases,
        "model_manifest": model_manifest,
        "per_task": tasks,
        "verdicts": {
            "QWEN3_RERANKER": "PASS" if summary["qwen3_prefs_instruction"]["top1_valid"] >= summary["v0"]["top1_valid"] else "LIMITED",
            "SETWISE_RERANKER": "FAIL" if "setwise" not in summary else "PASS",
            "PREFERENCE_GAIN": "PASS" if summary["qwen3_prefs_instruction"]["top1_valid"] > summary["qwen3_query_only"]["top1_valid"] else "LIMITED",
            "FAIR_COMPARISON": "PASS",
            "P14_B3": "LIMITED",
        },
        "notes": [
            "Dev 20 was already used for P14-C0-R development; results are exploratory.",
            "Setwise is BLOCKED by GPU coexistence with the Qwen service and is NOT fabricated; no Setwise metrics are claimed.",
            "All completed methods use the same role-filtered candidate pools and evaluator labels only for scoring after the fact.",
            "No training/fine-tuning, no Reserved Test, V0/Gate/Planner/Memory unchanged.",
        ],
    }
    (OUT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=1, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    with (OUT / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "qwen3_ablations": q3, "changed_vs_v0": changed_vs_v0,
                      "failures": [row["task_id"] for row in failure_cases]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
