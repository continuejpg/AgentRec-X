"""P14-B3-Eval100: BM25 vs V0 vs frozen Qwen3-Reranker-4B + Preference Instruction.

Recommendation metrics (80 tasks) are separate from non-recommend Decision Gate checks (20).
The 100 tasks were built and labeled before any model run; labels are used only here.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14b1.diagnose import verified_ids
from experiments.p14c0 import run_dev as rd
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14b3_eval100"
OUT = REPO / "runs/p14b3_eval100"
METHODS = ("bm25", "v0", "qwen3_query_only", "qwen3_query_prefs", "qwen3_prefs_instruction")
FORM_FACTORS = {"over_ear", "on_ear", "in_ear"}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank(pool: dict[str, Any], method: str, scores: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates = list(pool["candidates"])
    if method == "bm25":
        return sorted(candidates, key=lambda c: (c["bm25_rank"], c["item_id"]))
    if method == "v0":
        return sorted(candidates, key=lambda c: (c.get("v0_rank") if c.get("v0_rank") is not None else 10**9,
                                                 c["bm25_rank"], c["item_id"]))
    key = {"qwen3_query_only": "query_only", "qwen3_query_prefs": "query_prefs",
           "qwen3_prefs_instruction": "prefs_instruction"}[method]
    values = scores["scores"][key][pool["task_id"]]
    return sorted(candidates, key=lambda c: (-float(values.get(c["item_id"], float("-inf"))),
                                             c["bm25_rank"], c["item_id"]))


def pref_stats(product, preferences):
    matched = violated = unknown = 0
    form = getattr(product, "form_factor", None)
    attrs = set(product.attributes or {}) if product else set()
    for pref in preferences:
        value = str(pref.get("value"))
        negative = pref.get("polarity") == "negative"
        if value in FORM_FACTORS:
            observed, contradicts = form == value, form is not None and form != value
        else:
            observed, contradicts = value in attrs, False
        if observed and negative:
            violated += 1
        elif observed:
            matched += 1
        elif contradicts and not negative:
            violated += 1
        elif not contradicts:
            unknown += 1
    return {"matched": matched, "violated": violated, "unknown": unknown}


def hard_violations(product, hard):
    form = getattr(product, "form_factor", None)
    attrs = set(product.attributes or {}) if product else set()
    return sum(int(form is not None and form != str(req.get("value")))
               for req in hard if str(req.get("value")) in FORM_FACTORS) + \
           sum(0 for req in hard if str(req.get("value")) not in FORM_FACTORS)


def metrics_for(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if method not in rows[0]:
        return {}
    values = [row[method] for row in rows]
    return {
        "top1_valid": sum(1 for value in values if value["top1_valid"]),
        "top5_valid": sum(1 for value in values if value["top5_valid"]),
        "mrr_at_10": round(sum(value["mrr_at_10"] for value in values) / len(values), 6),
        "preference_matched_top1": sum(value["preference_stats"]["matched"] for value in values),
        "preference_violated_top1": sum(value["preference_stats"]["violated"] for value in values),
        "preference_unknown_top1": sum(value["preference_stats"]["unknown"] for value in values),
        "hard_constraint_violations": sum(value["hard_constraint_violations"] for value in values),
        "product_type_violations": sum(1 for value in values if value["product_type_violation"]),
    }


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    evaluator = {row["task_id"]: row for row in load_jsonl(DATA / "eval100.evaluator.jsonl")}
    agent = {row["task_id"]: row for row in load_jsonl(DATA / "eval100.agent.jsonl")}
    pools = {row["task_id"]: row for row in load_jsonl(OUT / "pools/eval100.jsonl")}
    scores = json.loads((OUT / "scores_eval100.json").read_text(encoding="utf-8"))
    recommend_ids = [task_id for task_id, row in evaluator.items() if row["expected_decision"] == "recommend"]
    nonrec_ids = [task_id for task_id, row in evaluator.items() if row["expected_decision"] != "recommend"]
    assert len(recommend_ids) == 80 and len(nonrec_ids) == 20

    task_rows = []
    for task_id in recommend_ids:
        row = {"task_id": task_id, "task_type": evaluator[task_id]["task_type"],
               "product_type": evaluator[task_id]["product_type_pool"],
               "pool_size": len(pools[task_id]["candidates"]),
               "candidate_unreachable": False, "methods": {}}
        verified = verified_ids(evaluator[task_id], catalogue)
        row["verified_total"] = len(verified)
        row["candidate_unreachable"] = bool(verified) and not (
            verified & {candidate["item_id"] for candidate in pools[task_id]["candidates"]})
        for method in METHODS:
            ranked = rank(pools[task_id], method, scores)
            ids = [candidate["item_id"] for candidate in ranked]
            valid_ranks = [index for index, item_id in enumerate(ids, start=1) if item_id in verified]
            product = catalogue.get(ids[0]) if ids else None
            row["methods"][method] = {
                "ranked_ids": ids, "top1": ids[0] if ids else None, "top5": ids[:5],
                "top1_valid": bool(ids) and ids[0] in verified,
                "top5_valid": bool(valid_ranks and valid_ranks[0] <= 5),
                "mrr_at_10": round(1.0 / valid_ranks[0], 6) if valid_ranks and valid_ranks[0] <= 10 else 0.0,
                "first_valid_rank": valid_ranks[0] if valid_ranks else None,
                "preference_stats": pref_stats(product, pools[task_id]["active_preferences"]),
                "hard_constraint_violations": hard_violations(product, pools[task_id]["hard_requirements"]),
                "product_type_violation": bool(product and pools[task_id]["target"]["product_type"]
                                               and product.product_type != pools[task_id]["target"]["product_type"]),
                "top1_title": product.title if product else None,
                "top1_product_type": product.product_type if product else None,
            }
        task_rows.append(row)

    summary_all = {method: metrics_for([row["methods"] for row in task_rows], method) for method in METHODS}
    reachable_rows = [row for row in task_rows if not row["candidate_unreachable"]]
    summary = {method: metrics_for([row["methods"] for row in reachable_rows], method) for method in METHODS}
    summary_v0 = summary["v0"]
    instruction = summary["qwen3_prefs_instruction"]

    # Category and scenario breakdown.
    by_type: dict[str, Any] = defaultdict(lambda: {"tasks": 0, "v0_top1": 0, "qwen3_top1": 0})
    for row in reachable_rows:
        group = by_type[row["task_type"]]
        group["tasks"] += 1
        group["v0_top1"] += int(row["methods"]["v0"]["top1_valid"])
        group["qwen3_top1"] += int(row["methods"]["qwen3_prefs_instruction"]["top1_valid"])
    by_category = {}
    for task_type, group in by_type.items():
        group["v0_top1_rate"] = round(group["v0_top1"] / group["tasks"], 3)
        group["qwen3_top1_rate"] = round(group["qwen3_top1"] / group["tasks"], 3)
        by_category[task_type] = group

    # Same-query persona pairs.
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        pair_groups[evaluator[row["task_id"]]["family_id"]].append(row)
    pairs = []
    for family, rows in pair_groups.items():
        if len(rows) != 2:
            continue
        pairs.append({
            "family_id": family,
            "task_ids": [row["task_id"] for row in rows],
            "v0_top1_changed": rows[0]["methods"]["v0"]["top1"] != rows[1]["methods"]["v0"]["top1"],
            "qwen3_top1_changed": (rows[0]["methods"]["qwen3_prefs_instruction"]["top1"] !=
                                    rows[1]["methods"]["qwen3_prefs_instruction"]["top1"]),
            "v0_correct": [row["methods"]["v0"]["top1_valid"] for row in rows],
            "qwen3_correct": [row["methods"]["qwen3_prefs_instruction"]["top1_valid"] for row in rows],
            "preferences": [pools[row["task_id"]]["preference_summary"] for row in rows],
        })

    # Gains/losses vs V0.
    gains = [row["task_id"] for row in reachable_rows
             if row["methods"]["qwen3_prefs_instruction"]["top1_valid"] and not row["methods"]["v0"]["top1_valid"]]
    losses = [row["task_id"] for row in reachable_rows
              if row["methods"]["v0"]["top1_valid"] and not row["methods"]["qwen3_prefs_instruction"]["top1_valid"]]
    unreachable = [row["task_id"] for row in task_rows if row["candidate_unreachable"]]

    # Non-recommend tasks: Decision Gate behavior via the frozen Rule executor (no Top-1 denominator).
    nonrec_rows = []
    gate_correct = 0
    for task_id in nonrec_ids:
        task = agent[task_id]
        record = rd._run_one(catalogue, task, planner_kind="rule")
        expected = evaluator[task_id]["expected_decision"]
        status = record["terminal"]["status"]
        if expected == "clarification_required":
            correct = status == "awaiting_clarification"
        elif expected == "no_verified_match":
            correct = status == "no_verified_candidate_in_scope"
        else:  # insufficient_evidence
            correct = status in {"no_verified_candidate_in_scope", "budget_exhausted"}
        gate_correct += int(correct)
        nonrec_rows.append({"task_id": task_id, "task_type": evaluator[task_id]["task_type"],
                            "expected": expected, "gate_status": status, "correct": correct,
                            "ask_question": record["terminal"].get("ask_question"),
                            "error": record.get("error")})
    gate_payload = {
        "schema": "agentrecx.p14b3eval100.nonrecommend_gate.v1",
        "evaluated_with": "frozen P14-C0-R Rule executor + Product Type/Role + Decision Gate; not Qwen planner",
        "tasks": nonrec_rows,
        "counts": dict(Counter(row["task_type"] for row in nonrec_rows)),
        "correct": gate_correct, "total": len(nonrec_rows),
        "note": "these 20 tasks are excluded from Top-1 denominators",
    }
    (OUT / "nonrecommend_gate.json").write_text(json.dumps(gate_payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-review queue: prefer negative-preference, update/withdraw, and V0/Qwen disagreement cases.
    prioritized = []
    for row in task_rows:
        ev = evaluator[row["task_id"]]
        reasons = []
        if any(str(pref.get("polarity")) == "negative" for pref in ev["effective_preferences"]):
            reasons.append("explicit negative preference")
        if ev["task_type"] in {"preference_update", "preference_withdrawal"}:
            reasons.append(ev["task_type"])
        if ev["withdrawn_preferences"]:
            reasons.append("withdrawn preference")
        if row["methods"]["v0"]["top1"] != row["methods"]["qwen3_prefs_instruction"]["top1"]:
            reasons.append("V0 vs Qwen3 top1 disagreement")
        if row["candidate_unreachable"]:
            reasons.append("candidate unreachable")
        if reasons:
            prioritized.append((len(reasons), row["task_id"], reasons))
    prioritized.sort(reverse=True)
    review = []
    for _score, task_id, reasons in prioritized[:27]:
        row = next(row for row in task_rows if row["task_id"] == task_id)
        ev = evaluator[task_id]
        review.append({
            "task_id": task_id, "task_type": ev["task_type"], "reasons": reasons,
            "review_status": "pending_human_review", "human_verified": False,
            "query": agent[task_id]["current_query"],
            "active_preferences": pools[task_id]["active_preferences"],
            "withdrawn_preferences": ev["withdrawn_preferences"],
            "hard_requirements": pools[task_id]["hard_requirements"],
            "pool_size": row["pool_size"], "candidate_unreachable": row["candidate_unreachable"],
            "v0_top1": row["methods"]["v0"]["top1"], "v0_top1_title": row["methods"]["v0"]["top1_title"],
            "qwen3_top1": row["methods"]["qwen3_prefs_instruction"]["top1"],
            "qwen3_top1_title": row["methods"]["qwen3_prefs_instruction"]["top1_title"],
            "qwen3_preference_stats": row["methods"]["qwen3_prefs_instruction"]["preference_stats"],
            "v0_preference_stats": row["methods"]["v0"]["preference_stats"],
            "verified_candidate_count": ev["labels"]["verified_candidate_count"],
            "label_basis": ev["construction_metadata"]["label_basis"],
        })
    (OUT / "review_checklist_divergence.json").write_text(json.dumps(
        {"schema": "agentrecx.p14b3eval100.review.v1", "human_verified": False,
         "review_status": "pending_human_review", "cases": review}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")

    qwen_resources = scores["resources"]
    model_manifest = {
        "schema": "agentrecx.p14b3eval100.model_manifest.v1",
        "qwen3_reranker": {
            "repo_id": "Qwen/Qwen3-Reranker-4B", "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
            "license": "Apache-2.0", "quantization": "bitsandbytes NF4 4-bit", "dtype": "fp16 compute",
            "max_length": 2048, "batch_size": 4, "scoring": "official yes/no token logits",
            "resources": qwen_resources,
            "instruction": "P14-B3 frozen preference-aware instruction; same as p14b3",
        },
        "v0": {"name": "Preference Scorer V0", "changed": False},
        "data": {"root": "data/p14b3_eval100", "validation": "runs/p14b3_eval100/validation.json",
                 "human_verified": False, "review_status": "pending_human_review"},
    }
    comparison = {
        "schema": "agentrecx.p14b3eval100.comparison.v1",
        "data_validation": json.loads((OUT / "validation.json").read_text(encoding="utf-8")),
        "pool": {"definition": "BM25 top-20 + Product Type/Role MATCH/NOT_REQUIRED",
                 "recommend_tasks": 80, "non_recommend_tasks": 20,
                 "candidates": sum(len(pool["candidates"]) for pool in pools.values()),
                 "reachable_recommend_tasks": len(reachable_rows),
                 "candidate_unreachable_tasks": unreachable},
        "summary": summary, "summary_all_80": summary_all,
        "reachable_tasks": len(reachable_rows), "candidate_unreachable_tasks": unreachable,
        "category_breakdown": by_category,
        "same_query_pairs": pairs,
        "qwen3_gained_vs_v0": gains, "qwen3_lost_vs_v0": losses,
        "nonrecommend_gate": gate_payload,
        "review_checklist": "runs/p14b3_eval100/review_checklist_divergence.json",
        "model_manifest": model_manifest,
        "per_task": task_rows,
        "preliminary_decision": None,  # filled by report after inspection
        "notes": ["Eval100 tasks are new, evaluation-only, and not used for model or label development.",
                  "Labels are independent evidence labels and remain provisional; no human verification.",
                  "All completed methods share the same fixed role-filtered candidate pools.",
                  "Setwise was not part of this evaluation.",
                  "Decision Gate results for non-recommend tasks are separate from ranking metrics."],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=1, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    with (OUT / "per_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in task_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "gains": gains, "losses": losses,
                      "unreachable": unreachable, "gate": {"correct": gate_correct, "total": len(nonrec_rows)},
                      "pairs": pairs}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
