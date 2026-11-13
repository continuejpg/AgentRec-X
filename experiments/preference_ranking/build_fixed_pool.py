"""Build a fixed-pool candidate-ranking benchmark from Original BGE and real tasks.

The label boundary is deliberately narrow:

* candidates are the fixed Original BGE Top-20 for each task;
* active preferences come from the legal Memory reader, never from evaluator labels;
* hard requirements come from the task query, not from labels;
* evaluator-only acceptable ids are joined after ranking for measurement only;
* every unlabelled candidate is recorded as `unjudged`, not as a negative.

This is a local CPU benchmark builder.  It does not change retrieval, the scorer, the
Gate, or the Planner.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.p14b1.diagnose import candidate_rows, verified_ids
from experiments.p14c0.electronics_executor import prepare_hard
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.product_role import assess_candidate_role, infer_product_target
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "runs/p20_agent_eval/task_suite.json"
TRACE = REPO / "runs/advanced_retrieval_v2/task_arm_original_bge.json"
OUT = REPO / "runs/preference_ranking_v1"
NAMESPACE = "amazon_electronics_c4"
TOP_K = 20


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pref_summary(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": item.get("memory_id"),
            "kind": item.get("kind"),
            "value": item.get("value"),
            "polarity": item.get("polarity"),
            "source_turn_id": item.get("source_turn_id"),
        }
        for item in preferences
    ]


def _aligned(product: Any, preferences: list[dict[str, Any]]) -> dict[str, Any]:
    attributes = {str(name).casefold() for name in (product.attributes or {})} if product else set()
    form = getattr(product, "form_factor", None) if product else None
    matched = []
    violated = []
    for preference in preferences:
        value = str(preference.get("value") or "").casefold()
        polarity = str(preference.get("polarity") or "positive").casefold()
        if not value:
            continue
        observed = (form == value) if value in {"over_ear", "on_ear", "in_ear"} else (value in attributes)
        if observed and polarity in {"avoid", "negative", "dislike"}:
            violated.append(preference.get("value"))
        elif observed:
            matched.append(preference.get("value"))
    return {"matched": matched, "violated": violated}


def _rank_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [row["product_ref"]["item_id"] for row in rows]


def _hit(ranked_ids: list[str], acceptable: set[str]) -> dict[str, Any]:
    first = next((index + 1 for index, item_id in enumerate(ranked_ids) if item_id in acceptable), None)
    return {
        "hit@1": bool(ranked_ids and ranked_ids[0] in acceptable),
        "hit@5": bool(any(item_id in acceptable for item_id in ranked_ids[:5])),
        "hit@10": bool(any(item_id in acceptable for item_id in ranked_ids[:10])),
        "first_acceptable_rank": first,
        "reciprocal_rank": (1.0 / first) if first else 0.0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in suite["tasks"]}
    traces = {record["task_id"]: record for record in json.loads(TRACE.read_text(encoding="utf-8"))["tasks"]}
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")

    dataset: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}

    for task_id in sorted(tasks):
        task = tasks[task_id]
        trace = traces.get(task_id) or {}
        query = task["current_query"]
        preferences, history = _memory_read(catalogue, task)
        hard = prepare_hard(query)
        target = infer_product_target(query)
        # The fixed pool is the first real Original-BGE retrieval result in the
        # P20 trace. This is the candidate order the original Agent actually saw.
        pool_ids: list[str] = []
        for step in trace.get("steps") or []:
            observation = step.get("observation_after") or {}
            ids = list(observation.get("candidate_ids") or ())
            if ids and not pool_ids:
                pool_ids = [str(item) for item in ids[:TOP_K]]
                break
        search_scores: dict[str, float] = {
            item_id: float(max(0.0, 1.0 - index / max(1, TOP_K)))
            for index, item_id in enumerate(pool_ids, start=1)
        }

        pool_task = {
            "task_id": task_id,
            "current_query": query,
            "candidate_ids": pool_ids,
            "search_scores": search_scores,
        }
        scorer_rows = candidate_rows(pool_task, catalogue, PreferenceAwareScorer(),
                                     preferences, hard)
        # candidate_rows sorts with hard tiers; retain deterministic order and add
        # the product-view fields needed by the structural hard-filter arm.
        for row in scorer_rows:
            item_id = row["product_ref"]["item_id"]
            product = catalogue.get(item_id)
            row["form_factor"] = product.form_factor if product else None
            row["attributes"] = dict(product.attributes or {}) if product else {}
            if product is not None:
                row.update(assess_candidate_role(
                    product_type=product.product_type, title=product.title, target=target,
                ).as_dict())
            else:
                row.update({"role_status": "UNKNOWN", "role_evidence": ["product not in catalogue"]})

        # A: fixed Original BGE order.
        a_ids = list(pool_ids)
        # B: existing PreferenceAwareScorer order.
        b_rows = list(scorer_rows)
        b_ids = _rank_rows(b_rows)
        # C: structural hard filter + role compatibility, then existing scorer order.
        c_rows = [
            row for row in scorer_rows
            if int(row.get("hard_tier") or 0) == 0
            and str(row.get("role_status") or "NOT_REQUIRED") in {"MATCH", "NOT_REQUIRED"}
        ]
        c_ids = _rank_rows(c_rows)

        evaluator = task.get("scored_evaluator") or {}
        acceptable = set()
        if evaluator and evaluator.get("expected_decision") == "recommend":
            acceptable = verified_ids(evaluator, catalogue)
        gate_selected = trace.get("selected_product_id")
        acceptable_in_pool = [item_id for item_id in pool_ids if item_id in acceptable]

        # Labelled only when the evaluator supplies acceptable ids; all other pool rows
        # remain unjudged.  Labels never enter the scorer/preference inputs.
        candidates: list[dict[str, Any]] = []
        for index, item_id in enumerate(pool_ids, start=1):
            product = catalogue.get(item_id)
            scorer_row = next((row for row in scorer_rows if row["product_ref"]["item_id"] == item_id), None)
            item = {
                "item_id": item_id,
                "original_bge_rank": index,
                "original_bge_score": search_scores.get(item_id),
                "title": product.title if product else None,
                "text": product.text if product else None,
                "attributes": dict(product.attributes or {}) if product else {},
                "form_factor": product.form_factor if product else None,
                "scorer": scorer_row,
                "label": "acceptable" if item_id in acceptable else "unjudged",
                "label_source": task.get("label_source"),
            }
            candidates.append(item)

        row = {
            "schema": "agentrecx.preference_ranking.fixed_pool.v1",
            "task_id": task_id,
            "split": task.get("split"),
            "category": task.get("category"),
            "expected_decision": task.get("expected"),
            "query": query,
            "historical_memory": _pref_summary(preferences),
            "memory_fixture": task.get("memory_fixture"),
            "hard_requirements": hard,
            "candidate_pool": candidates,
            "candidate_pool_size": len(candidates),
            "acceptable_ids_in_pool": acceptable_in_pool,
            "acceptable_total_catalog_scope": len(acceptable),
            "gate_selected_id": gate_selected,
            "label_source": task.get("label_source"),
            "label_status": "provisional_automated_verified_not_human_gold",
            "rankings": {
                "A_original_bge": a_ids,
                "B_existing_scorer": b_ids,
                "C_hard_filtered_scorer": c_ids,
            },
            "top1": {"A_original_bge": a_ids[0] if a_ids else None,
                     "B_existing_scorer": b_ids[0] if b_ids else None,
                     "C_hard_filtered_scorer": c_ids[0] if c_ids else None,
                     "gate": gate_selected},
            "metrics": {},
        }
        if acceptable:
            for label, ids in (("A_original_bge", a_ids), ("B_existing_scorer", b_ids),
                               ("C_hard_filtered_scorer", c_ids)):
                row["metrics"][label] = _hit(ids, acceptable)
        # Preference/hard evidence for top-1 only; no evaluator label used here.
        for label, item_id in row["top1"].items():
            if not item_id:
                continue
            product = catalogue.get(item_id)
            aligned = _aligned(product, preferences)
            scorer_row = next((item for item in scorer_rows
                               if item["product_ref"]["item_id"] == item_id), None)
            row["metrics"].setdefault(label, {}).update({
                "preference_matched": aligned["matched"],
                "preference_violated": aligned["violated"],
                "hard_tier": (scorer_row or {}).get("hard_tier"),
                "hard_status": (scorer_row or {}).get("hard_status"),
                "unknown_preferences": (scorer_row or {}).get("unknown_preferences"),
                "acceptable": bool(item_id in acceptable),
            })
        dataset.append(row)
        aggregate[task_id] = {
            "split": task.get("split"),
            "acceptable_in_pool": len(acceptable_in_pool),
            "candidate_pool_size": len(candidates),
            "metrics": row["metrics"],
        }

    data_path = OUT / "ranking_dataset.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                 for row in dataset), encoding="utf-8")
    manifest = {
        "schema": "agentrecx.preference_ranking.dataset_manifest.v1",
        "dataset": str(data_path),
        "sha256": _sha(data_path),
        "tasks": len(dataset),
        "recommend_tasks": sum(row["expected_decision"] == "recommend" for row in dataset),
        "fixed_pool": f"Original BGE Top-{TOP_K}",
        "retrieval": str(TRACE),
        "retrieval_sha256": _sha(TRACE),
        "trace": str(TRACE),
        "catalogue": "amazon_reviews_2023:electronics_c4",
        "label_source": "P14-E2/P14-B3 evaluator acceptable ids plus original Qwen trace selections",
        "label_status": "provisional_automated_verified_not_human_gold",
        "unjudged_policy": "candidates not in evaluator acceptable set are unjudged, not negative",
        "heldout": "eval100 task split is retained in this dataset; no training occurs here",
        "no_p18": True,
        "variants": ["A_original_bge", "B_existing_scorer", "C_hard_filtered_scorer"],
        "preference_source": "legal Memory reader active preferences; no evaluator labels",
        "hard_source": "prepare_hard(current_query); no evaluator labels",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    (OUT / "local_metrics.json").write_text(json.dumps(aggregate, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
