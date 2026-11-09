"""P14-B2: build fixed BM25 candidate pools (agent-visible only) for Train smoke and Dev 20.

Each pool stores: current query, legal active preferences (from M9 history-state read), hard
requirements, Product Type/Role target, BM25 order, role-filtered candidates with real
catalogue title/text, and the V0 Scorer order.  No evaluator labels are written here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from experiments.p14b1.diagnose import hard_requirements
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from recommendation.preference.product_role import assess_candidate_role, infer_product_target
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p14b2"
TOPK = 20
TRAIN_SMOKE_TYPES = ("no_history_recommend", "history_influenced", "preference_withdrawal",
                     "insufficient_information", "no_verified_match")


def _load(split: str, kind: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{split}.{kind}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_pool(catalogue: ElectronicsCatalogue, task: dict[str, Any], preferences: list[dict[str, Any]]) -> dict[str, Any]:
    query = task["current_query"]
    target = infer_product_target(query)
    hard = hard_requirements(query)
    result = catalogue.search_source.search(query, limit=TOPK)
    raw = []
    rejected = []
    for rank, hit in enumerate(result.hits, start=1):
        item_id = hit.parent_asin.split(":", 1)[-1]
        product = catalogue.get(item_id)
        if product is None:
            continue
        assessment = assess_candidate_role(product_type=product.product_type, title=product.title, target=target)
        candidate = {
            "item_id": item_id,
            "bm25_rank": rank,
            "bm25_score": round(hit.score, 6),
            "product_type": product.product_type,
            "role_status": assessment.status.value,
            "role": assessment.candidate_role.value,
            "title": product.title,
            "text": product.text[:450],
            "attributes": sorted((product.attributes or {}).keys()),
        }
        if assessment.status.value in {"MATCH", "NOT_REQUIRED"}:
            raw.append(candidate)
        else:
            rejected.append(candidate)
    # V0 Scorer over the same role-filtered candidates.
    scoring_rows = [
        {
            "product_ref": {"namespace": ELECTRONICS_NAMESPACE, "item_id": candidate["item_id"]},
            "query_rank": candidate["bm25_rank"],
            "query_score": candidate["bm25_score"],
            "attributes": {name: {"evidence": "catalog_meta:text"} for name in candidate["attributes"]},
            "form_factor": None,
        }
        for candidate in raw
    ]
    scorer = PreferenceAwareScorer()
    v0 = scorer.score(
        preferences=preferences, current_query=query, candidates=scoring_rows,
        hard_requirements=hard, namespace=ELECTRONICS_NAMESPACE,
    )
    v0_by_id = {row["product_ref"]["item_id"]: row for row in v0}
    for candidate in raw:
        row = v0_by_id.get(candidate["item_id"], {})
        candidate["v0_rank"] = row.get("rank")
        candidate["v0_score"] = row.get("final_score")
        candidate["v0_matched"] = [p.get("value") for p in row.get("matched_preferences", [])]
        candidate["v0_conflicting"] = [p.get("value") for p in row.get("conflicting_preferences", [])]
        candidate["v0_unknown"] = [p.get("value") for p in row.get("unknown_preferences", [])]
    prefs_summary = None
    if preferences:
        positive = [p["value"] for p in preferences if p.get("polarity") == "positive"]
        negative = [p["value"] for p in preferences if p.get("polarity") == "negative"]
        prefs_summary = "positive preferences: " + (", ".join(positive) or "none") + \
                        "; negative preferences: " + (", ".join(negative) or "none")
    return {
        "task_id": task["task_id"],
        "split": task["split"],
        "current_query": query,
        "hard_requirements": hard,
        "active_preferences": preferences,
        "preference_summary": prefs_summary,
        "target": target.as_dict(),
        "bm25_candidates": TOPK,
        "role_filtered": len(raw),
        "role_rejected": len(rejected),
        "role_rejected_types": sorted({c["product_type"] or "unknown" for c in rejected}),
        "candidates": raw,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pools").mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load()
    started = time.perf_counter()
    for split, kind, selection in (
        ("train_smoke", "train", "smoke"),
        ("dev", "dev", "all"),
    ):
        agents = _load(kind, "agent")
        evaluator = {row["task_id"]: row for row in _load(kind, "evaluator")}
        if selection == "smoke":
            tasks = []
            for task_type in TRAIN_SMOKE_TYPES:
                task = next((row for row in agents if evaluator[row["task_id"]]["task_type"] == task_type), None)
                if task:
                    tasks.append(task)
        else:
            tasks = agents
        rows = []
        for task in tasks:
            preferences, _history = _memory_read(catalogue, task)
            rows.append(build_pool(catalogue, task, preferences))
        path = OUT / "pools" / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(split, "tasks", len(rows), "empty_pools", sum(1 for row in rows if not row["candidates"]))
    manifest = {
        "schema": "agentrecx.p14b2.input_manifest.v1",
        "catalogue": {"namespace": ELECTRONICS_NAMESPACE, "stats": catalogue.stats},
        "top_k": TOPK,
        "pools": ["runs/p14b2/pools/train_smoke.jsonl", "runs/p14b2/pools/dev.jsonl"],
        "selection": "fixed BM25 top-20, role-filtered; V0 ordering inside same pool",
        "labels_in_pools": False,
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
