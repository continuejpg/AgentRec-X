"""P14-D1 candidate pools and real-history summaries.

Protocol A: natural BM25 top-20 (target may be absent/unreachable) and top-100 reachability.
Protocol B: target-included 1-in-5 diagnostic (target + 4 deterministic BM25 distractors).
No target identity is marked in model input; target IDs live only in dev_targets.jsonl.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
OUT = REPO / "runs/p14d1"
SEED = 20261109


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def history_summary(history: list[dict[str, Any]]) -> str:
    """Fixed, factual history summary; no inferred preferences or target features."""
    recent = history[-12:]
    lines = []
    for item in recent:
        title = (item.get("title") or "").strip()
        label = title[:150] if title else f"item {item['item_id']}"
        verified = "verified" if item.get("verified_purchase") else "unverified"
        lines.append(f"- [{item.get('category', 'Unknown')}] {label} ({verified})")
    return "\n".join(lines)


def history_proxy_preferences(history: list[dict[str, Any]], catalogue: ElectronicsCatalogue) -> list[dict[str, Any]]:
    """Weak interest proxy from observable catalogue attributes of pre-target history items.

    This is explicitly a weak proxy: observed attributes only, positive only, no negatives,
    no target item, and no claim that the user stated a preference.
    """
    counts: dict[tuple[str, str], int] = {}
    for item in history:
        product = catalogue.get(item["item_id"])
        if product is None:
            continue
        for attribute in product.attributes or {}:
            key = ("attribute", attribute)
            counts[key] = counts.get(key, 0) + 1
        if product.form_factor:
            key = ("form_factor", product.form_factor)
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    preferences = []
    for (kind, value), count in ranked[:5]:
        if count < 1:
            continue
        preferences.append({"kind": kind, "value": value, "polarity": "positive",
                            "evidence_count": count, "source": "history_proxy_observed_attributes"})
    return preferences


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    tasks = load_jsonl(DATA / "dev_tasks.jsonl")
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pools").mkdir(parents=True, exist_ok=True)
    natural = []
    target5 = []
    stats = []
    for index, task in enumerate(tasks):
        target = targets[task["task_id"]]
        target_id = target["target_parent_asin"]
        result = catalogue.search_source.search(task["query_original"], limit=100)
        hits = []
        for rank, hit in enumerate(result.hits, start=1):
            item_id = hit.parent_asin.split(":", 1)[-1]
            product = catalogue.get(item_id)
            if product is None:
                continue
            hits.append({"item_id": item_id, "bm25_rank": rank, "bm25_score": round(hit.score, 6),
                         "title": product.title, "text": product.text[:700]})
        target_rank100 = next((row["bm25_rank"] for row in hits if row["item_id"] == target_id), None)
        top20 = hits[:20]
        target_rank20 = next((row["bm25_rank"] for row in top20 if row["item_id"] == target_id), None)
        summary = history_summary(task["history"])
        proxy = history_proxy_preferences(task["history"], catalogue)
        common = {
            "task_id": task["task_id"], "query_original": task["query_original"],
            "query_rewritten": task.get("query_rewritten"),
            "history_summary": summary,
            "history_proxy_preferences": proxy,
            "history_length": len(task["history"]),
            "history_in_catalogue_count": sum(1 for item in task["history"] if item.get("in_catalogue")),
            "history_verified_count": sum(1 for item in task["history"] if item.get("verified_purchase")),
            "user_id": task["user_id"], "qid": task["qid"],
        }
        natural.append({**common, "candidates": top20})
        # Protocol B: target-included 1-in-5 diagnostic for every catalogue target.
        # Target need not be naturally retrieved; target row is added from the catalogue, while
        # four distractors are sampled deterministically from BM25 top-100.  The target is not
        # marked in the model input; target identity is kept only in dev_targets.jsonl.
        if target["target_in_catalogue"]:
            target_product = catalogue.get(target_id)
            target_row = next((row for row in hits if row["item_id"] == target_id), None)
            if target_row is None:
                target_row = {"item_id": target_id, "bm25_rank": 10**6, "bm25_score": 0.0,
                              "title": target_product.title, "text": target_product.text[:700]}
            distractors = [row for row in hits if row["item_id"] != target_id]
            rng = random.Random(SEED + index)
            chosen = rng.sample(distractors, min(4, len(distractors)))
            candidates = [target_row] + chosen
            # stable fixed order for the prompt/input; evaluation uses actual IDs
            candidates.sort(key=lambda row: (row["bm25_rank"], row["item_id"]))
            target5.append({**common, "candidates": candidates})
        stats.append({
            "task_id": task["task_id"], "target_in_catalogue": target["target_in_catalogue"],
            "target_rank_bm25_top100": target_rank100, "target_rank_bm25_top20": target_rank20,
            "history_length": len(task["history"]),
            "history_in_catalogue": common["history_in_catalogue_count"],
            "history_verified": common["history_verified_count"],
            "history_proxy_preferences": len(proxy),
        })
    (OUT / "pools/natural.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                                    for row in natural), encoding="utf-8")
    (OUT / "pools/target5.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                                     for row in target5), encoding="utf-8")
    manifest = {
        "schema": "agentrecx.p14d1.candidates.v1",
        "aligned_tasks": len(tasks),
        "target_in_catalogue": sum(1 for row in stats if row["target_in_catalogue"]),
        "target_in_bm25_top20": sum(1 for row in stats if row["target_rank_bm25_top20"] is not None),
        "target_in_bm25_top100": sum(1 for row in stats if row["target_rank_bm25_top100"] is not None),
        "target5_tasks": len(target5),
        "history_tasks": sum(1 for row in stats if row["history_length"] > 0),
        "history_items_in_catalogue": sum(row["history_in_catalogue"] for row in stats),
        "history_task_metadata_coverage": sum(1 for row in stats if row["history_in_catalogue"] > 0),
        "tasks_with_proxy_preferences": sum(1 for row in stats if row["history_proxy_preferences"] > 0),
        "protocol_a": "natural BM25 top-20; target not injected",
        "protocol_b": "target-included 1-in-5 diagnostic; 4 distractors sampled deterministically from BM25 top-100",
        "labels_in_pools": False,
        "per_task": stats,
    }
    (OUT / "candidate_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "per_task"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
