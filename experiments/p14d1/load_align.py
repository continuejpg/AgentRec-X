"""P14-D1 data adapter: align MemRerank Electronics Dev with P13-R tasks/catalogue.

Reads the released MemRerank dev rows + query dictionary + P13-R target timestamps/history,
verifies identity/query/timestamp/history constraints, and writes an evaluation-only view.
Target product identity is kept in a separate evaluator file; it is never part of model input.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/p14d1/raw"
OUT = REPO / "data/p14d1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    hf_dev = load_jsonl(RAW / "dev.jsonl")
    p13r = [row for row in load_jsonl(REPO / "data/p13r/electronics/tasks_raw.jsonl") if row["split"] == "dev"]
    query_map = json.loads((RAW / "amazon_c4_user_query_dict_o3-mini_rewritten_v2.json").read_text(encoding="utf-8"))
    hf_by_key = {(row["user_id"], int(row["query"]), row["pos_product"]): row for row in hf_dev}
    p13r_by_key = {(row["user_id"], int(row["qid"]), row["target_parent_asin"]): row for row in p13r}

    aligned = []
    targets = []
    stats = Counter()
    issues = []
    for key, hf_row in hf_by_key.items():
        stats["hf_dev_rows"] += 1
        task = p13r_by_key.get(key)
        if task is None:
            stats["missing_p13r_timestamp_row"] += 1
            issues.append({"key": key, "issue": "no P13-R timestamp row"})
            continue
        task_id = f"p14d1_{len(aligned) + 1:04d}"
        category = str(hf_row["pos_product_category"])
        qkey = f"{key[0]}-{key[1]}-{key[2]}-{category}"
        mapping = query_map.get(qkey)
        original = str((mapping or {}).get("amazon-c4") or task.get("query") or "").strip()
        rewritten = str((mapping or {}).get("o3-mini") or "").strip() or None
        if not original:
            stats["missing_original_query"] += 1
            issues.append({"task_id": task_id, "issue": "missing original query"})
            continue
        if task.get("query") and original != task.get("query"):
            stats["p13r_query_differs_from_original"] += 1
        if rewritten:
            stats["rewritten_available"] += 1
        target_timestamp = int(task.get("target_timestamp_ms") or 0)
        if target_timestamp <= 0:
            stats["missing_target_timestamp"] += 1
            issues.append({"task_id": task_id, "issue": "missing target timestamp"})
            continue
        history = []
        for item in task.get("history") or []:
            item_id = str(item.get("item_id") or "")
            timestamp = int(item.get("timestamp_ms") or 0)
            if not item_id or timestamp <= 0:
                stats["history_invalid_item"] += 1
                continue
            if item_id == key[2]:
                stats["history_contains_target"] += 1
                continue
            if timestamp >= target_timestamp:
                stats["history_future_or_target_time"] += 1
                continue
            product = catalogue.get(item_id)
            history.append({
                "item_id": item_id,
                "category": str(item.get("category") or "Unknown"),
                "timestamp_ms": timestamp,
                "verified_purchase": bool(item.get("verified")),
                "in_catalogue": product is not None,
                "title": product.title if product else str(item.get("text") or "")[:160] or None,
            })
            stats["history_items"] += 1
            if product is not None:
                stats["history_items_in_catalogue"] += 1
            if item.get("verified"):
                stats["history_verified_items"] += 1
        history.sort(key=lambda row: row["timestamp_ms"])
        if not history:
            stats["no_valid_history"] += 1
        # Check HF grouped history equivalence at the item-id set level.
        hf_ids = {item_id for items in (hf_row.get("grouped_purchase_history") or {}).values() for item_id, *_ in items}
        p13r_ids = {str(item.get("item_id")) for item in task.get("history") or []}
        if hf_ids != p13r_ids:
            stats["hf_p13r_history_id_mismatch"] += 1
        target_product = catalogue.get(key[2])
        stats["target_in_catalogue"] += int(target_product is not None)
        if target_product is None:
            stats["target_missing_catalogue"] += 1
        aligned.append({
            "task_id": task_id,
            "split": "dev",
            "user_id": key[0],
            "qid": key[1],
            "query_original": original,
            "query_rewritten": rewritten,
            "target_category": category,
            "target_timestamp_ms": target_timestamp,
            "history": history,
            "history_length": len(history),
            "history_in_catalogue": sum(1 for item in history if item["in_catalogue"]),
            "history_verified": sum(1 for item in history if item["verified_purchase"]),
            "target_in_catalogue": target_product is not None,
            "target_text": target_product.text if target_product else None,
        })
        targets.append({
            "task_id": task_id,
            "user_id": key[0],
            "qid": key[1],
            "target_parent_asin": key[2],
            "target_category": category,
            "target_timestamp_ms": target_timestamp,
            "target_in_catalogue": target_product is not None,
            "target_title": target_product.title if target_product else None,
            "target_text": target_product.text if target_product else None,
            "label_scope": "one observed positive; all other products unjudged",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                                for row in aligned), encoding="utf-8")
    (OUT / "dev_targets.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                                    for row in targets), encoding="utf-8")
    summary = {
        "schema": "agentrecx.p14d1.alignment.v1",
        "sources": {
            "release_rows": "data/p14d1/raw/dev.jsonl (MemRerank Electronics dev; downloaded)",
            "query_mapping": "data/p14d1/raw/amazon_c4_user_query_dict_o3-mini_rewritten_v2.json",
            "timestamps_and_history": "data/p13r/electronics/tasks_raw.jsonl (derived from same MemRerank release / Amazon-C4)",
            "catalogue": "data/p13r/electronics/catalog_meta.jsonl",
        },
        "counts": dict(stats),
        "aligned_tasks": len(aligned),
        "targets": len(targets),
        "issues": issues[:50],
        "notes": [
            "original Amazon-C4 query is the primary evaluation query; rewritten is recorded separately",
            "history is strictly before target timestamp; target item removed; verified_purchase preserved",
            "history product metadata is sparse in the independent C4 catalogue; this is reported as a limitation",
            "target identity lives only in dev_targets.jsonl and is never passed to a ranker",
        ],
    }
    (OUT / "alignment_manifest.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
