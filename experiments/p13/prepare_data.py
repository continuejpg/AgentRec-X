"""P13 Sports/MemRerank data preparation.

Builds the isolated P13 experiment space from Amazon-C4 user-purchase-history.  The
query is a public C4 proxy query; the target review is used only to obtain the target
timestamp and to validate the positive, never as prior history.  Histories are strictly
before the target timestamp when a timestamp is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data/p13/sports"
OUT = ROOT / "data/p13/sports/tasks.jsonl"
MANIFEST = ROOT / "data/p13/sports/manifest.json"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
TARGET_TS = ROOT / "runs/p13/reviews/target_timestamps.jsonl"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    mapping = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    item2id = mapping["item2id"]
    user2id = mapping["user2id"]
    c4 = {}
    with (DATA / "amazon_c4_test.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            c4[int(row["qid"])] = row
    target_rows = {
        int(row["qid"]): row
        for row in (
            json.loads(line)
            for line in TARGET_TS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    tasks = []
    stats = Counter()
    for split in ("train", "dev", "test"):
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats[f"{split}_raw"] += 1
            qid = int(row["query"])
            target_ts = target_rows.get(qid)
            target = str(row["pos_product"])
            c4row = c4.get(qid)
            if c4row is None or not str(c4row.get("query") or "").strip():
                stats[f"{split}_missing_query"] += 1
                continue
            if target_ts is None:
                stats[f"{split}_missing_target_timestamp"] += 1
                continue
            if target not in item2id or str(row["user_id"]) not in user2id:
                stats[f"{split}_target_not_in_current_catalogue"] += 1
                continue
            target_ms = int(target_ts["target_timestamp_ms"])
            history = []
            target_in_history = False
            for category, items in (row.get("grouped_purchase_history") or {}).items():
                for item in items:
                    asin = str(item[0])
                    timestamp = int(item[1])
                    if timestamp >= target_ms:
                        stats[f"{split}_future_history_dropped"] += 1
                        continue
                    if asin == target:
                        target_in_history = True
                        continue
                    history.append(
                        {
                            "parent_asin": asin,
                            "timestamp_ms": timestamp,
                            "category": str(category),
                            "verified": bool(item[2]) if len(item) > 2 else None,
                            "in_current_catalogue": asin in item2id,
                        }
                    )
            if target_in_history:
                stats[f"{split}_target_in_history_dropped"] += 1
                continue
            history.sort(key=lambda item: item["timestamp_ms"])
            task = {
                "sequence_id": f"p13-sports-{qid}",
                "qid": qid,
                "split": split,
                "source_dataset": "amazon-c4-user-purchase-history/Sports_and_Outdoors",
                "proxy_query": str(c4row["query"]),
                "user_id": str(row["user_id"]),
                "user_int_id": int(user2id[str(row["user_id"])]),
                "target_parent_asin": target,
                "target_item_id": int(item2id[target]),
                "target_timestamp_ms": target_ms,
                "target_review_timestamp_matched": True,
                "history": history,
                "history_length": len(history),
                "history_in_current_catalogue": sum(1 for item in history if item["in_current_catalogue"]),
                "label_scope": "observed positive; unlabelled products are unjudged, not judged negatives",
            }
            tasks.append(task)
            stats[f"{split}_kept"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    by_split = Counter(task["split"] for task in tasks)
    split_sets = {
        split: {
            "users": {task["user_id"] for task in tasks if task["split"] == split},
            "qids": {task["qid"] for task in tasks if task["split"] == split},
            "targets": {task["target_parent_asin"] for task in tasks if task["split"] == split},
        }
        for split in ("train", "dev", "test")
    }
    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap[f"{left}_{right}"] = {
            "users": len(split_sets[left]["users"] & split_sets[right]["users"]),
            "qids": len(split_sets[left]["qids"] & split_sets[right]["qids"]),
            "targets": len(split_sets[left]["targets"] & split_sets[right]["targets"]),
        }
    manifest = {
        "schema_version": "agentrecx.p13.sports_data.v1",
        "source": {
            "amazon_c4_user_purchase_history": (
                "https://huggingface.co/datasets/zhiyuanpeng/amazon-c4-user-purchase-history"
            ),
            "amazon_c4_queries": "https://huggingface.co/datasets/McAuley-Lab/Amazon-C4",
            "amazon_reviews_2023": "Sports_and_Outdoors raw reviews for target timestamps",
        },
        "category": "Sports_and_Outdoors",
        "catalogue": {
            "path": str(MAPPINGS.relative_to(ROOT)),
            "catalogue_size": len(item2id),
            "mapping_sha256": sha(MAPPINGS),
            "products_path": str(PRODUCTS.relative_to(ROOT)),
            "products_sha256": sha(PRODUCTS),
        },
        "splits": dict(by_split),
        "reserved_test": "split=test; not used for model selection or tuning in P13",
        "kept_rows": len(tasks),
        "stats": dict(stats),
        "query_semantics": "C4 proxy query generated from review; not a real search log",
        "label_semantics": "one observed positive per query; all other catalogue items unjudged",
        "history_scope": "only timestamps strictly before target timestamp; target review never used as history",
        "target_review_usage": "target review text only matched to obtain target timestamp; not a history feature",
        "split_overlap_check": overlap,
        "files": {
            "train": str((DATA / "train.jsonl").relative_to(ROOT)),
            "dev": str((DATA / "dev.jsonl").relative_to(ROOT)),
            "test": str((DATA / "test.jsonl").relative_to(ROOT)),
            "target_timestamps": str(TARGET_TS.relative_to(ROOT)),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
