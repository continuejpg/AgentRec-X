"""P13-R Electronics data preparation.

Reuses the P13 data protocol but rebuilds an independent Electronics catalogue from the
Amazon-C4 sampled metadata snapshot; it does not reuse the Sports 5-core mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
C4_META = ROOT / "runs/p13r/raw/c4_sampled_meta.jsonl"
C4_QUERIES = ROOT / "data/p13/sports/amazon_c4_test.csv"
MEM = ROOT / "data/p13/electronics"
TS_FILES = [
    ROOT / "runs/p13r/raw/Electronics.test.csv",
    ROOT / "runs/p13r/raw/Electronics.valid.csv",
]
OUT_DIR = ROOT / "data/p13r/electronics"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item = str(row.get("item_id") or "")
            if item:
                records[item] = {
                    "item_id": item,
                    "category": str(row.get("category") or ""),
                    "text": str(row.get("metadata") or ""),
                }
    return records


def load_queries(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(row["qid"])] = row
    return out


def target_timestamps(targets: set[tuple[str, str]], files: list[Path]) -> dict[tuple[str, str], list[int]]:
    found: dict[tuple[str, str], list[int]] = defaultdict(list)
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["user_id"], row["parent_asin"])
                if key in targets:
                    found[key].append(int(row["timestamp"]))
    return found


def compact(text: str, limit: int = 700) -> str:
    return " ".join(str(text or "").split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4-meta", type=Path, default=C4_META)
    parser.add_argument("--queries", type=Path, default=C4_QUERIES)
    parser.add_argument("--mem", type=Path, default=MEM)
    parser.add_argument("--ts-file", action="append", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.c4_meta)
    queries = load_queries(args.queries)
    catalog = {
        item: row for item, row in meta.items() if row["category"] == "Electronics"
    }
    raw_rows = []
    stats = Counter()
    for split in ("train", "dev", "test"):
        path = args.mem / f"{split}.jsonl"
        for line in path.open(encoding="utf-8"):
            if line.strip():
                raw_rows.append((split, json.loads(line), path))
    targets = {(row["user_id"], row["pos_product"]) for _, row, _ in raw_rows}
    ts_files = args.ts_file or TS_FILES
    timestamps = target_timestamps(targets, ts_files)

    tasks = []
    for split, row, _path in raw_rows:
        stats[f"{split}_raw"] += 1
        target = str(row["pos_product"])
        user = str(row["user_id"])
        qid = int(row["query"])
        c4 = queries.get(qid)
        if c4 is None or not str(c4.get("query") or "").strip():
            stats[f"{split}_missing_query"] += 1
            continue
        if target not in catalog:
            stats[f"{split}_target_not_in_catalog"] += 1
            continue
        target_ts_values = timestamps.get((user, target), [])
        if not target_ts_values:
            stats[f"{split}_missing_target_timestamp"] += 1
            continue
        if len(set(target_ts_values)) != 1:
            stats[f"{split}_ambiguous_target_timestamp"] += 1
            continue
        target_ts = target_ts_values[0]
        history = []
        target_in_history = False
        for category, items in (row.get("grouped_purchase_history") or {}).items():
            for item in items:
                item_id = str(item[0])
                ts = int(item[1])
                if ts >= target_ts:
                    stats[f"{split}_future_history_dropped"] += 1
                    continue
                if item_id == target:
                    target_in_history = True
                    continue
                history.append(
                    {
                        "item_id": item_id,
                        "category": str(category),
                        "timestamp_ms": ts,
                        "verified": bool(item[2]) if len(item) > 2 else None,
                        "has_text_metadata": item_id in meta,
                        "text": compact(meta[item_id]["text"]) if item_id in meta else "",
                    }
                )
        if target_in_history:
            stats[f"{split}_target_in_history_dropped"] += 1
            continue
        history.sort(key=lambda item: item["timestamp_ms"])
        tasks.append(
            {
                "sequence_id": f"p13r-electronics-{qid}",
                "qid": qid,
                "split": split,
                "source_dataset": "amazon-c4-user-purchase-history/Electronics",
                "query": str(c4["query"]),
                "user_id": user,
                "target_parent_asin": target,
                "target_timestamp_ms": target_ts,
                "target_category": catalog[target]["category"],
                "history": history,
                "history_length": len(history),
                "history_with_text": sum(1 for item in history if item["has_text_metadata"]),
                "label_scope": "observed positive; all other candidates unjudged",
            }
        )
        stats[f"{split}_kept"] += 1

    split_sets = {
        split: {
            "users": {t["user_id"] for t in tasks if t["split"] == split},
            "qids": {t["qid"] for t in tasks if t["split"] == split},
            "targets": {t["target_parent_asin"] for t in tasks if t["split"] == split},
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
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tasks.jsonl").write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8"
    )
    (args.out / "catalog.jsonl").write_text(
        "".join(json.dumps({"item_id": item, "category": row["category"], "text": compact(row["text"])}, ensure_ascii=False) + "\n" for item, row in sorted(catalog.items())),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "agentrecx.p13r.electronics_data.v1",
        "category": "Electronics",
        "catalog_source": "Amazon-C4 sampled_item_metadata_1M (Electronics subset)",
        "catalog_size": len(catalog),
        "catalog_sha256": sha(args.c4_meta),
        "catalog_note": "independent C4 sampled metadata catalogue; not the old Sports 5-core mapping and not the complete Electronics catalogue",
        "splits": dict(Counter(task["split"] for task in tasks)),
        "stats": dict(stats),
        "split_overlap": overlap,
        "reserved_test": "split=test; not used for P13-R model selection or ranking analysis",
        "query_semantics": "Amazon-C4 proxy query generated from the target review",
        "label_semantics": "one observed positive per query; other products unjudged",
        "history_scope": "strictly before target timestamp; target review never used as history",
        "target_timestamp_source": [str(path) for path in ts_files],
        "history_metadata_coverage": {
            "tasks": len(tasks),
            "with_text": sum(1 for task in tasks if task["history_with_text"] > 0),
            "total_history_items": sum(task["history_length"] for task in tasks),
            "history_items_with_text": sum(task["history_with_text"] for task in tasks),
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
