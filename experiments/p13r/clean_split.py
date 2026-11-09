"""Deterministic P13-R split cleanup for user/target overlap.

Original MemRerank splits are retained by default, but this script removes dev/test rows
whose user or target also appears in an earlier split.  Train is unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data/p13r/electronics/tasks_raw.jsonl"
OUT = ROOT / "data/p13r/electronics/tasks.jsonl"
MANIFEST = ROOT / "data/p13r/electronics/manifest.json"
MANIFEST_RAW = ROOT / "data/p13r/electronics/manifest_raw.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--manifest-raw", type=Path, default=MANIFEST_RAW)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [task for task in tasks if task["split"] == "train"]
    train_users = {task["user_id"] for task in train}
    train_targets = {task["target_parent_asin"] for task in train}
    dev_all = [task for task in tasks if task["split"] == "dev"]
    dev = [
        task for task in dev_all
        if task["user_id"] not in train_users and task["target_parent_asin"] not in train_targets
    ]
    dev_users = {task["user_id"] for task in dev}
    dev_targets = {task["target_parent_asin"] for task in dev}
    test_all = [task for task in tasks if task["split"] == "test"]
    test = [
        task for task in test_all
        if task["user_id"] not in train_users | dev_users
        and task["target_parent_asin"] not in train_targets | dev_targets
    ]
    for task in tasks:
        task["split_original"] = task["split"]
    clean = train + dev + test
    split_sets = {
        split: {
            "users": {task["user_id"] for task in clean if task["split"] == split},
            "qids": {task["qid"] for task in clean if task["split"] == split},
            "targets": {task["target_parent_asin"] for task in clean if task["split"] == split},
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
    args.out.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in clean), encoding="utf-8"
    )
    manifest = json.loads(args.manifest_raw.read_text(encoding="utf-8"))
    manifest["raw_tasks"] = len(tasks)
    manifest["splits_original"] = dict(Counter(task["split_original"] for task in tasks))
    manifest["splits"] = dict(Counter(task["split"] for task in clean))
    manifest["removed_for_isolation"] = {
        "dev": len(dev_all) - len(dev),
        "test": len(test_all) - len(test),
        "rule": "remove dev/test rows whose user or target appears in an earlier split; train unchanged",
    }
    manifest["split_overlap"] = overlap
    manifest["reserved_test"] = "split=test; not used for P13-R model selection/ranking analysis"
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "tasks": len(clean),
        "splits": dict(Counter(task["split"] for task in clean)),
        "removed": manifest["removed_for_isolation"],
        "overlap": overlap,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
