"""P13-R data/retrieval invariants (small, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

TASKS = Path("data/p13r/electronics/tasks.jsonl")
MANIFEST = Path("data/p13r/electronics/manifest.json")
RETRIEVAL = Path("runs/p13r/retrieval/summary.json")


def test_p13r_history_is_strictly_before_target_and_not_target() -> None:
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert tasks
    for task in tasks:
        assert task["query"].strip()
        assert task["target_category"] == "Electronics"
        for item in task["history"]:
            assert int(item["timestamp_ms"]) < int(task["target_timestamp_ms"])
            assert item["item_id"] != task["target_parent_asin"]


def test_p13r_split_is_isolated() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["split_overlap"] == {
        "train_dev": {"users": 0, "qids": 0, "targets": 0},
        "train_test": {"users": 0, "qids": 0, "targets": 0},
        "dev_test": {"users": 0, "qids": 0, "targets": 0},
    }
    assert manifest["reserved_test"]


def test_p13r_test_split_was_not_retrieved() -> None:
    summary = json.loads(RETRIEVAL.read_text(encoding="utf-8"))
    assert summary["test_retrieved"] is False
    assert summary["splits"] == {"train": 817, "dev": 123}
    assert summary["catalog_items"] == 76918
