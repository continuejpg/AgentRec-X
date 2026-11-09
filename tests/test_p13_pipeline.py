"""P13 data/rank invariant tests (small, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.p13.ranking import FEATURE_NAMES, metric_rows

TASKS = Path("data/p13/sports/tasks.jsonl")
MANIFEST = Path("data/p13/sports/manifest.json")


def test_p13_history_is_strictly_before_target() -> None:
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert tasks
    for task in tasks:
        assert task["split"] in {"train", "dev", "test"}
        for item in task["history"]:
            assert int(item["timestamp_ms"]) < int(task["target_timestamp_ms"])
            assert item["parent_asin"] != task["target_parent_asin"]
        assert task["proxy_query"].strip()


def test_reserved_test_is_present_but_not_used_for_ranking() -> None:
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(task["split"] == "test" for task in tasks)
    ranking = json.loads(Path("runs/p13/ranking/summary.json").read_text(encoding="utf-8"))
    assert ranking["reserved_test_used"] is False


def test_ranking_metric_is_deterministic_and_correct() -> None:
    scores = np.asarray([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    result = metric_rows(scores, target_index=2)
    assert result["rank"] == 2
    assert result["recall@10"] == 1.0
    assert result["mrr@10"] == 0.5


def test_p13_feature_names_do_not_include_target_or_future_fields() -> None:
    forbidden = ("target", "label", "future", "target_rank")
    assert not any(any(token in name for token in forbidden) for name in FEATURE_NAMES)
