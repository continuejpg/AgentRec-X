"""P12 review-matching invariants and P11 task isolation."""

from __future__ import annotations

import json
from pathlib import Path

from types import SimpleNamespace

from experiments.p12_match_reviews import normalize_ms
from recommendation.control.catalog_search import CatalogSearchSource

KEYS = Path("data/p12/review_keys.jsonl")
TASKS = Path("data/p11_benchmark/tasks.jsonl")


def test_timestamp_units_are_normalized() -> None:
    assert normalize_ms(1690000000) == 1690000000000
    assert normalize_ms(1690000000000) == 1690000000000
    assert normalize_ms("1690000000") == 1690000000000
    assert normalize_ms(None) is None


def test_review_keys_are_train_history_only_and_exact() -> None:
    keys = [json.loads(line) for line in KEYS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert keys
    for row in keys:
        assert row["source"] == "train_history_only"
        assert isinstance(row["timestamp_ms"], int)
        assert row["parent_asin"]
        assert row["user_id"]


class _FakeCatalog:
    records = {
        "A": SimpleNamespace(title="water filter alpha", categories=("water",), features=(), description=()),
        "B": SimpleNamespace(title="water filter beta", categories=("water",), features=(), description=()),
    }


class _FakeDense:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query: str, *, limit: int):
        self.calls.append((query, limit))
        return [
            SimpleNamespace(parent_asin="B", score=0.9, rank=1, title="water filter beta"),
            SimpleNamespace(parent_asin="A", score=0.8, rank=2, title="water filter alpha"),
        ][:limit]


def test_hybrid_bm25_head_is_opt_in_and_default_is_unchanged() -> None:
    dense = _FakeDense()
    legacy = CatalogSearchSource(_FakeCatalog(), mode="hybrid", dense_index=dense)
    bounded = CatalogSearchSource(
        _FakeCatalog(), mode="hybrid", dense_index=dense, hybrid_bm25_head=1
    )
    assert legacy.search("water filter", limit=2).matched
    assert bounded.search("water filter", limit=2).matched
    assert legacy.status()["hybrid_bm25_head"] is None
    assert bounded.status()["hybrid_bm25_head"] == 1


def test_p11_task_files_do_not_contain_future_targets() -> None:
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    for task in tasks:
        assert "validation_target" not in task
        assert "test_target" not in task
        assert "next_item_target" not in task
