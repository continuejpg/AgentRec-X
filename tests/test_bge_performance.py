"""Targeted BGE single-flight and warm-up tests (no network, no GPU)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.p14c1.search_backend import LocalBGEDenseIndex
from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.inference.dense_catalog import DenseHit


def _index(loader):
    vectors = np.vstack([np.eye(768, dtype=np.float32)[0], np.eye(768, dtype=np.float32)[1]])
    return LocalBGEDenseIndex(
        identities=["A", "B"], vectors=vectors, titles=["A title", "B title"],
        model_path="unused-model", encoder_loader=loader,
    )


def test_encoder_single_flight_loads_once_and_publishes_ready():
    calls = 0
    lock = threading.Lock()

    def loader(_path):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return SimpleNamespace(), SimpleNamespace(eval=lambda: None)

    index = _index(loader)
    barrier = threading.Barrier(8)
    results: list[object] = []

    def worker():
        barrier.wait()
        results.append(index._load_encoder())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls == 1
    assert len(results) == 8
    status = index.encoder_status()
    assert status["ready"] is True
    assert status["attempts"] == 1
    assert status["loading"] is False


def test_encoder_failure_is_retried_once_and_can_recover():
    calls = 0

    def loader(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated meta-device load failure")
        return SimpleNamespace(), SimpleNamespace(eval=lambda: None)

    index = _index(loader)
    with pytest.raises(RuntimeError, match="attempt 1/2"):
        index._load_encoder()
    assert index.encoder_status()["ready"] is False
    assert index.encoder_status()["error"]
    tokenizer, model = index._load_encoder()
    assert tokenizer is not None and model is not None
    assert calls == 2
    assert index.encoder_status()["ready"] is True
    assert index.encoder_status()["error"] is None


def test_encoder_repeated_failures_stop_after_two_attempts():
    calls = 0

    def loader(_path):
        nonlocal calls
        calls += 1
        raise RuntimeError("always fails")

    index = _index(loader)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            index._load_encoder()
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        index._load_encoder()
    assert calls == 2
    assert index.encoder_status()["ready"] is False


class _FakeDense:
    def __init__(self):
        self.warmup_queries: list[str] = []
        self.search_queries: list[str] = []

    def warmup(self, query):
        self.warmup_queries.append(query)
        return {"status": "ok", "query": query, "total_ms": 1.0}

    def search(self, query, *, limit=10):
        self.search_queries.append(query)
        return [DenseHit(parent_asin="amazon_electronics_c4:A", rank=1, score=0.5, title="A title")]

    def encoder_status(self):
        return {"ready": True, "attempts": 1}


class _Catalog:
    def __init__(self):
        self.records = {
            "amazon_electronics_c4:A": SimpleNamespace(title="A title"),
            "amazon_electronics_c4:B": SimpleNamespace(title="B title"),
        }


def test_catalog_source_warmup_exercises_real_dense_search_and_is_ready():
    dense = _FakeDense()
    source = CatalogSearchSource(_Catalog(), mode="bge", dense_index=dense)
    assert source.search_ready is True
    result = source.warmup("wireless headphones")
    assert result["status"] == "ok"
    assert dense.warmup_queries == ["wireless headphones"]
    assert dense.search_queries == ["wireless headphones"]
    assert result["retrieval_check"]["matched"] is True
    assert source.status()["search_ready"] is True


def test_bm25_source_warmup_is_explicitly_not_required():
    source = CatalogSearchSource(_Catalog(), mode="bm25")
    assert source.warmup()["status"] == "not_required"
    assert source.search_ready is True
