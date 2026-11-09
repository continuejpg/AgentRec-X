"""Compact ItemCF artifact and optional in-pool rerank signal."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.baselines.itemcf import ItemCF
from recommendation.baselines.itemcf_csr import ItemCFCSR
from recommendation.demo.itemcf_rerank import ItemCFRerankSignal


def _model():
    model = ItemCF(num_items=8, min_cooccurrence=1)
    model.fit(
        [
            (1, 2, 3),
            (1, 2, 4),
            (2, 3, 5),
            (3, 6, 7),
        ]
    )
    return model


def test_compact_csr_keeps_every_positive_relation_and_aggregates_exactly(tmp_path: Path) -> None:
    model = _model()
    compact = ItemCFCSR.from_model(model)
    assert compact.directed_pairs == 2 * model.stats.num_similarity_pairs
    assert compact.indptr[-1] == compact.directed_pairs

    history = (1, 2)
    original = {}
    for item in set(history):
        for neighbour, similarity in model.similarity.get(item, {}).items():
            if neighbour in history:
                continue
            original[neighbour] = original.get(neighbour, 0.0) + similarity
    assert compact.aggregate(history) == original

    path = tmp_path / "itemcf.npz"
    manifest = compact.save(path)
    assert manifest["directed_pairs"] == compact.directed_pairs
    loaded = ItemCFCSR.load(path)
    assert loaded.directed_pairs == compact.directed_pairs
    assert loaded.aggregate(history) == compact.aggregate(history)
    assert [loaded.neighbours(item)[0].tolist() for item in range(1, 9)] == [
        compact.neighbours(item)[0].tolist() for item in range(1, 9)
    ]


def _signal():
    compact = ItemCFCSR.from_model(_model())
    id_to_asin = {item: f"A{item}" for item in range(1, 9)}
    asin_to_id = {asin: item for item, asin in id_to_asin.items()}
    return ItemCFRerankSignal(
        compact,
        id_to_asin=id_to_asin,
        asin_to_id=asin_to_id,
        mappings_sha256="",
    )


def test_itemcf_signal_only_reorders_existing_candidates_with_real_evidence() -> None:
    signal = _signal()
    candidates = tuple(
        SimpleNamespace(parent_asin=identity) for identity in ("A3", "A2", "A8")
    )
    baseline = {"A2": 0.02, "A3": 0.02, "A8": 0.02}
    order = signal.order(candidates, ("A1",), baseline)
    assert order is not None
    assert set(order) == {"A3", "A2", "A8"}  # no candidate added or removed
    # A2 and A3 have learned similarity to history A1; A8 has no evidence and keeps
    # only its baseline score.  A2/A3 precede A8.
    assert order[-1] == "A8"
    assert order[0] in {"A2", "A3"}
    assert signal.order(candidates, (), baseline) is None
    no_evidence = tuple(SimpleNamespace(parent_asin="A8") for _ in range(1))
    assert signal.order(no_evidence, ("A1",), {"A8": 0.02}) is None
    assert signal.order(candidates, ("A1",), {"A2": 0.02}) is None  # missing baseline


def test_itemcf_signal_rejects_duplicate_candidates() -> None:
    signal = _signal()
    candidates = (
        SimpleNamespace(parent_asin="A2"),
        SimpleNamespace(parent_asin="A2"),
    )
    assert signal.order(candidates, ("A1",), {"A2": 0.02}) is None


def test_itemcf_artifact_rejects_mapping_mismatch() -> None:
    compact = ItemCFCSR.from_model(_model(), mappings_sha256="expected")
    with pytest.raises(ValueError):
        ItemCFRerankSignal(
            compact,
            id_to_asin={1: "A1"},
            asin_to_id={"A1": 1},
            mappings_sha256="other",
        )
