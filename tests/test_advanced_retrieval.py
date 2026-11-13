"""Advanced Retrieval v2: hybrid RRF and enriched-document tests (CPU, no model)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from recommendation.control.catalog_search import (
    CatalogSearchHit,
    CatalogSearchSource,
    reciprocal_rank_fusion,
)
from recommendation.inference.dense_catalog import DenseHit
from recommendation.retrieval import (
    EnrichedBM25Source,
    build_retrieval_document,
)


class _Dense:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, *, limit=10):
        return list(self.hits)[:limit]


def _catalog():
    return SimpleNamespace(records={
        "amazon_electronics_c4:A": SimpleNamespace(title="Wireless headphones A"),
        "amazon_electronics_c4:B": SimpleNamespace(title="Mechanical keyboard B"),
        "amazon_electronics_c4:C": SimpleNamespace(title="Wireless earbuds C"),
    })


def _hit(identity: str, rank: int, score: float) -> CatalogSearchHit:
    return CatalogSearchHit(parent_asin=identity, score=score, matched_fields=("bge_dense",),
                            title=identity)


def test_reciprocal_rank_fusion_is_deterministic_and_deduped() -> None:
    fused = reciprocal_rank_fusion([["A", "B", "A"], ["B", "C"]], k=60)
    assert fused == [("B", pytest.approx(1 / 62 + 1 / 61)),
                     ("A", pytest.approx(1 / 61)),
                     ("C", pytest.approx(1 / 62))]
    assert fused == reciprocal_rank_fusion([["A", "B", "A"], ["B", "C"]], k=60)


def test_hybrid_rrf_merges_bm25_and_dense_with_original_ranks() -> None:
    dense = _Dense([
        DenseHit(parent_asin="amazon_electronics_c4:C", rank=1, score=0.9, title="C"),
        DenseHit(parent_asin="amazon_electronics_c4:A", rank=2, score=0.8, title="A"),
    ])
    source = CatalogSearchSource(_catalog(), mode="hybrid_rrf", dense_index=dense)
    result = source.search("wireless", limit=10)
    assert source.mode == "hybrid"
    assert result.score_kind == "hybrid_bm25_bge"
    assert [hit.parent_asin for hit in result.hits] == [
        "amazon_electronics_c4:A", "amazon_electronics_c4:C",
    ]
    assert len({hit.parent_asin for hit in result.hits}) == len(result.hits)
    a = result.hits[0]
    assert a.bm25_rank == 1 and a.bge_rank == 2
    assert a.bm25_score is not None and a.bge_score == 0.8
    assert "bm25" in a.matched_fields and "bge_dense" in a.matched_fields
    assert result.hits[1].bm25_rank == 2 and result.hits[1].bge_rank == 1


def test_hybrid_drops_dense_identity_not_in_current_catalogue() -> None:
    dense = _Dense([
        DenseHit(parent_asin="amazon_electronics_c4:NOT_IN_CATALOGUE", rank=1,
                 score=1.0, title="foreign"),
        DenseHit(parent_asin="amazon_electronics_c4:A", rank=2, score=0.5, title="A"),
    ])
    source = CatalogSearchSource(_catalog(), mode="hybrid", dense_index=dense)
    result = source.search("wireless", limit=10)
    assert all(hit.parent_asin != "amazon_electronics_c4:NOT_IN_CATALOGUE"
               for hit in result.hits)
    assert result.hits


def test_bm25_mode_rejects_dense_index() -> None:
    with pytest.raises(ValueError, match="must not receive a dense index"):
        CatalogSearchSource(_catalog(), mode="bm25", dense_index=_Dense([]))


def _enriched_product():
    text = "Wireless Mechanical Keyboard for the desk with RGB backlighting."
    record = SimpleNamespace(description=(text,), source="data/p13r/electronics/catalog_meta.jsonl")
    meta = SimpleNamespace(
        dataset_version="sidecar-v1", features=("Wireless keyboard", "Wireless keyboard"),
        description=("A mechanical keyboard with brown switches.",),
        details={"Brand": "Example", "Form Factor": "On Ear", "Price": "Historical $99"},
        title="Example keyboard",
    )
    return SimpleNamespace(
        item_id="B000TEST01", namespaced_id="amazon_electronics_c4:B000TEST01",
        title="Wireless Mechanical Keyboard", text=text, product_type="keyboard",
        attributes={
            "wireless": {"status": "present"},
            "over_ear": {"status": "present"},
            "mechanical": {"status": "present"},
        },
        form_factor="over_ear", record=record, metadata=meta,
        metadata_facts={"brand": {"status": "present", "verification": "structured_metadata"}},
        metadata_conflicts={"form_factor": {"c4": ["over_ear"], "metadata": ["on_ear"],
                                            "resolution": "unknown"}},
    )


def test_enriched_document_omits_conflicting_form_and_historical_price() -> None:
    doc = build_retrieval_document(_enriched_product())
    assert doc.metadata_available is True
    assert "over-ear" not in doc.text.casefold()
    assert "on ear" not in doc.text.casefold()
    assert "price" not in doc.text.casefold()
    assert "Wireless Mechanical Keyboard" in doc.text
    assert "Description:" in doc.text
    assert len(doc.text) <= 1800
    assert "metadata:sidecar" in doc.source_fields


def test_enriched_bm25_searches_enriched_text() -> None:
    doc = build_retrieval_document(_enriched_product())
    source = EnrichedBM25Source({doc.item_id: doc.as_dict()})
    result = source.search("mechanical keyboard", limit=5)
    assert result.matched is True
    assert result.hits[0].parent_asin == "amazon_electronics_c4:B000TEST01"
    assert result.score_kind == "bm25_enriched_v2"
    assert source.name == "catalog_search_enriched_v2"
