"""Reranker interface tests; mock scorer only, no real cross-encoder claim."""

from __future__ import annotations

import pytest

from recommendation.retrieval.reranker import (
    CrossEncoderReranker,
    ExistingReranker,
    RerankCandidate,
    build_reranker,
)


def _candidates():
    return [
        RerankCandidate("A", "wireless headphones", 0.9, 1),
        RerankCandidate("B", "mechanical keyboard", 0.8, 2),
        RerankCandidate("C", "wireless earbuds", 0.7, 3),
    ]


def test_existing_reranker_preserves_order_and_ranks() -> None:
    reranked = ExistingReranker().rerank("wireless", _candidates())
    assert [item.item_id for item in reranked] == ["A", "B", "C"]
    assert [item.final_rank for item in reranked] == [1, 2, 3]
    assert all(item.reranker_score is None for item in reranked)


class _Scorer:
    def predict(self, pairs, **_kwargs):
        return [10.0 if "earbuds" in text else 1.0 for _query, text in pairs]


def test_cross_encoder_interface_sorts_scores_and_has_explicit_rank() -> None:
    reranker = CrossEncoderReranker(
        model_path="mock-model", loader=lambda _path, _device, _length: _Scorer(),
    )
    result = reranker.rerank("wireless", _candidates())
    assert [item.item_id for item in result] == ["C", "A", "B"]
    assert [item.final_rank for item in result] == [1, 2, 3]
    assert result[0].reranker_score == 10.0
    assert reranker.available is True


def test_cross_encoder_missing_model_is_explicit_not_fallback() -> None:
    reranker = CrossEncoderReranker(model_path="/definitely/missing-model")
    with pytest.raises(RuntimeError, match="model unavailable"):
        reranker.rerank("wireless", _candidates())
    assert reranker.available is False


def test_reranker_mode_validation_and_factory() -> None:
    assert isinstance(build_reranker("existing"), ExistingReranker)
    with pytest.raises(ValueError, match="reranker mode"):
        build_reranker("magic")
    with pytest.raises(ValueError, match="requires a model_path"):
        build_reranker("cross_encoder", model_path="")
