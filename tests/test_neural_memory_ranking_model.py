"""Shape/backward tests for the neural memory-conditioned ranker (CPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.neural_memory_ranking.model import ModelConfig, NeuralMemoryRanker  # noqa: E402


def _batch(item_dim=24, memory_dim=16, batch=3, candidates=4, history=5):
    torch.manual_seed(0)
    return {
        "candidate_item_features": torch.randn(batch, candidates, item_dim),
        "history_item_features": torch.randn(batch, history, item_dim),
        "history_ratings": torch.randint(1, 6, (batch, history)).float(),
        "history_days": torch.rand(batch, history) * 100,
        "history_known": torch.ones(batch, history),
        "history_mask": torch.ones(batch, history, dtype=torch.bool),
        "memory_features": torch.randn(batch, memory_dim),
    }


def test_memory_on_forward_backward_shapes():
    item_dim, memory_dim = 24, 32
    config = ModelConfig(item_dim=item_dim, memory_dim=memory_dim, hidden=16, dropout=0.0, memory_on=True)
    model = NeuralMemoryRanker(
        config,
        attr_indices=list(range(10)),
        high_attr_indices=list(range(10)),
        low_attr_indices=list(range(10, 20)),
    )
    batch = _batch(item_dim, memory_dim)
    scores, ratings = model(**batch)
    assert scores.shape == (3, 4)
    assert ratings.shape == (3, 4)
    loss = scores.mean() + ratings.mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_memory_off_forward_shapes_and_no_history_dependency():
    item_dim, memory_dim = 24, 32
    config = ModelConfig(item_dim=item_dim, memory_dim=memory_dim, hidden=16, dropout=0.0, memory_on=False)
    model = NeuralMemoryRanker(config, list(range(10)), list(range(10)), list(range(10, 20)))
    batch = _batch(item_dim, memory_dim)
    scores_a, ratings_a = model(**batch)
    changed = dict(batch)
    changed["history_item_features"] = torch.randn_like(batch["history_item_features"])
    changed["memory_features"] = torch.randn_like(batch["memory_features"])
    scores_b, ratings_b = model(**changed)
    assert torch.allclose(scores_a, scores_b)
    assert torch.allclose(ratings_a, ratings_b)
    assert scores_a.shape == (3, 4) and ratings_a.shape == (3, 4)


def test_attribute_ablation_disables_interaction_block():
    item_dim, memory_dim = 24, 32
    config = ModelConfig(
        item_dim=item_dim, memory_dim=memory_dim, hidden=16, dropout=0.0,
        memory_on=True, use_attributes=False,
    )
    model = NeuralMemoryRanker(config, list(range(10)), list(range(10)), list(range(10, 20)))
    batch = _batch(item_dim, memory_dim)
    scores, ratings = model(**batch)
    assert scores.shape == (3, 4)
    assert ratings.shape == (3, 4)
