"""Minimal M9 Fine Ranker model contract."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.ranking.fine_ranker import FineRankerMLP


def test_fine_ranker_has_a_lightweight_candidate_score_contract() -> None:
    model = FineRankerMLP(input_dim=16, hidden_dim=64, dropout=0.0)
    features = torch.randn(7, 16)
    logits = model(features)
    assert logits.shape == (7,)
    assert torch.isfinite(logits).all()
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000
