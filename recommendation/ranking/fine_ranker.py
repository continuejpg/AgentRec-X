"""Minimal MLP for the M9 candidate rerank pilot.

This is a small pointwise scorer over already-retrieved candidates.  It has no retrieval
responsibility and is not wired into default Serving V1.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FineRankerMLP"]


class FineRankerMLP(nn.Module):
    """Two small hidden layers; deliberately not a large sequence model."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
