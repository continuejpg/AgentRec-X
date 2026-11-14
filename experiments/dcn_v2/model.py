"""Small DCN-V2 / DCN-Mix-style feature-interaction ranker.

The model is intentionally simple: standardized dense features, stacked cross layers,
a parallel MLP, and a scalar ranking score.  It is prepared for the next round when
independent Query + Event-Sourced Memory + Candidate supervision exists.

This module is not wired into the production Scorer or Gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class CrossLayer(nn.Module):
    """DCN-V2 style element-wise cross layer: x0 * (W x) + x."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * (self.linear(x) + self.bias) + x


@dataclass
class DCNConfig:
    input_dim: int
    hidden: int = 128
    cross_layers: int = 3
    dropout: float = 0.1


class DCNV2Ranker(nn.Module):
    def __init__(self, config: DCNConfig):
        super().__init__()
        self.config = config
        dim = config.hidden
        self.input_proj = nn.Linear(config.input_dim, dim)
        self.cross_layers = nn.ModuleList([CrossLayer(dim) for _ in range(config.cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(config.input_dim, dim), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(dim, dim), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x0 = self.input_proj(features)
        x = x0
        for layer in self.cross_layers:
            x = layer(x0, x)
        deep = self.deep(features)
        return self.head(torch.cat([x, deep], dim=-1)).squeeze(-1)
