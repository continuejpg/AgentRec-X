"""LightGCN: transductive user-item graph collaborative filtering.

Classical LightGCN only: normalized bipartite propagation, layer averaging, no feature
transforms, no attention, no nonlinearity.  The model consumes the graph produced by the
training entry point; it does not build data or evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from recommendation.datasets.sasrec import PAD_ID

__all__ = ["LightGCN", "LightGCNConfig", "LightGCNError"]


class LightGCNError(ValueError):
    """Raised on an invalid graph/model configuration."""


@dataclass(frozen=True)
class LightGCNConfig:
    num_users: int
    num_items: int
    embedding_size: int = 64
    num_layers: int = 3

    def __post_init__(self) -> None:
        for name in ("num_users", "num_items", "embedding_size", "num_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LightGCNError(f"{name} must be a positive int, got {value!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_size": self.embedding_size,
            "num_layers": self.num_layers,
        }


class LightGCN(nn.Module):
    """LightGCN with a symmetric normalized adjacency supplied by the trainer."""

    def __init__(self, config: LightGCNConfig) -> None:
        super().__init__()
        self.config = config
        self.user_embedding = nn.Embedding(config.num_users, config.embedding_size)
        self.item_embedding = nn.Embedding(
            config.num_items + 1, config.embedding_size, padding_idx=PAD_ID
        )
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        with torch.no_grad():
            self.item_embedding.weight[PAD_ID].zero_()

    @property
    def num_users(self) -> int:
        return self.config.num_users

    @property
    def num_items(self) -> int:
        return self.config.num_items

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(user_embeddings, item_embeddings)`` after layer averaging.

        ``adjacency`` is a sparse ``[num_users + num_items]`` square tensor.  Rows
        ``0..num_users-1`` are users and rows ``num_users..`` are items 1..N.
        """
        expected = self.config.num_users + self.config.num_items
        if adjacency.dim() != 2 or tuple(adjacency.shape) != (expected, expected):
            raise LightGCNError(
                f"adjacency must be [{expected}, {expected}], got {tuple(adjacency.shape)}"
            )
        ego = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight[1:]], dim=0
        )
        layers = [ego]
        current = ego
        for _ in range(self.config.num_layers):
            current = torch.sparse.mm(adjacency, current)
            layers.append(current)
        final = torch.stack(layers, dim=0).mean(dim=0)
        users = final[: self.config.num_users]
        items = torch.cat(
            [self.item_embedding.weight[:1], final[self.config.num_users :]], dim=0
        )
        return users, items

    def scores(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        user_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch, num_items + 1]`` scores for selected user rows."""
        if user_indices.dim() != 1:
            raise LightGCNError("user_indices must be 1-D")
        return user_embeddings[user_indices] @ item_embeddings.t()

    def bpr_loss(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """BPR pairwise loss for one batch of graph edges."""
        u = user_embeddings[users]
        pos = (u * item_embeddings[positives]).sum(dim=-1)
        neg = (u * item_embeddings[negatives]).sum(dim=-1)
        return -torch.nn.functional.logsigmoid(pos - neg).mean()
