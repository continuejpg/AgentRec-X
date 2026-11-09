"""LightGCN training/inference graph utilities.

The graph is a data artifact derived only from `EvaluationCase.train_history`.  The
normalization is kept here so training and evaluation cannot disagree about item-row
offsets or edge weights.
"""

from __future__ import annotations

import numpy as np
import torch


class LightGCNDataError(ValueError):
    """Raised when graph edge arrays are malformed."""


def build_normalized_adjacency(
    edge_index: torch.Tensor | np.ndarray,
    *,
    num_users: int,
    num_items: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return symmetric normalized adjacency over users 0..U-1 and items 1..N.

    ``edge_index`` is ``[E, 2]`` of ``(user_index, item_id)``.  Item id ``i`` maps to
    graph row ``U + i - 1``; this offset is the part most easily gotten wrong.
    """
    edges = np.asarray(edge_index.detach().cpu() if isinstance(edge_index, torch.Tensor) else edge_index)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise LightGCNDataError(f"edge_index must be [E, 2], got {tuple(edges.shape)}")
    if edges.shape[0] == 0:
        raise LightGCNDataError("edge_index has no edges")
    users = edges[:, 0].astype(np.int64)
    items = edges[:, 1].astype(np.int64)
    if users.min() < 0 or users.max() >= num_users:
        raise LightGCNDataError("user index outside 0..num_users-1")
    if items.min() < 1 or items.max() > num_items:
        raise LightGCNDataError("item id outside 1..num_items")

    item_rows = items + num_users - 1
    rows = np.concatenate([users, item_rows])
    cols = np.concatenate([item_rows, users])
    degree = np.bincount(rows, minlength=num_users + num_items).astype(np.float64)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    values = inv_sqrt[rows] * inv_sqrt[cols]
    adjacency = torch.sparse_coo_tensor(
        torch.tensor(np.stack([rows, cols]), dtype=torch.long),
        torch.tensor(values, dtype=torch.float32),
        (num_users + num_items, num_users + num_items),
    ).coalesce()
    return adjacency.to(device) if device is not None else adjacency
