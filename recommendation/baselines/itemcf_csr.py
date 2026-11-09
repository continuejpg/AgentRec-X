"""Compact, versioned CSR artifact for the accepted cosine ItemCF.

The training-time fit still uses :class:`recommendation.baselines.itemcf.ItemCF`; this
module only changes **how the learned positive similarity graph is stored and read**.

Artifact layout (one compressed NumPy ``.npz``):
  format, num_items, mappings_sha256, sequences_sha256, min_cooccurrence,
  directed_pairs, undirected_pairs,
  indptr[int32, num_items+2], indices[int32, directed_pairs], data[float64, ...],
  item_freq[int32, num_items+2]

Rows are item ids ``1..num_items``; row ``0`` is the empty PAD row and the last
``indptr`` value ends the arrays.  Every positive directed relation is retained at
the original Python-float precision; no similarity is truncated or quantised.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from recommendation.baselines.itemcf import ItemCF

__all__ = ["ARTIFACT_FORMAT", "ItemCFCSR", "array_digest"]

ARTIFACT_FORMAT = "agentrecx.itemcf_cosine.csr.v1"


def array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class ItemCFCSR:
    """Read-only CSR view of a fitted ItemCF similarity graph."""

    def __init__(
        self,
        *,
        num_items: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        item_freq: np.ndarray,
        mappings_sha256: str = "",
        sequences_sha256: str = "",
        min_cooccurrence: int = 1,
        undirected_pairs: int = 0,
    ) -> None:
        self.num_items = int(num_items)
        self.indptr = np.asarray(indptr, dtype=np.int32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float64)
        self.item_freq = np.asarray(item_freq, dtype=np.int32)
        self.mappings_sha256 = str(mappings_sha256)
        self.sequences_sha256 = str(sequences_sha256)
        self.min_cooccurrence = int(min_cooccurrence)
        self.undirected_pairs = int(undirected_pairs)

    @property
    def directed_pairs(self) -> int:
        return int(self.indices.size)

    @classmethod
    def from_model(
        cls,
        model: ItemCF,
        *,
        mappings_sha256: str = "",
        sequences_sha256: str = "",
    ) -> "ItemCFCSR":
        if not model.is_fitted:
            raise ValueError("ItemCF model must be fitted before packing")
        n = int(model.num_items)
        indptr = np.zeros(n + 2, dtype=np.int32)
        counts = np.zeros(n + 1, dtype=np.int64)
        for item_id, neighbours in model.similarity.items():
            counts[int(item_id)] = len(neighbours)
        indptr[1:] = np.cumsum(counts, dtype=np.int64)
        directed = int(indptr[-1])
        indices = np.empty(directed, dtype=np.int32)
        data = np.empty(directed, dtype=np.float64)
        cursor = 0
        # Item ids are already deterministic; sort neighbour ids for a stable graph.
        for item_id in range(1, n + 1):
            neighbours = model.similarity.get(item_id, {})
            for neighbour in sorted(neighbours):
                indices[cursor] = int(neighbour)
                data[cursor] = float(neighbours[neighbour])
                cursor += 1
        freq = np.zeros(n + 2, dtype=np.int32)
        for item_id, count in model.item_freq.items():
            if 1 <= int(item_id) <= n:
                freq[int(item_id)] = int(count)
        return cls(
            num_items=n,
            indptr=indptr,
            indices=indices,
            data=data,
            item_freq=freq,
            mappings_sha256=mappings_sha256,
            sequences_sha256=sequences_sha256,
            min_cooccurrence=int(model.min_cooccurrence),
            undirected_pairs=int(model.stats.num_similarity_pairs),
        )

    def save(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format=np.asarray([ARTIFACT_FORMAT], dtype=str),
            num_items=np.asarray([self.num_items], dtype=np.int64),
            directed_pairs=np.asarray([self.directed_pairs], dtype=np.int64),
            undirected_pairs=np.asarray([self.undirected_pairs], dtype=np.int64),
            min_cooccurrence=np.asarray([self.min_cooccurrence], dtype=np.int64),
            mappings_sha256=np.asarray([self.mappings_sha256], dtype=str),
            sequences_sha256=np.asarray([self.sequences_sha256], dtype=str),
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            item_freq=self.item_freq,
        )
        payload = {
            "format": ARTIFACT_FORMAT,
            "num_items": self.num_items,
            "directed_pairs": self.directed_pairs,
            "undirected_pairs": self.undirected_pairs,
            "min_cooccurrence": self.min_cooccurrence,
            "bytes": path.stat().st_size,
            "indptr_sha256": array_digest(self.indptr),
            "indices_sha256": array_digest(self.indices),
            "data_sha256": array_digest(self.data),
            "item_freq_sha256": array_digest(self.item_freq),
        }
        return payload

    @classmethod
    def load(cls, path: Path) -> "ItemCFCSR":
        data = np.load(Path(path), allow_pickle=False)
        if str(data["format"][0]) != ARTIFACT_FORMAT:
            raise ValueError("ItemCF artifact format is missing or unsupported")
        artifact = cls(
            num_items=int(data["num_items"][0]),
            indptr=data["indptr"],
            indices=data["indices"],
            data=data["data"],
            item_freq=data["item_freq"],
            mappings_sha256=str(data["mappings_sha256"][0]),
            sequences_sha256=str(data["sequences_sha256"][0]),
            min_cooccurrence=int(data["min_cooccurrence"][0]),
            undirected_pairs=int(data["undirected_pairs"][0]),
        )
        if artifact.indptr.size != artifact.num_items + 2:
            raise ValueError("ItemCF artifact indptr length does not match num_items")
        if artifact.indices.size != artifact.data.size:
            raise ValueError("ItemCF artifact indices/data mismatch")
        if int(artifact.indptr[-1]) != artifact.indices.size:
            raise ValueError("ItemCF artifact terminal indptr does not match pairs")
        if int(data["directed_pairs"][0]) != artifact.indices.size:
            raise ValueError("ItemCF artifact declared pair count mismatch")
        if artifact.item_freq.size != artifact.num_items + 2:
            raise ValueError("ItemCF artifact item_freq length does not match num_items")
        return artifact

    def neighbours(self, item_id: int) -> tuple[np.ndarray, np.ndarray]:
        item_id = int(item_id)
        if not 1 <= item_id <= self.num_items:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        start = int(self.indptr[item_id])
        end = int(self.indptr[item_id + 1])
        return self.indices[start:end], self.data[start:end]

    def aggregate(self, history: Iterable[int]) -> dict[int, float]:
        """Aggregate similarity exactly as accepted ItemCF scoring does."""
        history_set = {int(item) for item in history}
        scores: dict[int, float] = {}
        for item_id in history_set:
            neighbours, values = self.neighbours(item_id)
            for neighbour, value in zip(neighbours.tolist(), values.tolist()):
                if neighbour in history_set:
                    continue
                scores[neighbour] = scores.get(neighbour, 0.0) + value
        return scores
