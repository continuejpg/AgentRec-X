"""Thin P14-C1 BGE search integration over the existing CatalogSearchSource.

Uses the P14-D2 identity-aligned dense index and the local BGE encoder.  The existing
CatalogSearchSource remains the single search backend (mode='bge'); BM25 remains the
default and is the recorded fallback when the BGE index/encoder cannot be used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.inference.dense_catalog import DenseHit

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPO / "runs/p14d2/bge_index.npz"
DEFAULT_MODEL = REPO / "models/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512


class LocalBGEDenseIndex:
    """Minimal dense index implementing the interface CatalogSearchSource expects."""

    def __init__(self, *, identities, vectors, titles, model_path: Path) -> None:
        self.identities = tuple(str(item) for item in identities)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.titles = tuple(titles)
        self.model_path = str(model_path)
        self._tokenizer = None
        self._model = None
        if self.vectors.shape != (len(self.identities), 768):
            raise ValueError("BGE index vectors do not align with identities")
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("BGE index has duplicate identities")
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("BGE index vectors are not normalized")

    @classmethod
    def load(cls, index_path: Path = DEFAULT_INDEX, model_path: Path = DEFAULT_MODEL) -> "LocalBGEDenseIndex":
        data = np.load(index_path, allow_pickle=False)
        return cls(identities=data["identities"], vectors=data["vectors"],
                   titles=data["titles"], model_path=model_path)

    def _load_encoder(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float32)
            self._model.eval()
        return self._tokenizer, self._model

    def search(self, query: str, *, limit: int = 10) -> list[DenseHit]:
        tokenizer, model = self._load_encoder()
        import torch
        text = QUERY_INSTRUCTION + str(query or "")
        inputs = tokenizer([text], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)[0].numpy()
        scores = self.vectors @ vector
        order = np.argsort(-scores, kind="stable")[: int(limit)]
        return [DenseHit(parent_asin=f"{ELECTRONICS_NAMESPACE}:{self.identities[int(position)]}", rank=rank,
                         score=float(scores[int(position)]), title=self.titles[int(position)])
                for rank, position in enumerate(order, start=1)]


def configure_catalogue(catalogue: Any, *, index_path: Path = DEFAULT_INDEX,
                        model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Switch the existing catalogue search source to BGE; BM25 remains fallback."""
    try:
        dense = LocalBGEDenseIndex.load(index_path, model_path)
        catalogue_ids = {product.item_id for product in catalogue._products.values()}
        if set(dense.identities) != catalogue_ids:
            raise ValueError("BGE index identities do not match the Electronics catalogue")
        source = CatalogSearchSource(catalogue._view, mode="bge", dense_index=dense)
        catalogue._search_source = source
        return {"mode": "bge", "status": "active", "index": str(index_path),
                "index_size": len(dense.identities), "model": str(model_path),
                "fallback": "bm25 (unavailable/error)"}
    except Exception as exc:
        # Keep the default BM25 source rather than silently serving a fake dense result.
        return {"mode": "bm25", "status": "fallback", "reason": f"{type(exc).__name__}: {exc}",
                "index": str(index_path)}
