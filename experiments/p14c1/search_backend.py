"""Thin P14-C1 BGE search integration over the existing CatalogSearchSource.

Uses the P14-D2 identity-aligned dense index and the local BGE encoder.  The existing
CatalogSearchSource remains the single search backend (mode='bge'); BM25 remains the
default and is the recorded fallback when the BGE index/encoder cannot be used.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.inference.dense_catalog import DenseHit

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPO / "runs/p14d2/bge_index.npz"
DEFAULT_MODEL = REPO / "models/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512
#: A real, small query used only to initialise the encoder/tokenizer caches.  It never changes the
#: catalogue index or the model; the actual user query is still encoded by the normal search path.
WARMUP_QUERY = "wireless earbuds for working out"
MAX_ENCODER_ATTEMPTS = 2


class LocalBGEDenseIndex:
    """Minimal dense index implementing the interface CatalogSearchSource expects."""

    def __init__(self, *, identities, vectors, titles, model_path: Path,
                 encoder_loader: Callable[[str], tuple[Any, Any]] | None = None) -> None:
        self.identities = tuple(str(item) for item in identities)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.titles = tuple(titles)
        self.model_path = str(model_path)
        self._encoder_loader = encoder_loader
        self._tokenizer = None
        self._model = None
        # Lazy encoder initialisation is single-flight.  ``_encoder_ready`` is set only after
        # tokenizer + model have both loaded successfully, so a second reader can never observe a
        # half-initialised model (the old meta-device race).
        self._encoder_lock = threading.Lock()
        self._encoder_ready = threading.Event()
        self._encoder_attempts = 0
        self._encoder_loading = False
        self._encoder_error: str | None = None
        self._encoder_load_ms: float | None = None
        self._tokenizer_load_ms: float | None = None
        self._model_load_ms: float | None = None
        self._last_warmup: dict[str, Any] | None = None
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
        """Load the encoder exactly once; concurrent callers wait for the same completed state.

        A failed initialisation is retried at most ``MAX_ENCODER_ATTEMPTS`` times by later callers;
        after that a clear error is raised instead of re-using a partially loaded model.  The model
        object is published only after the tokenizer, model and ``eval()`` all succeed.
        """
        if self._encoder_ready.is_set():
            return self._tokenizer, self._model
        if self._encoder_error and self._encoder_attempts >= MAX_ENCODER_ATTEMPTS:
            raise RuntimeError(
                "BGE encoder initialization failed after "
                f"{MAX_ENCODER_ATTEMPTS} attempts: {self._encoder_error}"
            )
        with self._encoder_lock:
            # Double-checked after acquiring the lock: another caller may have loaded it while we
            # were waiting.
            if self._encoder_ready.is_set():
                return self._tokenizer, self._model
            if self._encoder_error and self._encoder_attempts >= MAX_ENCODER_ATTEMPTS:
                raise RuntimeError(
                    "BGE encoder initialization failed after "
                    f"{MAX_ENCODER_ATTEMPTS} attempts: {self._encoder_error}"
                )
            self._encoder_attempts += 1
            self._encoder_loading = True
            started = time.perf_counter()
            try:
                if self._encoder_loader is not None:
                    tokenizer, model = self._encoder_loader(self.model_path)
                    if hasattr(model, "eval"):
                        model.eval()
                else:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    tokenizer_started = time.perf_counter()
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self._tokenizer_load_ms = round(
                        (time.perf_counter() - tokenizer_started) * 1000.0, 2
                    )
                    model_started = time.perf_counter()
                    model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float32)
                    model.eval()
                    self._model_load_ms = round(
                        (time.perf_counter() - model_started) * 1000.0, 2
                    )
                # Publish only after every piece is fully initialised.
                self._tokenizer = tokenizer
                self._model = model
                self._encoder_error = None
                self._encoder_ready.set()
                return tokenizer, model
            except Exception as exc:  # noqa: BLE001 - explicit error; caller may retry once
                self._tokenizer = None
                self._model = None
                self._encoder_error = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(
                    "BGE encoder initialization failed "
                    f"(attempt {self._encoder_attempts}/{MAX_ENCODER_ATTEMPTS}): "
                    f"{self._encoder_error}"
                ) from exc
            finally:
                self._encoder_loading = False
                self._encoder_load_ms = round((time.perf_counter() - started) * 1000.0, 2)

    def warmup(self, query: str = WARMUP_QUERY) -> dict[str, Any]:
        """Controlled real-query warm-up: load the encoder and run one forward pass.

        The catalogue index and search results are untouched.  The returned timings are the real
        measured costs; callers must not present a failed warm-up as ready.
        """
        started = time.perf_counter()
        try:
            try:
                tokenizer, model = self._load_encoder()
            except Exception:
                # A transient first load failure may be retried once inside the same
                # controlled warm-up.  If the retry also fails, the explicit error below
                # stops the startup; no half-initialised encoder is published.
                tokenizer, model = self._load_encoder()
            import torch

            text = QUERY_INSTRUCTION + str(query or WARMUP_QUERY)
            tokenize_started = time.perf_counter()
            inputs = tokenizer([text], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            tokenize_ms = round((time.perf_counter() - tokenize_started) * 1000.0, 2)
            forward_started = time.perf_counter()
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state[:, 0]
                _ = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)[0].numpy()
            forward_ms = round((time.perf_counter() - forward_started) * 1000.0, 2)
            result = {
                "status": "ok",
                "query": query,
                "encoder_load_ms": self._encoder_load_ms,
                "tokenizer_load_ms": self._tokenizer_load_ms,
                "model_load_ms": self._model_load_ms,
                "tokenization_ms": tokenize_ms,
                "forward_ms": forward_ms,
                "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            }
            self._last_warmup = result
            return result
        except Exception as exc:
            result = {
                "status": "error",
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
                "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            }
            self._last_warmup = result
            raise

    def encoder_status(self) -> dict[str, Any]:
        return {
            "ready": bool(self._encoder_ready.is_set()),
            "loading": bool(self._encoder_loading),
            "attempts": int(self._encoder_attempts),
            "error": self._encoder_error,
            "encoder_load_ms": self._encoder_load_ms,
            "tokenizer_load_ms": self._tokenizer_load_ms,
            "model_load_ms": self._model_load_ms,
            "last_warmup": dict(self._last_warmup) if self._last_warmup else None,
        }

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
