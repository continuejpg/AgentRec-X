"""Lazy Cross-Encoder reranker interface for Advanced Retrieval v2.

The existing preference scorer / Planner RERANK action is intentionally *not* called a
Cross-Encoder.  This module adds a separate, opt-in adapter.  No model is loaded unless a
caller explicitly selects ``cross_encoder``; a missing model is an explicit error, not a
silent fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

RERANKER_MODES = ("existing", "cross_encoder")


@dataclass(frozen=True)
class RerankCandidate:
    item_id: str
    text: str
    existing_score: float | None
    existing_rank: int
    reranker_score: float | None = None
    final_rank: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "text_chars": len(self.text),
            "existing_score": self.existing_score, "existing_rank": self.existing_rank,
            "reranker_score": self.reranker_score, "final_rank": self.final_rank,
        }


class Reranker(Protocol):
    mode: str

    @property
    def available(self) -> bool: ...

    def rerank(self, query: str, candidates: Iterable[RerankCandidate],
               *, top_n: int | None = None) -> list[RerankCandidate]: ...


class ExistingReranker:
    """The accepted existing ordering; no model is loaded."""

    mode = "existing"

    @property
    def available(self) -> bool:
        return True

    def rerank(self, query: str, candidates: Iterable[RerankCandidate],
               *, top_n: int | None = None) -> list[RerankCandidate]:
        ordered = sorted(candidates, key=lambda item: (item.existing_rank, item.item_id))
        if top_n is not None:
            ordered = ordered[:max(0, int(top_n))]
        return [RerankCandidate(
            item_id=item.item_id, text=item.text, existing_score=item.existing_score,
            existing_rank=item.existing_rank, reranker_score=None, final_rank=index + 1,
        ) for index, item in enumerate(ordered)]


class CrossEncoderReranker:
    """Lazy adapter around a local Cross-Encoder model.

    ``loader`` is injectable for tests/mock adapters.  The default loader uses
    ``sentence_transformers.CrossEncoder``, which supports the BGE reranker family.  Qwen3
    reranker scoring is intentionally left to a Phase-2 model-specific scorer; supplying
    an incompatible model path fails explicitly.
    """

    mode = "cross_encoder"

    def __init__(
        self, *, model_path: str, device: str = "cpu", batch_size: int = 8,
        max_length: int = 512,
        loader: Callable[[str, str, int], Any] | None = None,
    ) -> None:
        if not str(model_path or "").strip():
            raise ValueError("cross_encoder mode requires a model_path")
        self.model_path = str(model_path)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self._loader = loader
        self._model: Any | None = None
        self._load_error: str | None = None
        self.load_seconds: float | None = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        started = time.perf_counter()
        try:
            if self._loader is not None:
                model = self._loader(self.model_path, self.device, self.max_length)
            else:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.model_path, device=self.device,
                                     max_length=self.max_length)
        except Exception as exc:  # explicit; never silently falls back
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"cross-encoder model unavailable: {self._load_error}") from exc
        self._model = model
        self.load_seconds = round(time.perf_counter() - started, 6)
        return model

    def rerank(self, query: str, candidates: Iterable[RerankCandidate],
               *, top_n: int | None = None) -> list[RerankCandidate]:
        items = list(candidates)
        if not items:
            return []
        model = self._load()
        pairs = [[str(query or ""), item.text] for item in items]
        try:
            scores = model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False,
            )
        except TypeError:
            # A small injected scorer may expose ``predict(pairs)`` only.
            scores = model.predict(pairs)
        except Exception as exc:  # explicit failure, not a silent existing-order result
            raise RuntimeError(f"cross-encoder scoring failed: {type(exc).__name__}: {exc}") from exc
        values = [float(value) for value in scores]
        if len(values) != len(items):
            raise RuntimeError(
                f"cross-encoder returned {len(values)} scores for {len(items)} candidates"
            )
        scored = [
            RerankCandidate(
                item_id=item.item_id, text=item.text, existing_score=item.existing_score,
                existing_rank=item.existing_rank, reranker_score=value,
            )
            for item, value in zip(items, values)
        ]
        scored.sort(key=lambda item: (-float(item.reranker_score or 0.0), item.item_id))
        if top_n is not None:
            scored = scored[:max(0, int(top_n))]
        return [RerankCandidate(
            item_id=item.item_id, text=item.text, existing_score=item.existing_score,
            existing_rank=item.existing_rank, reranker_score=item.reranker_score,
            final_rank=index + 1,
        ) for index, item in enumerate(scored)]


def build_reranker(mode: str, *, model_path: str | None = None,
                   device: str = "cpu", batch_size: int = 8,
                   max_length: int = 512,
                   loader: Callable[[str, str, int], Any] | None = None) -> Reranker:
    normalised = str(mode or "").strip().casefold()
    if normalised == "existing":
        return ExistingReranker()
    if normalised == "cross_encoder":
        return CrossEncoderReranker(
            model_path=str(model_path or ""), device=device,
            batch_size=batch_size, max_length=max_length, loader=loader,
        )
    raise ValueError(f"reranker mode must be one of {RERANKER_MODES}, got {mode!r}")
