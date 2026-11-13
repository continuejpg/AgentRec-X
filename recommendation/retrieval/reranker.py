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

RERANKER_MODES = ("existing", "cross_encoder", "qwen3_cross_encoder", "bge_cross_encoder")


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
                   loader: Callable[[str, str, int], Any] | None = None,
                   quantization: str = "4bit",
                   instruction: str | None = None) -> Reranker:
    normalised = str(mode or "").strip().casefold()
    if normalised == "existing":
        return ExistingReranker()
    if normalised in {"cross_encoder", "bge_cross_encoder"}:
        return CrossEncoderReranker(
            model_path=str(model_path or ""), device=device,
            batch_size=batch_size, max_length=max_length, loader=loader,
        )
    if normalised == "qwen3_cross_encoder":
        return Qwen3Reranker(
            model_path=str(model_path or ""), device=device, batch_size=batch_size,
            max_length=max_length, quantization=quantization, instruction=instruction,
            loader=loader,
        )
    raise ValueError(f"reranker mode must be one of {RERANKER_MODES}, got {mode!r}")


class Qwen3Reranker:
    """Official Qwen3-Reranker yes/no-logit scorer (4-bit or BF16).

    This is a real model adapter, not a heuristic.  It follows the Qwen3-Reranker
    inference recipe: build the chat prompt, score the last-token logits for ``yes`` and
    ``no``, and use ``softmax([no, yes])[1]`` as the relevance probability.
    """

    mode = "qwen3_cross_encoder"

    def __init__(
        self, *, model_path: str, device: str = "cuda", batch_size: int = 4,
        max_length: int = 2048, quantization: str = "4bit",
        instruction: str | None = None,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        if not str(model_path or "").strip():
            raise ValueError("qwen3_cross_encoder requires a model_path")
        self.model_path = str(model_path)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self.quantization = str(quantization)
        self.instruction = instruction or (
            "Given a shopping query and a product document, judge whether the product "
            "matches the query. Judge only from the provided product evidence."
        )
        self._loader = loader
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None
        self.load_seconds: float | None = None
        self.scoring_seconds: float = 0.0
        self.pairs_scored: int = 0
        self.peak_vram_mib: float | None = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._prefix_ids: list[int] = []
        self._suffix_ids: list[int] = []

    @property
    def available(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        started = time.perf_counter()
        try:
            if self._loader is not None:
                model, tokenizer = self._loader(
                    self.model_path, self.device, self.max_length,
                )
            else:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, padding_side="left", trust_remote_code=True,
                )
                kwargs: dict[str, Any] = {"trust_remote_code": True}
                if self.quantization == "4bit":
                    from transformers import BitsAndBytesConfig

                    kwargs.update({
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                        ),
                        "device_map": {"": 0} if self.device.startswith("cuda") else "auto",
                        "torch_dtype": torch.float16,
                    })
                else:
                    kwargs.update({"device_map": "auto", "torch_dtype": torch.bfloat16})
                model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
                model.eval()
            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements "
                "based on the Query and the Instruct provided. Note that the answer can "
                "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            tokenizer = self._tokenizer or tokenizer
            self._yes_id = int(tokenizer.convert_tokens_to_ids("yes"))
            self._no_id = int(tokenizer.convert_tokens_to_ids("no"))
            self._prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
            self._suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
            self._tokenizer = tokenizer
            self._model = model
            self.load_seconds = round(time.perf_counter() - started, 6)
            return model, tokenizer
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"Qwen3 reranker unavailable: {self._load_error}") from exc

    def rerank(self, query: str, candidates: Iterable[RerankCandidate],
               *, top_n: int | None = None) -> list[RerankCandidate]:
        items = list(candidates)
        if not items:
            return []
        model, tokenizer = self._load()
        import torch
        started = time.perf_counter()
        texts = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {item.text}"
            for item in items
        ]
        values: list[float] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            input_ids = [
                self._prefix_ids
                + list(tokenizer.encode(text, add_special_tokens=False))[-self.max_length:]
                + self._suffix_ids
                for text in batch
            ]
            padded = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            padded = {key: value.to(model.device) for key, value in padded.items()}
            with torch.inference_mode():
                logits = model(**padded).logits[:, -1, :]
                probs = torch.softmax(torch.stack([
                    logits[:, self._no_id], logits[:, self._yes_id],
                ], dim=1), dim=1)[:, 1]
            values.extend(float(value) for value in probs.float().cpu())
            self.pairs_scored += len(batch)
        self.scoring_seconds += time.perf_counter() - started
        if torch.cuda.is_available():
            self.peak_vram_mib = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if len(values) != len(items):
            raise RuntimeError(
                f"Qwen3 reranker returned {len(values)} scores for {len(items)} candidates"
            )
        scored = [RerankCandidate(
            item_id=item.item_id, text=item.text, existing_score=item.existing_score,
            existing_rank=item.existing_rank, reranker_score=value,
        ) for item, value in zip(items, values)]
        scored.sort(key=lambda item: (-float(item.reranker_score or 0.0), item.item_id))
        if top_n is not None:
            scored = scored[:max(0, int(top_n))]
        return [RerankCandidate(
            item_id=item.item_id, text=item.text, existing_score=item.existing_score,
            existing_rank=item.existing_rank, reranker_score=item.reranker_score,
            final_rank=index + 1,
        ) for index, item in enumerate(scored)]
