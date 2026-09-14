"""SASRec inference engine for serving.

Pipeline::

    checkpoint loader -> SASRec model (eval, no grad) -> full-catalog scores
                      -> deterministic ranking (mask PAD + seen) -> recommendations

This module is framework independent: it has no HTTP, no FastAPI and no Agent logic.
The API layer must call :class:`SASRecInferenceEngine` rather than reimplementing
mapping, encoding, forward pass, masking or ranking.

Item identity contract
----------------------
External identity is the Amazon ``parent_asin``.  Internally the model uses integer
item ids ``1..num_items`` with ``0`` reserved for PAD.  The mapping comes from the
existing preprocessing artifact - no second mapping system is constructed here.

Two distinct history concepts, kept deliberately separate:

* **model window** - the most recent ``max_seq_len`` items, left-padded with 0, which
  the Transformer actually sees;
* **full supplied history** - every item the caller sent, which drives seen-item
  masking so an already-interacted item is never recommended.

Scores returned by this engine are raw SASRec model scores, **not** probabilities.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from recommendation import config
from recommendation.datasets.sasrec import encode_inference_history
from recommendation.models.sasrec import SASRec, SASRecConfig
from recommendation.training.checkpoint import CHECKPOINT_FORMAT, sha256_file

from .ranking import (
    PAD_ID,
    RankedItem,
    RankingError,
    eligible_candidate_count,
    rank_top_k,
    validate_k,
)

#: Accepted serving devices (same semantics as the Milestone 5 trainer).
SUPPORTED_DEVICES: tuple[str, ...] = ("cpu", "cuda", "cuda:0")

#: Model configuration keys that must agree across checkpoint / manifest / mapping.
REQUIRED_MODEL_KEYS: tuple[str, ...] = (
    "num_items",
    "max_seq_len",
    "hidden_size",
    "num_blocks",
    "num_heads",
    "dropout",
)


class InferenceError(RuntimeError):
    """Raised when the serving stack cannot be built or used safely."""


class UnknownItemError(InferenceError):
    """Raised when a caller supplies a ``parent_asin`` outside the served catalog."""


class RequestValidationError(InferenceError):
    """Raised when a recommendation request violates the input contract."""


@dataclass(frozen=True)
class InferenceConfig:
    """Serving configuration.

    Paths default to the repository layout but may be overridden explicitly; nothing
    machine-specific is hard-coded in the logic below.
    """

    checkpoint_path: Path
    mappings_path: Path
    manifest_path: Path | None = None
    device: str = "cpu"
    #: Optional expected checkpoint digest; when set, a mismatch fails startup.
    expected_checkpoint_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (paths omitted for API exposure)."""
        return {"device": self.device}


@dataclass(frozen=True)
class Recommendation:
    """One recommendation with external identity attached."""

    rank: int
    item_id: int
    parent_asin: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "parent_asin": self.parent_asin,
            "score": self.score,
        }


@dataclass
class RecommendationResult:
    """Structured result of one recommendation call."""

    recommendations: list[Recommendation]
    requested_k: int
    history_length: int
    effective_history_length: int
    history_truncated: bool
    eligible_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def returned_k(self) -> int:
        """How many recommendations were actually returned."""
        return len(self.recommendations)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "recommendations": [r.as_dict() for r in self.recommendations],
            "requested_k": self.requested_k,
            "returned_k": self.returned_k,
            "history_length": self.history_length,
            "effective_history_length": self.effective_history_length,
            "history_truncated": self.history_truncated,
            "eligible_candidates": self.eligible_candidates,
            "timings_ms": self.timings_ms,
        }


def resolve_device(requested: str) -> torch.device:
    """Resolve a serving device, failing fast when it is unusable.

    Reuses the Milestone 5 semantics: ``cpu`` always works; ``cuda`` / ``cuda:0``
    require CUDA to genuinely be available, and there is **no** silent CPU fallback.
    """
    if not isinstance(requested, str) or not requested.strip():
        raise InferenceError(f"device must be a non-empty string, got {requested!r}")
    normalised = requested.strip().lower()
    if normalised not in SUPPORTED_DEVICES:
        raise InferenceError(
            f"unsupported device {requested!r}; supported values are {SUPPORTED_DEVICES}"
        )
    if normalised == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise InferenceError(
            f"device {requested!r} requested but torch.cuda.is_available() is False; "
            "refusing to silently fall back to CPU"
        )
    return torch.device("cuda:0" if normalised == "cuda" else normalised)


def load_item_mapping(path: str | Path, expected_num_items: int | None = None) -> dict[str, Any]:
    """Load ``parent_asin <-> int id`` mappings from the preprocessing artifact.

    Validates cardinality so a mapping that disagrees with the model cannot be served
    silently: the integer ids must be exactly ``1..num_items`` with PAD ``0`` unused.
    """
    path = Path(path)
    if not path.exists():
        raise InferenceError(f"mappings artifact not found: {path}")
    with open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    for key in ("num_items", "item2id", "id2item"):
        if key not in payload:
            raise InferenceError(f"mappings artifact is missing {key!r}: {path}")

    item2id: dict[str, int] = payload["item2id"]
    num_items = int(payload["num_items"])

    if expected_num_items is not None and num_items != expected_num_items:
        raise InferenceError(
            f"mapping cardinality mismatch: artifact says num_items={num_items}, "
            f"model/manifest says {expected_num_items}"
        )
    if len(item2id) != num_items:
        raise InferenceError(
            f"mapping is inconsistent: num_items={num_items} but item2id has "
            f"{len(item2id)} entries"
        )
    ids = set(item2id.values())
    if PAD_ID in ids:
        raise InferenceError(
            f"mapping assigns PAD id {PAD_ID} to a real item; PAD must stay unused"
        )
    if min(ids) != config.FIRST_REAL_ID or max(ids) != num_items or len(ids) != num_items:
        raise InferenceError(
            f"mapping ids must be contiguous {config.FIRST_REAL_ID}..{num_items}; "
            f"got {len(ids)} distinct ids spanning {min(ids)}..{max(ids)}"
        )

    id2item = payload["id2item"]
    if len(id2item) != num_items + 1 or id2item[PAD_ID] is not None:
        raise InferenceError(
            "id2item must have length num_items + 1 with null at the PAD slot"
        )

    return {"num_items": num_items, "item2id": item2id, "id2item": id2item}


class SASRecInferenceEngine:
    """Loads an accepted SASRec checkpoint and serves recommendations.

    The model is loaded **once** at construction, put in ``eval()`` mode and never
    returned to training mode.  Every forward pass runs under ``torch.inference_mode``
    so no gradients or autograd graph are ever created.
    """

    def __init__(
        self,
        config: InferenceConfig,
        *,
        verify_checkpoint_sha256: bool = True,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)

        checkpoint_path = Path(config.checkpoint_path)
        if not checkpoint_path.exists():
            raise InferenceError(f"checkpoint not found: {checkpoint_path}")

        # ---- checkpoint identity ----------------------------------------- #
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        if (
            verify_checkpoint_sha256
            and config.expected_checkpoint_sha256
            and self.checkpoint_sha256 != config.expected_checkpoint_sha256
        ):
            raise InferenceError(
                "checkpoint SHA-256 mismatch: expected "
                f"{config.expected_checkpoint_sha256}, got {self.checkpoint_sha256}"
            )

        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise InferenceError(
                f"{checkpoint_path} is not an {CHECKPOINT_FORMAT} checkpoint"
            )
        for key in ("model_state_dict", "model_config", "num_items", "max_seq_len"):
            if key not in payload:
                raise InferenceError(f"checkpoint is missing required field {key!r}")

        self.checkpoint_metadata = {
            "format": payload["format"],
            "seed": payload.get("seed"),
            "max_seq_len": int(payload["max_seq_len"]),
            "num_items": int(payload["num_items"]),
            "model_config": payload.get("model_config", {}),
        }

        # ---- manifest cross-check (optional but validated when present) --- #
        manifest = self._load_manifest(config.manifest_path)
        self.manifest = manifest
        model_config_dict = payload["model_config"]
        if manifest is not None:
            self._crosscheck_manifest(manifest, payload)

        for key in REQUIRED_MODEL_KEYS:
            if key not in model_config_dict:
                raise InferenceError(f"checkpoint model_config is missing {key!r}")

        # ---- mapping cross-check ----------------------------------------- #
        mapping = load_item_mapping(config.mappings_path, expected_num_items=int(payload["num_items"]))
        self.num_items: int = mapping["num_items"]
        self._item2id: dict[str, int] = mapping["item2id"]
        self._id2item: list[str | None] = mapping["id2item"]

        # ---- model reconstruction ---------------------------------------- #
        model_config = SASRecConfig(
            num_items=int(model_config_dict["num_items"]),
            max_seq_len=int(model_config_dict["max_seq_len"]),
            hidden_size=int(model_config_dict["hidden_size"]),
            num_blocks=int(model_config_dict["num_blocks"]),
            num_heads=int(model_config_dict["num_heads"]),
            dropout=float(model_config_dict["dropout"]),
            feed_forward_multiplier=float(model_config_dict.get("feed_forward_multiplier", 4.0)),
            layer_norm_eps=float(model_config_dict.get("layer_norm_eps", 1e-8)),
            initializer_range=float(model_config_dict.get("initializer_range", 0.02)),
        )
        if model_config.num_items != self.num_items:
            raise InferenceError(
                f"model num_items {model_config.num_items} disagrees with mapping "
                f"{self.num_items}"
            )
        self.model_config = model_config

        # Seed the RNG for reproducible construction, then build directly from the
        # reconstructed config so non-default fields (feed_forward_multiplier,
        # layer_norm_eps, initializer_range) match the trained architecture exactly.
        torch.manual_seed(int(payload.get("seed") or 0))
        self.model: SASRec = SASRec(model_config)
        # Strict load: a mismatched architecture or unexpected key must fail loudly
        # rather than being ignored or silently resized.
        try:
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise InferenceError(
                f"checkpoint state_dict is incompatible with the reconstructed "
                f"architecture: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.max_seq_len: int = model_config.max_seq_len
        self.loaded_at = time.time()

    # -- construction helpers --------------------------------------------- #

    @staticmethod
    def _load_manifest(path: Path | None) -> dict[str, Any] | None:
        """Load the optional run manifest."""
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            raise InferenceError(f"run manifest not found: {path}")
        with open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _crosscheck_manifest(manifest: dict[str, Any], payload: dict[str, Any]) -> None:
        """Verify checkpoint/manifest agreement on the frozen configuration."""
        recorded = manifest.get("model_config") or {}
        checkpoint_config = payload.get("model_config") or {}
        for key in REQUIRED_MODEL_KEYS:
            if key in recorded and key in checkpoint_config:
                left, right = recorded[key], checkpoint_config[key]
                if isinstance(left, float) or isinstance(right, float):
                    agree = abs(float(left) - float(right)) < 1e-12
                else:
                    agree = int(left) == int(right)
                if not agree:
                    raise InferenceError(
                        f"manifest/checkpoint disagreement on {key}: "
                        f"manifest={left!r} checkpoint={right!r}"
                    )
        for key in ("max_seq_len", "num_items"):
            if key in manifest and key in payload:
                if int(manifest[key]) != int(payload[key]):
                    raise InferenceError(
                        f"manifest/checkpoint disagreement on {key}: "
                        f"manifest={manifest[key]} checkpoint={payload[key]}"
                    )

    # -- mapping ----------------------------------------------------------- #

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Map an integer item id to its ``parent_asin``."""
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise InferenceError(f"item_id must be an integer, got {type(item_id).__name__}")
        if item_id == PAD_ID:
            raise InferenceError("PAD (0) is not a real item and cannot be mapped")
        if not config.FIRST_REAL_ID <= item_id <= self.num_items:
            raise InferenceError(f"item_id {item_id} outside catalog 1..{self.num_items}")
        value = self._id2item[item_id]
        if value is None:  # pragma: no cover - guarded by load_item_mapping
            raise InferenceError(f"item_id {item_id} has no parent_asin in the mapping")
        return value

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Map a ``parent_asin`` to its integer item id, or raise explicitly."""
        if not isinstance(parent_asin, str):
            raise RequestValidationError(
                f"history entries must be parent_asin strings, got {type(parent_asin).__name__}"
            )
        item_id = self._item2id.get(parent_asin)
        if item_id is None:
            raise UnknownItemError(f"unknown parent_asin: {parent_asin!r}")
        return item_id

    def has_parent_asin(self, parent_asin: str) -> bool:
        """True when the catalog contains this ``parent_asin``."""
        return isinstance(parent_asin, str) and parent_asin in self._item2id

    # -- encoding ---------------------------------------------------------- #

    def encode_history(self, history_parent_asins: Sequence[str]) -> tuple[list[int], bool]:
        """Encode a caller history into the fixed-length model window.

        Returns ``(encoded_item_ids, truncated)``.  The encoded window is the most
        recent ``max_seq_len`` items, left-padded with PAD (0), preserving order.

        Duplicates are preserved: repeated interactions are genuine interactions and
        must not be collapsed before encoding.  The caller's sequence is never mutated.
        """
        item_ids = [self.parent_asin_to_item_id(value) for value in history_parent_asins]
        truncated = len(item_ids) > self.max_seq_len
        encoded = encode_inference_history(item_ids, self.max_seq_len, self.num_items)
        return list(encoded), truncated

    # -- scoring / recommendation ------------------------------------------ #

    def score_catalog(self, encoded_history: Sequence[int]) -> list[float]:
        """Return raw full-catalog scores for one encoded history (length N+1)."""
        tensor = torch.tensor([list(encoded_history)], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.inference_mode():
            scores = self.model.full_catalog_scores(tensor)
        if not bool(torch.isfinite(scores).all()):
            raise InferenceError("model produced non-finite catalog scores")
        return [float(value) for value in scores[0].tolist()]

    def recommend(
        self,
        history_parent_asins: Sequence[str],
        k: int = 10,
    ) -> RecommendationResult:
        """Produce deterministic top-``k`` recommendations for a history.

        ``history_parent_asins`` is a chronological sequence of ``parent_asin`` values.
        It may contain duplicates, and it may be longer than ``max_seq_len`` (only the
        newest ``max_seq_len`` items reach the model, but **all** of them are masked).

        Raises
        ------
        RequestValidationError
            Empty history, wrong types, or an invalid ``k``.
        UnknownItemError
            A history entry is not in the served catalog.
        """
        validate_k(k)
        if isinstance(history_parent_asins, (str, bytes)) or history_parent_asins is None:
            raise RequestValidationError("history must be a sequence of parent_asin strings")
        history = list(history_parent_asins)
        if not history:
            raise RequestValidationError("history must not be empty")

        requested_k = k
        history_length = len(history)
        # Strict behaviour: unknown items are rejected, never coerced to PAD or dropped.
        encoded, truncated = self.encode_history(history)
        effective_length = min(history_length, self.max_seq_len)

        seen_ids = [self.parent_asin_to_item_id(value) for value in history]

        started = time.perf_counter()
        scores = self.score_catalog(encoded)
        scoring_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        ranked: list[RankedItem] = rank_top_k(
            scores, num_items=self.num_items, seen_item_ids=seen_ids, k=requested_k
        )
        ranking_ms = (time.perf_counter() - started) * 1000.0

        recommendations = [
            Recommendation(
                rank=item.rank,
                item_id=item.item_id,
                parent_asin=self.item_id_to_parent_asin(item.item_id),
                score=item.score,
            )
            for item in ranked
        ]

        return RecommendationResult(
            recommendations=recommendations,
            requested_k=requested_k,
            history_length=history_length,
            effective_history_length=effective_length,
            history_truncated=truncated,
            eligible_candidates=eligible_candidate_count(self.num_items, seen_ids),
            timings_ms={"scoring": round(scoring_ms, 4), "ranking": round(ranking_ms, 4)},
        )

    # -- metadata ---------------------------------------------------------- #

    def model_metadata(self) -> dict[str, Any]:
        """Return non-sensitive model metadata for the ``/v1/model`` endpoint.

        Provenance is reported in two clearly separated parts so the serving source
        commit is never confused with the commit recorded during training.
        """
        manifest_git = (self.manifest or {}).get("git") or {}
        return {
            "model_type": "SASRec",
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "hidden_size": self.model_config.hidden_size,
            "num_blocks": self.model_config.num_blocks,
            "num_heads": self.model_config.num_heads,
            "dropout": self.model_config.dropout,
            "device": str(self.device),
            "checkpoint_sha256": self.checkpoint_sha256,
            # Serving freezes requires_grad, so count *all* parameters rather than
            # the Milestone 3 helper's trainable-only count.
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "model_parameters_frozen": all(
                not p.requires_grad for p in self.model.parameters()
            ),
            "provenance": {
                "formal_run_git": {
                    "commit": manifest_git.get("commit"),
                    "branch": manifest_git.get("branch"),
                    "dirty": manifest_git.get("dirty"),
                    "note": (
                        "Git state recorded by the formal Milestone 5 training run; the "
                        "model was trained from that (uncommitted) working tree."
                    ),
                },
                "serving_note": (
                    "The post-run code-only checkpoint captures the reviewed source "
                    "tree corresponding to the accepted Milestone 5 implementation."
                ),
            },
        }

    def is_ready(self) -> bool:
        """True when the model is loaded, in eval mode and on the expected device."""
        return (
            self.model is not None
            and not self.model.training
            and all(not p.requires_grad for p in self.model.parameters())
        )


__all__ = [
    "PAD_ID",
    "SUPPORTED_DEVICES",
    "InferenceConfig",
    "InferenceError",
    "Recommendation",
    "RecommendationResult",
    "RequestValidationError",
    "SASRecInferenceEngine",
    "UnknownItemError",
    "load_item_mapping",
    "resolve_device",
]
