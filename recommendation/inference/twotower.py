"""Serving engine for the Two-Tower retrieval model.

Why a separate module from the training code
--------------------------------------------
The accepted serving stack has one shape: an object exposing ``recommend(history, k=...)``
returning a :class:`~recommendation.inference.RecommendationResult`, wrapped by
``RecommendationTool``.  This module supplies that shape for Two-Tower, exactly as
:class:`~recommendation.inference.SASRecInferenceEngine` does for SASRec, so the two models
are interchangeable everywhere the repository consumes a recommender - the API, the demo, the
candidate plane and the benchmark.

What it preserves
-----------------
* **The same input convention.**  Histories are left-padded to ``max_seq_len`` with PAD id 0,
  the convention the model was trained with and that
  :func:`recommendation.datasets.sasrec.encode_inference_history` produces.
* **The same legality rule.**  The engine ranks the full catalogue and excludes the caller's
  seen items; it never emits PAD and never returns an item the caller already interacted with.
  Candidate legality therefore stays in one place per arm, and the benchmark's evaluator still
  owns the final masking decision for the arms it scores.
* **An honest score label.**  Scores are cosine similarities in ``[-1, 1]``, not SASRec logits
  and not probabilities.  They are exposed with :data:`TWO_TOWER_SCORE_KIND` so nothing can
  compare them against another source's raw values - the ledger records the same string.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from recommendation.datasets.sasrec import encode_inference_history
from recommendation.inference.ranking import rank_top_k
from recommendation.inference.sasrec import (
    InferenceError,
    load_item_mapping,
    Recommendation,
    RecommendationResult,
)
from recommendation.models.twotower import TwoTower, TwoTowerConfig

__all__ = [
    "TWO_TOWER_SCORE_KIND",
    "TwoTowerInferenceConfig",
    "TwoTowerInferenceEngine",
]

#: What this engine's scores measure.  Recorded on the ledger and in API responses so a
#: Two-Tower cosine is never read as a SASRec logit or a BM25 value.
TWO_TOWER_SCORE_KIND = "two_tower_cosine"


@dataclass(frozen=True)
class TwoTowerInferenceConfig:
    """Where a two-tower checkpoint, its category artifact and its id mapping live.

    ``mappings_path`` is the accepted ``parent_asin`` <-> integer-id artifact.  It is required
    in practice for serving: the plane grounds *identities*, so a candidate list of bare
    integers could not be verified against the trusted catalogue.
    """

    checkpoint_path: Path
    categories_path: Path | None = None
    mappings_path: Path | None = None
    device: str = "cpu"


class TwoTowerInferenceEngine:
    """Loads a Two-Tower checkpoint and serves full-catalogue recommendations.

    The model and the precomputed item matrix are built once at construction.  Item vectors
    are a pure function of static item features, so recomputing them per request would be
    wasted work; this is the property that makes the architecture a retrieval model rather
    than a reranker.
    """

    #: What this engine's scores mean, for consumers that label provenance.
    score_kind = TWO_TOWER_SCORE_KIND

    def __init__(self, config: TwoTowerInferenceConfig) -> None:
        checkpoint_path = Path(config.checkpoint_path)
        if not checkpoint_path.exists():
            raise InferenceError(f"two-tower checkpoint not found: {checkpoint_path}")
        self.device = torch.device(config.device)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or "model_config" not in payload:
            raise InferenceError(
                f"{checkpoint_path} is not a two-tower checkpoint (no model_config)"
            )
        extra = payload.get("extra") or {}
        if extra.get("architecture") not in (None, "two_tower_dssm"):
            raise InferenceError(
                f"{checkpoint_path} declares architecture {extra.get('architecture')!r}, "
                "not 'two_tower_dssm'"
            )

        resolved = dict(payload["model_config"])
        if "num_categories" in resolved:
            resolved["num_categories"] = tuple(resolved["num_categories"])
        self.config = TwoTowerConfig(**resolved)
        self.num_items = self.config.num_items
        self.max_seq_len = self.config.max_seq_len
        self.checkpoint_metadata = {
            "architecture": extra.get("architecture", "two_tower_dssm"),
            "objective": extra.get("objective"),
            "seed": payload.get("seed"),
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "pooling": self.config.pooling,
            "embedding_size": self.config.embedding_size,
        }

        model = TwoTower(self.config)
        model.load_state_dict(payload["model_state_dict"])
        if self.config.num_metadata_fields:
            categories_path = config.categories_path or (
                checkpoint_path.parent / "metadata_categories.json"
            )
            if not Path(categories_path).exists():
                raise InferenceError(
                    "the checkpoint was trained with categorical metadata but the category "
                    f"artifact is missing: {categories_path}"
                )
            from recommendation.datasets.twotower import (
                load_metadata_categories,
                metadata_tensor,
            )

            model.set_categories(metadata_tensor(load_metadata_categories(categories_path)))
        model.to(self.device)
        model.eval()
        self.model = model

        self._id2item: list[str | None] | None = None
        self._item2id: dict[str, int] = {}
        if config.mappings_path is not None:
            mapping = load_item_mapping(
                config.mappings_path, expected_num_items=self.num_items
            )
            self._id2item = list(mapping["id2item"])
            self._item2id = dict(mapping["item2id"])
        self.mappings_path = config.mappings_path

    # -- identity ---------------------------------------------------------- #

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        """Return the model item id for a canonical identity.

        Raises
        ------
        InferenceError
            No mapping was configured, or the identity is not in the served catalogue.  An
            engine that silently dropped or coerced an unknown history item would answer a
            different question than the one it was asked.
        """
        if self._id2item is None:
            raise InferenceError(
                "no mappings artifact is configured, so a parent_asin cannot be resolved to a "
                "model item id; construct the engine with mappings_path"
            )
        mapped = self._item2id.get(str(parent_asin))
        if mapped is None:
            raise InferenceError(
                f"unknown parent_asin {parent_asin!r}: it is not in the served catalogue"
            )
        return int(mapped)

    def has_parent_asin(self, parent_asin: str) -> bool:
        """True when the mapping knows this canonical identity."""
        return str(parent_asin) in self._item2id

    def item_id_to_parent_asin(self, item_id: int) -> str:
        """Return the canonical identity for a model item id.

        Raises
        ------
        InferenceError
            No mapping artifact was configured, or the id is not in the served catalogue.
            Returning a synthetic identity would let an ungroundable candidate into a
            candidate list, which is exactly what the plane's verifier exists to prevent.
        """
        if self._id2item is None:
            raise InferenceError(
                "no mappings artifact is configured, so a model item id cannot be resolved to "
                "a parent_asin; construct the engine with mappings_path"
            )
        if not 0 < int(item_id) < len(self._id2item):
            raise InferenceError(f"item id {item_id!r} is outside the served catalogue")
        identity = self._id2item[int(item_id)]
        if not identity:
            raise InferenceError(f"item id {item_id!r} has no identity in the mappings artifact")
        return str(identity)

    # -- metadata ---------------------------------------------------------- #

    def is_ready(self) -> bool:
        """The engine is always ready once constructed."""
        return True

    def model_metadata(self) -> dict[str, Any]:
        """Non-sensitive model metadata."""
        return {
            "model_type": "TwoTowerDSSM",
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "embedding_size": self.config.embedding_size,
            "hidden_size": self.config.hidden_size,
            "pooling": self.config.pooling,
            "metadata_fields": self.config.num_metadata_fields,
            "score_kind": TWO_TOWER_SCORE_KIND,
            "device": str(self.device),
            "parameter_count": int(sum(p.numel() for p in self.model.parameters())),
        }

    # -- serving ----------------------------------------------------------- #

    def score_catalog(self, encoded_history: Sequence[int]) -> list[float]:
        """Return one score per catalogue id (index 0 is PAD)."""
        tensor = torch.tensor([list(encoded_history)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            scores = self.model.full_catalog_scores(tensor)
        if not bool(torch.isfinite(scores).all()):
            raise InferenceError("the two-tower model produced non-finite catalog scores")
        return [float(value) for value in scores[0].tolist()]

    def recommend(self, history_item_ids: Sequence[int], k: int = 10) -> RecommendationResult:
        """Produce deterministic top-``k`` recommendations for an integer-id history.

        Raises
        ------
        InferenceError
            The history is empty, contains an out-of-range id, or ``k`` is not a positive int
            inside the accepted bounds.
        """
        if isinstance(history_item_ids, (str, bytes)) or history_item_ids is None:
            raise InferenceError("history must be a sequence of integer item ids")
        history = [int(value) for value in history_item_ids]
        if not history:
            raise InferenceError("history must not be empty")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise InferenceError(f"k must be a positive int, got {k!r}")
        for item_id in history:
            if not 0 < item_id <= self.num_items:
                raise InferenceError(
                    f"history item {item_id!r} is outside the served catalogue "
                    f"1..{self.num_items}"
                )

        encoded = encode_inference_history(history, self.max_seq_len, self.num_items)
        scores = self.score_catalog(encoded)
        # ``rank_top_k`` already excludes PAD (candidates are 1..num_items) and the caller's
        # seen items, and breaks ties by ascending item id.  Those are the accepted serving
        # semantics, so this engine reuses them instead of restating them.
        ranked = rank_top_k(scores, num_items=self.num_items, seen_item_ids=history, k=k)
        return RecommendationResult(
            recommendations=[
                Recommendation(
                    rank=item.rank,
                    item_id=item.item_id,
                    parent_asin=self.item_id_to_parent_asin(item.item_id),
                    score=item.score,
                )
                for item in ranked
            ],
            requested_k=k,
            history_length=len(history),
            effective_history_length=min(len(history), self.max_seq_len),
            history_truncated=len(history) > self.max_seq_len,
            eligible_candidates=max(self.num_items - len(set(history)), 0),
            timings_ms={"scoring": 0.0, "ranking": 0.0},
        )
