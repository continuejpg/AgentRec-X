"""Read-only full-catalogue scorers for the P6 classical benchmark.

Boundary contract
-----------------
A scorer turns one chronological history into one raw score per catalogue item::

    scorer.score_catalog(history_item_ids) -> list[float], length == num_items + 1

``scores[0]`` is the PAD slot (0); real item ids are ``1..num_items``.  The scorer

* does **not** receive the evaluation target,
* does **not** mask already-seen items,
* does **not** rank, break ties or compute metrics.

Masking, ranking and tie-breaking are owned by
:mod:`recommendation.evaluation.metrics`, and the P6 metric formulas by
:mod:`experiments.p6_benchmark.metrics`.  Keeping them out of the adapter means the P6
benchmark measures the accepted serving models under the same canonical ranking contract
as the rest of the repository instead of growing a second evaluator.

Nothing in this module changes Serving V1, trains a model, or writes a model artifact.
Both adapters call the existing full-catalogue ``score_catalog`` surface of the accepted
inference engines; the only new code is identity plumbing and provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from recommendation.datasets.sasrec import encode_inference_history
from recommendation.inference.sasrec import InferenceConfig, SASRecInferenceEngine
from recommendation.inference.twotower import TwoTowerInferenceConfig, TwoTowerInferenceEngine
from recommendation.training.checkpoint import sha256_file

#: P6 benchmark v0 as delivered: the test target is predicted from ``items[:-2]``.
P6_VARIANT = "p6_shared_history"
#: The repository-wide ``agentrecx.eval_protocol.v1`` test mode: the validation interaction
#: is an observed, earlier event and therefore part of the test history.
CANONICAL_VARIANT = "eval_protocol_v1_test_history"

#: The two documented classical history variants.  They are deliberately not merged:
#: P6 v0 froze the shorter history, while every formal SASRec / Two-Tower number in the
#: repository uses the longer one.  Results are therefore reported separately.
VARIANTS: dict[str, str] = {
    P6_VARIANT: "history = items[:-2] (P6 benchmark v0 as delivered)",
    CANONICAL_VARIANT: (
        "history = items[:-2] + validation target (repository eval_protocol.v1 test mode)"
    ),
}


def history_for_row(row: Mapping[str, Any], variant: str) -> tuple[int, ...]:
    """Return the model input history for one P6 classical row and one variant.

    The target is never added by this function: the two targets are the final two
    interactions of the user's sequence, and the only difference between the variants is
    whether the *validation* interaction (which happened before the test interaction) is
    visible at test time.

    Raises
    ------
    ValueError
        Unknown variant, or a row without the required integer fields.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown history variant {variant!r}; known: {sorted(VARIANTS)}")
    try:
        base = tuple(int(value) for value in row["history_item_ids"])
    except (KeyError, TypeError) as exc:
        raise ValueError("row is missing a usable 'history_item_ids' sequence") from exc
    if variant == P6_VARIANT:
        return base
    try:
        validation_target = int(row["validation_item_id"])
    except (KeyError, TypeError) as exc:
        raise ValueError("row is missing a usable 'validation_item_id'") from exc
    return base + (validation_target,)


@runtime_checkable
class FullCatalogueScorer(Protocol):
    """One model's raw full-catalogue scoring surface (PAD slot included)."""

    name: str
    score_kind: str
    num_items: int

    def score_catalog(self, history_item_ids: Sequence[int]) -> list[float]: ...

    def item_id_to_parent_asin(self, item_id: int) -> str: ...

    def model_metadata(self) -> dict[str, Any]: ...


class SASRecFullCatalogueScorer:
    """Read-only adapter over the accepted :class:`SASRecInferenceEngine`.

    The engine is loaded once, in ``eval`` mode, and only its public
    ``encode_history`` / ``score_catalog`` / identity methods are used.
    """

    name = "sasrec"
    score_kind = "sasrec_logit"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        mappings_path: str | Path,
        manifest_path: str | Path | None = None,
        device: str = "cpu",
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        config = InferenceConfig(
            checkpoint_path=Path(checkpoint_path),
            mappings_path=Path(mappings_path),
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        self.engine = SASRecInferenceEngine(config)
        self.num_items = int(self.engine.num_items)
        self.checkpoint_sha256 = self.engine.checkpoint_sha256
        self.checkpoint_path = str(Path(checkpoint_path))

    def score_catalog(self, history_item_ids: Sequence[int]) -> list[float]:
        """Raw SASRec logits for every catalogue id (index 0 = PAD)."""
        history_parent_asins = [
            self.engine.item_id_to_parent_asin(int(item_id)) for item_id in history_item_ids
        ]
        encoded, _truncated = self.engine.encode_history(history_parent_asins)
        scores = self.engine.score_catalog(encoded)
        if len(scores) != self.num_items + 1:
            raise ValueError(
                f"{self.name} returned {len(scores)} scores, expected {self.num_items + 1}"
            )
        return scores

    def item_id_to_parent_asin(self, item_id: int) -> str:
        return self.engine.item_id_to_parent_asin(int(item_id))

    def model_metadata(self) -> dict[str, Any]:
        payload = dict(self.engine.model_metadata())
        payload.update({
            "adapter": "SASRecFullCatalogueScorer",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score_kind": self.score_kind,
        })
        return payload


class TwoTowerFullCatalogueScorer:
    """Read-only adapter over the accepted :class:`TwoTowerInferenceEngine`."""

    name = "two_tower"
    score_kind = "two_tower_cosine"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        mappings_path: str | Path,
        categories_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        checkpoint = Path(checkpoint_path)
        config = TwoTowerInferenceConfig(
            checkpoint_path=checkpoint,
            categories_path=Path(categories_path) if categories_path is not None else None,
            mappings_path=Path(mappings_path),
            device=device,
        )
        self.engine = TwoTowerInferenceEngine(config)
        self.num_items = int(self.engine.num_items)
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_sha256 = sha256_file(checkpoint)

    def score_catalog(self, history_item_ids: Sequence[int]) -> list[float]:
        """Raw Two-Tower cosine scores for every catalogue id (index 0 = PAD)."""
        encoded = encode_inference_history(
            [int(item_id) for item_id in history_item_ids],
            self.engine.max_seq_len,
            self.num_items,
        )
        scores = self.engine.score_catalog(encoded)
        if len(scores) != self.num_items + 1:
            raise ValueError(
                f"{self.name} returned {len(scores)} scores, expected {self.num_items + 1}"
            )
        return scores

    def item_id_to_parent_asin(self, item_id: int) -> str:
        return self.engine.item_id_to_parent_asin(int(item_id))

    def model_metadata(self) -> dict[str, Any]:
        payload = dict(self.engine.model_metadata())
        payload.update({
            "adapter": "TwoTowerFullCatalogueScorer",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score_kind": self.score_kind,
        })
        return payload


def build_scorer(
    model: str,
    *,
    sasrec_checkpoint: str | Path,
    twotower_checkpoint: str | Path,
    twotower_categories: str | Path | None,
    mappings_path: str | Path,
    sasrec_manifest: str | Path | None = None,
    expected_sasrec_sha256: str | None = None,
    device: str = "cpu",
) -> FullCatalogueScorer:
    """Build one named full-catalogue scorer.

    Only ``cpu`` is accepted here: the P6 work described in the handoff is a CPU task and
    the local CUDA stack is not a supported runtime for this torch build.  Training is
    never involved.
    """
    if device != "cpu":
        raise ValueError(f"P6 classical evaluation runs on CPU, got device={device!r}")
    if model == "sasrec":
        return SASRecFullCatalogueScorer(
            checkpoint_path=sasrec_checkpoint,
            mappings_path=mappings_path,
            manifest_path=sasrec_manifest,
            device=device,
            expected_checkpoint_sha256=expected_sasrec_sha256,
        )
    if model == "two_tower":
        return TwoTowerFullCatalogueScorer(
            checkpoint_path=twotower_checkpoint,
            mappings_path=mappings_path,
            categories_path=twotower_categories,
            device=device,
        )
    raise ValueError(f"unknown model {model!r}; known: ('sasrec', 'two_tower')")


def scalar_identity_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """A deterministic digest of the identity facts this evaluation relies on.

    Used only for provenance/reproducibility.  It is not a metric and it never feeds the
    scorer.  It covers the split rows' case ids and target identities so a later run can
    prove it evaluated the same frozen cases.
    """
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            "\x1f".join(
                (
                    str(row.get("case_id", "")),
                    str(row.get("user_id", "")),
                    str(row.get("test_parent_asin", "")),
                    str(row.get("test_item_id", "")),
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_mapping_num_items(mappings_path: str | Path) -> int:
    """Read ``num_items`` from the accepted mappings artifact without loading both tables."""
    payload = json.loads(Path(mappings_path).read_text(encoding="utf-8"))
    num_items = payload.get("num_items")
    if not isinstance(num_items, int) or num_items < 1:
        raise ValueError(f"mappings artifact has no usable num_items: {mappings_path}")
    return int(num_items)


__all__ = [
    "CANONICAL_VARIANT",
    "P6_VARIANT",
    "VARIANTS",
    "FullCatalogueScorer",
    "SASRecFullCatalogueScorer",
    "TwoTowerFullCatalogueScorer",
    "build_scorer",
    "history_for_row",
    "load_mapping_num_items",
    "scalar_identity_checksum",
]
