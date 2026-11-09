"""Optional in-pool ItemCF rerank signal (experimental, default off).

This adapter never adds or removes candidates.  It reads the compact CSR artifact built
from the accepted ItemCF cosine graph and adds the fixed R5 reciprocal-rank term
``1/(60 + ItemCF in-pool rank)`` only for candidates with a real positive ItemCF score.
The caller must supply the ledger's baseline RRF scores; no score is fabricated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from recommendation.baselines.itemcf_csr import ItemCFCSR

__all__ = [
    "ENV_ITEMCF_RERANK",
    "ENV_ITEMCF_ARTIFACT",
    "ItemCFRerankSignal",
    "itemcf_rerank_enabled",
]

ENV_ITEMCF_RERANK = "AGENTRECX_ITEMCF_RERANK"
ENV_ITEMCF_ARTIFACT = "AGENTRECX_ITEMCF_ARTIFACT"
DEFAULT_MAPPINGS = "data/processed/Sports_and_Outdoors_mappings.json"
RRF_K = 60


def itemcf_rerank_enabled() -> bool:
    return (os.environ.get(ENV_ITEMCF_RERANK) or "").strip().lower() in {
        "1", "true", "on", "yes"
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ItemCFRerankSignal:
    """Read-only adapter: compact CSR + trusted item identity mapping."""

    def __init__(
        self,
        artifact: ItemCFCSR,
        *,
        id_to_asin: Mapping[int, str],
        asin_to_id: Mapping[str, int],
        mappings_sha256: str,
    ) -> None:
        self.artifact = artifact
        self.id_to_asin = dict(id_to_asin)
        self.asin_to_id = dict(asin_to_id)
        self.mappings_sha256 = mappings_sha256
        if artifact.mappings_sha256 and artifact.mappings_sha256 != mappings_sha256:
            raise ValueError("ItemCF artifact was built for different item mappings")

    @classmethod
    def from_env(cls) -> "ItemCFRerankSignal | None":
        if not itemcf_rerank_enabled():
            return None
        artifact_path = (os.environ.get(ENV_ITEMCF_ARTIFACT) or "").strip()
        if not artifact_path:
            raise ValueError(f"{ENV_ITEMCF_RERANK}=on requires {ENV_ITEMCF_ARTIFACT}")
        artifact = ItemCFCSR.load(Path(artifact_path))
        mappings_path = Path(DEFAULT_MAPPINGS)
        mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
        num_items = int(mappings["num_items"])
        if artifact.num_items != num_items:
            raise ValueError("ItemCF artifact catalogue size does not match mappings")
        id_to_asin = {int(value): str(key) for key, value in mappings["item2id"].items()}
        asin_to_id = {asin: item_id for item_id, asin in id_to_asin.items()}
        return cls(
            artifact,
            id_to_asin=id_to_asin,
            asin_to_id=asin_to_id,
            mappings_sha256=_sha256(mappings_path),
        )

    def order(
        self,
        candidates: Sequence[Any],
        history_asins: Iterable[str],
        baseline_scores: Mapping[str, float],
    ) -> tuple[str, ...] | None:
        """Return a new order over exactly ``candidates``, or None to leave it unchanged."""
        identities = tuple(str(getattr(item, "parent_asin", "")) for item in candidates)
        if len(set(identities)) != len(identities):
            return None
        history_ids = []
        for asin in history_asins:
            item_id = self.asin_to_id.get(str(asin))
            if item_id is not None:
                history_ids.append(item_id)
        if not history_ids:
            return None
        aggregate = self.artifact.aggregate(history_ids)
        evidence = {}
        for identity in identities:
            item_id = self.asin_to_id.get(identity)
            if item_id is not None and item_id in aggregate:
                evidence[identity] = aggregate[item_id]
        if not evidence:
            return None
        if any(identity not in baseline_scores for identity in identities):
            # The caller must provide the real baseline RRF score for every candidate;
            # otherwise there is nothing truthful to add the term to.
            return None
        evidence_order = sorted(evidence, key=lambda identity: (-evidence[identity], self.asin_to_id[identity]))
        itemcf_rank = {identity: rank for rank, identity in enumerate(evidence_order, start=1)}
        combined = {
            identity: float(baseline_scores[identity])
            + (1.0 / (RRF_K + itemcf_rank[identity]) if identity in itemcf_rank else 0.0)
            for identity in identities
        }
        return tuple(sorted(identities, key=lambda identity: (-combined[identity], self.asin_to_id[identity])))
