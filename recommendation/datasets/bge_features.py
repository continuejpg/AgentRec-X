"""Frozen BGE-base item features for the enhanced SASRec experiment.

Only the existing P8-B index is read.  No text encoding, model download, or second
embedding artifact is produced here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BGEFeatureError(ValueError):
    """The BGE index or its item mapping is not usable for training."""


def build_bge_item_matrix(
    index_path: str | Path,
    mappings_path: str | Path,
    *,
    num_items: int,
    dimension: int = 768,
) -> np.ndarray:
    """Return ``[num_items + 1, dimension]`` float32 features indexed by item id.

    Row 0 is PAD and is exactly zero.  Every real item id in the mappings must have a
    BGE vector; missing or duplicate identities fail loudly.  The vectors are frozen
    data, not model parameters.
    """
    index_path = Path(index_path)
    mappings_path = Path(mappings_path)
    if not index_path.exists():
        raise BGEFeatureError(f"BGE index not found: {index_path}")
    if not mappings_path.exists():
        raise BGEFeatureError(f"mappings not found: {mappings_path}")

    mapping = json.loads(mappings_path.read_text(encoding="utf-8"))
    item2id = mapping.get("item2id")
    if not isinstance(item2id, dict):
        raise BGEFeatureError("mappings artifact has no item2id dictionary")

    with np.load(index_path, allow_pickle=False) as payload:
        fmt = str(payload["format"][0]) if "format" in payload.files else ""
        if fmt != "agentrecx.dense_catalog.v3":
            raise BGEFeatureError(f"unexpected BGE index format {fmt!r}")
        identities = [str(value) for value in payload["identities"].tolist()]
        vectors = np.asarray(payload["vectors"], dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[1] != int(dimension):
        raise BGEFeatureError(
            f"BGE vectors must be N x {dimension}, got {vectors.shape}"
        )
    if len(identities) != vectors.shape[0]:
        raise BGEFeatureError("identities and vectors are not aligned")
    if len(set(identities)) != len(identities):
        raise BGEFeatureError("BGE index contains duplicate identities")
    if not np.isfinite(vectors).all():
        raise BGEFeatureError("BGE vectors contain NaN/Inf")

    matrix = np.zeros((int(num_items) + 1, int(dimension)), dtype=np.float32)
    seen: set[int] = set()
    for identity, vector in zip(identities, vectors):
        item_id = item2id.get(identity)
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            continue
        if not 1 <= item_id <= int(num_items):
            raise BGEFeatureError(
                f"item id {item_id!r} for {identity!r} is outside 1..{num_items}"
            )
        if item_id in seen:
            raise BGEFeatureError(f"duplicate mapped item id {item_id}")
        seen.add(item_id)
        matrix[item_id] = vector

    missing = sorted(set(int(v) for v in item2id.values()) - seen)
    if missing:
        raise BGEFeatureError(
            f"{len(missing)} mapped item(s) have no BGE vector; first {missing[0]}"
        )
    return matrix
