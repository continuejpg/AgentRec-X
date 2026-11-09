"""Real BGE-base Query-to-Product dense catalogue retrieval.

The only supported encoder is a local ``BAAI/bge-base-en-v1.5`` Sentence-Transformers
model.  Documents use catalogue text with no query instruction; queries use the official
English retrieval instruction and both sides are normalized.  There is deliberately no
hashed-TF-IDF or other fake dense fallback: a missing model is a configuration error.

The index is an exact NumPy matrix (156,746 x 768 float32) keyed by canonical
``parent_asin`` identities.  Loading validates the artifact's internal identity/vector
alignment and, when the current catalogue is supplied, its identity and text fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "BGE_EMBEDDING_DIM",
    "BGE_MODEL_ID",
    "BGE_QUERY_INSTRUCTION",
    "DenseCatalogIndex",
    "DenseHit",
    "build_text",
    "catalogue_fingerprint",
    "identity_fingerprint",
    "mapping_fingerprint",
]

INDEX_FORMAT = "agentrecx.dense_catalog.v3"
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
BGE_EMBEDDING_DIM = 768
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TEXT_BUILD_VERSION = "agentrecx.catalog_text.v1"
TEXT_MAX_CHARS = 2048
MAX_SEQ_LENGTH = 512


class DenseHit:
    __slots__ = ("parent_asin", "rank", "score", "title")

    def __init__(self, parent_asin: str, rank: int, score: float, title: str | None) -> None:
        self.parent_asin = parent_asin
        self.rank = rank
        self.score = score
        self.title = title

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "rank": self.rank,
            "score": self.score,
            "title": self.title,
        }


def build_text(record: Any, *, max_chars: int = TEXT_MAX_CHARS) -> str:
    """Stable trusted catalogue text; no user history, labels or inferred attributes."""
    parts: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, (tuple, list)):
            values = tuple(value)
        else:
            return
        for item in values:
            text = " ".join(str(item or "").split())
            if not text:
                continue
            folded = text.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            parts.append(text)

    add(getattr(record, "title", None))
    add(getattr(record, "store", None))
    for field in ("main_category", "categories", "features", "description"):
        add(getattr(record, field, None))
    text = " ".join(parts)
    return text[:max_chars]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_fingerprint(identities: Sequence[str]) -> str:
    return _sha256_text("\n".join(str(item) for item in identities))


def catalogue_fingerprint(records: Mapping[str, Any]) -> str:
    """Fingerprint of current identity + trusted dense text (not just the ID column)."""
    lines = [
        f"{identity}\t{build_text(records[identity])}"
        for identity in sorted(records)
    ]
    return _sha256_text("\n".join(lines))


def mapping_fingerprint(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DenseCatalogIndex:
    """Exact normalized BGE-base vectors keyed by canonical ``parent_asin``."""

    def __init__(
        self,
        *,
        identities: Sequence[str],
        vectors: np.ndarray,
        titles: Sequence[str | None],
        model_id: str,
        model_revision: str,
        model_path: str | None,
        query_instruction: str,
        text_build_version: str,
        catalogue_fingerprint_value: str,
        identity_fingerprint_value: str,
        mapping_fingerprint_value: str | None,
        text_max_chars: int,
        max_seq_length: int,
        built_at: str,
        encoder: Any | None = None,
    ) -> None:
        self.identities = tuple(str(item) for item in identities)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.titles = tuple(titles)
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.model_path = str(model_path) if model_path else ""
        self.query_instruction = str(query_instruction)
        self.text_build_version = str(text_build_version)
        self.catalogue_fingerprint = str(catalogue_fingerprint_value)
        self.identity_fingerprint = str(identity_fingerprint_value)
        self.mapping_fingerprint = mapping_fingerprint_value
        self.text_max_chars = int(text_max_chars)
        self.max_seq_length = int(max_seq_length)
        self.built_at = str(built_at)
        self.encoder = encoder

        if self.model_id != BGE_MODEL_ID:
            raise ValueError(f"unsupported BGE model id: {self.model_id!r}")
        if self.vectors.ndim != 2 or self.vectors.shape[1] != BGE_EMBEDDING_DIM:
            raise ValueError(
                f"BGE index must be N x {BGE_EMBEDDING_DIM}, got {self.vectors.shape}"
            )
        if self.vectors.shape[0] != len(self.identities):
            raise ValueError("dense index vectors do not align with identities")
        if len(self.titles) != len(self.identities):
            raise ValueError("dense index titles do not align with identities")
        if len(set(self.identities)) != len(self.identities):
            raise ValueError("dense index contains duplicate identities")
        if not np.isfinite(self.vectors).all():
            raise ValueError("dense index contains NaN/Inf vectors")
        norms = np.linalg.norm(self.vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("dense index vectors are not normalized")
        if identity_fingerprint(self.identities) != self.identity_fingerprint:
            raise ValueError("dense index identity fingerprint mismatch")
        if not self.model_revision:
            raise ValueError("dense index is missing model_revision")
        if not self.query_instruction:
            raise ValueError("dense index is missing query_instruction")

    @property
    def size(self) -> int:
        return len(self.identities)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def _load_encoder(self) -> Any:
        if self.encoder is not None:
            return self.encoder
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            self.model_path or self.model_id,
            device="cpu",
        )
        model.max_seq_length = self.max_seq_length
        self.encoder = model
        return model

    def validate_catalogue(
        self,
        records: Mapping[str, Any],
        *,
        mapping_sha: str | None = None,
    ) -> None:
        """Reject a dense index that does not belong to the current catalogue."""
        current = tuple(sorted(str(item) for item in records))
        if current != tuple(sorted(self.identities)):
            raise ValueError(
                "dense index identities do not match the current catalogue"
            )
        current_catalogue = catalogue_fingerprint(records)
        if current_catalogue != self.catalogue_fingerprint:
            raise ValueError(
                "dense index catalogue fingerprint does not match the current catalogue"
            )
        if mapping_sha is not None and self.mapping_fingerprint is not None:
            if mapping_sha != self.mapping_fingerprint:
                raise ValueError(
                    "dense index mapping fingerprint does not match the current mapping"
                )

    def search(self, query: str, *, limit: int = 10) -> list[DenseHit]:
        if limit < 1 or not self.identities:
            return []
        model = self._load_encoder()
        text = BGE_QUERY_INSTRUCTION + str(query or "")
        query_vector = np.asarray(
            model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                device=str(getattr(model, "device", "cpu")),
            )[0],
            dtype=np.float32,
        )
        if query_vector.shape[0] != BGE_EMBEDDING_DIM:
            raise ValueError("query embedding dimension does not match the BGE index")
        scores = self.vectors @ query_vector
        order = np.argsort(-scores, kind="stable")[: int(limit)]
        return [
            DenseHit(
                parent_asin=self.identities[int(position)],
                rank=rank,
                score=float(scores[int(position)]),
                title=self.titles[int(position)],
            )
            for rank, position in enumerate(order, start=1)
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "index_format": INDEX_FORMAT,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "embedding_dim": self.dim,
            "normalize_embeddings": True,
            "similarity": "cosine_dot_product",
            "query_instruction": self.query_instruction,
            "text_build_version": self.text_build_version,
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "identity_fingerprint": self.identity_fingerprint,
            "mapping_fingerprint": self.mapping_fingerprint,
            "item_count": self.size,
            "text_max_chars": self.text_max_chars,
            "max_seq_length": self.max_seq_length,
            "built_at": self.built_at,
        }
        np.savez(
            target,
            format=np.asarray([INDEX_FORMAT], dtype=str),
            metadata=np.asarray([json.dumps(metadata, ensure_ascii=False)], dtype=str),
            identities=np.asarray(self.identities, dtype=str),
            vectors=self.vectors,
            titles=np.asarray([title or "" for title in self.titles], dtype=str),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model_path: str | None = None,
        catalogue_records: Mapping[str, Any] | None = None,
        mapping_sha: str | None = None,
    ) -> "DenseCatalogIndex":
        data = np.load(Path(path), allow_pickle=False)
        if "format" not in data.files or str(data["format"][0]) != INDEX_FORMAT:
            raise ValueError("dense index format is missing or unsupported")
        metadata = json.loads(str(data["metadata"][0]))
        identities = tuple(str(item) for item in data["identities"])
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        titles = tuple(str(item) or None for item in data["titles"])
        index = cls(
            identities=identities,
            vectors=vectors,
            titles=titles,
            model_id=str(metadata["model_id"]),
            model_revision=str(metadata["model_revision"]),
            model_path=(model_path or str(metadata.get("model_path") or "")),
            query_instruction=str(metadata["query_instruction"]),
            text_build_version=str(metadata["text_build_version"]),
            catalogue_fingerprint_value=str(metadata["catalogue_fingerprint"]),
            identity_fingerprint_value=str(metadata["identity_fingerprint"]),
            mapping_fingerprint_value=metadata.get("mapping_fingerprint"),
            text_max_chars=int(metadata["text_max_chars"]),
            max_seq_length=int(metadata["max_seq_length"]),
            built_at=str(metadata["built_at"]),
        )
        if catalogue_records is not None:
            index.validate_catalogue(catalogue_records, mapping_sha=mapping_sha)
        # Fail at load time, not at the first user query: a missing/incompatible local
        # BGE model must never be silently treated as a working dense retriever.
        index._load_encoder()
        return index

    @classmethod
    def build(
        cls,
        records: Mapping[str, Any],
        *,
        model_path: str,
        model_revision: str,
        mapping_sha: str | None = None,
        text_max_chars: int = TEXT_MAX_CHARS,
        max_seq_length: int = MAX_SEQ_LENGTH,
        device: str = "cuda",
        batch_size: int = 64,
    ) -> "DenseCatalogIndex":
        if not model_path:
            raise ValueError("building a BGE index requires a local model_path")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_path, device=device)
        model.max_seq_length = int(max_seq_length)
        ordered = tuple(sorted(str(item) for item in records))
        texts = [build_text(records[identity], max_chars=text_max_chars) for identity in ordered]
        vectors = np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=int(batch_size),
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.shape != (len(ordered), BGE_EMBEDDING_DIM):
            raise ValueError(
                f"BGE encoder returned {vectors.shape}, expected "
                f"({len(ordered)}, {BGE_EMBEDDING_DIM})"
            )
        titles = tuple(
            (str(getattr(records[identity], "title", "") or "") or None)
            for identity in ordered
        )
        return cls(
            identities=ordered,
            vectors=vectors,
            titles=titles,
            model_id=BGE_MODEL_ID,
            model_revision=model_revision,
            model_path=model_path,
            query_instruction=BGE_QUERY_INSTRUCTION,
            text_build_version=TEXT_BUILD_VERSION,
            catalogue_fingerprint_value=catalogue_fingerprint(records),
            identity_fingerprint_value=identity_fingerprint(ordered),
            mapping_fingerprint_value=mapping_sha,
            text_max_chars=text_max_chars,
            max_seq_length=max_seq_length,
            built_at=datetime.now(timezone.utc).isoformat(),
            encoder=model,
        )
