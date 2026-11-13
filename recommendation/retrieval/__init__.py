"""Versioned retrieval documents and a BM25 source for Advanced Retrieval v2.

The C4 catalogue and the original BGE index are never overwritten.  This builder creates
an additive document view from the existing catalogue plus the metadata sidecar.  It
keeps only title/features/description and directly stated structured specifications;
historical price/rating are intentionally omitted from retrieval text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE
from recommendation.control.catalog_search import CatalogSearchSource

RETRIEVAL_DOC_SCHEMA = "agentrecx.retrieval_doc.v2"
RETRIEVAL_DOC_VERSION = "electronics_retrieval_docs_v2"
RETRIEVAL_DOC_MAX_CHARS = 1800
DEFAULT_DOCS_PATH = Path("data/enrichment/electronics_retrieval_docs_v2.jsonl")
DEFAULT_MANIFEST_PATH = Path("runs/advanced_retrieval_v2/retrieval_docs_v2_manifest.json")
_ALLOWED_DETAIL_KEYS = (
    "Connectivity Technology", "Wireless Technology", "Network Technology",
    "Form Factor", "Ear Placement", "Special Feature", "Material",
)
_CONFLICT_FORM_ATTRS = ("over_ear", "on_ear", "in_ear")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _dedupe(values: Iterable[str], *, seen: set[str], limit: int = 12) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _clean(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RetrievalDocument:
    item_id: str
    title: str
    text: str
    sections: dict[str, str]
    source_fields: tuple[str, ...]
    metadata_available: bool
    metadata_version: str | None
    conflicts: dict[str, Any] = field(default_factory=dict)
    source_record: str = ""
    builder_version: str = RETRIEVAL_DOC_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RETRIEVAL_DOC_SCHEMA,
            "item_id": self.item_id,
            "namespace": ELECTRONICS_NAMESPACE,
            "title": self.title,
            "text": self.text,
            "sections": dict(self.sections),
            "source_fields": list(self.source_fields),
            "metadata_available": self.metadata_available,
            "metadata_version": self.metadata_version,
            "conflicts": dict(self.conflicts),
            "source_record": self.source_record,
            "builder_version": self.builder_version,
        }


def build_retrieval_document(product: Any) -> RetrievalDocument:
    """Build one enriched retrieval document from an existing ElectronicsProduct.

    Conflicting form factors are omitted rather than asserted.  Structured details are
    copied only from an allow-list and are never used to rewrite the C4 text.
    """
    meta = getattr(product, "metadata", None)
    record = getattr(product, "record", None)
    conflicts = dict(getattr(product, "metadata_conflicts", {}) or {})
    title = _clean(getattr(product, "title", None) or getattr(meta, "title", None))
    product_type = _clean(getattr(product, "product_type", None))
    sections: dict[str, str] = {}
    sources: set[str] = {"c4_catalog:text"}
    seen: set[str] = set()
    if title:
        sections["Title"] = title
    if product_type:
        sections["Product Type"] = product_type
        sources.add("derived:product_type")
    facts = dict(getattr(product, "metadata_facts", {}) or {})
    c4_attributes = {
        name for name, fact in (getattr(product, "attributes", {}) or {}).items()
        if isinstance(fact, dict) and fact.get("status") == "present"
    }
    trusted_attributes = sorted(
        c4_attributes | {
            name for name, fact in facts.items()
            if isinstance(fact, dict)
            and fact.get("status") == "present"
            and fact.get("verification") == "structured_metadata"
        }
    )
    if trusted_attributes:
        if "form_factor" in conflicts:
            trusted_attributes = [
                name for name in trusted_attributes if name not in _CONFLICT_FORM_ATTRS
            ]
        if trusted_attributes:
            sections["Attributes"] = ", ".join(trusted_attributes)
            sources.add("c4_catalog:attributes")
            if any(name in facts for name in trusted_attributes):
                sources.add("metadata:attribute_facts")
    form = _clean(getattr(product, "form_factor", None))
    if "form_factor" in conflicts:
        # The source disagreement is explicit; never put either value in the retrieval
        # document as a confident specification.
        sources.add("metadata:conflict_omitted")
    elif form:
        sections["Form Factor"] = form
    if meta is not None:
        sources.add("metadata:sidecar")
        if getattr(meta, "dataset_version", None):
            sources.add("metadata:dataset_version")
        detail_parts: list[str] = []
        details = getattr(meta, "details", {}) or {}
        for wanted in _ALLOWED_DETAIL_KEYS:
            if wanted.casefold() in {str(key).casefold() for key in conflicts}:
                continue
            if wanted.casefold() in {"form factor", "ear placement"} and "form_factor" in conflicts:
                continue
            for key, value in details.items():
                if str(key).casefold() == wanted.casefold():
                    text = _clean(value)
                    if text:
                        detail_parts.append(f"{key}: {text}")
                    break
        if detail_parts:
            sections["Specifications"] = " | ".join(detail_parts)
            sources.add("metadata:details")
        features = _dedupe(getattr(meta, "features", ()) or (), seen=seen)
        if features:
            sections["Features"] = "\n".join(f"- {item}" for item in features)
            sources.add("metadata:features")
        descriptions = _dedupe(getattr(meta, "description", ()) or (), seen=seen, limit=3)
        if descriptions:
            sections["Description"] = "\n".join(descriptions)
            sources.add("metadata:description")
    # C4 text remains the fallback/context when metadata has no feature or description.
    if "Features" not in sections and "Description" not in sections:
        c4_description = _dedupe(
            (getattr(record, "description", ()) or (getattr(product, "text", ""),)),
            seen=seen, limit=2,
        )
        if c4_description:
            sections["Description"] = "\n".join(c4_description)
            sources.add("c4_catalog:description")
    # Build the final text in a stable order and cap it.  Empty sections are omitted.
    ordered = []
    for heading in ("Title", "Product Type", "Attributes", "Form Factor",
                    "Specifications", "Features", "Description"):
        body = sections.get(heading)
        if body:
            ordered.append(f"{heading}: {body}" if heading != "Features" else f"Features:\n{body}")
    text = "\n".join(ordered)[:RETRIEVAL_DOC_MAX_CHARS]
    return RetrievalDocument(
        item_id=str(product.item_id),
        title=title,
        text=text,
        sections=sections,
        source_fields=tuple(sorted(sources)),
        metadata_available=meta is not None,
        metadata_version=(getattr(meta, "dataset_version", None) if meta is not None else None),
        conflicts=conflicts,
        source_record=str(getattr(record, "source", "data/p13r/electronics/catalog_meta.jsonl")),
    )


def build_enriched_documents(catalogue: Any) -> Iterator[RetrievalDocument]:
    for identity in sorted(
        product.item_id for product in getattr(catalogue, "products", ()) or ()
    ):
        product = catalogue.get(identity)
        if product is not None:
            yield build_retrieval_document(product)


def write_retrieval_documents(catalogue: Any, output: Path) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = empty = conflict_count = 0
    lengths: list[int] = []
    with output.open("w", encoding="utf-8") as handle:
        for document in build_enriched_documents(catalogue):
            payload = document.as_dict()
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            lengths.append(len(document.text))
            if not document.text:
                empty += 1
            if document.conflicts:
                conflict_count += 1
    return {
        "count": count, "empty_text": empty, "conflict_documents": conflict_count,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
    }


def load_retrieval_documents(path: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item = str(row.get("item_id") or "")
            if not item:
                raise ValueError("retrieval document has no item_id")
            if item in documents:
                raise ValueError(f"duplicate retrieval document: {item}")
            documents[item] = row
    return documents


class _RetrievalView:
    """Minimal CatalogTextLike view over enriched document text."""

    def __init__(self, documents: Mapping[str, Mapping[str, Any]]) -> None:
        self.records = {
            f"{ELECTRONICS_NAMESPACE}:{item_id}": SimpleNamespace(
                title=row.get("title"),
                features=(row.get("text") or "",),
                description=(),
                categories=(),
                main_category=None,
                store=None,
            )
            for item_id, row in documents.items()
        }


class EnrichedBM25Source(CatalogSearchSource):
    """BM25 over versioned enriched retrieval documents; no dense encoder is loaded."""

    def __init__(self, documents: Mapping[str, Mapping[str, Any]]) -> None:
        self.documents = dict(documents)
        super().__init__(_RetrievalView(self.documents), fields=("title", "features"),
                         mode="bm25")

    @property
    def name(self) -> str:
        return "catalog_search_enriched_v2"

    @property
    def score_kind(self) -> str:
        return "bm25_enriched_v2"


def load_enriched_bm25(path: Path) -> EnrichedBM25Source:
    return EnrichedBM25Source(load_retrieval_documents(path))


def build_manifest(*, catalogue_path: Path, sidecar_path: Path, docs_path: Path,
                   docs_stats: Mapping[str, Any], catalogue: Any) -> dict[str, Any]:
    return {
        "schema": "agentrecx.advanced_retrieval.v2_docs_manifest.v1",
        "builder_version": RETRIEVAL_DOC_VERSION,
        "doc_schema": RETRIEVAL_DOC_SCHEMA,
        "catalogue_path": str(catalogue_path),
        "catalogue_sha256": _sha256_file(Path(catalogue_path)),
        "metadata_sidecar_path": str(sidecar_path),
        "metadata_sidecar_sha256": _sha256_file(Path(sidecar_path)),
        "documents_path": str(docs_path),
        "documents_sha256": _sha256_file(Path(docs_path)),
        "documents": int(docs_stats.get("count") or 0),
        "empty_text": int(docs_stats.get("empty_text") or 0),
        "conflict_documents": int(docs_stats.get("conflict_documents") or 0),
        "char_stats": {key: docs_stats.get(key) for key in
                       ("min_chars", "max_chars", "mean_chars")},
        "identity_namespace": ELECTRONICS_NAMESPACE,
        "catalogue_identity_fingerprint": hashlib.sha256(
            "\n".join(sorted(product.item_id for product in catalogue.products)).encode("utf-8")
        ).hexdigest(),
        "rules": [
            "C4 catalogue and BGE index are not modified",
            "historical price/rating are omitted from retrieval text",
            "conflicting form factor is omitted, never asserted",
            "features/descriptions are deduplicated and length-capped",
            "source fields are retained per document",
        ],
    }
