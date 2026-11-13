"""Optional Amazon Reviews 2023 metadata sidecar for the Electronics catalogue.

The sidecar is deliberately *additive*.  It never changes the C4 ``catalog_meta.jsonl``
identity file, the BM25/BGE search documents, or the BGE index.  The catalogue loader
attaches a sidecar record to an already accepted ``ElectronicsProduct``; only then do
the existing executor, Eligibility and P15 evidence layers read the extra fields.

Design rules:

* The sidecar is keyed by the same ``parent_asin``/``item_id`` as the C4 catalogue.
* Search text is unchanged, so retrieval results and the frozen BGE index remain valid.
* Structured metadata attributes are added only when the C4 record did not already
  provide the attribute.  If a mutually exclusive form factor disagrees, the C4
  attribute is removed and the disagreement is recorded; no hard constraint is allowed
  to pass on an ambiguous fact.
* Price/rating are historical crawl-time metadata, never live price/stock.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "DEFAULT_SIDECAR",
    "ENV_MODE",
    "ENV_SIDECAR",
    "MetadataRecord",
    "MetadataSidecar",
    "attach_metadata",
    "metadata_mode",
    "read_metadata_mode",
]

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SIDECAR = REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"
ENV_MODE = "AGENTRECX_METADATA_MODE"
ENV_SIDECAR = "AGENTRECX_METADATA_SIDECAR"
METADATA_MODES = ("original", "enriched")
SIDECAR_SCHEMA = "agentrecx.electronics_metadata_sidecar.v1"


def read_metadata_mode(value: str | None = None) -> str:
    """Return the validated catalogue metadata mode (``original`` by default)."""
    mode = str(value if value is not None else os.environ.get(ENV_MODE, "original")).strip().casefold()
    if not mode:
        mode = "original"
    if mode not in METADATA_MODES:
        raise ValueError(f"{ENV_MODE} must be one of {METADATA_MODES}, got {mode!r}")
    return mode


def metadata_mode(value: str | None = None) -> str:
    """Alias used by callers that want a short, explicit name."""
    return read_metadata_mode(value)


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _clean_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if isinstance(item, str) and item.strip())


def _details(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items() if str(k).strip()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return {str(k): v for k, v in parsed.items() if str(k).strip()} if isinstance(parsed, dict) else {}


def _price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "")
    if not text or text.casefold() in {"none", "null", "nan"}:
        return None
    text = text.replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _number(value: Any, *, integer: bool = False) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if integer:
        return int(number)
    return number


def _detail_lookup(details: Mapping[str, Any], *needles: str) -> str | None:
    """Return the first non-empty details value whose key contains a needle."""
    lowered = [(str(key).casefold(), value) for key, value in details.items()]
    for needle in needles:
        wanted = needle.casefold()
        for key, value in lowered:
            if wanted in key:
                if isinstance(value, (list, tuple)):
                    text = ", ".join(str(part).strip() for part in value if str(part).strip())
                elif isinstance(value, Mapping):
                    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    text = str(value).strip()
                if text:
                    return text
    return None


@dataclass(frozen=True)
class MetadataRecord:
    """One upstream Raw Meta Electronics record for a current catalogue item."""

    item_id: str
    title: str | None
    features: tuple[str, ...]
    description: tuple[str, ...]
    categories: tuple[str, ...]
    store: str | None
    details: Mapping[str, Any]
    price: float | None
    average_rating: float | None
    rating_number: int | None
    source: str = "amazon_reviews_2023:raw_meta_Electronics"
    dataset_version: str = ""
    main_category: str | None = None
    schema_version: str = SIDECAR_SCHEMA

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "MetadataRecord":
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            raise ValueError("metadata row has no item_id")
        return cls(
            item_id=item_id,
            title=_clean_text(row.get("title")),
            features=_clean_list(row.get("features")),
            description=_clean_list(row.get("description")),
            categories=_clean_list(row.get("categories")),
            store=_clean_text(row.get("store")),
            details=_details(row.get("details")),
            price=_price(row.get("price")),
            average_rating=_number(row.get("average_rating")),
            rating_number=_number(row.get("rating_number"), integer=True),
            source=str(row.get("source") or "amazon_reviews_2023:raw_meta_Electronics"),
            dataset_version=str(row.get("dataset_version") or ""),
            main_category=_clean_text(row.get("main_category")),
        )

    @property
    def brand(self) -> str | None:
        return _detail_lookup(self.details, "Brand", "Manufacturer")

    @property
    def material(self) -> str | None:
        return _detail_lookup(self.details, "Material")

    @property
    def size(self) -> str | None:
        return _detail_lookup(self.details, "Size")

    @property
    def product_dimensions(self) -> str | None:
        return _detail_lookup(self.details, "Product Dimensions", "Item Dimensions")

    @property
    def weight(self) -> str | None:
        return _detail_lookup(self.details, "Item Weight", "Package Weight")

    def attribute_facts(self) -> dict[str, dict[str, Any]]:
        """Derive a small, conservative set of attributes from structured details.

        Only directly stated details values are used.  Free-text marketing phrases are
        not turned into hard facts; ``mechanical`` is deliberately omitted because
        "Mechanical Feeling" is ambiguous.  The returned ``evidence`` is the actual
        details line so the P15 span validator can point at real catalog text.
        """
        facts: dict[str, dict[str, Any]] = {}
        connectivity = _detail_lookup(
            self.details, "Connectivity Technology", "Wireless Technology",
            "Network Technology", "Interface",
        )
        combined = (connectivity or "").casefold()
        if combined:
            evidence = f"Connectivity Technology: {connectivity}"
            if any(token in combined for token in ("wireless", "bluetooth", "wi-fi", "wifi")):
                facts["wireless"] = {
                    "status": "present", "evidence": evidence,
                    "source": "metadata:details.Connectivity Technology",
                    "verification": "structured_metadata",
                }
            if any(token in combined for token in ("wired", "3.5mm", "3.5 mm", "aux")):
                facts["wired"] = {
                    "status": "present", "evidence": evidence,
                    "source": "metadata:details.Connectivity Technology",
                    "verification": "structured_metadata",
                }
            if "usb" in combined:
                facts["usb"] = {
                    "status": "present", "evidence": evidence,
                    "source": "metadata:details.Connectivity Technology",
                    "verification": "structured_metadata",
                }
        form = _detail_lookup(self.details, "Form Factor", "Ear Placement")
        form_low = (form or "").casefold()
        if form_low:
            evidence = f"Form Factor: {form}"
            for name, tokens in (
                ("in_ear", ("in ear", "in-ear", "earbud")),
                ("on_ear", ("on ear", "on-ear", "supra")),
                ("over_ear", ("over ear", "over-ear", "circumaural", "around ear")),
            ):
                if any(token in form_low for token in tokens):
                    facts[name] = {
                        "status": "present", "evidence": evidence,
                        "source": "metadata:details.Form Factor",
                        "verification": "structured_metadata",
                    }
        special = _detail_lookup(self.details, "Special Feature")
        special_low = (special or "").casefold()
        if special_low:
            for name, tokens in (
                ("noise_cancelling", ("noise cancelling", "noise cancellation", "active noise", "anc")),
                ("waterproof", ("waterproof", "water resistant", "water-resistant")),
                ("rechargeable", ("rechargeable", "battery")),
                ("foldable", ("foldable", "folding")),
            ):
                if any(token in special_low for token in tokens):
                    facts[name] = {
                        "status": "present", "evidence": f"Special Feature: {special}",
                        "source": "metadata:details.Special Feature",
                        "verification": "structured_metadata",
                    }
        brand = self.brand
        if brand:
            facts["brand"] = {
                "status": "present", "evidence": f"Brand: {brand}",
                "source": "metadata:details.Brand",
                "verification": "structured_metadata", "value": brand,
            }
        material = self.material
        if material:
            facts["material"] = {
                "status": "present", "evidence": f"Material: {material}",
                "source": "metadata:details.Material",
                "verification": "structured_metadata", "value": material,
            }
        return facts

    def summary(self) -> dict[str, Any]:
        return {
            "available": True,
            "schema": self.schema_version,
            "source": self.source,
            "dataset_version": self.dataset_version,
            "title": self.title,
            "store": self.store,
            "price": self.price,
            "price_kind": "historical_crawl_time",
            "average_rating": self.average_rating,
            "rating_number": self.rating_number,
            "categories": list(self.categories),
            "features_count": len(self.features),
            "description_chars": sum(len(item) for item in self.description),
            "details_keys": sorted(self.details),
        }


@dataclass
class MetadataSidecar:
    """Loaded sidecar, keyed by catalogue ``item_id``."""

    records: dict[str, MetadataRecord] = field(default_factory=dict)
    path: str = ""
    schema: str = SIDECAR_SCHEMA
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SIDECAR) -> "MetadataSidecar":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"metadata sidecar not found: {source}")
        records: dict[str, MetadataRecord] = {}
        malformed = 0
        duplicates = 0
        # Iterate the file rather than ``read_text().splitlines()``: JSON strings contain
        # Unicode line/paragraph separators that ``splitlines`` would split incorrectly.
        with source.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    record = MetadataRecord.from_row(row)
                except (json.JSONDecodeError, ValueError, TypeError):
                    malformed += 1
                    continue
                if record.item_id in records:
                    duplicates += 1
                    continue
                records[record.item_id] = record
        return cls(
            records=records,
            path=str(source),
            stats={
                "path": str(source),
                "bytes": source.stat().st_size,
                "records": len(records),
                "malformed_lines": malformed,
                "duplicate_ids": duplicates,
                "dataset_versions": sorted({r.dataset_version for r in records.values() if r.dataset_version}),
            },
        )

    def get(self, item_id: str) -> MetadataRecord | None:
        return self.records.get(str(item_id or "").strip())

    def __len__(self) -> int:
        return len(self.records)

    def coverage(self, ids: Iterable[str]) -> dict[str, Any]:
        wanted = {str(item) for item in ids}
        matched = wanted.intersection(self.records)
        return {
            "catalogue_ids": len(wanted),
            "metadata_records": len(self.records),
            "matched_ids": len(matched),
            "match_rate": round(len(matched) / len(wanted), 6) if wanted else 0.0,
        }


def attach_metadata(catalogue: Any, sidecar: MetadataSidecar) -> dict[str, Any]:
    """Attach sidecar records to existing ``ElectronicsProduct`` objects in place.

    C4 search records, titles and product_type are left untouched.  The only changes are:
    (a) the product's optional ``metadata`` field, (b) non-conflicting structured
    attributes from details, and (c) form-factor conflict clearing.  The C4/BGE search
    text is never modified.
    """
    products = getattr(catalogue, "_products", {})
    form_names = ("over_ear", "on_ear", "in_ear")
    matched = 0
    attrs_added = 0
    conflicts: list[dict[str, Any]] = []
    for product in products.values():
        record = sidecar.get(getattr(product, "item_id", ""))
        if record is None:
            continue
        matched += 1
        product.metadata = record
        facts = record.attribute_facts()
        product.metadata_facts = facts
        existing = dict(getattr(product, "attributes", {}) or {})
        existing_forms = {name for name in form_names if name in existing}
        meta_forms = {name for name in form_names if name in facts}
        if existing_forms and meta_forms and not (existing_forms & meta_forms):
            # Mutually exclusive form factors disagree: do not let either side decide a
            # hard constraint.  Keep the C4 record for audit but clear the derived view.
            for name in existing_forms:
                existing.pop(name, None)
            product.attributes = existing
            product.form_factor = None
            product.metadata_conflicts = {
                "form_factor": {
                    "c4": sorted(existing_forms), "metadata": sorted(meta_forms),
                    "resolution": "unknown",
                }
            }
            conflicts.append({"item_id": product.item_id, "kind": "form_factor",
                              "c4": sorted(existing_forms), "metadata": sorted(meta_forms)})
            continue
        changed = False
        for name, fact in facts.items():
            if name in existing:
                continue
            # A metadata attribute that directly contradicts an existing C4
            # connectivity fact is retained separately as a conflict, not used as C4
            # evidence.  wireless/wired can both be true on multi-interface products.
            if name in {"wireless", "wired"} and (
                (name == "wireless" and "wired" in existing) or
                (name == "wired" and "wireless" in existing)
            ):
                product.metadata_conflicts.setdefault("connectivity", {})[name] = fact.get("evidence")
                continue
            existing[name] = fact
            changed = True
            attrs_added += 1
        product.attributes = existing
        if product.form_factor is None:
            for name in form_names:
                if name in existing and name in facts:
                    product.form_factor = name
                    break
    return {
        "mode": "enriched",
        "sidecar": str(sidecar.path),
        "matched_products": matched,
        "metadata_attributes_added": attrs_added,
        "conflicts_count": len(conflicts),
        "conflicts": conflicts[:20],
        "sidecar_stats": dict(sidecar.stats),
    }
