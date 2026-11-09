"""P14-E thin Electronics catalogue adapter over the real P13-R C4 metadata.

This is deliberately the *thinnest* adapter (P14-E D1): it reuses the accepted
``CatalogSearchSource`` (BM25) and ``ProductMetadata`` contracts, and only adds:

* identity validation of the P13-R ``catalog_meta.jsonl`` file;
* a text-evidence attribute extractor (no fabricated brand/price/size/material);
* a read-only ``inspect`` view with explicit evidence spans and UNKNOWN states.

Nothing here invents a product fact.  ``text`` is the only source of attributes, and
every extracted attribute keeps the matched evidence span.  Price/brand/material/stock
are **UNKNOWN** in this catalogue because the source file does not contain them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from recommendation.catalog.schemas import ProductMetadata
from recommendation.control.catalog_search import CatalogSearchSource, CatalogSearchResult

__all__ = [
    "ATTR_NAMES",
    "ELECTRONICS_NAMESPACE",
    "PRODUCT_TYPES",
    "ElectronicsCatalogue",
    "ElectronicsProduct",
    "extract_attributes",
]

ELECTRONICS_NAMESPACE = "amazon_electronics_c4"
CATALOG_SOURCE = "data/p13r/electronics/catalog_meta.jsonl"

PRODUCT_TYPES: dict[str, str] = {
    "headphones": r"(?:headphones?|headset)\b",
    "earbuds": r"(?:earbuds?|earphones?|in-ear)\b",
    "keyboard": r"(?:keyboards?)\b",
    "mouse": r"(?:mouse|mice|trackball)\b",
    "speaker": r"(?:speakers?|soundbar)\b",
}

#: evidence-based attributes; absence in the text is "unknown", never "false".
ATTR_PATTERNS: dict[str, str] = {
    "wireless": r"(?:wireless|bluetooth|2\.4g)",
    "wired": r"(?:wired|3\.5mm|3\.5 mm|aux cable|audio cable)",
    "noise_cancelling": r"(?:noise[- ]?cancel|anc\b|active noise)",
    "over_ear": r"(?:over[- ]?ear|over the ear|around[- ]ear|circumaural)",
    "on_ear": r"(?:on[- ]?ear|supra[- ]?aural)",
    "in_ear": r"(?:in[- ]?ear|earbud|earphone|canal)",
    "microphone": r"(?:microphone|\bmic\b|built-in mic)",
    "gaming": r"(?:gaming|gamer)",
    "waterproof": r"(?:waterproof|water[- ]resistant|sweatproof|sweat[- ]proof|ipx[4-8])",
    "foldable": r"(?:foldable|folding|collapsible)",
    "mechanical": r"(?:mechanical)",
    "rgb": r"(?:\brgb\b|backlit|backlight)",
    "rechargeable": r"(?:rechargeable|recharge|battery)",
    "multidevice": r"(?:multi[- ]device|multipoint|multi[- ]point)",
    "usb": r"(?:\busb\b|usb-c|type-c|type c)",
    "silent": r"(?:silent|quiet|low[- ]profile|whisper)",
    "ergonomic": r"(?:ergonomic)",
}
ATTR_NAMES = tuple(ATTR_PATTERNS)
FORM_ATTRS = ("over_ear", "on_ear", "in_ear")
BOOK_MARKER = re.compile(
    r"\b(paperback|hardcover|novel|textbook|isbn|book review|about the author|"
    r"publisher|edition|memoir|fiction)\b",
    re.I,
)
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")

UNKNOWN_FACTS = {
    "brand": "catalog_meta.jsonl has no brand field",
    "price": "catalog_meta.jsonl has no price field",
    "size": "catalog_meta.jsonl has no size field",
    "material": "catalog_meta.jsonl has no material field",
    "stock": "catalog_meta.jsonl has no stock field",
}


@dataclass
class ElectronicsProduct:
    """One accepted P13-R catalogue record plus its text-evidence attributes."""

    item_id: str
    namespaced_id: str
    title: str
    text: str
    product_type: str | None
    attributes: dict[str, dict[str, Any]] = field(default_factory=dict)
    form_factor: str | None = None
    record: ProductMetadata | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "namespaced_id": self.namespaced_id,
            "title": self.title,
            "product_type": self.product_type,
            "form_factor": self.form_factor,
            "attributes": self.attributes,
            "text_chars": len(self.text),
            "source": CATALOG_SOURCE,
        }


def _first_sentence(text: str, limit: int = 220) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return ""
    match = re.search(r"(?<=[.!?])\s", clean)
    title = clean[: match.start()] if match else clean
    return title[:limit].strip()


def _type_for_title(title: str) -> str | None:
    low = title.lower()
    for name, pattern in PRODUCT_TYPES.items():
        if re.search(pattern, low):
            return name
    return None


def _evidence(text: str, match: re.Match[str], width: int = 60) -> str:
    start = max(0, match.start() - width)
    end = min(len(text), match.end() + width)
    return " ".join(text[start:end].split())


def extract_attributes(text: str, title: str) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Extract only attributes for which the catalogue text has direct evidence."""
    haystack = f"{title}. {text}".strip()
    low = haystack.lower()
    attributes: dict[str, dict[str, Any]] = {}
    for name, pattern in ATTR_PATTERNS.items():
        match = re.search(pattern, low)
        if match:
            attributes[name] = {
                "status": "present",
                "evidence": _evidence(haystack, match),
                "source": "catalog_meta:text",
            }
    form_factor = None
    for name in FORM_ATTRS:
        if name in attributes:
            form_factor = name
            break
    return attributes, form_factor


class _View:
    """Minimal ``CatalogTextLike`` view for the accepted BM25 source."""

    def __init__(self, records: Mapping[str, ProductMetadata]) -> None:
        self.records = dict(records)


class ElectronicsCatalogue:
    """Read-only validated view of the P13-R Electronics catalogue."""

    def __init__(self, products: Mapping[str, ElectronicsProduct], stats: Mapping[str, Any]) -> None:
        self._products = dict(products)
        self._stats = dict(stats)
        self._view = _View({product.namespaced_id: product.record for product in self._products.values()})
        self._search_source: CatalogSearchSource | None = None

    # -- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path = CATALOG_SOURCE) -> "ElectronicsCatalogue":
        path = Path(path)
        stats: dict[str, Any] = {
            "source": CATALOG_SOURCE,
            "rows": 0,
            "accepted": 0,
            "quarantined": 0,
            "quarantine_reasons": {},
            "type_counts": {},
            "typed": 0,
            "untyped": 0,
            "attribute_presence": {},
            "duplicate_text_values": 0,
        }
        products: dict[str, ElectronicsProduct] = {}
        seen_ids: set[str] = set()
        text_values: dict[str, str] = {}
        quarantined = 0

        def quarantine(reason: str) -> None:
            nonlocal quarantined
            quarantined += 1
            stats["quarantine_reasons"][reason] = stats["quarantine_reasons"].get(reason, 0) + 1

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["rows"] += 1
                row = json.loads(line)
                item_id = str(row.get("item_id") or "").strip()
                text = " ".join(str(row.get("text") or "").split())
                if not item_id or not ASIN_RE.match(item_id):
                    quarantine("invalid_item_id")
                    continue
                if item_id in seen_ids:
                    quarantine("duplicate_item_id")
                    continue
                seen_ids.add(item_id)
                if len(text) < 20:
                    quarantine("empty_or_truncated_text")
                    continue
                title = _first_sentence(text)
                if not title:
                    quarantine("missing_title")
                    continue
                if BOOK_MARKER.search(title):
                    quarantine("book_or_media_title")
                    continue
                product_type = _type_for_title(title)
                attributes, form_factor = extract_attributes(text, title)
                if text in text_values:
                    stats["duplicate_text_values"] += 1
                text_values.setdefault(text, item_id)
                namespaced = f"{ELECTRONICS_NAMESPACE}:{item_id}"
                record = ProductMetadata(
                    parent_asin=namespaced,
                    title=title,
                    main_category="Electronics",
                    categories=("Electronics",),
                    description=(text,),
                    source=CATALOG_SOURCE,
                )
                product = ElectronicsProduct(
                    item_id=item_id,
                    namespaced_id=namespaced,
                    title=title,
                    text=text,
                    product_type=product_type,
                    attributes=attributes,
                    form_factor=form_factor,
                    record=record,
                )
                products.setdefault(item_id, product)
                if product_type is None:
                    stats["untyped"] += 1
                else:
                    stats["typed"] += 1
                    stats["type_counts"][product_type] = stats["type_counts"].get(product_type, 0) + 1
                for name in attributes:
                    stats["attribute_presence"][name] = stats["attribute_presence"].get(name, 0) + 1

        stats["accepted"] = len(products)
        stats["quarantined"] = quarantined
        stats["type_counts"] = dict(sorted(stats["type_counts"].items()))
        stats["attribute_presence"] = dict(sorted(stats["attribute_presence"].items()))
        return cls(products, stats)

    # -- metadata ---------------------------------------------------------- #
    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    @property
    def records(self) -> Mapping[str, ProductMetadata]:
        return self._view.records

    @property
    def namespace(self) -> str:
        return ELECTRONICS_NAMESPACE

    @property
    def catalogue_size(self) -> int:
        return len(self._products)

    @property
    def search_source(self) -> CatalogSearchSource:
        if self._search_source is None:
            self._search_source = CatalogSearchSource(self._view, mode="bm25")
        return self._search_source

    def by_type(self, product_type: str) -> tuple[ElectronicsProduct, ...]:
        return tuple(p for p in self._products.values() if p.product_type == product_type)

    def __contains__(self, identity: object) -> bool:
        if not isinstance(identity, str):
            return False
        return identity in self.records or self._products.get(identity) is not None

    def get(self, identity: str) -> ElectronicsProduct | None:
        text = str(identity or "").strip()
        if ":" in text:
            namespace, item_id = text.split(":", 1)
            if namespace != ELECTRONICS_NAMESPACE:
                return None
            return self._products.get(item_id)
        return self._products.get(text)

    # -- tools ------------------------------------------------------------- #
    def search(self, query: str, *, limit: int = 10) -> CatalogSearchResult:
        return self.search_source.search(str(query or ""), limit=limit)

    def inspect(self, item_id: str, requested: Iterable[str] | None = None) -> dict[str, Any]:
        product = self.get(item_id)
        requested_names = list(requested) if requested is not None else []
        if product is None:
            return {
                "status": "missing",
                "product_id": str(item_id),
                "facts": {},
                "unknown_fields": requested_names,
                "evidence": [],
            }
        facts: dict[str, dict[str, Any]] = {}
        unknown: list[str] = []
        unsupported: list[str] = []
        for name in requested_names:
            if name in product.attributes:
                fact = dict(product.attributes[name])
                fact.update({"value": True, "value_type": "boolean", "verification": "text_evidence"})
                facts[name] = fact
            elif name in ATTR_NAMES or name in FORM_ATTRS:
                facts[name] = {
                    "status": "unknown",
                    "value": None,
                    "source": CATALOG_SOURCE,
                    "reason": "no lexical evidence in catalogue text",
                    "verification": "absent",
                }
                unknown.append(name)
            elif name in UNKNOWN_FACTS:
                facts[name] = {
                    "status": "unknown",
                    "value": None,
                    "source": CATALOG_SOURCE,
                    "reason": UNKNOWN_FACTS[name],
                    "verification": "absent",
                }
                unknown.append(name)
            else:
                facts[name] = {"status": "unsupported", "value": None}
                unsupported.append(name)
        return {
            "status": "ok",
            "product_id": product.item_id,
            "namespaced_id": product.namespaced_id,
            "product_type": product.product_type,
            "title": product.title,
            "form_factor": product.form_factor,
            "facts": facts,
            "unknown_fields": unknown,
            "unsupported_fields": unsupported,
            "source": CATALOG_SOURCE,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "namespace": ELECTRONICS_NAMESPACE,
            "catalogue_size": self.catalogue_size,
            "source": CATALOG_SOURCE,
            "limited_catalogue": True,
            "stats": self.stats,
        }
