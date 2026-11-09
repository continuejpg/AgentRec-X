"""Real limited Shopping Companion catalogue on the accepted catalogue contract.

The official release ships ``products.jsonl`` (~20 GiB, 1,298,797 products).  P14-A
does **not** materialise that file: a contiguous real sample is projected by
``experiments/p14a_build_catalogue.py`` into ``docs.jsonl`` (see
``data/p14a/catalogue/manifest.json``), and this module loads that projection.

Two things are deliberately reused instead of reimplemented:

* product records are normalised into the accepted
  :class:`recommendation.catalog.schemas.ProductMetadata` contract, so the
  existing catalogue readers and the grounded constraint reasoner accept them
  without modification;
* retrieval is the existing
  :class:`recommendation.control.catalog_search.CatalogSearchSource` BM25
  implementation, i.e. exactly the same tokenizer, BM25 constants and IDF that the
  rest of the repository uses.

What is new here is only the *translation*: Shopping Companion fields ->
``ProductMetadata`` fields, plus the explicit ``shopping_companion:<product_id>``
identity.  The catalogue is a limited sample and every response says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from recommendation.catalog.schemas import ProductMetadata
from recommendation.control.catalog_search import CatalogSearchResult, CatalogSearchSource

from .namespace import (
    SHOPPING_COMPANION_NAMESPACE,
    ShoppingCompanionIdentityError,
    namespaced_id,
    raw_product_id,
    split_namespaced_id,
)

__all__ = [
    "CATALOGUE_SCHEMA",
    "SEARCHABLE_FIELDS",
    "ShoppingCompanionCatalogue",
    "ShoppingCompanionProduct",
    "product_from_doc",
]

#: Schema tag written into the projected docs file's manifest (not on each line).
CATALOGUE_SCHEMA = "agentrecx.p14a.catalogue.v1"

#: Fields the existing BM25 source searches; identical to the accepted catalogue
#: contract's searchable surface.  ``ProductMetadata`` supplies all of them.
SEARCHABLE_FIELDS: tuple[str, ...] = (
    "title",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
)

#: Provenance label attached to every normalised record.
_RECORD_SOURCE = "shopping_companion:products.jsonl"

#: Hard cap on the description stored in an agent-facing product view (the full
#: field is kept in the catalogue record; the tool truncates for display only).
_DESCRIPTION_CHARS = 1200


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _clean_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    return tuple(items)


def _attribute_items(value: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Normalise the official ``attributes`` mapping (``key -> [values]``)."""
    if not isinstance(value, Mapping):
        return ()
    items: list[tuple[str, tuple[str, ...]]] = []
    for key, raw in value.items():
        text_key = _clean_text(key)
        values = _clean_list(raw)
        if text_key is None or not values:
            continue
        items.append((text_key, values))
    return tuple(items)


def _option_items(value: object) -> tuple[dict[str, tuple[str, ...]], ...]:
    """Normalise the official ``options`` list of variant mappings."""
    if not isinstance(value, (list, tuple)):
        return ()
    options: list[dict[str, tuple[str, ...]]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        option: dict[str, tuple[str, ...]] = {}
        for key, raw in entry.items():
            text_key = _clean_text(key)
            values = _clean_list(raw)
            if text_key is None or not values:
                continue
            option[text_key] = values
        if option:
            options.append(option)
    return tuple(options)


def _price_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _feature_lines(
    attributes: tuple[tuple[str, tuple[str, ...]], ...],
    options: tuple[dict[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    """Flatten attributes/options into ``key: v1, v2`` lines (deduplicated)."""
    lines: list[str] = []
    seen: set[str] = set()
    for key, values in attributes:
        line = f"{key}: {', '.join(values)}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    for option in options:
        for key, values in option.items():
            line = f"{key}: {', '.join(values)}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return tuple(lines)


def _details(
    attributes: tuple[tuple[str, tuple[str, ...]], ...],
    options: tuple[dict[str, tuple[str, ...]], ...],
    seller_id: str | None,
) -> tuple[tuple[str, str], ...]:
    """Build the accepted ``details`` bag from the source's own attribute names."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, values in attributes:
        pair = (key, ", ".join(values))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    for option in options:
        for key, values in option.items():
            pair = (key, ", ".join(values))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if seller_id:
        pairs.append(("seller_id", seller_id))
    return tuple(pairs)


def _category_path(value: object) -> tuple[str, ...]:
    text = _clean_text(value)
    if text is None:
        return ()
    return tuple(part.strip() for part in text.split(" - ") if part.strip())


@dataclass(frozen=True)
class ShoppingCompanionProduct:
    """One Shopping Companion product, adapted to the accepted catalogue contract."""

    product_id: str
    namespaced_id: str
    title: str | None
    category_path: tuple[str, ...]
    price: float | None
    seller_id: str | None
    seller_name: str | None
    attributes: tuple[tuple[str, tuple[str, ...]], ...]
    options: tuple[dict[str, tuple[str, ...]], ...]
    description: str | None
    record: ProductMetadata

    @property
    def category(self) -> str | None:
        return " - ".join(self.category_path) if self.category_path else None

    @property
    def description_truncated(self) -> str | None:
        if self.description is None:
            return None
        return self.description[:_DESCRIPTION_CHARS]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (no fabricated fields)."""
        return {
            "product_id": self.product_id,
            "namespaced_id": self.namespaced_id,
            "title": self.title,
            "category": self.category,
            "price": self.price,
            "seller_id": self.seller_id,
            "seller_name": self.seller_name,
            "attributes": {key: list(values) for key, values in self.attributes},
            "options": [{key: list(values) for key, values in option.items()} for option in self.options],
            "description": self.description_truncated,
            "source": _RECORD_SOURCE,
        }


def product_from_doc(doc: Mapping[str, Any]) -> ShoppingCompanionProduct:
    """Build an adapter product from one projected ``docs.jsonl`` line.

    Raises :class:`ShoppingCompanionIdentityError` for a missing/blank product id:
    an identity-less catalogue record cannot be referenced safely and is not
    silently dropped here (the builder already drops them and counts the drops).
    """
    product = doc.get("product")
    if not isinstance(product, Mapping):
        raise ShoppingCompanionIdentityError("catalogue doc has no 'product' object", code="malformed_doc")
    product_id = raw_product_id(str(product.get("product_id") or doc.get("id") or ""))
    if not product_id:
        raise ShoppingCompanionIdentityError("catalogue doc has no product id", code="missing_product_id")

    title = _clean_text(product.get("product_name"))
    category_path = _category_path(product.get("category"))
    price = _price_value(product.get("price"))
    seller_id = _clean_text(product.get("seller_id"))
    seller_name = _clean_text(product.get("seller_name"))
    attributes = _attribute_items(product.get("attributes"))
    options = _option_items(product.get("options"))
    description = _clean_text(product.get("description"))
    identity = namespaced_id(product_id)

    record = ProductMetadata(
        parent_asin=identity,
        title=title,
        store=seller_name,
        main_category=category_path[0] if category_path else None,
        categories=category_path,
        features=_feature_lines(attributes, options),
        description=(description,) if description else (),
        price_text=None if price is None else str(price),
        details=_details(attributes, options, seller_id),
        source=_RECORD_SOURCE,
    )
    return ShoppingCompanionProduct(
        product_id=product_id,
        namespaced_id=identity,
        title=title,
        category_path=category_path,
        price=price,
        seller_id=seller_id,
        seller_name=seller_name,
        attributes=attributes,
        options=options,
        description=description,
        record=record,
    )


class _CatalogueView:
    """Minimal ``CatalogTextLike`` view over the normalised records."""

    def __init__(self, records: Mapping[str, ProductMetadata]) -> None:
        self._records = records

    @property
    def records(self) -> Mapping[str, ProductMetadata]:
        return self._records


class ShoppingCompanionCatalogue:
    """The P14-A limited Shopping Companion catalogue.

    The catalogue is static for the lifetime of the process.  The BM25 index is
    built lazily on first search (a single pass over the records) so loading a
    catalogue for identity-only work (Product Info) does not pay for retrieval.
    """

    def __init__(
        self,
        products: Mapping[str, ShoppingCompanionProduct],
        *,
        docs_path: str | Path | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self._products = dict(products)
        self._records: dict[str, ProductMetadata] = {
            identity: product.record for identity, product in self._products.items()
        }
        self._docs_path = str(docs_path) if docs_path is not None else None
        self._manifest = dict(manifest or {})
        self._view = _CatalogueView(self._records)
        self._search_source: CatalogSearchSource | None = None
        #: Load-time diagnostics (malformed lines / duplicates) filled by ``load``.
        self._load_stats: dict[str, Any] = {"malformed_lines": 0, "duplicate_id_count": 0, "duplicate_ids": []}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def load(
        cls,
        docs_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
    ) -> "ShoppingCompanionCatalogue":
        """Load a projected ``docs.jsonl`` file; the file must be the real sample."""
        path = Path(docs_path)
        products: dict[str, ShoppingCompanionProduct] = {}
        duplicate_ids: list[str] = []
        malformed = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                    product = product_from_doc(doc)
                except (json.JSONDecodeError, ShoppingCompanionIdentityError):
                    malformed += 1
                    continue
                if product.namespaced_id in products:
                    duplicate_ids.append(product.product_id)
                    continue
                products[product.namespaced_id] = product
        manifest: dict[str, Any] = {}
        if manifest_path is not None:
            candidate = Path(manifest_path)
            if candidate.is_file():
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
        catalogue = cls(products, docs_path=path, manifest=manifest)
        catalogue._load_stats = {
            "malformed_lines": malformed,
            "duplicate_ids": duplicate_ids[:10],
            "duplicate_id_count": len(duplicate_ids),
        }
        return catalogue

    @classmethod
    def from_docs(
        cls,
        docs: Iterable[Mapping[str, Any]],
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> "ShoppingCompanionCatalogue":
        """Build a catalogue from in-memory projected docs (used by tests)."""
        products: dict[str, ShoppingCompanionProduct] = {}
        for doc in docs:
            product = product_from_doc(doc)
            products.setdefault(product.namespaced_id, product)
        return cls(products, manifest=manifest)

    # -- metadata ---------------------------------------------------------- #

    @property
    def namespace(self) -> str:
        return SHOPPING_COMPANION_NAMESPACE

    @property
    def catalogue_size(self) -> int:
        return len(self._products)

    @property
    def docs_path(self) -> str | None:
        return self._docs_path

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def limited(self) -> bool:
        """True unless the manifest explicitly proves a full-catalogue build."""
        selection = str(self._manifest.get("selection_method") or "").lower()
        if "not the full" in selection or "limited" in selection or not self._manifest:
            return True
        return False

    @property
    def records(self) -> Mapping[str, ProductMetadata]:
        """Namespaced identity -> normalised catalogue record (accepted contract)."""
        return self._records

    def __contains__(self, identity: object) -> bool:
        """Catalogue membership for grounded reasoners; never raises for bad input."""
        if not isinstance(identity, str):
            return False
        return identity in self._products

    def get(self, identity: str) -> ShoppingCompanionProduct | None:
        """Look up a product by namespaced or bare id; cross-namespace raises."""
        text = str(identity or "").strip()
        if not text:
            return None
        if ":" in text:
            _, item_id = split_namespaced_id(text)
            return self._products.get(namespaced_id(item_id))
        return self._products.get(namespaced_id(text))

    # -- retrieval --------------------------------------------------------- #

    @property
    def search_source(self) -> CatalogSearchSource:
        """The accepted BM25 catalogue source (built once, reused)."""
        if self._search_source is None:
            self._search_source = CatalogSearchSource(
                self._view,
                fields=SEARCHABLE_FIELDS,
                mode="bm25",
            )
        return self._search_source

    def search(self, query: str, *, limit: int = 10) -> CatalogSearchResult:
        """BM25 search over the limited catalogue, using the accepted implementation."""
        return self.search_source.search(query, limit=limit)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CATALOGUE_SCHEMA,
            "namespace": self.namespace,
            "catalogue_size": self.catalogue_size,
            "limited_catalogue": self.limited,
            "docs_path": self._docs_path,
            "manifest": dict(self._manifest),
        }



