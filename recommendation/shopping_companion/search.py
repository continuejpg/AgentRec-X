"""P14-A ``product_search`` tool: real BM25 over the limited Shopping Companion catalogue.

The tool is a thin, namespace-aware wrapper around the *existing* catalogue
retrieval implementation (:class:`recommendation.control.catalog_search.CatalogSearchSource`,
exposed by :class:`recommendation.shopping_companion.catalogue.ShoppingCompanionCatalogue`).
It does not re-rank, filter by target products, or add results that the catalogue
search did not return.

Contract (P14-0 ``docs/P14_PREP_INTERFACE.md`` section 3.1):

* in: current query, catalogue namespace, top-k / budget, optional ``shop_id`` and
  official ``price`` range filter, retrieval options;
* out: product ids (raw + namespaced), source, BM25 score + score kind,
  matched fields, traceable retrieval metadata (query, catalogue size, limited
  catalogue flag, latency, filters).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .catalogue import ShoppingCompanionCatalogue
from .namespace import SHOPPING_COMPANION_NAMESPACE, raw_product_id

__all__ = [
    "DEFAULT_MAX_K",
    "SearchHit",
    "SearchResult",
    "ShoppingCompanionSearchError",
    "ShoppingCompanionSearchTool",
    "parse_price_filter",
]

#: The official tool searches up to 50 products per call; keep the same ceiling.
DEFAULT_MAX_K = 50

#: Score kind reported for every hit (never compared across sources).
SCORE_KIND = "bm25"


class ShoppingCompanionSearchError(ValueError):
    """An invalid search request (bad k, bad filter, wrong namespace)."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def parse_price_filter(price: str | None) -> tuple[float | None, float | None]:
    """Parse the official ``"min-max"`` / ``"min-"`` / ``"-max"`` price filter.

    Returns ``(price_min, price_max)`` with ``None`` meaning unbounded.  This mirrors
    the official ``product_search`` semantics and validates it the same way.
    """
    if price is None:
        return None, None
    text = str(price).strip()
    if not text:
        return None, None
    parts = text.split("-")
    if len(parts) != 2:
        raise ShoppingCompanionSearchError(
            f"invalid price range {price!r}; expected 'min-max' or 'min-'", code="invalid_price"
        )
    low_text, high_text = parts[0].strip(), parts[1].strip()
    try:
        price_min = float(low_text) if low_text else None
        price_max = float(high_text) if high_text else None
    except ValueError as exc:
        raise ShoppingCompanionSearchError(
            f"invalid price range {price!r}; expected numeric bounds", code="invalid_price"
        ) from exc
    if price_min is not None and price_min < 0:
        raise ShoppingCompanionSearchError("price minimum must be >= 0", code="invalid_price")
    if price_min is not None and price_max is not None and price_min > price_max:
        raise ShoppingCompanionSearchError("price minimum must be <= maximum", code="invalid_price")
    return price_min, price_max


@dataclass(frozen=True)
class SearchHit:
    """One real catalogue hit, in the tool's returned order."""

    product_id: str
    namespaced_id: str
    rank: int
    score: float
    score_kind: str
    matched_fields: tuple[str, ...]
    title: str | None
    seller_id: str | None
    seller_name: str | None
    price: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "namespaced_id": self.namespaced_id,
            "rank": self.rank,
            "score": self.score,
            "score_kind": self.score_kind,
            "matched_fields": list(self.matched_fields),
            "title": self.title,
            "seller_id": self.seller_id,
            "seller_name": self.seller_name,
            "price": self.price,
        }


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one real catalogue search, with traceable retrieval metadata."""

    query: str
    namespace: str
    hits: tuple[SearchHit, ...]
    matched: bool
    catalogue_size: int
    limited_catalogue: bool
    score_kind: str
    latency_ms: float
    filters: dict[str, Any]
    catalogue_source: str

    @property
    def product_ids(self) -> tuple[str, ...]:
        return tuple(hit.product_id for hit in self.hits)

    @property
    def namespaced_ids(self) -> tuple[str, ...]:
        return tuple(hit.namespaced_id for hit in self.hits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "namespace": self.namespace,
            "hits": [hit.as_dict() for hit in self.hits],
            "matched": self.matched,
            "catalogue_size": self.catalogue_size,
            "limited_catalogue": self.limited_catalogue,
            "score_kind": self.score_kind,
            "latency_ms": round(self.latency_ms, 3),
            "filters": dict(self.filters),
            "catalogue_source": self.catalogue_source,
        }


class ShoppingCompanionSearchTool:
    """Real catalogue search bound to one :class:`ShoppingCompanionCatalogue`."""

    def __init__(
        self,
        catalogue: ShoppingCompanionCatalogue,
        *,
        max_k: int = DEFAULT_MAX_K,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_k < 1:
            raise ValueError("max_k must be >= 1")
        self._catalogue = catalogue
        self._max_k = max_k
        self._clock = clock

    @property
    def catalogue(self) -> ShoppingCompanionCatalogue:
        return self._catalogue

    @property
    def namespace(self) -> str:
        return SHOPPING_COMPANION_NAMESPACE

    @property
    def max_k(self) -> int:
        return self._max_k

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        shop_id: str | None = None,
        price: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
    ) -> SearchResult:
        """Search the real limited catalogue; never fabricates or reorders results.

        The official filter semantics are preserved: when a shop or price filter is
        present the tool searches a wider head and then applies the filter, keeping
        the catalogue's own order.  Because the P14-A catalogue is a limited sample,
        the returned ``limited_catalogue`` flag and ``catalogue_size`` are recorded on
        every response.
        """
        text = str(query or "").strip()
        if not text:
            raise ShoppingCompanionSearchError("query must be non-empty", code="empty_query")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ShoppingCompanionSearchError("limit must be an integer", code="invalid_limit")
        if limit < 1 or limit > self._max_k:
            raise ShoppingCompanionSearchError(
                f"limit must be between 1 and {self._max_k}, got {limit}", code="invalid_limit"
            )
        parsed_min, parsed_max = parse_price_filter(price)
        if price_min is not None:
            parsed_min = float(price_min)
        if price_max is not None:
            parsed_max = float(price_max)
        if parsed_min is not None and parsed_min < 0:
            raise ShoppingCompanionSearchError("price_min must be >= 0", code="invalid_price")
        if parsed_min is not None and parsed_max is not None and parsed_min > parsed_max:
            raise ShoppingCompanionSearchError("price_min must be <= price_max", code="invalid_price")
        shop_filter = str(shop_id).strip() if shop_id is not None else None
        if shop_filter == "":
            shop_filter = None

        filtered = shop_filter is not None or parsed_min is not None or parsed_max is not None
        capacity = min(
            self._catalogue.catalogue_size,
            max(limit * 20, 200) if filtered else limit,
        )
        started = self._clock()
        result = self._catalogue.search(text, limit=max(capacity, 1))
        elapsed_ms = (self._clock() - started) * 1000.0

        hits: list[SearchHit] = []
        for catalogue_hit in result.hits:
            product = self._catalogue.get(catalogue_hit.parent_asin)
            if product is None:  # pragma: no cover - defensive; index and store are one map
                continue
            if shop_filter is not None:
                if product.seller_id != shop_filter and product.seller_name != shop_filter:
                    continue
            if parsed_min is not None and (product.price is None or product.price < parsed_min):
                continue
            if parsed_max is not None and (product.price is None or product.price > parsed_max):
                continue
            hits.append(
                SearchHit(
                    product_id=raw_product_id(product.namespaced_id),
                    namespaced_id=product.namespaced_id,
                    rank=len(hits) + 1,
                    score=float(catalogue_hit.score),
                    score_kind=result.score_kind,
                    matched_fields=tuple(catalogue_hit.matched_fields),
                    title=product.title,
                    seller_id=product.seller_id,
                    seller_name=product.seller_name,
                    price=product.price,
                )
            )
            if len(hits) >= limit:
                break

        return SearchResult(
            query=text,
            namespace=SHOPPING_COMPANION_NAMESPACE,
            hits=tuple(hits),
            matched=bool(hits) and result.matched,
            catalogue_size=self._catalogue.catalogue_size,
            limited_catalogue=self._catalogue.limited,
            score_kind=result.score_kind,
            latency_ms=elapsed_ms,
            filters={
                "shop_id": shop_filter,
                "price": price,
                "price_min": parsed_min,
                "price_max": parsed_max,
                "mode": "bm25",
                "capacity": capacity,
            },
            catalogue_source=str(self._catalogue.docs_path or "shopping_companion:docs.jsonl"),
        )
