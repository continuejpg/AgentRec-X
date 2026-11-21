"""Category-scoped BM25 index used only by the M3-C experiment.

The production BM25 path searches the full catalogue.  The candidate strategy asks the same
BM25 query over the subset of products whose title/sidecar category denotes the requested
product type.  This is the single experimental variable: candidate scope.  Query construction
and BM25 parameters stay unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue, ElectronicsProduct
from recommendation.catalog.enrichment import MetadataSidecar
from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.preference.product_role import infer_product_target

__all__ = [
    "CATEGORY_TERMS",
    "CategoryScope",
    "SCOPE_RULES",
    "classify_product",
    "is_accessory_title",
]

SCOPE_RULES = {
    "laptop": {
        "title": re.compile(r"\b(laptop|notebook)\b", re.IGNORECASE),
        "categories": (
            "laptops", "gaming laptops", "traditional laptops", "2 in 1 laptops",
            "notebooks",
        ),
    },
    "headphones": {
        "title": re.compile(
            r"\b(headphone|headphones|headset|headsets|earbud|earbuds|earphone|earphones)\b",
            re.IGNORECASE,
        ),
        "categories": (
            "headphones", "earbud headphones", "over-ear headphones",
            "on-ear headphones", "in-ear headphones", "earbuds", "earphones",
            "headsets",
        ),
    },
    "tablet": {
        "title": re.compile(r"\b(tablet|tablets|ipad)\b", re.IGNORECASE),
        "categories": ("tablets", "tablet computers"),
    },
    "mouse": {
        "title": re.compile(r"\b(mouse|mice)\b", re.IGNORECASE),
        "categories": ("mice", "computer mice"),
    },
    "keyboard": {
        "title": re.compile(r"\b(keyboard|keyboards)\b", re.IGNORECASE),
        "categories": ("keyboards", "computer keyboards"),
    },
}

CATEGORY_TERMS = frozenset(SCOPE_RULES)

_COMMON_ACCESSORY = (
    r"chargers?|adapters?|batter(?:y|ies)|replacement|stands?|risers?|sleeves?|cases?|"
    r"covers?|bags?|backpacks?|totes?|briefcases?|desks?|handbags?|messengers?|screens?|"
    r"lcd|digitizers?|fans?|cooling|heatsinks?|cables?|cords?|power|supply|"
    r"motherboards?|inverters?|stickers?|skins?|decals?|docks?|hubs?|carry-on|luggage|"
    r"pockets?|slimcases?|drives?|dvd|cd|protection|plans?|warrant(?:y|ies)|"
    r"accessor(?:y|ies)|projectors?|stations?|office product|rubber feet|feet|foot|"
    r"portfolios?|cradles?"
)
_TARGET_ACCESSORY = {
    "laptop": (
        r"memory|ram|ssd|hard drive|keyboards?|keycaps?|pads?|speakers?|webcams?|cameras?|"
        r"lamps?|lights?|screws?|toolkits?|mounts?|trays?|holders?|protectors?|filters?|"
        r"privacy|repairs?|parts?|warmers?|mouse|mice|headsets?|headphones?|table(?!t)"
    ),
    "headphones": (
        r"memory|foam|cushions?|earpads?|ear pad|replacement|pads?|speakers?|stands?|hooks?|hangers?"
    ),
    "tablet": (
        r"keyboards?|keycaps?|pads?|styluss?|pens?|holders?|mounts?|trays?|protectors?|"
        r"filters?|privacy|repairs?|parts?|speakers?|cameras?|webcams?"
    ),
    "mouse": (
        r"pads?|mousepads?|bungees?|skates?|grips?|feet|weights?|holders?|mounts?|trays?|"
        r"protectors?|repairs?|parts?"
    ),
    "keyboard": (
        r"pads?|mousepads?|covers?|skins?|stickers?|keycaps?|pullers?|removers?|cleaners?|"
        r"wrist|rests?|holders?|mounts?|trays?|protectors?|repairs?|parts?"
    ),
}
_ACCESSORY_BY_TARGET = {
    target: re.compile(rf"\b({_COMMON_ACCESSORY}|{extra})\b", re.IGNORECASE)
    for target, extra in _TARGET_ACCESSORY.items()
}

_ACCESSORY_LEAF = re.compile(
    r"(accessor|case|cover|bag|sleeve|stand|riser|charger|adapter|battery|cable|"
    r"pad|dock|hub|screen|keyboard cover|skin|sticker)",
    re.IGNORECASE,
)


def is_accessory_title(target: str, title: str) -> bool:
    pattern = _ACCESSORY_BY_TARGET.get(target)
    return bool(pattern and pattern.search(str(title or "")))


def _leaf_categories(categories: tuple[str, ...]) -> list[str]:
    generic = {
        "electronics", "computers & accessories", "computer accessories & peripherals",
        "audio & video accessories", "cell phones & accessories",
        "computers & tablets", "context of use",
    }
    return [str(c).casefold() for c in categories if str(c).casefold() not in generic]


def _sidecar_categories(product: ElectronicsProduct, sidecar: MetadataSidecar) -> tuple[str, ...]:
    record = sidecar.get(product.item_id)
    return tuple(record.categories) if record is not None else ()


def classify_product(
    product: ElectronicsProduct,
    target: str,
    sidecar: MetadataSidecar,
) -> bool:
    """Return whether a product belongs to the target retrieval scope.

    The rule uses only the real title and, when present, the Raw Meta leaf categories.  It
    never uses BM25 score, rank or a retrieval result.
    """
    rule = SCOPE_RULES.get(target)
    if rule is None:
        return False
    title = str(product.title or "")
    if is_accessory_title(target, title):
        return False
    # The earliest product-type word is the head type; a compatibility clause later in the
    # title ("headphones for ... Tablet") must not make the product a tablet.
    title_type = infer_product_target(title).product_type
    title_match = title_type == target or (
        title_type is None and bool(rule["title"].search(title))
    )
    leaves = _leaf_categories(_sidecar_categories(product, sidecar))
    sidecar_match = any(
        any(term in leaf for term in rule["categories"])
        for leaf in leaves
        if not _ACCESSORY_LEAF.search(leaf)
    )
    return title_match or (title_type is None and sidecar_match)


@dataclass
class CategoryScope:
    """Lazily built BM25 sources over title/category-selected catalogue subsets."""

    catalogue: ElectronicsCatalogue
    sidecar: MetadataSidecar
    top_k: int = 5
    _sources: dict[str, CatalogSearchSource] | None = None
    _members: dict[str, list[ElectronicsProduct]] | None = None

    def __post_init__(self) -> None:
        if self._sources is None:
            self._sources = {}
        if self._members is None:
            self._members = {}

    @property
    def catalogue_size(self) -> int:
        return self.catalogue.catalogue_size

    def members(self, target: str) -> list[ElectronicsProduct]:
        assert self._members is not None
        if target not in self._members:
            self._members[target] = [
                product for product in self.catalogue._products.values()
                if classify_product(product, target, self.sidecar)
            ]
        return list(self._members[target])

    def source(self, target: str) -> CatalogSearchSource:
        assert self._sources is not None
        if target not in self._sources:
            products = self.members(target)
            records = {
                product.namespaced_id: product.record
                for product in products
                if product.record is not None
            }
            view = SimpleNamespace(records=records)
            self._sources[target] = CatalogSearchSource(view, mode="bm25")
        return self._sources[target]

    def sizes(self) -> dict[str, int]:
        return {target: len(self.members(target)) for target in sorted(CATEGORY_TERMS)}

    def search(self, target: str, query: str, *, limit: int | None = None) -> Any:
        return self.source(target).search(query, limit=limit or self.top_k)
