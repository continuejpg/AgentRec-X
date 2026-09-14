"""Synthetic catalogue/retrieval fixtures for the Milestone 8 tests.

Everything here is tiny, deterministic and offline: no real metadata artifact, no
checkpoint and no catalogue.  Real-data verification lives in the M8 smoke scripts.

The fixtures intentionally use product identities that do **not** all start with
``B``, so any accidental assumption that ``parent_asin`` has a particular shape fails
a test rather than silently passing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.schemas import (  # noqa: E402
    MetadataRecord,
    MissingMetadata,
    ProductMetadata,
)
from recommendation.catalog.metadata import (  # noqa: E402
    MetadataIndex,
    normalize_product_record,
)
from recommendation.tools.schemas import (  # noqa: E402
    RecommendationToolResult,
    ToolRecommendation,
)

#: Non-``B`` first characters, to prove opaque identifier handling.
OPAQUE_ASINS: tuple[str, ...] = (
    "boot-001",
    "9mat-002",
    "_tent-003",
    "Ünïcode-004",
)

#: Synthetic source records exercising the fields M8 normalizes.
RAW_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "parent_asin": "boot-001",
        "title": "Waterproof Hiking Boots",
        "store": "Acme Outdoors",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Outdoor Recreation", "Hiking"],
        "features": ["waterproof membrane", "vibram sole", "ankle support"],
        "description": ["Built for wet trails."],
        "price": 89.99,
        "average_rating": 4.6,
        "rating_number": 1200,
        "details": {"Brand Name": "Acme Outdoors", "Color": "brown", "Material": "leather"},
    },
    {
        "parent_asin": "9mat-002",
        "title": "Non-Slip Yoga Mat",
        "store": "ZenWorks",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Fitness", "Yoga"],
        "features": ["non-slip surface", "cushioned grip"],
        "description": ["Extra cushioning for joint comfort."],
        "price": 29.5,
        "average_rating": 4.2,
        "rating_number": 320,
        "details": {"Brand Name": "ZenWorks", "Color": "teal"},
    },
    {
        "parent_asin": "_tent-003",
        "title": "Two Person Backpacking Tent",
        "store": "CampFix",
        "main_category": "Sports & Outdoors",
        "categories": ["Sports & Outdoors", "Camping", "Tents"],
        "features": ["two person capacity", "aluminium poles"],
        "description": [],
        "price": "159.00",
        "average_rating": 4.8,
        "rating_number": 87,
        "details": {"Brand Name": "CampFix"},
    },
    {
        # A record whose only populated fields are not searchable text.
        "parent_asin": "Ünïcode-004",
        "title": None,
        "features": [],
        "description": [],
        "details": {},
    },
)


def raw_records() -> list[dict[str, Any]]:
    """Return fresh copies of the synthetic source records."""
    return [dict(record) for record in RAW_RECORDS]


def records() -> list[ProductMetadata]:
    """Return the normalized synthetic records, in a stable order."""
    return [normalize_product_record(record) for record in RAW_RECORDS]


def index(asins: Iterable[str] | None = None) -> MetadataIndex:
    """Build an in-memory metadata index, optionally restricted to ``asins``."""
    selected = set(asins) if asins is not None else None
    chosen = [record for record in records() if selected is None or record.parent_asin in selected]
    return MetadataIndex.from_records(chosen)


class RecordingMetadataLookup:
    """A metadata lookup that records every identifier it was asked about.

    Used to prove candidate-universe isolation: the set of identifiers that reached
    the metadata layer must equal the candidate set exactly.
    """

    def __init__(self, delegate: MetadataIndex) -> None:
        self._delegate = delegate
        self.looked_up: list[str] = []
        self.many_calls: list[tuple[str, ...]] = []

    @property
    def requested_asins(self) -> tuple[str, ...]:
        """Every identifier requested, in request order (duplicates preserved)."""
        return tuple(self.looked_up)

    def lookup(self, parent_asin: str) -> MetadataRecord:
        """Record the request and delegate."""
        self.looked_up.append(parent_asin)
        return self._delegate.lookup(parent_asin)

    def lookup_many(self, parent_asins: Sequence[str]) -> tuple[MetadataRecord, ...]:
        """Record the request and delegate, preserving alignment."""
        self.many_calls.append(tuple(parent_asins))
        self.looked_up.extend(parent_asins)
        return self._delegate.lookup_many(parent_asins)

    def __contains__(self, parent_asin: object) -> bool:
        """Delegate membership."""
        return parent_asin in self._delegate


def tool_result(
    candidates: Sequence[tuple[str, int, float]],
    *,
    requested_k: int | None = None,
    history_length: int = 4,
) -> RecommendationToolResult:
    """Build a Tool result from ``(parent_asin, item_id, score)`` triples."""
    recommendations = [
        ToolRecommendation(rank=position, parent_asin=asin, item_id=item_id, score=score)
        for position, (asin, item_id, score) in enumerate(candidates, start=1)
    ]
    return RecommendationToolResult(
        recommendations=recommendations,
        requested_k=requested_k if requested_k is not None else len(recommendations),
        returned_k=len(recommendations),
        history_length=history_length,
        effective_history_length=history_length,
        history_truncated=False,
        eligible_candidates=1000,
        timings_ms={"scoring": 1.0, "ranking": 0.5},
    )


def missing(parent_asin: str) -> MissingMetadata:
    """Return an explicit missing-metadata result for ``parent_asin``."""
    return MissingMetadata(parent_asin=parent_asin)


__all__ = [
    "OPAQUE_ASINS",
    "RAW_RECORDS",
    "RecordingMetadataLookup",
    "index",
    "missing",
    "raw_records",
    "records",
    "tool_result",
]
