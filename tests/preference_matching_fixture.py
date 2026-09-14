"""Fixtures for the Milestone 10A preference-matching tests.

Small, deterministic and offline: synthetic M8 metadata, synthetic M9 preferences and
synthetic candidates.  No checkpoint, no real metadata artifact and no database.

Candidates are built as real M8 ``EnrichedRecommendation`` objects so the matcher is
exercised against the accepted integration type rather than a stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog.metadata import normalize_product_record  # noqa: E402
from recommendation.catalog.schemas import MissingMetadata  # noqa: E402
from recommendation.memory.schemas import (  # noqa: E402
    PreferenceKind,
    PreferenceMemoryEntry,
    PreferenceMemorySnapshot,
    PreferenceMode,
    PreferencePolarity,
    PreferenceStatus,
)
from recommendation.rag.schemas import EnrichedRecommendation  # noqa: E402
from recommendation.tools.schemas import ToolRecommendation  # noqa: E402

__all__ = [
    "CANDIDATE_ROWS",
    "make_candidate",
    "make_entry",
    "make_preference_candidate",
    "make_snapshot",
    "metadata_for",
]


#: Candidate metadata used across the suite.  Deliberately covers: full categorical
#: data, colour only, no colour, a non-numeric price placeholder, and no record at all.
CANDIDATE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "parent_asin": "cand-red",
        "item_id": 101,
        "score": 5.5,
        "metadata": {
            "title": "Waterproof Hiking Boots",
            "store": "Acme",
            "price": 89.99,
            "features": ["waterproof membrane", "lightweight build"],
            "categories": ["Sports & Outdoors", "Camping & Hiking"],
            "details": {"Color": "red", "Material": "leather", "Brand Name": "Acme"},
        },
    },
    {
        "parent_asin": "cand-blue",
        "item_id": 102,
        "score": 4.25,
        "metadata": {
            "title": "Yoga Mat",
            "store": "ZenWorks",
            "price": 29.5,
            "features": ["non-slip surface"],
            "categories": ["Sports & Outdoors", "Fitness"],
            "details": {"Color": "blue", "Material": "foam"},
        },
    },
    {
        "parent_asin": "cand-nocolor",
        "item_id": 103,
        "score": 3.75,
        "metadata": {
            "title": "Tent Pole Repair Kit",
            "store": "CampFix",
            "price": 159.0,
            "features": ["aluminium poles"],
            "categories": ["Sports & Outdoors", "Camping"],
            "details": {},
        },
    },
    {
        "parent_asin": "cand-badprice",
        "item_id": 104,
        "score": 2.5,
        "metadata": {
            "title": "Mystery Item",
            "store": None,
            "price": "—",
            "features": [],
            "categories": [],
            "details": {"Colour": "red"},
        },
    },
    {
        "parent_asin": "cand-nometa",
        "item_id": 105,
        "score": 1.25,
        "metadata": None,
    },
)


def metadata_for(row: dict[str, Any]) -> Any:
    """Return the normalized metadata for a candidate row, or ``None``."""
    raw = row["metadata"]
    return None if raw is None else normalize_product_record({**raw, "parent_asin": row["parent_asin"]})


def make_candidate(
    parent_asin: str = "cand-x",
    *,
    rank: int = 1,
    item_id: int = 1,
    score: float = 1.0,
    metadata: Any = None,
    metadata_status: str | None = None,
) -> EnrichedRecommendation:
    """Build a real M8 enrichment item for the matcher to consume."""
    if metadata is None and metadata_status is None:
        metadata_status = "missing"
    if metadata_status is None:
        metadata_status = "found" if metadata is not None else "missing"
    return EnrichedRecommendation(
        recommendation=ToolRecommendation(
            rank=rank, parent_asin=parent_asin, item_id=item_id, score=score
        ),
        metadata_status=metadata_status,  # type: ignore[arg-type]
        metadata=metadata,
    )


def candidates_from_rows(rows: Sequence[dict[str, Any]] = CANDIDATE_ROWS) -> list[EnrichedRecommendation]:
    """Build the standard candidate list, in the row order given."""
    return [
        make_candidate(
            row["parent_asin"],
            rank=position,
            item_id=row["item_id"],
            score=row["score"],
            metadata=metadata_for(row),
        )
        for position, row in enumerate(rows, start=1)
    ]


def make_entry(
    *,
    memory_id: str = "mem-1",
    kind: PreferenceKind | str = PreferenceKind.COLOR,
    value: str = "red",
    polarity: PreferencePolarity | str = PreferencePolarity.AVOID,
    status: PreferenceStatus | str = PreferenceStatus.ACTIVE,
    source_text: str = "user statement",
    source_turn_id: str = "t1",
    logical_seq: int = 1,
    user_key: str = "alice",
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> PreferenceMemoryEntry:
    """Build a stored preference entry."""
    return PreferenceMemoryEntry(
        memory_id=memory_id,
        user_key=user_key,
        kind=kind,
        value=value,
        polarity=polarity,
        source_text=source_text,
        source_turn_id=source_turn_id,
        extractor="test",
        status=status,
        logical_seq=logical_seq,
        created_at=1000.0 + logical_seq,
        supersedes=supersedes,
        superseded_by=superseded_by,
    )


def make_snapshot(*entries: PreferenceMemoryEntry, user_key: str = "alice") -> PreferenceMemorySnapshot:
    """Build a preference snapshot from entries."""
    return PreferenceMemorySnapshot(user_key=user_key, entries=tuple(entries))


def make_preference_candidate(
    *,
    kind: PreferenceKind | str = PreferenceKind.COLOR,
    value: str = "red",
    polarity: PreferencePolarity | str = PreferencePolarity.AVOID,
    source_text: str = "user statement",
    mode: PreferenceMode | str = PreferenceMode.ADD,
    replaces: str | None = None,
) -> Any:
    """Build an extraction candidate (for symmetric extraction tests if needed)."""
    from recommendation.memory.schemas import PreferenceCandidate

    return PreferenceCandidate(
        kind=kind,
        value=value,
        polarity=polarity,
        source_text=source_text,
        mode=mode,
        replaces=replaces,
    )


__all__ += ["candidates_from_rows"]
