"""Catalog product-metadata schemas (Milestone 8-A).

The recommendation stack identifies a product by an **opaque** external
``parent_asin`` string.  Nothing in this module assumes anything about the shape of
that string -- in particular it does not assume it starts with ``B``.  The accepted
SASRec mapping stays authoritative for ``parent_asin <-> item_id``; catalog metadata
attaches to that identity and never redefines it.

Source fidelity is the governing rule here:

* only fields that really exist in the source schema are represented;
* a field the source does not supply stays ``None`` / empty -- never inferred;
* no brand is guessed from a title, no category from free text, no price from a
  description, no material from a name;
* text is normalised for structure only (whitespace), never paraphrased, summarised
  or generated;
* the original price token is preserved as a string rather than being reformatted,
  so no source value is silently changed.

The upstream schema documented by the Amazon Reviews 2023 dataset card for
``meta_<Category>.jsonl.gz`` records is: ``main_category``, ``title``,
``average_rating``, ``rating_number``, ``features``, ``description``, ``price``,
``images``, ``videos``, ``store``, ``categories``, ``details``, ``parent_asin``,
``bought_together`` (plus a rarely-present ``subtitle``/``author``).  See
``recommendation/catalog/README.md`` for exactly which of those M8 normalises and
which it deliberately ignores.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "MISSING_METADATA_STATUS",
    "NORMALIZATION_VERSION",
    "CandidateMetadataStatus",
    "MetadataIndexProtocol",
    "MetadataLookup",
    "MetadataRecord",
    "MissingMetadata",
    "ProductMetadata",
]

#: Bumped whenever the normalisation rules change in a way that alters artifacts.
NORMALIZATION_VERSION = 1

#: Explicit status for a catalog item the metadata source does not cover.
MISSING_METADATA_STATUS = "missing"

#: A non-empty identifier string.  Deliberately opaque: no pattern, no prefix rule.
OpaqueIdentifier = Annotated[str, Field(min_length=1)]

#: A non-empty text fragment.
NonEmptyText = Annotated[str, Field(min_length=1)]


def _clean_text(value: object) -> str | None:
    """Strip structural whitespace; map blank/absent values to ``None``.

    This is the **only** transformation applied to product text.  Nothing is
    paraphrased, truncated for meaning, summarised or generated, and interior
    spacing is preserved.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected a string or null")
    stripped = value.strip()
    return stripped or None


def _clean_text_list(value: object) -> tuple[str, ...]:
    """Normalise a list of text fragments, preserving logical order.

    Order is preserved because it is meaningful in the source (feature bullets and
    category paths are ordered).  Blank entries are dropped.  Entries that are
    exact duplicates **within the same field of the same record** are collapsed,
    because they carry no additional information and would otherwise be scored
    twice by retrieval; no other deduplication or reordering happens.

    The function is deliberately idempotent: feeding it its own output is a no-op,
    so a record produced by the normalization pipeline can be re-validated without
    being altered.
    """
    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(entry, str) for entry in value):
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    seen: set[str] = set()
    cleaned: list[str] = []
    for entry in value:
        text = _clean_text(entry)
        if text is None or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return tuple(cleaned)


def _clean_details(value: object) -> tuple[tuple[str, str], ...]:
    """Normalise the source ``details`` mapping into ordered key/value pairs.

    The source ``details`` object carries attributes such as ``Brand Name``,
    ``Color`` or ``Material`` exactly as the catalogue supplies them.  Keys and
    values are whitespace-stripped, blank entries are dropped, and the source key
    order is preserved (there is no documented reason to reorder it).

    Like :func:`_clean_text_list` this is idempotent, so already-normalized pairs
    pass through unchanged.
    """
    if value is None:
        return ()
    if isinstance(value, tuple) and all(
        isinstance(entry, tuple)
        and len(entry) == 2
        and all(isinstance(part, str) for part in entry)
        for entry in value
    ):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("expected an object of string keys and values")
    items: list[tuple[str, str]] = []
    for key, raw in value.items():
        text_key = _clean_text(key)
        text_value = _clean_text(raw)
        if text_key is None or text_value is None:
            continue
        items.append((text_key, text_value))
    return tuple(items)


class ProductMetadata(BaseModel):
    """One normalized catalogue product record.

    Every field is either present exactly as the source supplied it, or ``None`` /
    empty.  ``parent_asin`` is the only required field, because it is the join key
    to the accepted recommendation catalog.

    ``details`` keeps the source's own attribute names (for example
    ``"Brand Name"``, ``"Color"``, ``"Material"``) rather than renaming them into
    invented fields, so a consumer can always trace a fact back to the source key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: OpaqueIdentifier = Field(
        ..., description="Opaque external product identity; never assumed to start with 'B'."
    )

    # -- descriptive text -------------------------------------------------- #
    title: NonEmptyText | None = None
    subtitle: NonEmptyText | None = None
    author: NonEmptyText | None = None
    store: NonEmptyText | None = Field(
        default=None, description="Source 'store' value. Exposed as-is, not renamed to brand."
    )
    main_category: NonEmptyText | None = None
    categories: tuple[NonEmptyText, ...] = ()
    features: tuple[NonEmptyText, ...] = ()
    description: tuple[NonEmptyText, ...] = ()

    # -- commercial / social signals --------------------------------------- #
    price_text: NonEmptyText | None = Field(
        default=None,
        description=(
            "Price as text. The source field is a JSON number (occasionally a "
            "placeholder string such as an em dash), so a numeric value has already "
            "been parsed to a Python number before this layer sees it; this field is "
            "therefore the canonical `repr` of the parsed value, NOT the exact raw "
            "lexical token (e.g. a source `1e2` becomes '100.0'). A source string is "
            "kept stripped and verbatim. Nothing is reformatted into a currency and "
            "nothing is inferred from another field."
        ),
    )
    average_rating: float | None = None
    rating_number: int | None = None

    # -- source attribute bag ---------------------------------------------- #
    details: tuple[tuple[NonEmptyText, NonEmptyText], ...] = ()

    # -- provenance -------------------------------------------------------- #
    source: NonEmptyText = Field(
        ..., description="Provenance label, e.g. the normalisation stage that produced this record."
    )

    @field_validator("title", "subtitle", "author", "store", "main_category", "price_text")
    @classmethod
    def _validate_text(cls, value: object) -> str | None:
        return _clean_text(value)

    @field_validator("categories", "features", "description", mode="before")
    @classmethod
    def _validate_text_list(cls, value: object) -> tuple[str, ...]:
        return _clean_text_list(value)

    @field_validator("details", mode="before")
    @classmethod
    def _validate_details(cls, value: object) -> tuple[tuple[str, str], ...]:
        return _clean_details(value)

    # -- derived helpers --------------------------------------------------- #

    @property
    def has_searchable_text(self) -> bool:
        """True when at least one retrieval-bearing text field is populated."""
        return bool(self.title or self.features or self.description or self.categories)

    def details_dict(self) -> dict[str, str]:
        """Return ``details`` as a plain mapping (order preserved)."""
        return dict(self.details)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


class MissingMetadata(BaseModel):
    """Explicit representation of "the catalogue has no metadata for this item".

    This is a first-class result, not an error: a SASRec candidate without metadata
    remains a valid candidate.  It is never replaced, dropped or invented.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: OpaqueIdentifier
    status: Literal["missing"] = MISSING_METADATA_STATUS
    reason: NonEmptyText = Field(
        default="no metadata record for this parent_asin",
        description="Why metadata is absent (source does not cover this item).",
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return self.model_dump()


#: Result of a single lookup: either a real record or an explicit absence.
MetadataRecord = ProductMetadata | MissingMetadata


@runtime_checkable
class MetadataLookup(Protocol):
    """The narrow lookup surface the rest of the system depends on.

    Implementations must be read-only, deterministic, network-free and must not know
    anything about recommendation, ranking or candidate generation.
    """

    def lookup(self, parent_asin: str) -> MetadataRecord:
        """Return metadata for one opaque identifier, or an explicit absence."""
        ...

    def lookup_many(self, parent_asins: Sequence[str]) -> tuple[MetadataRecord, ...]:
        """Return results positionally aligned with ``parent_asins``."""
        ...

    def __contains__(self, parent_asin: object) -> bool:
        """True when a metadata record exists for ``parent_asin``."""
        ...


#: Kept as a public alias so callers can type against the protocol name.
MetadataIndexProtocol = MetadataLookup

#: Metadata status for one enriched candidate: ``found`` or ``missing``.
CandidateMetadataStatus = Literal["found", "missing"]
