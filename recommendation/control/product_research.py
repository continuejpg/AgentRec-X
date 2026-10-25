"""Read-only product research over the normalized catalogue (Phase 6A).

Two tools live here:

``inspect_product``
    Read one trusted ``parent_asin`` and return the catalogue's own facts, the explicit
    absences, and where each fact came from.

``compare_products``
    Read two to five trusted identities in the caller's order and compare only the attributes
    the catalogue can actually support.

The rules are deliberately stricter than a product page formatter:

* an unknown identity is an error, never a lookup miss mapped to another product;
* a missing attribute is ``missing``; an unparseable numeric string is ``unreadable``; neither
  is converted to zero, a default value, or a prose guess;
* raw catalogue values and derived numeric values are separate fields (``raw`` / ``value``);
* a comparison only runs when every product carries a value of the same field, type and unit;
* no "better" ranking is produced.  The numeric result is a deterministic min/max/equality
  observation, not a recommendation;
* input order is preserved and never sorted.

This module is independent of the action protocol and of a policy.  ``GroundedReasoner`` and
``ReasoningExecutor`` adapt it to the control plane.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .grounded_reasoning import parse_price, parse_weight_kg

__all__ = [
    "DEFAULT_ATTRIBUTES",
    "MAX_ATTRIBUTES",
    "MAX_COMPARE_PRODUCTS",
    "ProductComparison",
    "ProductComparisonRow",
    "ProductFact",
    "ProductInspection",
    "ProductResearchError",
    "ProductResearchTool",
    "SUPPORTED_ATTRIBUTES",
    "compare_records",
    "inspect_record",
]

MAX_COMPARE_PRODUCTS = 5
MAX_ATTRIBUTES = 12

#: Canonical names, in the order used when the caller does not name attributes.
SUPPORTED_ATTRIBUTES: tuple[str, ...] = (
    "title",
    "brand",
    "store",
    "main_category",
    "categories",
    "price",
    "color",
    "material",
    "features",
    "description",
    "weight",
    "average_rating",
)

DEFAULT_ATTRIBUTES: tuple[str, ...] = (
    "title",
    "brand",
    "store",
    "main_category",
    "categories",
    "price",
    "color",
    "material",
    "features",
    "description",
)

_ALIASES: dict[str, str] = {
    "title": "title",
    "name": "title",
    "brand": "brand",
    "brand name": "brand",
    "manufacturer": "brand",
    "store": "store",
    "seller": "store",
    "main_category": "main_category",
    "main category": "main_category",
    "category": "main_category",
    "categories": "categories",
    "category path": "categories",
    "price": "price",
    "price_text": "price",
    "color": "color",
    "colour": "color",
    "material": "material",
    "features": "features",
    "feature": "features",
    "description": "description",
    "weight": "weight",
    "item weight": "weight",
    "average_rating": "average_rating",
    "rating": "average_rating",
}

#: Fields that are derived from a different raw representation.  The distinction is reported
#: in every fact as ``derived`` and is never hidden by presenting the number alone.
_DERIVED_FIELDS = frozenset({"price", "weight"})


class ProductResearchError(ValueError):
    """A product-research request that cannot be answered without inventing a fact."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ProductFact(BaseModel):
    """One requested attribute for one product.

    ``status`` is the operational answer:

    * ``value``       - the catalogue carries a fact (raw or derived);
    * ``missing``     - the field is absent from the record;
    * ``unreadable``  - the field exists but its value cannot be parsed safely;
    * ``unsupported`` - the requested attribute is not in the declared surface.

    ``raw`` is always the catalogue-level value (or ``None``), while ``value`` is the display /
    parsed value.  ``derived`` distinguishes the two: a derived price is a number computed from
    ``price_text``; a raw title is not reinterpreted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attribute: str
    status: Literal["value", "missing", "unreadable", "unsupported"]
    raw: Any = None
    value: Any = None
    value_type: Literal["text", "list", "number", "none"] = "none"
    unit: str | None = None
    derived: bool = False
    source: str | None = None
    verification: Literal["catalog_value", "derived", "absent", "unreadable", "unsupported"]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @property
    def is_present(self) -> bool:
        return self.status == "value"


class ProductInspection(BaseModel):
    """The structured result of inspecting one catalogue identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_asin: str
    status: Literal["ok", "empty"] = "ok"
    facts: dict[str, ProductFact] = Field(default_factory=dict)
    requested_attributes: tuple[str, ...] = ()
    missing_attributes: tuple[str, ...] = ()
    unreadable_attributes: tuple[str, ...] = ()
    unsupported_attributes: tuple[str, ...] = ()
    catalog_source: str | None = None
    verification: Literal["catalog_verified"] = "catalog_verified"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ProductComparisonRow(BaseModel):
    """One compared attribute, with every product's fact and an explicit comparability verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attribute: str
    status: Literal["comparable", "partial", "not_comparable", "unsupported"]
    facts: dict[str, ProductFact] = Field(default_factory=dict)
    reason: str | None = None
    comparison: dict[str, Any] | None = None


class ProductComparison(BaseModel):
    """Two-to-five product comparison, in the caller's order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    products: tuple[str, ...] = ()
    status: Literal["ok", "empty"] = "ok"
    rows: tuple[ProductComparisonRow, ...] = ()
    requested_attributes: tuple[str, ...] = ()
    missing_attributes: tuple[str, ...] = ()
    unreadable_attributes: tuple[str, ...] = ()
    unsupported_attributes: tuple[str, ...] = ()
    catalog_source: str | None = None
    verification: Literal["catalog_verified"] = "catalog_verified"


# --------------------------------------------------------------------------- #
# Fact extraction
# --------------------------------------------------------------------------- #


def _attribute_text(record: Any, names: Sequence[str]) -> tuple[str | None, str | None]:
    details = getattr(record, "details", ()) or ()
    values = dict(details)
    for name in names:
        value = values.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip(), f"catalog:details:{name}"
    return None, None


def _raw_text(record: Any, field: str) -> tuple[Any, str | None]:
    value = getattr(record, field, None)
    if isinstance(value, str):
        value = value.strip() or None
    elif isinstance(value, (tuple, list)):
        value = [str(item) for item in value if str(item).strip()]
        value = value or None
    return value, f"catalog:{field}"


def _fact_for(record: Any, requested: str) -> ProductFact:
    canonical = _ALIASES.get(" ".join(str(requested).split()).casefold())
    if canonical is None:
        return ProductFact(
            attribute=str(requested),
            status="unsupported",
            verification="unsupported",
        )

    source: str | None
    if canonical == "brand":
        raw, source = _attribute_text(record, ("Brand Name", "Brand", "Manufacturer"))
    elif canonical == "color":
        raw, source = _attribute_text(record, ("Color",))
    elif canonical == "material":
        raw, source = _attribute_text(record, ("Material",))
    elif canonical == "weight":
        raw, source = _attribute_text(record, ("Item Weight", "Package Weight"))
    elif canonical == "price":
        raw, source = _raw_text(record, "price_text")
    elif canonical == "store":
        raw, source = _raw_text(record, "store")
    elif canonical == "main_category":
        raw, source = _raw_text(record, "main_category")
    elif canonical == "categories":
        raw, source = _raw_text(record, "categories")
    elif canonical == "features":
        raw, source = _raw_text(record, "features")
    elif canonical == "description":
        raw, source = _raw_text(record, "description")
    elif canonical == "average_rating":
        raw, source = _raw_text(record, "average_rating")
    else:  # title
        raw, source = _raw_text(record, "title")

    if raw is None:
        return ProductFact(
            attribute=canonical,
            status="missing",
            verification="absent",
            source=None,
        )

    if canonical == "price":
        parsed = parse_price(raw)
        if parsed is None:
            return ProductFact(
                attribute=canonical,
                status="unreadable",
                raw=raw,
                value_type="number",
                unit="catalog_price",
                derived=True,
                source=source,
                verification="unreadable",
            )
        return ProductFact(
            attribute=canonical,
            status="value",
            raw=raw,
            value=parsed,
            value_type="number",
            unit="catalog_price",
            derived=True,
            source=source,
            verification="derived",
        )

    if canonical == "weight":
        parsed = parse_weight_kg(raw)
        if parsed is None:
            return ProductFact(
                attribute=canonical,
                status="unreadable",
                raw=raw,
                value_type="number",
                unit="kg",
                derived=True,
                source=source,
                verification="unreadable",
            )
        return ProductFact(
            attribute=canonical,
            status="value",
            raw=raw,
            value=parsed,
            value_type="number",
            unit="kg",
            derived=True,
            source=source,
            verification="derived",
        )

    if canonical == "average_rating":
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            return ProductFact(
                attribute=canonical,
                status="unreadable",
                raw=raw,
                value_type="number",
                unit="rating",
                source=source,
                verification="unreadable",
            )
        if not math.isfinite(parsed):
            return ProductFact(
                attribute=canonical,
                status="unreadable",
                raw=raw,
                value_type="number",
                unit="rating",
                source=source,
                verification="unreadable",
            )
        return ProductFact(
            attribute=canonical,
            status="value",
            raw=raw,
            value=parsed,
            value_type="number",
            unit="rating",
            source=source,
            verification="catalog_value",
        )

    if isinstance(raw, list):
        return ProductFact(
            attribute=canonical,
            status="value",
            raw=raw,
            value=list(raw),
            value_type="list",
            source=source,
            verification="catalog_value",
        )

    return ProductFact(
        attribute=canonical,
        status="value",
        raw=raw,
        value=raw,
        value_type="text",
        source=source,
        verification="catalog_value",
    )


def _requested(attributes: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(str(value).strip() for value in (attributes or ()))
    if len(values) > MAX_ATTRIBUTES:
        raise ProductResearchError(
            f"at most {MAX_ATTRIBUTES} attributes may be requested",
            code="too_many_attributes",
        )
    return values or DEFAULT_ATTRIBUTES


def _record(records: Mapping[str, Any], parent_asin: str) -> Any:
    if not isinstance(parent_asin, str) or not parent_asin.strip():
        raise ProductResearchError("a product identity must be a non-empty string", code="invalid_product")
    identity = parent_asin.strip()
    if identity not in records:
        raise ProductResearchError(
            f"unknown product identity: {identity!r}",
            code="unknown_product",
        )
    return records[identity]


# --------------------------------------------------------------------------- #
# Public operations
# --------------------------------------------------------------------------- #


def inspect_record(
    records: Mapping[str, Any],
    parent_asin: str,
    *,
    attributes: Sequence[str] | None = None,
) -> ProductInspection:
    """Inspect one known identity; unknown identity raises, never substitutes."""
    record = _record(records, parent_asin)
    requested = _requested(attributes)

    facts: dict[str, ProductFact] = {}
    missing: list[str] = []
    unreadable: list[str] = []
    unsupported: list[str] = []
    for requested_name in requested:
        fact = _fact_for(record, requested_name)
        facts[requested_name] = fact
        if fact.status == "missing":
            missing.append(requested_name)
        elif fact.status == "unreadable":
            unreadable.append(requested_name)
        elif fact.status == "unsupported":
            unsupported.append(requested_name)

    status = "ok" if any(fact.is_present for fact in facts.values()) else "empty"
    source = getattr(record, "source", None)
    return ProductInspection(
        parent_asin=parent_asin.strip(),
        status=status,
        facts=facts,
        requested_attributes=requested,
        missing_attributes=tuple(missing),
        unreadable_attributes=tuple(unreadable),
        unsupported_attributes=tuple(unsupported),
        catalog_source=str(source) if isinstance(source, str) and source else None,
    )


def _comparable_row(
    attribute: str,
    facts: dict[str, ProductFact],
    order: tuple[str, ...],
) -> ProductComparisonRow:
    if any(fact.status == "unsupported" for fact in facts.values()):
        return ProductComparisonRow(
            attribute=attribute,
            status="unsupported",
            facts=facts,
            reason="the requested attribute is not in the declared catalogue surface",
        )

    present = [identity for identity in order if facts[identity].is_present]
    missing = [identity for identity in order if facts[identity].status == "missing"]
    unreadable = [identity for identity in order if facts[identity].status == "unreadable"]

    if not present:
        return ProductComparisonRow(
            attribute=attribute,
            status="not_comparable",
            facts=facts,
            reason="no product carries a value for this attribute",
        )
    if missing or unreadable:
        reasons = []
        if missing:
            reasons.append("missing on " + ", ".join(missing))
        if unreadable:
            reasons.append("unreadable on " + ", ".join(unreadable))
        return ProductComparisonRow(
            attribute=attribute,
            status="partial",
            facts=facts,
            reason="; ".join(reasons),
        )

    value_types = {facts[identity].value_type for identity in order}
    units = {facts[identity].unit for identity in order}
    if len(value_types) != 1:
        return ProductComparisonRow(
            attribute=attribute,
            status="not_comparable",
            facts=facts,
            reason="the values do not share one value type",
        )
    if len(units) != 1:
        return ProductComparisonRow(
            attribute=attribute,
            status="not_comparable",
            facts=facts,
            reason="the values do not share one unit",
        )

    value_type = next(iter(value_types))
    unit = next(iter(units))
    if value_type == "number":
        values = {identity: facts[identity].value for identity in order}
        numeric = [float(values[identity]) for identity in order]
        minimum = min(numeric)
        maximum = max(numeric)
        comparison = {
            "kind": "numeric",
            "unit": unit,
            "min": minimum,
            "max": maximum,
            "min_products": [identity for identity in order if float(values[identity]) == minimum],
            "max_products": [identity for identity in order if float(values[identity]) == maximum],
            "values": {identity: values[identity] for identity in order},
        }
    else:
        values = {identity: facts[identity].value for identity in order}
        rendered = [str(values[identity]) for identity in order]
        comparison = {
            "kind": "equality",
            "all_equal": len(set(rendered)) == 1,
            "distinct_values": sorted(set(rendered)),
            "values": {identity: values[identity] for identity in order},
        }
    return ProductComparisonRow(
        attribute=attribute,
        status="comparable",
        facts=facts,
        comparison=comparison,
    )


def compare_records(
    records: Mapping[str, Any],
    parent_asins: Sequence[str],
    *,
    attributes: Sequence[str] | None = None,
) -> ProductComparison:
    """Compare 2–5 known identities in the caller's order.

    Duplicate identities and out-of-range product counts are rejected: a comparison against
    the same product twice would be meaningless, and silently collapsing it would hide a
    malformed request.
    """
    order = tuple(str(value).strip() for value in parent_asins)
    if not 2 <= len(order) <= MAX_COMPARE_PRODUCTS:
        raise ProductResearchError(
            f"comparison needs 2..{MAX_COMPARE_PRODUCTS} products",
            code="invalid_product_count",
        )
    if any(not identity for identity in order):
        raise ProductResearchError("product identities must be non-empty", code="invalid_product")
    if len(set(order)) != len(order):
        raise ProductResearchError("duplicate product identities cannot be compared", code="duplicate_product")

    requested = _requested(attributes)
    records_by_identity = {
        identity: _record(records, identity)
        for identity in order
    }
    source = getattr(next(iter(records_by_identity.values())), "source", None)
    rows: list[ProductComparisonRow] = []
    missing_attributes: list[str] = []
    unreadable_attributes: list[str] = []
    unsupported_attributes: list[str] = []
    for requested_name in requested:
        facts = {identity: _fact_for(records_by_identity[identity], requested_name) for identity in order}
        row = _comparable_row(requested_name, facts, order)
        rows.append(row)
        if row.status == "unsupported":
            unsupported_attributes.append(requested_name)
        else:
            if any(facts[identity].status == "missing" for identity in order):
                missing_attributes.append(requested_name)
            if any(facts[identity].status == "unreadable" for identity in order):
                unreadable_attributes.append(requested_name)

    return ProductComparison(
        products=order,
        status="ok" if any(row.status in {"comparable", "partial"} for row in rows) else "empty",
        rows=tuple(rows),
        requested_attributes=requested,
        missing_attributes=tuple(missing_attributes),
        unreadable_attributes=tuple(unreadable_attributes),
        unsupported_attributes=tuple(unsupported_attributes),
        catalog_source=str(source) if isinstance(source, str) and source else None,
    )


class ProductResearchTool:
    """A narrow, read-only tool over a catalogue records mapping.

    The caller supplies the trusted record mapping (normally ``MetadataIndex.records``); the
    tool never loads a second catalogue and never resolves an identity through a different
    map.  The action executor validates run membership before calling this tool.
    """

    def __init__(self, records: Mapping[str, Any]) -> None:
        self._records = records

    @property
    def catalogue_size(self) -> int:
        return len(self._records)

    def inspect_product(
        self, parent_asin: str, *, attributes: Sequence[str] | None = None
    ) -> ProductInspection:
        return inspect_record(self._records, parent_asin, attributes=attributes)

    def compare_products(
        self,
        parent_asins: Sequence[str],
        *,
        attributes: Sequence[str] | None = None,
    ) -> ProductComparison:
        return compare_records(self._records, parent_asins, attributes=attributes)
