"""P14-A Product Info tool for the Shopping Companion catalogue.

This is the ``product_view`` / Product-Info capability from the P14-0 interface
contract.  It reads **only** the real catalogue record and never invents an
attribute: a value the catalogue snapshot does not carry is reported as
``missing``/``unknown``, never filled from the model's knowledge and never inferred
from another field.

It also enforces the catalogue namespace: a request for a product id in a different
namespace (Sports integer space, P13-R Electronics ``parent_asin``, ...) is refused
rather than re-interpreted as a Shopping Companion product.

The tool reuses the accepted product-fact vocabulary
(:class:`recommendation.control.product_research.ProductFact`) for per-attribute
facts, so P14-B/C consumers see the same four-state semantics
(``value``/``missing``/``unreadable``/``unsupported``) as the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from recommendation.control.product_research import ProductFact

from .catalogue import ShoppingCompanionCatalogue, ShoppingCompanionProduct
from .namespace import (
    SHOPPING_COMPANION_NAMESPACE,
    ShoppingCompanionIdentityError,
    raw_product_id,
)

__all__ = [
    "DEFAULT_ATTRIBUTES",
    "ProductInfoResult",
    "ShoppingCompanionProductInfoError",
    "ShoppingCompanionProductInfoTool",
]

#: Attributes the tool can serve from the Shopping Companion catalogue snapshot.
DEFAULT_ATTRIBUTES: tuple[str, ...] = (
    "name",
    "category",
    "price",
    "seller",
    "attributes",
    "options",
    "description",
)

#: Canonical attribute -> catalogue field mapping (closed surface; anything else is
#: reported as ``unsupported`` rather than guessed).
_ATTRIBUTE_FIELDS: dict[str, str] = {
    "name": "title",
    "title": "title",
    "category": "category",
    "main_category": "category",
    "price": "price",
    "currency": "currency",
    "seller": "seller",
    "shop": "seller",
    "seller_id": "seller_id",
    "seller_name": "seller_name",
    "attributes": "attributes",
    "options": "options",
    "description": "description",
    "rating": "average_rating",
    "average_rating": "average_rating",
}

#: Fields the official product snapshot does not provide at all.
_UNAVAILABLE_FIELDS: dict[str, str] = {
    "currency": "the catalogue snapshot stores a numeric price without a currency",
    "stock": "the catalogue snapshot carries no stock/availability information",
    "live_availability": "the catalogue snapshot is not a live marketplace feed",
}


class ShoppingCompanionProductInfoError(ValueError):
    """An invalid Product-Info request (namespace, identity or attribute surface)."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _normalise_facts(facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return per-attribute facts as plain JSON-safe dicts.

    ``_fact`` builds the accepted :class:`ProductFact` pydantic object; the response
    model stores plain dicts so a Product-Info result is directly JSON-serialisable
    inside a trajectory.
    """
    normalised: dict[str, dict[str, Any]] = {}
    for key, value in facts.items():
        normalised[key] = value.as_dict() if isinstance(value, ProductFact) else value
    return normalised


@dataclass(frozen=True)
class ProductInfoResult:
    """The structured Product-Info response for one Shopping Companion product."""

    namespace: str
    product_id: str | None
    namespaced_id: str | None
    status: str
    name: str | None = None
    category: str | None = None
    price: dict[str, Any] | None = None
    seller: dict[str, Any] | None = None
    attributes: dict[str, list[str]] = field(default_factory=dict)
    options: list[dict[str, list[str]]] = field(default_factory=list)
    description: dict[str, Any] | None = None
    requested_attributes: tuple[str, ...] = ()
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    unsupported_fields: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "agentrecx.p14a.product_info.v1",
            "namespace": self.namespace,
            "product_id": self.product_id,
            "namespaced_id": self.namespaced_id,
            "status": self.status,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "seller": self.seller,
            "attributes": self.attributes,
            "options": self.options,
            "description": self.description,
            "requested_attributes": list(self.requested_attributes),
            "facts": self.facts,
            "missing_fields": list(self.missing_fields),
            "unknown_fields": list(self.unknown_fields),
            "unsupported_fields": list(self.unsupported_fields),
            "provenance": self.provenance,
            "notes": list(self.notes),
        }


def _price_field(product: ShoppingCompanionProduct) -> dict[str, Any]:
    if product.price is None:
        return {
            "status": "missing",
            "value": None,
            "currency": "UNKNOWN",
            "currency_status": "unknown",
            "source": "catalog:product.price",
            "note": "the catalogue snapshot carries no price for this product",
        }
    return {
        "status": "value",
        "value": product.price,
        "currency": "UNKNOWN",
        "currency_status": "unknown",
        "source": "catalog:product.price",
        "note": "numeric snapshot price; the catalogue does not provide a currency field",
    }


def _seller_field(product: ShoppingCompanionProduct) -> dict[str, Any]:
    if product.seller_name is None and product.seller_id is None:
        return {
            "status": "missing",
            "seller_id": None,
            "seller_name": None,
            "source": "catalog:product.seller_id/seller_name",
        }
    return {
        "status": "value" if product.seller_name is not None else "partial",
        "seller_id": product.seller_id,
        "seller_name": product.seller_name,
        "source": "catalog:product.seller_id/seller_name",
    }


def _fact(attribute: str, *, status: str, raw: Any = None, value: Any = None,
          value_type: str = "none", source: str | None = None,
          verification: str = "absent") -> ProductFact:
    return ProductFact(
        attribute=attribute,
        status=status,  # type: ignore[arg-type]  # Literal validated by pydantic
        raw=raw,
        value=value,
        value_type=value_type,  # type: ignore[arg-type]
        derived=False,
        source=source,
        verification=verification,  # type: ignore[arg-type]
    )


class ShoppingCompanionProductInfoTool:
    """Read-only Product Info over one :class:`ShoppingCompanionCatalogue`."""

    def __init__(
        self,
        catalogue: ShoppingCompanionCatalogue,
        *,
        description_chars: int = 600,
    ) -> None:
        if description_chars < 1:
            raise ValueError("description_chars must be >= 1")
        self._catalogue = catalogue
        self._description_chars = description_chars

    @property
    def namespace(self) -> str:
        return SHOPPING_COMPANION_NAMESPACE

    def inspect(
        self,
        product_id: str,
        *,
        namespace: str = SHOPPING_COMPANION_NAMESPACE,
        attributes: Sequence[str] | None = None,
    ) -> ProductInfoResult:
        """Return real catalogue facts for one product id.

        A namespace mismatch is a refused request, not a lookup: this is the rule that
        prevents Sports integers or P13-R ASINs from being silently interpreted as
        Shopping Companion products.
        """
        requested_namespace = str(namespace or "").strip()
        if requested_namespace != SHOPPING_COMPANION_NAMESPACE:
            raise ShoppingCompanionProductInfoError(
                f"Product Info refuses namespace {requested_namespace!r}; this catalogue only "
                f"resolves {SHOPPING_COMPANION_NAMESPACE!r}",
                code="namespace_mismatch",
            )
        text = str(product_id or "").strip()
        if not text:
            raise ShoppingCompanionProductInfoError("product_id must be non-empty", code="empty_product_id")
        try:
            raw_id = raw_product_id(text)
        except ShoppingCompanionIdentityError as exc:
            raise ShoppingCompanionProductInfoError(str(exc), code=exc.code) from exc

        requested = tuple(attributes) if attributes is not None else DEFAULT_ATTRIBUTES
        unknown_fields: list[str] = []
        missing: list[str] = []
        unsupported: list[str] = []
        facts: dict[str, dict[str, Any]] = {}

        product = self._catalogue.get(raw_id)
        if product is None:
            for attribute in requested:
                mapped = _ATTRIBUTE_FIELDS.get(str(attribute).strip().lower())
                if mapped is None:
                    unsupported.append(str(attribute))
                    facts[str(attribute)] = _fact(
                        str(attribute), status="unsupported", verification="unsupported"
                    ).as_dict()
                    continue
                facts[str(attribute)] = _fact(
                    str(attribute), status="missing", source=f"catalog:{mapped}", verification="absent"
                ).as_dict()
                missing.append(str(attribute))
            return ProductInfoResult(
                namespace=SHOPPING_COMPANION_NAMESPACE,
                product_id=raw_id,
                namespaced_id=None,
                status="not_found",
                requested_attributes=tuple(str(a) for a in requested),
                facts=_normalise_facts(facts),
                missing_fields=tuple(missing),
                unknown_fields=("product_id",),
                unsupported_fields=tuple(unsupported),
                provenance={
                    "source": "shopping_companion/products.jsonl (limited catalogue)",
                    "catalogue_size": self._catalogue.catalogue_size,
                    "catalogue_limited": self._catalogue.limited,
                    "catalogue_docs_path": self._catalogue.docs_path,
                    "verification": "identity_absent_from_loaded_catalogue",
                },
                notes=(
                    "the product id is not present in the loaded limited catalogue; "
                    "this is an absent-identity result, not a claim that the product does not exist",
                ),
            )

        # The record exists: build the real fields.
        price = _price_field(product)
        seller = _seller_field(product)
        description_status = "value" if product.description else "missing"
        description = {
            "status": description_status,
            "summary": product.description_truncated[: self._description_chars] if product.description else None,
            "truncated": bool(product.description and len(product.description) > self._description_chars),
            "source": "catalog:product.description",
        }
        attribute_map = {key: list(values) for key, values in product.attributes}
        option_list = [{key: list(values) for key, values in option.items()} for option in product.options]

        for attribute in requested:
            name = str(attribute).strip().lower()
            mapped = _ATTRIBUTE_FIELDS.get(name)
            if mapped is None:
                unsupported.append(str(attribute))
                facts[str(attribute)] = _fact(
                    str(attribute), status="unsupported", verification="unsupported"
                ).as_dict()
                continue
            if mapped == "title":
                value = product.title
                facts[str(attribute)] = (
                    _fact(str(attribute), status="value", raw=value, value=value, value_type="text",
                          source="catalog:product.product_name", verification="catalog_value")
                    if value else _fact(str(attribute), status="missing", source="catalog:product.product_name")
                )
                if not value:
                    missing.append(str(attribute))
                continue
            if mapped == "category":
                value = product.category
                facts[str(attribute)] = (
                    _fact(str(attribute), status="value", raw=value, value=value, value_type="text",
                          source="catalog:product.category", verification="catalog_value")
                    if value else _fact(str(attribute), status="missing", source="catalog:product.category")
                )
                if not value:
                    missing.append(str(attribute))
                continue
            if mapped == "price":
                if product.price is None:
                    facts[str(attribute)] = _fact(str(attribute), status="missing", source="catalog:product.price")
                    missing.append(str(attribute))
                else:
                    facts[str(attribute)] = _fact(
                        str(attribute), status="value", raw=product.price, value=product.price,
                        value_type="number", source="catalog:product.price", verification="catalog_value",
                    )
                continue
            if mapped == "currency":
                unknown_fields.append(str(attribute))
                facts[str(attribute)] = _fact(
                    str(attribute), status="missing", source="catalog:product.price"
                )
                continue
            if mapped in {"seller", "shop"}:
                if product.seller_name is None and product.seller_id is None:
                    facts[str(attribute)] = _fact(
                        str(attribute), status="missing", source="catalog:product.seller_id/seller_name"
                    )
                    missing.append(str(attribute))
                else:
                    facts[str(attribute)] = _fact(
                        str(attribute), status="value", raw=product.seller_name,
                        value=product.seller_name or product.seller_id, value_type="text",
                        source="catalog:product.seller_id/seller_name", verification="catalog_value",
                    )
                continue
            if mapped == "seller_id":
                facts[str(attribute)] = (
                    _fact(str(attribute), status="value", raw=product.seller_id, value=product.seller_id,
                          value_type="text", source="catalog:product.seller_id", verification="catalog_value")
                    if product.seller_id else
                    _fact(str(attribute), status="missing", source="catalog:product.seller_id")
                )
                if not product.seller_id:
                    missing.append(str(attribute))
                continue
            if mapped == "seller_name":
                facts[str(attribute)] = (
                    _fact(str(attribute), status="value", raw=product.seller_name, value=product.seller_name,
                          value_type="text", source="catalog:product.seller_name", verification="catalog_value")
                    if product.seller_name else
                    _fact(str(attribute), status="missing", source="catalog:product.seller_name")
                )
                if not product.seller_name:
                    missing.append(str(attribute))
                continue
            if mapped == "attributes":
                facts[str(attribute)] = _fact(
                    str(attribute), status="value" if attribute_map else "missing",
                    raw=attribute_map, value=attribute_map, value_type="list",
                    source="catalog:product.attributes",
                    verification="catalog_value" if attribute_map else "absent",
                )
                if not attribute_map:
                    missing.append(str(attribute))
                continue
            if mapped == "options":
                facts[str(attribute)] = _fact(
                    str(attribute), status="value" if option_list else "missing",
                    raw=option_list, value=option_list, value_type="list",
                    source="catalog:product.options",
                    verification="catalog_value" if option_list else "absent",
                )
                if not option_list:
                    missing.append(str(attribute))
                continue
            if mapped == "description":
                facts[str(attribute)] = _fact(
                    str(attribute), status="value" if product.description else "missing",
                    raw=product.description, value=product.description_truncated,
                    value_type="text", source="catalog:product.description",
                    verification="catalog_value" if product.description else "absent",
                )
                if not product.description:
                    missing.append(str(attribute))
                continue
            # average_rating and any other declared-but-unavailable field
            unknown_fields.append(str(attribute))
            facts[str(attribute)] = _fact(
                str(attribute), status="missing", source=f"catalog:{mapped}"
            )

        unavailable_notes = tuple(
            f"{name}: {reason}" for name, reason in _UNAVAILABLE_FIELDS.items() if name in requested
        )
        return ProductInfoResult(
            namespace=SHOPPING_COMPANION_NAMESPACE,
            product_id=product.product_id,
            namespaced_id=product.namespaced_id,
            status="ok",
            name=product.title,
            category=product.category,
            price=price,
            seller=seller,
            attributes=attribute_map,
            options=option_list,
            description=description,
            requested_attributes=tuple(str(a) for a in requested),
            facts=_normalise_facts(facts),
            missing_fields=tuple(sorted(set(missing))),
            unknown_fields=tuple(sorted(set(unknown_fields))),
            unsupported_fields=tuple(sorted(set(unsupported))),
            provenance={
                "source": "shopping_companion/products.jsonl (limited catalogue)",
                "catalogue_size": self._catalogue.catalogue_size,
                "catalogue_limited": self._catalogue.limited,
                "catalogue_docs_path": self._catalogue.docs_path,
                "product_id_raw": product.product_id,
                "verification": "catalog_verified",
            },
            notes=(
                "values are read from the official research catalogue snapshot; they are not live "
                "marketplace availability, price or stock information",
                *unavailable_notes,
            ),
        )
