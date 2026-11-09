"""Catalogue-namespace contract for product identity (P14-0).

Why this module exists
----------------------
AgentRec-X has shipped three different product-identity spaces:

* the legacy Amazon Reviews 2023 *Sports_and_Outdoors* 5-core pipeline uses an
  integer ``item_id`` starting at **1**, with ``item_id == 0`` permanently reserved
  for padding (see ``AGENTS.md`` and ``recommendation/preprocess.py``);
* the P13-R research catalogue uses opaque Amazon ``parent_asin`` strings over an
  independently sampled *Electronics* catalogue (76,918 items, see
  ``data/p13r/electronics/manifest.json``);
* the P14 Shopping Companion will introduce its own catalogue and product ids.

The same numeric/string value can therefore name different products in different
catalogues.  P14-0 introduces the rule that an item reference must carry an explicit
**catalogue namespace**; nothing in this module changes any existing code path or
data mapping.  It only gives P14 (and future dataset/catalog/backend adapters) one
small, testable place to express and validate that rule.

Deliberately not included: catalogue registration, storage, I/O, backend adapters or
a plugin registry.  A namespace is a validated label; that is all.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Final, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "KNOWN_CATALOG_NAMESPACES",
    "LEGACY_INTEGER_NAMESPACE",
    "PAD_ITEM_ID",
    "SHOPPING_COMPANION_NAMESPACE",
    "CatalogNamespace",
    "NamespaceMismatchError",
    "ProductRef",
    "ensure_same_namespace",
]

#: The legacy integer identity space reserves ``0`` for padding.  No real product
#: in that namespace may use it (``AGENTS.md`` section 7).
PAD_ITEM_ID: Final = "0"

#: Legacy Sports namespace: integer item ids in the 5-core mapping.  Kept here as a
#: documented constant, not as a new mapping.
LEGACY_INTEGER_NAMESPACE: Final = "amazon_sports_5core"

#: P14-A Shopping Companion namespace: the official Lazada-derived product ids from
#: ``products.jsonl`` (1,298,797 products in the full catalogue).  The id is kept
#: verbatim as an opaque string; it is never assumed to be an Amazon ASIN and never
#: coerced into the legacy integer space.
SHOPPING_COMPANION_NAMESPACE: Final = "shopping_companion"

#: Namespaces that currently exist in the repository.  This is documentation, not a
#: closed registry: new P14 catalogues use a new valid name and add it here when they
#: are actually introduced (in the same change as their adapter).
KNOWN_CATALOG_NAMESPACES: Final[Mapping[str, str]] = MappingProxyType(
    {
        LEGACY_INTEGER_NAMESPACE: (
            "Amazon Reviews 2023 Sports_and_Outdoors 5-core mapping: integer item_id >= 1, "
            "item_id 0 is PAD; checkpoint identity is parent_asin <-> item_id"
        ),
        "amazon_electronics_c4": (
            "P13-R independent Amazon-C4 sampled Electronics catalogue (76,918 items): "
            "parent_asin identity, not the old Sports mapping and not the complete Electronics catalogue"
        ),
        "demo_catalog": (
            "synthetic demo catalogue used by the local demo / CI smoke (demo-* parent_asin values)"
        ),
        SHOPPING_COMPANION_NAMESPACE: (
            "P14-A official Shopping Companion Lazada product catalogue (products.jsonl, "
            "1,298,797 products in the full release; P14-A indexes a limited real sample): "
            "opaque string product_id kept verbatim, never an Amazon ASIN, never mixed with "
            "Sports integer ids or P13-R parent_asin"
        ),
    }
)

_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class NamespaceMismatchError(ValueError):
    """Raised when two product references from different catalogues are mixed."""


class CatalogNamespace(BaseModel):
    """A validated catalogue namespace label.

    The label is part of the product identity.  It is deliberately not inferred from
    the item id: callers state it explicitly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=2,
        max_length=64,
        description="lowercase dataset/catalog namespace, e.g. 'amazon_electronics_c4'",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not _NAMESPACE_PATTERN.fullmatch(value):
            raise ValueError(
                "catalog namespace must match ^[a-z][a-z0-9_]{1,63}$ "
                f"(lowercase, digits, underscore); got {value!r}"
            )
        return value

    @property
    def is_known(self) -> bool:
        """True when the namespace is one of the catalogues that exist today."""
        return self.name in KNOWN_CATALOG_NAMESPACES


class ProductRef(BaseModel):
    """A product reference that cannot be confused with another catalogue's item.

    ``item_id`` is opaque within its namespace.  It is not redefined or normalised
    here -- in particular this module does not build or cache any ``item_id`` mapping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: CatalogNamespace
    item_id: str = Field(min_length=1, max_length=256)
    parent_asin: str | None = Field(default=None, max_length=256)

    @field_validator("item_id")
    @classmethod
    def _validate_item_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("item_id must not carry leading/trailing whitespace")
        if ":" in value:
            raise ValueError(
                "item_id must not contain ':' -- the colon separates namespace from item id "
                "in the serialised form"
            )
        return value

    @field_validator("parent_asin")
    @classmethod
    def _validate_parent_asin(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("parent_asin must be non-empty when provided")
        return value

    @model_validator(mode="after")
    def _reject_legacy_padding(self) -> "ProductRef":
        if self.namespace.name == LEGACY_INTEGER_NAMESPACE and self.item_id == PAD_ITEM_ID:
            raise ValueError(
                "item_id 0 is the legacy padding sentinel; no real product may use it "
                f"in namespace {LEGACY_INTEGER_NAMESPACE!r}"
            )
        return self

    @property
    def namespaced_id(self) -> str:
        """Canonical serialised form ``"<namespace>:<item_id>"``."""
        return f"{self.namespace.name}:{self.item_id}"

    @classmethod
    def from_namespaced(cls, value: str) -> "ProductRef":
        """Parse the canonical ``"<namespace>:<item_id>"`` form.

        The item id may itself contain ':' only if this parser is not used -- prefer
        constructing :class:`ProductRef` with the fields directly for exotic ids.
        """
        parts = value.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"expected '<namespace>:<item_id>', got {value!r}"
            )
        namespace, item_id = parts
        return cls(namespace=CatalogNamespace(name=namespace), item_id=item_id)


def ensure_same_namespace(*refs: "ProductRef") -> CatalogNamespace:
    """Return the common namespace of ``refs`` or raise on a cross-catalogue mix."""
    if not refs:
        raise ValueError("ensure_same_namespace needs at least one ProductRef")
    namespace = refs[0].namespace
    for ref in refs[1:]:
        if ref.namespace.name != namespace.name:
            raise NamespaceMismatchError(
                f"mixed catalog namespaces: {namespace.name!r} and {ref.namespace.name!r} "
                "must never be resolved into one ranking"
            )
    return namespace
