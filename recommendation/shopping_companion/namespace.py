"""Identity helpers for the P14-A ``shopping_companion`` catalogue namespace.

Every product touched by the Shopping Companion adapter is a
:class:`recommendation.contracts.namespaces.ProductRef` in the
``shopping_companion`` namespace.  The official product id is kept **verbatim** as
an opaque string: it is not an Amazon ASIN, it is never parsed as an integer and it
is never allowed to cross into the legacy Sports integer namespace or the P13-R
Electronics ``parent_asin`` namespace.
"""

from __future__ import annotations

from recommendation.contracts.namespaces import (
    CatalogNamespace,
    NamespaceMismatchError,
    ProductRef,
    SHOPPING_COMPANION_NAMESPACE,
)

__all__ = [
    "SHOPPING_COMPANION",
    "SHOPPING_COMPANION_NAMESPACE",
    "ShoppingCompanionIdentityError",
    "is_shopping_companion_identity",
    "namespaced_id",
    "product_ref",
    "raw_product_id",
    "split_namespaced_id",
]

#: The validated namespace object; construct once, reuse everywhere.
SHOPPING_COMPANION = CatalogNamespace(name=SHOPPING_COMPANION_NAMESPACE)

#: Separator used by the serialised identity form ``<namespace>:<item_id>``.
_SEPARATOR = ":"


class ShoppingCompanionIdentityError(ValueError):
    """A product identity is not a valid Shopping Companion identity."""

    def __init__(self, message: str, *, code: str = "invalid_identity") -> None:
        super().__init__(message)
        self.code = code


def product_ref(product_id: str, *, parent_asin: str | None = None) -> ProductRef:
    """Build a validated ``shopping_companion`` product reference from a raw id."""
    text = str(product_id or "").strip()
    if not text:
        raise ShoppingCompanionIdentityError("product_id must be non-empty", code="empty_product_id")
    try:
        return ProductRef(namespace=SHOPPING_COMPANION, item_id=text, parent_asin=parent_asin)
    except Exception as exc:  # pydantic validation error -> stable adapter error
        raise ShoppingCompanionIdentityError(
            f"invalid shopping_companion product_id {product_id!r}: {exc}",
            code="invalid_product_id",
        ) from exc


def namespaced_id(product_id: str) -> str:
    """Return the canonical ``shopping_companion:<product_id>`` identity string."""
    return product_ref(product_id).namespaced_id


def split_namespaced_id(value: str) -> tuple[str, str]:
    """Split a serialised identity and require the Shopping Companion namespace.

    Returns ``(namespace, raw_product_id)``.  A reference from any other catalogue
    raises :class:`ShoppingCompanionIdentityError` instead of being re-interpreted.
    """
    text = str(value or "").strip()
    if _SEPARATOR not in text:
        raise ShoppingCompanionIdentityError(
            f"expected '<namespace>:<product_id>', got {value!r}; a bare id is ambiguous",
            code="missing_namespace",
        )
    namespace, _, item_id = text.partition(_SEPARATOR)
    if namespace != SHOPPING_COMPANION_NAMESPACE:
        raise ShoppingCompanionIdentityError(
            f"identity {value!r} belongs to namespace {namespace!r}, not "
            f"{SHOPPING_COMPANION_NAMESPACE!r}",
            code="namespace_mismatch",
        )
    if not item_id.strip():
        raise ShoppingCompanionIdentityError(
            f"identity {value!r} has an empty product id", code="empty_product_id"
        )
    return namespace, item_id


def raw_product_id(value: str) -> str:
    """Return the raw official product id for a namespaced or bare identity.

    A namespaced value from a *different* namespace is rejected.  A bare value is
    returned stripped (the caller is inside the Shopping Companion adapter and the
    namespace is then attached explicitly).
    """
    text = str(value or "").strip()
    if not text:
        raise ShoppingCompanionIdentityError("product id must be non-empty", code="empty_product_id")
    if _SEPARATOR in text:
        _, item_id = split_namespaced_id(text)
        return item_id
    return text


def is_shopping_companion_identity(value: object) -> bool:
    """True when ``value`` is a serialised identity in our namespace (never raises)."""
    if not isinstance(value, str) or _SEPARATOR not in value:
        return False
    try:
        split_namespaced_id(value)
    except ShoppingCompanionIdentityError:
        return False
    return True
