"""P14-A namespace tests: ``shopping_companion`` identity contract.

The point of these tests is the *refusal* behaviour: a Shopping Companion product id
must never be re-interpreted as a Sports integer id or a P13-R Electronics ASIN, and
the legacy namespaces must keep working unchanged.
"""

from __future__ import annotations

import pytest

from recommendation.contracts.namespaces import (
    KNOWN_CATALOG_NAMESPACES,
    LEGACY_INTEGER_NAMESPACE,
    SHOPPING_COMPANION_NAMESPACE,
    CatalogNamespace,
    ProductRef,
    ensure_same_namespace,
)
from recommendation.shopping_companion.namespace import (
    SHOPPING_COMPANION,
    ShoppingCompanionIdentityError,
    is_shopping_companion_identity,
    namespaced_id,
    product_ref,
    raw_product_id,
    split_namespaced_id,
)


def test_shopping_companion_namespace_is_registered() -> None:
    assert SHOPPING_COMPANION_NAMESPACE == "shopping_companion"
    assert SHOPPING_COMPANION_NAMESPACE in KNOWN_CATALOG_NAMESPACES
    assert CatalogNamespace(name=SHOPPING_COMPANION_NAMESPACE).is_known is True
    assert SHOPPING_COMPANION.name == SHOPPING_COMPANION_NAMESPACE


def test_product_identity_keeps_official_id_verbatim() -> None:
    ref = product_ref("422373")
    assert ref.namespace.name == SHOPPING_COMPANION_NAMESPACE
    assert ref.item_id == "422373"
    assert ref.namespaced_id == "shopping_companion:422373"
    assert namespaced_id("422373") == "shopping_companion:422373"
    assert split_namespaced_id("shopping_companion:422373") == ("shopping_companion", "422373")
    assert raw_product_id("shopping_companion:422373") == "422373"
    assert raw_product_id("422373") == "422373"
    assert is_shopping_companion_identity("shopping_companion:422373") is True


@pytest.mark.parametrize(
    "foreign",
    [
        "amazon_sports_5core:12",
        "amazon_electronics_c4:B0000501GV",
        "demo_catalog:demo-socks",
        "amazon_sports_5core:0",
    ],
)
def test_foreign_namespace_ids_are_rejected(foreign: str) -> None:
    with pytest.raises(ShoppingCompanionIdentityError) as excinfo:
        split_namespaced_id(foreign)
    assert excinfo.value.code == "namespace_mismatch"
    with pytest.raises(ShoppingCompanionIdentityError):
        raw_product_id(foreign)
    assert is_shopping_companion_identity(foreign) is False


@pytest.mark.parametrize("bad", ["", "   ", "no-namespace-here", ":422373", "shopping_companion:"])
def test_malformed_identities_are_rejected(bad: str) -> None:
    with pytest.raises(ShoppingCompanionIdentityError):
        split_namespaced_id(bad)


def test_mixing_namespaces_in_one_ranking_is_rejected() -> None:
    sports = ProductRef(namespace=CatalogNamespace(name=LEGACY_INTEGER_NAMESPACE), item_id="12")
    shopping = product_ref("422373")
    with pytest.raises(ValueError):
        ensure_same_namespace(sports, shopping)


def test_legacy_padding_rule_is_unchanged() -> None:
    # The legacy namespace still forbids item id 0; the Shopping Companion namespace
    # treats ids as opaque strings and does not apply the integer padding rule.
    with pytest.raises(ValueError):
        ProductRef(namespace=CatalogNamespace(name=LEGACY_INTEGER_NAMESPACE), item_id="0")
    assert product_ref("0").item_id == "0"
