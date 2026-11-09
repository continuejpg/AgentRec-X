"""Regression tests for the P14-0 catalogue-namespace contract.

The contract only pins product-identity rules; these tests deliberately assert the
*rejections* (invalid namespace, cross-catalogue mixing, legacy padding id) because
silent id mixing is exactly the failure mode P14-0 must prevent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from recommendation.contracts.namespaces import (
    KNOWN_CATALOG_NAMESPACES,
    LEGACY_INTEGER_NAMESPACE,
    PAD_ITEM_ID,
    CatalogNamespace,
    NamespaceMismatchError,
    ProductRef,
    ensure_same_namespace,
)


def test_known_namespaces_are_valid_and_documented() -> None:
    assert LEGACY_INTEGER_NAMESPACE in KNOWN_CATALOG_NAMESPACES
    assert PAD_ITEM_ID == "0"
    for name, description in KNOWN_CATALOG_NAMESPACES.items():
        namespace = CatalogNamespace(name=name)
        assert namespace.name == name
        assert namespace.is_known is True
        assert description.strip()


@pytest.mark.parametrize(
    "bad_name",
    ["", "x", "Electronics", "9sports", "a b", "sports:c4", "x" * 65, "sports-c4"],
)
def test_invalid_namespace_is_rejected(bad_name: str) -> None:
    with pytest.raises(ValidationError):
        CatalogNamespace(name=bad_name)


def test_unknown_but_valid_namespace_is_allowed_for_p14() -> None:
    namespace = CatalogNamespace(name="p14_shopping_companion")
    assert namespace.name == "p14_shopping_companion"
    assert namespace.is_known is False


def test_namespaced_id_roundtrip() -> None:
    ref = ProductRef(
        namespace=CatalogNamespace(name="amazon_electronics_c4"),
        item_id="B0000501GV",
        parent_asin="B0000501GV",
    )
    assert ref.namespaced_id == "amazon_electronics_c4:B0000501GV"
    parsed = ProductRef.from_namespaced(ref.namespaced_id)
    assert parsed.namespace == ref.namespace
    assert parsed.item_id == ref.item_id
    # The serialised identity form carries namespace + item_id only; parent_asin is
    # catalogue metadata and stays out of the identity string.
    assert parsed.parent_asin is None
    assert ref.namespace.is_known is True


@pytest.mark.parametrize("bad_item_id", ["", " B0000501GV", "B0000501GV ", "a:b"])
def test_item_id_is_validated(bad_item_id: str) -> None:
    with pytest.raises(ValidationError):
        ProductRef(namespace=CatalogNamespace(name="demo_catalog"), item_id=bad_item_id)


def test_legacy_padding_item_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProductRef(namespace=CatalogNamespace(name=LEGACY_INTEGER_NAMESPACE), item_id=PAD_ITEM_ID)


def test_zero_is_only_forbidden_in_the_legacy_integer_namespace() -> None:
    # "0" is an ordinary opaque id in a catalogue that has no integer padding rule.
    ref = ProductRef(namespace=CatalogNamespace(name="demo_catalog"), item_id="0")
    assert ref.item_id == "0"


@pytest.mark.parametrize(
    "bad_value",
    ["", "no-colon", ":B0000501GV", "amazon_electronics_c4:", "a:b:c", "Bad Namespace:x"],
)
def test_from_namespaced_rejects_malformed_values(bad_value: str) -> None:
    with pytest.raises(ValueError):
        ProductRef.from_namespaced(bad_value)


def test_ensure_same_namespace_returns_common_namespace() -> None:
    namespace = CatalogNamespace(name="demo_catalog")
    refs = [ProductRef(namespace=namespace, item_id=value) for value in ("a", "b")]
    assert ensure_same_namespace(*refs) == namespace
    with pytest.raises(ValueError):
        ensure_same_namespace()


def test_mixing_catalogues_is_rejected() -> None:
    sports = ProductRef(namespace=CatalogNamespace(name=LEGACY_INTEGER_NAMESPACE), item_id="12")
    electronics = ProductRef(namespace=CatalogNamespace(name="amazon_electronics_c4"), item_id="B0000501GV")
    with pytest.raises(NamespaceMismatchError, match="mixed catalog namespaces"):
        ensure_same_namespace(sports, electronics)


def test_product_ref_is_frozen() -> None:
    ref = ProductRef(namespace=CatalogNamespace(name="demo_catalog"), item_id="demo-socks")
    with pytest.raises(ValidationError):
        ref.item_id = "demo-other"  # type: ignore[misc]
