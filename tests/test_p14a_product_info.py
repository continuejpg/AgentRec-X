"""P14-A Product Info tests: real catalogue facts, explicit UNKNOWN, namespace refusal."""

from __future__ import annotations

import pytest

from recommendation.shopping_companion.catalogue import ShoppingCompanionCatalogue
from recommendation.shopping_companion.product_info import (
    ShoppingCompanionProductInfoError,
    ShoppingCompanionProductInfoTool,
)

DOCS = [
    {
        "id": "2001",
        "contents": "Full Record Product\nbrand: Acme",
        "product": {
            "product_id": "2001",
            "product_name": "Full Record Product",
            "seller_id": "s9",
            "seller_name": "Acme Store",
            "category": "Home - Kitchen - Kettles",
            "price": 23.45,
            "attributes": {"brand": ["Acme"], "capacity": ["1.7"]},
            "options": [{"color": ["Steel"]}],
            "description": "A stainless steel kettle with a large capacity.",
        },
    },
    {
        "id": "2002",
        "contents": "Sparse Product",
        "product": {
            "product_id": "2002",
            "product_name": "Sparse Product",
            "seller_id": None,
            "seller_name": None,
            "category": None,
            "price": None,
            "attributes": {},
            "options": [],
            "description": None,
        },
    },
]


@pytest.fixture()
def catalogue() -> ShoppingCompanionCatalogue:
    return ShoppingCompanionCatalogue.from_docs(
        DOCS, manifest={"selection_method": "LIMITED catalogue"}
    )


@pytest.fixture()
def tool(catalogue: ShoppingCompanionCatalogue) -> ShoppingCompanionProductInfoTool:
    return ShoppingCompanionProductInfoTool(catalogue)


def test_full_record_is_served_verbatim(tool: ShoppingCompanionProductInfoTool) -> None:
    result = tool.inspect("shopping_companion:2001")
    assert result.status == "ok"
    assert result.product_id == "2001"
    assert result.namespaced_id == "shopping_companion:2001"
    assert result.name == "Full Record Product"
    assert result.category == "Home - Kitchen - Kettles"
    assert result.price is not None
    assert result.price["value"] == 23.45
    assert result.price["status"] == "value"
    assert result.price["currency"] == "UNKNOWN"
    assert result.seller["seller_name"] == "Acme Store"
    assert result.attributes == {"brand": ["Acme"], "capacity": ["1.7"]}
    assert result.options == [{"color": ["Steel"]}]
    assert result.description["status"] == "value"
    assert result.description["summary"].startswith("A stainless steel kettle")
    assert result.provenance["verification"] == "catalog_verified"
    assert any("not live" in note for note in result.notes)


def test_currency_is_unknown_not_invented(tool: ShoppingCompanionProductInfoTool) -> None:
    result = tool.inspect("2001", attributes=["price", "currency"])
    assert result.facts["price"]["status"] == "value"
    assert result.facts["currency"]["status"] == "missing"
    assert "currency" in result.unknown_fields


def test_missing_fields_stay_missing(tool: ShoppingCompanionProductInfoTool) -> None:
    result = tool.inspect("2002")
    assert result.status == "ok"
    assert result.price is not None and result.price["status"] == "missing"
    assert result.seller is not None and result.seller["status"] == "missing"
    assert result.description is not None and result.description["status"] == "missing"
    assert "price" in result.missing_fields
    assert result.provenance["verification"] == "catalog_verified"


def test_unknown_attribute_is_unsupported(tool: ShoppingCompanionProductInfoTool) -> None:
    result = tool.inspect("2001", attributes=["name", "battery_life"])
    assert result.facts["name"]["status"] == "value"
    assert result.facts["battery_life"]["status"] == "unsupported"
    assert "battery_life" in result.unsupported_fields


def test_absent_identity_is_reported_not_guessed(tool: ShoppingCompanionProductInfoTool) -> None:
    result = tool.inspect("999999")
    assert result.status == "not_found"
    assert result.namespaced_id is None
    assert "product_id" in result.unknown_fields
    assert result.provenance["verification"] == "identity_absent_from_loaded_catalogue"


@pytest.mark.parametrize(
    "foreign",
    ["amazon_sports_5core:12", "amazon_electronics_c4:B0000501GV", "demo_catalog:demo-socks"],
)
def test_cross_namespace_request_is_refused(tool: ShoppingCompanionProductInfoTool, foreign: str) -> None:
    with pytest.raises(ShoppingCompanionProductInfoError) as excinfo:
        tool.inspect(foreign)
    assert excinfo.value.code == "namespace_mismatch"
    with pytest.raises(ShoppingCompanionProductInfoError) as excinfo:
        tool.inspect("2001", namespace="amazon_sports_5core")
    assert excinfo.value.code == "namespace_mismatch"


def test_empty_product_id_is_refused(tool: ShoppingCompanionProductInfoTool) -> None:
    with pytest.raises(ShoppingCompanionProductInfoError):
        tool.inspect("")


def test_repeated_queries_are_consistent(tool: ShoppingCompanionProductInfoTool) -> None:
    first = tool.inspect("2001").as_dict()
    second = tool.inspect("2001").as_dict()
    assert first == second
