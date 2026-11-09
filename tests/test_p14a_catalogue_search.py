"""P14-A catalogue + real-search tool tests on an in-memory fixture catalogue."""

from __future__ import annotations

import pytest

from recommendation.shopping_companion.catalogue import (
    ShoppingCompanionCatalogue,
    product_from_doc,
)
from recommendation.shopping_companion.namespace import ShoppingCompanionIdentityError
from recommendation.shopping_companion.search import (
    ShoppingCompanionSearchError,
    ShoppingCompanionSearchTool,
    parse_price_filter,
)

DOCS = [
    {
        "id": "1001",
        "contents": "Acme Wireless Mouse\nbrand: Acme\ncolor: Black",
        "product": {
            "product_id": "1001",
            "product_name": "Acme Wireless Mouse",
            "seller_id": "s1",
            "seller_name": "Acme Store",
            "category": "Electronics - Computer Accessories - Mice",
            "price": 15.5,
            "attributes": {"brand": ["Acme"], "color": ["Black"]},
            "options": [{"size": ["M"]}],
            "description": "A compact wireless mouse.",
        },
    },
    {
        "id": "1002",
        "contents": "Beta Wireless Mouse Pro\nbrand: Beta",
        "product": {
            "product_id": "1002",
            "product_name": "Beta Wireless Mouse Pro",
            "seller_id": "s2",
            "seller_name": "Beta Shop",
            "category": "Electronics - Computer Accessories - Mice",
            "price": 42.0,
            "attributes": {"brand": ["Beta"]},
            "options": [],
            "description": "A premium wireless mouse.",
        },
    },
    {
        "id": "1003",
        "contents": "Gamma Wired Mouse\nbrand: Gamma",
        "product": {
            "product_id": "1003",
            "product_name": "Gamma Wired Mouse",
            "seller_id": "s3",
            "seller_name": "Gamma Store",
            "category": "Electronics - Computer Accessories - Mice",
            "price": 9.0,
            "attributes": {"brand": ["Gamma"]},
            "options": [],
            "description": "A simple wired mouse.",
        },
    },
]

MANIFEST = {
    "selection_method": "contiguous byte prefix; LIMITED catalogue, NOT the full catalogue",
    "docs_sha256": "fixture",
}


@pytest.fixture()
def catalogue() -> ShoppingCompanionCatalogue:
    return ShoppingCompanionCatalogue.from_docs(DOCS, manifest=MANIFEST)


def test_catalogue_loads_real_schema_and_keeps_identity(catalogue: ShoppingCompanionCatalogue) -> None:
    assert catalogue.catalogue_size == 3
    assert catalogue.limited is True
    product = catalogue.get("shopping_companion:1001")
    assert product is not None
    assert product.product_id == "1001"
    assert product.title == "Acme Wireless Mouse"
    assert product.seller_name == "Acme Store"
    assert product.price == 15.5
    assert product.category_path[0] == "Electronics"
    # the accepted ProductMetadata contract is what the ledger/reasoner see
    record = catalogue.records["shopping_companion:1001"]
    assert record.parent_asin == "shopping_companion:1001"
    assert record.title == "Acme Wireless Mouse"
    assert record.price_text == "15.5"
    assert ("brand", "Acme") in record.details


def test_catalogue_membership_and_cross_namespace_lookup(catalogue: ShoppingCompanionCatalogue) -> None:
    assert "shopping_companion:1001" in catalogue
    assert "shopping_companion:9999" not in catalogue
    assert "not-a-namespaced-id" not in catalogue
    with pytest.raises(ShoppingCompanionIdentityError):
        catalogue.get("amazon_sports_5core:12")


def test_product_from_doc_rejects_missing_identity() -> None:
    with pytest.raises(ShoppingCompanionIdentityError):
        product_from_doc({"id": "", "product": {"product_id": "", "product_name": "x"}})
    with pytest.raises(ShoppingCompanionIdentityError):
        product_from_doc({"contents": "no product object"})


def test_search_returns_real_catalogue_hits(catalogue: ShoppingCompanionCatalogue) -> None:
    tool = ShoppingCompanionSearchTool(catalogue)
    result = tool.search("wireless mouse", limit=5)
    assert result.matched is True
    assert result.limited_catalogue is True
    assert result.catalogue_size == 3
    assert result.score_kind == "bm25"
    assert result.latency_ms >= 0.0
    assert len(result.hits) >= 2
    for hit in result.hits:
        assert hit.score_kind == "bm25"
        assert hit.namespaced_id.startswith("shopping_companion:")
        assert hit.product_id
        # every hit exists in the catalogue (search never invents a product)
        assert catalogue.get(hit.namespaced_id) is not None
    # deterministic ordering on repeated searches
    again = tool.search("wireless mouse", limit=5)
    assert [hit.product_id for hit in again.hits] == [hit.product_id for hit in result.hits]


def test_search_zero_hits_is_honest(catalogue: ShoppingCompanionCatalogue) -> None:
    tool = ShoppingCompanionSearchTool(catalogue)
    result = tool.search("zzzznotinthecat", limit=5)
    assert result.hits == ()
    assert result.matched is False


def test_search_filters_are_applied_without_reordering(catalogue: ShoppingCompanionCatalogue) -> None:
    tool = ShoppingCompanionSearchTool(catalogue)
    shop = tool.search("mouse", limit=5, shop_id="s1")
    assert [hit.product_id for hit in shop.hits] == ["1001"]
    priced = tool.search("mouse", limit=5, price="0-20")
    assert all((hit.price or 0) <= 20 for hit in priced.hits)
    expensive = tool.search("mouse", limit=5, price_min=10.0, price_max=20.0)
    assert [hit.product_id for hit in expensive.hits] == ["1001"]
    none = tool.search("mouse", limit=5, price="100-200")
    assert none.hits == ()


def test_search_input_validation(catalogue: ShoppingCompanionCatalogue) -> None:
    tool = ShoppingCompanionSearchTool(catalogue)
    with pytest.raises(ShoppingCompanionSearchError):
        tool.search("")
    with pytest.raises(ShoppingCompanionSearchError):
        tool.search("mouse", limit=0)
    with pytest.raises(ShoppingCompanionSearchError):
        tool.search("mouse", limit=10_000)
    with pytest.raises(ShoppingCompanionSearchError):
        tool.search("mouse", price="not-a-range")
    with pytest.raises(ShoppingCompanionSearchError):
        tool.search("mouse", price="20-10")


def test_parse_price_filter_official_semantics() -> None:
    assert parse_price_filter(None) == (None, None)
    assert parse_price_filter("0-100") == (0.0, 100.0)
    assert parse_price_filter("1000-") == (1000.0, None)
    assert parse_price_filter("-50") == (None, 50.0)
