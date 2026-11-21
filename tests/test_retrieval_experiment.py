"""Fast tests for the M3-C retrieval query set, scope and label rules."""

from __future__ import annotations

from types import SimpleNamespace

from agent_eval.retrieval.labeling import LABEL_POLICY, label_title
from agent_eval.retrieval.queries import load_queries
from agent_eval.retrieval.scope import (
    CATEGORY_TERMS,
    CategoryScope,
    classify_product,
    is_accessory_title,
)


def _product(item_id: str, title: str):  # noqa: ANN202
    return SimpleNamespace(item_id=item_id, title=title, namespaced_id=f"x:{item_id}")


def test_query_set_has_ten_unique_queries_and_required_categories() -> None:
    query_set = load_queries()
    assert query_set.dataset_version == "1.0.0"
    assert len(query_set.queries) == 10
    assert len({query.query_id for query in query_set.queries}) == 10
    assert {"laptop", "headphones", "tablet"} <= {
        query.expected_category for query in query_set.queries
    }
    assert {query.query_id for query in query_set.queries if query.expected_features} == {
        "mouse_wireless_300", "keyboard_mechanical_500",
    }


def test_label_policy_is_explicitly_machine_assisted() -> None:
    assert "machine-assisted" in LABEL_POLICY
    assert "pending human review" in LABEL_POLICY


def test_accessory_and_head_type_rules() -> None:
    assert is_accessory_title("laptop", "Laptop Stand, Adjustable Laptop Stand")
    assert is_accessory_title("laptop", "The Office Stickers for Laptop")
    assert not is_accessory_title("laptop", "ASUS Laptop E551LA-XB51")
    assert is_accessory_title("tablet", "Tablet Case for Kids")
    assert not is_accessory_title("tablet", "Kids Tablet 7 Inch Android")


def test_classify_product_uses_head_type_not_compatibility_mentions() -> None:
    sidecar = SimpleNamespace(get=lambda item_id: None)
    headphones = _product(
        "B1",
        "Bluetooth Headphones with Microphone for PC Tablet Study",
    )
    laptop = _product("B2", "MSI GF65 Thin 15.6 Gaming Laptop")
    accessory = _product("B3", "Laptop Stand for Desk")
    assert not classify_product(headphones, "tablet", sidecar)
    assert classify_product(laptop, "laptop", sidecar)
    assert not classify_product(accessory, "laptop", sidecar)


def test_label_title_requires_category_and_explicit_intent() -> None:
    assert label_title("laptop", "MSI GF65 Thin Gaming Laptop", use_cases=["gaming"]) == (
        1,
        "title matches target category and query intents",
    )
    grade, reason = label_title(
        "laptop", "Dell Latitude Business Laptop", use_cases=["gaming"]
    )
    assert grade == 0 and "gaming" in reason
    grade, reason = label_title("tablet", "Bluetooth Headphones for Tablet")
    assert grade == 0 and "head product type" in reason
    assert label_title(
        "keyboard", "Hexgears GK13 Mechanical Keyboard", features=["mechanical"]
    )[0] == 1
    assert label_title(
        "keyboard", "Cheap Wired Keyboard", features=["mechanical"]
    )[0] == 0


def test_category_scope_sizes_are_computed_without_bm25_scores() -> None:
    # A tiny fake catalogue demonstrates that the scope is a document filter, not a score.
    sidecar = SimpleNamespace(get=lambda item_id: None)
    products = {
        "a": _product("a", "Gaming Laptop"),
        "b": _product("b", "Laptop Stand"),
        "c": _product("c", "Kids Tablet"),
    }
    catalogue = SimpleNamespace(
        _products=products,
        catalogue_size=3,
    )
    scope = CategoryScope(catalogue=catalogue, sidecar=sidecar)
    assert {product.item_id for product in scope.members("laptop")} == {"a"}
    assert {product.item_id for product in scope.members("tablet")} == {"c"}
    assert CATEGORY_TERMS == {"laptop", "headphones", "tablet", "mouse", "keyboard"}
