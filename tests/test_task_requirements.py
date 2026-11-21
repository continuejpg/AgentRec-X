"""Task requirements (budget/use case/category) and price-budget verification (M1-E).

The real C4 Electronics text catalogue has no structured price, so the **real** MCP path must
refuse a budget it cannot verify.  These tests pin the pure requirement/price logic, including
the 7999/8000/8001 boundary, with explicitly labelled test-fixture prices; the real end-to-end
refusal is covered in ``tests/test_web_requirement_update_e2e.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue, ElectronicsProduct
from recommendation.catalog.enrichment import MetadataSidecar, attach_metadata
from recommendation.catalog.schemas import ProductMetadata
from recommendation.preference.task_requirements import (
    PRICE_CURRENCY_MISMATCH,
    PRICE_NOT_APPLICABLE,
    PRICE_OVER,
    PRICE_UNKNOWN,
    PRICE_WITHIN,
    PriceBudget,
    TaskRequirements,
    augment_query,
    evaluate_price_budget,
    extract_task_requirements,
)

TEST_PRICE_SOURCE = "test-fixture:cn-laptop-price"


def _fixture_price(amount: float, currency: str = "CNY") -> dict:
    """A clearly labelled test-fixture price fact (never presented as a real MCP result)."""
    return {
        "status": "historical_metadata",
        "value": amount,
        "currency": currency,
        "price_kind": "historical_crawl_time",
        "source": TEST_PRICE_SOURCE,
        "verification": "upstream_metadata",
    }


# --------------------------------------------------------------------------- #
# extraction / merge
# --------------------------------------------------------------------------- #


def test_chinese_query_budget_and_category_are_extracted() -> None:
    requirements = extract_task_requirements("推荐一台 5000 元的笔记本")
    assert requirements.budget is not None
    assert requirements.budget.max_amount == 5000.0
    assert requirements.budget.currency == "CNY"
    assert requirements.category == "laptop"
    assert requirements.use_cases == ()
    assert requirements.budget.source_text == "推荐一台 5000 元的笔记本"


def test_chinese_reply_budget_update_and_use_case_are_extracted() -> None:
    requirements = extract_task_requirements("游戏，预算改成 8000 元")
    assert requirements.budget is not None
    assert requirements.budget.max_amount == 8000.0
    assert requirements.budget.currency == "CNY"
    assert requirements.use_cases == ("gaming",)
    assert requirements.category is None


def test_latest_budget_replaces_the_original_and_keeps_the_task() -> None:
    original = extract_task_requirements("推荐一台 5000 元的笔记本")
    reply = extract_task_requirements("游戏，预算改成 8000 元")
    merged = original.merged(reply)
    assert merged.budget is not None
    assert merged.budget.max_amount == 8000.0, "the new budget must replace 5000, not coexist"
    assert merged.budget.currency == "CNY"
    assert merged.category == "laptop", "the original product type must survive the update"
    assert merged.use_cases == ("gaming",)
    # Both user statements are kept as provenance; the effective budget is the latest one.
    assert len(merged.source_text) == 2


def test_a_reply_without_a_budget_keeps_the_original_budget() -> None:
    original = extract_task_requirements("推荐一台 5000 元的笔记本")
    merged = original.merged(extract_task_requirements("游戏"))
    assert merged.budget is not None and merged.budget.max_amount == 5000.0
    assert merged.use_cases == ("gaming",)
    assert merged.category == "laptop"


def test_currency_is_explicit_or_unknown_never_inferred() -> None:
    assert extract_task_requirements("budget under $1200").budget.currency == "USD"
    assert extract_task_requirements("预算 8000 元").budget.currency == "CNY"
    unknown = extract_task_requirements("budget under 8000").budget
    assert unknown is not None and unknown.max_amount == 8000.0
    assert unknown.currency is None
    assert evaluate_price_budget(_fixture_price(7000), unknown).status == PRICE_UNKNOWN


def test_a_bare_number_without_currency_or_budget_phrase_is_not_a_budget() -> None:
    requirements = extract_task_requirements("推荐一台 5000 元的笔记本")
    assert requirements.budget is not None
    # A model number / rating is not a budget.
    assert extract_task_requirements("I need the 8000 model of the keyboard").budget is None


def test_augmented_query_adds_english_terms_but_never_budget_digits() -> None:
    requirements = extract_task_requirements("推荐一台 5000 元的笔记本").merged(
        extract_task_requirements("游戏，预算改成 8000 元")
    )
    query = augment_query("推荐一台 5000 元的笔记本", requirements)
    assert "laptop" in query and "gaming" in query
    assert "8000" not in query, "budget digits pollute lexical retrieval"


def test_requirements_are_json_round_trippable() -> None:
    requirements = extract_task_requirements("推荐一台 5000 元的笔记本").merged(
        extract_task_requirements("游戏，预算改成 8000 元")
    )
    payload = json.loads(json.dumps(requirements.as_dict()))
    restored = TaskRequirements.from_dict(payload)
    assert restored == requirements
    assert TaskRequirements.from_dict(None) == TaskRequirements()
    assert TaskRequirements.from_dict({"schema": "other"}) == TaskRequirements()
    assert restored.as_dict()["budget"] == {
        "max_amount": 8000.0, "min_amount": None, "currency": "CNY",
        "source_text": "游戏，预算改成 8000 元",
    }


# --------------------------------------------------------------------------- #
# price boundary (7999 / 8000 / 8001)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "amount,status",
    [
        (7999.0, PRICE_WITHIN),
        (8000.0, PRICE_WITHIN),  # a budget is inclusive: 8000 <= 8000
        (8001.0, PRICE_OVER),
    ],
)
def test_budget_boundary_is_inclusive_at_the_limit(amount: float, status: str) -> None:
    budget = PriceBudget(max_amount=8000.0, currency="CNY", source_text="预算改成 8000 元")
    evaluation = evaluate_price_budget(_fixture_price(amount), budget)
    assert evaluation.status == status
    assert evaluation.amount == amount
    assert evaluation.currency == "CNY"
    assert evaluation.source == TEST_PRICE_SOURCE
    assert evaluation.price_kind == "historical_crawl_time"


def test_currency_mismatch_is_unverifiable_not_compliance() -> None:
    budget = PriceBudget(max_amount=8000.0, currency="CNY", source_text="预算改成 8000 元")
    evaluation = evaluate_price_budget(_fixture_price(49.99, currency="USD"), budget)
    assert evaluation.status == PRICE_CURRENCY_MISMATCH
    assert not evaluation.verified
    assert "CNY" in evaluation.reason and "USD" in evaluation.reason


def test_missing_price_or_currency_is_unknown() -> None:
    budget = PriceBudget(max_amount=8000.0, currency="CNY")
    assert evaluate_price_budget(None, budget).status == PRICE_UNKNOWN
    assert evaluate_price_budget(
        {"status": "unknown", "value": None}, budget
    ).status == PRICE_UNKNOWN
    assert evaluate_price_budget(
        {"status": "historical_metadata", "value": 5000.0, "currency": None}, budget
    ).status == PRICE_UNKNOWN


def test_no_budget_means_not_applicable() -> None:
    assert evaluate_price_budget(_fixture_price(1.0), None).status == PRICE_NOT_APPLICABLE
    assert evaluate_price_budget(
        _fixture_price(1.0), PriceBudget()
    ).status == PRICE_NOT_APPLICABLE


def test_price_floor_uses_the_same_comparison() -> None:
    budget = PriceBudget(min_amount=8000.0, currency="CNY")
    assert evaluate_price_budget(_fixture_price(8000.0), budget).status == PRICE_WITHIN
    assert evaluate_price_budget(_fixture_price(7999.0), budget).status == PRICE_OVER


# --------------------------------------------------------------------------- #
# the real enriched sidecar: numeric price, explicit unit and source
# --------------------------------------------------------------------------- #


def _sidecar(path: Path) -> MetadataSidecar:
    row = {
        "schema": "agentrecx.electronics_metadata_sidecar.v1",
        "item_id": "B000TEST01",
        "title": "Upstream gaming laptop",
        "features": ["16 GB RAM"],
        "description": ["Upstream description"],
        "categories": ["Electronics", "Computers", "Laptops"],
        "store": "Acme",
        "details": {"Brand": "Acme"},
        "price": 49.99,
        "average_rating": 4.5,
        "rating_number": 120,
        "main_category": "All Electronics",
        "source": "amazon_reviews_2023:raw_meta_Electronics",
        "dataset_version": "test-rev",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return MetadataSidecar.load(path)


def test_real_sidecar_price_has_numeric_value_unit_and_source(tmp_path: Path) -> None:
    item_id = "B000TEST01"
    product = ElectronicsProduct(
        item_id=item_id,
        namespaced_id=f"{ELECTRONICS_NAMESPACE}:{item_id}",
        title="C4 short title",
        text="C4 text",
        product_type="laptop",
        record=ProductMetadata(
            parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
            title="C4 short title",
            main_category="Electronics",
            categories=("Electronics",),
            description=("C4 text",),
            source="data/p13r/electronics/catalog_meta.jsonl",
        ),
    )
    catalogue = ElectronicsCatalogue({item_id: product}, {})
    attach_metadata(catalogue, _sidecar(tmp_path / "sidecar.jsonl"))
    view = catalogue.inspect(item_id, requested=["price"])
    fact = view["facts"]["price"]
    assert fact["status"] == "historical_metadata"
    assert fact["value"] == 49.99
    assert fact["currency"] == "USD"
    assert fact["source"] == "amazon_reviews_2023:raw_meta_Electronics"
    assert fact["price_kind"] == "historical_crawl_time"
    assert "price" in view["unknown_fields"], "a historical price is not a live price"

    # The user's 8000 CNY budget cannot be verified against this USD historical price.
    budget = PriceBudget(max_amount=8000.0, currency="CNY")
    evaluation = evaluate_price_budget(fact, budget)
    assert evaluation.status == PRICE_CURRENCY_MISMATCH
    assert evaluation.source == "amazon_reviews_2023:raw_meta_Electronics"
