"""Budget enforcement through the real electronics executor (sourced test-fixture prices).

The production C4 catalogue has no price, so this uses an explicitly labelled test fixture
(``test-fixture:usd-laptop-price``) to exercise the real executor/Gate integration and the
7999/8000/8001 boundary.  These are **not** MCP end-to-end recommendations; the real MCP path
is covered by ``tests/test_web_requirement_update_e2e.py`` (where the budget is refused
because the catalogue price cannot be verified).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue, ElectronicsProduct
from recommendation.catalog.enrichment import MetadataSidecar, attach_metadata
from recommendation.catalog.schemas import ProductMetadata
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.preference.task_requirements import PriceBudget
from recommendation.shopping_companion.planner_bridge import (
    PHASE_PRODUCT_RETRIEVE,
    P14ARulePlanner,
)

TEST_PRICE_SOURCE = "test-fixture:usd-laptop-price"
ITEMS = {
    "B0PRICE7999": 7999.0,
    "B0PRICE8000": 8000.0,
    "B0PRICE8001": 8001.0,
}


def _catalogue(tmp_path: Path) -> ElectronicsCatalogue:
    products: dict[str, ElectronicsProduct] = {}
    rows: list[dict] = []
    for item_id, price in ITEMS.items():
        title = f"Ultra Gaming Laptop {item_id[-4:]}"
        text = f"{title} gaming laptop with RGB keyboard and 16 GB RAM."
        record = ProductMetadata(
            parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
            title=title,
            main_category="Electronics",
            categories=("Electronics", "Computers", "Laptops"),
            description=(text,),
            source="test-fixture:catalogue",
        )
        products[item_id] = ElectronicsProduct(
            item_id=item_id,
            namespaced_id=f"{ELECTRONICS_NAMESPACE}:{item_id}",
            title=title,
            text=text,
            product_type=None,
            record=record,
        )
        rows.append({
            "schema": "agentrecx.electronics_metadata_sidecar.v1",
            "item_id": item_id,
            "title": title,
            "features": ["RGB keyboard"],
            "description": [text],
            "categories": ["Electronics", "Computers", "Laptops"],
            "store": "Acme",
            "details": {"Brand": "Acme"},
            "price": price,
            "average_rating": 4.5,
            "rating_number": 100,
            "main_category": "All Electronics",
            "source": TEST_PRICE_SOURCE,
            "dataset_version": "test-fixture",
        })
    sidecar_path = tmp_path / "fixture_sidecar.jsonl"
    sidecar_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    catalogue = ElectronicsCatalogue(products, {})
    attach_metadata(catalogue, MetadataSidecar.load(sidecar_path))
    return catalogue


def _executor(catalogue: ElectronicsCatalogue, budget: PriceBudget | None) -> ElectronicsPlannerExecutor:
    return ElectronicsPlannerExecutor(
        catalogue=catalogue,
        current_query="gaming laptop",
        preferences=[],
        hard=[],
        price_budget=budget,
    )


def _run(executor: ElectronicsPlannerExecutor) -> None:
    observation = PlannerObservation(
        step=0,
        query="gaming laptop",
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": "gaming laptop",
                  "preferences": []},
    )
    run_planner(
        planner=P14ARulePlanner(), executor=executor,
        initial_observation=observation, max_steps=MAX_STEPS,
    )


def test_over_budget_candidates_are_excluded_before_the_gate(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path)
    executor = _executor(
        catalogue,
        PriceBudget(max_amount=8000.0, currency="USD", source_text="budget under $8000"),
    )
    _run(executor)
    budget = executor.last_budget or {}
    assert budget["candidates_checked"] == 3
    assert budget["within"] == 2, "7999 and 8000 are within the inclusive budget"
    assert budget["over"] == 1
    assert budget["excluded_candidate_ids"] == ["B0PRICE8001"]
    assert executor.terminal_status == "finalized"
    assert executor.selected_candidate_id in {"B0PRICE7999", "B0PRICE8000"}, (
        "a within-budget candidate must be selected, never the 8001 one"
    )
    # The selected candidate's price fact is the sourced test fixture, in USD.
    evidence = (executor._rows_by_rank[executor.selected_candidate_id] or {}).get(
        "budget_evidence"
    )
    assert evidence["status"] == "within"
    assert evidence["currency"] == "USD"
    assert evidence["source"] == TEST_PRICE_SOURCE


def test_all_over_budget_is_an_explicit_refusal(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path)
    executor = _executor(
        catalogue,
        PriceBudget(max_amount=7000.0, currency="USD", source_text="budget under $7000"),
    )
    _run(executor)
    assert executor.terminal_status == "budget_exceeded"
    assert executor.selected_candidate_id is None
    assert (executor.last_budget or {})["over"] == 3


def test_currency_mismatch_is_never_finalized(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path)
    executor = _executor(
        catalogue,
        PriceBudget(max_amount=8000.0, currency="CNY", source_text="预算 8000 元"),
    )
    _run(executor)
    assert executor.terminal_status == "budget_unverifiable"
    assert executor.selected_candidate_id is None
    assert (executor.last_budget or {})["currency_mismatch"] == 3


def test_without_a_budget_the_executor_behaves_as_before(tmp_path: Path) -> None:
    catalogue = _catalogue(tmp_path)
    executor = _executor(catalogue, None)
    _run(executor)
    assert executor.last_budget is None
    assert executor.terminal_status == "finalized"
    assert executor.selected_candidate_id is not None
