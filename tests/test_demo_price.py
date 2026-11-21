"""Demo historical USD->CNY price adapter, laptop role, retrieval and Gate (M1-F)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue, ElectronicsProduct
from recommendation.catalog.demo_price import (
    DEMO_FX_RATE,
    DEMO_PRICE_MODE,
    DemoPriceIndex,
    read_price_mode,
)
from recommendation.control.rag_evidence import build_product_evidence
from recommendation.catalog.schemas import ProductMetadata
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.preference.product_role import (
    CandidateRole,
    RoleStatus,
    assess_candidate_role,
    infer_product_target,
)
from recommendation.preference.task_requirements import (
    PriceBudget,
    augment_query,
    extract_task_requirements,
    strip_budget_expressions,
)
from recommendation.shopping_companion.planner_bridge import (
    PHASE_PRODUCT_RETRIEVE,
    P14ARulePlanner,
)
from web_demo import viewmodels as vm

REPO = Path(__file__).resolve().parents[1]
SIDECAR = REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"


def _product(item_id: str, title: str, text: str | None = None) -> ElectronicsProduct:
    body = text or f"{title} gaming laptop with RGB keyboard"
    return ElectronicsProduct(
        item_id=item_id,
        namespaced_id=f"{ELECTRONICS_NAMESPACE}:{item_id}",
        title=title,
        text=body,
        product_type=None,
        record=ProductMetadata(
            parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
            title=title,
            main_category="Electronics",
            categories=("Electronics", "Computers", "Laptops"),
            description=(body,),
            source="test-fixture:catalogue",
        ),
    )


def _write_sidecar(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _sidecar_row(item_id: str, price, **extra) -> dict:
    row = {
        "schema": "agentrecx.electronics_metadata_sidecar.v1",
        "item_id": item_id,
        "title": f"Upstream {item_id}",
        "price": price,
        "source": "amazon_reviews_2023:raw_meta_Electronics",
        "dataset_version": "test-version",
    }
    row.update(extra)
    return row


def _requirements():
    return extract_task_requirements("推荐一台 5000 元的笔记本").merged(
        extract_task_requirements("游戏，预算改成 8000 元")
    )


# --------------------------------------------------------------------------- #
# conversion / retention
# --------------------------------------------------------------------------- #


def test_fixed_fx_conversion_is_decimal_with_two_decimals(tmp_path: Path) -> None:
    path = _write_sidecar(tmp_path / "sidecar.jsonl", [
        _sidecar_row("B0A", 899.0),
        _sidecar_row("B0B", 19.99),
        _sidecar_row("B0C", "10.00"),
    ])
    index = DemoPriceIndex.load(path, item_ids={"B0A", "B0B", "B0C"})
    assert index.get("B0A").usd_amount == Decimal("899.0")
    assert index.get("B0A").cny_amount == Decimal("6293.00")
    assert index.get("B0B").cny_amount == Decimal("139.93")
    assert index.get("B0C").cny_amount == Decimal("70.00")
    assert index.stats["priced_records"] == 3
    assert index.stats["fx_rate"] == "7.00"
    assert index.stats["fx_source"] == "demo_fixed_rate"

    fact = index.get("B0A").as_fact()
    assert fact["item_id"] == "B0A"
    assert fact["status"] == "historical_metadata"
    assert fact["currency"] == "CNY"
    assert fact["value"] == 6293.0
    assert fact["original_usd_price"] == 899.0
    assert fact["original_currency"] == "USD"
    assert fact["converted_cny_price"] == "6293.00"
    assert fact["fx_rate"] == "7.00"
    assert fact["source"] == "amazon_reviews_2023:raw_meta_Electronics"
    assert fact["dataset_version"] == "test-version"
    assert fact["price_kind"] == "historical_demo_converted"
    assert fact["source_price_kind"] == "historical_crawl_time"
    assert fact["history_year"] == 2023
    assert fact["historical_time"] is None
    assert "confirmed source dataset year=2023" in fact["historical_time_note"]
    assert fact["price_note"] == "演示参考价：历史美元价格按固定汇率换算，非当前市场售价。"
    assert "not a live price" in fact["reason"]


def test_invalid_or_missing_prices_stay_unknown(tmp_path: Path) -> None:
    path = _write_sidecar(tmp_path / "sidecar.jsonl", [
        _sidecar_row("B0NONE", None),
        _sidecar_row("B0NEG", -1),
        _sidecar_row("B0TEXT", "not-a-price"),
        _sidecar_row("B0ZERO", 0),
        _sidecar_row("B0OK", 12.5),
    ])
    index = DemoPriceIndex.load(
        path, item_ids={"B0NONE", "B0NEG", "B0TEXT", "B0ZERO", "B0OK", "B0MISSING"}
    )
    assert set(index.records) == {"B0OK"}
    assert index.stats["missing_price"] == 1
    assert index.stats["invalid_price"] == 3
    assert index.get("B0MISSING") is None


def test_price_mode_is_strict_by_default_and_demo_only_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTRECX_PRICE_MODE", raising=False)
    assert read_price_mode() == "strict"
    assert read_price_mode("demo_historical_fx") == DEMO_PRICE_MODE
    with pytest.raises(ValueError):
        read_price_mode("live")
    assert DEMO_FX_RATE == Decimal("7.00")


def test_strict_catalogue_does_not_expose_a_demo_price() -> None:
    product = _product("B0STRICT", "Gaming Laptop")
    catalogue = ElectronicsCatalogue({"B0STRICT": product}, {})
    assert catalogue.price_mode == "strict"
    assert catalogue.demo_price_for("B0STRICT") is None
    fact = catalogue.inspect("B0STRICT", requested=["price"])["facts"]["price"]
    assert fact["status"] == "unknown"


def test_demo_catalogue_fact_is_the_converted_cny_price(tmp_path: Path) -> None:
    product = _product("B0DEMO", "MSI Gaming Laptop")
    catalogue = ElectronicsCatalogue({"B0DEMO": product}, {})
    index = DemoPriceIndex.load(
        _write_sidecar(tmp_path / "sidecar.jsonl", [_sidecar_row("B0DEMO", 899.0)]),
        item_ids={"B0DEMO"},
    )
    catalogue.attach_demo_prices(index)
    assert catalogue.price_mode == "demo_historical_fx"
    fact = catalogue.inspect("B0DEMO", requested=["price"])["facts"]["price"]
    assert fact["status"] == "historical_metadata"
    assert fact["value"] == 6293.0
    assert fact["currency"] == "CNY"
    assert fact["original_usd_price"] == 899.0
    assert fact["fx_rate"] == "7.00"


def test_product_card_is_marked_as_historical_fx_not_live(tmp_path: Path) -> None:
    product = _product("B0CARD", "MSI GF65 Gaming Laptop")
    catalogue = ElectronicsCatalogue({"B0CARD": product}, {})
    index = DemoPriceIndex.load(
        _write_sidecar(tmp_path / "sidecar.jsonl", [_sidecar_row("B0CARD", 899.0)]),
        item_ids={"B0CARD"},
    )
    catalogue.attach_demo_prices(index)
    info = catalogue.inspect("B0CARD", requested=["title", "price"])
    card = vm.product_card(
        {
            "product_id": "B0CARD",
            "selected_product": {"item_id": "B0CARD", "title": product.title},
            "product_info": info,
            "product_evidence": [],
            "unknown_fields": [],
        },
        source="recommendation",
        retrieval_source=vm.RETRIEVAL_CATALOGUE,
        match_status="recommended",
        recommended=True,
    )
    assert card["price"]["status"] == "historical_metadata"
    assert card["price"]["display"] == "¥6,293.00 CNY"
    assert card["price"]["reason"] == "演示参考价：历史美元价格按固定汇率换算，非当前市场售价。"
    assert card["price"]["price_kind"] == "historical_demo_converted"
    assert card["price"]["history_year"] == 2023
    assert card["price"]["original_usd_price"] == 899.0
    assert card["price"]["fx_rate"] == "7.00"
    assert card["price"]["currency"] == "CNY"


def test_demo_price_text_becomes_evidence_and_does_not_mark_price_unknown(
    tmp_path: Path,
) -> None:
    product = _product("B0EVID", "MSI GF65 Gaming Laptop")
    catalogue = ElectronicsCatalogue({"B0EVID": product}, {})
    index = DemoPriceIndex.load(
        _write_sidecar(tmp_path / "sidecar.jsonl", [_sidecar_row("B0EVID", 899.0)]),
        item_ids={"B0EVID"},
    )
    catalogue.attach_demo_prices(index)
    _, evidence, unknown = build_product_evidence(
        catalogue=catalogue, item_id="B0EVID", namespace=ELECTRONICS_NAMESPACE, task_id="t"
    )
    assert [item.raw_field for item in evidence].count("price_text") == 1
    assert all(item.attribute != "price" for item in unknown)
    assert any("899.00 USD" in item.span_text and "6293.00 CNY" in item.span_text
               for item in evidence)


# --------------------------------------------------------------------------- #
# laptop role and retrieval revision
# --------------------------------------------------------------------------- #


def test_laptop_target_and_accessories_are_distinguished() -> None:
    target = infer_product_target("gaming laptop")
    assert target.product_type == "laptop" and target.role is CandidateRole.MAIN
    match = assess_candidate_role(
        product_type=None, title="MSI GF65 Thin ... Win10 Gaming Laptop.", target=target
    )
    assert match.status is RoleStatus.MATCH
    bundled = assess_candidate_role(
        product_type=None, title="Gaming Laptop with 16GB RAM and RGB keyboard", target=target
    )
    assert bundled.status is RoleStatus.MATCH
    for title in (
        "Keyboard Cover Compatible with 2022 ASUS TUF Gaming F15 Laptop",
        "Laptop Charger 65W AC Adapter",
        "Laptop Battery for Dell Latitude E5450",
        "MSI GE62 LED LCD Screen 15.6 FHD Gaming Laptop",
        "Laptop Cooling Pad RGB",
        "Laptop Stand for Desk",
    ):
        assessment = assess_candidate_role(product_type=None, title=title, target=target)
        assert assessment.status is RoleStatus.MISMATCH, title
    charger_target = infer_product_target("laptop charger")
    assert charger_target.product_type == "laptop"
    assert charger_target.role is CandidateRole.ACCESSORY


def test_retrieval_text_drops_expired_budget_but_keeps_original_history() -> None:
    requirements = _requirements()
    query = "推荐一台 5000 元的笔记本"
    retrieval = augment_query(query, requirements)
    assert "5000" not in retrieval
    assert "8000" not in retrieval
    assert "笔记本" in retrieval
    assert "laptop" in retrieval and "gaming" in retrieval
    # The original text is not mutated and remains the history carrier.
    assert query == "推荐一台 5000 元的笔记本"
    assert "5000" in requirements.source_text[0]
    assert strip_budget_expressions("gaming laptop under $8000") == "gaming laptop"
    assert strip_budget_expressions("预算改成 8000 元的笔记本") == "笔记本"


# --------------------------------------------------------------------------- #
# real catalogue / real sidecar matching
# --------------------------------------------------------------------------- #


def test_real_catalogue_matches_real_sidecar_prices_and_accepts_8000_cny() -> None:
    catalogue = ElectronicsCatalogue.load(price_mode="demo_historical_fx")
    stats = catalogue.demo_price_stats
    assert catalogue.catalogue_size == 76263
    assert stats["matched_records"] == 76263
    assert stats["priced_records"] == 25736
    assert stats["match_rate"] == 1.0
    assert stats["priced_rate"] == pytest.approx(0.337464, abs=1e-6)
    assert stats["fx_rate"] == "7.00"

    fact = catalogue.inspect("B0BY2Z27LC", requested=["price"])["facts"]["price"]
    assert fact["value"] == 6293.0
    assert fact["original_usd_price"] == 899.0
    assert fact["fx_rate"] == "7.00"
    assert fact["price_note"].startswith("演示参考价：历史美元价格按固定汇率换算")
    assert 6293.0 <= 8000.0

    hits = catalogue.search("laptop gaming", limit=20).hits
    priced = [
        catalogue.demo_price_for(hit.parent_asin.split(":", 1)[-1]) for hit in hits
    ]
    priced = [item for item in priced if item is not None]
    assert len(priced) >= 8
    assert any(item.item_id == "B0BY2Z27LC" and item.cny_amount <= Decimal("8000.00")
               for item in priced)


# --------------------------------------------------------------------------- #
# the real Gate uses the converted CNY price
# --------------------------------------------------------------------------- #


def _gate_catalogue(tmp_path: Path) -> ElectronicsCatalogue:
    products = {
        "B0WITHIN": _product("B0WITHIN", "Gaming Laptop Within"),
        "B0OVER": _product("B0OVER", "Gaming Laptop Over"),
        "B0CHEAP": _product("B0CHEAP", "Gaming Laptop Cheap"),
    }
    catalogue = ElectronicsCatalogue(products, {})
    index = DemoPriceIndex.load(
        _write_sidecar(tmp_path / "gate_sidecar.jsonl", [
            _sidecar_row("B0WITHIN", 1142.71),   # 7998.97 CNY
            _sidecar_row("B0OVER", 1142.86),     # 8000.02 CNY
            _sidecar_row("B0CHEAP", 800.00),     # 5600.00 CNY
        ]),
        item_ids=set(products),
    )
    catalogue.attach_demo_prices(index)
    return catalogue


def _run_executor(executor: ElectronicsPlannerExecutor, query: str) -> None:
    observation = PlannerObservation(
        step=0, query=query,
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": query, "preferences": []},
    )
    run_planner(
        planner=P14ARulePlanner(), executor=executor,
        initial_observation=observation, max_steps=MAX_STEPS,
    )


def test_gate_selects_a_converted_cny_price_within_8000(tmp_path: Path) -> None:
    catalogue = _gate_catalogue(tmp_path)
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue,
        current_query="gaming laptop",
        preferences=[],
        hard=[],
        price_budget=PriceBudget(max_amount=8000.0, currency="CNY"),
    )
    _run_executor(executor, "gaming laptop")
    assert executor.terminal_status == "finalized"
    assert executor.selected_candidate_id in {"B0WITHIN", "B0CHEAP"}
    budget = executor.last_budget or {}
    assert budget["within"] == 2 and budget["over"] == 1
    assert budget["excluded_candidate_ids"] == ["B0OVER"]
    selected_price = budget["selected_price"]
    assert selected_price["currency"] == "CNY"
    assert selected_price["amount"] <= 8000.0
    assert selected_price["fx_rate"] == "7.00"
    assert selected_price["original_usd_price"] in {1142.71, 800.0}
    assert selected_price["converted_cny_price"] in {"7998.97", "5600.00"}


def test_gate_falls_back_from_an_over_budget_top_pick(tmp_path: Path) -> None:
    products = {
        "B0OVER": _product(
            "B0OVER", "Gaming Laptop Gaming Laptop Gaming Laptop Over Budget"
        ),
        "B0WITHIN": _product("B0WITHIN", "Gaming Laptop Within Budget"),
    }
    catalogue = ElectronicsCatalogue(products, {})
    index = DemoPriceIndex.load(
        _write_sidecar(tmp_path / "fallback_sidecar.jsonl", [
            _sidecar_row("B0OVER", 1142.86),   # 8000.02 CNY, over
            _sidecar_row("B0WITHIN", 800.00),  # 5600.00 CNY, within
        ]),
        item_ids=set(products),
    )
    catalogue.attach_demo_prices(index)
    first = catalogue.search("gaming laptop", limit=5).hits[0].parent_asin
    assert first.endswith("B0OVER"), "the fixture must rank the over-budget item first"

    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue,
        current_query="gaming laptop",
        preferences=[],
        hard=[],
        price_budget=PriceBudget(max_amount=8000.0, currency="CNY"),
    )
    _run_executor(executor, "gaming laptop")
    assert executor.terminal_status == "finalized"
    budget = executor.last_budget or {}
    assert budget["gate_selected_candidate_id"] == "B0OVER"
    assert budget["selected_candidate_id"] == "B0WITHIN"
    assert budget["selected_status"] == "within"
    assert budget["selected_price"]["amount"] == 5600.0
    assert "selection_note" in budget


def test_all_over_budget_is_not_a_success(tmp_path: Path) -> None:
    catalogue = _gate_catalogue(tmp_path)
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue,
        current_query="gaming laptop",
        preferences=[],
        hard=[],
        price_budget=PriceBudget(max_amount=5000.0, currency="CNY"),
    )
    _run_executor(executor, "gaming laptop")
    assert executor.terminal_status == "budget_exceeded"
    assert executor.selected_candidate_id is None
    assert (executor.last_budget or {})["over"] >= 1


def test_strict_catalogue_refuses_the_same_budget_without_a_price(tmp_path: Path) -> None:
    catalogue = _gate_catalogue(tmp_path)
    # Build a strict copy from the same products but with no demo attachment.
    strict = ElectronicsCatalogue(
        {item_id: _product(item_id, product.title) for item_id, product in
         catalogue._products.items()},
        {},
    )
    executor = ElectronicsPlannerExecutor(
        catalogue=strict,
        current_query="gaming laptop",
        preferences=[],
        hard=[],
        price_budget=PriceBudget(max_amount=8000.0, currency="CNY"),
    )
    _run_executor(executor, "gaming laptop")
    assert executor.terminal_status == "budget_unverifiable"
    assert executor.selected_candidate_id is None
