"""Phase 6A: structured product research (inspect_product / compare_products).

The fixtures are intentionally small and explicit.  They include a fully populated record,
a record with several absences, and a record whose price cannot be parsed, because the
properties under test are precisely the distinctions between value / missing / unreadable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.catalog import MetadataIndex, ProductMetadata  # noqa: E402
from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    CompareProductsArguments,
    GroundedReasoner,
    InspectProductArguments,
    LoopLimits,
    PolicyActionError,
    ProductComparisonObservation,
    ProductFact,
    ProductInspectionObservation,
    ProductResearchError,
    ProductResearchTool,
    ReasoningExecutor,
)
from recommendation.control.schemas import ValidatedAction  # noqa: E402
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import ScriptedPolicy, build_control_harness  # noqa: E402


def _record(
    parent_asin: str,
    *,
    title: str | None,
    store: str | None = None,
    price: str | None = None,
    details: tuple[tuple[str, str], ...] = (),
    description: tuple[str, ...] = (),
    average_rating: float | None = None,
) -> ProductMetadata:
    return ProductMetadata(
        parent_asin=parent_asin,
        title=title,
        store=store,
        main_category="Sports & Outdoors" if title else None,
        categories=("Sports & Outdoors", "Exercise") if title else (),
        features=("feature one", "feature two") if title else (),
        description=description,
        price_text=price,
        average_rating=average_rating,
        details=details,
        source="fixture:product-research",
    )


@pytest.fixture(scope="module")
def index() -> MetadataIndex:
    return MetadataIndex.from_records(
        [
            _record(
                "prod-full",
                title="Trail Runner 2",
                store="Acme Outdoors",
                price="25.0",
                details=(
                    ("Brand Name", "Acme"),
                    ("Color", "Red"),
                    ("Material", "Leather"),
                    ("Item Weight", "0.8 Kilograms"),
                ),
                description=("A trail shoe.", "Built for wet ground."),
                average_rating=4.6,
            ),
            _record(
                "prod-sparse",
                title="Road Runner",
                store="Acme Outdoors",
                price="40.0",
                details=(("Color", "Blue"), ("Item Weight", "1.1 Kilograms")),
                average_rating=4.1,
            ),
            _record(
                "prod-unreadable",
                title="Budget Runner",
                store=None,
                price="not-a-price",
                details=(("Material", "Canvas"), ("Item Weight", "heavy")),
            ),
            _record(
                "prod-no-price",
                title="Minimal Runner",
                store="CampFix",
                price=None,
                details=(("Color", "Green"),),
            ),
        ]
    )


@pytest.fixture()
def tool(index: MetadataIndex) -> ProductResearchTool:
    return ProductResearchTool(index.records)


def _validated(kind: ActionKind, arguments: Any) -> ValidatedAction:
    return ValidatedAction(
        action=kind,
        action_id=f"act:{kind.value}",
        step_index=0,
        run_id="run",
        k=1,
        arguments=arguments,
    )


# --------------------------------------------------------------------------- #
# Tool-level facts
# --------------------------------------------------------------------------- #


def test_known_product_has_raw_values_derived_values_and_sources(tool: ProductResearchTool) -> None:
    result = tool.inspect_product(
        "prod-full",
        attributes=("title", "brand", "store", "price", "color", "description"),
    )
    assert result.status == "ok"
    assert result.parent_asin == "prod-full"
    assert result.catalog_source == "fixture:product-research"

    title = result.facts["title"]
    assert title.status == "value"
    assert title.raw == "Trail Runner 2"
    assert title.value == "Trail Runner 2"
    assert title.derived is False
    assert title.source == "catalog:title"

    price = result.facts["price"]
    assert price.status == "value"
    assert price.raw == "25.0"
    assert price.value == 25.0
    assert price.derived is True
    assert price.verification == "derived"
    assert price.source == "catalog:price_text"

    brand = result.facts["brand"]
    assert brand.raw == "Acme"
    assert brand.source == "catalog:details:Brand Name"

    description = result.facts["description"]
    assert description.status == "value"
    assert description.raw == ["A trail shoe.", "Built for wet ground."]
    assert description.value == ["A trail shoe.", "Built for wet ground."]


def test_unknown_product_is_an_error_not_an_empty_inspection(tool: ProductResearchTool) -> None:
    with pytest.raises(ProductResearchError) as raised:
        tool.inspect_product("does-not-exist")
    assert raised.value.code == "unknown_product"

    with pytest.raises(ProductResearchError) as compare:
        tool.compare_products(("prod-full", "does-not-exist"))
    assert compare.value.code == "unknown_product"


def test_missing_fields_are_missing_and_never_defaulted(tool: ProductResearchTool) -> None:
    result = tool.inspect_product(
        "prod-sparse", attributes=("brand", "material", "description", "price")
    )
    for name in ("brand", "material", "description"):
        fact = result.facts[name]
        assert fact.status == "missing"
        assert fact.raw is None
        assert fact.value is None
        assert fact.verification == "absent"
    assert result.facts["price"].value == 40.0
    assert set(result.missing_attributes) == {"brand", "material", "description"}
    # A product whose requested facts are all absent is a normal empty result, not a failure.
    empty = tool.inspect_product("prod-sparse", attributes=("brand", "material"))
    assert empty.status == "empty"


def test_unreadable_values_are_distinguished_from_missing(tool: ProductResearchTool) -> None:
    result = tool.inspect_product("prod-unreadable", attributes=("price", "weight", "material"))
    assert result.facts["price"].status == "unreadable"
    assert result.facts["price"].raw == "not-a-price"
    assert result.facts["price"].value is None
    assert result.facts["weight"].status == "unreadable"
    assert result.facts["material"].value == "Canvas"
    assert result.unreadable_attributes == ("price", "weight")


def test_unknown_requested_attribute_is_reported_not_guessed(tool: ProductResearchTool) -> None:
    result = tool.inspect_product("prod-full", attributes=("title", "waterproof", "colour"))
    assert result.facts["waterproof"].status == "unsupported"
    assert result.facts["colour"].attribute == "color"
    assert result.unsupported_attributes == ("waterproof",)
    assert result.facts["colour"].value == "Red"


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_comparison_preserves_input_order_and_attributes(tool: ProductResearchTool) -> None:
    result = tool.compare_products(
        ("prod-sparse", "prod-full"),
        attributes=("title", "price", "color", "material"),
    )
    assert result.products == ("prod-sparse", "prod-full")
    assert [row.attribute for row in result.rows] == ["title", "price", "color", "material"]
    assert "material" in {row.attribute for row in result.rows if row.status == "partial"}
    material = next(row for row in result.rows if row.attribute == "material")
    assert material.facts["prod-sparse"].status == "missing"
    assert material.facts["prod-full"].value == "Leather"
    assert "prod-sparse" in (material.reason or "")
    assert "material" in result.missing_attributes


def test_numeric_comparison_is_min_max_not_a_better_verdict(tool: ProductResearchTool) -> None:
    result = tool.compare_products(("prod-full", "prod-sparse"), attributes=("price",))
    row = result.rows[0]
    assert row.status == "comparable"
    assert row.comparison is not None
    assert row.comparison["kind"] == "numeric"
    assert row.comparison["min"] == 25.0
    assert row.comparison["max"] == 40.0
    assert row.comparison["min_products"] == ["prod-full"]
    assert row.comparison["max_products"] == ["prod-sparse"]
    assert "better" not in row.comparison
    assert "best" not in row.comparison


def test_missing_or_unreadable_blocks_numeric_comparison(tool: ProductResearchTool) -> None:
    result = tool.compare_products(
        ("prod-full", "prod-unreadable"), attributes=("price", "material")
    )
    price = next(row for row in result.rows if row.attribute == "price")
    assert price.status == "partial"
    assert price.comparison is None
    assert price.facts["prod-unreadable"].status == "unreadable"
    assert price.facts["prod-full"].value == 25.0


def test_comparison_does_not_replace_missing_with_zero(tool: ProductResearchTool) -> None:
    result = tool.compare_products(("prod-no-price", "prod-full"), attributes=("price",))
    row = result.rows[0]
    assert row.status == "partial"
    assert row.facts["prod-no-price"].status == "missing"
    assert row.facts["prod-no-price"].value is None
    assert row.comparison is None


def test_comparison_limits_and_duplicates_are_rejected(tool: ProductResearchTool) -> None:
    with pytest.raises(ProductResearchError) as one:
        tool.compare_products(("prod-full",))
    assert one.value.code == "invalid_product_count"

    with pytest.raises(ProductResearchError) as many:
        tool.compare_products(("a", "b", "c", "d", "e", "f"))
    assert many.value.code == "invalid_product_count"

    with pytest.raises(ProductResearchError) as duplicate:
        tool.compare_products(("prod-full", "prod-full"))
    assert duplicate.value.code == "duplicate_product"


def test_argument_schemas_enforce_product_and_attribute_bounds() -> None:
    with pytest.raises(ValidationError):
        InspectProductArguments(parent_asin="x", attributes=tuple(f"a{i}" for i in range(13)))
    with pytest.raises(ValidationError):
        CompareProductsArguments(parent_asins=("a", "b", "c", "d", "e", "f"))
    with pytest.raises(ValidationError):
        CompareProductsArguments(parent_asins=("a",))
    with pytest.raises(ValidationError):
        InspectProductArguments(parent_asin="x", unexpected=True)


# --------------------------------------------------------------------------- #
# Executor / loop integration
# --------------------------------------------------------------------------- #


def test_executor_records_tool_name_identity_and_status(index: MetadataIndex) -> None:
    executor = ReasoningExecutor(GroundedReasoner(index))
    action = _validated(
        ActionKind.INSPECT_PRODUCT,
        InspectProductArguments(parent_asin="prod-full", attributes=("title", "price")),
    )
    observation = executor.execute(action, authorized_candidates=("prod-full",))
    assert isinstance(observation, ProductInspectionObservation)
    assert observation.tool == "inspect_product"
    assert observation.parent_asin == "prod-full"
    assert observation.status == "ok"
    assert observation.facts["price"]["derived"] is True


def test_executor_refuses_unknown_identity_with_stable_code(index: MetadataIndex) -> None:
    executor = ReasoningExecutor(GroundedReasoner(index))
    with pytest.raises(PolicyActionError) as missing:
        executor.execute(
            _validated(ActionKind.INSPECT_PRODUCT, InspectProductArguments(parent_asin="nope")),
            authorized_candidates=None,
        )
    assert missing.value.code == "unknown_product"

    with pytest.raises(PolicyActionError) as unauthorized:
        executor.execute(
            _validated(
                ActionKind.INSPECT_PRODUCT,
                InspectProductArguments(parent_asin="prod-full"),
            ),
            authorized_candidates=("prod-sparse",),
        )
    assert unauthorized.value.code == "candidate_not_in_run"


def test_executor_comparison_preserves_order_and_returns_structured_rows(
    index: MetadataIndex,
) -> None:
    executor = ReasoningExecutor(GroundedReasoner(index))
    action = _validated(
        ActionKind.COMPARE_PRODUCTS,
        CompareProductsArguments(
            parent_asins=("prod-sparse", "prod-full"),
            attributes=("title", "price", "material"),
        ),
    )
    observation = executor.execute(
        action, authorized_candidates=("prod-full", "prod-sparse")
    )
    assert isinstance(observation, ProductComparisonObservation)
    assert observation.tool == "compare_products"
    assert observation.products == ("prod-sparse", "prod-full")
    assert [row["attribute"] for row in observation.rows] == ["title", "price", "material"]
    assert observation.rows[2]["status"] == "partial"


class _RecommendInspectCompareFinish:
    """A bounded deterministic script for the Phase 6A loop integration."""

    name = "p6a-product-research"

    def __init__(self) -> None:
        self._step = 0

    def choose(self, context: Any) -> ActionProposal:
        from recommendation.control import PolicyContext  # noqa: F401 - documentation

        steps = (
            ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3),
            ActionProposal(
                action=ActionKind.INSPECT_PRODUCT,
                arguments=InspectProductArguments(
                    parent_asin="cand-red", attributes=("title", "price", "color")
                ),
            ),
            ActionProposal(
                action=ActionKind.COMPARE_PRODUCTS,
                arguments=CompareProductsArguments(
                    parent_asins=("cand-red", "cand-blue"),
                    attributes=("title", "price"),
                ),
            ),
            ActionProposal(action=ActionKind.FINISH),
        )
        index = min(self._step, len(steps) - 1)
        self._step += 1
        return steps[index]


def test_loop_calls_both_tools_and_leaves_the_candidate_set_unchanged() -> None:
    harness = build_control_harness(
        policy=_RecommendInspectCompareFinish(),
        limits=LoopLimits(max_steps=6, max_tool_calls=4, max_retries=1),
    )
    harness.controller._reasoning = ReasoningExecutor(  # noqa: SLF001 - deliberate wiring
        GroundedReasoner(harness.parts["enricher"].metadata)
    )
    result = harness.controller.run("recommend and inspect gear", HISTORY, run_id="p6a")
    assert result.trajectory.actions() == (
        "recommend_from_history",
        "inspect_product",
        "compare_products",
        "finish",
    )
    observations = {
        step.observation["kind"]: step.observation
        for step in result.trajectory.steps
        if step.observation
    }
    assert observations["product_inspection"]["tool"] == "inspect_product"
    assert observations["product_inspection"]["parent_asin"] == "cand-red"
    assert observations["product_comparison"]["tool"] == "compare_products"
    assert observations["product_comparison"]["products"] == ("cand-red", "cand-blue")

    before = [item.parent_asin for item in result.state["tool_result"].recommendations]
    assert before == ["cand-red", "cand-blue", "cand-black"]
    assert result.control.tool_call_count == 3  # recommendation + two product-tool calls


def test_loop_turns_unknown_product_into_a_failure_observation_not_empty() -> None:
    class _BadInspect:
        name = "p6a-bad-inspect"

        def __init__(self) -> None:
            self.step = 0

        def choose(self, context: Any) -> ActionProposal:
            steps = (
                ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=3),
                ActionProposal(
                    action=ActionKind.INSPECT_PRODUCT,
                    arguments=InspectProductArguments(parent_asin="cand-green"),
                ),
                ActionProposal(action=ActionKind.FINISH),
            )
            index = min(self.step, len(steps) - 1)
            self.step += 1
            return steps[index]

    harness = build_control_harness(
        policy=_BadInspect(),
        limits=LoopLimits(max_steps=5, max_tool_calls=3, max_retries=1),
    )
    harness.controller._reasoning = ReasoningExecutor(  # noqa: SLF001
        GroundedReasoner(harness.parts["enricher"].metadata)
    )
    result = harness.controller.run("recommend gear", HISTORY, run_id="p6a-bad")
    failures = [
        step.observation
        for step in result.trajectory.steps
        if step.observation and step.observation.get("status") == "failed"
    ]
    assert failures, "the refusal must reach the trajectory"
    assert failures[0]["code"] == "candidate_not_in_run"
    assert failures[0]["tool"] == "inspect_product"
    assert failures[0]["products"] == ("cand-green",)
    assert failures[0]["status"] == "failed"
    assert all(
        step.observation.get("kind") != "product_inspection"
        for step in result.trajectory.steps
        if step.observation
    )


def test_product_fact_model_is_strict_about_status() -> None:
    with pytest.raises(ValidationError):
        ProductFact(
            attribute="title",
            status="guessed",
            verification="catalog_value",
        )


def test_demo_agent_service_wires_product_research(tmp_path: Path) -> None:
    """The served Agent composes the executor over the same metadata it already uses."""
    from tests.demo_fixture import build_harness

    harness = build_harness(tmp_path)
    try:
        service = harness.runtime.agent_service()
        executor = service._reasoning_executor  # noqa: SLF001 - deliberate composition check
        assert executor is not None
        offered = set(executor.available_actions())
        assert ActionKind.INSPECT_PRODUCT in offered
        assert ActionKind.COMPARE_PRODUCTS in offered
    finally:
        harness.runtime.close()
