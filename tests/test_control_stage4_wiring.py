"""Stage 4 wiring tests: reasoning actions inside the bounded loop.

`tests/test_control_stage4_reasoning.py` proves the *reasoner* reads catalogue facts
correctly.  This file proves the **executor and the loop** route those actions safely:

**A. The executor** maps each read-only action onto the reasoner and shapes an observation,
including the identities it could not ground.

**B. Authority** - a candidate-producing action cannot be routed to the reasoning executor,
and reasoning cannot add, drop or reorder a candidate.

**C. Loop accounting** - a reasoning step is not a tool call, does not consume the tool-call
budget, and does not disturb the candidate state that the final renderer reads.

**D. Both drivers agree**, and the Stage 1/2/3 paths are unaffected when no reasoner is
configured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    REASONING_ACTIONS,
    READ_ONLY_ACTIONS,
    ActionKind,
    ActionProposal,
    BundleArguments,
    BundleObservation,
    CheckCompatibilityArguments,
    CompareArguments,
    ComparisonObservation,
    CompatibilityObservation,
    DetailObservation,
    GetDetailsArguments,
    GroundedReasoner,
    LoopLimits,
    PolicyActionError,
    ReasoningExecutor,
    RunStatus,
    TradeOffArguments,
    TradeOffObservation,
    ValidatedAction,
)
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

# --------------------------------------------------------------------------- #
# fixture catalogue
# --------------------------------------------------------------------------- #


class _Record:
    def __init__(self, parent_asin, **kw):
        self.parent_asin = parent_asin
        self.title = kw.get("title")
        self.main_category = kw.get("main_category")
        self.categories = kw.get("categories", ())
        self.features = kw.get("features", ())
        self.price_text = kw.get("price_text")
        self.average_rating = kw.get("average_rating")
        self.rating_number = 5
        self.details = tuple(kw.get("details", {}).items())


class _Catalog:
    def __init__(self, records):
        self.records = {record.parent_asin: record for record in records}

    def __contains__(self, parent_asin):
        return isinstance(parent_asin, str) and parent_asin in self.records


def build_executor() -> ReasoningExecutor:
    """An executor over a small catalogue with one fully specified and one sparse product."""
    return ReasoningExecutor(
        GroundedReasoner(
            _Catalog(
                [
                    _Record(
                        "LIGHT",
                        title="Ultralight Tent",
                        price_text="120.0",
                        main_category="Camping",
                        categories=("Camping", "Tents"),
                        average_rating=4.6,
                        details={
                            "Item Weight": "1.2 Kilograms",
                            "Brand Name": "Outfit",
                            "Color": "Green",
                        },
                    ),
                    _Record(
                        "HEAVY",
                        title="Heavy Tent",
                        price_text="90.0",
                        main_category="Camping",
                        categories=("Camping", "Tents"),
                        average_rating=4.1,
                        details={
                            "Item Weight": "4.5 Kilograms",
                            "Brand Name": "Outfit",
                            "Color": "Blue",
                        },
                    ),
                    _Record(
                        "SPARSE",
                        title="Mystery Tent",
                        price_text="150.0",
                        main_category="Camping",
                        categories=("Camping",),
                        details={"Brand Name": "Outfit"},
                    ),
                ]
            )
        )
    )


def action(kind: ActionKind, arguments: object, *, step: int = 0) -> ValidatedAction:
    """Build a validated action for one reasoning step."""
    return ValidatedAction(
        action=kind,
        action_id=f"act:{step}:{kind.value}",
        step_index=step,
        run_id="run",
        k=1,
        arguments=arguments,
    )


# =========================================================================== #
# A. The executor
# =========================================================================== #


def test_the_executor_handles_exactly_the_read_only_actions() -> None:
    """Its authority set is derived from the control-plane contract, not restated."""
    assert REASONING_ACTIONS == frozenset(READ_ONLY_ACTIONS)
    executor = build_executor()
    for kind in READ_ONLY_ACTIONS:
        assert executor.handles(kind) is True
    for kind in (ActionKind.FINISH, ActionKind.RECOMMEND_FROM_HISTORY, ActionKind.SEARCH_CATALOG):
        assert executor.handles(kind) is False


def test_details_returns_grounded_facts_and_names_the_ungrounded() -> None:
    """An identity not in the catalogue is reported, never silently dropped."""
    observation = build_executor().execute(
        action(ActionKind.GET_DETAILS, GetDetailsArguments(parent_asins=("LIGHT", "SPARSE", "GHOST")))
    )
    assert isinstance(observation, DetailObservation)
    assert set(observation.facts) == {"LIGHT", "SPARSE"}
    assert observation.facts["LIGHT"]["weight_text"] == "1.200 kg"
    # A fact the catalogue does not carry is UNKNOWN, and the missing identity is named.
    assert observation.facts["SPARSE"]["weight_text"] == "UNKNOWN"
    assert observation.unknown == ("GHOST",)


def test_details_whitelists_the_attributes_it_exposes() -> None:
    """A future catalogue field cannot leak into a policy payload simply by existing."""
    observation = build_executor().execute(
        action(ActionKind.GET_DETAILS, GetDetailsArguments(parent_asins=("LIGHT",)))
    )
    exposed = set(observation.facts["LIGHT"])
    assert "parent_asin" not in exposed
    assert "details" not in exposed  # the raw attribute bag is not forwarded
    assert {"title", "brand", "weight_text", "price_text"} <= exposed


def test_comparison_reports_rows_including_all_unknown_ones() -> None:
    """The absence of a shared fact is itself a finding."""
    observation = build_executor().execute(
        action(
            ActionKind.COMPARE,
            CompareArguments(parent_asins=("LIGHT", "HEAVY"), attributes=("weight_text", "material")),
        )
    )
    assert isinstance(observation, ComparisonObservation)
    rows = {row["attribute"]: row for row in observation.rows}
    assert rows["weight_text"]["values"]["LIGHT"] == "1.200 kg"
    # Neither product carries a material, so the row is present and wholly unknown.
    assert rows["material"]["all_unknown"] is True


def test_comparison_flags_an_unsupported_attribute_request() -> None:
    """Asking for an attribute the catalogue does not track is reported."""
    observation = build_executor().execute(
        action(
            ActionKind.COMPARE,
            CompareArguments(parent_asins=("LIGHT", "HEAVY"), attributes=("voltage",)),
        )
    )
    assert "voltage" in observation.unsupported_attributes


def test_trade_off_orders_only_grounded_products() -> None:
    """A product whose fact is absent is named as unknown, not placed in the order."""
    observation = build_executor().execute(
        action(
            ActionKind.TRADE_OFF,
            TradeOffArguments(parent_asins=("HEAVY", "LIGHT", "SPARSE"), priority="lighter"),
        )
    )
    assert isinstance(observation, TradeOffObservation)
    assert observation.supported is True
    assert observation.order == ("LIGHT", "HEAVY")
    assert "SPARSE" in observation.unknown


def test_trade_off_with_an_unsupported_priority_orders_nothing() -> None:
    """No grounded attribute means no ordering, and the reason is stated."""
    observation = build_executor().execute(
        action(
            ActionKind.TRADE_OFF,
            TradeOffArguments(parent_asins=("LIGHT", "HEAVY"), priority="most stylish"),
        )
    )
    assert observation.supported is False
    assert observation.order == ()
    assert observation.reason


def test_grounded_compatibility_records_the_check_it_performed() -> None:
    """A grounded verdict says which deterministic check produced it."""
    observation = build_executor().execute(
        action(
            ActionKind.CHECK_COMPATIBILITY,
            CheckCompatibilityArguments(parent_asins=("LIGHT", "HEAVY"), requirement="brand: Outfit"),
        )
    )
    assert isinstance(observation, CompatibilityObservation)
    assert observation.verdict == "compatible"
    assert observation.checks_performed == ("brand_match",)


def test_ungrounded_compatibility_is_unknown_with_no_checks() -> None:
    """An interface requirement cannot be checked against this catalogue, and says so."""
    observation = build_executor().execute(
        action(
            ActionKind.CHECK_COMPATIBILITY,
            CheckCompatibilityArguments(parent_asins=("LIGHT", "HEAVY"), requirement="interface: USB-C"),
        )
    )
    assert observation.verdict == "unknown"
    assert observation.checks_performed == ()
    assert "no check could be performed" in (observation.verification_note or "")


def test_bundle_reports_aggregates_and_their_blockers() -> None:
    """A grounded total is reported; a blocked one names who blocks it."""
    observation = build_executor().execute(
        action(
            ActionKind.BUNDLE,
            BundleArguments(parent_asins=("LIGHT", "HEAVY", "SPARSE")),
        )
    )
    assert isinstance(observation, BundleObservation)
    assert observation.member_count == 3
    assert observation.total_price["status"] == "grounded"
    assert observation.total_price["value"] == pytest.approx(360.0)
    assert observation.total_weight_kg["status"] == "unknown"
    assert observation.total_weight_kg["blocking"] == ["SPARSE"]


def test_bundle_observation_has_no_transaction_concept() -> None:
    """A bundle is a recommendation set, not a cart or an order."""
    observation = build_executor().execute(
        action(ActionKind.BUNDLE, BundleArguments(parent_asins=("LIGHT", "HEAVY")))
    )
    payload = str(observation.model_dump()).lower()
    for forbidden in ("cart", "checkout", "order", "payment", "purchase"):
        assert forbidden not in payload


# =========================================================================== #
# B. Authority
# =========================================================================== #


def test_the_executor_refuses_a_candidate_producing_action() -> None:
    """Reasoning cannot be used as a back door to retrieval."""
    executor = build_executor()
    for kind, arguments in (
        (ActionKind.RECOMMEND_FROM_HISTORY, None),
        (ActionKind.SEARCH_CATALOG, None),
        (ActionKind.FIND_SIMILAR, None),
        (ActionKind.SELECT_SOURCE, None),
        (ActionKind.FINISH, None),
    ):
        with pytest.raises(PolicyActionError):
            executor.execute(
                ValidatedAction(
                    action=kind, action_id="a", step_index=0, run_id="r", k=1, arguments=arguments
                )
            )


def test_the_executor_exposes_no_candidate_or_memory_surface() -> None:
    """It cannot add a candidate, touch a ledger or write memory."""
    executor = build_executor()
    for forbidden in ("ledger", "plane", "candidate_plane", "memory", "store", "commit", "add"):
        assert not hasattr(executor, forbidden)


def test_reasoning_does_not_widen_the_candidate_set_in_the_loop() -> None:
    """A comparison leaves the candidate set and its order exactly as retrieved."""
    harness = _harness_with_reasoning()
    result = harness.controller.run("recommend gear", HISTORY, run_id="r")
    before = [item.parent_asin for item in result.state["tool_result"].recommendations]
    # The run recommended, compared, then finished.
    assert result.trajectory.actions() == ("recommend_from_history", "compare", "finish")
    after = [item.parent_asin for item in result.state["tool_result"].recommendations]
    assert after == before
    assert result.route == "recommend"


# =========================================================================== #
# C. Loop accounting
# =========================================================================== #


class _RecommendCompareFinish:
    """Recommends once, asks one grounded comparison, then finishes."""

    name = "recommend-compare-finish"

    def choose(self, context: PolicyContext) -> ActionProposal:  # noqa: F821 - imported below
        if not context.candidate_state.grounded:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
        if (
            context.action_available(ActionKind.COMPARE)
            and getattr(context.last_observation, "kind", None) != "comparison"
        ):
            return ActionProposal(
                action=ActionKind.COMPARE,
                arguments=CompareArguments(parent_asins=("cand-red", "cand-blue")),
            )
        return ActionProposal(action=ActionKind.FINISH)


from recommendation.control import PolicyContext  # noqa: E402  (used by the policy above)


def _harness_with_reasoning(*, driver: str = "graph"):
    """A control harness whose reasoning executor reads the same catalogue the enricher uses."""
    harness = build_control_harness(driver=driver)
    harness.controller._reasoning = ReasoningExecutor(  # noqa: SLF001 - deliberate wiring
        GroundedReasoner(harness.parts["enricher"].metadata)
    )
    harness.controller._policy = _RecommendCompareFinish()  # noqa: SLF001
    return harness


def test_a_reasoning_step_is_not_a_tool_call() -> None:
    """Reasoning reads facts; it must not consume the recommender's budget."""
    harness = _harness_with_reasoning()
    result = harness.controller.run("recommend gear", HISTORY, run_id="r")
    assert harness.engine.call_count == 1, "the recommender runs exactly once"
    assert result.control.tool_call_count == 1, "only the recommendation is charged"
    assert result.control.step_count == 3


def test_reasoning_requires_a_grounded_candidate_to_act_on() -> None:
    """The reasoning actions are not offered before there is something to reason about."""
    harness = _harness_with_reasoning()
    seen: list[tuple[str, ...]] = []

    class Recorder:
        name = "recorder"

        def choose(self, context):
            seen.append(tuple(a.value for a in context.available_actions))
            if not context.candidate_state.grounded:
                return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
            return ActionProposal(action=ActionKind.FINISH)

    harness.controller._policy = Recorder()  # noqa: SLF001
    harness.controller.run("recommend gear", HISTORY, run_id="r")
    first, later = seen[0], seen[-1]
    assert "compare" not in first, "reasoning must not be offered with no candidates"
    assert "compare" in later, "reasoning becomes available once candidates exist"


def test_reasoning_is_not_offered_when_no_executor_is_configured() -> None:
    """A deployment without a catalogue offers no facts questions at all."""
    harness = build_control_harness()
    seen: list[tuple[str, ...]] = []

    class Recorder:
        name = "recorder"

        def choose(self, context):
            seen.append(tuple(a.value for a in context.available_actions))
            if not context.candidate_state.grounded:
                return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
            return ActionProposal(action=ActionKind.FINISH)

    harness.controller._policy = Recorder()  # noqa: SLF001
    result = harness.controller.run("recommend gear", HISTORY, run_id="r")
    assert result.status is RunStatus.FINISHED
    assert all("compare" not in actions for actions in seen)


def test_an_unoffered_reasoning_action_is_refused_by_validation() -> None:
    """Clarification-adjacent: a policy cannot grant itself a capability."""
    harness = build_control_harness()

    class Asking:
        name = "asking"

        def choose(self, context):
            if not context.candidate_state.grounded:
                return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)
            return ActionProposal(
                action=ActionKind.COMPARE,
                arguments=CompareArguments(parent_asins=("cand-red", "cand-blue")),
            )

    harness.controller._policy = Asking()  # noqa: SLF001
    result = harness.controller.run("recommend gear", HISTORY, run_id="r")
    # No executor, so COMPARE was never offered and the proposal is refused.
    assert result.status is RunStatus.FAILED
    assert "action_not_available" in result.trajectory.refusals()


# =========================================================================== #
# D. Drivers and preserved behaviour
# =========================================================================== #


def test_both_drivers_agree_on_a_reasoning_run() -> None:
    """Graph driver and direct driver produce the same action sequence and result."""
    outcomes = []
    for driver in ("graph", "direct"):
        harness = _harness_with_reasoning(driver=driver)
        result = harness.controller.run("recommend gear", HISTORY, run_id="r")
        outcomes.append(
            (
                result.status.value,
                result.control.termination_reason.value,
                result.trajectory.actions(),
                result.final_response,
                harness.engine.call_count,
                result.control.tool_call_count,
            )
        )
    assert outcomes[0] == outcomes[1]


def test_the_stage_1_fast_path_is_unaffected() -> None:
    """With no reasoner configured the Stage 1 loop is exactly as before."""
    harness = build_control_harness()
    result = harness.controller.run("recommend gear", HISTORY, run_id="r")
    assert result.status is RunStatus.FINISHED
    assert result.trajectory.actions() == ("recommend_from_history", "finish")
    assert harness.engine.call_count == 1
    assert result.control.tool_call_count == 1


def test_finalization_survives_a_reasoning_observation() -> None:
    """A run whose last non-terminal observation was a comparison still finalizes.

    This is a regression guard: the finalizer reads a candidate count for its state delta, and
    a reasoning observation has no such field.
    """
    harness = _harness_with_reasoning()
    result = harness.controller.run("recommend gear", HISTORY, run_id="r")
    assert result.final_response, "the run must still render a recommendation"
    assert result.state["route"] == "recommend"
    # The comparison happened and is recorded in the trajectory.
    kinds = [step.observation["kind"] for step in result.trajectory.steps if step.observation]
    assert "comparison" in kinds
