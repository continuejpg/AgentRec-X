"""P0: current-task hard exclusions are executed, not just scored.

These are focused CPU executor tests over a real ``ElectronicsPlannerExecutor`` and a
synthetic catalogue.  They intentionally use the same ``CurrentTaskConstraints`` objects
that the Web reply coordinator persists, so a passing test means the hard constraint reached
the executor and the Gate, not merely the extraction layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.p14c0.electronics_executor import (  # noqa: E402
    MAX_STEPS,
    ElectronicsPlannerExecutor,
)
from experiments.p14e.catalogue import (  # noqa: E402
    ELECTRONICS_NAMESPACE,
    ElectronicsCatalogue,
    ElectronicsProduct,
)
from recommendation.catalog.schemas import ProductMetadata  # noqa: E402
from recommendation.demo.planner_contract import PlannerObservation, run_planner  # noqa: E402
from recommendation.preference.current_task_constraints import (  # noqa: E402
    CurrentTaskConstraint,
    CurrentTaskConstraints,
)
from recommendation.shopping_companion.planner_bridge import (  # noqa: E402
    PHASE_PRODUCT_RETRIEVE,
    P14ARulePlanner,
)


def _product(item_id: str, title: str, *, attributes: dict) -> ElectronicsProduct:
    text = f"{title}. Bluetooth audio product."
    return ElectronicsProduct(
        item_id=item_id,
        namespaced_id=f"{ELECTRONICS_NAMESPACE}:{item_id}",
        title=title,
        text=text,
        product_type="headphones",
        attributes=attributes,
        form_factor="in_ear",
        record=ProductMetadata(
            parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
            title=title,
            main_category="Electronics",
            categories=("Electronics", "Headphones"),
            description=(text,),
            source="test-fixture:task-constraints",
        ),
    )


def _catalogue() -> ElectronicsCatalogue:
    products = {
        # Deliberately first / strongest lexical match: Memory/BM25 must not override the
        # task's hard exclusion.
        "B0WIRED": _product(
            "B0WIRED",
            "Wired Headphones with Microphone",
            attributes={"wired": {"status": "present", "evidence": "wired"}},
        ),
        "B0WIRELESS": _product(
            "B0WIRELESS",
            "Wireless Headphones with Microphone",
            attributes={"wireless": {"status": "present", "evidence": "wireless"}},
        ),
        "B0UNKNOWN": _product(
            "B0UNKNOWN",
            "Headphones with Microphone",
            attributes={},
        ),
    }
    return ElectronicsCatalogue(products, {})


def _constraints() -> CurrentTaskConstraints:
    return CurrentTaskConstraints(
        hard_exclusions=[
            CurrentTaskConstraint(
                attribute="wired",
                value="wired",
                kind="feature",
                polarity="exclude",
                source_text="Please avoid wired headphones.",
                source_turn_id="reply-1",
            )
        ],
        soft_preferences=[],
        source_turns=["reply-1"],
    )


def _run(
    *,
    constraints: CurrentTaskConstraints | None,
    preferences: list[dict] | None = None,
) -> ElectronicsPlannerExecutor:
    executor = ElectronicsPlannerExecutor(
        catalogue=_catalogue(),
        current_query="wireless headphones",
        preferences=preferences or [],
        hard=[],
        task_constraints=constraints,
    )
    observation = PlannerObservation(
        step=0,
        query="wireless headphones",
        m5_state={
            "phase": PHASE_PRODUCT_RETRIEVE,
            "next_query": "wireless headphones",
            "preferences": executor.preferences,
        },
    )
    run_planner(
        planner=P14ARulePlanner(),
        executor=executor,
        initial_observation=observation,
        max_steps=MAX_STEPS,
    )
    return executor


def test_hard_exclusion_excludes_wired_and_unknown_candidates() -> None:
    executor = _run(constraints=_constraints())
    assert executor.terminal_status == "finalized", executor.tool_calls
    assert executor.selected_candidate_id == "B0WIRELESS"
    assert [row["product_ref"]["item_id"] for row in executor.last_rows] == ["B0WIRELESS"]
    summary = executor.last_task_constraint_summary
    assert summary["task_constraint_violations"] == 1
    assert summary["task_constraint_unknown"] == 1
    assert set(summary["excluded_candidate_ids"]) == {"B0WIRED", "B0UNKNOWN"}
    assert executor._rows_by_rank["B0WIRELESS"]["task_constraint_hard_verified"]
    assert executor._rows_by_rank["B0WIRELESS"]["task_constraint_status"]["status"] == "verified"


def test_task_hard_exclusion_overrides_positive_m9_preference() -> None:
    # M9 says the user likes wired; the current task says avoid wired.  The task wins for
    # this purchase, but the M9 input list itself is not mutated.
    m9 = [{"value": "wired", "polarity": "positive", "memory_id": "m9-wired"}]
    executor = _run(constraints=_constraints(), preferences=m9)
    assert executor.history_preferences == m9
    assert any(
        item["value"] == "wired" and item["polarity"] == "negative"
        for item in executor.preferences
    )
    assert executor.selected_candidate_id == "B0WIRELESS"
    assert "B0WIRED" not in [row["product_ref"]["item_id"] for row in executor.last_rows]


def test_all_candidates_unknown_or_violating_is_a_safe_refusal() -> None:
    products = {
        "B0WIRED": _product(
            "B0WIRED",
            "Wired Headphones",
            attributes={"wired": {"status": "present", "evidence": "wired"}},
        ),
        "B0UNKNOWN": _product("B0UNKNOWN", "Headphones", attributes={}),
    }
    executor = ElectronicsPlannerExecutor(
        catalogue=ElectronicsCatalogue(products, {}),
        current_query="wireless headphones",
        preferences=[],
        hard=[],
        task_constraints=_constraints(),
    )
    observation = PlannerObservation(
        step=0,
        query="wireless headphones",
        m5_state={
            "phase": PHASE_PRODUCT_RETRIEVE,
            "next_query": "wireless headphones",
            "preferences": [],
        },
    )
    run_planner(
        planner=P14ARulePlanner(),
        executor=executor,
        initial_observation=observation,
        max_steps=MAX_STEPS,
    )
    assert executor.selected_candidate_id is None
    assert executor.terminal_status == "no_verified_candidate_in_scope"
    assert executor.last_rows == []
    assert set(executor.last_task_constraint_summary["excluded_candidate_ids"]) == {
        "B0WIRED", "B0UNKNOWN",
    }
