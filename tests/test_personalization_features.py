"""Feature-flag, memory-retrieval and behavior-rerank unit tests.

The executor tests use a deterministic fake search source so the four flag combinations
are exercised without depending on BM25 tokenisation.  Real FastAPI + MCP + BM25 coverage
is in ``test_web_personalization_e2e.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14e.catalogue import (
    ELECTRONICS_NAMESPACE,
    ElectronicsCatalogue,
    ElectronicsProduct,
)
from recommendation.catalog.schemas import ProductMetadata
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.mcp.server import _memory_query_values
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import (
    PHASE_PRODUCT_RETRIEVE,
    P14ARulePlanner,
)
from web_demo.behavior import build_behavior_snapshot
from web_demo.config import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_IDS = ("B0BASE1", "B0BASE2", "B0BASE3", "B0BASE4", "B0BASE5")
MEMORY_ID = "B0MEMORY"


def _product(item_id: str, title: str) -> ElectronicsProduct:
    text = f"{title}. Bluetooth audio product."
    return ElectronicsProduct(
        item_id=item_id,
        namespaced_id=f"{ELECTRONICS_NAMESPACE}:{item_id}",
        title=title,
        text=text,
        product_type="headphones",
        attributes={"wireless": {"status": "present", "evidence": "wireless"}},
        form_factor="in_ear",
        record=ProductMetadata(
            parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
            title=title,
            main_category="Electronics",
            categories=("Electronics", "Headphones"),
            description=(text,),
            source="test-fixture:personalization",
        ),
    )


def _catalogue() -> ElectronicsCatalogue:
    products = {
        item_id: _product(item_id, f"Wireless Studio Headphones {index}")
        for index, item_id in enumerate(BASE_IDS, start=1)
    }
    products[MEMORY_ID] = _product(MEMORY_ID, "Wireless Studio Headphones Memory")
    catalogue = ElectronicsCatalogue(products, {})

    class _Source:
        def search(self, query: str, limit: int = 20):
            ids = [MEMORY_ID] if "aux" in query else list(BASE_IDS)
            hits = [
                SimpleNamespace(
                    parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}",
                    score=float(10 - index),
                )
                for index, item_id in enumerate(ids)
            ]
            return SimpleNamespace(
                hits=hits, score_kind="fixture", catalogue_size=len(products)
            )

    catalogue._search_source = _Source()
    return catalogue


def _run(
    *,
    memory_retrieval: bool,
    memory_queries: tuple[str, ...] = (),
    behavior_snapshot: dict | None = None,
) -> ElectronicsPlannerExecutor:
    executor = ElectronicsPlannerExecutor(
        catalogue=_catalogue(),
        current_query="wireless studio headphones",
        preferences=[],
        hard=[],
        memory_retrieval=memory_retrieval,
        memory_queries=memory_queries,
        behavior_snapshot=behavior_snapshot,
    )
    observation = PlannerObservation(
        step=0,
        query="wireless studio headphones",
        m5_state={
            "phase": PHASE_PRODUCT_RETRIEVE,
            "next_query": "wireless studio headphones",
            "preferences": [],
        },
    )
    run_planner(
        planner=P14ARulePlanner(),
        executor=executor,
        initial_observation=observation,
        max_steps=MAX_STEPS,
    )
    return executor


def _behavior(score: float = 3.0) -> dict:
    return {
        "schema": "agentrecx.behavior_snapshot.v1",
        "product_interests": [{"product_id": "B0BASE2", "score": score}],
        "attribute_interests": [],
    }


def test_feature_flags_default_off_and_parse_independently() -> None:
    default = load_settings({})
    assert default.memory_aware_retrieval is False
    assert default.behavior_aware_rerank is False

    memory_only = load_settings({"AGENTRECX_MEMORY_AWARE_RETRIEVAL": "1"})
    assert memory_only.memory_aware_retrieval is True
    assert memory_only.behavior_aware_rerank is False

    behavior_only = load_settings({"AGENTRECX_BEHAVIOR_AWARE_RERANK": "true"})
    assert behavior_only.memory_aware_retrieval is False
    assert behavior_only.behavior_aware_rerank is True

    both = load_settings({
        "AGENTRECX_MEMORY_AWARE_RETRIEVAL": "1",
        "AGENTRECX_BEHAVIOR_AWARE_RERANK": "1",
    })
    assert both.memory_aware_retrieval is True
    assert both.behavior_aware_rerank is True


def test_memory_queries_are_positive_bounded_and_deduplicated() -> None:
    queries = _memory_query_values(
        "studio headphones",
        [
            {"value": "wireless", "polarity": "positive"},
            {"value": "noise_cancelling", "polarity": "prefer"},
            {"value": "wired", "polarity": "negative"},
            {"value": "wireless", "polarity": "positive"},
            {"value": "usb", "polarity": "positive"},
        ],
    )
    assert queries == [
        "studio headphones wireless",
        "studio headphones noise_cancelling",
    ]
    assert _memory_query_values("wireless headphones", [
        {"value": "wireless", "polarity": "positive"},
    ]) == []


def test_behavior_snapshot_is_bounded_and_contains_no_raw_evidence() -> None:
    profile = {
        "schema": "behavior_feedback_profile_v1.2",
        "generated_at": "2026-11-22T00:00:00Z",
        "as_of": "2026-11-22T00:00:00Z",
        "window_seconds": 3600,
        "half_life_seconds": 1800,
        "product_interests": [{
            "product_id": "B0BASE2", "product_title": "raw title",
            "score": 2.0, "positive_score": 2.0, "negative_score": 0.0,
            "counted_events": 1, "last_event_at": "now",
        }],
        "attribute_interests": [{
            "attribute": "wireless", "score": 2.0, "distinct_products": 1,
            "evidence_count": 1, "source_event_types": ["explicit_item_feedback"],
            "evidence": [{"event_id": "raw-event"}],
        }],
    }
    snapshot = build_behavior_snapshot(profile, session_id="web-1")
    assert snapshot["schema"] == "agentrecx.behavior_snapshot.v1"
    assert snapshot["product_interests"][0]["product_id"] == "B0BASE2"
    assert snapshot["attribute_interests"][0]["attribute"] == "wireless"
    assert "product_title" not in snapshot["product_interests"][0]
    assert "evidence" not in snapshot["attribute_interests"][0]


def test_scorer_behavior_is_soft_bounded_and_click_is_weaker_than_like() -> None:
    scorer = PreferenceAwareScorer()
    candidates = [
        {
            "product_ref": {"namespace": "ns", "item_id": f"P{index}"},
            "query_rank": index,
            "attributes": {},
            "form_factor": None,
        }
        for index in range(1, 6)
    ]
    no_behavior = scorer.score(
        preferences=[], current_query="q", candidates=candidates, namespace="ns",
    )
    like = scorer.score(
        preferences=[], current_query="q", candidates=candidates, namespace="ns",
        behavior_snapshot={
            "product_interests": [{"product_id": "P5", "score": 3.0}],
        },
    )
    click = scorer.score(
        preferences=[], current_query="q", candidates=candidates, namespace="ns",
        behavior_snapshot={
            "product_interests": [{"product_id": "P5", "score": 1.0}],
        },
    )
    dislike = scorer.score(
        preferences=[], current_query="q", candidates=candidates, namespace="ns",
        behavior_snapshot={
            "product_interests": [{"product_id": "P1", "score": -2.0}],
        },
    )
    by_no = {row["product_ref"]["item_id"]: row for row in no_behavior}
    by_like = {row["product_ref"]["item_id"]: row for row in like}
    by_click = {row["product_ref"]["item_id"]: row for row in click}
    by_dislike = {row["product_ref"]["item_id"]: row for row in dislike}
    assert by_like["P5"]["behavior_score"] > 0
    assert by_like["P5"]["rank"] < by_no["P5"]["rank"]
    assert by_click["P5"]["behavior_score"] < by_like["P5"]["behavior_score"]
    assert by_dislike["P1"]["behavior_score"] < 0
    assert abs(by_like["P5"]["behavior_score"]) <= 0.30


def test_four_flag_combinations_are_independent() -> None:
    off_off = _run(memory_retrieval=False)
    assert off_off.last_memory_retrieval["queries"] == []
    assert off_off.last_search["hit_ids"] == list(BASE_IDS)
    assert off_off.selected_candidate_id == "B0BASE1"

    on_off = _run(memory_retrieval=True, memory_queries=("aux",))
    assert on_off.last_memory_retrieval["new_item_ids"] == [MEMORY_ID]
    assert on_off.selected_candidate_id == "B0BASE1"
    assert "B0MEMORY" in on_off.last_search["hit_ids"]

    off_on = _run(memory_retrieval=False, behavior_snapshot=_behavior())
    assert off_on.last_memory_retrieval["queries"] == []
    assert off_on.last_behavior["candidates_with_behavior_signal"] == 1
    assert off_on.selected_candidate_id == "B0BASE2"

    on_on = _run(
        memory_retrieval=True, memory_queries=("aux",),
        behavior_snapshot=_behavior(),
    )
    assert on_on.last_memory_retrieval["new_item_ids"] == [MEMORY_ID]
    assert on_on.last_behavior["candidates_with_behavior_signal"] == 1
    assert on_on.selected_candidate_id == "B0BASE2"
