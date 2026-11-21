"""Pure recommendation metadata extraction (B1c-1a-1)."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_demo.demo_flow import (  # noqa: E402
    EXISTING_RANKER_VERSION,
    _attach_recommendation,
    prepare_recommendation_products,
)
from web_demo.sessions import Session  # noqa: E402


def _products() -> dict:
    return {
        "query": "wireless headphones",
        "search_mode": "bm25",
        "items": [{"item_id": "B0TEST", "title": "Headphones"}],
        "gate": {"action": "finalize", "allow_finalize": True},
    }


def test_prepare_returns_new_mapping_with_metadata_and_keeps_input() -> None:
    products = _products()
    before = copy.deepcopy(products)

    prepared = prepare_recommendation_products(
        products, recommendation_id="rec-fixed123456", turn_id="web-t-t1"
    )

    assert products == before, "the pure function must not mutate its input"
    assert prepared is not products
    assert prepared["recommendation_id"] == "rec-fixed123456"
    assert prepared["request_id"] == "web-t-t1"
    assert prepared["ranker_version"] == EXISTING_RANKER_VERSION
    assert prepared["items"] == before["items"]
    assert prepared["gate"] == before["gate"]


def test_prepare_is_deterministic_and_never_invents_an_id() -> None:
    first = prepare_recommendation_products(_products(), recommendation_id="rec-a", turn_id="t")
    second = prepare_recommendation_products(_products(), recommendation_id="rec-a", turn_id="t")
    assert first == second

    empty = prepare_recommendation_products(_products(), recommendation_id="", turn_id="t")
    assert empty["recommendation_id"] == "", "no uuid may be generated inside the pure function"


def test_prepared_output_roundtrips_through_json() -> None:
    prepared = prepare_recommendation_products(
        _products(), recommendation_id="rec-1", turn_id="t"
    )
    assert json.loads(json.dumps(prepared)) == prepared


def test_attach_recommendation_keeps_side_effects_and_identity() -> None:
    session = Session(session_id="web-t", profile_id="profile_0057")
    recorded: list[dict] = []
    original = session.record_recommendation

    def spy(products, **kwargs):
        recorded.append(copy.deepcopy(products))
        return original(products, **kwargs)

    session.record_recommendation = spy  # type: ignore[method-assign]
    products = _products()

    returned = _attach_recommendation(
        session, products, turn_id="web-t-t1", ranking_source="cpu_live:test"
    )

    assert returned is products, "the wrapper still returns the same dict object"
    assert re.fullmatch(r"rec-[0-9a-f]{12}", products["recommendation_id"])
    assert products["request_id"] == "web-t-t1"
    assert products["ranker_version"] == EXISTING_RANKER_VERSION
    assert len(recorded) == 1
    assert recorded[0]["recommendation_id"] == products["recommendation_id"]
    assert session.latest_recommendation_id == products["recommendation_id"]


def test_wrapper_fields_match_manual_old_construction() -> None:
    session = Session(session_id="web-t", profile_id="profile_0057")
    products = _products()
    _attach_recommendation(
        session, products, turn_id="web-t-t9", ranking_source="cpu_live:test"
    )
    expected_keys = set(_products()) | {
        "recommendation_id", "request_id", "ranker_version"
    }
    assert set(products) == expected_keys
    assert products["query"] == "wireless headphones"
    assert products["gate"]["allow_finalize"] is True
