"""Round B: the durable reply assembler equals the in-memory BFF presentation.

The warm reply must not invent its own presentation.  These tests prove that
``web_demo.reply_assembly`` returns exactly the products, narrative, memory update and
preference state that the existing in-memory ``POST /api/clarification/reply`` produces for
the same real MCP payloads, and that anything incomplete is rejected instead of shown.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tests.test_web_demo_unit as twu  # noqa: E402
from web_demo.reply_assembly import (  # noqa: E402
    ReplyAssemblyError,
    ReplyAssemblyRequest,
    assemble_reply_presentation,
    new_recommendation_id,
    prepare_recommendation_products,
    reply_response_from_result,
)
from web_demo.reply_executor import CPU_RESUME_RANKING_SOURCE  # noqa: E402
from web_demo.reply_result_contract import build_reply_payload  # noqa: E402

QUERY = "I need some new headphones. Not sure what matters yet."
CATALOGUE_SIZE = 76263


def _request(**overrides) -> ReplyAssemblyRequest:
    recommendation = overrides.pop("recommendation", None) or twu.recommend_payload()
    values = dict(
        recommendation=recommendation,
        product_infos={recommendation.get("selected_product_id"): twu.info_payload()},
        reply=twu.reply_payload(),
        preference_payload=twu.preferences_payload(),
        query=QUERY,
        recommendation_id="rec-assembly",
        ranking_source=CPU_RESUME_RANKING_SOURCE,
        profile_id="profile_0057",
        turn_id="web-asm-r1",
    )
    values.update(overrides)
    return ReplyAssemblyRequest(**values)


def _memory_mode_presentation(tmp_path: Path) -> dict:
    """What the existing in-memory clarification reply returns for the same payloads."""
    manager = twu.FakeManager(writes_enabled=True)
    twu._clarification_setup(manager)
    calls = {"recommend": 0}

    def recommend(profile_id, arguments):  # noqa: ARG001
        calls["recommend"] += 1
        return twu.recommend_payload(
            status="awaiting_clarification" if calls["recommend"] == 1 else "finalized"
        )

    manager.responses.update(
        {
            "recommend_products": recommend,
            "submit_clarification_reply": twu.reply_payload(),
            "get_active_preferences": twu.preferences_payload(),
        }
    )
    with TestClient(twu.make_app(tmp_path, manager)) as client:
        session = twu.create_session(client, "profile_0057")
        chat = client.post(
            "/api/chat/stream",
            json={"session_id": session["session_id"], "message": QUERY},
        )
        _ = chat.text
        response = client.post(
            "/api/clarification/reply",
            json={
                "session_id": session["session_id"],
                "clarification_id": "clar_fake_handle",
                "message": "I prefer noise cancelling.",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        return {
            "recommendation": body["recommendation"],
            "message": body["message"],
            "memory_update": body["memory_update"],
            "preferences": body["preferences"],
        }


def _normalised(payload: dict, recommendation_id: str) -> dict:
    """Drop only the identities/timestamps that legitimately differ per run."""
    clone = copy.deepcopy(payload)
    products = clone.get("recommendation")
    if isinstance(products, dict):
        products["recommendation_id"] = recommendation_id
        products["request_id"] = "web-asm-r1"
    for container in (clone.get("preferences"), (clone.get("memory_update") or {}).get("preferences")):
        if isinstance(container, dict):
            container["fetched_at"] = None
    return clone


def test_assembler_matches_the_in_memory_bff_field_by_field(tmp_path: Path) -> None:
    memory = _memory_mode_presentation(tmp_path)
    assembly = assemble_reply_presentation(_request(), catalogue_size=CATALOGUE_SIZE)
    assert assembly["recommendation_id"] == "rec-assembly"

    durable = {
        "recommendation": assembly["products"],
        "message": assembly["narrative"],
        "memory_update": assembly["memory_update"],
        "preferences": assembly["preferences"],
    }
    assert _normalised(durable, "rec-memory") == _normalised(memory, "rec-memory")

    first_item = assembly["products"]["items"][0]
    assert first_item["product_id"] == "B004K6R43G"
    assert first_item["match_status"] == "recommended"
    assert first_item["recommended"] is True
    assert first_item["source"] == "recommendation"
    assert first_item["displayed_position"] == 1
    assert assembly["products"]["recommended_product_id"] == "B004K6R43G"
    assert assembly["products"]["catalogue_size"] == CATALOGUE_SIZE
    assert assembly["products"]["ranker_version"]
    assert assembly["memory_update"]["added"] == 2
    assert assembly["memory_update"]["preferences"]["profile_id"] == "profile_0057"


def test_assembler_rejects_missing_selected_product_enrichment() -> None:
    with pytest.raises(ReplyAssemblyError, match="not enriched"):
        assemble_reply_presentation(
            _request(product_infos={}), catalogue_size=CATALOGUE_SIZE
        )
    incomplete = twu.info_payload()
    incomplete.pop("selected_product")
    with pytest.raises(ReplyAssemblyError, match="missing selected_product"):
        assemble_reply_presentation(
            _request(product_infos={"B004K6R43G": incomplete}),
            catalogue_size=CATALOGUE_SIZE,
        )
    no_title = copy.deepcopy(twu.info_payload())
    no_title["selected_product"]["title"] = ""
    with pytest.raises(ReplyAssemblyError, match="no title"):
        assemble_reply_presentation(
            _request(product_infos={"B004K6R43G": no_title}),
            catalogue_size=CATALOGUE_SIZE,
        )


def test_assembler_rejects_non_final_or_failed_recommendations() -> None:
    clarifying = twu.recommend_payload(status="awaiting_clarification")
    with pytest.raises(ReplyAssemblyError, match="finalized"):
        assemble_reply_presentation(
            _request(recommendation=clarifying, product_infos={}),
            catalogue_size=CATALOGUE_SIZE,
        )
    failed = {**twu.recommend_payload(), "status": "error", "error": {"code": "boom"}}
    with pytest.raises(ReplyAssemblyError):
        assemble_reply_presentation(_request(recommendation=failed), catalogue_size=CATALOGUE_SIZE)


def test_assembler_never_mutates_its_inputs() -> None:
    request = _request()
    snapshot = copy.deepcopy({
        "recommendation": request.recommendation,
        "product_infos": request.product_infos,
        "reply": request.reply,
        "preference_payload": request.preference_payload,
    })
    assemble_reply_presentation(request, catalogue_size=CATALOGUE_SIZE)
    assert {
        "recommendation": request.recommendation,
        "product_infos": request.product_infos,
        "reply": request.reply,
        "preference_payload": request.preference_payload,
    } == snapshot


def test_recommendation_id_rule_is_issued_by_the_caller_once() -> None:
    first = new_recommendation_id()
    second = new_recommendation_id()
    assert first.startswith("rec-") and len(first) == len("rec-") + 12
    assert first != second
    int(first[len("rec-"):], 16)  # 12 hex characters


def test_prepare_recommendation_products_adds_identity_without_mutation() -> None:
    products = {"items": [], "query": "q"}
    prepared = prepare_recommendation_products(
        products, recommendation_id="rec-1", turn_id="web-r-r1"
    )
    assert prepared == {
        "items": [], "query": "q",
        "recommendation_id": "rec-1", "request_id": "web-r-r1",
        "ranker_version": prepared["ranker_version"],
    }
    assert products == {"items": [], "query": "q"}


def test_reply_response_from_result_rebuilds_the_same_response() -> None:
    request = _request()
    presentation = assemble_reply_presentation(request, catalogue_size=CATALOGUE_SIZE)
    contract = build_reply_payload(
        reply_turn_id="web-asm-r1",
        pending_id="pend-1",
        user_message="I prefer noise cancelling.",
        recommendation=request.recommendation,
        narrative=presentation["narrative"],
        recommendation_id=presentation["recommendation_id"],
        ranking_source=presentation["ranking_source"],
        products=presentation["products"],
    )
    result = {
        "reply": contract,
        "preferences": presentation["preferences"],
        "memory_update": presentation["memory_update"],
        "steps": [{"tool": "recommend_products", "status": "ok", "latency_ms": 1.0}],
    }
    response = reply_response_from_result(result, clarification_id="clar-1")
    assert response["turn_id"] == "web-asm-r1"
    assert response["message"] == presentation["narrative"]
    assert response["recommendation"] == presentation["products"]
    assert response["preferences"] == presentation["preferences"]
    assert response["memory_update"] == presentation["memory_update"]
    assert response["steps"] == result["steps"]
    assert response["clarification"] is None
    assert response["error"] is None
    with pytest.raises(ReplyAssemblyError):
        reply_response_from_result({"reply": {}}, clarification_id="clar-1")
