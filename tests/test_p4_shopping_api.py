"""Phase 4.1: the multi-turn shopping route over HTTP.

The route is deliberately thin: session lookup, server-owned history, and a call to
``DemoAgentService.run_shopping_turn``.  These tests therefore assert the wire contract
(validation, clarification round-trip, session isolation, expiry and duplicate handling)
rather than re-testing the shopping logic, which Phase 4's tests and scenario suite cover.
"""

from __future__ import annotations

import re

from tests.demo_fixture import PROFILE_A, PROFILE_B
from tests.test_demo_agent_api import HISTORY, harness  # noqa: F401 - pytest fixture

BASE = "/v1/demo"
CREATE = BASE + "/sessions"
SHOPPING = BASE + "/sessions/{session_id}/agent/shopping"
CAND_ID = re.compile(r"cand-[a-z]+")


def _session(harness, profile: str = PROFILE_A) -> str:
    response = harness.client.post(CREATE, json={"profile_id": profile})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def _shopping(harness, session_id: str, message: str, **overrides):
    body = {"message": message, "k": 3, "sources": ["history", "catalog_search"]}
    body.update(overrides)
    return harness.client.post(SHOPPING.format(session_id=session_id), json=body)


def _ids(text: str) -> list[str]:
    return CAND_ID.findall(text)


# --------------------------------------------------------------------------- #
# published contract
# --------------------------------------------------------------------------- #


def test_the_shopping_route_is_published_in_openapi(harness) -> None:
    schema = harness.client.get("/openapi.json").json()
    assert SHOPPING in schema["paths"]
    assert "post" in schema["paths"][SHOPPING]


def test_request_validation_rejects_out_of_range_k_and_extra_fields(harness) -> None:
    session = _session(harness)
    assert _shopping(harness, session, "hiking gear", k=0).status_code == 422
    assert _shopping(harness, session, "hiking gear", k=101).status_code == 422
    # The session owns history and memory; a caller cannot smuggle either through the body.
    smuggled = harness.client.post(
        SHOPPING.format(session_id=session),
        json={"message": "hiking gear", "history": ["cand-red"], "user_key": "other"},
    )
    assert smuggled.status_code == 422


def test_a_normal_turn_returns_the_final_list_intent_and_reasons(harness) -> None:
    session = _session(harness)
    response = _shopping(harness, session, "Recommend some hiking gear.")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["route"] == "recommend"
    assert body["recommendations"], "the synthetic history returns candidates"
    assert body["intent"] is not None
    assert body["clarification"] is None
    assert body["terminal"]["status"] == "finished"
    for item in body["recommendations"]:
        assert item["reason"], "every final item carries an evidence-bound reason"
    # The rendered explanation names exactly the final identities, in order.
    assert _ids(body["message"]) == [item["parent_asin"] for item in body["recommendations"]]


# --------------------------------------------------------------------------- #
# clarification round-trip
# --------------------------------------------------------------------------- #


def test_a_conflict_suspends_with_one_pending_question(harness) -> None:
    session = _session(harness)
    response = _shopping(
        harness,
        session,
        "Recommend some hiking gear.",
        constraints=[
            {"kind": "price_max", "value": "20"},
            {"kind": "price_min", "value": "30"},
        ],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recommendations"] == []
    assert body["terminal"]["status"] == "waiting_for_user"
    assert body["terminal"]["termination_reason"] == "awaiting_user"
    assert body["clarification"]["kind"] == "conflict"
    assert body["clarification"]["question"]


def test_answering_with_the_id_updates_intent_and_recommends(harness) -> None:
    session = _session(harness)
    first = _shopping(
        harness,
        session,
        "Recommend some hiking gear.",
        constraints=[
            {"kind": "price_max", "value": "20"},
            {"kind": "price_min", "value": "30"},
        ],
    ).json()
    clarification_id = first["clarification"]["clarification_id"]
    second = _shopping(
        harness, session, "make it under $30 instead", clarification_id=clarification_id
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["clarification"] is None
    assert body["recommendations"]
    assert body["terminal"]["status"] == "finished"
    hard = {(entry["kind"], entry["value"]) for entry in body["intent"]["hard"]}
    assert ("price_max", "30") in hard
    for item in body["recommendations"]:
        assert item["reason"]


def test_an_unverifiable_constraint_asks_and_then_completes(harness) -> None:
    session = _session(harness)
    first = _shopping(
        harness,
        session,
        "Recommend some hiking gear.",
        constraints=[{"kind": "free_form_constraint", "value": "extra-large"}],
    )
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["clarification"]["kind"] == "unverified"
    assert body["recommendations"] == []
    second = _shopping(
        harness,
        session,
        "make it under $30 instead",
        clarification_id=body["clarification"]["clarification_id"],
    )
    assert second.status_code == 200, second.text
    assert second.json()["recommendations"]
    assert second.json()["clarification"] is None


# --------------------------------------------------------------------------- #
# failure and isolation
# --------------------------------------------------------------------------- #


def test_an_unknown_clarification_id_is_404(harness) -> None:
    session = _session(harness)
    response = _shopping(harness, session, "hiking gear", clarification_id="does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_clarification"


def test_a_cross_session_answer_is_rejected_without_consuming_the_question(harness) -> None:
    first_session = _session(harness)
    second_session = _session(harness, PROFILE_B)
    body = _shopping(
        harness,
        first_session,
        "Recommend some hiking gear.",
        constraints=[
            {"kind": "price_max", "value": "20"},
            {"kind": "price_min", "value": "30"},
        ],
    ).json()
    clarification_id = body["clarification"]["clarification_id"]
    denial = _shopping(
        harness, second_session, "make it under $30 instead", clarification_id=clarification_id
    )
    assert denial.status_code == 404
    assert denial.json()["error"] == "unknown_clarification"
    # The first session's pending question was not destroyed by the cross-session attempt.
    accepted = _shopping(
        harness,
        first_session,
        "make it under $30 instead",
        clarification_id=clarification_id,
    )
    assert accepted.status_code == 200, accepted.text


def test_a_duplicate_answer_is_404(harness) -> None:
    session = _session(harness)
    body = _shopping(
        harness,
        session,
        "Recommend some hiking gear.",
        constraints=[
            {"kind": "price_max", "value": "20"},
            {"kind": "price_min", "value": "30"},
        ],
    ).json()
    clarification_id = body["clarification"]["clarification_id"]
    assert _shopping(
        harness, session, "make it under $30 instead", clarification_id=clarification_id
    ).status_code == 200
    duplicate = _shopping(
        harness, session, "make it under $30 instead", clarification_id=clarification_id
    )
    assert duplicate.status_code == 404


def test_an_expired_clarification_is_410(harness) -> None:
    from recommendation.demo.agent_service import CLARIFICATION_TTL_SECONDS

    session = _session(harness)
    body = _shopping(
        harness,
        session,
        "Recommend some hiking gear.",
        constraints=[
            {"kind": "price_max", "value": "20"},
            {"kind": "price_min", "value": "30"},
        ],
    ).json()
    clarification_id = body["clarification"]["clarification_id"]
    service = harness.runtime.agent_service()
    with service._lock:
        service._pending_clarifications[clarification_id]["created_at"] -= (
            CLARIFICATION_TTL_SECONDS + 1
        )
    response = _shopping(
        harness, session, "make it under $30 instead", clarification_id=clarification_id
    )
    assert response.status_code == 410
    assert response.json()["error"] == "clarification_expired"
    # still gone afterwards
    again = _shopping(
        harness, session, "make it under $30 instead", clarification_id=clarification_id
    )
    assert again.status_code == 404


# --------------------------------------------------------------------------- #
# constraints, identity and evidence
# --------------------------------------------------------------------------- #


def test_a_hard_budget_is_bound_on_the_presented_items(harness) -> None:
    from recommendation.control.grounded_reasoning import parse_price

    session = _session(harness)
    body = _shopping(
        harness,
        session,
        "Recommend some hiking gear.",
        constraints=[{"kind": "price_max", "value": "20"}],
    ).json()
    assert body["recommendations"]
    for item in body["recommendations"]:
        record = harness.runtime.metadata.records[item["parent_asin"]]
        price = parse_price(record.price_text)
        assert price is not None and price <= 20.0
        assert "catalogue price" in item["reason"]


def test_each_reason_uses_that_products_own_price(harness) -> None:
    from recommendation.control.grounded_reasoning import parse_price

    session = _session(harness)
    body = _shopping(harness, session, "Recommend some hiking gear.")
    for item in body.json()["recommendations"]:
        record = harness.runtime.metadata.records[item["parent_asin"]]
        price = parse_price(record.price_text)
        if price is not None:
            assert f"${price:.2f}" in item["reason"]


def test_the_traditional_agent_endpoint_still_works(harness) -> None:
    response = harness.client.post(
        "/v1/demo/agent/recommend",
        json={"message": "Recommend some hiking gear.", "history": list(HISTORY), "k": 3},
    )
    assert response.status_code == 200, response.text
    assert response.json()["recommendations"]
