"""Phase 4.2: the browser demo serves the shopping controls and wires the right endpoint.

The demo has no browser test harness, so this is a static contract check over the two assets
the FastAPI app actually serves: the shopping panel must exist in ``index.html`` and the
client must call the shopping endpoint with the clarification id.  The behaviour behind it
is covered by ``tests/test_p4_shopping_api.py`` and the Phase 4 intent/clarification tests.
"""

from __future__ import annotations

from tests.demo_fixture import PROFILE_A
from tests.test_demo_agent_api import harness  # noqa: F401 - pytest fixture


def test_the_demo_page_carries_the_shopping_controls(harness) -> None:
    page = harness.client.get("/demo/")
    assert page.status_code == 200
    for node_id in (
        "shop-form",
        "shop-message",
        "shop-max-price",
        "shop-send",
        "shop-status",
        "shop-transcript",
        "shop-clarification",
        "shop-intent",
        "shop-cards",
    ):
        assert f'id="{node_id}"' in page.text, node_id


def test_the_demo_script_calls_the_shopping_endpoint_with_the_clarification_id(harness) -> None:
    script = harness.client.get("/demo/app.js")
    assert script.status_code == 200
    body = script.text
    assert "/agent/shopping" in body
    assert "clarification_id" in body
    assert "unknown_clarification" in body
    assert "clarification_expired" in body
    # The page only renders server data; it must not inject markup or invent attributes.
    assert ".innerHTML" not in body
    assert "price_max" in body


def test_the_demo_flow_from_request_to_answer(harness) -> None:
    """The exact HTTP sequence the page performs: request, question, answer, cards."""
    session = harness.client.post(
        "/v1/demo/sessions", json={"profile_id": PROFILE_A}
    )
    assert session.status_code == 201, session.text
    session_id = session.json()["session_id"]
    endpoint = f"/v1/demo/sessions/{session_id}/agent/shopping"

    first = harness.client.post(endpoint, json={
        "message": "Recommend some hiking gear.",
        "k": 3,
        "constraints": [
            {"kind": "price_max", "value": "20"},
            {"kind": "price_min", "value": "30"},
        ],
    })
    assert first.status_code == 200, first.text
    question = first.json()["clarification"]
    assert question["kind"] == "conflict"

    second = harness.client.post(endpoint, json={
        "message": "make it under $30 instead",
        "k": 3,
        "clarification_id": question["clarification_id"],
    })
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["clarification"] is None
    assert body["recommendations"]
    assert all(item.get("reason") for item in body["recommendations"])
    assert body["intent"] is not None
    assert body["terminal"]["status"] == "finished"
