"""Phase 6C: research clarifications and their recoverable answers.

The tests drive the real shopping route through the demo harness: the synthetic catalogue,
the real Phase 6A research tools, the real Phase 6B dynamic policy, and the same session /
pending-clarification mechanism.  No external model is called.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.agent_schemas import AgentResearchClarificationView  # noqa: E402
from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402

BASE = "/v1/demo/sessions/{session}/agent/shopping"
FIRST = "compare the first two by price, color, and material"


def _harness(tmp_path: Path) -> Any:
    return build_harness(tmp_path)


def _session(harness: Any) -> str:
    return harness.create_session(PROFILE_A)["session_id"]


def _first(harness: Any, message: str = FIRST) -> tuple[str, dict[str, Any]]:
    session = _session(harness)
    response = harness.client.post(
        BASE.format(session=session),
        json={"message": message, "k": 3, "sources": ["history", "catalog_search"]},
    )
    assert response.status_code == 200, response.text
    return session, response.json()


def _answer(
    harness: Any,
    session: str,
    clarification_id: str,
    *,
    message: str = "show facts",
    option: str | None = None,
) -> Any:
    body = {
        "message": message,
        "k": 3,
        "sources": ["history", "catalog_search"],
        "clarification_id": clarification_id,
    }
    if option is not None:
        body["clarification_option"] = option
    return harness.client.post(BASE.format(session=session), json=body)


def _research_options(payload: dict[str, Any]) -> dict[str, Any]:
    clarification = payload["clarification"]
    return {option["option_id"]: option for option in clarification["research_options"]}


def test_missing_field_produces_real_structured_options(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        _, payload = _first(harness)
        clarification = payload["clarification"]
        assert clarification["reason"] == "missing_catalogue_attribute"
        assert clarification["missing_attributes"] == ["material"]
        assert clarification["products"] == ["cand-red", "cand-blue"]
        assert clarification["known_facts"] == {
            "cand-red": {"price": 25.0, "color": "red"},
            "cand-blue": {"price": 40.0, "color": "blue"},
        }
        options = _research_options(payload)
        assert set(options) == {
            "compare:price", "compare:color", "known_facts", "stop_research"
        }
        assert options["compare:price"]["action"] == "compare_attributes"
        assert options["compare:price"]["attributes"] == ["price"]
        assert options["compare:price"]["products"] == ["cand-red", "cand-blue"]
        assert options["known_facts"]["action"] == "show_known_facts"
        assert options["stop_research"]["action"] == "stop_research"
        # The first response must not present products while the question is pending.
        assert payload["recommendations"] == []
        assert payload["research"]["executed"] is True
        assert payload["terminal"]["tool_calls"] == 4
    finally:
        harness.runtime.close()


def test_other_attribute_selection_runs_only_the_new_comparison(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session, payload = _first(harness)
        first_id = payload["clarification"]["clarification_id"]
        response = _answer(
            harness,
            session,
            first_id,
            message="compare price instead",
            option="compare:price",
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert body["clarification"] is None
        assert body["trajectory"]["actions"] == [
            "recommend_from_history",
            "search_catalog",
            "compare_products",
            "finish",
        ]
        # The missing material comparison is not repeated and no inspect is issued.
        assert "inspect_product" not in body["trajectory"]["actions"]
        comparisons = [
            action for action in body["research"]["actions"]
            if action["tool"] == "compare_products"
        ]
        assert comparisons
        assert "price" in comparisons[-1]["attributes"]
        assert "material" not in comparisons[-1]["attributes"]
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]
        assert body["terminal"]["tool_calls"] == 3
    finally:
        harness.runtime.close()


def test_known_facts_option_returns_facts_without_another_tool(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session, payload = _first(harness)
        first_id = payload["clarification"]["clarification_id"]
        response = _answer(
            harness, session, first_id, message="show facts", option="known_facts"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert body["recommendations"]
        assert body["trajectory"]["actions"] == [
            "recommend_from_history", "search_catalog", "finish"
        ]
        assert body["terminal"]["tool_calls"] == 2
        assert body["research"]["executed"] is True
        assert "previously verified facts" in body["research"]["note"]
        # The facts are the real verified values, not a statement from this message.
        compare_actions = [
            action for action in body["research"]["actions"]
            if action["tool"] == "compare_products"
        ]
        assert compare_actions
        facts = compare_actions[0]["rows"]
        price = next(row for row in facts if row["attribute"] == "price")
        assert price["facts"]["cand-red"]["value"] == 25.0
        assert price["facts"]["cand-blue"]["value"] == 40.0
    finally:
        harness.runtime.close()


def test_stop_option_does_not_call_a_research_tool(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session, payload = _first(harness)
        first_id = payload["clarification"]["clarification_id"]
        response = _answer(
            harness, session, first_id, message="stop research", option="stop_research"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert body["trajectory"]["actions"] == [
            "recommend_from_history", "search_catalog", "finish"
        ]
        assert body["terminal"]["tool_calls"] == 2
        assert "research stopped" in body["research"]["note"]
    finally:
        harness.runtime.close()


def test_invalid_option_does_not_consume_the_clarification(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session, payload = _first(harness)
        first_id = payload["clarification"]["clarification_id"]
        invalid = _answer(
            harness, session, first_id, message="unknown", option="not-an-option"
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"] == "invalid_clarification_option"
        # The pending question survived the invalid attempt and can still be answered.
        valid = _answer(
            harness, session, first_id, message="show facts", option="known_facts"
        )
        assert valid.status_code == 200, valid.text
        assert valid.json()["terminal"]["status"] == "finished"
    finally:
        harness.runtime.close()


def test_free_text_answer_never_becomes_a_catalogue_fact(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session, payload = _first(harness)
        first_id = payload["clarification"]["clarification_id"]
        response = _answer(
            harness,
            session,
            first_id,
            message="the material is leather",
            option=None,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        # The user's words selected the safe known-facts fallback; no material value was
        # invented anywhere in the research result.
        for action in body["research"]["actions"]:
            for fact in (action.get("facts") or {}).values():
                assert fact["value"] != "leather"
            for row in action.get("rows") or []:
                for fact in (row.get("facts") or {}).values():
                    assert fact["value"] != "leather"
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]
    finally:
        harness.runtime.close()


def test_research_view_is_structured_and_serialisable(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        _, payload = _first(harness)
        clarification = AgentResearchClarificationView.model_validate(
            {
                "reason": payload["clarification"]["reason"],
                "missing_attributes": payload["clarification"]["missing_attributes"],
                "products": payload["clarification"]["products"],
                "known_facts": payload["clarification"]["known_facts"],
                "options": payload["clarification"]["research_options"],
            }
        )
        assert clarification.options[0].option_id == "compare:price"
        assert clarification.missing_attributes == ("material",)
    finally:
        harness.runtime.close()


def test_cross_session_option_is_rejected_without_consuming_the_original(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    try:
        first_session, payload = _first(harness)
        second_session = _session(harness)
        clarification_id = payload["clarification"]["clarification_id"]
        cross = _answer(
            harness,
            second_session,
            clarification_id,
            message="show facts",
            option="known_facts",
        )
        assert cross.status_code == 404
        # The first session still owns the pending question.
        valid = _answer(
            harness,
            first_session,
            clarification_id,
            message="show facts",
            option="known_facts",
        )
        assert valid.status_code == 200, valid.text
    finally:
        harness.runtime.close()


def test_expired_research_clarification_is_410(tmp_path: Path, monkeypatch: Any) -> None:
    harness = _harness(tmp_path)
    try:
        session, payload = _first(harness)
        clarification_id = payload["clarification"]["clarification_id"]
        from recommendation.demo import agent_service

        monkeypatch.setattr(agent_service, "CLARIFICATION_TTL_SECONDS", -1)
        expired = _answer(
            harness,
            session,
            clarification_id,
            message="show facts",
            option="known_facts",
        )
        assert expired.status_code == 410
        # The expired entry was removed; retrying is not a second consumption path.
        again = _answer(
            harness,
            session,
            clarification_id,
            message="show facts",
            option="known_facts",
        )
        assert again.status_code == 404
    finally:
        harness.runtime.close()


def test_recovery_preserves_hard_constraints_and_final_identity(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session = _session(harness)
        first = harness.client.post(
            BASE.format(session=session),
            json={
                "message": "compare the first two by price, color, and material",
                "k": 3,
                "sources": ["history", "catalog_search"],
                "constraints": [{"kind": "price_max", "value": "30"}],
            },
        ).json()
        assert first["clarification"]
        recovered = _answer(
            harness,
            session,
            first["clarification"]["clarification_id"],
            message="show facts",
            option="known_facts",
        )
        assert recovered.status_code == 200, recovered.text
        body = recovered.json()
        assert body["terminal"]["status"] == "finished"
        ids = [item["parent_asin"] for item in body["recommendations"]]
        assert "cand-blue" not in ids  # $40 exceeds the hard $30 ceiling
        assert ids == ["cand-red", "cand-black"]
        for item in body["recommendations"]:
            assert item["reason"]
    finally:
        harness.runtime.close()
