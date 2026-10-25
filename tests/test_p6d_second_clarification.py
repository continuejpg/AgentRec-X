"""Phase 6D: one bounded second research clarification, then a safe end.

The second-missing scenario is built by adding a partial description to one product in the
existing synthetic catalogue.  This is test-fixture data only; the catalogue in the process is
otherwise unchanged and no production asset is modified.  It lets the first choice
(``compare:description``) be a real, executable option while the second comparison still shows
a missing value, which is exactly the recovery path under test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402

BASE = "/v1/demo/sessions/{session}/agent/shopping"
FIRST = "compare the first two by price, color, material, and description"


def _harness(tmp_path: Path) -> Any:
    harness = build_harness(tmp_path)
    metadata = harness.runtime.metadata
    red = metadata.records["cand-red"]
    blue = metadata.records["cand-blue"]
    metadata.records["cand-red"] = red.model_copy(
        update={"description": ("red description",)}
    )
    metadata.records["cand-blue"] = blue.model_copy(update={"description": ()})
    return harness


def _session(harness: Any) -> str:
    return harness.create_session(PROFILE_A)["session_id"]


def _post(harness: Any, session: str, body: dict[str, Any]) -> Any:
    return harness.client.post(BASE.format(session=session), json=body)


def _first(harness: Any) -> tuple[str, dict[str, Any]]:
    session = _session(harness)
    response = _post(
        harness,
        session,
        {"message": FIRST, "k": 3, "sources": ["history", "catalog_search"]},
    )
    assert response.status_code == 200, response.text
    return session, response.json()


def _answer(
    harness: Any,
    session: str,
    clarification_id: str,
    *,
    message: str,
    option: str | None,
) -> Any:
    body = {
        "message": message,
        "k": 3,
        "sources": ["history", "catalog_search"],
        "clarification_id": clarification_id,
    }
    if option is not None:
        body["clarification_option"] = option
    return _post(harness, session, body)


def _options(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        option["option_id"]: option
        for option in payload["clarification"]["research_options"]
    }


def test_second_missing_attribute_has_one_follow_up_and_no_third_question(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    try:
        session, first = _first(harness)
        first_clarification = first["clarification"]
        assert "material" in first_clarification["missing_attributes"]
        assert "description" in first_clarification["missing_attributes"]
        assert "compare:description" in _options(first)
        assert first["terminal"]["tool_calls"] == 4

        second = _answer(
            harness,
            session,
            first_clarification["clarification_id"],
            message="compare description instead",
            option="compare:description",
        )
        assert second.status_code == 200, second.text
        body = second.json()
        assert body["terminal"]["status"] == "waiting_for_user"
        assert body["terminal"]["tool_calls"] == 4
        assert body["trajectory"]["actions"] == [
            "recommend_from_history",
            "search_catalog",
            "compare_products",
            "inspect_product",
            "ask_clarification",
        ]
        # The second question is about the newly selected attribute, and it no longer
        # offers a compare option: the task-level research budget is exhausted.
        assert "description" in body["clarification"]["missing_attributes"]
        assert "compare:description" not in _options(body)
        assert "compare:price" not in _options(body)
        assert set(_options(body)) == {"known_facts", "stop_research"}
        # All four research actions are distinct observations: material compare + inspect,
        # then description compare + inspect.  No same product/field was inspected twice.
        research_actions = body["research"]["actions"]
        assert [action["tool"] for action in research_actions] == [
            "compare_products",
            "inspect_product",
            "compare_products",
            "inspect_product",
        ]
        inspected = [
            (action["products"][0], tuple(action.get("attributes") or ()))
            for action in research_actions
            if action["tool"] == "inspect_product"
        ]
        assert len(inspected) == len(set(inspected)) == 2
        assert body["recommendations"] == []

        # A third research attempt is unavailable; an explicit compare option is refused and
        # the pending second question remains available for a safe stop.
        third_id = body["clarification"]["clarification_id"]
        refused = _answer(
            harness,
            session,
            third_id,
            message="compare price",
            option="compare:price",
        )
        assert refused.status_code == 422
        stopped = _answer(
            harness,
            session,
            third_id,
            message="stop research",
            option="stop_research",
        )
        assert stopped.status_code == 200, stopped.text
        final = stopped.json()
        assert final["terminal"]["status"] == "finished"
        assert final["terminal"]["tool_calls"] == 2
        assert final["recommendations"]
        assert [item["parent_asin"] for item in final["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]
    finally:
        harness.runtime.close()


def test_second_clarification_keeps_known_facts_and_stop_does_no_research(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    try:
        session, first = _first(harness)
        second = _answer(
            harness,
            session,
            first["clarification"]["clarification_id"],
            message="compare description",
            option="compare:description",
        ).json()
        second_id = second["clarification"]["clarification_id"]
        stopped = _answer(
            harness,
            session,
            second_id,
            message="stop research",
            option="stop_research",
        )
        assert stopped.status_code == 200
        final = stopped.json()
        assert "research stopped" in final["research"]["note"]
        # The known facts survive both rounds.
        assert any(
            action.get("facts") or action.get("rows")
            for action in final["research"]["actions"]
        )
        assert "description" in {
            attribute
            for row in (
                row for action in final["research"]["actions"] for row in action.get("rows") or []
            )
            for attribute in [row.get("attribute")]
        } or any(
            "description" in (action.get("facts") or {})
            for action in final["research"]["actions"]
            if action.get("tool") == "inspect_product"
        )
        assert final["terminal"]["tool_calls"] == 2
    finally:
        harness.runtime.close()


def test_first_alternate_attribute_can_still_finish_without_second_question(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path)
    try:
        session, first = _first(harness)
        response = _answer(
            harness,
            session,
            first["clarification"]["clarification_id"],
            message="compare price",
            option="compare:price",
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terminal"]["status"] == "finished"
        assert body["clarification"] is None
        assert [item["parent_asin"] for item in body["recommendations"]] == [
            "cand-red", "cand-blue", "cand-black"
        ]
        assert body["trajectory"]["actions"] == [
            "recommend_from_history", "search_catalog", "compare_products", "finish"
        ]
        assert body["terminal"]["tool_calls"] == 3
    finally:
        harness.runtime.close()


def test_hard_constraint_and_reasons_survive_two_recovery_rounds(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session = _session(harness)
        first = _post(
            harness,
            session,
            {
                "message": FIRST,
                "k": 3,
                "sources": ["history", "catalog_search"],
                "constraints": [{"kind": "price_max", "value": "30"}],
            },
        ).json()
        assert first["clarification"]
        second = _answer(
            harness,
            session,
            first["clarification"]["clarification_id"],
            message="compare description",
            option="compare:description",
        ).json()
        assert second["clarification"]
        final = _answer(
            harness,
            session,
            second["clarification"]["clarification_id"],
            message="stop",
            option="stop_research",
        ).json()
        assert final["terminal"]["status"] == "finished"
        ids = [item["parent_asin"] for item in final["recommendations"]]
        assert ids == ["cand-red", "cand-black"]
        assert "cand-blue" not in ids
        for item in final["recommendations"]:
            assert item["reason"]
            assert item["parent_asin"] not in ("", None)
        assert final["terminal"]["tool_calls"] == 2
    finally:
        harness.runtime.close()
