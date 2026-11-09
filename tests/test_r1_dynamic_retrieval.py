"""R1 memory-guided retrieval integration through the shopping HTTP path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_reranking_fixture import CANDIDATE_ROWS, build_metadata
from tests.demo_fixture import PROFILE_A, build_harness, close_harness

ENDPOINT = "/v1/demo/sessions/{session}/agent/shopping"
EXTRA_RED = ("cand-extra-red", 105, 0.55, "red", "Red Extra Widget Pack", None)


def _rows():
    return (*CANDIDATE_ROWS, EXTRA_RED)


def _add_extra_catalogue(harness) -> None:
    """Add the extra row to the live catalogue and rebuild the lexical index once."""
    from recommendation.control.catalog_search import CatalogSearchSource

    harness.runtime.metadata.records[EXTRA_RED[0]] = build_metadata(EXTRA_RED)
    harness.runtime.agent_service()._catalog_search = CatalogSearchSource(
        harness.runtime.metadata
    )


def _first_question(harness, session_id: str, message: str, k: int):
    response = harness.client.post(
        ENDPOINT.format(session=session_id),
        json={
            "message": message,
            "k": k,
            "sources": ["history", "catalog_search"],
            "preference_discovery": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preference"] is not None, body
    return body


def _answer(harness, session_id: str, body, option_id: str, k: int):
    response = harness.client.post(
        ENDPOINT.format(session=session_id),
        json={
            "message": "choose",
            "k": k,
            "sources": ["history", "catalog_search"],
            "preference_discovery": True,
            "clarification_id": body["clarification"]["clarification_id"],
            "clarification_option": option_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_answered_categorical_preference_can_recall_an_out_of_pool_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_M7_GROUNDED_ACTIONS", "on")
    monkeypatch.setenv("AGENTRECX_M7_BUDGET", "1")
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    monkeypatch.setenv("AGENTRECX_STATE_MODE", "shadow")
    monkeypatch.setenv("AGENTRECX_STATE_DB", str(tmp_path / "state.sqlite3"))
    from recommendation.demo import agent_service as agent_service_module
    from recommendation.demo.retrieval_plan import RetrievalPlan

    def fake_plan(*args, **kwargs):
        return RetrievalPlan(
            "RETRIEVE",
            "changed_categorical_preference_not_covered",
            query_terms=("red",),
            sources=("history", "catalog_search"),
            changed_attributes=("color",),
        )

    monkeypatch.setattr(agent_service_module, "plan_dynamic_retrieval", fake_plan)
    harness = build_harness(tmp_path, rows=_rows())
    try:
        _add_extra_catalogue(harness)
        session = harness.create_session(PROFILE_A)["session_id"]
        question = _first_question(
            harness, session, "recommend gear in different colors", k=3
        )
        assert question["preference"]["attribute"] == "color"
        before = tuple(question["preference"]["products"])
        option = next(
            item
            for item in question["preference"]["options"]
            if item["action"] == "value" and item["value"] == "red"
        )
        answer = _answer(harness, session, question, option["option_id"], k=3)
        after = tuple(item["parent_asin"] for item in answer["recommendations"])
        service = harness.runtime.agent_service()
        assert "cand-extra-red" in after
        diagnostic = service.r1_diagnostics(session)
        assert diagnostic is not None
        assert diagnostic["action"] == "RETRIEVE"
        assert "cand-extra-red" in diagnostic["added"]
        assert "catalog_search" in diagnostic["sources"]
        assert diagnostic["after"] == list(after)
    finally:
        close_harness(harness)


def test_numeric_preference_with_sufficient_candidates_is_rerank_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_M7_GROUNDED_ACTIONS", "on")
    monkeypatch.setenv("AGENTRECX_M7_BUDGET", "1")
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    monkeypatch.setenv("AGENTRECX_STATE_MODE", "shadow")
    monkeypatch.setenv("AGENTRECX_STATE_DB", str(tmp_path / "state.sqlite3"))
    harness = build_harness(tmp_path, rows=_rows())
    try:
        _add_extra_catalogue(harness)
        session = harness.create_session(PROFILE_A)["session_id"]
        question = _first_question(
            harness, session, "gear under a tight budget", k=2
        )
        assert question["preference"]["attribute"] == "price"
        before = tuple(question["preference"]["products"])
        option = next(
            item
            for item in question["preference"]["options"]
            if item["action"] == "lower"
        )
        answer = _answer(harness, session, question, option["option_id"], k=2)
        after = tuple(item["parent_asin"] for item in answer["recommendations"])
        assert set(after) == set(before)
    finally:
        close_harness(harness)


def test_free_text_goal_correction_retrieves_again_but_unchanged_turn_reranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_R1_DYNAMIC_RETRIEVAL", "on")
    monkeypatch.setenv("AGENTRECX_STATE_MODE", "shadow")
    monkeypatch.setenv("AGENTRECX_STATE_DB", str(tmp_path / "state.sqlite3"))
    harness = build_harness(tmp_path, rows=_rows())
    try:
        _add_extra_catalogue(harness)
        session = harness.create_session(PROFILE_A)["session_id"]
        first = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "water filter",
                "k": 2,
                "sources": ["history", "catalog_search"],
                "preference_discovery": False,
            },
        )
        assert first.status_code == 200, first.text
        first_ids = tuple(
            item["parent_asin"] for item in first.json()["recommendations"]
        )
        assert first_ids

        unchanged = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "water filter",
                "k": 2,
                "sources": ["history", "catalog_search"],
                "preference_discovery": False,
            },
        )
        assert unchanged.status_code == 200, unchanged.text
        service = harness.runtime.agent_service()
        unchanged_diag = service.r1_diagnostics(session)
        assert unchanged_diag["action"] == "RERANK"
        assert unchanged_diag["before"] == unchanged_diag["after"]

        corrected = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "actually I want a replacement mouthpiece for a water filter",
                "k": 2,
                "sources": ["history", "catalog_search"],
                "preference_discovery": False,
            },
        )
        assert corrected.status_code == 200, corrected.text
        corrected_diag = service.r1_diagnostics(session)
        assert corrected_diag["action"] == "RETRIEVE"
        assert corrected_diag["mode"] == "normal_turn"
        assert any("replacement" in term for term in corrected_diag["query_terms"])
        assert corrected_diag["sources"] == ["catalog_search"]

        withdrawn = harness.client.post(
            ENDPOINT.format(session=session),
            json={
                "message": "forget the replacement mouthpiece requirement",
                "k": 2,
                "sources": ["history", "catalog_search"],
                "preference_discovery": False,
            },
        )
        assert withdrawn.status_code == 200, withdrawn.text
        withdrawn_diag = service.r1_diagnostics(session)
        assert withdrawn_diag["action"] == "RETRIEVE"
        # The correction's mouthpiece query is no longer the active anchor.
        assert not any(
            "mouthpiece" in term for term in withdrawn_diag["query_terms"]
        )
    finally:
        close_harness(harness)
