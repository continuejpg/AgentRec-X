"""E-01A-3B: the tool-selection decision mode as it is actually served over the chat API.

These are integration tests for the *wiring*, not for the parts it composes (those are covered by
``tests/test_llm_tool_selection.py``, ``tests/test_llm_selected_controller.py`` and
``tests/test_agent_chat_adapter.py``).  Every provider call is answered by the repository's
existing ``ScriptedModelClient`` and the whole stack runs over the synthetic engine, catalogue and
SQLite store of ``tests.demo_fixture`` - no checkpoint, no 300 MB artifact, no network, no
credential.

What is proven here, in the order of the milestone's acceptance criteria:

* ``decision_mode="llm_agent"`` is a closed wire value that a real chat request can select, that a
  session carries, and that is refused - never quietly downgraded - when its provider is missing or
  when the deployment cannot serve it;
* a single-source and a multi-source plan both return **real products from the controller** with
  the provenance the ledger recorded, and the executed run is visible in the response's trace;
* the menu offered to the model is derived from the deployment's live source set (a TIGER worker
  that is not ready is simply not offered), the budget is the controller's own bound, and the
  model is shown no product identity and no score;
* ``direct`` starts no recommendation controller at all and reuses the accepted direct answer;
* an unoffered source, a duplicated source, a provider failure, a failing tool and an empty result
  are all safely visible, and none of them is answered by a default source list;
* the tool-selection turn writes what the shopper states through the accepted M9 service and
  reports the store's own summary, while the *run* stays memory-free and its audit says so; a direct
  turn keeps the accepted graph's memory behaviour;
* the response carries no model prose and no new top-level field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.control import LoopLimits
from recommendation.control.model_client import ModelCallError, ScriptedModelClient
from recommendation.demo.agent_policy import SOURCE_PLANS, SOURCE_PURPOSES
from recommendation.demo.agent_service import DemoAgentService
from recommendation.demo.llm_agent import LLMAgentTurn, offered_sources
from recommendation.demo.schemas import DECISION_MODES, DecisionMode
from recommendation.demo.tool_selection import ToolSelectionError, ToolSelectionPlanner
from tests.demo_fixture import build_harness, close_harness
from tests.agent_reranking_fixture import CANDIDATE_ROWS, INITIAL_ORDER

#: The accepted provider configuration, declared so the *availability* path is exercised.  Nothing
#: connects: every test installs a scripted client, and the placeholder is not a credential.
PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}

#: The chat contract's keys, copied here so this file pins "no new top-level field" independently
#: of the other mode's suite (a shared import would let one file's edit hide the other's drift).
#: ``agent`` is the one additive field E-01A-3B introduced; it is non-null exactly on this mode's
#: turns, which the tests below assert in both directions.
#: A numbered product line of the chat's card-derived prose (E-02C-NARRATION-FIX).
_NARRATION_LINE = re.compile(r"^(\d+)\. (\S+) \(item \d+\)", re.MULTILINE)

#: Phrases only the controller's own narration contains; the chat's prose must not reuse them.
_CONTROLLER_NARRATION_MARKERS = (
    "from the sequential recommender",
    "Original SASRec ranks are shown",
)

CHAT_RESPONSE_KEYS = {
    "api_version",
    "session_id",
    "turn_id",
    "turn",
    "route",
    "message",
    "active_preferences",
    "memory_update",
    "recommendations",
    "audit",
    "trace",
    "agent",
}

#: Fixture catalogue identities, so a fabricated candidate cannot pass as a real one.
CATALOGUE = {str(row[0]) for row in CANDIDATE_ROWS}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that looks fully configured, without any real provider."""
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)


def _client(*answers: Any) -> Any:
    """A scripted provider client that declares the identity a real adapter declares."""
    client = ScriptedModelClient(list(answers), repeat_last=True)
    client.profile = SimpleNamespace(name="deepseek")
    client.model = "deepseek-chat"
    return client


def plan(*sources: str, route: str = "recommend", reason: str | None = "because") -> dict[str, Any]:
    payload: dict[str, Any] = {"route": route, "selected_sources": list(sources)}
    if reason is not None:
        payload["reason"] = reason
    return payload


def _harness(tmp_path: Path, *answers: Any) -> Any:
    """A demo harness, with a scripted client installed when the test supplies a plan."""
    harness = build_harness(tmp_path)
    if answers:
        harness.runtime.use_llm_client(_client(*answers))
    return harness


def _chat(harness: Any, session_id: str, message: str, mode: str | None = None, k: int = 3) -> Any:
    body: dict[str, Any] = {"message": message, "k": k}
    if mode is not None:
        body["decision_mode"] = mode
    return harness.client.post(f"/v1/demo/sessions/{session_id}/chat", json=body)


def _ok(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _modes(harness: Any) -> dict[str, dict[str, Any]]:
    body = harness.client.get("/v1/demo/decision-modes").json()
    return {item["mode"]: item for item in body["modes"]}


def _ids(*, k: int = 3) -> list[str]:
    return list(INITIAL_ORDER[:k])


# --------------------------------------------------------------------------- #
# 1. the mode is a real, closed wire value
# --------------------------------------------------------------------------- #


def test_the_three_modes_are_advertised_with_their_own_availability(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, plan("history"))
    try:
        modes = _modes(harness)
        assert set(modes) == set(DECISION_MODES) == set(DecisionMode.__args__)
        assert modes["deterministic"]["available"] is True
        for mode in ("llm", "llm_agent"):
            assert modes[mode]["available"] is True, mode
            assert modes[mode]["provider"] == "deepseek"
            assert modes[mode]["model"] == "deepseek-chat"
            assert modes[mode]["detail"] is None
        assert modes["llm_agent"]["label"] != modes["llm"]["label"], (
            "two different modes must not be labelled identically in the served list"
        )
    finally:
        close_harness(harness)


def test_a_session_can_select_the_mode_and_keeps_it(
    tmp_path: Path, configured: None
) -> None:
    client = _client(plan("history"))
    harness = _harness(tmp_path)
    harness.runtime.use_llm_client(client)
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find me a widget", mode="llm_agent"))
        assert body["route"] == "recommend"
        assert body["trace"]["decision_mode"] == "llm_agent"
        assert harness.state(session_id)["decision_mode"] == "llm_agent"

        # A later turn that omits the field inherits the session's mode and plans again.
        inherited = _ok(_chat(harness, session_id, "another widget"))
        assert inherited["trace"]["decision_mode"] == "llm_agent"
        assert client.call_count == 2
        assert body["turn"] == 1 and inherited["turn"] == 2
    finally:
        close_harness(harness)


def test_an_unknown_mode_is_still_refused(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path, plan("history"))
    try:
        session_id = harness.create_session()["session_id"]
        response = _chat(harness, session_id, "find gear", mode="adaptive")
        assert response.status_code == 422
        assert harness.state(session_id)["decision_mode"] == "deterministic"
    finally:
        close_harness(harness)


def test_a_missing_credential_refuses_the_mode_before_any_turn(tmp_path: Path) -> None:
    """No credential: the mode is unavailable and asking for it changes nothing."""
    harness = _harness(tmp_path, plan("history"))
    try:
        entry = _modes(harness)["llm_agent"]
        assert entry["available"] is False
        assert entry["reason"] in {"missing_api_key", "not_configured"}
        assert entry["detail"]

        created = harness.client.post(
            "/v1/demo/sessions", json={"profile_id": "demo-user-1", "decision_mode": "llm_agent"}
        )
        assert created.status_code == 503
        assert created.json()["error"] == "llm_unavailable"

        session_id = harness.create_session()["session_id"]
        refused = _chat(harness, session_id, "find gear", mode="llm_agent")
        assert refused.status_code == 503
        assert refused.json()["error"] == "llm_unavailable"
        assert harness.state(session_id)["decision_mode"] == "deterministic"
        assert harness.engine.call_count == 0, "a refused mode must run nothing"
    finally:
        close_harness(harness)


def test_the_mode_does_not_need_the_control_plane_the_plan_mode_needs(
    tmp_path: Path, configured: None
) -> None:
    """The two policy-backed modes execute on different planes, and each reports its own answer."""
    harness = _harness(tmp_path, plan("history"))
    try:
        harness.runtime.control_plane = "loop"
        modes = _modes(harness)
        assert modes["llm"]["available"] is False
        assert modes["llm"]["reason"] == "control_plane_unsupported"
        assert modes["llm_agent"]["available"] is True, (
            "the tool-selection mode runs the runtime's own loop agent service, not the graph"
        )

        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find gear", mode="llm_agent"))
        assert body["route"] == "recommend"
        assert body["trace"]["decision_mode"] == "llm_agent"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 2. the controller really runs: real products, real provenance
# --------------------------------------------------------------------------- #


def test_a_single_source_plan_returns_the_controllers_products(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, plan("history"))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find me a widget", mode="llm_agent"))

        assert harness.engine.call_count == 1, "the accepted Tool really retrieved the candidates"
        cards = body["recommendations"]
        assert [card["parent_asin"] for card in cards] == _ids()
        assert {card["parent_asin"] for card in cards} <= CATALOGUE
        assert [card["sasrec_score"] for card in cards] == [row[2] for row in CANDIDATE_ROWS[:3]]
        assert body["audit"]["candidate_count"] == len(cards)
        assert body["trace"]["candidate_count"] == len(cards)
        # The history source is the accepted capability path: its candidates are the run's
        # grounded universe (the accepted Tool's own verified list) but they have no ledger
        # record, which is why no card carries a source.
        assert all(card["fusion_sources"] == [] for card in cards)

        agent = body["agent"]
        assert agent["outcome"] == "recommended"
        assert agent["planned_sources"] == ["history"], "the plan's own names, in plan order"
        assert agent["controller_executed"] is True
        assert agent["action_counts"] == {"recommend_from_history": 1, "finish": 1}
        assert agent["terminal"]["succeeded"] is True
        assert agent["terminal"]["termination_reason"] == "completed"
        assert agent["grounded"]["grounded_count"] == len(cards)
        assert agent["grounded"]["sources_present"] == ["history"]
    finally:
        close_harness(harness)


def test_a_multi_source_plan_executes_exactly_the_sources_the_model_chose(
    tmp_path: Path, configured: None
) -> None:
    """Both planned sources run, in plan order, and nothing else is consulted.

    Each test builds its own harness, so this is the **first** run of the controller this plan
    compiles to.  That matters and is not incidental: ``DemoSourcePlanPolicy`` keeps its
    "already-served sources" set across the runs of one cached controller (a pre-existing leak,
    reproduced independently in ``docs/V2_E01A_CLOSE_REPORT.md`` §5), so a *second* run of the same
    plan can skip a planned source.  This test therefore asserts first-run behaviour - the
    behaviour a fresh session's first turn gets - and the pre-existing second-run behaviour is
    documented rather than asserted here.
    """
    harness = _harness(tmp_path, plan("catalog_search", "history"))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "I want a RedWidget", mode="llm_agent"))

        agent = body["agent"]
        assert agent["planned_sources"] == ["catalog_search", "history"]
        # Both chosen sources really executed, in the plan's order, and nothing else did.
        assert agent["action_counts"] == {"search_catalog": 1, "recommend_from_history": 1,
                                          "finish": 1}
        assert "select_source" not in agent["action_counts"], (
            "an unselected source must not be consulted"
        )
        assert [step["action"] for step in agent["trajectory"]["steps"]] == [
            "search_catalog", "recommend_from_history", "finish"]
        assert agent["outcome"] == "recommended"
        assert agent["terminal"]["tool_calls"] == 2

        cards = body["recommendations"]
        assert cards, "the fused run must return the candidates the plane retrieved"
        assert {card["parent_asin"] for card in cards} <= CATALOGUE
        assert body["trace"]["proposed_route"] == "recommend"
        assert body["trace"]["route"] == body["route"]
        recorded = [card for card in cards if "catalog_search" in card["fusion_sources"]]
        assert recorded, f"the plane-recorded source must be visible per candidate: {cards}"
        for card in recorded:
            assert card["score_kind"] == "bm25", (
                "a fused source score must be labelled with the source that produced it"
            )
        # Ledger provenance on the response itself: which source proposed what, and how it scored.
        provenance = [record
                      for candidate in agent["grounded"]["candidates"]
                      for record in candidate["provenance"]]
        assert any(record["source"] == "catalog_search" and record["score_kind"] == "bm25"
                   for record in provenance)
        assert agent["grounded"]["grounded_count"] == len(agent["grounded"]["candidates"])
    finally:
        close_harness(harness)


def test_the_products_are_shown_with_their_real_title_and_nothing_invented(
    tmp_path: Path, configured: None
) -> None:
    """A real title is displayed; the attributes the response never carried stay unavailable."""
    harness = _harness(tmp_path, plan("history"))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find me a widget", mode="llm_agent"))

        titles = {row[0]: row[4] for row in CANDIDATE_ROWS}
        for card in body["recommendations"]:
            assert card["metadata_status"] == "found"
            assert card["metadata"]["title"] == titles[card["parent_asin"]]
            assert [card["metadata"]["store"], card["metadata"]["main_category"],
                    card["metadata"]["price_text"]] == [None, None, None], (
                "the run carried a title and nothing else; nothing else may appear"
            )
            assert card["metadata"]["categories"] == [] and card["metadata"]["details"] == []
            assert card["score_kind"] == "sasrec_logit", (
                "the history source's own score kind must survive the projection"
            )
    finally:
        close_harness(harness)


def test_a_candidate_the_response_gives_no_title_for_is_reported_as_missing() -> None:
    """A missing title stays missing, and an unnamed score kind is labelled as the controller's."""
    from recommendation.demo.agent_chat_adapter import build_agent_chat_response
    from recommendation.demo.agent_schemas import AgentRecommendResponse

    response = AgentRecommendResponse.model_validate({
        "api_version": "1.0.0", "route": "recommend", "message": "", "control_plane": "loop",
        "recommendations": [{"rank": 1, "item_id": 7, "parent_asin": "BUNKNOWN", "score": 0.5,
                             "score_kind": None, "title": None, "sources": ()}],
        "grounded": {"grounded_count": 0, "ungrounded_count": 0, "multi_source_count": 0,
                     "sources_present": (), "candidates": ()},
        "eligibility": {"evaluated": False},
        "terminal": {"run_id": "r", "status": "finished", "succeeded": True,
                     "termination_reason": "completed", "termination_detail": None, "steps": 1,
                     "tool_calls": 1, "retries": 0, "limits": {}},
        "trajectory": {"step_count": 0, "actions": (), "refusals": (), "steps": ()},
        "timing": {"total_ms": 1.0, "agent_ms": 1.0, "serialize_ms": 0.0},
    })
    chat = build_agent_chat_response(response, session_id="s", turn_id="s:1", turn=1,
                                     planned_sources=("history",), proposed_route="recommend")
    card = chat.recommendations[0]
    assert card.metadata_status == "missing" and card.metadata is None
    assert card.score_kind == "agent_controller_score", (
        "the response named no kind, so the projection must say the score is the controller's"
    )
    assert chat.agent is not None and chat.agent.outcome == "recommended"


def test_the_new_mode_is_not_a_second_pipeline(
    tmp_path: Path, configured: None
) -> None:
    """Same message, same history, same plan: the chat mode executes the accepted agent run.

    The stateless ``/v1/demo/agent/recommend`` endpoint is the accepted way to run one
    request-declared plan.  The chat mode must reach the *same* result for the same inputs - the
    same products, in the same order, with the same scores and the same per-source provenance -
    because it is the same controller over the same plane, not a parallel implementation of it.
    """
    from recommendation.demo import DemoProfile

    profile = DemoProfile(
        profile_id="demo-user-1",
        display_name="Test shopper 1",
        trusted_user_history=tuple(INITIAL_ORDER),
        source_user_int_id=1,
        source_length=len(INITIAL_ORDER),
    )
    harness = build_harness(tmp_path, profiles={"demo-user-1": profile})
    harness.runtime.use_llm_client(_client(plan("history", "catalog_search")))
    try:
        body = {
            "message": "I want a RedWidget",
            "history": list(INITIAL_ORDER),
            "k": 3,
            "sources": ["history", "catalog_search"],
        }

        # The catalogue-search source a service shares across its controllers answers its very
        # first query of the process differently from later ones - a pre-existing property,
        # reproducible through this endpoint alone, so nothing to do with the chat mode.  The
        # reference run is therefore taken after one warm-up request on each path.
        assert harness.client.post("/v1/demo/agent/recommend", json=body).status_code == 200
        accepted = harness.client.post("/v1/demo/agent/recommend", json=body)
        assert accepted.status_code == 200, accepted.text
        reference = accepted.json()

        session_id = harness.create_session()["session_id"]
        _chat(harness, session_id, "I want a RedWidget", mode="llm_agent")  # warm-up turn
        chat = _ok(_chat(harness, session_id, "I want a RedWidget", mode="llm_agent"))

        assert [(card["parent_asin"], card["sasrec_score"], card["score_kind"])
                for card in chat["recommendations"]] == [
            (card["parent_asin"], card["score"], card["score_kind"])
            for card in reference["recommendations"]
        ], "the chat mode must present exactly what the accepted agent run presented"
        assert [list(card["fusion_sources"]) for card in chat["recommendations"]] == [
            list(card["sources"]) for card in reference["recommendations"]
        ], "provenance must be the ledger's own, unchanged by the chat projection"
        # E-02C-NARRATION-FIX changed this contract, and this assertion previously pinned the
        # defect: the chat used to reuse the controller's own narration, which is rendered *inside*
        # the run - before the multi-source accumulation, the memory-aware ranking and the k limit -
        # so the prose could disagree with the cards the same response published.  The chat now
        # narrates its own final cards, while the accepted endpoint (whose response *is* the run)
        # keeps the controller's text.  Both facts are asserted, so neither can regress silently.
        assert chat["message"] != reference["message"], (
            "the chat narrates its final cards, the accepted endpoint narrates the run"
        )
        listed = [
            (int(position), parent_asin)
            for position, parent_asin in _NARRATION_LINE.findall(chat["message"])
        ]
        assert listed == [
            (position, card["parent_asin"])
            for position, card in enumerate(chat["recommendations"], start=1)
        ], "the chat's prose lists exactly its own cards, in their final order"
        for marker in _CONTROLLER_NARRATION_MARKERS:
            assert marker not in chat["message"], (
                f"{marker!r} belongs to the controller's intermediate narration"
            )
        assert chat["route"] == reference["route"]
        # The execution record is the controller's own: the same trajectory and the same ledger
        # provenance the accepted endpoint publishes, and a terminal record identical except for
        # the run id, which every run mints fresh.
        assert chat["agent"]["trajectory"] == reference["trajectory"]
        assert chat["agent"]["grounded"] == reference["grounded"]
        assert {k: v for k, v in chat["agent"]["terminal"].items() if k != "run_id"} == {
            k: v for k, v in reference["terminal"].items() if k != "run_id"
        }
        assert chat["agent"]["outcome"] == "recommended"
        assert chat["agent"]["planned_sources"] == ["history", "catalog_search"]
    finally:
        close_harness(harness)


def test_the_menu_and_budget_come_from_the_deployment(
    tmp_path: Path, configured: None
) -> None:
    """What the model is offered is read from this deployment, and it is offered nothing else."""
    client = _client(plan("history"))
    harness = _harness(tmp_path)
    harness.runtime.use_llm_client(client)
    try:
        session_id = harness.create_session()["session_id"]
        _ok(_chat(harness, session_id, "find me a widget", mode="llm_agent"))

        service = harness.runtime.agent_service()
        request = client.last_request
        assert request is not None
        offered = [tool["name"] for tool in request.context_payload["available_tools"]]
        assert offered == list(service.available_sources), (
            "the menu must be the deployment's own advertised set, in its order"
        )
        for absent in ("tiger", "two_tower", "similar_item"):
            assert absent not in offered, f"{absent} is not served by this deployment"
        assert request.context_payload["tool_budget"] == service.tool_budget == 4, (
            "the budget is the controller's own max_tool_calls"
        )
        assert [tool["purpose"] for tool in request.context_payload["available_tools"]], (
            "every offered tool must be described to the model"
        )

        # No product identity and no score may reach the provider.
        sent = json.dumps(request.as_messages())
        for identity in CATALOGUE:
            assert identity not in sent, f"{identity} leaked into the planning prompt"
        for row in CANDIDATE_ROWS:
            assert str(row[2]) not in sent, f"score {row[2]} leaked into the planning prompt"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 3. a direct plan starts no controller
# --------------------------------------------------------------------------- #


def test_a_direct_plan_starts_no_recommendation_controller(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, plan(route="direct", reason=None))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "hello there", mode="llm_agent"))

        assert body["route"] == "direct"
        assert body["recommendations"] == []
        assert body["trace"]["proposed_route"] == "direct"
        assert body["trace"]["decision_mode"] == "llm_agent"
        service = harness.runtime.agent_service()
        # Monotonic since E-01A-STATE-FIX (controllers are composed per request, not cached),
        # so "still zero" is the real claim: a direct plan composed none.
        assert service.controllers_built == 0, "no controller may be built for a direct plan"
        assert harness.engine.call_count == 0, "the recommender must not run"

        agent = body["agent"]
        assert agent["outcome"] == "direct", "the model chose direct; nothing was attempted"
        assert agent["planned_sources"] == [], "a direct plan names no source"
        assert agent["controller_executed"] is False
        assert [agent["terminal"], agent["trajectory"], agent["grounded"], agent["timing"]] == [
            None, None, None, None], "a turn that ran no controller has no controller record"
    finally:
        close_harness(harness)


def test_a_direct_plan_keeps_the_accepted_direct_answer(
    tmp_path: Path, configured: None
) -> None:
    """The direct branch reuses the accepted direct capability instead of inventing an answer."""
    harness = _harness(tmp_path, plan(route="direct", reason=None))
    try:
        deterministic = _ok(_chat(harness, harness.create_session()["session_id"], "hello there"))
        agent = _ok(
            _chat(harness, harness.create_session()["session_id"], "hello there", mode="llm_agent")
        )
        assert deterministic["route"] == agent["route"] == "direct"
        assert agent["message"] == deterministic["message"]
        assert deterministic["trace"]["decision_mode"] == "deterministic"
        assert agent["trace"]["decision_mode"] == "llm_agent"
        assert deterministic["agent"] is None
        assert agent["agent"]["outcome"] == "direct"
    finally:
        close_harness(harness)


def test_a_deterministic_turn_never_composes_the_agent_service(
    tmp_path: Path, configured: None
) -> None:
    """Mode isolation: the new path must not be reachable, or paid for, from the old one."""
    harness = _harness(tmp_path, plan("history"))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find me a widget"))
        assert body["trace"]["decision_mode"] == "deterministic"
        assert body["agent"] is None, (
            "the execution record belongs to the tool-selection mode; the other modes report null"
        )
        assert harness.runtime.build_report()["agent_service_composed"] is False
    finally:
        close_harness(harness)


def test_the_plan_backed_mode_reports_no_agent_execution(
    tmp_path: Path, configured: None
) -> None:
    """The ``llm`` mode is untouched by this milestone, including the new optional field."""
    harness = _harness(tmp_path, {"route": "recommend", "add": [], "remove": []})
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find me a widget", mode="llm"))
        assert body["trace"]["decision_mode"] == "llm"
        assert body["agent"] is None
        assert set(body) == CHAT_RESPONSE_KEYS
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 4. refusals, failures and empty results are safely visible
# --------------------------------------------------------------------------- #


def test_an_unoffered_source_is_refused_and_runs_nothing(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, plan("tiger"))
    try:
        session_id = harness.create_session()["session_id"]
        response = _chat(harness, session_id, "find gear", mode="llm_agent")
        assert response.status_code == 502
        payload = response.json()
        assert payload["error"] == "llm_invalid_plan"
        assert set(payload) == {"error", "detail"}
        assert harness.engine.call_count == 0, "a refused plan must consult no source"
        assert harness.state(session_id)["decision_mode"] == "llm_agent", (
            "a refused turn must not move the session onto another mode's path"
        )
    finally:
        close_harness(harness)


def test_a_duplicate_source_is_refused(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path, plan("history", "history"))
    try:
        session_id = harness.create_session()["session_id"]
        response = _chat(harness, session_id, "find gear", mode="llm_agent")
        assert response.status_code == 502
        assert response.json()["error"] == "llm_invalid_plan"
        assert harness.engine.call_count == 0
    finally:
        close_harness(harness)


class _TimeoutClient:
    """A client whose transport always times out: a provider failure with a distinct code."""

    profile = SimpleNamespace(name="deepseek")
    model = "deepseek-chat"

    def __init__(self) -> None:
        self.requests: list[Any] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, request: Any) -> Any:
        self.requests.append(request)
        raise ModelCallError("the model request timed out", code="timeout")


def test_a_provider_failure_is_reported_as_a_provider_error(
    tmp_path: Path, configured: None
) -> None:
    client = _TimeoutClient()
    harness = _harness(tmp_path)
    harness.runtime.use_llm_client(client)
    try:
        session_id = harness.create_session()["session_id"]
        response = _chat(harness, session_id, "find gear", mode="llm_agent")
        assert response.status_code == 504
        assert response.json()["error"] == "llm_timeout"
        assert client.call_count == 1, "a transport failure is not retried"
        assert harness.engine.call_count == 0
    finally:
        close_harness(harness)


def test_a_failing_tool_yields_no_fabricated_products(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, plan("history"))
    try:
        harness.engine.error = RuntimeError("engine exploded")
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find gear", mode="llm_agent"))

        assert body["recommendations"] == [], "a failed tool must not yield candidates"
        assert body["route"] == "direct", "the honest route of a run with no candidate set"
        assert body["audit"]["candidate_count"] == 0
        # The plan and the outcome stay comparable instead of one being rewritten to match.
        assert body["trace"]["proposed_route"] == "recommend"
        assert body["trace"]["candidate_count"] is None
        # The run's own record says what happened: the guard refused it after the execution
        # failed, and both facts travel - the refusal as the reason, the failure in the detail
        # and in the recorded refusal codes. Nothing here is inferred from the empty list.
        agent = body["agent"]
        assert agent["controller_executed"] is True
        assert agent["outcome"] == "refused"
        assert agent["terminal"]["succeeded"] is False
        assert agent["terminal"]["termination_reason"] == "completion_refused"
        assert "last_execution_failed" in (agent["terminal"]["termination_detail"] or "")
        assert "execution_failed" in agent["trajectory"]["refusals"]
        assert agent["action_counts"]["recommend_from_history"] >= 1, (
            "the trajectory must show how often the failing source was really called"
        )
        assert body["message"], "a failed turn must still answer with words"
    finally:
        close_harness(harness)


def test_an_empty_result_is_reported_as_an_empty_recommendation_not_a_direct_turn(
    tmp_path: Path, configured: None
) -> None:
    """Scenario E: zero candidates is a refused run, and it must not look like a direct answer."""
    harness = _harness(tmp_path, plan("catalog_search"))
    try:
        # A direct turn of the same mode, for the comparison the outcome field exists to make.
        direct_session = harness.create_session()["session_id"]
        harness.runtime.use_llm_client(_client(plan(route="direct", reason=None)))
        direct = _ok(_chat(harness, direct_session, "hello there", mode="llm_agent"))
        assert direct["agent"]["outcome"] == "direct"

        harness.runtime.use_llm_client(_client(plan("catalog_search")))
        session_id = harness.create_session()["session_id"]
        empty = _ok(
            _chat(harness, session_id, "zzzqqq nothing here matches", mode="llm_agent")
        )

        assert empty["recommendations"] == [], "no products may be produced from nothing"
        assert empty["route"] == "direct", "the run produced no candidate set"
        assert empty["trace"]["proposed_route"] == "recommend", (
            "the plan did ask for a recommendation; the executed route is the separate fact"
        )
        assert empty["trace"]["candidate_count"] is None

        agent = empty["agent"]
        assert agent["outcome"] == "refused"
        assert agent["controller_executed"] is True
        assert agent["terminal"]["termination_reason"] == "completion_refused"
        assert agent["grounded"]["grounded_count"] == 0
        assert agent["grounded"]["sources_present"] == []

        # The two empties are distinguishable in the response itself, and neither is silent.
        assert direct["agent"]["outcome"] != empty["agent"]["outcome"]
        assert direct["agent"]["controller_executed"] is False
        assert empty["message"] and empty["message"] != direct["message"]
        assert direct["message"] and direct["message"] != empty["message"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 5b. the memory boundary the new mode must keep (scenario F)
# --------------------------------------------------------------------------- #


def test_a_stored_preference_is_not_folded_into_this_turns_query(
    tmp_path: Path, configured: None
) -> None:
    """Scenario F: memory holds `trail running shoes`, the request says something else.

    The stored preference may be *read* (the panel is a safe projection), but it must not be
    appended to this turn's catalogue query - which the ledger's own ``source_query`` makes
    checkable from the response rather than from a log.
    """
    harness = _harness(tmp_path, plan("catalog_search"))
    try:
        session_id = harness.create_session()["session_id"]
        # Put the preference in this session's memory through the accepted deterministic path.
        stored = _ok(_chat(harness, session_id, "I prefer trail running shoes."))
        assert stored["memory_update"]["changed"] is True
        assert harness.active_values(session_id), "the preference really is stored"

        body = _ok(_chat(harness, session_id, "RedWidget basketball", mode="llm_agent"))

        records = [record
                   for candidate in body["agent"]["grounded"]["candidates"]
                   for record in candidate["provenance"]
                   if record["source"] == "catalog_search"]
        assert records, "the ledger must record what the source was actually asked for"
        for record in records:
            query = (record["source_query"] or "").lower()
            assert "trail running" not in query, (
                f"a stored preference reached this turn's query: {record['source_query']!r}"
            )
            assert "redwidget" in query or "basketball" in query, (
                f"the query must be this turn's own request: {record['source_query']!r}"
            )

        # The panel still shows what memory holds (a read-only projection), and this path wrote
        # nothing of its own.
        assert body["memory_update"]["changed"] is False
        assert body["active_preferences"], "the stored preference stays visible in the panel"
        assert harness.active_values(session_id), "and it is still stored afterwards"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 5. preference memory: what the turn states is written and reported; the run stays memory-free
# --------------------------------------------------------------------------- #


def test_a_recommendation_turn_writes_what_the_user_states_and_reports_it(
    tmp_path: Path, configured: None
) -> None:
    """E-01B-1 superseded the earlier boundary here, and this is the new contract.

    Before E-01B-1 a tool-selection *recommendation* turn wrote nothing, so it reported an empty
    memory update.  The session's preference loop now runs on this path too: what the shopper states
    this turn is persisted once through the accepted M9 service, keyed by the session's own
    namespace and this turn's server-owned id, and the response reports the store's own summary.
    What has *not* changed is that the run itself is memory-free - nothing ranks with a preference
    yet, and the audit says so.
    """
    harness = _harness(tmp_path, plan("history"))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(
            _chat(harness, session_id, "I prefer lightweight products.", mode="llm_agent")
        )

        assert body["memory_update"]["changed"] is True
        assert [item["value"] for item in body["memory_update"]["added"]] == ["lightweight"]
        assert body["trace"]["memory_changes"]["changed"] is True
        assert [item["value"] for item in body["active_preferences"]] == ["lightweight"]
        assert body["audit"]["ranked_with_preference_count"] == 0, (
            "the run loaded no preference snapshot, and the audit says so"
        )
        user_key = harness.manager.get(session_id).user_key
        stored = harness.store.get_entries(user_key, active_only=False)
        assert [entry.value for entry in stored if entry.is_active] == ["lightweight"], (
            "the reported update must be the store's own state, not a claim"
        )
    finally:
        close_harness(harness)


def test_a_direct_plan_keeps_the_accepted_memory_behaviour(
    tmp_path: Path, configured: None
) -> None:
    """The direct branch reuses the accepted graph turn, so its memory write is real and reported.

    The same message is sent twice - once with no decision policy, once with a direct plan - and
    the two turns must persist, report and panel the same thing.  A direct plan is the model saying
    "this turn needs no product search", not a reason to stop remembering stated preferences.
    """
    harness = _harness(tmp_path, plan(route="direct", reason=None))
    try:
        message = "I prefer lightweight products."
        deterministic = _ok(_chat(harness, harness.create_session()["session_id"], message))
        agent = _ok(
            _chat(harness, harness.create_session()["session_id"], message, mode="llm_agent")
        )

        assert agent["route"] == deterministic["route"]
        assert agent["message"] == deterministic["message"]
        assert agent["memory_update"] == deterministic["memory_update"]
        assert agent["memory_update"]["changed"] is True
        assert [item["value"] for item in agent["memory_update"]["added"]] == ["lightweight"]
        assert agent["active_preferences"] == deterministic["active_preferences"]
        assert agent["trace"]["memory_changes"] == deterministic["trace"]["memory_changes"]
        # The model's proposal and the accepted graph's own decision stay distinguishable.
        assert agent["trace"]["proposed_route"] == "direct"
        assert agent["trace"]["decision_mode"] == "llm_agent"
        assert deterministic["trace"]["decision_mode"] == "deterministic"
        assert harness.runtime.agent_service().controllers_built == 0, (
            "the direct branch must still start no recommendation controller"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 6. the response and the trace carry facts, never model prose
# --------------------------------------------------------------------------- #


def test_the_response_adds_no_field_and_no_model_prose(
    tmp_path: Path, configured: None
) -> None:
    marker = "RAW_MODEL_MARKER"
    harness = _harness(tmp_path, plan("history", reason=marker))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "find me a widget", mode="llm_agent"))

        assert set(body) == CHAT_RESPONSE_KEYS, "the new mode must not add a top-level field"
        assert marker not in json.dumps(body), (
            "the plan's reason describes a tool choice; it is not product text or answer text"
        )
        trace = body["trace"]
        assert trace["decision_mode"] == "llm_agent"
        assert trace["provider"] == "deepseek" and trace["model"] == "deepseek-chat"
        assert trace["proposed_route"] == "recommend"
        assert trace["preference_actions"] == [], "a tool plan proposes no preference mutation"
        assert trace["source"] is None and trace["evidence"] is None
        assert trace["grounding"] is None, (
            "execution facts live in the agent block, not in the decision trace"
        )
        agent = body["agent"]
        assert set(agent) == {
            "outcome", "planned_sources", "controller_executed", "action_counts", "terminal",
            "trajectory", "grounded", "timing",
        }, "the execution record is a fixed, closed projection"
        assert marker not in json.dumps(agent), "no model prose in the execution record either"
        for banned in ("prompt", "reasoning", "thought", "context_payload", "traceback"):
            assert banned not in json.dumps(body).lower(), banned
    finally:
        close_harness(harness)


def test_the_browser_labels_a_score_by_its_kind_rather_than_calling_it_sasrec() -> None:
    """§6C: a fused or agent-path score must not be printed as a SASRec logit."""
    js = (Path(__file__).resolve().parents[1] / "recommendation" / "web" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "card.score_kind" in js, "the card view must read the score's declared kind"
    assert "Ranking score (\" + scoreKind" in js
    assert '"Raw SASRec ranking score "' in js, (
        "the SASRec wording stays for the scores that really are SASRec logits"
    )


def test_the_cards_are_not_reordered_or_rescored_by_the_adapter(
    tmp_path: Path, configured: None
) -> None:
    """Presentation follows the controller: one card per controller candidate, in its order."""
    harness = _harness(tmp_path, plan("catalog_search"))
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "I want a RedWidget", mode="llm_agent"))
        cards = body["recommendations"]
        assert [card["original_rank"] for card in cards] == list(range(1, len(cards) + 1))
        assert all(card["reranked_rank"] is None for card in cards), (
            "no M10B reranking ran on this path, so no card may claim a reranked position"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 7. the pieces the wiring rests on, pinned directly
# --------------------------------------------------------------------------- #


def test_every_source_has_a_purpose_and_the_menu_preserves_order() -> None:
    assert set(SOURCE_PURPOSES) == set(SOURCE_PLANS), (
        "a source the plan vocabulary can execute must be describable, and vice versa"
    )
    assert [option.name for option in offered_sources(("two_tower", "history"))] == [
        "two_tower",
        "history",
    ], "the offered menu must preserve the deployment's own order"
    assert [option.name for option in offered_sources(("similar_item", "history"))] == ["history"], (
        "a name the accepted plan vocabulary cannot execute must not be offered as a tool"
    )
    assert offered_sources(()) == ()


def test_the_planner_budget_is_the_controllers_bound() -> None:
    """A plan larger than the loop's tool budget is refused before anything runs."""
    from tests.control_fixture import build_full_capability

    _capability, engine, parts = build_full_capability()
    service = DemoAgentService(
        tool=parts["tool"],
        enricher=parts["enricher"],
        metadata=parts["enricher"].metadata,
        matcher=parts["matcher"],
        reranker=parts["reranker"],
        engine=engine,
        limits=LoopLimits(max_tool_calls=1),
    )
    assert service.tool_budget == 1
    assert service.available_sources == ("history", "catalog_search")

    planner = ToolSelectionPlanner(_client(plan("history", "catalog_search")))
    turn = LLMAgentTurn(
        planner, service, k=3, user_message="find gear", history=["cand-red", "cand-blue"]
    )
    with pytest.raises(ToolSelectionError) as excinfo:
        turn.plan()
    assert excinfo.value.code == "over_budget"


def test_the_production_provider_path_composes_the_same_planner(
    tmp_path: Path, configured: None
) -> None:
    """Without an injected client the mode builds the real provider adapter, over one client.

    Nothing is called: constructing the adapter reads ``AGENTRECX_LLM_*`` and nothing else, so this
    pins the production composition path - the same one a deployment uses - without a provider call.
    """
    harness = build_harness(tmp_path)  # deliberately no use_llm_client()
    try:
        planner = harness.runtime.tool_planner()
        assert isinstance(planner, ToolSelectionPlanner)
        assert planner.client is harness.runtime.llm_client()
        assert harness.runtime.llm_planner().client is planner.client, (
            "both policy-backed modes must share one client, not build one each"
        )
        assert planner.calls == 0, "composing a planner must not call the provider"
    finally:
        close_harness(harness)


def test_the_request_the_turn_executes_is_memory_free() -> None:
    """The memory decision is visible in the request the turn builds, not only in a comment."""
    from tests.control_fixture import build_full_capability

    _capability, engine, parts = build_full_capability()
    service = DemoAgentService(
        tool=parts["tool"],
        enricher=parts["enricher"],
        metadata=parts["enricher"].metadata,
        matcher=parts["matcher"],
        reranker=parts["reranker"],
        engine=engine,
    )
    turn = LLMAgentTurn(
        ToolSelectionPlanner(_client(plan("history"))),
        service,
        k=3,
        user_message="find gear",
        history=["cand-red"],
    )
    request = turn.request()
    assert request.user_key is None
    assert tuple(request.history) == ("cand-red",)
    assert request.k == 3
