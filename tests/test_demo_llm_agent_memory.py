"""E-01B-1: the chat session's M9 preferences, wired into the ``llm_agent`` decision.

What this file proves, end to end and with the real components (Chat route → DemoRuntime → the real
``ToolSelectionPlanner`` building a real ``ModelRequest`` → ``DemoAgentService`` → the bounded loop,
plus the real M9 ``PreferenceMemoryService`` over a real store):

* a preference stated in one turn is written by M9 and reaches the **next** turn's planner as
  structured context (never as query text, never as the user's own words);
* a superseded or removed preference stops reaching it;
* two sessions never share a namespace;
* the current request keeps priority: stored preferences never enter this turn's search query;
* the response's ``memory_update`` / ``active_preferences`` report what the store really did - not
  what the model said it did, and never a success for a write that did not happen;
* the old chat modes and the tool-execution behaviour of E-01A do not regress.

Only the model provider is a fake (the repository's own ``ScriptedModelClient``); every other
component is the production one, and no store or response is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.agent import AgentInput
from recommendation.control.model_client import ScriptedModelClient
from recommendation.demo.llm_agent import MAX_PLANNER_PREFERENCES, planner_preference_context
from recommendation.memory.query_adapter import project_active_preferences
from recommendation.memory.schemas import PreferenceMemoryEntry, PreferenceMemorySnapshot
from tests.demo_fixture import build_harness, close_harness

#: The accepted provider configuration, declared so the availability path is exercised.  Nothing
#: connects: every turn installs a scripted client, and this is not a credential.
PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}

#: Phrasings the accepted rule-based extractor already handles (see tests/test_demo_multiturn.py):
#: ADD, a REPLACE that supersedes, and a retraction.
PREFER_BLACK = "I prefer black."
REPLACE_WITH_BLUE = "Actually, I prefer blue instead."
RETRACT_COLOUR = "I don't care about color anymore."


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)


def _client(*answers: Any) -> Any:
    client = ScriptedModelClient(list(answers), repeat_last=True)
    client.profile = SimpleNamespace(name="deepseek")
    client.model = "deepseek-chat"
    return client


def plan(*sources: str, route: str = "recommend", reason: str = "because") -> dict[str, Any]:
    return {"route": route, "selected_sources": list(sources), "reason": reason}


DIRECT = plan(route="direct", reason="no products needed")
HISTORY = plan("history")
CATALOG = plan("catalog_search")
BOTH = plan("history", "catalog_search")


def _turn(harness: Any, session_id: str, message: str, *answers: Any, k: int = 3) -> tuple[Any, Any]:
    """One llm_agent turn with its own scripted client; returns ``(body, client)``."""
    client = _client(*answers)
    harness.runtime.use_llm_client(client)
    response = harness.client.post(
        f"/v1/demo/sessions/{session_id}/chat",
        json={"message": message, "k": k, "decision_mode": "llm_agent"},
    )
    return response, client


def _ok(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _sent_context(client: Any) -> Any:
    """The ``context`` block of the real ``ModelRequest`` the planner built, or ``None``."""
    request = client.last_request
    assert request is not None, "the planner must have made exactly one provider call"
    return request.context_payload.get("context")


def _sent_request(client: Any) -> str:
    request = client.last_request
    assert request is not None
    return str(request.context_payload["user_request"])


def _preferences(values: set[tuple[str, str, str]]) -> PreferenceMemorySnapshot:
    """A hand-built snapshot, used only for the projection unit tests at the end of this file."""
    return PreferenceMemorySnapshot(
        user_key="u",
        entries=tuple(
            PreferenceMemoryEntry(
                memory_id=f"m{index}", user_key="u", kind=kind, value=value, polarity=polarity,
                source_text="the user said so", source_turn_id="turn-1", logical_seq=index,
                created_at=1.0,
            )
            for index, (kind, polarity, value) in enumerate(sorted(values), start=1)
        ),
    )


# --------------------------------------------------------------------------- #
# A. same session: a preference stated in one turn reaches the next turn's planner
# --------------------------------------------------------------------------- #


def test_a_stated_preference_is_written_and_reaches_the_next_turns_planner(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]

        # Turn 1: the shopper states a preference; the model routes the statement direct, so the
        # accepted graph writes it through M9 exactly as the deterministic mode does.
        statement, _client1 = _turn(harness, session, PREFER_BLACK, DIRECT)
        first = _ok(statement)
        assert first["route"] == "direct"
        assert first["memory_update"]["changed"] is True
        assert [(item["kind"], item["polarity"], item["value"])
                for item in first["memory_update"]["added"]] == [("color", "prefer", "black")]
        assert first["agent"]["outcome"] == "direct"
        assert first["agent"]["controller_executed"] is False
        # The store really holds it - this is the accepted M9 state, not a response field.
        assert harness.active_values(session) == {("color", "prefer", "black")}
        user_key = harness.manager.get(session).user_key
        stored = harness.store.get_entries(user_key, active_only=False)
        assert [entry.value for entry in stored if entry.is_active] == ["black"]
        assert len(stored) == 1, (
            "the statement must be stored exactly once: the graph owns the write on a direct turn "
            "and the runtime must not submit it a second time"
        )

        # Turn 2: a recommendation request.  The real planner must be handed the preference as
        # structured context, while the request text stays exactly what the shopper typed.
        reply, client2 = _turn(harness, session, "Recommend something for me.", HISTORY)
        second = _ok(reply)
        assert second["route"] == "recommend"
        assert _sent_request(client2) == "Recommend something for me."
        assert _sent_context(client2) == {
            "active_preferences": [
                {"kind": "color", "polarity": "prefer", "value": "black"}],
            "active_preference_count": 1,
        }
        # The preference was background, not the request: it never entered the request text.
        assert "black" not in _sent_request(client2)
        # And the panel still reports the store's real state after a turn that stored nothing.
        assert second["memory_update"]["changed"] is False
        assert [(item["kind"], item["polarity"], item["value"])
                for item in second["active_preferences"]] == [("color", "prefer", "black")]
    finally:
        close_harness(harness)


def test_a_preference_stated_on_a_recommendation_turn_is_written_once_and_reported(
    tmp_path: Path, configured: None
) -> None:
    """A recommendation turn writes what the shopper stated *on that turn*, and says so."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        reply, _client1 = _turn(harness, session, "I prefer black, recommend something.", HISTORY)
        body = _ok(reply)

        assert body["route"] == "recommend"
        assert body["memory_update"]["changed"] is True
        assert [item["value"] for item in body["memory_update"]["added"]] == ["black"]
        assert harness.active_values(session) == {("color", "prefer", "black")}
        # The run itself is still memory-free: this turn's retrieval was not ranked with it.
        assert body["audit"]["ranked_with_preference_count"] == 0

        # A later turn that repeats the same statement changes nothing and says so.
        again, _client2 = _turn(harness, session, "I prefer black, recommend something.", BOTH)
        repeated = _ok(again)
        assert repeated["memory_update"]["changed"] is False
        assert repeated["memory_update"]["skipped_duplicates"] == 1
        assert harness.active_values(session) == {("color", "prefer", "black")}
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# B. supersede and removal stop reaching the planner
# --------------------------------------------------------------------------- #


def test_a_superseded_preference_stops_reaching_the_planner(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _ok(_turn(harness, session, PREFER_BLACK, DIRECT)[0])

        replaced, _client = _turn(harness, session, REPLACE_WITH_BLUE, DIRECT)
        body = _ok(replaced)
        assert [item["value"] for item in body["memory_update"]["added"]] == ["blue"]
        assert [item["value"] for item in body["memory_update"]["superseded"]] == ["black"]
        assert harness.active_values(session) == {("color", "prefer", "blue")}

        _reply, client = _turn(harness, session, "Recommend something.", HISTORY)
        assert _sent_context(client) == {
            "active_preferences": [{"kind": "color", "polarity": "prefer", "value": "blue"}],
            "active_preference_count": 1,
        }
        assert "black" not in json.dumps(_sent_context(client))
    finally:
        close_harness(harness)


def test_a_removed_preference_stops_reaching_the_planner(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _ok(_turn(harness, session, PREFER_BLACK, DIRECT)[0])
        removed, _client = _turn(harness, session, RETRACT_COLOUR, DIRECT)
        body = _ok(removed)
        assert body["memory_update"]["removed"], body["memory_update"]
        assert harness.active_values(session) == set()

        reply, client = _turn(harness, session, "Recommend something.", HISTORY)
        final = _ok(reply)
        assert _sent_context(client) is None, "a withdrawn preference must not reach the planner"
        assert final["active_preferences"] == []
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# C. two sessions never share a namespace
# --------------------------------------------------------------------------- #


def test_two_sessions_do_not_share_preferences(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        first = harness.create_session()["session_id"]
        second = harness.create_session()["session_id"]
        assert harness.manager.get(first).user_key != harness.manager.get(second).user_key

        _ok(_turn(harness, first, PREFER_BLACK, DIRECT)[0])
        assert harness.active_values(first) == {("color", "prefer", "black")}

        # Session B has stored nothing, so its planner is told nothing and its panel is empty.
        reply, client = _turn(harness, second, "Recommend something.", HISTORY)
        body = _ok(reply)
        assert _sent_context(client) is None
        assert body["active_preferences"] == []
        assert harness.active_values(second) == set()

        # And a statement in session B never shows up in session A.
        _ok(_turn(harness, second, "I prefer blue.", DIRECT)[0])
        assert harness.active_values(second) == {("color", "prefer", "blue")}
        assert harness.active_values(first) == {("color", "prefer", "black")}
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# D. the current request keeps priority
# --------------------------------------------------------------------------- #


def test_stored_preferences_never_enter_this_turns_search_query(
    tmp_path: Path, configured: None
) -> None:
    """Both channels at once: memory reaches the *planner*, and only the request reaches the tool."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        _ok(_turn(harness, session, "I prefer trail running shoes.", DIRECT)[0])
        assert harness.active_values(session) == {("category", "prefer", "trail running")}

        reply, client = _turn(harness, session, "RedWidget basketball", CATALOG)
        body = _ok(reply)

        # The planner was told about the stored preference ...
        assert _sent_context(client) == {
            "active_preferences": [
                {"kind": "category", "polarity": "prefer", "value": "trail running"}],
            "active_preference_count": 1,
        }
        assert _sent_request(client) == "RedWidget basketball"

        # ... and the search that actually ran used the current request alone.
        queries = [record["source_query"]
                   for candidate in body["agent"]["grounded"]["candidates"]
                   for record in candidate["provenance"]
                   if record["source"] == "catalog_search"]
        assert queries, "the ledger must record what the source was asked for"
        for query in queries:
            assert "trail running" not in query.lower(), (
                f"a stored preference was appended to this turn's query: {query!r}"
            )
            assert "redwidget" in query.lower() or "basketball" in query.lower(), query
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# E. the old chat modes are unchanged
# --------------------------------------------------------------------------- #


def test_the_deterministic_chat_mode_keeps_its_memory_behaviour(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        added = _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": PREFER_BLACK, "k": 3}))
        assert added["memory_update"]["changed"] is True
        assert harness.active_values(session) == {("color", "prefer", "black")}

        # The deterministic path ranks *with* the stored preference - the difference this milestone
        # deliberately does not change for the llm_agent path.
        ranked = _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": "Recommend again.", "k": 3}))
        assert ranked["audit"]["ranked_with_preference_count"] == 1

        replaced = _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": REPLACE_WITH_BLUE, "k": 3}))
        assert [item["value"] for item in replaced["memory_update"]["superseded"]] == ["black"]
        assert harness.active_values(session) == {("color", "prefer", "blue")}
    finally:
        close_harness(harness)


def test_the_plan_backed_chat_mode_keeps_its_memory_behaviour(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        client = _client({"route": "recommend", "add": [{"value": "durable", "kind": "feature"}],
                          "remove": []})
        harness.runtime.use_llm_client(client)
        body = _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": "I want something durable.", "k": 3, "decision_mode": "llm"}))
        assert [item["value"] for item in body["memory_update"]["added"]] == ["durable"]
        assert harness.active_values(session) == {("feature", "prefer", "durable")}
        assert body["agent"] is None, "the plan-backed mode is not an agent execution turn"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# F. tool execution does not regress
# --------------------------------------------------------------------------- #


def test_tool_execution_and_controller_isolation_do_not_regress(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        # A direct turn composes no controller at all.
        direct_session = harness.create_session()["session_id"]
        direct, _client1 = _turn(harness, direct_session, "hello there", DIRECT)
        assert _ok(direct)["agent"]["outcome"] == "direct"
        assert harness.runtime.agent_service().controllers_built == 0

        # A multi-source plan consults both sources on *every* turn, and a different session does
        # not inherit the previous turn's controller state.
        for _ in range(2):
            session = harness.create_session()["session_id"]
            body = _ok(_turn(harness, session, "I want a RedWidget", BOTH)[0])
            assert body["agent"]["action_counts"] == {
                "recommend_from_history": 1, "search_catalog": 1, "finish": 1}
            assert body["agent"]["outcome"] == "recommended"

        # A refused plan writes nothing: the plan is obtained before any preference is persisted.
        refused_session = harness.create_session()["session_id"]
        refused, _client2 = _turn(harness, refused_session, "I prefer black, recommend gear.",
                                  plan("tiger"))
        assert refused.status_code == 502
        assert refused.json()["error"] == "llm_invalid_plan"
        assert harness.active_values(refused_session) == set(), (
            "a refused plan must not leave a preference behind"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# G. no fake success, no pollution
# --------------------------------------------------------------------------- #


def test_a_turn_that_states_nothing_reports_no_change(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        reply, _client = _turn(harness, session, "Recommend something for me.", HISTORY)
        body = _ok(reply)
        assert body["memory_update"] == {
            "changed": False, "added": [], "superseded": [], "removed": [],
            "skipped_duplicates": 0, "already_processed": False, "removal_directives": 0,
        }
        assert harness.active_values(session) == set()
    finally:
        close_harness(harness)


def test_a_blank_memory_identity_writes_nothing_and_pollutes_nothing(
    tmp_path: Path, configured: None
) -> None:
    """The namespace comes from the session; a turn without one must not touch memory at all."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        user_key = harness.manager.get(session).user_key
        before = harness.store.get_entries(user_key, active_only=False)
        assert harness.runtime._process_turn_memory(
            user_key="   ", user_message=PREFER_BLACK, turn_id="t1") is None
        assert harness.runtime._active_preference_snapshot("   ") is None

        harness.runtime.use_llm_client(_client(HISTORY))
        response, evidence = harness.runtime.llm_agent_turn(
            3,
            user_key="",
            agent_input=AgentInput(
                user_message="I prefer black, recommend something.",
                trusted_user_history=harness.manager.get(session).trusted_user_history,
                turn_id=f"{session}:000001",
            ),
            session_id=session,
            turn_id=f"{session}:000001",
            turn_number=1,
        )
        assert response.recommendations or response.route == "recommend"
        assert response.memory_update.changed is False
        assert response.active_preferences == ()
        assert evidence["memory_write_attempted"] == "none"
        assert harness.store.get_entries(user_key, active_only=False) == before
    finally:
        close_harness(harness)


def test_a_failing_store_write_is_reported_as_a_failure_not_a_success(
    tmp_path: Path, configured: None
) -> None:
    """A write that does not happen may never be presented as a memory update."""
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        user_key = harness.manager.get(session).user_key
        real = harness.runtime.memory_service

        class _FailingWrites:
            """The real service, with only its write failing - the store stays real."""

            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

            def process_turn(self, **_: Any) -> Any:
                raise RuntimeError("the preference store is unavailable")

        harness.runtime.memory_service = _FailingWrites()
        response, _client = _turn(harness, session, "I prefer black, recommend something.", HISTORY)
        assert response.status_code >= 500, response.text
        body = response.json()
        assert body.get("error"), body
        assert "changed" not in json.dumps(body), "no memory claim may survive a failed write"
        assert harness.store.get_entries(user_key, active_only=False) == ()
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# H. the projection itself: narrow, capped, and never the snapshot
# --------------------------------------------------------------------------- #


def test_the_planner_context_carries_only_the_decision_relevant_values() -> None:
    snapshot = _preferences({("color", "prefer", "black"), ("category", "avoid", "boots")})
    projection = project_active_preferences(snapshot)
    context = planner_preference_context(projection)

    assert context["active_preference_count"] == 2
    assert {entry["value"] for entry in context["active_preferences"]} == {"black", "boots"}
    for entry in context["active_preferences"]:
        assert set(entry) == {"kind", "polarity", "value"}, entry
    rendered = json.dumps(context)
    for leaked in ("source_text", "the user said so", "memory_id", "user_key", "turn-1", "m1"):
        assert leaked not in rendered, leaked
    assert not hasattr(projection, "user_key")


def test_the_planner_context_is_capped_and_counts_what_it_left_out() -> None:
    # ``kind`` is a closed vocabulary in M9, so the *values* are what vary here.
    values = {("feature", "prefer", f"value{i}") for i in range(MAX_PLANNER_PREFERENCES + 4)}
    context = planner_preference_context(project_active_preferences(_preferences(values)))
    assert len(context["active_preferences"]) == MAX_PLANNER_PREFERENCES
    assert context["active_preference_count"] == MAX_PLANNER_PREFERENCES + 4, (
        "a capped list must never be mistaken for the whole picture"
    )


def test_no_memory_means_no_context_at_all() -> None:
    assert planner_preference_context(None) == {}
    assert planner_preference_context(project_active_preferences(None)) == {}
    assert planner_preference_context(project_active_preferences(
        PreferenceMemorySnapshot(user_key="u", entries=()))) == {}


# --------------------------------------------------------------------------- #
# I. the context is part of the decision, not decoration
# --------------------------------------------------------------------------- #


class _PreferenceAwareClient:
    """A client that routes on the disclosed context: a stand-in for a model that uses memory.

    It is deliberately still a *fake*: the point of this test is that the projection reaches the
    decision input and can change which tool is chosen, which is what the wiring has to deliver.
    Whether a real model chooses *better* with it is E-01B-2's question, not this one's.
    """

    profile = SimpleNamespace(name="deepseek")
    model = "deepseek-chat"

    def __init__(self) -> None:
        self.requests: list[Any] = []

    @property
    def last_request(self) -> Any:
        return self.requests[-1] if self.requests else None

    def complete(self, request: Any) -> Any:
        from recommendation.control.model_client import ModelResponse

        self.requests.append(request)
        context = (request.context_payload or {}).get("context") or {}
        sources = ["catalog_search"] if context.get("active_preferences") else ["history"]
        return ModelResponse(
            text=json.dumps({"route": "recommend", "selected_sources": sources,
                             "reason": "routed on the disclosed context"}),
            model_id="scripted",
        )


def test_the_disclosed_context_can_change_which_tool_is_chosen(
    tmp_path: Path, configured: None
) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        # A request the fixture catalogue can actually answer, so a chosen source returns something
        # and the trajectory shows one execution rather than a bounded retry.
        request = "Recommend me a RedWidget."

        # Nothing stored: the client has no context to route on and picks the history source.
        client = _PreferenceAwareClient()
        harness.runtime.use_llm_client(client)
        before = _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": request, "k": 3, "decision_mode": "llm_agent"}))
        assert (client.last_request.context_payload or {}).get("context") is None
        assert before["agent"]["planned_sources"] == ["history"]
        assert before["agent"]["action_counts"] == {"recommend_from_history": 1, "finish": 1}

        # Store a preference, then ask again: the same client now routes to the other source, and
        # the executed trajectory follows the plan it validated.
        harness.runtime.use_llm_client(_client(DIRECT))
        _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": "I prefer black.", "k": 3, "decision_mode": "llm_agent"}))

        client2 = _PreferenceAwareClient()
        harness.runtime.use_llm_client(client2)
        after = _ok(harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": request, "k": 3, "decision_mode": "llm_agent"}))
        assert (client2.last_request.context_payload or {})["context"][
            "active_preferences"] == [{"kind": "color", "polarity": "prefer", "value": "black"}]
        assert after["agent"]["planned_sources"] == ["catalog_search"]
        assert after["agent"]["action_counts"] == {"search_catalog": 1, "finish": 1}
        assert "recommend_from_history" not in after["agent"]["action_counts"]
    finally:
        close_harness(harness)
