"""E-01B-2: memory-driven recommendation, accepted end to end over the real Chat API.

The claim this file has to establish is **not** that memory appears in a prompt: it is that a
preference the shopper states through Chat (1) is written by the real M9 service, (2) is read by the
next turn's real ``ToolSelectionPlanner`` call, (3) changes which trusted source the plan consults,
and (4) therefore changes what the real controller executes and which products come back - all
visible in one ``ChatResponse``.

How the four levels are made observable
---------------------------------------
The fixture gives the two candidate sources **disjoint** products, so a plan change cannot be
mistaken for a scoring difference:

* the history source returns ``hist-*`` identities;
* the lexical catalogue source can only match ``cat-*`` identities (a request containing
  ``BasketWidget``), and the grounding map knows both sets so either plan is groundable.

The model provider is the repository's ``ScriptedModelClient`` seam, driven by a controlled client
that routes on the disclosed memory context (``context.active_preferences`` present -> consult the
catalogue; absent -> use history).  That rule is **this test's** rule, not a claim about a hosted
model: it exists to prove the memory channel reaches the decision and propagates to execution.  What
a real provider would choose with the same context is a separate question and is not asserted here.

Everything else is production code: the Chat route, the session registry and its ``user_key``
binding, ``DemoRuntime.llm_agent_turn``, the real M9 ``PreferenceMemoryService`` over a real SQLite
store, ``DemoAgentService``, ``LoopController``, ``ActionValidator``, ``CandidatePlane``,
``GroundingVerifier`` and ``CandidateLedger``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recommendation.api.app import create_app
from recommendation.catalog import MetadataIndex
from recommendation.catalog.metadata import normalize_product_record
from recommendation.control.model_client import ModelResponse
from recommendation.demo import build_demo_runtime
from recommendation.memory import SQLitePreferenceStore
from tests.demo_fixture import (
    DemoEngine,
    DemoHarness,
    close_harness,
    demo_profiles,
    make_settings,
)

#: The history source's products.  Deliberately NOT matchable by the lexical query below.
HISTORY_ROWS: tuple[tuple[Any, ...], ...] = (
    ("hist-alpha", 901, 0.91, "grey", "TrailRunner Alpha", "Acme"),
    ("hist-beta", 902, 0.82, "green", "TrailRunner Beta", "Acme"),
    ("hist-gamma", 903, 0.73, "black", "TrailRunner Gamma", None),
)

#: The catalogue-only products: only a lexical search for ``BasketWidget`` can reach them.
CATALOG_ROWS: tuple[tuple[Any, ...], ...] = (
    ("cat-delta", 911, 0.0, "orange", "BasketWidget Delta", "SportCo"),
    ("cat-epsilon", 912, 0.0, "white", "BasketWidget Epsilon", "SportCo"),
)

#: The one request used before and after every preference change.
#:
#: Phrased as an imperative on purpose.  The accepted rule-based extractor reads first-person
#: desire phrases ("I want/like/need ...") as *preferences*, so a request like "I want a
#: BasketWidget" would itself store one and the baseline would already carry memory.  "Recommend a
#: BasketWidget" states no preference, which is what makes the before/after comparison controlled.
REQUEST = "Recommend a BasketWidget"

PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}

PREFER_BLACK = "I prefer black."
REPLACE_WITH_BLUE = "Actually, I prefer blue instead."
RETRACT_COLOUR = "I don't care about color anymore."


class _DisjointEngine(DemoEngine):
    """The history source's rows, with grounding identities covering both sources.

    ``DemoEngine`` already answers the trusted identity protocol over its own rows; a catalogue that
    contains products the history source never returns needs the mapping to span both, or the plane
    would correctly treat the lexical candidates as ungroundable.
    """

    def __init__(self, history_rows: Any, catalogue_rows: Any) -> None:
        super().__init__(history_rows)
        rows = tuple(history_rows) + tuple(catalogue_rows)
        self._by_asin = {str(row[0]): int(row[1]) for row in rows}
        self._by_id = {int(row[1]): str(row[0]) for row in rows}


def _catalogue_index() -> MetadataIndex:
    """Both sources' products, so either plan's candidates can be grounded."""
    return MetadataIndex.from_records([
        normalize_product_record({"parent_asin": row[0], "title": row[4],
                                  **({"store": row[5]} if row[5] else {})})
        for row in tuple(HISTORY_ROWS) + tuple(CATALOG_ROWS)
    ])


class _MemoryAwareClient:
    """A controlled client that routes on the memory context the planner discloses.

    A single, explicit rule - the catalogue source is consulted when the session holds a stored
    preference, history otherwise - so the *wiring* is what is under test.  It records the real
    ``ModelRequest`` objects it was handed, which is how a test inspects the disclosed context.
    """

    profile = type("P", (), {"name": "deepseek"})()
    model = "deepseek-chat"

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.calls = 0

    @property
    def last_request(self) -> Any:
        return self.requests[-1] if self.requests else None

    def complete(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        self.calls += 1
        context = (request.context_payload or {}).get("context") or {}
        sources = ["catalog_search"] if context.get("active_preferences") else ["history"]
        return ModelResponse(
            text=json.dumps({"route": "recommend", "selected_sources": sources,
                             "reason": "routed on the disclosed memory context"}),
            model_id="scripted",
        )


class _ScriptedClient:
    """A client that answers with a fixed script (used for the statement turns)."""

    profile = type("P", (), {"name": "deepseek"})()
    model = "deepseek-chat"

    def __init__(self, *answers: Any) -> None:
        self._answers = list(answers)
        self.requests: list[Any] = []

    @property
    def last_request(self) -> Any:
        return self.requests[-1] if self.requests else None

    def complete(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        answer = self._answers.pop(0) if self._answers else {"route": "direct",
                                                             "selected_sources": []}
        return ModelResponse(text=json.dumps(answer), model_id="scripted")


DIRECT = {"route": "direct", "selected_sources": [], "reason": "a statement, no products"}


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)


def _harness(tmp_path: Path, client: Any) -> DemoHarness:
    """The full demo app over the disjoint-source fixture."""
    from fastapi.testclient import TestClient

    profiles = demo_profiles()
    engine = _DisjointEngine(HISTORY_ROWS, CATALOG_ROWS)
    store = SQLitePreferenceStore(tmp_path / "acceptance_memory.sqlite3")
    runtime = build_demo_runtime(
        engine=engine, metadata=_catalogue_index(), store=store, profiles=profiles,
        max_sessions=8,
    )
    runtime.use_llm_client(client)
    app = create_app(make_settings(tmp_path), engine=engine, load_on_startup=False,
                     demo=runtime, enable_demo=True)
    test_client = TestClient(app)
    test_client.__enter__()
    return DemoHarness(app=app, runtime=runtime, engine=engine, client=test_client,
                       store=store, profiles=profiles)


def _install(harness: DemoHarness, client: Any) -> Any:
    """Install the client the next turn will use, and return it so its request can be inspected."""
    harness.runtime.use_llm_client(client)
    return client


def _response(harness: DemoHarness, session: str, message: str, k: int = 3) -> Any:
    """One llm_agent turn over the real route, returning the raw response."""
    return harness.client.post(
        f"/v1/demo/sessions/{session}/chat",
        json={"message": message, "k": k, "decision_mode": "llm_agent"},
    )


def _chat(harness: DemoHarness, session: str, message: str, k: int = 3) -> dict[str, Any]:
    response = _response(harness, session, message, k)
    assert response.status_code == 200, response.text
    return response.json()


def _cards(body: dict[str, Any]) -> list[str]:
    return [card["parent_asin"] for card in body["recommendations"]]


def _planned(body: dict[str, Any]) -> list[str]:
    return list(body["agent"]["planned_sources"])


def _counts(body: dict[str, Any]) -> dict[str, int]:
    return dict(body["agent"]["action_counts"])


def _context(client: Any) -> Any:
    request = client.last_request
    assert request is not None, "the planner must have been asked exactly once"
    return (request.context_payload or {}).get("context")


def _user_request(client: Any) -> str:
    return str(client.last_request.context_payload["user_request"])


def _field_names(value: Any) -> set[str]:
    """Every key name anywhere in a nested payload - used to prove a field is absent."""
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            names.add(str(key))
            names |= _field_names(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            names |= _field_names(item)
    return names


def _provenance(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every ledger provenance record of a run; empty for a turn that ran no controller."""
    grounded = body["agent"].get("grounded") or {}
    return [record
            for candidate in (grounded.get("candidates") or [])
            for record in candidate["provenance"]]


# --------------------------------------------------------------------------- #
# A. the acceptance flow: ADD -> recommend -> supersede -> recommend -> REMOVE -> recommend
# --------------------------------------------------------------------------- #


def test_memory_changes_the_plan_the_execution_and_the_products(
    tmp_path: Path, configured: None
) -> None:
    """One controlled multi-turn scenario, with all three evidence levels asserted.

    Level 1: the planner is handed the ACTIVE preference as context (and nothing before it exists).
    Level 2: the plan it validates differs between the two states - history vs catalogue.
    Level 3: the controller executes the plan and the presented products differ, with the ledger's
    own provenance, and both the supersede and the removal paths are exercised afterwards.
    """
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]

        # --- baseline: same request, no stored preference ------------------------------ #
        routing = _install(harness, _MemoryAwareClient())
        before = _chat(harness, session, REQUEST)
        assert _context(routing) is None, "nothing is stored yet, so nothing is disclosed"
        assert _user_request(routing) == REQUEST
        assert _planned(before) == ["history"]
        assert _counts(before) == {"recommend_from_history": 1, "finish": 1}
        baseline_cards = _cards(before)
        assert baseline_cards == [row[0] for row in HISTORY_ROWS]
        assert before["agent"]["grounded"]["sources_present"] == ["history"]
        assert before["memory_update"]["changed"] is False

        # --- the shopper states a preference through Chat ------------------------------ #
        statement = _install(harness, _ScriptedClient(DIRECT))
        stored = _chat(harness, session, PREFER_BLACK)
        assert stored["memory_update"]["changed"] is True
        assert [item["value"] for item in stored["memory_update"]["added"]] == ["black"]
        assert harness.active_values(session) == {("color", "prefer", "black")}
        user_key = harness.manager.get(session).user_key
        assert len(harness.store.get_entries(user_key, active_only=False)) == 1, (
            "the statement must be written exactly once"
        )

        # --- the same request again: memory now reaches the decision ------------------- #
        after_routing = _install(harness, _MemoryAwareClient())
        after = _chat(harness, session, REQUEST)
        assert _context(after_routing) == {
            "active_preferences": [{"kind": "color", "polarity": "prefer", "value": "black"}],
            "active_preference_count": 1,
        }, "level 1: the planner must be handed the stored preference"
        assert _user_request(after_routing) == REQUEST, "the current request is unchanged"
        assert _planned(after) == ["catalog_search"], "level 2: the plan follows the context"
        assert _counts(after) == {"search_catalog": 1, "finish": 1}, "level 3: and it executed"
        after_cards = _cards(after)
        assert after_cards == [row[0] for row in CATALOG_ROWS]
        assert set(after_cards) & set(baseline_cards) == set(), (
            "the fixture makes the two plans' products disjoint, so the difference is unmistakable"
        )
        provenance = _provenance(after)
        assert {record["source"] for record in provenance} == {"catalog_search"}
        assert {record["score_kind"] for record in provenance} == {"bm25"}
        assert after["agent"]["grounded"]["sources_present"] == ["catalog_search"]
        assert after["memory_update"]["changed"] is False, "this turn stated no new preference"
        assert [item["value"] for item in after["active_preferences"]] == ["black"]

        # --- supersede: a correction changes memory, and only the new value is disclosed - #
        _install(harness, _ScriptedClient(DIRECT))
        replaced = _chat(harness, session, REPLACE_WITH_BLUE)
        assert [item["value"] for item in replaced["memory_update"]["added"]] == ["blue"]
        assert [item["value"] for item in replaced["memory_update"]["superseded"]] == ["black"]
        assert harness.active_values(session) == {("color", "prefer", "blue")}

        superseded_routing = _install(harness, _MemoryAwareClient())
        after_supersede = _chat(harness, session, REQUEST)
        assert _context(superseded_routing)["active_preferences"] == [
            {"kind": "color", "polarity": "prefer", "value": "blue"}]
        assert "black" not in json.dumps(_context(superseded_routing))
        assert _planned(after_supersede) == ["catalog_search"]
        assert _counts(after_supersede) == {"search_catalog": 1, "finish": 1}

        # --- removal: the plan returns to the no-preference state ---------------------- #
        _install(harness, _ScriptedClient(DIRECT))
        removed = _chat(harness, session, RETRACT_COLOUR)
        assert removed["memory_update"]["removed"], removed["memory_update"]
        assert harness.active_values(session) == set()

        restored_routing = _install(harness, _MemoryAwareClient())
        restored = _chat(harness, session, REQUEST)
        assert _context(restored_routing) is None, "a withdrawn preference is not disclosed"
        assert _planned(restored) == ["history"], "the decision is back to the baseline"
        assert _counts(restored) == _counts(before)
        assert _cards(restored) == baseline_cards, (
            "removal restores the pre-preference behaviour exactly"
        )
        assert restored["active_preferences"] == []
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# B. correction and removal keep the disclosure honest
# --------------------------------------------------------------------------- #


def test_only_the_current_active_value_is_disclosed_after_a_correction(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        _install(harness, _ScriptedClient(DIRECT))
        _chat(harness, session, PREFER_BLACK)
        _chat(harness, session, REPLACE_WITH_BLUE)

        routing = _install(harness, _MemoryAwareClient())
        body = _chat(harness, session, REQUEST)
        context = _context(routing)
        assert context["active_preference_count"] == 1
        assert [entry["value"] for entry in context["active_preferences"]] == ["blue"]
        assert body["memory_update"]["changed"] is False
        # The superseded entry still exists in M9 (accepted semantics: provenance is retained) but
        # it is not ACTIVE, so it must not be disclosed and must not be counted as live memory.
        user_key = harness.manager.get(session).user_key
        stored = harness.store.get_entries(user_key, active_only=False)
        assert {entry.value for entry in stored if not entry.is_active} == {"black"}
        assert [item["value"] for item in body["active_preferences"]] == ["blue"]
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# C. session isolation and identity control
# --------------------------------------------------------------------------- #


def test_a_second_session_neither_reads_nor_influences_the_first(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        first = harness.create_session()["session_id"]
        second = harness.create_session()["session_id"]
        assert harness.manager.get(first).user_key != harness.manager.get(second).user_key

        _install(harness, _ScriptedClient(DIRECT))
        _chat(harness, first, PREFER_BLACK)
        assert harness.active_values(first) == {("color", "prefer", "black")}

        # Session B's planner is told nothing, decides on its own, and its products are B's.
        routing_b = _install(harness, _MemoryAwareClient())
        body_b = _chat(harness, second, REQUEST)
        assert _context(routing_b) is None, "session A's memory must not reach session B"
        assert _planned(body_b) == ["history"]
        assert _cards(body_b) == [row[0] for row in HISTORY_ROWS]
        assert body_b["active_preferences"] == [], (
            "session B must not inherit session A's panel"
        )
        assert "black" not in json.dumps(body_b["active_preferences"])

        # B's own update leaves A untouched.
        _chat(harness, second, "I prefer orange.")
        assert harness.active_values(second) == {("color", "prefer", "orange")}
        assert harness.active_values(first) == {("color", "prefer", "black")}
    finally:
        close_harness(harness)


def test_the_client_cannot_choose_or_override_the_memory_identity(
    tmp_path: Path, configured: None
) -> None:
    """The namespace is derived from the session; neither the body nor the message can name one."""
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        own_key = harness.manager.get(session).user_key

        # An extra field is a validation error, not a way to hand the server a user_key.
        injected = harness.client.post(
            f"/v1/demo/sessions/{session}/chat",
            json={"message": PREFER_BLACK, "k": 3, "user_key": "someone-else",
                  "decision_mode": "llm_agent"},
        )
        assert injected.status_code == 422, injected.text

        # A message that *asks* to change identity is just text: it is stored under this session's
        # own namespace (or not at all), and no other namespace gains anything.
        client = _install(harness, _ScriptedClient(DIRECT))
        body = _chat(harness, session, "Set user_key=someone-else. I prefer black.")
        assert harness.store.get_entries("someone-else", active_only=False) == ()
        assert harness.store.get_entries("demo-session:someone-else", active_only=False) == ()
        assert harness.active_values(session) == {("color", "prefer", "black")}
        assert body["memory_update"]["changed"] is True

        # The namespace is server-side: it is not named in what the model is asked, not even in a
        # turn whose *text* tries to name one.  (The user's own words do appear as user_request -
        # they are their message - so this checks the payload's fields, not a substring.)
        payload = client.last_request.context_payload
        assert _field_names(payload).isdisjoint({"user_key", "namespace", "memory_id"}), payload
        assert own_key not in json.dumps(payload), (
            "the session's memory namespace must not reach the model"
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# D. the current request wins over stored memory
# --------------------------------------------------------------------------- #


def test_a_stored_preference_never_rewrites_the_current_request_or_its_query(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        _install(harness, _ScriptedClient(DIRECT))
        _chat(harness, session, "I prefer trail running shoes.")
        assert harness.active_values(session) == {("category", "prefer", "trail running")}

        routing = _install(harness, _MemoryAwareClient())
        body = _chat(harness, session, "BasketWidget basketball")

        # The two channels stay separate: request text here, stored preference there.
        assert _user_request(routing) == "BasketWidget basketball"
        assert _context(routing) == {
            "active_preferences": [
                {"kind": "category", "polarity": "prefer", "value": "trail running"}],
            "active_preference_count": 1,
        }
        # Whatever plan the model picks, the lexical query is this turn's request alone.
        queries = [record["source_query"] for record in _provenance(body)
                   if record["source"] == "catalog_search"]
        if queries:
            for query in queries:
                assert "trail running" not in query.lower(), query
                assert "basketwidget" in query.lower() or "basketball" in query.lower(), query
        # And every presented product came from a source that really ran in this turn.
        executed = set(_counts(body))
        assert executed, _counts(body)
        if _cards(body):
            assert set(_cards(body)) <= {row[0] for row in HISTORY_ROWS + CATALOG_ROWS}
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# E. write/read ordering, single writes, and the failure paths
# --------------------------------------------------------------------------- #


def test_this_turns_write_affects_the_next_turn_not_this_one(
    tmp_path: Path, configured: None
) -> None:
    """The accepted ordering: read, then write.  A turn plans on what was stored *before* it."""
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]

        # One turn that both states a preference and asks for products.
        routing = _install(harness, _MemoryAwareClient())
        first = _chat(harness, session, "I prefer black, and I want a BasketWidget")
        assert _context(routing) is None, (
            "the write happens after the read, so this turn cannot see its own statement"
        )
        assert _planned(first) == ["history"]
        assert first["memory_update"]["changed"] is True, "the statement is still persisted"
        assert harness.active_values(session) == {("color", "prefer", "black")}

        # The very next turn sees it.
        next_routing = _install(harness, _MemoryAwareClient())
        second = _chat(harness, session, "I want a BasketWidget")
        assert _context(next_routing)["active_preferences"] == [
            {"kind": "color", "polarity": "prefer", "value": "black"}]
        assert _planned(second) == ["catalog_search"]
        assert _cards(second) != _cards(first)
    finally:
        close_harness(harness)


def test_repeating_a_preference_creates_no_second_active_entry(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        _install(harness, _ScriptedClient(DIRECT, DIRECT))
        first = _chat(harness, session, PREFER_BLACK)
        second = _chat(harness, session, "Recommend a BasketWidget, and I prefer black.")
        assert first["memory_update"]["changed"] is True
        assert second["memory_update"]["changed"] is False
        assert second["memory_update"]["skipped_duplicates"] == 1
        user_key = harness.manager.get(session).user_key
        assert len(harness.store.get_entries(user_key, active_only=False)) == 1
        assert harness.active_values(session) == {("color", "prefer", "black")}
    finally:
        close_harness(harness)


def test_a_failed_memory_write_is_never_reported_as_a_successful_update(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        user_key = harness.manager.get(session).user_key
        real = harness.runtime.memory_service

        class _FailingWrites:
            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

            def process_turn(self, **_: Any) -> Any:
                raise RuntimeError("the preference store is unavailable")

        harness.runtime.memory_service = _FailingWrites()
        _install(harness, _ScriptedClient(DIRECT))
        response = _response(harness, session, PREFER_BLACK)
        assert response.status_code >= 500, response.text
        assert "changed" not in json.dumps(response.json())
        assert harness.store.get_entries(user_key, active_only=False) == ()
    finally:
        close_harness(harness)


def test_an_invalid_plan_writes_nothing_and_fabricates_nothing(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        # An unoffered source, and a self-contradictory direct-with-sources plan.
        for answer in ({"route": "recommend", "selected_sources": ["tiger"], "reason": "x"},
                       {"route": "direct", "selected_sources": ["history"], "reason": "x"}):
            _install(harness, _ScriptedClient(answer))
            response = _response(harness, session, PREFER_BLACK)
            assert response.status_code == 502, response.text
            assert response.json()["error"] == "llm_invalid_plan"
            assert harness.store.get_entries(
                harness.manager.get(session).user_key, active_only=False) == ()
        assert harness.active_values(session) == set()
    finally:
        close_harness(harness)


def test_an_empty_retrieval_is_reported_as_a_refused_run_not_as_a_recommendation(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _MemoryAwareClient())
    try:
        session = harness.create_session()["session_id"]
        # A lexical plan whose query matches nothing in the catalogue: the controller must not
        # invent a product, and the response must say the run could not be completed.
        _install(harness, _ScriptedClient({"route": "recommend",
                                          "selected_sources": ["catalog_search"],
                                          "reason": "lexical"}))
        response = _response(harness, session, "zzzqqq nothing here")
        body = response.json()
        assert response.status_code == 200
        assert body["recommendations"] == []
        assert body["route"] == "direct"
        assert body["agent"]["outcome"] == "refused"
        assert body["agent"]["terminal"]["termination_reason"] == "completion_refused"
        assert body["agent"]["grounded"]["grounded_count"] == 0
        assert body["message"], "the turn still answers in words"
    finally:
        close_harness(harness)
