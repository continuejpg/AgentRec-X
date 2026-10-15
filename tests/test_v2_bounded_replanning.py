"""E-03B: bounded replanning — one extra source, chosen by the model, inside the same run.

Every test drives the real Chat API through the real chain: ToolSelectionPlanner parsing, the
LoopController, ActionValidator, CandidatePlane, GroundingVerifier, the accumulated-candidate
fusion, memory-aware ranking and product RAG.  Only the model is a fake, and it answers from the
payload it is handed (plan / source choice / explanation claims), so call order is never scripted.

The fixture is local because the scenarios need sources whose products *differ*: the sequential
source returns what it returns, and the lexical source can only reach products in the catalogue
index, so a first source with one product and a second with three is expressible directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recommendation.api.app import create_app
from recommendation.catalog import MetadataIndex
from recommendation.catalog.metadata import normalize_product_record
from recommendation.control import LoopLimits
from recommendation.control.model_client import ModelCallError, ModelResponse
from recommendation.demo import build_demo_runtime
from recommendation.demo.replan_policy import (
    ENV_AGENT_REPLANNING,
    REPLAN_STATUSES,
    parse_replan_answer,
    replanning_enabled,
)
from recommendation.memory import SQLitePreferenceStore
from tests.demo_fixture import (
    DemoEngine,
    DemoHarness,
    close_harness,
    demo_profiles,
    make_settings,
)

#: (parent_asin, item_id, score, colour, title, store).
#: ``k-hist`` is the sequential source's only product; the other three are reachable only through
#: the lexical source, whose titles contain the token the request uses.
HISTORY_ROWS: tuple[tuple[Any, ...], ...] = (
    ("k-hist", 801, 0.90, "grey", "Hist Widget", "Acme"),
)
CATALOG_ROWS: tuple[tuple[Any, ...], ...] = (
    ("k-red", 802, 0.0, "red", "Red Widget", "SportCo"),
    ("k-blue", 803, 0.0, "blue", "Blue Widget", "SportCo"),
    ("k-black", 804, 0.0, "black", "Black Widget", "Acme"),
)
ALL_ROWS = HISTORY_ROWS + CATALOG_ROWS
REQUEST = "Widget"
PLAN_HISTORY = {"route": "recommend", "selected_sources": ["history"], "reason": "history"}
PLAN_CATALOG = {"route": "recommend", "selected_sources": ["catalog_search"], "reason": "lexical"}
DIRECT = {"route": "direct", "selected_sources": [], "reason": "statement"}

PROVIDER_ENV = {
    "AGENTRECX_LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "AGENTRECX_LLM_MODEL": "deepseek-chat",
    "AGENTRECX_LLM_API_KEY": "unit-test-placeholder-not-a-credential",
    "AGENTRECX_LLM_PROFILE": "deepseek",
}


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(ENV_AGENT_REPLANNING, raising=False)


class _Engine(DemoEngine):
    """The sequential source: its own rows, over a mapping that knows the whole catalogue."""

    def __init__(self, rows: Any = HISTORY_ROWS, known: Any = ALL_ROWS) -> None:
        super().__init__(rows)
        self._by_asin = {str(row[0]): int(row[1]) for row in known}
        self._by_id = {int(row[1]): str(row[0]) for row in known}


def _index(rows: Any = ALL_ROWS, *, colour: bool = True) -> MetadataIndex:
    return MetadataIndex.from_records([
        normalize_product_record({
            "parent_asin": row[0],
            "title": row[4],
            **({"details": {"Color": row[3]}} if colour and row[3] else {}),
            **({"store": row[5]} if row[5] else {}),
        })
        for row in rows
    ])


class _ReactiveClient:
    """A fake model that answers each call from its payload, never from a call counter."""

    profile = SimpleNamespace(name="deepseek")
    model = "deepseek-chat"

    def __init__(
        self,
        plan: Any = PLAN_HISTORY,
        *,
        replan_source: str | None = None,
        replan_answer: str | None = None,
        replan_repeat: bool = False,
        explain: bool = False,
    ) -> None:
        self.plan = plan
        self.replan_source = replan_source
        self.replan_answer = replan_answer
        self.replan_repeat = replan_repeat
        self.explain = explain
        self.plan_calls = 0
        self.replan_calls = 0
        self.explanation_calls = 0
        self.replan_payloads: list[dict[str, Any]] = []

    def complete(self, request: Any) -> ModelResponse:
        payload = request.context_payload or {}
        if "products" in payload:
            self.explanation_calls += 1
            return ModelResponse(text=json.dumps({"explanations": []}), model_id="reactive")
        if "eligible_sources" in payload:
            self.replan_calls += 1
            self.replan_payloads.append(payload)
            if self.replan_answer is not None:
                return ModelResponse(text=self.replan_answer, model_id="reactive")
            source = self.replan_source
            if self.replan_repeat:
                source = (payload.get("eligible_sources") or [{}])[0].get("name")
            if source is None:
                return ModelResponse(
                    text=json.dumps({"replan": False, "source": None, "reason": "enough"}),
                    model_id="reactive",
                )
            return ModelResponse(
                text=json.dumps({"replan": True, "source": source, "reason": "another source"}),
                model_id="reactive",
            )
        self.plan_calls += 1
        return ModelResponse(text=json.dumps(self.plan), model_id="reactive")


def _harness(
    tmp_path: Path,
    *,
    replanning: bool = True,
    explanations: bool = False,
    history_rows: Any = HISTORY_ROWS,
    index_rows: Any = ALL_ROWS,
    known: Any = ALL_ROWS,
    colour: bool = True,
) -> DemoHarness:
    from fastapi.testclient import TestClient

    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = _Engine(history_rows, known)
    store = SQLitePreferenceStore(tmp_path / "replan.sqlite3")
    runtime = build_demo_runtime(
        engine=engine, metadata=_index(index_rows, colour=colour), store=store,
        profiles=demo_profiles(), max_sessions=8, explanations=explanations,
        replanning=replanning,
    )
    runtime.agent_service()._memory_ranking_enabled = True
    app = create_app(make_settings(tmp_path), engine=engine, load_on_startup=False, demo=runtime,
                     enable_demo=True)
    client = TestClient(app)
    client.__enter__()
    return DemoHarness(app=app, runtime=runtime, engine=engine, client=client, store=store,
                       profiles=runtime.profiles)


def _turn(
    harness: DemoHarness,
    client: Any,
    message: str = REQUEST,
    *,
    k: int = 4,
    session: str | None = None,
) -> dict[str, Any]:
    """One chat turn; ``session`` continues an existing conversation (so its memory applies)."""
    harness.runtime.use_llm_client(client)
    if session is None:
        session = harness.create_session()["session_id"]
    response = harness.client.post(
        f"/v1/demo/sessions/{session}/chat",
        json={"message": message, "k": k, "decision_mode": "llm_agent"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _state_a_preference(
    harness: DemoHarness, session: str, message: str = "I prefer black."
) -> None:
    body = _turn(harness, _ReactiveClient(DIRECT), message, session=session)
    assert body["memory_update"]["changed"] is True


def _ids(body: dict[str, Any]) -> list[str]:
    return [card["parent_asin"] for card in body["recommendations"]]


def _cards(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {card["parent_asin"]: card for card in body["recommendations"]}


def _replan(body: dict[str, Any]) -> dict[str, Any]:
    return body["audit"]["replan"]


# --------------------------------------------------------------------------- #
# A. enough candidates: no second decision, no extra call
# --------------------------------------------------------------------------- #


def test_enough_candidates_never_ask_the_model_to_replan(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, index_rows=ALL_ROWS)
    try:
        # A plan of both sources: the answer is sufficient, so the gate closes before any call.
        client = _ReactiveClient(
            {"route": "recommend", "selected_sources": ["history", "catalog_search"],
             "reason": "both"},
            replan_source="catalog_search",
        )
        body = _turn(harness, client)
        assert set(_ids(body)) == {"k-hist", "k-red", "k-blue", "k-black"}
        assert client.replan_calls == 0, "no replan decision when the answer is already complete"
        assert client.plan_calls == 1
        assert body["agent"]["terminal"]["tool_calls"] == 2
        decision = _replan(body)
        assert decision["status"] == "not_triggered"
        assert decision["model_called"] is False
        assert decision["reason_code"] == "candidates_sufficient"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# B. short candidates: the model appends a source whose products are new
# --------------------------------------------------------------------------- #


def test_a_short_answer_is_completed_by_one_model_chosen_source(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body = _turn(harness, client, k=4)

        # The first source produced one product; the appended one produced three, and all four are
        # in the one fused answer.  (Their order is the plane's own fusion order, not this test's.)
        assert set(_ids(body)) == {"k-hist", "k-red", "k-blue", "k-black"}, (
            "the first source's candidate survives, and the appended source's products join it"
        )
        assert _ids(body)[0] == "k-hist"
        assert client.replan_calls == 1, "the model was asked exactly once"
        assert body["agent"]["action_counts"] == {
            "recommend_from_history": 1, "search_catalog": 1, "finish": 1}
        # One request, one budget: the extra source is inside the same run.
        terminal = body["agent"]["terminal"]
        assert (terminal["tool_calls"], terminal["steps"]) == (2, 3)
        assert terminal["limits"] == {"max_steps": 6, "max_tool_calls": 4, "max_retries": 1}
        assert terminal["termination_reason"] == "completed"

        decision = _replan(body)
        assert decision["status"] == "appended_candidates"
        assert decision["chosen_source"] == "catalog_search"
        assert decision["model_called"] is True
        assert decision["initial_candidate_count"] == 1, "the initial observation is its own fact"
        assert decision["final_candidate_count"] == 4

        # The appended source is a real ledger sighting.  It also returned the history product
        # (its title carries the request token), so that product legitimately carries the lexical
        # sighting and the lexical score - and the multi-source count stays zero, because the
        # sequential capability's own candidates are not ledger sightings (E-02A-V).
        assert _cards(body)["k-red"]["fusion_sources"] == ["catalog_search"]
        assert _cards(body)["k-hist"]["fusion_sources"] == ["catalog_search"]
        assert _cards(body)["k-hist"]["score_kind"] == "bm25"
        assert body["agent"]["grounded"]["multi_source_count"] == 0
        assert body["agent"]["grounded"]["sources_present"] == ["history", "catalog_search"]
    finally:
        close_harness(harness)


def test_the_request_the_model_receives_is_bounded_and_factual(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        _turn(harness, client, k=4)
        payload = client.replan_payloads[0]
        assert sorted(payload) == [
            "eligible_sources", "instructions", "observation", "stated_preferences",
            "user_request",
        ]
        assert payload["user_request"] == REQUEST
        assert [item["name"] for item in payload["eligible_sources"]] == ["catalog_search"], (
            "only sources the deployment offers and this run has not consulted"
        )
        observation = payload["observation"]
        assert observation["requested_k"] == 4
        assert observation["consulted_sources"] == ["history"]
        assert observation["source_outcomes"] == {"history": "candidates"}
        assert observation["verified_candidates"] == 1
        assert observation["remaining_tool_calls"] == 3 and observation["remaining_steps"] == 5
        # No candidate identity, no title, no score, no memory or store internals.
        rendered = json.dumps(payload)
        for forbidden in ("k-hist", "k-red", "801", "Hist Widget", "source_text", "memory_id",
                          "user_key", "trusted_user_history", "sasrec"):
            assert forbidden not in rendered, forbidden
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# C. zero candidates: recovered *before* the completion guard could refuse
# --------------------------------------------------------------------------- #


def test_zero_candidates_are_recovered_inside_the_run(
    tmp_path: Path, configured: None
) -> None:
    """The key scenario: the accepted guard refuses an empty recommendation, so the decision has
    to be taken before that proposal - and it is, because it is the policy that proposes."""
    harness = _harness(tmp_path)
    try:
        # The lexical source cannot match this request at all; the sequential one can answer it.
        client = _ReactiveClient(PLAN_CATALOG, replan_source="history")
        body = _turn(harness, client, message="NothingMatchesThisRequest")

        assert body["agent"]["action_counts"] == {
            "search_catalog": 1, "recommend_from_history": 1, "finish": 1}, (
            "the empty source ran, then the appended one, then the run finished"
        )
        assert body["agent"]["terminal"]["termination_reason"] == "completed", (
            "the run was completed, not refused: it recovered before the guard saw an empty answer"
        )
        assert body["agent"]["outcome"] == "recommended"
        assert _ids(body) == ["k-hist"], "real products, from the source the model chose"
        assert body["route"] == "recommend"

        decision = _replan(body)
        assert decision["status"] == "appended_candidates"
        assert decision["chosen_source"] == "history"
        assert decision["initial_candidate_count"] == 0
        assert decision["final_candidate_count"] == 1
    finally:
        close_harness(harness)


def test_zero_candidates_without_a_usable_second_source_still_refuse(
    tmp_path: Path, configured: None
) -> None:
    """Both sources empty: the run keeps the accepted refusal, and no third attempt happens."""
    harness = _harness(tmp_path, history_rows=(), index_rows=(), known=ALL_ROWS)
    try:
        client = _ReactiveClient(PLAN_CATALOG, replan_source="history", replan_repeat=True)
        body = _turn(harness, client, message="NothingMatchesThisRequest")

        assert body["recommendations"] == []
        assert body["agent"]["outcome"] == "refused"
        assert body["agent"]["terminal"]["termination_reason"] == "completion_refused"
        assert client.replan_calls == 1, "exactly one replan decision, never a loop"
        assert body["agent"]["action_counts"].get("search_catalog", 0) == 1, (
            "the consulted source is not retried: the run's own record says it was consulted"
        )
        decision = _replan(body)
        assert decision["status"] == "appended_empty"
        assert decision["chosen_source"] == "history"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# D. budget: neither the tool nor the step budget may be exceeded
# --------------------------------------------------------------------------- #


def test_no_tool_budget_means_no_replanning(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        harness.runtime.agent_service()._limits = LoopLimits(
            max_steps=6, max_tool_calls=1, max_retries=0
        )
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body = _turn(harness, client, k=9)

        assert client.replan_calls == 0, "no decision can be taken without a tool call left"
        assert body["agent"]["terminal"]["tool_calls"] == 1
        assert _ids(body) == ["k-hist"]
        decision = _replan(body)
        assert decision["status"] == "gate_blocked"
        assert decision["model_called"] is False
    finally:
        close_harness(harness)


def test_not_enough_steps_to_both_consult_and_finish_blocks_the_replan(
    tmp_path: Path, configured: None
) -> None:
    """One step can terminate a run; it cannot also consult a source and then finish."""
    harness = _harness(tmp_path)
    try:
        harness.runtime.agent_service()._limits = LoopLimits(
            max_steps=2, max_tool_calls=4, max_retries=0
        )
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body = _turn(harness, client, k=9)

        assert client.replan_calls == 0
        assert body["agent"]["terminal"]["steps"] <= 2
        assert body["agent"]["action_counts"] == {"recommend_from_history": 1, "finish": 1}
        assert _replan(body)["status"] == "gate_blocked"
    finally:
        close_harness(harness)


def test_a_three_step_budget_is_just_enough(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        harness.runtime.agent_service()._limits = LoopLimits(
            max_steps=3, max_tool_calls=4, max_retries=0
        )
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body = _turn(harness, client, k=4)
        assert client.replan_calls == 1, "one source + finish needs exactly two steps"
        assert body["agent"]["terminal"]["steps"] == 3
        assert set(_ids(body)) == {"k-hist", "k-red", "k-blue", "k-black"}
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# E. invalid model answers never execute anything
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "answer", "status"), [
    ("invented_source", json.dumps({"replan": True, "source": "tiger"}), "invalid_output"),
    ("consulted_source", json.dumps({"replan": True, "source": "history"}), "invalid_output"),
    ("unauthorised_tool", json.dumps({"replan": True, "source": "catalog_search",
                                      "budget": 99, "tool": "rm -rf"}), None),
    ("illegal_json", "not json at all", "invalid_output"),
    ("declined", json.dumps({"replan": False, "source": None}), "model_declined"),
])
def test_invalid_or_declining_answers_change_nothing(
    tmp_path: Path, configured: None, name: str, answer: str, status: str | None
) -> None:
    harness = _harness(tmp_path)
    try:
        # ``history`` is the only consulted source here, so naming it is not eligible; a request
        # that also carries extra keys may only be honoured for the fields the contract defines.
        client = _ReactiveClient(PLAN_HISTORY, replan_answer=answer)
        body = _turn(harness, client, k=9)

        assert client.replan_calls == 1
        if name == "unauthorised_tool":
            # The extra keys are ignored: the one legal source named is executed, nothing else.
            assert body["agent"]["action_counts"].get("search_catalog") == 1
            assert _replan(body)["status"] == "appended_candidates"
        else:
            assert body["agent"]["action_counts"] == {
                "recommend_from_history": 1, "finish": 1}
            assert _replan(body)["status"] == status
            assert _replan(body)["chosen_source"] is None
        # The answer is untouched either way.
        assert "k-hist" in _ids(body)
        assert body["agent"]["terminal"]["tool_calls"] <= 2
    finally:
        close_harness(harness)


def test_a_model_that_asks_twice_still_gets_one_decision(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        client = _ReactiveClient(PLAN_HISTORY, replan_repeat=True)
        body = _turn(harness, client, k=9)
        assert client.replan_calls == 1, "one decision per request, whatever the model wants"
        assert body["agent"]["action_counts"] == {
            "recommend_from_history": 1, "search_catalog": 1, "finish": 1}
        assert _replan(body)["status"] == "appended_candidates"
    finally:
        close_harness(harness)


def test_a_failing_model_call_leaves_the_run_alone(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        class _FailsOnReplan(_ReactiveClient):
            def complete(self, request: Any) -> ModelResponse:
                payload = request.context_payload or {}
                if "eligible_sources" in payload:
                    self.replan_calls += 1
                    raise ModelCallError("provider timed out")
                return super().complete(request)

        client = _FailsOnReplan(PLAN_HISTORY)
        body = _turn(harness, client, k=9)
        assert client.replan_calls == 1, "no retry after a failed call"
        assert _ids(body) == ["k-hist"]
        decision = _replan(body)
        assert decision["status"] == "model_failed"
        assert decision["model_called"] is True
    finally:
        close_harness(harness)


def test_an_unauthorised_plan_is_refused_before_any_run(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path)
    try:
        client = _ReactiveClient(
            {"route": "recommend", "selected_sources": ["tiger"], "reason": "x"})
        harness.runtime.use_llm_client(client)
        response = harness.client.post(
            f"/v1/demo/sessions/{harness.create_session()['session_id']}/chat",
            json={"message": REQUEST, "k": 4, "decision_mode": "llm_agent"},
        )
        assert response.status_code == 502
        assert response.json()["error"] == "llm_invalid_plan"
        assert client.replan_calls == 0, "replanning never runs for a refused plan"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# F. the appended candidates join the one fusion, ranking and explanation pool
# --------------------------------------------------------------------------- #


def test_the_appended_source_joins_the_same_fusion_and_ranking_pool(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, explanations=True)
    try:
        # A preference for black: it must reorder the *final* pool, which spans both sources.
        session = harness.create_session()["session_id"]
        _state_a_preference(harness, session)

        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body = _turn(harness, client, k=4, session=session)

        assert set(_ids(body)) == {"k-hist", "k-red", "k-blue", "k-black"}
        assert _ids(body)[0] == "k-black", (
            "memory-aware ranking ran on the accumulated pool, not on one source's list"
        )
        assert _replan(body)["status"] == "appended_candidates"
        # Provenance, scores and preference evidence are the run's own, per product.
        black = _cards(body)["k-black"]
        assert (black["score_kind"], black["fusion_sources"]) == ("bm25", ["catalog_search"])
        assert black["match_count"] == 1 and black["violation_count"] == 0
        assert _cards(body)["k-hist"]["fusion_sources"] == ["catalog_search"]
        assert _cards(body)["k-hist"]["match_count"] == 0
        # RAG explains only what is presented, and the prose lists exactly the cards.
        assert body["audit"]["rag"]["products_examined"] == 4
        listing = [
            (int(position), identity)
            for position, identity in __import__("re").findall(
                r"^(\d+)\. (\S+) \(item \d+\)", body["message"], __import__("re").MULTILINE)
        ]
        assert listing == [(index + 1, identity)
                           for index, identity in enumerate(_ids(body))]
    finally:
        close_harness(harness)


def test_the_second_source_runs_through_the_same_validation_and_grounding(
    tmp_path: Path, configured: None
) -> None:
    """An appended source whose products the mapping does not know contributes nothing."""
    harness = _harness(tmp_path, index_rows=ALL_ROWS + (("k-ghost", 805, 0.0, None,
                                                        "Ghost Widget", None),))
    try:
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body = _turn(harness, client, k=4)
        assert client.replan_calls == 1
        assert "k-ghost" not in _ids(body), "the appended source's unknown product is refused"
        assert body["agent"]["grounded"]["ungrounded_count"] >= 1
        assert body["agent"]["action_counts"]["search_catalog"] == 1
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# G. two sessions, and concurrency
# --------------------------------------------------------------------------- #


def test_two_sessions_keep_their_own_memory_and_replan_state(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path)
    try:
        first = harness.create_session()["session_id"]
        second = harness.create_session()["session_id"]
        assert harness.manager.get(first).user_key != harness.manager.get(second).user_key

        _state_a_preference(harness, first)

        client_a = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body_a = _turn(harness, client_a, k=4, session=first)
        assert _ids(body_a)[0] == "k-black"
        assert len(client_a.replan_payloads[0]["stated_preferences"]) == 1

        client_b = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body_b = _turn(harness, client_b, k=4, session=second)
        assert client_b.replan_payloads[0]["stated_preferences"] == [], (
            "the second session's decision sees its own memory, and only its own"
        )
        assert _ids(body_b)[0] == "k-hist", "no preference, so no reordering"
        assert harness.active_values(first) == {("color", "prefer", "black")}
        assert harness.active_values(second) == set()
    finally:
        close_harness(harness)


def test_concurrent_turns_keep_their_own_budgets_and_candidates(
    tmp_path: Path, configured: None
) -> None:
    import threading

    harness = _harness(tmp_path)
    try:
        sessions = [harness.create_session()["session_id"] for _ in range(4)]
        # One client for all four turns: the runtime holds one client slot, and this fake answers
        # from the payload it is handed rather than from per-turn state - which is how a real
        # deployment behaves (one provider, four concurrent requests).
        client = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        harness.runtime.use_llm_client(client)
        results: dict[int, list[str]] = {}
        errors: list[BaseException] = []

        def _run(index: int) -> None:
            try:
                response = harness.client.post(
                    f"/v1/demo/sessions/{sessions[index]}/chat",
                    json={"message": REQUEST, "k": 4, "decision_mode": "llm_agent"},
                )
                assert response.status_code == 200, response.text
                results[index] = _ids(response.json())
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=_run, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors
        for index in range(4):
            assert set(results[index]) == {"k-hist", "k-red", "k-blue", "k-black"}, index
        assert client.replan_calls == 4, (
            "each request took exactly one decision; the concurrent turns did not share one"
        )
        assert client.plan_calls == 4
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# H. the switch: off means nothing changes
# --------------------------------------------------------------------------- #


def test_with_the_switch_off_nothing_extra_happens(tmp_path: Path, configured: None) -> None:
    assert replanning_enabled() is False
    assert replanning_enabled("1") is True
    assert replanning_enabled("no") is False

    off = _harness(tmp_path / "off", replanning=False)
    on = _harness(tmp_path / "on", replanning=True)
    try:
        client_off = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body_off = _turn(off, client_off, k=4)
        client_on = _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search")
        body_on = _turn(on, client_on, k=4)

        assert client_off.replan_calls == 0, "no extra model call with the switch off"
        assert client_on.replan_calls == 1
        assert body_off["agent"]["terminal"]["tool_calls"] == body_on["agent"]["terminal"][
            "tool_calls"] - 1, "and no extra tool call"
        assert _ids(body_off) == ["k-hist"], "the answer is the plan's own"
        assert set(_ids(body_on)) == {"k-hist", "k-red", "k-blue", "k-black"}
        # The E-03A terminal gate keeps its own contract in both states, and the grounded block
        # still bounds what may be presented.
        for body in (body_off, body_on):
            assert body["agent"]["grounded"]["grounded_count"] >= len(body["recommendations"])
            assert _replan(body)["decision_source"] == "deterministic_rules"
        assert _replan(body_off)["status"] is None, (
            "with the capability off there is no execution record to report"
        )
        assert _replan(body_on)["status"] == "appended_candidates"
        # With the switch off, an empty first source keeps the accepted refusal.
        empty = _harness(tmp_path / "empty", replanning=False)
        try:
            client_empty = _ReactiveClient(PLAN_CATALOG, replan_source="history")
            body_empty = _turn(empty, client_empty, message="NothingMatchesThisRequest")
            assert body_empty["recommendations"] == []
            assert body_empty["agent"]["terminal"]["termination_reason"] == "completion_refused"
            assert client_empty.replan_calls == 0
        finally:
            close_harness(empty)
    finally:
        close_harness(off)
        close_harness(on)


def test_the_other_paths_are_not_replanned(tmp_path: Path, configured: None) -> None:
    """The request-declared agent endpoint and the deterministic chat never replan."""
    harness = _harness(tmp_path, replanning=True)
    try:
        session = harness.create_session()["session_id"]
        agent_response = harness.client.post(
            "/v1/demo/agent/recommend",
            json={"message": REQUEST, "k": 4, "sources": ["history"],
                  "history": list(harness.manager.get(session).trusted_user_history)},
        )
        assert agent_response.status_code == 200, agent_response.text
        body = agent_response.json()
        assert body["replan_execution"] is None, (
            "bounded replanning belongs to the LLM-selected path only"
        )
        assert body["agent"] is None if "agent" in body else True
        assert [card["parent_asin"] for card in body["recommendations"]] == ["k-hist"]

        deterministic = harness.client.post(
            f"/v1/demo/sessions/{session}/chat", json={"message": "Recommend something.", "k": 4})
        assert deterministic.status_code == 200
        assert deterministic.json()["audit"]["replan"] is None
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# the answer contract of the source choice itself
# --------------------------------------------------------------------------- #


def test_the_source_choice_contract_is_closed() -> None:
    assert parse_replan_answer('{"replan": true, "source": "catalog_search"}',
                              eligible=["catalog_search"]) == ("catalog_search", "chosen")
    assert parse_replan_answer('{"replan": false, "source": "catalog_search"}',
                              eligible=["catalog_search"]) == (None, "model_declined")
    assert parse_replan_answer('{"replan": true, "source": "history"}',
                              eligible=["catalog_search"]) == (None, "invalid_output")
    assert parse_replan_answer('{"replan": true}', eligible=["catalog_search"]) == (
        None, "invalid_output")
    assert parse_replan_answer('```json\n{"replan": true, "source": "history"}\n```',
                              eligible=["history"]) == ("history", "chosen"), (
        "a fenced answer is still readable - the contract is the object, not the packaging"
    )
    assert "appended_candidates" in REPLAN_STATUSES and "appended_empty" in REPLAN_STATUSES


# --------------------------------------------------------------------------- #
# the gate itself: the deterministic rules, on constructed run states
# --------------------------------------------------------------------------- #


def test_the_gate_rules_are_exactly_what_the_documentation_claims() -> None:
    """The gate's inputs are the run's own facts; each rule is asserted on its own."""
    from types import SimpleNamespace

    from recommendation.control import ActionKind
    from recommendation.demo.replan_policy import BoundedReplanPolicy

    policy = BoundedReplanPolicy(["history"], default_k=4, client=lambda: None)
    policy._state.consulted = ["history"]  # one source really was consulted
    base: dict[str, Any] = {
        "run_status": SimpleNamespace(value="running"),
        "available_actions": (ActionKind.SEARCH_CATALOG,),
        "last_proposal_rejected": False,
        "remaining_tool_calls": 3,
        "remaining_steps": 5,
        "step_index": 1,
        "candidate_state": SimpleNamespace(candidate_count=1),
        "hard_constraint_count": 0,
        "ineligible_count": 0,
        "verified_eligible_count": 0,
    }

    def gate(**overrides: Any) -> Any:
        return policy._gate(SimpleNamespace(**{**base, **overrides}))

    assert gate() is None, "budget, an untried legal source and a short answer: allowed"
    assert gate(candidate_state=SimpleNamespace(candidate_count=4)) == "candidates_sufficient"
    assert gate(remaining_tool_calls=0) == "tool_budget_exhausted"
    assert gate(remaining_steps=1) == "not_enough_steps_to_finish", (
        "one step can terminate a run but cannot consult a source and then finish"
    )
    assert gate(available_actions=(ActionKind.FINISH,)) == "sources_exhausted"
    assert gate(last_proposal_rejected=True) == "blocked_by_termination", (
        "a refusal is never a reason to retrieve again"
    )
    assert gate(run_status=SimpleNamespace(value="finished")) == "run_not_running"
    # A hard constraint makes the sufficiency claim stricter: four products that fail it are not an
    # answer, so the gate stays open on the one product that passes.
    assert gate(
        candidate_state=SimpleNamespace(candidate_count=4),
        hard_constraint_count=1,
        ineligible_count=3,
        verified_eligible_count=1,
    ) is None
    # One decision per run, whatever the state looks like afterwards.
    policy._replans = 1
    assert gate() == "replan_already_used"
