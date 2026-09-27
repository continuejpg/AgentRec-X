"""The optional LLM decision mode as it is actually served over the demo HTTP API.

These are integration tests for the *wiring*, not for the policy core (that is
``tests/test_demo_llm_policy.py``).  Every provider call is answered by the repository's existing
``ScriptedModelClient``, so the suite needs no network and no credential; the deployment is
nonetheless declared fully configured, because the point is that the mode is reached through the
same request path a browser uses.

What is proven here:

* an existing client that never sends a decision mode keeps exactly its old behaviour, and no
  response gains a new top-level field;
* the session carries the mode, and a turn that omits the field inherits it;
* the provider abstraction really is invoked, once per turn;
* validated plan actions reach the accepted Milestone 9 service, and the recommendation truth
  still comes from the accepted components;
* a missing credential, a provider failure, a timeout, malformed output and an unsupported
  action all fail closed - no memory mutation, no fabricated success, and never a silent switch
  to the deterministic path while the browser is showing ``LLM Agent``;
* the trace distinguishes the modes and can never carry model-supplied product identity, scores,
  evidence or ranks, or any prompt/reasoning text.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import pytest

from recommendation.control.model_client import ModelCallError, ModelResponse, ScriptedModelClient
from recommendation.demo.llm_policy import DECISION_MODES, DEFAULT_DECISION_MODE, LLMTurnPlan
from recommendation.demo.schemas import DecisionMode, RecommendationTrace
from recommendation.demo.sessions import UnsupportedDecisionMode
from tests.demo_fixture import build_harness, close_harness

FREE_TEXT = (
    "I'm going hiking in the rain. Drop the lightweight preference, "
    "but I'd still prefer something durable."
)
SUBSTITUTION_PLAN = {
    "route": "recommend",
    "add": [{"value": "durable", "kind": "feature"}],
    "remove": ["lightweight"],
}
PREFER_BOTH = "I prefer lightweight and durable products."
REMOVE_LIGHT = "Remove my lightweight preference."
DIRECT_PLAN = {"route": "direct", "add": [], "remove": []}

#: The response keys the demo published before the decision mode existed.  The mode is carried
#: inside ``trace``, so this set must not change.
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
}


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that looks fully configured, without any real provider.

    The variables are the accepted provider configuration; nothing connects, because every test
    installs a scripted client.  Declaring the deployment configured is what makes the
    *availability* path and the *no silent fallback* assertions meaningful.
    """
    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")


def _client(*answers: Any, provider: str = "deepseek", model: str = "deepseek-chat") -> Any:
    """A scripted provider client that declares the identity a real adapter declares."""
    client = ScriptedModelClient(list(answers), repeat_last=True)
    client.profile = SimpleNamespace(name=provider)
    client.model = model
    return client


class _TimeoutClient:
    """A client whose transport always times out: a provider failure with a distinct code."""

    profile = SimpleNamespace(name="deepseek")
    model = "deepseek-chat"

    def __init__(self) -> None:
        self.requests: list[Any] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def complete(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        raise ModelCallError("the model request timed out", code="timeout")


def _harness(tmp_path: Path, client: Any = None) -> Any:
    harness = build_harness(tmp_path)
    if client is not None:
        harness.runtime.use_llm_client(client)
    return harness


def _chat(harness: Any, session_id: str, message: str, mode: str | None = None) -> Any:
    body: dict[str, Any] = {"message": message, "k": 5}
    if mode is not None:
        body["decision_mode"] = mode
    return harness.client.post(f"/v1/demo/sessions/{session_id}/chat", json=body)


def _ok(response: Any) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _prefer_lightweight(harness: Any, session_id: str) -> None:
    """Put one explicit preference in memory through the accepted deterministic path."""
    _ok(_chat(harness, session_id, "I prefer lightweight products."))


# --------------------------------------------------------------------------- #
# 1. backward compatibility and the deterministic default
# --------------------------------------------------------------------------- #


def test_existing_clients_default_to_deterministic(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        created = harness.client.post("/v1/demo/sessions", json={"profile_id": "demo-user-1"})
        assert created.status_code == 201
        assert created.json()["decision_mode"] == DEFAULT_DECISION_MODE

        session_id = created.json()["session_id"]
        body = _ok(_chat(harness, session_id, "Recommend some hiking gear."))
        assert body["trace"]["decision_mode"] == "deterministic"
        assert set(body) == CHAT_RESPONSE_KEYS, "the mode must not add a top-level field"
        assert harness.state(session_id)["decision_mode"] == "deterministic"
    finally:
        close_harness(harness)


def test_deterministic_trace_claims_no_provider_route_or_action(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session_id = harness.create_session()["session_id"]
        trace = _ok(_chat(harness, session_id, "Recommend some hiking gear."))["trace"]
        assert trace["decision_mode"] == "deterministic"
        assert trace["provider"] is None
        assert trace["model"] is None
        assert trace["proposed_route"] is None
        assert trace["preference_actions"] == []
    finally:
        close_harness(harness)


def test_deterministic_behaviour_is_unchanged(tmp_path: Path) -> None:
    """The accepted preference semantics still hold with the LLM path present and unused."""
    harness = _harness(tmp_path)
    try:
        session_id = harness.create_session()["session_id"]
        _ok(_chat(harness, session_id, PREFER_BOTH))
        assert harness.active_values(session_id) == {
            ("feature", "prefer", "durable"),
            ("feature", "prefer", "lightweight"),
        }

        body = _ok(_chat(harness, session_id, REMOVE_LIGHT))
        assert body["route"] == "direct"
        assert body["message"] == (
            "Got it. I removed your lightweight preference. Active preference: durable."
        )
        assert harness.active_values(session_id) == {("feature", "prefer", "durable")}
        assert body["trace"]["decision_mode"] == "deterministic"
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 2. session mode
# --------------------------------------------------------------------------- #


def test_session_mode_is_created_and_persisted(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path, _client(SUBSTITUTION_PLAN))
    try:
        created = harness.client.post(
            "/v1/demo/sessions", json={"profile_id": "demo-user-1", "decision_mode": "llm"}
        )
        assert created.status_code == 201, created.text
        session_id = created.json()["session_id"]
        assert created.json()["decision_mode"] == "llm"
        assert harness.state(session_id)["decision_mode"] == "llm"

        # A turn that omits the field inherits the session's mode - it does not fall back.
        body = _ok(_chat(harness, session_id, FREE_TEXT))
        assert body["trace"]["decision_mode"] == "llm"
        assert body["trace"]["provider"] == "deepseek"
    finally:
        close_harness(harness)


def test_request_field_switches_the_session_mode_and_it_sticks(
    tmp_path: Path, configured: None
) -> None:
    client = _client(SUBSTITUTION_PLAN)
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        assert _ok(_chat(harness, session_id, "Recommend gear."))["trace"]["decision_mode"] == (
            "deterministic"
        )

        switched = _ok(_chat(harness, session_id, "Recommend gear.", mode="llm"))
        assert switched["trace"]["decision_mode"] == "llm"
        assert harness.state(session_id)["decision_mode"] == "llm"

        inherited = _ok(_chat(harness, session_id, "Recommend gear."))
        assert inherited["trace"]["decision_mode"] == "llm"
        assert client.call_count == 2, "each LLM turn plans once"

        back = _ok(_chat(harness, session_id, "Recommend gear.", mode="deterministic"))
        assert back["trace"]["decision_mode"] == "deterministic"
        assert harness.state(session_id)["decision_mode"] == "deterministic"
        assert client.call_count == 2, "a deterministic turn makes no provider call"
    finally:
        close_harness(harness)


def test_unknown_decision_mode_is_rejected(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        session_id = harness.create_session()["session_id"]
        response = _chat(harness, session_id, "Recommend gear.", mode="adaptive")
        assert response.status_code == 422
        assert harness.state(session_id)["decision_mode"] == "deterministic"
    finally:
        close_harness(harness)


def test_session_manager_refuses_an_unknown_mode(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    try:
        with pytest.raises(UnsupportedDecisionMode):
            harness.manager.create("demo-user-1", decision_mode="adaptive")
        session = harness.manager.get(harness.create_session()["session_id"])
        with pytest.raises(UnsupportedDecisionMode):
            session.switch_decision_mode("adaptive")
        assert session.decision_mode == "deterministic", "a refused switch changes nothing"
    finally:
        close_harness(harness)


def test_mode_literal_matches_the_policy_layer(tmp_path: Path) -> None:
    """The wire contract and the policy layer name the same two modes, in the same default."""
    assert set(get_args(DecisionMode)) == set(DECISION_MODES)
    assert DEFAULT_DECISION_MODE == DECISION_MODES[0]


# --------------------------------------------------------------------------- #
# 3/4. the provider is invoked, and trusted code still owns the truth
# --------------------------------------------------------------------------- #


def test_llm_mode_invokes_the_provider_abstraction_once(tmp_path: Path, configured: None) -> None:
    client = _client(SUBSTITUTION_PLAN)
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, FREE_TEXT, mode="llm"))

        assert client.call_count == 1, "one turn is one provider call"
        request = client.last_request
        assert request is not None
        assert request.context_payload == {"user_request": FREE_TEXT}
        assert body["trace"]["proposed_route"] == "recommend"
        assert body["route"] == "recommend"
    finally:
        close_harness(harness)


def test_validated_actions_reach_the_accepted_memory_service(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _client(SUBSTITUTION_PLAN))
    try:
        session_id = harness.create_session()["session_id"]
        _prefer_lightweight(harness, session_id)
        before = harness.active_values(session_id)

        body = _ok(_chat(harness, session_id, FREE_TEXT, mode="llm"))
        assert harness.active_values(session_id) == {("feature", "prefer", "durable")}

        update = body["memory_update"]
        assert update["changed"] is True
        assert [item["value"] for item in update["added"]] == ["durable"]
        assert [item["value"] for item in update["removed"]] == ["lightweight"]

        actions = body["trace"]["preference_actions"]
        assert [(item["action"], item["value"], item["applied"]) for item in actions] == [
            ("remove", "lightweight", True),
            ("add", "durable", True),
        ]
        assert before == {("free_form_constraint", "prefer", "lightweight")}

        # Provenance: the accepted store recorded where the value came from.
        user_key = harness.manager.get(session_id).user_key
        entries = harness.store.get_entries(user_key, active_only=False)
        assert any(entry.extractor == "llm_policy_v1" for entry in entries)
    finally:
        close_harness(harness)


def test_recommendation_truth_still_comes_from_trusted_code(
    tmp_path: Path, configured: None
) -> None:
    """The model never sees, and cannot produce, a candidate identity, score or rank."""
    from tests.agent_reranking_fixture import CANDIDATE_ROWS

    client = _client(SUBSTITUTION_PLAN)
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, FREE_TEXT, mode="llm"))

        # The engine the accepted Tool drives is the only source of candidates and scores.
        assert harness.engine.call_count == 1
        expected = {row[0]: row[2] for row in CANDIDATE_ROWS[:5]}
        assert body["audit"]["original_order"] == [row[0] for row in CANDIDATE_ROWS[:5]]
        cards = body["recommendations"]
        assert cards, "the accepted Tool still produced the candidates"
        for card in cards:
            assert card["parent_asin"] in expected
            assert card["sasrec_score"] == pytest.approx(expected[card["parent_asin"]])

        # And none of that identity or scoring ever reached the provider.  The word "score"
        # appears only in the prompt's prohibition ("never name ... a score"), so what is
        # checked is the identity, the score values and the field names themselves.
        sent = json.dumps(client.last_request.as_messages())
        for identity in expected:
            assert identity not in sent
        for score in expected.values():
            assert str(score) not in sent
        for field in ("sasrec_score", "item_id", "parent_asin", "evidence", "rank"):
            assert field not in sent
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 5. failure behaviour
# --------------------------------------------------------------------------- #


def test_provider_failure_leaves_memory_untouched(tmp_path: Path, configured: None) -> None:
    harness = _harness(tmp_path, _client(TimeoutError("transport down")))
    try:
        session_id = harness.create_session()["session_id"]
        _prefer_lightweight(harness, session_id)
        before = harness.active_values(session_id)

        response = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert response.status_code == 502
        assert response.json()["error"] == "llm_provider_error"
        assert "no preference was changed" in response.json()["detail"]
        assert harness.active_values(session_id) == before
    finally:
        close_harness(harness)


def test_timeout_is_reported_distinctly(tmp_path: Path, configured: None) -> None:
    client = _TimeoutClient()
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        _prefer_lightweight(harness, session_id)
        before = harness.active_values(session_id)

        response = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert response.status_code == 504
        assert response.json()["error"] == "llm_timeout"
        assert client.call_count == 1, "a transport failure is not retried"
        assert harness.active_values(session_id) == before
    finally:
        close_harness(harness)


def test_malformed_output_is_fail_closed(tmp_path: Path, configured: None) -> None:
    client = _client("this is not json")
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        _prefer_lightweight(harness, session_id)
        before = harness.active_values(session_id)

        response = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert response.status_code == 502
        assert response.json()["error"] == "llm_invalid_plan"
        assert client.call_count == 2, "the retry stays inside the policy's bound"
        assert harness.active_values(session_id) == before
    finally:
        close_harness(harness)


def test_unsupported_action_is_fail_closed(tmp_path: Path, configured: None) -> None:
    client = _client(
        {"route": "recommend", "add": [], "remove": [], "action": "SELECT_SOURCE"}
    )
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        _prefer_lightweight(harness, session_id)
        before = harness.active_values(session_id)

        response = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert response.status_code == 502
        assert response.json()["error"] == "llm_invalid_plan"
        assert harness.active_values(session_id) == before
    finally:
        close_harness(harness)


def test_injected_truth_keys_are_a_validation_error(tmp_path: Path, configured: None) -> None:
    """A plan that tries to carry product identity, scores or evidence is rejected outright."""
    for truth_key, value in (
        ("parent_asin", "B000000001"),
        ("item_id", 7),
        ("score", 0.99),
        ("evidence", ["match"]),
        ("rank", 1),
        ("recommendations", [{"parent_asin": "B000000001"}]),
    ):
        client = _client(
            {"route": "recommend", "add": [], "remove": [], truth_key: value}
        )
        case_dir = tmp_path / truth_key
        case_dir.mkdir(parents=True, exist_ok=True)
        harness = _harness(case_dir, client)
        try:
            session_id = harness.create_session()["session_id"]
            response = _chat(harness, session_id, FREE_TEXT, mode="llm")
            assert response.status_code == 502, truth_key
            payload = response.json()
            assert payload["error"] == "llm_invalid_plan", truth_key
            # The error body is authored by the API layer: it echoes neither the injected key
            # nor anything else the model returned.
            assert set(payload) == {"error", "detail"}, truth_key
            assert truth_key not in json.dumps(payload), truth_key
        finally:
            close_harness(harness)


def test_no_silent_fallback_after_a_failed_llm_turn(tmp_path: Path, configured: None) -> None:
    """The session stays in LLM mode, so the next field-less turn fails the same way."""
    client = _client(TimeoutError("transport down"))
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        first = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert first.status_code == 502
        assert harness.state(session_id)["decision_mode"] == "llm"

        second = _chat(harness, session_id, FREE_TEXT)
        assert second.status_code == 502, "a failed LLM session must not answer deterministically"
        assert second.json()["error"] == "llm_provider_error"

        switched = _ok(_chat(harness, session_id, "Recommend gear.", mode="deterministic"))
        assert switched["trace"]["decision_mode"] == "deterministic"
    finally:
        close_harness(harness)


def test_missing_credential_is_refused_before_any_turn(tmp_path: Path) -> None:
    """No credential: the mode is unavailable, and asking for it changes nothing."""
    harness = _harness(tmp_path)
    try:
        modes = harness.client.get("/v1/demo/decision-modes").json()
        llm = [item for item in modes["modes"] if item["mode"] == "llm"][0]
        assert llm["available"] is False
        assert llm["reason"] in {"missing_api_key", "not_configured"}
        assert llm["detail"]

        session_id = harness.create_session()["session_id"]
        _prefer_lightweight(harness, session_id)
        before = harness.active_values(session_id)

        chat = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert chat.status_code == 503
        assert chat.json()["error"] == "llm_unavailable"
        created = harness.client.post(
            "/v1/demo/sessions", json={"profile_id": "demo-user-1", "decision_mode": "llm"}
        )
        assert created.status_code == 503
        assert created.json()["error"] == "llm_unavailable"
        assert harness.active_values(session_id) == before
        assert harness.state(session_id)["decision_mode"] == "deterministic"
    finally:
        close_harness(harness)


def test_llm_mode_is_refused_on_a_control_plane_it_cannot_run(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _client(SUBSTITUTION_PLAN))
    try:
        harness.runtime.control_plane = "loop"
        modes = harness.client.get("/v1/demo/decision-modes").json()
        llm = [item for item in modes["modes"] if item["mode"] == "llm"][0]
        assert llm["available"] is False
        assert llm["reason"] == "control_plane_unsupported"

        session_id = harness.create_session()["session_id"]
        response = _chat(harness, session_id, FREE_TEXT, mode="llm")
        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"
    finally:
        close_harness(harness)


def test_direct_route_plan_still_persists_through_the_graph(
    tmp_path: Path, configured: None
) -> None:
    """A plan may route to a direct turn; the accepted memory node still runs on that route."""
    client = _client({"route": "direct", "add": [{"value": "durable", "kind": "feature"}], "remove": []})
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, "I would prefer something durable.", mode="llm"))
        assert body["route"] == "direct"
        assert body["trace"]["proposed_route"] == "direct"
        assert harness.active_values(session_id) == {("feature", "prefer", "durable")}
        assert body["message"] == (
            "Got it. I saved your durable preference for future turns. "
            "Active preference: durable."
        )
    finally:
        close_harness(harness)


# --------------------------------------------------------------------------- #
# 6. the trace
# --------------------------------------------------------------------------- #


def test_trace_distinguishes_deterministic_from_llm(tmp_path: Path, configured: None) -> None:
    client = _client(SUBSTITUTION_PLAN)
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        deterministic = _ok(_chat(harness, session_id, "Recommend gear."))["trace"]
        llm = _ok(_chat(harness, session_id, FREE_TEXT, mode="llm"))["trace"]

        assert deterministic["decision_mode"] == "deterministic"
        assert llm["decision_mode"] == "llm"
        assert deterministic["provider"] is None and llm["provider"] == "deepseek"
        assert deterministic["model"] is None and llm["model"] == "deepseek-chat"
        assert deterministic["proposed_route"] is None
        assert llm["proposed_route"] == "recommend"
        assert deterministic["preference_actions"] == []
        assert [item["value"] for item in llm["preference_actions"]] == ["lightweight", "durable"]
    finally:
        close_harness(harness)


def test_trace_never_exposes_prompts_or_model_text(tmp_path: Path, configured: None) -> None:
    marker = "RAW_MODEL_MARKER"
    client = _client({"route": "recommend", "add": [{"value": marker, "kind": "feature"}], "remove": []})
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        body = _ok(_chat(harness, session_id, FREE_TEXT, mode="llm"))

        blob = json.dumps(body["trace"]).lower()
        for banned in ("prompt", "chain", "reasoning", "thought", "context_payload"):
            assert banned not in blob, banned
        # The model's own text may appear only as a validated preference VALUE, never as prose.
        assert '"text"' not in json.dumps(body["trace"])
        assert set(RecommendationTrace.model_fields) == {
            "route",
            "candidate_count",
            "decision_mode",
            "provider",
            "model",
            "proposed_route",
            "preference_actions",
            "active_preferences",
            "memory_changes",
            "evidence",
            "ranking_changes",
            "source",
            "grounding",
        }
    finally:
        close_harness(harness)


def test_trace_reports_only_applied_actions(tmp_path: Path, configured: None) -> None:
    """A validated action the store declined is reported as not applied, never as a change."""
    client = _client(SUBSTITUTION_PLAN)
    harness = _harness(tmp_path, client)
    try:
        session_id = harness.create_session()["session_id"]
        # Nothing is active, so the removal matches nothing and the add is the only change.
        body = _ok(_chat(harness, session_id, FREE_TEXT, mode="llm"))
        actions = {item["action"]: item for item in body["trace"]["preference_actions"]}
        assert actions["remove"]["applied"] is False
        assert actions["add"]["applied"] is True
        assert body["memory_update"]["removed"] == []
    finally:
        close_harness(harness)


def test_plan_schema_cannot_express_truth() -> None:
    """The plan the model answers with has no field for identity, scores, evidence or ranks."""
    fields = set(LLMTurnPlan.model_fields)
    assert fields == {"route", "add", "remove"}
    for banned in ("parent_asin", "item_id", "score", "evidence", "rank", "recommendations"):
        assert banned not in fields


# --------------------------------------------------------------------------- #
# 7. the browser reflects real availability
# --------------------------------------------------------------------------- #


def test_decision_modes_endpoint_reports_real_availability(
    tmp_path: Path, configured: None
) -> None:
    harness = _harness(tmp_path, _client(SUBSTITUTION_PLAN))
    try:
        body = harness.client.get("/v1/demo/decision-modes").json()
        assert body["default"] == "deterministic"
        by_mode = {item["mode"]: item for item in body["modes"]}
        assert by_mode["deterministic"]["available"] is True
        assert by_mode["llm"]["available"] is True
        assert by_mode["llm"]["provider"] == "deepseek"
        assert by_mode["llm"]["model"] == "deepseek-chat"
    finally:
        close_harness(harness)


def test_browser_selector_reflects_provider_availability() -> None:
    web = Path(__file__).resolve().parents[1] / "recommendation" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")

    assert 'id="mode-deterministic"' in html
    assert 'id="mode-llm"' in html
    assert 'id="mode-note"' in html
    assert "Decision Mode" in html
    # Deterministic is checked in the markup, and the LLM option starts disabled: availability
    # is confirmed by the server before it can be selected.
    assert 'value="deterministic" checked' in html
    assert 'id="mode-llm" name="decision-mode" value="llm" disabled' in html

    assert '"/decision-modes"' in js
    assert "els.modeLlm.disabled = !llmAvailable" in js
    assert "turnBody.decision_mode = state.decisionMode" in js
    assert 'trace.decision_mode === "llm" ? "LLM Agent" : "Deterministic"' in js
    assert "proposed_route" in js
    assert "preference_actions" in js

    # A disabled option cannot be clicked, so the reason must be visible while it is disabled.
    note = js.split("function renderModeNote(", 1)[1].split("function renderModeChoice(", 1)[0]
    assert "llm && !llm.available" in note
    assert "LLM Agent unavailable: " in note
    assert "llm.detail" in note, "the reason shown must be the server's own detail"


def test_client_omits_the_field_against_a_server_without_the_mode_list() -> None:
    """Backward compatible in both directions: the field is sent only when it is understood.

    A server built before the decision mode existed rejects unknown request fields
    (``extra="forbid"``), so a client that always sent ``decision_mode`` would fail to open a
    session there.  The client therefore sends it only after the server has advertised the mode
    list, and reports the mode as unavailable otherwise.
    """
    web = Path(__file__).resolve().parents[1] / "recommendation" / "web"
    js = (web / "app.js").read_text(encoding="utf-8")

    assert "if (state.modes && state.modes.length) {" in js
    assert "state.modeProbeFailed = true" in js
    assert "does not advertise decision modes" in js
    for block, assignment in (
        ("function createSession(", "body.decision_mode = requested;"),
        ("function send(", "turnBody.decision_mode = state.decisionMode;"),
    ):
        guarded = js.split(block, 1)[1].split("\n  }", 1)[0]
        assert assignment in guarded, block
        assert "if (state.modes && state.modes.length) {" in guarded, block


def test_unavailable_mode_is_not_selectable_but_explains_itself() -> None:
    """The served client disables the option and names the reason without a click."""
    web = Path(__file__).resolve().parents[1] / "recommendation" / "web"
    js = (web / "app.js").read_text(encoding="utf-8")
    choice = js.split("function renderModeChoice(", 1)[1].split("function loadDecisionModes(", 1)[0]
    assert 'els.modeLlm.setAttribute("title"' in choice
    assert "unavailable" in choice
