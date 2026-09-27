"""Recommendation Trace: the browser demo's consolidated, read-only turn view.

The trace is a VIEW over values the chat response already carries. These tests exist to prove
that, not to re-test recommendation behaviour:

* every trace value agrees with the authoritative value it was copied from;
* the trace cannot invent a product identity or a memory change;
* the acknowledgement comes from the persisted mutation result, never from user text;
* the browser chat path never claims a source or a grounding count it does not have;
* the response stays backward compatible for a client that ignores ``trace``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recommendation.demo.schemas import ChatResponse, RecommendationTrace
from recommendation.demo.serialization import build_trace, preference_only_acknowledgement
from tests.demo_fixture import build_harness

CAMPING = "Recommend some useful gear for a weekend camping trip."
PREFER = "I prefer lightweight and durable products."
AGAIN_WITH_PREFS = "Recommend the camping gear again using my preferences."
REMOVE_LIGHT = "Remove my lightweight preference."
AGAIN = "Recommend the camping gear again."


@pytest.fixture()
def harness(tmp_path: Path) -> Any:
    return build_harness(tmp_path)


def _chat(harness: Any, session_id: str, message: str) -> dict[str, Any]:
    return harness.chat(session_id, message)


# --------------------------------------------------------------------------- #
# 1. backward compatibility
# --------------------------------------------------------------------------- #


def test_response_still_validates_and_trace_is_optional(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, CAMPING)

    parsed = ChatResponse.model_validate(body)
    assert parsed.trace is not None
    # a client that ignores trace reads exactly the fields it always read
    assert parsed.route == body["route"]
    assert parsed.message == body["message"]
    assert parsed.audit.candidate_count == body["audit"]["candidate_count"]

    # the field is optional in the schema, so an older/partial payload still validates
    partial = {k: v for k, v in body.items() if k != "trace"}
    assert ChatResponse.model_validate(partial).trace is None


def test_trace_is_additive_only(harness: Any) -> None:
    """Nothing that existed before may change shape; trace is the only new key."""
    session_id = harness.create_session()["session_id"]
    body_before_shape = set(_chat(harness, session_id, CAMPING))
    assert body_before_shape == {
        "api_version", "session_id", "turn_id", "turn", "route", "message",
        "active_preferences", "memory_update", "recommendations", "audit", "trace",
    }


# --------------------------------------------------------------------------- #
# 2/3/12. ranks, evidence and ordering agree with the rendered cards
# --------------------------------------------------------------------------- #


def test_trace_ranks_match_the_rendered_cards(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, CAMPING)

    trace = body["trace"]
    cards = body["recommendations"]
    assert len(trace["ranking_changes"]) == len(cards)

    for row, card in zip(trace["ranking_changes"], cards):
        assert row["parent_asin"] == card["parent_asin"]
        assert row["original_rank"] == card["original_rank"]
        expected_final = card["reranked_rank"] or card["original_rank"]
        assert row["final_rank"] == expected_final


def test_trace_candidate_count_matches_the_audit_block(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, CAMPING)
    assert body["trace"]["candidate_count"] == body["audit"]["candidate_count"]
    assert body["trace"]["candidate_count"] == len(body["recommendations"])


def test_no_preference_turn_reports_no_evidence(harness: Any) -> None:
    """Turn 1 ranks with no preferences, so there is no evidence to report."""
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, CAMPING)
    assert body["trace"]["evidence"] is None, "a zeroed block would imply a measurement"
    assert all(row["moved"] is False for row in body["trace"]["ranking_changes"]), (
        "with no preferences there is nothing to move a candidate"
    )
    assert [row["original_rank"] for row in body["trace"]["ranking_changes"]] == list(
        range(1, len(body["recommendations"]) + 1)
    )


def test_trace_evidence_counts_agree_with_per_candidate_evidence(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    _chat(harness, session_id, PREFER)
    body = _chat(harness, session_id, AGAIN_WITH_PREFS)

    trace = body["trace"]
    assert trace["evidence"] is not None, "this turn ranked against active preferences"
    summary = trace["evidence"]

    cards = body["recommendations"]
    matched = sum(1 for c in cards if any(e["status"] == "match" for e in c["evidence"]))
    violated = sum(1 for c in cards if any(e["status"] == "violation" for e in c["evidence"]))
    unknown = sum(1 for c in cards if any(e["status"] == "unknown" for e in c["evidence"]))
    assert summary["match_candidates"] == matched
    assert summary["violation_candidates"] == violated
    assert summary["unknown_candidates"] == unknown


# --------------------------------------------------------------------------- #
# 4/5. preferences and mutations come from persisted state
# --------------------------------------------------------------------------- #


def test_trace_active_preferences_match_persisted_state(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    _chat(harness, session_id, PREFER)
    body = _chat(harness, session_id, AGAIN_WITH_PREFS)

    trace_active = {(p["kind"], p["value"]) for p in body["trace"]["active_preferences"]}
    persisted = {
        (p["kind"], p["value"]) for p in harness.state(session_id)["active_preferences"]
    }
    assert trace_active == persisted
    assert trace_active == {("feature", "lightweight"), ("feature", "durable")}


def test_trace_memory_change_reflects_the_persisted_removal(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    _chat(harness, session_id, PREFER)
    body = _chat(harness, session_id, REMOVE_LIGHT)

    changes = body["trace"]["memory_changes"]
    assert changes["changed"] is True
    assert changes["removed"] == ["lightweight"]

    persisted = {p["value"] for p in harness.state(session_id)["active_preferences"]}
    assert "lightweight" not in persisted, "the trace reported a removal that really happened"
    assert persisted == {"durable"}


def test_trace_addition_reflects_the_persisted_additions(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, PREFER)
    assert sorted(body["trace"]["memory_changes"]["added"]) == ["durable", "lightweight"]
    assert body["trace"]["memory_changes"]["removed"] == []


# --------------------------------------------------------------------------- #
# 6. determinism
# --------------------------------------------------------------------------- #


def test_trace_is_deterministic_for_identical_state(tmp_path: Path) -> None:
    def run(name: str) -> str:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
        h = build_harness(tmp_path / name)
        sid = h.create_session()["session_id"]
        _chat(h, sid, CAMPING)
        _chat(h, sid, PREFER)
        body = _chat(h, sid, AGAIN_WITH_PREFS)
        return json.dumps(body["trace"], sort_keys=True)

    first = run("run-a")
    second = run("run-b")
    assert first == second, "the trace must be a pure function of the turn's authoritative state"


# --------------------------------------------------------------------------- #
# 7. the trace cannot invent identities
# --------------------------------------------------------------------------- #


def test_trace_cannot_invent_a_parent_asin(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    _chat(harness, session_id, PREFER)
    body = _chat(harness, session_id, AGAIN_WITH_PREFS)

    in_result = {c["parent_asin"] for c in body["recommendations"]}
    in_trace = {row["parent_asin"] for row in body["trace"]["ranking_changes"]}
    assert in_trace <= in_result, "the trace may only name identities already in the result"


def test_trace_title_is_only_the_card_title(harness: Any) -> None:
    """A title is copied from the card when present; never looked up by the trace."""
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, CAMPING)
    cards = {c["parent_asin"]: c for c in body["recommendations"]}
    for row in body["trace"]["ranking_changes"]:
        card = cards[row["parent_asin"]]
        expected = (card.get("metadata") or {}).get("title")
        assert row["title"] == expected


# --------------------------------------------------------------------------- #
# 8/9. acknowledgement is truthful
# --------------------------------------------------------------------------- #


def test_acknowledgement_uses_the_persisted_mutation_not_user_text(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    _chat(harness, session_id, PREFER)
    body = _chat(harness, session_id, REMOVE_LIGHT)

    assert body["route"] == "direct"
    assert body["message"] == (
        "Got it. I removed your lightweight preference. Active preference: durable."
    )
    assert "AgentRec-X demo" not in body["message"], "the generic capability text is replaced"


def test_unmatched_removal_is_not_acknowledged_as_success(harness: Any) -> None:
    """Removing something that was never stored must not claim success."""
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, "Remove my titanium preference.")

    assert body["memory_update"]["changed"] is False
    assert body["trace"]["memory_changes"]["changed"] is False
    assert "removed" not in body["message"].lower()
    assert body["message"].startswith("I'm a local AgentRec-X demo"), (
        "with no persisted change the honest generic response is kept"
    )


def test_trace_memory_changes_always_equal_the_persisted_update(harness: Any) -> None:
    """Whatever the store decided, the trace reports exactly that and nothing more."""
    session_id = harness.create_session()["session_id"]
    for message in (PREFER, "I prefer durable", "Remove my titanium preference.", REMOVE_LIGHT):
        body = _chat(harness, session_id, message)
        update = body["memory_update"]
        changes = body["trace"]["memory_changes"]
        assert changes["changed"] == update["changed"]
        assert changes["added"] == [item["value"] for item in update["added"]]
        assert changes["removed"] == [item["value"] for item in update["removed"]]
        assert changes["superseded"] == [item["value"] for item in update["superseded"]]
        if not update["changed"]:
            assert "Got it." not in body["message"]


def test_acknowledgement_helper_returns_none_without_a_change() -> None:
    class _Update:
        changed = False
        added: tuple = ()
        removed: tuple = ()
        superseded: tuple = ()

    assert preference_only_acknowledgement(_Update(), ()) is None


# --------------------------------------------------------------------------- #
# omitted fields on the browser path
# --------------------------------------------------------------------------- #


def test_browser_trace_omits_source_and_grounding(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    _chat(harness, session_id, PREFER)
    body = _chat(harness, session_id, AGAIN_WITH_PREFS)

    trace = body["trace"]
    assert trace["source"] is None, "no authoritative source exists on the browser path"
    assert trace["grounding"] is None, "no ledger or plane exists on the browser path"
    blob = json.dumps(trace)
    for banned in ("SASRec", "RecommendationTool", "two_tower", "tiger"):
        assert banned not in blob, banned


def test_browser_trace_never_exposes_prompts_or_reasoning(harness: Any) -> None:
    session_id = harness.create_session()["session_id"]
    body = _chat(harness, session_id, CAMPING)
    blob = json.dumps(body["trace"]).lower()
    for banned in ("prompt", "chain", "reasoning", "thought", "score"):
        assert banned not in blob, banned


# --------------------------------------------------------------------------- #
# the builder is a view, not a second computation
# --------------------------------------------------------------------------- #


def test_trace_module_does_not_import_evaluation_or_recompute_evidence() -> None:
    source = (Path(__file__).resolve().parents[1] / "recommendation/demo/serialization.py").read_text()
    for banned in ("from recommendation.evaluation", "import evaluation", "match_candidates(", "ndcg"):
        assert banned not in source, banned


def test_build_trace_requires_no_state_object() -> None:
    """The builder takes serialized values only, so it cannot reach back into raw state."""
    import inspect

    params = set(inspect.signature(build_trace).parameters)
    assert params == {
        "route", "audit", "cards", "active_preferences", "memory_update"
    }, params


def test_trace_schema_marks_source_and_grounding_optional() -> None:
    fields = RecommendationTrace.model_fields
    assert fields["source"].default is None
    assert fields["grounding"].default is None
    assert fields["candidate_count"].default is None


# --------------------------------------------------------------------------- #
# browser asset existence
# --------------------------------------------------------------------------- #


def test_browser_panel_is_named_recommendation_trace() -> None:
    web = Path(__file__).resolve().parents[1] / "recommendation/web"
    html_js = (web / "app.js").read_text() + (web / "index.html").read_text()
    assert "Recommendation Trace" in html_js
    assert "renderTrace" in (web / "app.js").read_text()
    assert "Agent Trace" not in html_js


def test_browser_panel_omits_source_and_grounding_when_absent() -> None:
    js = (Path(__file__).resolve().parents[1] / "recommendation/web/app.js").read_text()
    start = js.index("function renderTrace(")
    end = js.index("function renderPreferences(")
    trace_renderer = js[start:end]
    assert "if (trace.source)" in trace_renderer
    assert "if (trace.grounding)" in trace_renderer
    assert "SASRec" not in trace_renderer, "the panel must not hardcode a source name"
    assert "RecommendationTool" not in trace_renderer
