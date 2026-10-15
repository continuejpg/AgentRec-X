"""E-01A-3A/3B: the agent -> chat adapter converts real results and invents nothing.

Fixtures are **schema-valid** ``AgentRecommendResponse`` objects (built through pydantic, with
required fields filled by introspection so the test cannot drift from the real contract).  The
adapter itself is executed for real; nothing is mocked.

E-01A-3B extended this file with the two corrections the chat integration needed: the execution
record (``ChatResponse.agent``) that makes a direct turn, an empty result and a refused completion
distinguishable, and the title-only catalogue view that lets a real product title be displayed
without inventing the attributes around it.
"""

from __future__ import annotations

from typing import Any

import pytest

import re

from recommendation.demo.agent_chat_adapter import (
    AGENT_SCORE_KIND,
    agent_execution_digest,
    build_agent_chat_response,
)
from recommendation.demo.agent_schemas import AgentRecommendResponse
from recommendation.demo.schemas import ChatResponse


def _fill(model: Any, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """A schema-valid payload: required fields get type-appropriate placeholders.

    Nested models are filled recursively so a required sub-model (``eligibility``) is satisfied by
    the schema rather than by a placeholder the schema would reject.
    """
    overrides = overrides or {}
    payload: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name in overrides:
            payload[name] = overrides[name]
            continue
        if not field.is_required():
            continue
        annotation = field.annotation
        payload[name] = _placeholder(annotation)
    return payload


def _placeholder(annotation: Any) -> Any:
    from enum import Enum

    from pydantic import BaseModel

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _fill(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return list(annotation)[0].value
    text = str(annotation)
    if "bool" in text:
        return False
    if "int" in text:
        return 0
    if "float" in text:
        return 0.0
    if "dict" in text or "Mapping" in text:
        return {}
    if "tuple" in text or "list" in text or "Sequence" in text or "set" in text:
        return ()
    return "x"


def agent_response(*, recommendations=(), route="recommend", succeeded=True,
                   termination_reason="completed", grounded_count=0, actions=(), message="ok",
                   steps=(), sources_present=("history",), outcome_detail=None):
    payload = _fill(AgentRecommendResponse)
    payload.update({
        "route": route,
        "message": message,
        "control_plane": "loop",
        "recommendations": recommendations,
        "grounded": {
            "grounded_count": grounded_count,
            "ungrounded_count": 0,
            "multi_source_count": 0,
            "sources_present": tuple(sources_present),
            "candidates": (),
        },
        "terminal": {"run_id": "run-1", "status": "finished", "succeeded": succeeded,
                     "termination_reason": termination_reason,
                     "termination_detail": outcome_detail,
                     "steps": 2, "tool_calls": 1, "retries": 0,
                     "limits": {"max_steps": 6, "max_tool_calls": 4}},
        "trajectory": {"step_count": len(steps) or len(actions), "actions": tuple(actions),
                       "refusals": (), "steps": tuple(steps)},
        "timing": {"total_ms": 1.0, "agent_ms": 0.9, "serialize_ms": 0.1},
    })
    try:
        return AgentRecommendResponse.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface the payload so the fixture is debuggable
        raise AssertionError(f"fixture is not schema-valid: {exc}\npayload={payload!r}") from exc


def step(index: int, action: str, *, validated: bool = True, candidates: int = 0,
         observation: str | None = None) -> dict[str, Any]:
    """One projected trajectory step, in the shape ``build_trajectory_summary`` produces."""
    return {"step": index, "action": action, "validated": validated,
            "observation_type": observation, "candidate_count": candidates,
            "eligibility_state": None, "refusal_codes": (), "note": None}


def candidate(rank: int, asin: str, item_id: int, score: float, *, sources=("history",),
              title=None, score_kind=None) -> dict[str, Any]:
    return {"rank": rank, "item_id": item_id, "parent_asin": asin, "score": score,
            "sources": tuple(sources), "title": title, "score_kind": score_kind}


def convert(response, **kwargs):
    defaults = {"session_id": "s1", "turn_id": "s1:1", "turn": 1}
    defaults.update(kwargs)
    return build_agent_chat_response(response, **defaults)


#: A numbered product line of the card-derived prose (E-02C-NARRATION-FIX).
_NARRATION_LINE = re.compile(r"^(\d+)\. (\S+) \(item \d+\)", re.MULTILINE)


def test_a_single_source_result_keeps_identity_order_and_provenance() -> None:
    response = agent_response(recommendations=[candidate(1, "B01", 11, 0.5, title="Tent")])
    chat = convert(response)
    assert isinstance(chat, ChatResponse)
    card = chat.recommendations[0]
    assert (card.parent_asin, card.item_id, card.original_rank) == ("B01", 11, 1)
    assert card.fusion_sources == ("history",), "real provenance must survive"
    assert card.metadata_status == "found"
    # A title is a catalogue fact and may be shown; it is not a licence to invent the rest.
    assert card.metadata is not None and card.metadata.title == "Tent"
    assert [card.metadata.store, card.metadata.main_category, card.metadata.price_text] == [
        None, None, None], "a title alone must not be dressed up as a full catalogue record"
    assert card.metadata.categories == () and card.metadata.details == ()
    assert agent_execution_digest(response)["titles"] == {"B01": "Tent"}
    assert card.score_kind == AGENT_SCORE_KIND, "the score's origin must be labelled"


def test_a_multi_source_result_keeps_every_candidate_in_order() -> None:
    response = agent_response(recommendations=[
        candidate(1, "B01", 11, 0.5, sources=("history", "catalog_search")),
        candidate(2, "B02", 12, 0.4, sources=("catalog_search",)),
        candidate(3, "B03", 13, 0.3, sources=("history",)),
    ], grounded_count=3)
    chat = convert(response)
    assert [card.parent_asin for card in chat.recommendations] == ["B01", "B02", "B03"]
    assert [card.fusion_sources for card in chat.recommendations] == [
        ("history", "catalog_search"), ("catalog_search",), ("history",)]
    assert chat.route == "recommend"
    # E-02C-NARRATION-FIX: a response that presents products narrates *those* products.  The stub's
    # own "ok" is the run's intermediate text, and reusing it is what let the prose disagree with
    # the cards; the prose is now the card-derived list, in the cards' order.
    assert chat.message != "ok"
    listed = [
        (int(position), parent_asin)
        for position, parent_asin in _NARRATION_LINE.findall(chat.message)
    ]
    assert listed == [(1, "B01"), (2, "B02"), (3, "B03")]
    # A caller that has its own text still overrides the renderer.
    assert convert(response, message="ok").message == "ok"


def test_an_empty_run_produces_no_card_at_all() -> None:
    chat = convert(agent_response(recommendations=(), route="direct"))
    assert chat.recommendations == ()
    assert chat.audit.candidate_count == 0
    assert chat.route == "direct", "an empty run must not be presented as a recommendation"


def test_a_refused_completion_is_not_dressed_up_as_success() -> None:
    response = agent_response(recommendations=(), route="direct", succeeded=False,
                              termination_reason="completion_refused")
    chat = convert(response)
    assert chat.recommendations == ()
    assert chat.route == "direct"
    digest = agent_execution_digest(response)
    assert digest["terminal"]["succeeded"] is False
    assert digest["terminal"]["termination_reason"] == "completion_refused"


def test_missing_optional_attributes_are_reported_as_missing_not_invented() -> None:
    response = agent_response(recommendations=[candidate(1, "B01", 11, 0.5, title=None)])
    card = convert(response).recommendations[0]
    assert card.metadata_status == "missing"
    assert card.metadata is None, "no title may be manufactured"
    assert agent_execution_digest(response)["titles"] == {}, "an absent title stays absent"


def test_the_controller_score_is_never_relabelled_as_a_sasrec_logit() -> None:
    response = agent_response(recommendations=[
        candidate(1, "B01", 11, 0.5, score_kind="rrf_rank_fusion")])
    card = convert(response).recommendations[0]
    assert card.score_kind == "rrf_rank_fusion", "the response's own score kind wins"
    assert card.sasrec_score == 0.5


def test_the_output_round_trips_through_the_chat_schema() -> None:
    response = agent_response(recommendations=[candidate(1, "B01", 11, 0.5, title="Tent")])
    chat = convert(response)
    dumped = chat.model_dump_json()
    assert ChatResponse.model_validate_json(dumped) == chat


def test_the_execution_digest_carries_only_real_evidence() -> None:
    response = agent_response(recommendations=[candidate(1, "B01", 11, 0.5)],
                              actions=("recommend_from_history", "finish"), grounded_count=1)
    digest = agent_execution_digest(response)
    assert digest["trajectory"]["actions"] == ["recommend_from_history", "finish"]
    assert digest["grounding"]["grounded_count"] == 1
    assert digest["terminal"]["tool_calls"] == 1
    assert digest["timing_ms"]["total_ms"] == 1.0


def test_the_legacy_state_builder_is_untouched() -> None:
    """The adapter is additive: the old graph-state path still builds a valid response."""
    from recommendation.demo.serialization import build_chat_response

    state = {"route": "direct", "message": "hello"}
    chat = build_chat_response(state, session_id="s1", turn_id="s1:1", turn_number=1)
    assert isinstance(chat, ChatResponse) and chat.route == "direct"
    assert chat.agent is None, "the graph path must not claim an agent execution record"


# --------------------------------------------------------------------------- #
# E-01A-3B: the execution record, and the outcomes route alone cannot express
# --------------------------------------------------------------------------- #


def test_a_direct_plan_is_recorded_as_a_direct_turn_with_no_controller() -> None:
    from recommendation.demo.agent_chat_adapter import direct_agent_execution
    from recommendation.demo.serialization import build_chat_response

    agent = direct_agent_execution(())
    assert agent.outcome == "direct"
    assert agent.planned_sources == ()
    assert agent.controller_executed is False
    assert agent.action_counts == {}
    assert (agent.terminal, agent.trajectory, agent.grounded, agent.timing) == (
        None, None, None, None), "a turn that ran no controller has no controller record"

    # What the chat layer attaches for a real direct turn: the graph answers, the plan is known,
    # and the record says plainly that nothing was recommended.
    chat = build_chat_response({"route": "direct", "final_response": "hello"},
                               session_id="s1", turn_id="s1:1", turn_number=1,
                               agent=direct_agent_execution(()))
    assert chat.agent is not None and chat.agent.outcome == "direct"
    assert chat.agent.controller_executed is False and chat.recommendations == ()


def test_the_five_outcomes_are_derived_from_the_runs_own_record() -> None:
    """``route`` cannot separate them; the outcome must, and only from execution state."""
    from recommendation.demo.agent_chat_adapter import agent_outcome
    from recommendation.demo.agent_schemas import AGENT_OUTCOMES

    assert set(AGENT_OUTCOMES) == {
        "direct", "recommended", "no_candidates", "refused", "failed"}

    # 1. the plan asked for no products: no controller ran.
    assert agent_outcome(agent_response(route="direct"), proposed_route="direct") == "direct"
    # 2. a controller ran and produced candidates.
    assert agent_outcome(
        agent_response(recommendations=[candidate(1, "B01", 11, 0.5)]),
        proposed_route="recommend") == "recommended"
    # 3. a controller ran, completed, and had nothing to present.
    assert agent_outcome(
        agent_response(recommendations=(), route="direct", succeeded=True),
        proposed_route="recommend") == "no_candidates"
    # 4. the completion guard refused the run - the controller's own reason, not an inference.
    assert agent_outcome(
        agent_response(recommendations=(), route="direct", succeeded=False,
                       termination_reason="completion_refused"),
        proposed_route="recommend") == "refused"
    # 5. a run that neither completed nor was refused (failure, invalid action, budget).
    for reason in ("execution_failed", "invalid_action", "max_tool_calls", "max_steps"):
        assert agent_outcome(
            agent_response(recommendations=(), route="direct", succeeded=False,
                           termination_reason=reason),
            proposed_route="recommend") == "failed", reason


def test_the_three_empty_looking_cases_are_actually_distinguishable() -> None:
    """The regression this field exists for: empty cards must not read as one outcome.

    All three turns below return ``route="direct"`` with an empty recommendation list.  Only the
    execution record separates the model's own direct choice from a recommendation the controller
    could not complete - and each answers in its own words instead of with silence.
    """
    from recommendation.demo.agent_chat_adapter import direct_agent_execution
    from recommendation.demo.serialization import build_chat_response

    chose_direct = build_chat_response({"route": "direct", "final_response": "hello"},
                                       session_id="s1", turn_id="s1:1", turn_number=1,
                                       agent=direct_agent_execution(()))
    refused = convert(
        agent_response(recommendations=(), route="direct", message="", succeeded=False,
                       termination_reason="completion_refused"),
        proposed_route="recommend")
    completed_empty = convert(
        agent_response(recommendations=(), route="direct", message="", succeeded=True,
                       termination_reason="completed"),
        proposed_route="recommend")

    assert [chat.agent.outcome for chat in (chose_direct, refused, completed_empty)] == [
        "direct", "refused", "no_candidates"]
    assert {chat.route for chat in (chose_direct, refused, completed_empty)} == {"direct"}
    assert all(chat.recommendations == () for chat in (chose_direct, refused, completed_empty))

    assert chose_direct.agent.controller_executed is False
    assert refused.agent.controller_executed is True
    assert completed_empty.agent.controller_executed is True
    assert refused.agent.terminal is not None
    assert refused.agent.terminal.termination_reason == "completion_refused"
    assert completed_empty.agent.terminal is not None
    assert completed_empty.agent.terminal.termination_reason == "completed"

    messages = {chose_direct.message, refused.message, completed_empty.message}
    assert len(messages) == 3 and all(messages), (
        f"each case must answer differently and in words: {messages}"
    )


def test_a_silent_failure_is_given_the_outcomes_own_answer() -> None:
    """An empty controller message is answered with an authored sentence, never with silence."""
    from recommendation.demo.agent_chat_adapter import OUTCOME_MESSAGES

    refused = convert(
        agent_response(recommendations=(), route="direct", message="", succeeded=False,
                       termination_reason="completion_refused"),
        proposed_route="recommend")
    assert refused.message == OUTCOME_MESSAGES["refused"] and refused.message
    assert OUTCOME_MESSAGES["refused"] != OUTCOME_MESSAGES["failed"]

    # A run that did render text keeps its own text: this fallback never overwrites the controller.
    spoken = convert(
        agent_response(recommendations=(), route="direct", message="nothing matched",
                       succeeded=False, termination_reason="completion_refused"),
        proposed_route="recommend")
    assert spoken.message == "nothing matched"


def test_the_execution_record_reports_actions_with_their_real_counts() -> None:
    """A source the controller consulted twice is visible as twice, and a refusal is not a run."""
    response = agent_response(
        recommendations=[candidate(1, "B01", 11, 0.5, sources=("catalog_search",),
                                   score_kind="bm25", title="Tent")],
        actions=("search_catalog", "search_catalog", "finish", "search_catalog"),
        steps=(step(0, "search_catalog", candidates=0),
               step(1, "search_catalog", candidates=1),
               step(2, "finish", candidates=1),
               step(3, "search_catalog", validated=False)),
        grounded_count=1, sources_present=("catalog_search",))
    agent = convert(response, proposed_route="recommend",
                    planned_sources=("catalog_search",)).agent
    assert agent is not None
    assert agent.outcome == "recommended"
    assert agent.planned_sources == ("catalog_search",), "the plan's own names, in plan order"
    assert agent.controller_executed is True
    assert agent.action_counts == {"search_catalog": 2, "finish": 1}, (
        "counts are executions; the refused proposal is not one"
    )
    assert agent.grounded is not None and agent.grounded.sources_present == ("catalog_search",)
    assert agent.terminal is not None and agent.terminal.run_id == "run-1"
    assert agent.trajectory is not None and len(agent.trajectory.steps) == 4
    card = convert(response, planned_sources=("catalog_search",)).recommendations[0]
    assert card.score_kind == "bm25", "the source's own score kind must survive"


def test_the_execution_record_round_trips_and_carries_no_prose() -> None:
    response = agent_response(recommendations=[candidate(1, "B01", 11, 0.5, title="Tent")],
                              actions=("recommend_from_history", "finish"),
                              steps=(step(0, "recommend_from_history", candidates=1),
                                     step(1, "finish", candidates=1)),
                              grounded_count=1)
    chat = convert(response, proposed_route="recommend", planned_sources=("history",))
    dumped = chat.model_dump_json()
    assert ChatResponse.model_validate_json(dumped) == chat
    blob = dumped.lower()
    for banned in ("prompt", "reasoning", "thought", "chain_of_thought", "traceback"):
        assert banned not in blob, banned
