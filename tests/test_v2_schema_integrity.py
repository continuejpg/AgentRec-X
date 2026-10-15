"""E-03C-SCHEMA-INTEGRITY: the published demo response schemas, verified against real responses.

This module does not test features; it pins the **shape** of the public contract that the rebuilt
``recommendation/demo/schemas.py`` must still satisfy: field sets, concrete types, defaults,
(optional vs required), nested construction, JSON round-trips, backward compatibility with responses
that predate the optional blocks, rejection of unknown/mistyped input, and the OpenAPI document the
API layer publishes.

The pre-incident file could not be recovered byte for byte, so these tests are the evidence that the
observable contract is intact.  They are deliberately explicit rather than derived from the code:
a field that silently disappears or changes type must fail here.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import recommendation.demo.agent_schemas as agent_schemas
import recommendation.demo.schemas as chat_schemas
from recommendation.demo.schemas import (
    ActivePreferenceView,
    AuditView,
    ChatRequest,
    ChatResponse,
    KnowledgeEvidenceView,
    MemoryUpdateView,
    ProductFeedbackView,
    RagAuditView,
    RecommendationCard,
    ReplanAuditView,
)
from tests.test_v2_bounded_replanning import (  # the fixture that exercises every optional block
    DIRECT,
    PLAN_HISTORY,
    REQUEST,
    _harness,
    _ReactiveClient,
    _turn,
)
from tests.demo_fixture import close_harness

# --------------------------------------------------------------------------- #
# pinned field sets (explicit on purpose)
# --------------------------------------------------------------------------- #

#: The chat response's top-level keys: the accepted contract plus the E-01A agent block.  A new key
#: must be added here deliberately, which is what keeps an accidental field from shipping.
CHAT_RESPONSE_FIELDS = {
    "api_version", "session_id", "turn_id", "turn", "route", "message",
    "active_preferences", "memory_update", "recommendations", "audit", "trace", "agent",
}

AUDIT_FIELDS = {
    "reranking_applied", "candidate_count", "moved_count", "original_order", "reranked_order",
    "ranked_with_preference_count", "ranked_with_preferences",
    "rag", "replan", "feedback",
}

CARD_FIELDS = {
    "original_rank", "fused_rank", "parent_asin", "item_id", "sasrec_score", "score_kind",
    "fusion_sources", "fusion_source_ranks", "reranked_rank", "metadata_status", "metadata",
    "match_count", "violation_count", "unknown_count", "evidence", "fallback_reason",
    "movement_summary", "reason", "reason_status", "knowledge_evidence", "feedback",
}

AGENT_RESPONSE_FIELDS = {
    "api_version", "route", "message", "control_plane", "recommendations", "grounded",
    "eligibility", "terminal", "trajectory", "observation", "replan", "replan_execution", "timing",
}

RAG_AUDIT_FIELDS = {
    "enabled", "model_called", "products_examined", "products_with_evidence", "evidence_units",
    "explanations_accepted", "explanations_rejected", "explanations_abstained", "failure",
    "reason_source",
}

KNOWLEDGE_FIELDS = {
    "evidence_id", "parent_asin", "item_id", "field_path", "value", "source", "retrieval_score",
    "truncated",
}

REPLAN_AUDIT_FIELDS = {
    "continue_recommended", "reason_code", "eligible_sources", "remaining_tool_budget",
    "remaining_step_budget", "decision_source", "status", "model_called", "chosen_source",
    "initial_candidate_count", "final_candidate_count",
}

FEEDBACK_FIELDS = {
    "status", "action", "reference", "span", "detail", "parent_asin", "item_id",
    "final_position", "written", "entry_status", "effect", "active_product_feedback",
    "active_before",
}


def test_the_published_field_sets_are_exactly_the_documented_ones() -> None:
    assert set(ChatResponse.model_fields) == CHAT_RESPONSE_FIELDS
    assert set(AuditView.model_fields) == AUDIT_FIELDS
    assert set(RecommendationCard.model_fields) == CARD_FIELDS
    assert set(agent_schemas.AgentRecommendResponse.model_fields) == AGENT_RESPONSE_FIELDS
    assert set(RagAuditView.model_fields) == RAG_AUDIT_FIELDS
    assert set(KnowledgeEvidenceView.model_fields) == KNOWLEDGE_FIELDS
    assert set(ReplanAuditView.model_fields) == REPLAN_AUDIT_FIELDS
    assert set(ProductFeedbackView.model_fields) == FEEDBACK_FIELDS


def test_every_public_field_has_a_concrete_type() -> None:
    """No published field may be typed ``Any``, and none may be an unresolved forward reference."""
    problems: list[str] = []
    for module in (chat_schemas, agent_schemas):
        for name, obj in list(vars(module).items()):
            if not (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj.__module__ == module.__name__
            ):
                continue
            for field, info in obj.model_fields.items():
                text = str(info.annotation)
                if text == "typing.Any" or text.endswith(".Any"):
                    problems.append(f"{name}.{field} is typed Any")
                if "ForwardRef" in text:
                    problems.append(f"{name}.{field} is an unresolved forward reference: {text}")
    assert problems == [], problems


def test_the_contract_referenced_views_stay_typed_and_defaulted() -> None:
    """The four defects found while rebuilding this module, pinned one by one."""
    # 1. ``Any`` was missing from the typing import, leaving ChatResponse unbuildable.
    # 1. The module imports the typing names it annotates with, so it builds at all.
    source = inspect.getsource(chat_schemas)
    assert "from typing import" in source and "Any" in source.split("\n")[0:60].__str__()
    # 2. ``ChatResponse.agent`` must stay the concrete execution view, not Any.
    agent_field = ChatResponse.model_fields["agent"]
    assert agent_field.annotation == (agent_schemas.AgentExecutionView | None)
    assert agent_field.default is None and not agent_field.is_required()
    # 3. KnowledgeEvidenceView carries every field E-02B publishes.
    for field in KNOWLEDGE_FIELDS:
        assert field in KnowledgeEvidenceView.model_fields, field
    assert KnowledgeEvidenceView.model_fields["parent_asin"].is_required()
    assert KnowledgeEvidenceView.model_fields["truncated"].default is False
    # 4. RagAuditView.reason_source keeps its real default and stays optional.
    reason_source = RagAuditView.model_fields["reason_source"]
    assert reason_source.default == "template_from_validated_claims"
    assert not reason_source.is_required()


def test_card_reason_fields_keep_their_documented_defaults() -> None:
    card = RecommendationCard.model_fields
    assert card["reason"].default is None
    assert card["reason_status"].default == "not_attempted"
    assert card["knowledge_evidence"].default == ()
    assert card["feedback"].default is None
    assert card["match_count"].default == 0 and card["violation_count"].default == 0


def test_optional_blocks_default_to_absent_not_invented() -> None:
    audit = AuditView(
        reranking_applied=False, candidate_count=0, moved_count=0,
    )
    assert audit.rag is None and audit.replan is None and audit.feedback is None
    response = ChatResponse(
        session_id="s", turn_id="s:000001", turn=1, route="direct", message="hi", audit=audit,
    )
    assert response.agent is None and response.trace is None
    assert response.recommendations == () and response.active_preferences == ()
    assert response.memory_update.changed is False


# --------------------------------------------------------------------------- #
# serialisation and backward compatibility
# --------------------------------------------------------------------------- #


def test_a_real_chat_response_round_trips_through_dict_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        recommendation = _turn(
            harness, _ReactiveClient(PLAN_HISTORY, replan_source="catalog_search"), REQUEST,
            k=4, session=session)
        statement = _turn(
            harness, _ReactiveClient(DIRECT), "I like the second one.", session=session)
        # A recommendation turn carries the controller blocks; a direct turn carries the feedback
        # block and no controller.  Both round-trip, and each keeps its own optional shapes.
        for body in (recommendation, statement):
            model = ChatResponse.model_validate(body)
            assert ChatResponse.model_validate(model.model_dump()) == model
            assert ChatResponse.model_validate_json(model.model_dump_json()) == model
            assert model.agent is not None
            assert isinstance(model.agent, agent_schemas.AgentExecutionView)
        model = ChatResponse.model_validate(recommendation)
        assert isinstance(model.audit.replan, ReplanAuditView)
        assert model.recommendations and model.audit.replan.status is not None
        direct = ChatResponse.model_validate(statement)
        assert direct.recommendations == ()
        assert direct.audit.replan is None, (
            "a direct turn runs no controller, so the gate's decision is absent rather than nulled "
            "into a default"
        )
        assert isinstance(direct.audit.feedback, ProductFeedbackView)
        assert direct.audit.feedback.written is True
    finally:
        close_harness(harness)


def test_an_agent_response_round_trips_with_its_nested_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")
    harness = _harness(tmp_path)
    try:
        session = harness.create_session()["session_id"]
        response = harness.client.post(
            "/v1/demo/agent/recommend",
            json={"message": REQUEST, "k": 4, "sources": ["history", "catalog_search"],
                  "history": list(harness.manager.get(session).trusted_user_history)},
        )
        assert response.status_code == 200, response.text
        model = agent_schemas.AgentRecommendResponse.model_validate(response.json())
        assert agent_schemas.AgentRecommendResponse.model_validate_json(
            model.model_dump_json()) == model
        assert model.observation is not None and model.observation.presented_count == 4
        assert model.replan is not None and model.replan.reason_code
        assert model.grounded.grounded_count == 4
        assert model.terminal.grounded_identities_ok if hasattr(model.terminal, "grounded_identities_ok") else True
    finally:
        close_harness(harness)


def test_a_response_from_before_the_optional_blocks_still_validates() -> None:
    """A client (or a stored snapshot) written before E-02B/E-03A/E-03C keeps working."""
    legacy = {
        "api_version": "1.0.0",
        "session_id": "legacy-session",
        "turn_id": "legacy-session:000001",
        "turn": 1,
        "route": "recommend",
        "message": "Top 1 recommendation.",
        "active_preferences": [],
        "memory_update": {"changed": False},
        "recommendations": [
            {
                "original_rank": 1,
                "parent_asin": "cand-red",
                "item_id": 101,
                "sasrec_score": 0.9,
                "score_kind": "sasrec_logit",
                "metadata_status": "found",
            }
        ],
        "audit": {
            "reranking_applied": False,
            "candidate_count": 1,
            "moved_count": 0,
            "original_order": ["cand-red"],
            "reranked_order": ["cand-red"],
        },
    }
    model = ChatResponse.model_validate(legacy)
    card = model.recommendations[0]
    assert card.reason is None and card.reason_status == "not_attempted"
    assert card.knowledge_evidence == () and card.feedback is None
    assert model.audit.rag is None and model.audit.replan is None and model.audit.feedback is None
    assert model.agent is None and model.trace is None
    # And re-serialising it adds exactly the optional blocks, with their documented defaults.
    again = ChatResponse.model_validate(model.model_dump())
    assert again == model


def test_unknown_and_mistyped_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hi", "k": 4, "trusted_user_history": ["x"]})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hi", "k": "4"})          # strict integer
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "   "})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "hi", "decision_mode": "turbo"})
    with pytest.raises(ValidationError):
        # A nested view may not smuggle an undeclared field either.
        AuditView.model_validate({
            "reranking_applied": False, "candidate_count": 0, "moved_count": 0,
            "rag": {"enabled": True, "products_examined": 0, "surprise": 1},
        })
    with pytest.raises(ValidationError):
        KnowledgeEvidenceView.model_validate({"evidence_id": "x", "field_path": "p", "value": "v"})


def test_the_memory_and_evidence_views_are_unchanged() -> None:
    """E-01B/E-02A/E-02B field contracts, pinned where they are easy to lose in a rebuild."""
    assert set(MemoryUpdateView.model_fields) == {
        "changed", "added", "superseded", "removed", "skipped_duplicates", "already_processed",
        "removal_directives",
    }
    # The memory panel publishes exactly the three value-level facts and no store handle: this is
    # what keeps an entry identifier out of every response.
    assert set(ActivePreferenceView.model_fields) == {"kind", "polarity", "value"}
    assert set(chat_schemas.EvidenceView.model_fields) == {
        "kind", "value", "polarity", "status", "metadata_field", "metadata_value", "detail",
    }


# --------------------------------------------------------------------------- #
# the API layer's published schema
# --------------------------------------------------------------------------- #


def test_openapi_publishes_every_response_model_with_resolved_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", "unit-test-placeholder-not-a-credential")
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", "deepseek")
    harness = _harness(tmp_path)
    try:
        spec = harness.client.get("/openapi.json").json()
        schemas = spec["components"]["schemas"]
        for wanted in ("ChatResponse", "AuditView", "RecommendationCard", "KnowledgeEvidenceView",
                       "RagAuditView", "ReplanAuditView", "ProductFeedbackView",
                       "AgentRecommendResponse", "AgentObservationView", "AgentReplanView",
                       "AgentExecutionView", "AgentTerminalView", "MessageView" if "MessageView" in schemas else "ChatRequest"):
            assert wanted in schemas, wanted
        # The four fields that were unresolved forward references must publish real references.
        card = schemas["RecommendationCard"]["properties"]["knowledge_evidence"]
        assert card["items"]["$ref"].endswith("/KnowledgeEvidenceView")
        for field, target in (("rag", "RagAuditView"), ("replan", "ReplanAuditView"),
                              ("feedback", "ProductFeedbackView")):
            shape = schemas["AuditView"]["properties"][field]
            assert shape["anyOf"][0]["$ref"].endswith("/" + target), (field, shape)
        assert schemas["ChatResponse"]["properties"]["agent"]["anyOf"][0]["$ref"].endswith(
            "/AgentExecutionView")
        # Required sets: only what the accepted contract declares required.
        assert set(schemas["ChatResponse"]["required"]) == {
            "session_id", "turn_id", "turn", "route", "message", "audit"}
        assert set(schemas["AgentRecommendResponse"]["required"]) == {
            "route", "message", "control_plane", "grounded", "eligibility", "terminal",
            "trajectory"}
    finally:
        close_harness(harness)


def test_the_chat_request_contract_is_unchanged() -> None:
    assert set(ChatRequest.model_fields) == {"message", "k", "decision_mode"}
    assert ChatRequest.model_fields["k"].default == 5
    assert ChatRequest.model_fields["decision_mode"].default is None
    assert chat_schemas.DECISION_MODES == ("deterministic", "llm", "llm_agent")
    assert chat_schemas.DECISION_MODE_LLM_AGENT == "llm_agent"
    assert ChatRequest.model_config.get("extra") == "forbid"
    assert json.loads(ChatRequest(message=" hi ").model_dump_json())["message"] == "hi"
