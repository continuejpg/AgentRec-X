"""Phase 4: the additive policy wrappers and their wire-contract fields."""

from __future__ import annotations

from recommendation.control import ActionKind, ActionProposal, SearchCatalogArguments
from recommendation.demo.agent_policy import ClarificationPolicy, IntentQueryPolicy
from recommendation.demo.agent_schemas import (
    AgentClarificationView,
    AgentIntentView,
    AgentRecommendRequest,
    AgentRecommendResponse,
    AgentRecommendationView,
)


class _Context:
    def __init__(
        self,
        *,
        step: int = 0,
        available=(ActionKind.ASK_CLARIFICATION, ActionKind.FINISH),
        awaiting: bool = False,
        verified: int = 0,
        unresolved: int = 0,
        constraints=(),
    ) -> None:
        self.step_index = step
        self.available_actions = tuple(available)
        self.awaiting_user = awaiting
        self.verified_eligible_count = verified
        self.unresolved_count = unresolved
        self.task_constraints = tuple(constraints)

    def action_available(self, action) -> bool:
        return action in self.available_actions


class _Inner:
    def __init__(self, proposal: ActionProposal | None = None) -> None:
        self.calls = 0
        self._proposal = proposal or ActionProposal(action=ActionKind.FINISH, rationale="done")

    def choose(self, context):
        self.calls += 1
        return self._proposal


def test_a_conflict_question_is_asked_before_any_source_runs() -> None:
    inner = _Inner()
    policy = ClarificationPolicy(inner, question="Which budget should I keep?")
    proposal = policy.choose(_Context())
    assert proposal.action is ActionKind.ASK_CLARIFICATION
    assert proposal.arguments.question == "Which budget should I keep?"
    assert inner.calls == 0


def test_an_undecided_candidate_set_produces_one_question_not_a_success() -> None:
    inner = _Inner()
    policy = ClarificationPolicy(inner, ask_on_unresolved=True)
    proposal = policy.choose(
        _Context(unresolved=2, constraints=("price_max: prefers 20",))
    )
    assert proposal.action is ActionKind.ASK_CLARIFICATION
    assert "couldn't verify" in proposal.arguments.question
    assert "2 candidate(s)" in proposal.arguments.question


def test_a_verified_candidate_set_never_asks() -> None:
    inner = _Inner()
    policy = ClarificationPolicy(inner, ask_on_unresolved=True)
    proposal = policy.choose(_Context(verified=3, unresolved=1))
    assert proposal.action is ActionKind.FINISH
    assert inner.calls == 1


def test_only_one_question_is_asked_per_run() -> None:
    inner = _Inner()
    policy = ClarificationPolicy(inner, ask_on_unresolved=True)
    context = _Context(unresolved=2, constraints=("brand: prefers Acme",))
    assert policy.choose(context).action is ActionKind.ASK_CLARIFICATION
    context.step_index = 1
    assert policy.choose(context).action is ActionKind.FINISH


def test_the_intent_query_replaces_the_lexical_terms() -> None:
    inner = _Inner(
        ActionProposal(
            action=ActionKind.SEARCH_CATALOG,
            rationale="lexical",
            arguments=SearchCatalogArguments(terms=("old",)),
        )
    )
    policy = IntentQueryPolicy(inner, terms=("yoga mat", "camping"))
    proposal = policy.choose(_Context(available=(ActionKind.SEARCH_CATALOG, ActionKind.FINISH)))
    assert proposal.action is ActionKind.SEARCH_CATALOG
    assert proposal.arguments.terms == ("yoga mat", "camping")


def test_the_new_wire_fields_are_additive_and_optional() -> None:
    assert "clarification_id" in AgentRecommendRequest.model_fields
    for name in ("clarification", "intent"):
        assert name in AgentRecommendResponse.model_fields
    assert "reason" in AgentRecommendationView.model_fields
    assert AgentClarificationView(clarification_id="c1", question="q?").kind == "unverified"
    assert AgentIntentView().active_preference_count == 0
