"""Turn-scoped wiring that makes the optional LLM decision policy usable in the demo.

This module is **composition, not policy**.  The policy core
(:mod:`recommendation.demo.llm_policy`) already turns one user message into one validated
:class:`~recommendation.demo.llm_policy.LLMTurnPlan`; this module binds that plan to the
trusted execution path the browser demo already uses:

    one user message
      -> LLMTurnPlanner.plan                 (one provider call, validated plan)
      -> PlanDecisionModel                   (route: the plan's route, nothing else)
      -> PreferenceMemoryService(LLM plan)   (the SAME plan, no second call)
      -> accepted AgentGraph                 (Tool, enricher, matcher, reranker, M9 store)

Why the plan is obtained once, before anything runs
---------------------------------------------------
A turn is planned **before** any trusted component executes.  Three properties follow, and
all three are deliberate:

* a provider failure cannot leave a partial mutation behind, because no trusted component
  has run yet;
* the route and the preference extraction come from the *same* validated plan, so the demo
  makes exactly one provider call per turn (a second call could return a different answer
  and would be billed twice);
* there is nothing to fall back to.  A failed plan raises, and the HTTP layer reports a
  failed turn - the session never silently runs the deterministic path while the browser
  shows "LLM Agent".

Authority
---------
The plan schema has no field for a product id, a score, evidence, a rank or a mutation
result, and ``extra="forbid"`` makes an attempt to add one a validation error.  What this
module lets the model decide is therefore exactly two things: which of the two accepted
routes the turn takes, and which preference *values* are stored or withdrawn.  Retrieval,
scoring, masking, matching, reranking, persistence and every metric stay in trusted code.

The route is a real decision, not a decoration: it is handed to the accepted graph through
the accepted decision-model seam (the channel that already carries ``k``), which is why the
trace reports it as the model's *proposed* route while ``route`` reports what the run
actually did.
"""

from __future__ import annotations

from typing import Any, Sequence

from recommendation.agent import AgentDecision, DecisionMessage
from recommendation.control.model_client import ModelCallError
from recommendation.memory import PreferenceMemoryService
from recommendation.memory.schemas import PreferenceExtraction

from .decision import DIRECT_RESPONSE_TEXT
from .llm_policy import (
    LLM_EXTRACTOR_NAME,
    LLMPolicyError,
    LLMTurnPlan,
    LLMTurnPlanner,
    RouteAction,
    plan_to_extraction,
)
from .schemas import DECISION_MODE_LLM, TracePreferenceAction

__all__ = [
    "DECISION_MODE_LLM",
    "LLM_MAX_ATTEMPTS",
    "LLMTurnRun",
    "PlanDecisionModel",
    "PlanPreferenceExtractor",
    "build_llm_planner",
    "provider_facts",
]

#: Bounded retries for unparseable provider output, matching the policy core's default.
#: A transport failure is never retried here: the policy raises it immediately so a failed
#: call cannot be mistaken for a decision.
LLM_MAX_ATTEMPTS = 2


def build_llm_planner(client_factory: Any = None) -> LLMTurnPlanner:
    """Build the planner over the existing provider seam.

    ``client_factory`` is an injection point for tests and for a deployment that supplies its
    own transport; production uses :func:`~recommendation.control.provider_adapter.
    build_provider_client`, which reads ``AGENTRECX_LLM_*`` at construction time.

    An unconfigured provider is reported as a policy-layer failure with the stable
    ``not_configured`` code, so every caller sees one error type rather than two.
    """
    if client_factory is not None:
        client = client_factory()
    else:
        from recommendation.control.provider_adapter import build_provider_client

        try:
            client = build_provider_client()
        except ModelCallError as exc:
            raise LLMPolicyError(
                "no provider is configured for the LLM decision mode",
                code="not_configured",
            ) from exc
    return LLMTurnPlanner(client, max_attempts=LLM_MAX_ATTEMPTS)


def provider_facts(client: Any) -> tuple[str | None, str | None]:
    """``(provider, model)`` declared by the client that answered, or ``(None, None)``.

    Read from the client object that actually made the call, never assumed from
    configuration: an adapter built for DeepSeek reports DeepSeek, an injected test double
    that declares no identity reports ``None``, and the trace then says so instead of
    guessing.  No credential is readable through this path.
    """
    profile = getattr(client, "profile", None)
    name = getattr(profile, "name", None)
    if not isinstance(name, str) and isinstance(profile, str):
        name = profile
    model = getattr(client, "model", None)
    return (
        name if isinstance(name, str) and name.strip() else None,
        model if isinstance(model, str) and model.strip() else None,
    )


class LLMTurnRun:
    """One turn planned by the LLM policy layer.

    The run owns the turn's single plan.  :meth:`plan` performs the provider call the first
    time it is needed and returns the cached validated plan afterwards, so the decision model
    and the extractor share one answer.
    """

    def __init__(self, planner: LLMTurnPlanner, *, k: int, user_message: str) -> None:
        if not isinstance(user_message, str) or not user_message.strip():
            raise LLMPolicyError("a turn with no user message cannot be planned", code="empty_message")
        self._planner = planner
        self._k = int(k)
        self._user_message = user_message
        self._plan: LLMTurnPlan | None = None
        self._planned_for: str | None = None
        self._provider_calls = 0

    # -- introspection ----------------------------------------------------- #

    @property
    def k(self) -> int:
        """The candidate count this turn asks the Tool for (request-scoped, not model-chosen)."""
        return self._k

    @property
    def user_message(self) -> str:
        """The one user message this turn was planned for."""
        return self._user_message

    @property
    def provider_calls(self) -> int:
        """How many provider calls this turn actually made (diagnostic; never a trace field)."""
        return self._provider_calls

    @property
    def client(self) -> Any:
        """The structured-model client the plan came from."""
        return self._planner.client

    @property
    def provider(self) -> str | None:
        """Provider name declared by the client, or ``None`` when it declares none."""
        return provider_facts(self.client)[0]

    @property
    def model(self) -> str | None:
        """Model identifier declared by the client, or ``None`` when it declares none."""
        return provider_facts(self.client)[1]

    @property
    def proposed_route(self) -> str:
        """The route the model proposed (``recommend`` or ``direct``)."""
        return self.plan().route.value

    # -- the one plan ------------------------------------------------------ #

    def plan(self) -> LLMTurnPlan:
        """Return this turn's validated plan, calling the provider at most once.

        Raises
        ------
        LLMPolicyError
            The provider could not be reached, or the model never returned a valid plan.
            Nothing has been executed when this raises.
        """
        if self._plan is None:
            self._provider_calls += 1
            self._plan = self._planner.plan(self._user_message)
            self._planned_for = self._user_message
        return self._plan

    def plan_for(self, user_message: str) -> LLMTurnPlan:
        """Return the plan for ``user_message``, refusing a message this turn did not plan.

        Trusted code hands the extractor the message recorded on the run's own state.  A
        different message would mean the wiring is about to apply one turn's plan to another
        turn's text, so it is refused rather than re-planned: re-planning would be a second
        provider call, and silently applying the wrong plan would be worse.
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise LLMPolicyError(
                "this turn has no user message to extract from", code="empty_message"
            )
        plan = self.plan()
        if user_message != self._planned_for:
            raise LLMPolicyError(
                "a different user message reached this planned turn", code="turn_mismatch"
            )
        return plan

    # -- the accepted seams ------------------------------------------------ #

    def decision_model(self) -> "PlanDecisionModel":
        """The accepted decision-model seam, carrying the plan's route and the request's ``k``."""
        return PlanDecisionModel(self)

    def memory_service(self, store: Any) -> PreferenceMemoryService:
        """A memory service whose extraction is the already-validated plan.

        The accepted Milestone 9 service owns validation and persistence, so a preference the
        model proposed travels the same validate-then-apply path as a rule-extracted one.
        """
        return PreferenceMemoryService(store, PlanPreferenceExtractor(self))

    def preference_actions(self) -> tuple[TracePreferenceAction, ...]:
        """The plan's validated preference actions, for the trace.

        Withdrawals are listed before additions, which is how a substitution reads: what was
        dropped, then what was gained.  Within each group the plan's own order is preserved.

        ``applied`` is left ``False`` here on purpose: whether an action actually changed
        memory is decided by the accepted memory write, and the serializer fills it in from
        that persisted result.  A proposal is never reported as a change on its own.
        """
        plan = self.plan()
        actions = [
            TracePreferenceAction(action="remove", value=value, kind=None)
            for value in plan.remove
        ]
        actions.extend(
            TracePreferenceAction(action="add", value=item.value, kind=item.kind.value)
            for item in plan.add
        )
        return tuple(actions)


class PlanDecisionModel:
    """Present one validated plan's route through the accepted decision-model seam.

    This is the *only* channel through which the LLM decision mode can affect a run, and what
    it can express is bounded by the accepted :class:`~recommendation.agent.AgentDecision`
    contract: one of the two routes this demo already had, plus the request's ``k``.  The
    direct route's reply text is the demo's accepted text, not model prose.
    """

    def __init__(self, run: LLMTurnRun) -> None:
        self._run = run
        self.calls: list[tuple[DecisionMessage, ...]] = []

    @property
    def k(self) -> int:
        """The candidate count passed through to the Tool."""
        return self._run.k

    @property
    def call_count(self) -> int:
        """How many decisions were produced (diagnostics only)."""
        return len(self.calls)

    def decide(self, messages: Sequence[DecisionMessage]) -> AgentDecision:
        """Return the plan's route as an accepted decision.

        The messages are recorded for inspection only and are never re-sent to the provider:
        the plan was made before the graph started, from the same user message the graph is
        about to pass in.
        """
        self.calls.append(tuple(messages))
        plan = self._run.plan()
        if plan.route is RouteAction.RECOMMEND:
            return AgentDecision(action="recommend", k=self._run.k)
        return AgentDecision(action="direct_response", direct_response=DIRECT_RESPONSE_TEXT)


class PlanPreferenceExtractor:
    """The accepted extractor interface, backed by this turn's validated plan.

    ``name`` is the LLM provenance label the policy layer already defines, so entries written
    through this path are distinguishable in the accepted memory audit trail.
    """

    def __init__(self, run: LLMTurnRun) -> None:
        self._run = run

    @property
    def name(self) -> str:
        """Extractor label recorded on produced memory entries."""
        return LLM_EXTRACTOR_NAME

    def extract(self, user_message: str) -> PreferenceExtraction:
        """Return the plan's preferences and removals for this turn's message."""
        return plan_to_extraction(self._run.plan_for(user_message), source_text=user_message)
