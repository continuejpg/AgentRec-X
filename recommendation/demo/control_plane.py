"""Opt-in 2.0-alpha control plane for the local demo (AgentRec-X 2.0-alpha).

What this module is
-------------------
The accepted demo path is the Milestone 7B-10D DAG
(:class:`~recommendation.agent.graph.AgentGraph`).  AgentRec-X 2.0-alpha adds a second
control plane - the bounded agent loop - and this module lets the local demo run on it
**without changing the HTTP contract, the accepted default, or any recommendation
behaviour**.

Selection is explicit and defaults to the accepted path::

    AGENTRECX_CONTROL_PLANE=graph   # default: the accepted M7B-10D DAG
    AGENTRECX_CONTROL_PLANE=loop    # 2.0-alpha: the bounded agent loop

Why the loop is opt-in rather than the new default
--------------------------------------------------
Stage 1's rule is *change who decides the next step, not how a recommendation is made*.
Making the loop the default in the same change that introduces it would conflate the two:
a regression in the new control plane would then look like a regression in recommendations.
The default therefore stays the tested DAG, the loop is exercised explicitly (unit tests,
the control-plane tests and an HTTP equivalence test), and promoting it is a separate,
deliberate decision.

`k` and the decision model
--------------------------
The accepted trust boundary routes ``k`` exclusively through the decision object, which is
why the DAG caches one compiled graph per ``(k, user_key)``.  The loop keeps that property
from the other side: the demo's decision model still chooses the route *text* and ``k``, and
the runner turns that choice into the loop's initial policy configuration.  A policy cannot
invent a ``k`` that the request did not ask for, because the validator re-checks it against
the accepted Tool range and only the request-scoped model supplies it.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from recommendation.agent import AgentGraphState, AgentInput, DecisionModel
from recommendation.control import (
    LoopController,
    LoopLimits,
    LoopResult,
    RecommendFromHistoryCapability,
    RuleBasedPolicy,
)
from recommendation.memory import PreferenceMemoryService
from recommendation.preference_matching import PreferenceCandidateMatcher
from recommendation.rag import ProductEnricher
from recommendation.reranking import PreferenceReranker
from recommendation.tools import RecommendationTool

__all__ = [
    "CONTROL_PLANE_ENV_VAR",
    "CONTROL_PLANE_GRAPH",
    "CONTROL_PLANE_LOOP",
    "DEFAULT_CONTROL_PLANE",
    "DEFAULT_LOOP_LIMITS",
    "DemoLoopRunner",
    "resolve_control_plane",
]

#: Environment variable that selects the demo's control plane.
CONTROL_PLANE_ENV_VAR = "AGENTRECX_CONTROL_PLANE"

#: The accepted Milestone 7B-10D DAG.  The default.
CONTROL_PLANE_GRAPH = "graph"

#: The 2.0-alpha bounded agent loop.  Opt-in.
CONTROL_PLANE_LOOP = "loop"

#: Default control plane: the accepted, already-accepted path.
DEFAULT_CONTROL_PLANE = CONTROL_PLANE_GRAPH

#: Loop budgets for the demo.  Small and finite on purpose: the loop exists to prove it is
#: bounded, and the canonical turn needs exactly two steps (recommend, then finish).
DEFAULT_LOOP_LIMITS = LoopLimits(max_steps=6, max_tool_calls=4, max_retries=1)


def resolve_control_plane(configured: str | None = None) -> str:
    """Resolve the demo's control plane from configuration, defaulting to the DAG.

    An unrecognised value is rejected rather than silently ignored: a typo in an
    environment variable must not quietly select a different control plane.
    """
    value = configured if configured is not None else os.environ.get(CONTROL_PLANE_ENV_VAR)
    if value is None or not str(value).strip():
        return DEFAULT_CONTROL_PLANE
    normalised = str(value).strip().lower()
    if normalised not in (CONTROL_PLANE_GRAPH, CONTROL_PLANE_LOOP):
        raise ValueError(
            f"{CONTROL_PLANE_ENV_VAR} must be '{CONTROL_PLANE_GRAPH}' or "
            f"'{CONTROL_PLANE_LOOP}', got {value!r}"
        )
    return normalised


class DemoLoopRunner:
    """Run the demo's turns on the bounded agent loop, exposing the DAG's state shape.

    The runner is deliberately a *runnable*, not a graph: it shares the graph's public
    surface (``invoke``) and returns the same
    :class:`~recommendation.agent.state.AgentGraphState` mapping, so the accepted demo
    serializer and the HTTP layer need no knowledge of which control plane produced a turn.
    """

    def __init__(
        self,
        *,
        decision_model: DecisionModel,
        tool: RecommendationTool,
        enricher: ProductEnricher,
        matcher: PreferenceCandidateMatcher,
        reranker: PreferenceReranker,
        memory_service: PreferenceMemoryService,
        user_key: str,
        limits: LoopLimits = DEFAULT_LOOP_LIMITS,
    ) -> None:
        self._decision_model = decision_model
        self._tool = tool
        capability = RecommendFromHistoryCapability(
            tool,
            product_enricher=enricher,
            preference_matcher=matcher,
            preference_reranker=reranker,
        )
        self._policy = _DecisionModelPolicy(decision_model)
        self._controller = LoopController(
            self._policy,
            capability,
            memory_service=memory_service,
            user_key=user_key,
            limits=limits,
        )
        self._last_result: LoopResult | None = None

    # -- metadata ---------------------------------------------------------- #

    @property
    def decision_model(self) -> DecisionModel:
        """The injected decision model (exposed for lifecycle inspection/tests)."""
        return self._decision_model

    @property
    def tool(self) -> RecommendationTool:
        """The accepted Recommendation Tool this runner drives."""
        return self._tool

    @property
    def policy(self) -> Any:
        """The Stage 1 policy the loop consults."""
        return self._policy

    @property
    def controller(self) -> LoopController:
        """The loop controller (exposed for inspection/tests)."""
        return self._controller

    @property
    def limits(self) -> LoopLimits:
        """The loop's termination budgets."""
        return self._controller.limits

    @property
    def control_plane(self) -> str:
        """Which control plane this runner implements."""
        return CONTROL_PLANE_LOOP

    @property
    def last_result(self) -> LoopResult | None:
        """The most recent loop result, for diagnostics and tests."""
        return self._last_result

    # -- invocation -------------------------------------------------------- #

    def invoke(self, agent_input: AgentInput) -> AgentGraphState:
        """Run one turn and return the accepted state shape.

        Raises
        ------
        PolicyActionError
            The input is not a validated :class:`AgentInput`.
        MalformedDecision
            Propagated from the loop when the decision model produced nothing usable.
        RecommendationToolError
            Propagated unchanged from the accepted Tool.
        """
        result = self._controller.invoke(agent_input)
        self._last_result = result
        if not result.succeeded and result.status.value == "failed":
            # A failed loop run is a backend failure for the same reasons a failed DAG run
            # is; the API layer maps it identically.  The engine already recorded the
            # reason, so raise the same class the DAG would.
            from recommendation.agent import AgentGraphError

            raise AgentGraphError(
                f"the agent loop failed: {result.control.termination_reason}"
            )
        return result.state


class _DecisionModelPolicy:
    """Adapt the demo's decision model into a Stage 1 policy.

    The decision model remains the authority on *routing text* and ``k`` - the accepted
    trust boundary routes ``k`` through it - while the loop's policy interface is what the
    controller drives.  A model that cannot decide raises, and the loop terminates
    deterministically rather than defaulting to an action.
    """

    name = "demo-decision-model"

    def __init__(self, decision_model: DecisionModel) -> None:
        self._decision_model = decision_model
        #: The most recent decision, exposed so the loop's finalizer can branch on what the
        #: policy actually decided (in particular, render a direct turn's reply text).
        self.last_decision: Any = None

    def choose(self, context: Any) -> Any:
        """Turn the decision model's route choice into one action proposal.

        Two mapping rules, both deliberate:

        **One decision per run.**  The accepted DAG consults its decision model exactly once
        per turn, and the demo must keep that contract: a "recommend" turn means *one*
        recommendation round, not a search loop.  The policy therefore proposes ``FINISH``
        once a verified candidate set exists for the run, which is what makes the loop's
        behaviour identical to the DAG's for the same decision model rather than a
        multi-round variant of it.

        **Map onto the offered action space.**  The model's choice is translated into the
        actions the **system** currently offers.  When the model asks for a recommendation
        the run cannot execute - the tool-call budget is spent, or a candidate set is
        already grounded - the policy proposes ``FINISH`` rather than an action that would
        be refused.  That is a policy decision inside the policy's own remit, and it is
        documented here so it cannot be mistaken for the controller silently rewriting a
        proposal.

        A real LLM policy for a later stage would be free to choose differently: the
        one-round rule lives in *this* demo policy, not in the control plane.
        """
        from recommendation.agent import build_decision_messages
        from recommendation.agent.decision import AgentAction
        from recommendation.control import ActionKind, ActionProposal

        # One decision per run: the DAG's contract, preserved.
        if context.candidate_state.grounded:
            self.last_decision = None
            return ActionProposal(
                action=ActionKind.FINISH,
                rationale="this run has already produced its recommendation",
            )

        decision = self._decision_model.decide(
            build_decision_messages(context.user_request)
        )
        self.last_decision = decision
        wants_recommendation = decision.action is AgentAction.RECOMMEND
        if wants_recommendation and context.action_available(
            ActionKind.RECOMMEND_FROM_HISTORY
        ):
            return ActionProposal(
                action=ActionKind.RECOMMEND_FROM_HISTORY,
                k=decision.requested_k,
                rationale="the decision model selected the recommendation route",
            )
        return ActionProposal(
            action=ActionKind.FINISH,
            rationale=(
                "the decision model asked to recommend, but the system no longer offers "
                "that action for this run"
            )
            if wants_recommendation
            else "the decision model selected the direct route",
        )
