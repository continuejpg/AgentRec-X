"""Run-level observation and the deterministic replan gate (V2 E-03A).

What this module is
-------------------
After a run has finished, this module answers two questions from the run's **own trusted record**:

1. *What actually happened?* - an :class:`~recommendation.demo.agent_schemas.AgentObservationView`
   built from the projected terminal, grounding, eligibility and trajectory records: which sources
   executed, how often, with which outcome; how many candidates were verified, how many were
   refused by the grounding verifier and how many are presented; the request's ``k``; the
   termination reason; and what budget is left.
2. *Is another retrieval worth attempting?* - a
   :class:`~recommendation.demo.agent_schemas.ReplanDecisionView` from the deterministic gate below.
   This round only **decides**; nothing here executes a second plan.

Why it cannot change the run
----------------------------
The observation is built **after** the controller has finished, from views that are already frozen,
and it only reads them.  It never calls a tool, never re-runs the planner, never touches a query and
never writes controller state, so the recommendations, their order, their scores, their provenance,
the preference evidence, the explanations and every other response field are exactly what they were.

What it deliberately does not do
--------------------------------
* It does not score relevance.  "Fewer than ``k`` candidates" is a count, and the observation says
  only that: the repository has no trustworthy relevance judge, so absence of quantity is never
  reported as absence of quality.
* It does not infer tool outcomes from prose.  Outcomes come from the per-step records the
  controller wrote (a failure code, a candidate count), never from ``ChatResponse.message``, a
  product title or a model's own account.
* It does not decide anything the deterministic rules already settle: safety refusals, exhausted
  budgets and exhausted sources are hard stops, and a model-driven gate (E-03B) may only choose
  inside what they leave open.

Reuse
-----
This adds no second control abstraction.  The per-step observations, the trajectory summary, the
grounding block, the eligibility block and the terminal record already exist; the run-level view is
their composition, and every field below names the record it came from.
"""

from __future__ import annotations

from typing import Any, Sequence

from .agent_schemas import AgentObservationView, ReplanDecisionView

__all__ = [
    "ACTION_SOURCES",
    "BLOCKING_TERMINATIONS",
    "DECISION_SOURCE_RULES",
    "REPLAN_REASON_CODES",
    "build_observation",
    "decide_replan",
    "observation_context",
]

#: Which trusted candidate source each candidate-producing action consults.
#:
#: ``select_source`` carries its source in the action's arguments, which the projected trajectory
#: does not publish, so it is reported under its own action name rather than guessed at - an
#: observation must not invent a source.  The two actions this deployment's plans compile to are
#: named exactly.
ACTION_SOURCES: dict[str, str] = {
    "recommend_from_history": "history",
    "search_catalog": "catalog_search",
    "find_similar": "similar_item",
}

#: The candidate-source vocabulary a name must belong to before it is called a source.
_SOURCE_NAMES = frozenset({"history", "catalog_search", "similar_item", "two_tower", "tiger"})

#: Terminations a replan must never route around: the controller stopped because a proposed action
#: was illegal, because the completion guard refused to end the run, because no legal action
#: existed, or because execution failed.  Suggesting another retrieval would be an attempt to
#: bypass the run's own safety decision, so the gate refuses regardless of budget.
BLOCKING_TERMINATIONS = frozenset(
    {
        "invalid_action",
        "completion_refused",
        "no_verified_candidate",
        "no_available_action",
        "execution_failed",
    }
)

#: The closed reason vocabulary of the gate.
REPLAN_REASON_CODES = (
    "not_a_recommendation",
    "blocked_by_termination",
    "budget_exhausted",
    "sources_exhausted",
    "candidates_sufficient",
    "tool_failed",
    "no_candidates",
    "candidates_below_k",
)

#: Where a decision came from.  Only the deterministic rules exist today; E-03B may add a validated
#: model decision, which must stay inside what these rules allow.
DECISION_SOURCE_RULES = "deterministic_rules"

#: Outcome of one source's steps, from the per-step records.
OUTCOME_CANDIDATES = "candidates"
OUTCOME_EMPTY = "empty"
OUTCOME_FAILED = "failed"


#: The candidate-producing actions whose source cannot be named from the projection alone.
#: ``select_source`` carries its source in the action's arguments, which the projected step does
#: not publish, so it is reported under its own action name rather than guessed at.
_UNNAMED_SOURCE_ACTIONS = frozenset({"select_source"})

#: Observations a candidate-producing step can record: the plane's ``candidate_set`` and the
#: accepted sequential capability's ``recommendation``.  Both carry ``returned_k``.
_CANDIDATE_OBSERVATIONS = frozenset({"candidate_set", "recommendation"})


def _is_candidate_action(action: str) -> bool:
    return action in ACTION_SOURCES or action in _UNNAMED_SOURCE_ACTIONS


def _step_facts(trajectory: Any) -> tuple[dict[str, int], dict[str, str]]:
    """Per-source call counts and outcomes, from the run's own per-step records.

    The projected step carries the action, the observation kind, the candidate count it observed and
    the refusal codes of its validation/verification.  Those facts separate the three cases that
    must never be conflated:

    * a **failed** tool - a refusal code, or a failure observation;
    * a tool that **succeeded and returned nothing** - a candidate observation with a zero count;
    * a tool that **returned candidates** - a candidate observation with a positive count.

    Only candidate-producing steps are counted as source calls; a later step for the same source
    wins, so its outcome describes where that source ended up.  A step the projection cannot
    attribute (no action) is skipped rather than attributed to a guess.
    """
    calls: dict[str, int] = {}
    outcomes: dict[str, str] = {}
    for step in getattr(trajectory, "steps", ()) or ():
        action = str(getattr(step, "action", "") or "")
        if not action or not _is_candidate_action(action):
            continue
        key = ACTION_SOURCES.get(action, action)
        calls[key] = calls.get(key, 0) + 1
        observation_type = str(getattr(step, "observation_type", "") or "")
        refusals = tuple(getattr(step, "refusal_codes", ()) or ())
        if refusals or observation_type == "failure":
            outcomes[key] = OUTCOME_FAILED
        elif observation_type in _CANDIDATE_OBSERVATIONS:
            outcomes[key] = (
                OUTCOME_CANDIDATES
                if int(getattr(step, "candidate_count", 0) or 0) > 0
                else OUTCOME_EMPTY
            )
    return calls, outcomes


def build_observation(
    *,
    terminal: Any,
    grounded: Any,
    eligibility: Any,
    trajectory: Any,
    plan: Sequence[str],
    available_sources: Sequence[str],
    requested_k: int,
    presented_count: int,
) -> AgentObservationView:
    """Compose this run's observation from records the controller already wrote.

    Every argument is a frozen view of the finished run (or a deployment fact), so this function is
    pure: it reads, derives counts, and returns.  ``requested_k`` and ``presented_count`` come from
    the request and the final recommendation list, which is what makes "below k" a statement about
    the answer rather than about retrieval.
    """
    calls, outcomes = _step_facts(trajectory)
    # First-seen order, so the observation reads in the order the run consulted them.  A source is
    # "consulted" because a candidate-producing step named it, and every consulted source keeps its
    # call count even when the projection could not classify its outcome.
    consulted = tuple(name for name in calls if name in _SOURCE_NAMES)
    available = tuple(str(name) for name in available_sources)
    unexecuted = tuple(name for name in available if name not in consulted)
    limits = dict(getattr(terminal, "limits", {}) or {})
    steps_limit = int(limits.get("max_steps", 0) or 0)
    tool_limit = int(limits.get("max_tool_calls", 0) or 0)
    steps_used = int(getattr(terminal, "steps", 0) or 0)
    tool_used = int(getattr(terminal, "tool_calls", 0) or 0)
    remaining_steps = max(0, steps_limit - steps_used)
    remaining_tools = max(0, tool_limit - tool_used)
    failed = tuple(name for name, outcome in outcomes.items() if outcome == OUTCOME_FAILED)
    empty = tuple(name for name, outcome in outcomes.items() if outcome == OUTCOME_EMPTY)
    grounded_count = int(getattr(grounded, "grounded_count", 0) or 0)
    return AgentObservationView(
        run_ref=str(getattr(terminal, "run_id", "") or ""),
        requested_k=int(requested_k),
        plan=tuple(str(name) for name in plan),
        executed_sources=consulted,
        unexecuted_sources=unexecuted,
        source_calls=calls,
        source_outcomes=outcomes,
        failed_sources=failed,
        empty_sources=empty,
        grounded_count=grounded_count,
        ungrounded_count=int(getattr(grounded, "ungrounded_count", 0) or 0),
        multi_source_count=int(getattr(grounded, "multi_source_count", 0) or 0),
        presented_count=int(presented_count),
        candidates_sufficient=int(presented_count) >= int(requested_k),
        termination_status=str(getattr(terminal, "status", "") or ""),
        termination_reason=(
            None
            if getattr(terminal, "termination_reason", None) is None
            else str(terminal.termination_reason)
        ),
        steps_used=steps_used,
        steps_limit=steps_limit,
        tool_calls_used=tool_used,
        tool_calls_limit=tool_limit,
        retries_used=int(getattr(terminal, "retries", 0) or 0),
        remaining_steps=remaining_steps,
        remaining_tool_calls=remaining_tools,
        has_further_budget=remaining_steps > 0 and remaining_tools > 0,
        eligibility_evaluated=bool(getattr(eligibility, "evaluated", False)),
        eligible_candidate_count=int(
            getattr(eligibility, "verified_eligible_count", 0) or 0
        ),
    )


def decide_replan(observation: AgentObservationView) -> ReplanDecisionView:
    """The deterministic gate: may another retrieval be attempted, and why?

    The rules are ordered, first match wins, and each one is a fact the observation carries:

    ======================================== ===================================================
    condition                                decision
    ======================================== ===================================================
    no source was consulted at all           ``not_a_recommendation`` - nothing to replan
    the run stopped on a safety refusal      ``blocked_by_termination`` - never routed around
    ``k`` products presented, no failure     ``candidates_sufficient`` - nothing to add
    no step or tool budget left              ``budget_exhausted``
    no other legal source remains            ``sources_exhausted``
    a consulted source failed                ``tool_failed`` - a different source may work
    nothing presented                        ``no_candidates``
    fewer than ``k`` presented               ``candidates_below_k``
    ======================================== ===================================================

    "Sufficient" is checked before the budget and source limits on purpose: an answer that already
    presents the requested ``k`` has nothing to add, so reporting a spent budget or an exhausted
    source list as its reason would understate it.  Those limits still decide every case where the
    answer is short, and a sufficiency claim is skipped whenever a consulted source failed (§4A
    conditions it on there being no real execution error).

    A decided-in-favour answer is a *suggestion* with a bounded scope: it is offered only when the
    run has budget left and a legal source it has not consulted, and it can never widen the tool
    budget, name a source the deployment does not offer, or undo a refusal.
    """
    eligible = tuple(observation.unexecuted_sources)
    base = {
        "eligible_sources": eligible,
        "remaining_tool_budget": observation.remaining_tool_calls,
        "remaining_step_budget": observation.remaining_steps,
        "decision_source": DECISION_SOURCE_RULES,
    }

    def decision(cont: bool, code: str, explanation: str) -> ReplanDecisionView:
        return ReplanDecisionView(
            continue_recommended=cont, reason_code=code, explanation=explanation, **base
        )

    if not observation.executed_sources:
        return decision(
            False,
            "not_a_recommendation",
            "No candidate source was consulted in this run, so there is nothing to replan.",
        )
    if observation.termination_reason in BLOCKING_TERMINATIONS:
        return decision(
            False,
            "blocked_by_termination",
            f"The run stopped on '{observation.termination_reason}'; another retrieval would "
            "attempt to route around that decision.",
        )
    if observation.candidates_sufficient and not observation.failed_sources:
        return decision(
            False,
            "candidates_sufficient",
            f"{observation.presented_count} product(s) are presented for a request of "
            f"{observation.requested_k}, and no consulted source failed.",
        )
    if not observation.has_further_budget:
        return decision(
            False,
            "budget_exhausted",
            "No step or tool-call budget remains, so another retrieval is not permitted.",
        )
    if not eligible:
        return decision(
            False,
            "sources_exhausted",
            "Every source this deployment offers has already been consulted.",
        )
    if observation.failed_sources:
        return decision(
            True,
            "tool_failed",
            "These sources failed: " + ", ".join(observation.failed_sources) + ".",
        )
    if observation.presented_count == 0:
        return decision(
            True,
            "no_candidates",
            "No verified candidate reached the answer, budget and untried sources remain.",
        )
    return decision(
        True,
        "candidates_below_k",
        f"{observation.presented_count} of {observation.requested_k} requested product(s) are "
        "presented; this counts products, not relevance.",
    )


def observation_context(
    observation: AgentObservationView, decision: ReplanDecisionView
) -> dict[str, Any]:
    """The bounded, disclosure-only projection a decision layer may see.

    Counts, source **names** the deployment already advertises, budgets and reason codes: no
    product identity, no title, no score, no query, no memory text, no store handle, and nothing
    from another session.  This is the whole channel - a later model-driven gate (E-03B) receives
    this and nothing else, which is why the observation itself (identities included) stays on the
    response rather than in a prompt.
    """
    return {
        "requested_k": observation.requested_k,
        "executed_sources": list(observation.executed_sources),
        "unexecuted_sources": list(observation.unexecuted_sources),
        "source_outcomes": dict(observation.source_outcomes),
        "grounded_count": observation.grounded_count,
        "ungrounded_count": observation.ungrounded_count,
        "presented_count": observation.presented_count,
        "candidates_sufficient": observation.candidates_sufficient,
        "termination_reason": observation.termination_reason,
        "remaining_tool_budget": observation.remaining_tool_calls,
        "remaining_step_budget": observation.remaining_steps,
        "continue_recommended": decision.continue_recommended,
        "reason_code": decision.reason_code,
        "decision_source": decision.decision_source,
    }
