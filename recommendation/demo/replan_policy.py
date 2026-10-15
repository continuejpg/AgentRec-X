"""Bounded replanning: one extra source, chosen by the model, inside the same run (V2 E-03B).

Where the decision happens, and why there
-----------------------------------------
A run's terminal record cannot be used to recover from an empty or short candidate set: by the time
the response exists, the accepted ``CompletionGuard`` has already refused the completion, and
E-03A's contract forbids routing around a refusal.  So the decision has to be taken **before** that
proposal is made, at the one place the controller already offers a post-tool decision: the
**policy**.  The loop asks its policy for the next action after every step, with the run's real
state in front of it, and this policy answers.

It is a decorator: :class:`DemoSourcePlanPolicy` still serves the authorised plan, in order, exactly
as before.  This class watches the run through the same projection and, when that plan is served,
may add **at most one** source the plan did not name - once per run, inside the same run, on the
same step and tool budgets - and then lets the plan's ``FINISH`` through.

What the gate reads, and from where
-----------------------------------
Everything comes from the controller's own projection of the run, which the policy receives on every
step:

* **consulted sources and their outcomes** - accumulated from the run's own step *observations*
  (``PolicyContext.last_observation``), which the controller writes after verifying each step.  A
  source counts as consulted because the controller recorded an observation for it, not because this
  policy intended to ask for it; a proposal that was refused produces no such record.
* **candidates held** - ``PolicyContext.candidate_state`` (the verified set the run holds) plus the
  best count any observed source reported, so a run whose most recent source was empty is not
  mistaken for a run that never had candidates.
* **budget** - ``remaining_steps`` and ``remaining_tool_calls``, and the rule that an appended
  source plus the ``FINISH`` that must follow it needs **two** steps: a run with one step left can
  terminate but cannot afford another source, so the gate stays shut.
* **legal sources** - the deployment's own plan vocabulary filtered to the actions the controller
  currently offers (``available_actions``), minus the ones already consulted.
* **safety** - a run whose previous proposal was rejected is not replanned: the controller refused
  something, and another retrieval is not a way around that.

What the model may decide
-------------------------
Nothing but *which* of the remaining legal sources to consult, or that none is worth it.  The
request it receives carries the current request, the value-level projection of the ACTIVE
preferences, a counts-and-names observation, and the eligible sources with the deployment's own
purpose strings.  Its answer is parsed and validated against that list; a name that is not eligible,
an unreadable answer, or a transport failure all mean "do not replan" - there is no retry, no
second opinion, and no way to widen a budget, invent a source, re-consult one already used, or ask
for another replan.  Whatever it answers, the resulting action still goes through the accepted
``ActionValidator``, the candidate plane and the grounding verifier like any other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from recommendation.control.schemas import PolicyActionError

from .agent_policy import SOURCE_ARGUMENTS, SOURCE_PLANS, SOURCE_PURPOSES, DemoSourcePlanPolicy

__all__ = [
    "BoundedReplanPolicy",
    "normalise_source",
    "REPLAN_STATUSES",
    "ReplanRecord",
    "build_replan_request_payload",
    "parse_replan_answer",
    "replanning_enabled",
]

#: Feature switch: bounded replanning is opt-in.
#:
#: Off by default.  With it off the controller is composed exactly as it was before this milestone
#: - same plan policy, no extra model call, no extra tool call - so every existing behaviour and
#: every existing response field is unchanged.
ENV_AGENT_REPLANNING = "AGENTRECX_AGENT_REPLANNING"


def replanning_enabled(value: Any = None) -> bool:
    """Whether bounded replanning is enabled for this deployment (default off)."""
    raw = os.environ.get(ENV_AGENT_REPLANNING) if value is None else value
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


#: The closed vocabulary of what happened, so a reader can never confuse the cases.
REPLAN_STATUSES = (
    "not_triggered",       # the run never needed it (the plan already delivered k)
    "gate_blocked",        # the deterministic rules refused (budget, sources, safety, ...)
    "model_declined",      # the model was asked and chose not to add a source
    "invalid_output",      # the model's answer did not satisfy the contract
    "model_failed",        # the model call itself failed
    "appended_candidates",  # a source was added and it produced candidates
    "appended_empty",      # a source was added and it produced none
    "appended_failed",     # a source was added and its execution failed
)

#: Gate reasons, reusing the E-03A vocabulary where the rule is the same.
GATE_SUFFICIENT = "candidates_sufficient"
GATE_NO_ELIGIBLE_SOURCE = "sources_exhausted"  # the E-03A name for "nothing legal is left"
GATE_NO_TOOL_BUDGET = "tool_budget_exhausted"
GATE_NOT_ENOUGH_STEPS = "not_enough_steps_to_finish"
GATE_AFTER_REFUSAL = "blocked_by_termination"
GATE_ALREADY_REPLANNED = "replan_already_used"
GATE_NO_SOURCE_STATE = "not_a_recommendation"
GATE_RUN_NOT_RUNNING = "run_not_running"
GATE_CANDIDATES_INELIGIBLE = "candidates_fail_hard_constraints"

_SYSTEM_PROMPT = """You decide whether one more trusted retrieval source is worth consulting.

A recommendation run has already consulted some sources, and you are shown what they returned. You
may add AT MOST ONE of the remaining legal sources, or decline.

Return JSON only:

{"replan": true, "source": "<one name from eligible_sources>", "reason": "<short factual reason>"}

or

{"replan": false, "source": null, "reason": "<short factual reason>"}

Rules:
1. "source" must be exactly one of the names in eligible_sources. Never invent a name, never name a
   source that is not listed, never ask for two.
2. Decline when the products already found are enough for the request, or when another source is
   unlikely to help. Declining is a correct answer, not a failure.
3. You cannot change any budget, tool, candidate, query or stored preference. You are choosing one
   retrieval source, nothing else.
4. Judge only from the observation given to you. Do not claim any source succeeded or failed beyond
   what the observation says, and do not describe products.
5. Output JSON only - no prose, no markdown."""


@dataclass(frozen=True)
class ReplanRecord:
    """What the replanning policy actually did in one run (for the response and for tests)."""

    status: str = "not_triggered"
    reason_code: str = ""
    model_called: bool = False
    chosen_source: str | None = None
    eligible_sources: tuple[str, ...] = ()
    consulted_sources: tuple[str, ...] = ()
    source_outcomes: dict[str, str] = field(default_factory=dict)
    initial_candidates: int = 0
    #: The same two facts as they stood before the appended source ran.
    initial_consulted_sources: tuple[str, ...] = ()
    initial_source_outcomes: dict[str, str] = field(default_factory=dict)
    final_candidates: int = 0
    remaining_tool_calls: int = 0
    remaining_steps: int = 0
    decision_step: int | None = None
    appended_outcome: str | None = None


@dataclass
class _RunState:
    """Run-scoped bookkeeping, rebuilt on the first decision of every run."""

    consulted: list[str] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)
    best_candidates: int = 0
    #: The step index at which the one allowed replan decision was taken (``None`` until then).
    decided_at: int | None = None
    #: The source whose observation the policy is waiting for after appending it.
    awaiting: str | None = None
    status: str = "not_triggered"
    reason_code: str = ""
    model_called: bool = False
    eligible: tuple[str, ...] = ()
    initial_candidates: int = 0
    #: The consulted sources and their outcomes **at the moment of the decision**, frozen before
    #: anything is appended: "what the first sources found" must stay its own fact.
    initial_consulted: tuple[str, ...] = ()
    initial_outcomes: dict[str, str] = field(default_factory=dict)
    chosen: str | None = None
    remaining_tool_calls: int = 0
    remaining_steps: int = 0


#: The action name a capability-based source observation carries, mapped to its plan name.
#: ``RecommendationObservation.source`` defaults to the action (``"recommend_from_history"``) while
#: a plane observation carries the ``CandidateSource`` value, which is already the plan name - so
#: this mapping is what keeps "consulted" comparable with the plan vocabulary.
_ACTION_SOURCES: dict[str, str] = {
    "recommend_from_history": "history",
    "search_catalog": "catalog_search",
    "find_similar": "similar_item",
}


def normalise_source(name: Any) -> str:
    """One source name, in the plan vocabulary, whatever record it came from."""
    value = str(name)
    return _ACTION_SOURCES.get(value, value)


def _source_argument(name: str) -> str:
    """The ``CandidateSource`` value a plan name selects (kept beside the plan vocabulary)."""
    from .agent_policy import SOURCE_ARGUMENTS

    return str(SOURCE_ARGUMENTS.get(name, name))


def _observation_source(observation: Any) -> str | None:
    """The trusted source a recorded observation names, in the plan vocabulary, or ``None``."""
    source = getattr(observation, "source", None)
    if not isinstance(source, str) or not source:
        return None
    return normalise_source(source)


def _observation_outcome(observation: Any) -> str | None:
    """``candidates`` / ``empty`` / ``failed`` for a candidate observation, else ``None``."""
    if observation is None:
        return None
    kind = str(getattr(observation, "kind", "") or "")
    status = str(getattr(observation, "status", "") or "")
    if kind == "failure" or status == "failed":
        return "failed"
    if kind in ("candidate_set", "recommendation"):
        if status == "empty":
            return "empty"
        count = int(getattr(observation, "returned_k", 0) or 0)
        return "candidates" if count > 0 else "empty"
    return None


def _preference_rows(projection: Any) -> list[dict[str, str]]:
    """The value-level rows of an ACTIVE preference projection (kind, polarity, value)."""
    rows: list[dict[str, str]] = []
    for entry in getattr(projection, "entries", ()) or ():
        rows.append(
            {
                "kind": str(getattr(getattr(entry, "kind", None), "value", entry.kind)),
                "polarity": str(
                    getattr(getattr(entry, "polarity", None), "value", entry.polarity)
                ),
                "value": str(entry.value),
            }
        )
    return rows


def build_replan_request_payload(
    *,
    context: Any,
    k: int,
    eligible: Sequence[str],
    consulted: Sequence[str],
    outcomes: dict[str, str],
    candidates: int,
    preferences: Any = None,
) -> dict[str, Any]:
    """The bounded, disclosure-only payload the source choice is made from.

    Counts, source names and the deployment's own purpose strings, plus the request and the
    value-level preference projection the controller already publishes to policies.  No product
    identity, no title, no score, no history, no store handle, no other session's memory.
    """
    # The controller publishes the turn's ACTIVE projection to a policy when it has memory bound;
    # an LLM-selected run deliberately binds none (its sources must not fold stored preferences into
    # a query), so the caller's own snapshot projection is used instead.  Either way this is the
    # value-level view only: no ``user_key``, no ``source_text``, no withdrawn entry.
    rows = _preference_rows(getattr(context, "active_preferences", None)) or _preference_rows(
        preferences
    )
    return {
        "user_request": str(getattr(context, "user_request", "")),
        "stated_preferences": rows,
        "observation": {
            "requested_k": int(k),
            "verified_candidates": int(candidates),
            "consulted_sources": list(consulted),
            "source_outcomes": dict(outcomes),
            "remaining_tool_calls": int(getattr(context, "remaining_tool_calls", 0) or 0),
            "remaining_steps": int(getattr(context, "remaining_steps", 0) or 0),
            "step_index": int(getattr(context, "step_index", 0) or 0),
        },
        "eligible_sources": [
            {"name": name, "purpose": SOURCE_PURPOSES.get(name, "trusted candidate source")}
            for name in eligible
        ],
        "instructions": (
            "Add at most one of the eligible sources, or decline. "
            "You cannot change budgets, tools, candidates or queries."
        ),
    }


def parse_replan_answer(text: Any, *, eligible: Sequence[str]) -> tuple[str | None, str]:
    """Parse and validate one answer; returns ``(chosen_source_or_None, code)``.

    Anything that is not exactly one eligible source name - an unreadable answer, a missing or
    unknown name, an empty list of sources - is a *decline* with a code, never an execution.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "invalid_output"
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    try:
        answer = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None, "invalid_output"
        try:
            answer = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None, "invalid_output"
    if not isinstance(answer, dict):
        return None, "invalid_output"
    if not bool(answer.get("replan")):
        return None, "model_declined"
    source = answer.get("source")
    if not isinstance(source, str) or source not in tuple(eligible):
        return None, "invalid_output"
    return source, "chosen"


class BoundedReplanPolicy:
    """Serve the authorised plan, then add at most one model-chosen legal source.

    The plan is served by an inner :class:`DemoSourcePlanPolicy`, unchanged; this policy only
    intercepts the ``FINISH`` that would end the run.*  ``client`` is a callable returning a
    ``StructuredModelClient`` (or ``None`` when this deployment has no model), ``gate`` is
    consulted for tests and observability, and ``max_replans`` is fixed at one by the milestone.
    """

    #: Recorded in the trajectory's policy metadata, so a run says which policy served it.
    name = "demo-source-plan+bounded-replan"

    def __init__(
        self,
        plan: Sequence[str],
        *,
        query_builder: Any = None,
        default_k: int = 5,
        use_memory: bool = False,
        client: Any = None,
        preferences: Any = None,
        max_replans: int = 1,
    ) -> None:
        self._inner = DemoSourcePlanPolicy(
            plan, query_builder=query_builder, default_k=default_k, use_memory=use_memory
        )
        self._client = client
        #: The caller's ACTIVE preference projection, used for the source-choice context when the
        #: controller itself publishes none (an LLM-selected run binds no memory to the loop).
        self._preferences = preferences
        self._max_replans = max(1, int(max_replans))
        self._plan = tuple(plan)
        self._default_k = int(default_k)
        self._state = _RunState()
        self._replans = 0
        self._call_count = 0
        self._pending_proposal: Any | None = None

    # -- what the run did, for the response --------------------------------- #

    @property
    def plan(self) -> tuple[str, ...]:
        return self._plan

    @property
    def replans_used(self) -> int:
        return self._replans

    @property
    def call_count(self) -> int:
        return self._call_count

    def record(self, *, final_candidates: int) -> ReplanRecord:
        """This run's replanning record, as the response reports it."""
        state = self._state
        status = state.status
        if state.chosen is not None and state.awaiting is not None:
            # The appended source was proposed and the run moved on without an observation for it:
            # its execution did not happen or did not survive validation.
            status = "appended_failed"
        if state.chosen is not None and state.awaiting is None:
            # The appended source's observation has been folded in: report what it produced.
            outcome = state.outcomes.get(state.chosen)
            status = {
                "candidates": "appended_candidates",
                "empty": "appended_empty",
                "failed": "appended_failed",
            }.get(str(outcome), "appended_empty")
        return ReplanRecord(
            status=status,
            reason_code=state.reason_code,
            model_called=state.model_called,
            chosen_source=state.chosen,
            eligible_sources=state.eligible,
            consulted_sources=tuple(state.consulted),
            source_outcomes=dict(state.outcomes),
            initial_candidates=state.initial_candidates,
            initial_consulted_sources=state.initial_consulted,
            initial_source_outcomes=dict(state.initial_outcomes),
            final_candidates=int(final_candidates),
            remaining_tool_calls=state.remaining_tool_calls,
            remaining_steps=state.remaining_steps,
            decision_step=state.decided_at,
            appended_outcome=(
                None if state.chosen is None else state.outcomes.get(state.chosen)
            ),
        )

    # -- the policy seam ---------------------------------------------------- #

    def choose(self, context: Any) -> Any:
        """Propose the plan's next unconsulted source, or one appended source, or ``FINISH``.

        Three things happen here, in order:

        1. the run's own last observation is folded into this run's record (what a source really
           did, not what this policy asked for);
        2. the inner plan policy is asked, and its proposal is passed through **when it names a
           source the run has not actually consulted yet** - which is the unchanged plan behaviour.
           A proposal to repeat an already-consulted source is not passed through: with candidates
           absent the accepted plan policy re-proposes the same source, which spends the whole tool
           budget retrying a source that already answered.  This policy treats "consulted" as the
           run's own record says, so the plan is finished instead;
        3. once the plan is served, gate and (at most once per run) let the model name one legal
           source to append.  Everything the controller checks still applies to that proposal.
        """
        self._call_count += 1
        if int(getattr(context, "step_index", 0) or 0) == 0:
            # The first decision of a fresh run: nothing has been consulted in *this* run yet.
            self._state = _RunState()
            self._replans = 0
            self._pending_proposal = None
        self._fold_observation(context)

        proposal = self._inner.choose(context)
        source = self._source_of(proposal)
        if source is not None and source not in self._state.consulted:
            return proposal

        if self._state.decided_at is None:
            appended = self._decide_and_maybe_append(context)
            if appended is not None:
                return appended
        return self._finish_proposal(context)

    def _source_of(self, proposal: Any) -> str | None:
        """Which trusted source a proposal names, or ``None`` when it proposes to finish.

        Read from the proposal the policy itself built: the action for the two directly mapped
        sources, and the arguments for ``SELECT_SOURCE``, whose whole purpose is to name one.  A
        name outside the plan vocabulary is reported as ``None`` rather than guessed at.
        """
        action = getattr(proposal, "action", None)
        value = str(getattr(action, "value", "") or "")
        if value in _ACTION_SOURCES:
            return _ACTION_SOURCES[value]
        arguments = getattr(proposal, "arguments", None)
        declared = getattr(arguments, "source", None)
        if declared is not None:
            name = normalise_source(getattr(declared, "value", declared))
            return name if name in SOURCE_PLANS else None
        return None

    def _finish_proposal(self, context: Any) -> Any:
        """The proposal that ends the run, built here when this policy has taken over the plan."""
        from recommendation.control import ActionKind, ActionProposal

        if not getattr(context, "action_available", lambda _a: False)(ActionKind.FINISH):
            raise PolicyActionError(
                "this policy has nothing left to propose and FINISH is not offered"
            )
        return ActionProposal(
            action=ActionKind.FINISH,
            rationale=(
                "the authorised plan is served"
                + ("" if self._replans == 0 else " and the one permitted additional source ran")
            ),
        )

    # -- internals ---------------------------------------------------------- #

    def _fold_observation(self, context: Any) -> None:
        """Record what the run's own last observation proved, before deciding anything."""
        observation = getattr(context, "last_observation", None)
        source = _observation_source(observation)
        if source is None:
            return
        if source not in self._state.consulted:
            self._state.consulted.append(source)
        outcome = _observation_outcome(observation)
        if outcome is not None:
            self._state.outcomes[source] = outcome
        count = int(getattr(observation, "returned_k", 0) or 0)
        if count > self._state.best_candidates:
            self._state.best_candidates = count
        if self._state.awaiting == source:
            self._state.awaiting = None

    def _candidates_held(self, context: Any) -> int:
        """Verified candidates the run holds: the controller's count, or the best observed.

        When the task carries a hard constraint and the controller has judged some candidates
        ineligible, only the *verified eligible* ones may count towards "enough": a product that
        violates the request is not an answer, so it cannot close the gate against a retrieval that
        might satisfy it.  With no hard constraint (``hard_constraint_count == 0``) nothing was
        evaluated and the verified count stands as it is - "not evaluated" is never read as
        "everything passed".
        """
        candidate_state = getattr(context, "candidate_state", None)
        held = max(
            int(getattr(candidate_state, "candidate_count", 0) or 0),
            self._state.best_candidates,
        )
        constraints = int(getattr(context, "hard_constraint_count", 0) or 0)
        ineligible = int(getattr(context, "ineligible_count", 0) or 0)
        if constraints > 0 and ineligible > 0:
            verified = int(getattr(context, "verified_eligible_count", 0) or 0)
            held = min(held, verified)
        return held

    def _eligible_sources(self, context: Any) -> tuple[str, ...]:
        """Legal sources this deployment offers, minus the ones already consulted."""
        available = set(getattr(context, "available_actions", ()) or ())
        eligible: list[str] = []
        for source in SOURCE_PLANS:
            action = SOURCE_PLANS[source]
            if action not in available:
                continue
            if source in self._state.consulted:
                continue
            eligible.append(source)
        return tuple(eligible)

    def _gate(self, context: Any) -> str | None:
        """The deterministic gate: ``None`` when a replan is allowed, else the blocking code.

        The caller guarantees this is the run's *first* decision (it records ``decided_at`` just
        before asking), so the only "already used" condition left is the real one: a source has
        actually been appended in this run.
        """
        if self._replans >= self._max_replans:
            return GATE_ALREADY_REPLANNED
        if str(getattr(getattr(context, "run_status", None), "value", "") or "") not in (
            "",
            "running",
        ):
            # Not a run that can act: nothing to replan, whatever the rest of the state says.
            return GATE_RUN_NOT_RUNNING
        if not self._state.consulted:
            return GATE_NO_SOURCE_STATE
        if bool(getattr(context, "last_proposal_rejected", False)):
            return GATE_AFTER_REFUSAL
        if self._candidates_held(context) >= self._default_k:
            return GATE_SUFFICIENT
        if int(getattr(context, "remaining_tool_calls", 0) or 0) < 1:
            return GATE_NO_TOOL_BUDGET
        if not self._eligible_sources(context):
            return GATE_NO_ELIGIBLE_SOURCE
        if int(getattr(context, "remaining_steps", 0) or 0) < 2:
            # One step can terminate a run but cannot both consult a source and finish it.
            return GATE_NOT_ENOUGH_STEPS
        return None

    def _decide_and_maybe_append(self, context: Any) -> Any | None:
        """Take the one allowed decision; return the appended source's proposal, or ``None``."""
        state = self._state
        state.decided_at = int(getattr(context, "step_index", 0) or 0)
        state.remaining_tool_calls = int(getattr(context, "remaining_tool_calls", 0) or 0)
        state.remaining_steps = int(getattr(context, "remaining_steps", 0) or 0)
        state.initial_candidates = self._candidates_held(context)
        # Freeze what the run had found *before* anything is appended: "the first sources' result"
        # and "the request's result" must never be read as the same fact.
        state.initial_consulted = tuple(state.consulted)
        state.initial_outcomes = dict(state.outcomes)
        eligible = self._eligible_sources(context)
        state.eligible = eligible

        blocked = self._gate(context)
        if blocked is not None:
            state.status = "not_triggered" if blocked == GATE_SUFFICIENT else "gate_blocked"
            state.reason_code = blocked
            return None
        if self._client is None:
            state.status = "gate_blocked"
            state.reason_code = "no_model_client"
            return None

        state.model_called = True
        chosen, code = self._ask_model(context, eligible=eligible)
        if chosen is None:
            state.status = (
                "model_declined" if code == "model_declined"
                else "model_failed" if code == "model_failed"
                else "invalid_output"
            )
            state.reason_code = code
            return None
        proposal = self._inner.proposal_for_source(chosen, context)
        if proposal is None:
            # The plan vocabulary cannot build this source: decline rather than invent.
            state.status = "invalid_output"
            state.reason_code = "not_in_plan_vocabulary"
            return None
        self._replans += 1
        state.chosen = chosen
        state.awaiting = chosen
        state.status = "appended_pending"
        state.reason_code = "appended_source_proposed"
        return proposal

    def _ask_model(self, context: Any, *, eligible: Sequence[str]) -> tuple[str | None, str]:
        """One model call, validated; never retried, never a second opinion."""
        from recommendation.control.model_client import ModelCallError, ModelRequest

        client = self._client() if callable(self._client) else self._client
        if client is None:
            return None, "no_model_client"
        request = ModelRequest(
            system_prompt=_SYSTEM_PROMPT,
            context_payload=build_replan_request_payload(
                context=context,
                k=self._default_k,
                eligible=eligible,
                consulted=self._state.consulted,
                outcomes=self._state.outcomes,
                candidates=self._state.initial_candidates,
                preferences=self._preferences,
            ),
        )
        try:
            response = client.complete(request)
        except ModelCallError:
            return None, "model_failed"
        except Exception:  # noqa: BLE001 - any transport problem is the same outcome
            return None, "model_failed"
        return parse_replan_answer(getattr(response, "text", ""), eligible=eligible)
