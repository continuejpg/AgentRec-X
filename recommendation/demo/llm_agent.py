"""LLM tool selection executed by the bounded agent loop (V2 E-01A-3B).

One chat turn, in order::

    user message
      -> ToolSelectionPlanner.plan            (one provider call, validated plan)
      -> DemoAgentService.run_selection       (exactly the planned sources, in plan order)
      -> LoopController -> ActionValidator -> CandidatePlane -> GroundingVerifier
         -> CandidateLedger
      -> AgentRecommendResponse               (real products, real provenance, a digest)

This module is **composition, not policy**: the planner, the controller, the plane, the
verifier, the ledger and the fusion are the accepted components; what is added here is the
binding between them for a request that arrives over the chat endpoint.

What the model may decide
-------------------------
Exactly one thing per turn: which of the trusted candidate sources this deployment can serve
*right now* to consult, in which order - or that the turn needs no product search at all.  The
plan schema (:class:`~recommendation.demo.tool_selection.SourceSelection`) has no field for a
product identity, a score, a rank, an evidence verdict or a preference, and ``extra="forbid"``
makes an attempt to carry one a validation error rather than a stored fact.  Retrieval, masking,
grounding, ranking, fusion and the response text all stay in trusted code.

The offered menu is read, never assumed
---------------------------------------
:func:`offered_sources` derives the menu on every turn from
:attr:`~recommendation.demo.agent_service.DemoAgentService.available_sources`, so a source this
deployment cannot serve right now - a TIGER worker before its handshake, an item-item index the
deployment did not opt into - is simply not in the menu and is therefore unreachable, while a
source that becomes available needs no code change to be offered.  The advertised set is then
restricted to the names the accepted plan vocabulary can execute
(:data:`~recommendation.demo.agent_policy.SOURCE_PLANS`), so every validated selection executes:
a name the deployment advertises but the plan vocabulary cannot name (today ``similar_item``,
which the controller's own action menu reaches through ``FIND_SIMILAR``) is not offered, because
offering a tool that execution would refuse is not a choice.  The budget is the controller's own
bound (:attr:`~recommendation.demo.agent_service.DemoAgentService.tool_budget`), so the model is
never asked for a plan the loop could not run.

Preference memory: read as context, never merged into the request
---------------------------------------------------------------
The session's own ACTIVE preferences reach this path as **structured context** for the tool
decision (``planner_preference_context``), not as extra query text.  The two channels stay
separate on purpose:

* the current request travels as ``user_request`` and is the only text a lexical search runs on
  (``use_memory=False`` in
  :meth:`~recommendation.demo.agent_service.DemoAgentService.run_selection`), so a stored
  preference can never be spliced into the shopper's own words and passed off as this turn's
  intent;
* the stored preferences travel as ``context.active_preferences``: for each ACTIVE entry only its
  kind, polarity and value, capped - no ``user_key``, no ``memory_id``, no ``source_text``, no turn
  id, no timestamp, and nothing REMOVED or superseded.  An empty memory sends no context at all.

The run itself is still **memory-free** (it executes in the agent service's no-``user_key`` mode),
so nothing in this path ranks candidates with preferences yet; making the ranking memory-aware is a
later milestone's delivery, not something to fake here.

Why the *run* stays memory-free
-------------------------------
The write happens in the runtime, once per turn, through the accepted M9 service keyed by the
session's own namespace and the server-owned turn id - not inside the loop.  Three facts decide
that, and every one is a property of the accepted contracts rather than a preference:

* :class:`~recommendation.demo.agent_schemas.AgentRecommendResponse` carries no preference write, so
  a chat response built from a *loop*-written mutation could not report it truthfully - and
  reporting "nothing changed" after a change is exactly the fabricated audit this layer must never
  produce;
* :class:`~recommendation.demo.agent_schemas.AgentRecommendRequest` carries no session turn id, so
  a loop-side write would key its idempotency on a message-derived id instead of this session's
  turn;
* an LLM-selected run is deliberately the isolated path for memory *influence*, and that isolation
  is what keeps this turn's retrieval answerable to this turn's request.

A *direct* turn is different by design: it runs no controller and is answered by the accepted
direct capability, whose memory behaviour (load, extract, persist, report) is unchanged - and is
the only place a direct turn's preferences are written, so nothing is submitted twice.

The direct route starts no controller
-------------------------------------
``route="direct"`` means the model decided that this turn needs no product search.  No plan is
executed and no controller is built: the turn goes to the accepted direct capability - the same
component the deterministic mode answers a greeting or a preference statement with.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .agent_policy import SOURCE_PLANS, SOURCE_PURPOSES
from .agent_schemas import MAX_HISTORY_ITEMS, AgentRecommendRequest, AgentRecommendResponse
from .agent_service import DemoAgentError, DemoAgentService
from .tool_selection import ROUTE_RECOMMEND, SourceSelection, ToolOption, ToolSelectionPlanner

__all__ = ["LLMAgentTurn", "MAX_PLANNER_PREFERENCES", "offered_sources",
           "planner_preference_context"]

#: How many stored preferences one tool-selection request may carry.
#:
#: Bounded deliberately: the prompt is proportional to what a *tool choice* can depend on, not to
#: how much the shopper has ever stated.  The store's own order (kind, then value, deterministic)
#: decides which ones fit, and the count of the rest is still reported.
MAX_PLANNER_PREFERENCES = 10


def planner_preference_context(projection: Any | None) -> dict[str, Any]:
    """The memory part of a tool-selection request, as structured data.

    ``projection`` is the same restricted ACTIVE-preference projection the controller hands a
    policy (:func:`recommendation.memory.query_adapter.project_active_preferences`); this function
    narrows it once more, to what choosing a *tool* can depend on:

    * ``kind`` / ``polarity`` / ``value`` per preference - and nothing else: no ``user_key``, no
      ``memory_id``, no ``source_text``, no turn id, no timestamp, and only ACTIVE entries, because
      the projection already excludes everything else;
    * at most :data:`MAX_PLANNER_PREFERENCES` entries, in the projection's own order;
    * an explicit count, so a capped list is never mistaken for the whole picture.

    Returns an empty mapping for no memory, so a session with nothing stored sends exactly the
    request it sent before this round.
    """
    entries = tuple(getattr(projection, "entries", ()) or ())
    if not entries:
        return {}
    return {
        "active_preferences": [
            {
                "kind": str(getattr(entry.kind, "value", entry.kind)),
                "polarity": str(getattr(entry.polarity, "value", entry.polarity)),
                "value": str(entry.value),
            }
            for entry in entries[:MAX_PLANNER_PREFERENCES]
        ],
        "active_preference_count": len(entries),
    }


def offered_sources(available_sources: Sequence[str]) -> tuple[ToolOption, ...]:
    """The menu of tools this deployment can actually serve, described for the model.

    ``available_sources`` is the caller's runtime-derived set, in its own order; this function
    neither adds nor reorders a name.  A name the accepted plan vocabulary cannot execute is left
    out (see the module docstring), and each remaining name is described by
    :data:`~recommendation.demo.agent_policy.SOURCE_PURPOSES` - a retrieval capability, never a
    product fact.
    """
    return tuple(
        ToolOption(name=str(name), description=SOURCE_PURPOSES[str(name)])
        for name in available_sources
        if str(name) in SOURCE_PLANS
    )


class LLMAgentTurn:
    """One chat turn: planned by the model, executed by the controller.

    Mirrors :class:`~recommendation.demo.llm_runtime.LLMTurnRun`'s contract for this path - the
    plan is obtained once, before anything runs, and is reused rather than re-asked - with one
    difference that is the point of this mode: the plan names **tools**, not a route plus
    preference mutations.

    Parameters
    ----------
    planner:
        The validated tool-selection planner, holding the one process-scoped provider client.
    service:
        The runtime's single-turn agent service: it owns the controller cache, so a turn reuses
        a compiled controller instead of rebuilding the plane.
    k:
        The candidate budget for the run.  Request-scoped and never model-chosen: the plan
        decides *which* sources run, not how many candidates they may return.
    user_message:
        The turn's user text - the only thing the model reads.
    history:
        The session's trusted history, supplied to the run exactly as the session holds it.
    preference_context:
        The session's ACTIVE preferences, already narrowed by :func:`planner_preference_context`.
        Passed to the planner as structured context (never as query text); an empty mapping sends
        no context, which is what a session with nothing stored sends.
    """

    def __init__(
        self,
        planner: ToolSelectionPlanner,
        service: DemoAgentService,
        *,
        k: int,
        user_message: str,
        history: Sequence[str],
        preference_context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(user_message, str) or not user_message.strip():
            raise DemoAgentError(
                "a turn with no user message cannot select tools", code="invalid_request"
            )
        self._planner = planner
        self._service = service
        self._k = int(k)
        self._user_message = user_message
        self._history = tuple(str(item) for item in history)
        self._preference_context = dict(preference_context or {})
        self._selection: SourceSelection | None = None

    # -- introspection ----------------------------------------------------- #

    @property
    def k(self) -> int:
        """The candidate budget this turn asks each source for."""
        return self._k

    @property
    def user_message(self) -> str:
        """The one user message this turn was planned for."""
        return self._user_message

    @property
    def menu(self) -> tuple[ToolOption, ...]:
        """The tools this turn offers the model (recomputed from the service's live set)."""
        return offered_sources(self._service.available_sources)

    @property
    def preference_context(self) -> dict[str, Any]:
        """The stored preferences this turn discloses to the planner (empty when there are none)."""
        return dict(self._preference_context)

    @property
    def provider(self) -> str | None:
        """Provider declared by the client that answered, or ``None`` when it declares none."""
        from .llm_runtime import provider_facts

        return provider_facts(self._planner.client)[0]

    @property
    def model(self) -> str | None:
        """Model identifier declared by the client that answered, or ``None``."""
        from .llm_runtime import provider_facts

        return provider_facts(self._planner.client)[1]

    # -- the turn ---------------------------------------------------------- #

    def plan(self) -> SourceSelection:
        """Ask the model once, validate the answer, and cache it for this turn.

        The request it builds carries the current request text, the offered tools, the budget and -
        when the session has stored preferences - the memory context from
        :func:`planner_preference_context`.  The preferences are a separate field, so the planner
        can tell what the shopper asked for *now* from what they stated earlier, and no stored
        value ever becomes part of the request text.

        Raises
        ------
        ToolSelectionError
            The answer was refused (an unoffered tool, a duplicate, an over-budget or
            self-contradictory plan) or the provider call failed.  Nothing has run at this point,
            and no failure path substitutes a default source list.
        """
        if self._selection is None:
            self._selection = self._planner.plan(
                user_request=self._user_message,
                allowed_sources=self.menu,
                budget=self._service.tool_budget,
                context=self._preference_context or None,
            )
        return self._selection

    def request(self) -> AgentRecommendRequest:
        """The stateless agent request this turn executes.

        Built from the session's own history, with no ``user_key``: see the module docstring for
        why this path is memory-free.  A history the request schema cannot carry is refused here,
        loudly, rather than by an opaque validation error deeper in the stack.

        ``sources`` keeps the schema's default on purpose.  The plan is *not* the request body's
        business: :meth:`~recommendation.demo.agent_service.DemoAgentService.run_selection`
        executes the validated selection and re-checks availability itself, so writing the plan
        here could only suggest that the body decided the run.
        """
        if not self._history:
            raise DemoAgentError(
                "this session has no trusted history for an agent run", code="invalid_history"
            )
        if len(self._history) > MAX_HISTORY_ITEMS:
            raise DemoAgentError(
                f"the session history ({len(self._history)} items) exceeds the agent request "
                f"bound of {MAX_HISTORY_ITEMS}",
                code="invalid_history",
            )
        return AgentRecommendRequest(
            message=self._user_message, history=list(self._history), k=self._k
        )

    def run(
        self, *, preference_snapshot: Any = None, replan_client: Any = None
    ) -> AgentRecommendResponse:
        """Execute the validated selection through the service's own controller.

        Only a ``recommend`` selection reaches here: a direct turn is answered by the accepted
        direct capability and must never start a recommendation controller, so asking this method
        for one is an error rather than an empty run.

        ``preference_snapshot`` is the *trusted caller's* read of this session's M9 memory, passed in
        at execution time rather than when the turn was composed.  That separation is deliberate:
        what the model is told (``preference_context``) and what the trusted ranking may use are two
        different decisions, and only the caller that owns the session namespace can make the
        second one.  The model never names a ``user_key`` and never supplies this value.
        """
        selection = self.plan()
        if selection.route != ROUTE_RECOMMEND:
            raise DemoAgentError(
                "a direct selection does not start a recommendation run",
                code="not_a_recommendation",
            )
        return self._service.run_selection(
            selection,
            self.request(),
            preference_snapshot=preference_snapshot,
            # The bounded replanning client (E-03B), when the deployment enabled it: the service
            # composes the run's policy with it, so the decision happens inside this same run.
            replan_client=replan_client,
        )
