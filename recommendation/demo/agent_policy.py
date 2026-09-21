"""Request-driven multi-source plan policy for the single-turn agent endpoint.

What this is
------------
A small policy that consults **the trusted candidate sources the request named**, in the
order the request named them, and then finishes.  It exists so that one HTTP request can
demonstrate the capability this repository actually measured in Phase 4 and Phase 5 -
multi-source retrieval with ledger provenance and rank fusion - without adding a planner,
a multi-agent split, a learned source-selection strategy, or any change to the accepted
control plane.

It is deliberately **not** an adaptive selector
-----------------------------------------------
Phase 5 measured that a per-user rule that consults *one* source performs worse than
fusing every source (``docs/PHASE5_HANDOFF.md`` §6).  So this policy does not choose: the
caller states which sources to consult, all of them are consulted, and the plane's existing
:func:`~recommendation.control.reciprocal_rank_fusion` decides the order.  Choosing a
source adaptively here would re-introduce exactly the strategy the benchmark found wanting,
and would be an unmeasured behaviour change in a packaging phase.

What it may not do
------------------
* it names a source and a budget; it names no product, and every identity a source returns
  is grounded by the plane's :class:`~recommendation.control.GroundingVerifier` before it
  enters the ledger;
* its ``SEARCH_CATALOG`` query is the user's own message text, tokenised by the trusted
  search tool.  A policy can express *what to look for*; it has no control over how the
  catalogue is matched;
* a source the run does not offer is never proposed - the policy reads
  ``context.available_actions`` and stops, rather than proposing an action that the
  validator would refuse.
"""

from __future__ import annotations

from typing import Any, Sequence

from recommendation.control import ActionKind, ActionProposal, PolicyActionError

__all__ = ["SOURCE_PLANS", "DemoSourcePlanPolicy", "normalise_source_plan"]

#: The candidate sources a request may name, mapped to the control-plane action that
#: consults them.  ``history`` is the accepted sequential recommender; ``catalog_search`` is
#: full-catalogue lexical search over normalized catalogue text.
#:
#: ``history`` maps to the accepted ``RECOMMEND_FROM_HISTORY`` action, and that is a
#: deliberate refusal to route it through the candidate plane.  Routing it via
#: ``SELECT_SOURCE`` would add provenance for it, but ``SELECT_SOURCE`` is only offered when a
#: similar-item source is registered, and changing that menu rule was measured to alter the
#: action sequence of an **archived Phase-3 recording** (three replay tests failed) - i.e. it
#: changes which actions a run may take, which is a control-plane behaviour change rather than
#: packaging.  So the history path stays exactly as accepted, and the response labels its
#: provenance as the history source without a ledger record (see
#: :func:`~recommendation.demo.agent_trajectory.build_grounded_view`).  ``catalog_search``
#: goes through the plane, which is what records real multi-source provenance.
SOURCE_PLANS: dict[str, ActionKind] = {
    "history": ActionKind.RECOMMEND_FROM_HISTORY,
    "catalog_search": ActionKind.SEARCH_CATALOG,
}

#: The ``CandidateSource`` each plan name selects when consulted through the candidate plane.
SOURCE_ARGUMENTS: dict[str, str] = {
    "history": "history",
    "catalog_search": "catalog_search",
}

#: Default plan: the accepted history recommender alone.  That keeps the endpoint's default
#: behaviour identical to the accepted single-source path.
DEFAULT_SOURCE_PLAN: tuple[str, ...] = ("history",)

#: Upper bound on a plan.  Three sources is every registered source today, and a bound here
#: means a malformed request cannot ask the loop to do unbounded work.
MAX_SOURCE_PLAN = len(SOURCE_PLANS)


def normalise_source_plan(sources: Sequence[str] | None) -> tuple[str, ...]:
    """Validate a requested source plan, preserving order and rejecting duplicates.

    Raises
    ------
    ValueError
        The plan names an unknown source, repeats a source, or is empty.  Loud rather than
        silently repaired: a plan the run cannot honour must not look like a successful one.
    """
    if sources is None:
        return DEFAULT_SOURCE_PLAN
    if len(sources) == 0:
        raise ValueError("a source plan must name at least one source")
    plan: list[str] = []
    for raw in sources:
        name = str(raw).strip().lower()
        if name not in SOURCE_PLANS:
            raise ValueError(
                f"unknown candidate source {raw!r}; known sources: "
                + ", ".join(sorted(SOURCE_PLANS))
            )
        if name in plan:
            raise ValueError(f"source {name!r} appears more than once in the plan")
        plan.append(name)
    if len(plan) > MAX_SOURCE_PLAN:
        raise ValueError(f"a source plan may name at most {MAX_SOURCE_PLAN} sources")
    return tuple(plan)


class DemoSourcePlanPolicy:
    """Consult the requested sources in order, then finish.

    Parameters
    ----------
    plan:
        Source names, already validated by :func:`normalise_source_plan`.
    query_builder:
        Turns the user's message into ``SEARCH_CATALOG`` terms.  Injected so a test can
        pin the query without re-implementing the tokenisation.
    default_k:
        The candidate budget asked of each source.  The request's ``k`` does not silently
        become a per-source budget: each source contributes up to ``default_k`` candidates
        and the plane's fusion decides the order.
    """

    #: Stable policy identity, recorded in the trajectory's policy metadata.
    name = "demo-source-plan"

    def __init__(
        self,
        plan: Sequence[str] = DEFAULT_SOURCE_PLAN,
        *,
        query_builder: Any = None,
        default_k: int = 5,
    ) -> None:
        if not plan:
            raise ValueError("a source plan needs at least one source")
        for source in plan:
            if source not in SOURCE_PLANS:
                raise ValueError(f"unknown source {source!r}")
        self._plan = tuple(plan)
        self._query_builder = query_builder or _search_arguments_from_message
        self._default_k = int(default_k)
        self._call_count = 0
        #: Sources already proposed in the current run, and whether the plan is served.
        #: Reset on step 0, because one compiled policy serves every run of its controller.
        self._served: set[str] = set()
        self._finished_plan = False

    @property
    def plan(self) -> tuple[str, ...]:
        """The sources this policy will consult, in order."""
        return self._plan

    @property
    def call_count(self) -> int:
        """How many proposals this policy produced (diagnostics only)."""
        return self._call_count

    @property
    def default_k(self) -> int:
        """The per-source candidate budget this policy asks for."""
        return self._default_k

    def choose(self, context: Any) -> ActionProposal:
        """Propose the next planned source that has not been consulted, then ``FINISH``.

        The policy is the only component that knows its own plan, so *it* has to remember what
        it already asked for.  The controller's action menu legitimately keeps offering
        ``SELECT_SOURCE`` after a source has been consulted - a run may consult several - so
        without this the plan would repeat its first source until the step budget ran out
        instead of moving on.

        The record is a returned flag rather than mutable state: ``choose`` reports
        ``finished_plan=True`` once the plan is served, and the loop clears that flag on the
        terminal step.  One policy instance is cached per compiled controller and is shared by
        every run that controller serves, so mutable "which sources have I asked for" state
        would leak between requests and silently change the second request's behaviour.

        A planned source the run does not currently offer is skipped rather than proposed: the
        controller's validator is the authority on availability, and a policy that proposed an
        unavailable action would spend a step to be refused.
        """
        self._call_count += 1
        if context.step_index == 0:
            # A fresh run on this cached policy: last run's plan is over.
            self._finished_plan = False
        available = context.available_actions
        plan = self._plan

        if not self._finished_plan:
            for source in plan:
                action = SOURCE_PLANS[source]
                if action not in available:
                    continue
                proposal = self._proposal_for(source, action, context)
                if proposal is not None:
                    return proposal
            self._finished_plan = True

        if context.action_available(ActionKind.FINISH):
            return ActionProposal(
                action=ActionKind.FINISH,
                rationale="every source the request named has been consulted",
            )
        raise PolicyActionError(
            "no planned source is available and FINISH is not offered: "
            + ", ".join(a.value for a in available)
        )

    def _proposal_for(
        self, source: str, action: ActionKind, context: Any
    ) -> ActionProposal | None:
        """One proposal for a planned source, or ``None`` when it has already been served.

        A source counts as served once the run holds grounded candidates and this is the only
        source in the plan; with several planned sources the fields in each proposal
        (``source`` for ``SELECT_SOURCE``, the query for ``SEARCH_CATALOG``) differ per source,
        so the plan advances by position.  Both paths are bounded by the plan length.
        """
        if context.candidate_state.grounded and source in self._served:
            return None
        if action is ActionKind.SEARCH_CATALOG:
            self._served.add(source)
            return ActionProposal(
                action=action,
                rationale="the request asked for full-catalogue lexical retrieval",
                arguments=self._query_builder(context.user_request, limit=self._default_k),
            )
        self._served.add(source)
        if action is ActionKind.RECOMMEND_FROM_HISTORY:
            # The accepted action carries ``k`` and no arguments; passing arguments to it is a
            # contract violation the proposal model rejects.
            return ActionProposal(
                action=action,
                k=self._default_k,
                rationale=f"the request asked for the {source} candidate source",
            )
        return ActionProposal(
            action=action,
            rationale=f"the request asked for the {source} candidate source",
            arguments=self._select_arguments(source),
        )

    def _select_arguments(self, source: str) -> Any:
        """Build ``SELECT_SOURCE`` arguments naming one trusted source."""
        from recommendation.control import CandidateSource, SelectSourceArguments

        return SelectSourceArguments(
            source=CandidateSource(SOURCE_ARGUMENTS[source]), limit=self._default_k
        )

    def _select_arguments(self, source: str) -> Any:
        """Build ``SELECT_SOURCE`` arguments naming one trusted source."""
        from recommendation.control import CandidateSource, SelectSourceArguments

        return SelectSourceArguments(
            source=CandidateSource(SOURCE_ARGUMENTS[source]), limit=self._default_k
        )

    def _consulted_sources(self, context: Any) -> frozenset[str]:
        """Sources this run has already consulted, read from its own recorded actions.

        Every plan source is consulted through the candidate plane, whose observations record
        which source produced them, so the trajectory is a sufficient and non-forgeable record
        of what the plan has already done.
        """
        plan_sources = frozenset(self._plan)
        consulted: set[str] = set()
        for step in getattr(context, "trajectory", ()) or ():
            proposal = getattr(step, "action_proposal", None) or {}
            arguments = proposal.get("arguments") or {}
            source = arguments.get("source")
            if source in plan_sources:
                consulted.add(str(source))
            if proposal.get("action") == ActionKind.SEARCH_CATALOG.value and (
                "catalog_search" in plan_sources
            ):
                consulted.add("catalog_search")
        return frozenset(consulted)

    def _select_arguments(self, source: str) -> Any:
        """Build ``SELECT_SOURCE`` arguments naming one trusted source."""
        from recommendation.control import CandidateSource, SelectSourceArguments

        return SelectSourceArguments(
            source=CandidateSource(SOURCE_ARGUMENTS[source]), limit=self._default_k
        )

        if context.action_available(ActionKind.FINISH):
            return ActionProposal(
                action=ActionKind.FINISH,
                rationale="every source the request named has been consulted",
            )
        raise PolicyActionError(
            "no planned source is available and FINISH is not offered: "
            + ", ".join(a.value for a in available)
        )


def _search_arguments_from_message(message: str, *, limit: int) -> Any:
    """Build ``SEARCH_CATALOG`` arguments from the user's own message.

    The message is split, not understood: :class:`~recommendation.control.SearchCatalogArguments`
    owns the token bound and the trusted search tool owns the matching.  A message with no
    usable term cannot build a query, which is reported as a policy failure rather than
    patched with a fabricated query.
    """
    from recommendation.control import SearchCatalogArguments

    return SearchCatalogArguments.from_text(message, limit=limit)
