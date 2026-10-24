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

__all__ = [
    "SOURCE_ARGUMENTS",
    "SOURCE_PLANS",
    "SOURCE_PURPOSES",
    "ClarificationPolicy",
    "DemoSourcePlanPolicy",
    "IntentQueryPolicy",
    "normalise_source_plan",
]

#: The candidate sources a request may name, mapped to the control-plane action that
#: consults them.  ``history`` is the accepted sequential recommender; ``catalog_search`` is
#: full-catalogue lexical search over normalized catalogue text.
#:
#: ``history`` maps to the accepted ``RECOMMEND_FROM_HISTORY`` action, and that is a
#: deliberate refusal to route it through the candidate plane: the accepted single-source path
#: stays exactly as it was, and the response labels its provenance as the history source without
#: a ledger record (see :func:`~recommendation.demo.agent_trajectory.build_grounded_view`).
#: ``catalog_search`` goes through the plane, which is what records real multi-source provenance.
#:
#: ``two_tower`` and ``tiger`` are consulted through ``SELECT_SOURCE``, the action that names one
#: trusted source.  That action used to be offered only when a *similar-item* source was
#: registered, which left both unreachable in a deployment that does not ship the 6.21 GB
#: item-item index.  The menu rule now offers it whenever the plane can serve any source the
#: action can name (see ``recommendation/control/loop.py:_plane_candidate_actions``); a
#: deployment that does register similar-item sees the identical action tuple as before.
SOURCE_PLANS: dict[str, ActionKind] = {
    "history": ActionKind.RECOMMEND_FROM_HISTORY,
    "catalog_search": ActionKind.SEARCH_CATALOG,
    # Two-Tower and TIGER are consulted through ``SELECT_SOURCE``, the action that names one
    # trusted source, because neither is the accepted history recommender nor a lexical search.
    "two_tower": ActionKind.SELECT_SOURCE,
    "tiger": ActionKind.SELECT_SOURCE,
}

#: The ``CandidateSource`` each plan name selects when consulted through the candidate plane.
SOURCE_ARGUMENTS: dict[str, str] = {
    "history": "history",
    "catalog_search": "catalog_search",
    "two_tower": "two_tower",
    "tiger": "tiger",
}

#: One short, factual purpose per source name, for callers that must *describe* a source rather
#: than consult it - today the tool-selection prompt, which is handed the menu of sources this
#: deployment can actually serve.  It lives beside :data:`SOURCE_PLANS` so the described set and
#: the executable set cannot drift apart, and each wording names a retrieval capability only: a
#: purpose may not claim a product fact, a quality, an identity or a score.
SOURCE_PURPOSES: dict[str, str] = {
    "history": "sequential recommendation from the shopper's own trusted purchase history",
    "catalog_search": "lexical full-catalogue search over normalized product text",
    "two_tower": "embedding retrieval from the trained Two-Tower candidate model",
    "tiger": "generative semantic-id retrieval from the TIGER backend",
}

#: Default plan: the accepted history recommender alone.  That keeps the endpoint's default
#: behaviour identical to the accepted single-source path.
DEFAULT_SOURCE_PLAN: tuple[str, ...] = ("history",)

#: Upper bound on a plan.  Four sources is every registered source today, and a bound here
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
        use_memory: bool = True,
    ) -> None:
        if not plan:
            raise ValueError("a source plan needs at least one source")
        for source in plan:
            if source not in SOURCE_PLANS:
                raise ValueError(f"unknown source {source!r}")
        self._plan = tuple(plan)
        self._query_builder = query_builder or _search_arguments_from_message
        self._default_k = int(default_k)
        #: Whether this plan may fold the user's stored preferences into the lexical query.
        #: Defaults to the accepted legacy behaviour; an LLM-selected run turns it off so an
        #: unrelated historical preference can never be appended behind the current request.
        self._use_memory = bool(use_memory)
        self._call_count = 0
        #: Sources already proposed **in the current run**, and whether the current run's plan is
        #: served.  Both are cleared on the first step of a run (see :meth:`choose`), because a
        #: compiled policy may serve more than one run - whether the controller is reused or the
        #: same instance is handed to a second engine - and run-scoped state that survived into the
        #: next run used to make that run skip a planned source.
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

        That record is **run-scoped**, and this method is what makes it so: ``step_index == 0``
        is the first decision of a run (the engine that carries the step counter is built per
        run), so both fields are cleared there.  Relying on a fresh policy instance instead would
        be relying on how a *caller* composes its controller - and a caller that reuses one (as
        this deployment did) would then silently change the second run's behaviour: a source
        already served in the previous run is still in ``_served``, so once the first source of
        the new run has grounded candidates the next planned source is skipped and the plan ends
        early.  Cleaning up here keeps one run's decisions from reaching the next one whatever
        the caller does.

        The record is a returned flag rather than mutable state: ``choose`` reports
        ``finished_plan=True`` once the plan is served, and the loop clears that flag on the
        terminal step.  Note that a *refused* FINISH does not restart the plan: the flag stays
        true, the policy keeps proposing FINISH, and the step budget bounds the run.

        A planned source the run does not currently offer is skipped rather than proposed: the
        controller's validator is the authority on availability, and a policy that proposed an
        unavailable action would spend a step to be refused.
        """
        self._call_count += 1
        if context.step_index == 0:
            # The first decision of a run: nothing has been consulted *in this run* yet.
            self._served = set()
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
                arguments=_adapt_search_arguments(
                    self._query_builder(context.user_request, limit=self._default_k),
                    # The controller hands a policy the *projection* of the user's ACTIVE
                    # preferences, never the M9 snapshot (see PolicyContext); the adapter reads the
                    # same six values out of it either way, so this stays duck-typed and the
                    # baseline path is unchanged when memory is off.
                    getattr(context, "active_preferences", None) if self._use_memory else None,
                ),
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

    def proposal_for_source(self, source: str, context: Any) -> ActionProposal | None:
        """One proposal for exactly ``source``, built the same way the plan builds it.

        Public because a *bounded replanning* policy (E-03B) may add one source the authorised
        plan did not name, and that source's action and arguments must be constructed by the same
        rule as any other: the query comes from the current request, the budget from this policy's
        ``default_k``, and the source name from :data:`SOURCE_PLANS`.  Returns ``None`` when this
        deployment's plan vocabulary does not know the name - the caller then declines rather than
        inventing an action.
        """
        action = SOURCE_PLANS.get(str(source))
        if action is None:
            return None
        return self._proposal_for(str(source), action, context)

    def _select_arguments(self, source: str) -> Any:
        """Build ``SELECT_SOURCE`` arguments naming one trusted source.

        One definition only.  The FINISH fallback lives in :meth:`choose`, which is the only
        place that may decide the plan is served; a second copy here was unreachable.
        """
        from recommendation.control import CandidateSource, SelectSourceArguments

        return SelectSourceArguments(
            source=CandidateSource(SOURCE_ARGUMENTS[source]), limit=self._default_k
        )


def _adapt_search_arguments(arguments: Any, snapshot: Any | None) -> Any:
    """Fold this user's ACTIVE preferences into the lexical query, if any apply.

    The policy owns *what to look for*; the trusted search tool still owns matching.  With no
    usable preference the original arguments object is returned **unchanged**, so the baseline path
    is bit-for-bit what it was.
    """
    from recommendation.control import SearchCatalogArguments
    from recommendation.control.arguments import MAX_SEARCH_TERMS
    from recommendation.memory.query_adapter import adapt_terms

    base = tuple(getattr(arguments, "terms", ()) or ())
    adaptation = adapt_terms(base, snapshot)
    if not adaptation.changed:
        return arguments
    # The adapter builds the object directly, so it must respect the argument schema's own term
    # bound (``MAX_SEARCH_TERMS``) rather than assume the constructor already capped the list.
    return SearchCatalogArguments(terms=adaptation.terms[:MAX_SEARCH_TERMS])


def _search_arguments_from_message(message: str, *, limit: int) -> Any:
    """Build ``SEARCH_CATALOG`` arguments from the user's own message.

    The message is split, not understood: :class:`~recommendation.control.SearchCatalogArguments`
    owns the token bound and the trusted search tool owns the matching.  A message with no
    usable term cannot build a query, which is reported as a policy failure rather than
    patched with a fabricated query.
    """
    from recommendation.control import SearchCatalogArguments

    return SearchCatalogArguments.from_text(message, limit=limit)


class IntentQueryPolicy:
    """Wrap a plan policy and rebuild its catalogue query from the current intent.

    The wrapped policy still owns *which* sources to consult and in what order.  Only the
    ``SEARCH_CATALOG`` terms are replaced, with the terms the shopping-intent projection
    computed for this turn (hard category/brand, then soft, then the user's own message).
    Removed and superseded values are absent from the intent, so they cannot leak into the
    query; the cap is the argument schema's own.
    """

    name = "demo-source-plan-with-intent-query"

    def __init__(self, inner: Any, *, terms: Sequence[str]) -> None:
        self._inner = inner
        self._terms = tuple(str(term) for term in terms)

    def choose(self, context: Any) -> ActionProposal:
        proposal = self._inner.choose(context)
        if proposal.action is ActionKind.SEARCH_CATALOG and self._terms:
            from recommendation.control import SearchCatalogArguments

            return proposal.model_copy(
                update={"arguments": SearchCatalogArguments(terms=self._terms)}
            )
        return proposal


class ClarificationPolicy:
    """Wrap a plan policy and ask one question when the intent cannot be resolved.

    Two triggers, both explicit:

    * ``question`` -- a pre-computed conflict question (for example a price floor above the
      ceiling).  Asked on the first decision, before any source runs.
    * ``ask_on_unresolved`` -- at the point the plan would finish, if the run holds no
      verified candidate and at least one candidate is still undecided.  The question says
      exactly that; it never claims the undecided candidates are non-compliant.

    One question per run, and only while the controller offers the action.  The guard, the
    constraints and the candidate set are untouched; a clarification is a suspension, not a
    success, and no product is presented.
    """

    name = "demo-source-plan-with-clarification"

    def __init__(
        self,
        inner: Any,
        *,
        question: str | None = None,
        ask_on_unresolved: bool = False,
    ) -> None:
        self._inner = inner
        self._question = question
        self._ask_on_unresolved = bool(ask_on_unresolved)
        self._asked = False

    def choose(self, context: Any) -> ActionProposal:
        if getattr(context, "step_index", 0) == 0:
            self._asked = False
        if (
            not self._asked
            and not getattr(context, "awaiting_user", False)
            and context.action_available(ActionKind.ASK_CLARIFICATION)
        ):
            question = self._question
            if (
                question is None
                and self._ask_on_unresolved
                and int(getattr(context, "verified_eligible_count", 0) or 0) == 0
                and int(getattr(context, "unresolved_count", 0) or 0) > 0
            ):
                from recommendation.demo.shopping_intent import build_unresolved_question

                question = build_unresolved_question(context)
            if question:
                from recommendation.control.arguments import AskClarificationArguments

                self._asked = True
                return ActionProposal(
                    action=ActionKind.ASK_CLARIFICATION,
                    rationale=(
                        "the current intent cannot be resolved from the catalogue; the answer "
                        "would change which candidates are eligible"
                    ),
                    arguments=AskClarificationArguments(
                        question=str(question), blocks="candidate_set"
                    ),
                )
        return self._inner.choose(context)
