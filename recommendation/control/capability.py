"""``RecommendFromHistoryCapability`` - the accepted pipeline behind a control boundary.

This is a **migration wrapper, not a recommendation rewrite**.

Stage 1 of AgentRec-X 2.0-alpha moves control authority without touching recommendation
semantics, so the accepted Milestones 7A-10B pipeline is treated as one indivisible
capability::

    ValidatedAction(RECOMMEND_FROM_HISTORY)
        |
        v
    RecommendFromHistoryCapability
        |  recommend     -> RecommendationTool -> SASRecInferenceEngine -> SASRec
        |  enrich        -> ProductEnricher    -> MetadataIndex
        |  match         -> PreferenceCandidateMatcher
        |  rerank        -> PreferenceReranker
        v
    RecommendationDomainResult   (raw; NOT policy-visible)

Every one of those four stages is the **same collaborator instance** the accepted
:class:`~recommendation.agent.graph.AgentGraph` uses, called through the same shared
stage functions.  Nothing is re-implemented here, and no stage can be skipped or
reordered by the policy: the capability fixes the sequence and the callers of the
pipelines are the only thing that changed.

Trust boundary
--------------
The capability receives trusted history through an injected
:class:`TrustedHistoryReader` - a *function*, not a value from the action.  A
:class:`~recommendation.control.schemas.ValidatedAction` has no history field, so there
is no path by which a policy could supply, forge or extend the history this pipeline
scores.  The capability never reads history from anywhere else.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from recommendation.agent.graph import (
    AgentGraphError,
    PreferenceMatcherLike,
    PreferenceMemoryLike,
    PreferenceRerankerLike,
    ProductEnricherLike,
)
from recommendation.agent.rendering import memory_context
from recommendation.agent.state import TrustedHistory
from recommendation.tools import (
    MissingUserHistory,
    RecommendationContext,
    RecommendationTool,
    RecommendationToolRequest,
)

from .schemas import (
    ActionKind,
    PolicyActionError,
    RecommendationDomainResult,
    ValidatedAction,
)

__all__ = [
    "CAPABILITY_NAME",
    "RecommendFromHistoryCapability",
    "TrustedHistoryReader",
]

#: Stable capability identity.
CAPABILITY_NAME = "recommend_from_history"


@runtime_checkable
class TrustedHistoryReader(Protocol):
    """Read-only access to the application-owned trusted history.

    Deliberately one method with no argument: a capability cannot ask for "the history of
    user X" or "the history matching this payload".  It can only ask the application for
    the history of the run it is already executing.
    """

    def __call__(self) -> TrustedHistory:
        """Return the run's trusted chronological ``parent_asin`` history."""
        ...


def _preference_lines(snapshot: Any) -> list[str]:
    """Return the rendered constraint lines of a snapshot, tolerating an absent one.

    Used only to decide *whether* to augment the retrieval query.  The augmentation
    itself goes through the accepted Milestone 9 renderer, so the query this capability
    builds is character-for-character the query the DAG builds.
    """
    if snapshot is None:
        return []
    return [entry for entry in snapshot.active_entries]


class RecommendFromHistoryCapability:
    """Execute the accepted recommendation pipeline for one validated action.

    Parameters
    ----------
    tool:
        The accepted Milestone 7A :class:`RecommendationTool`.  The capability calls it
        in process and never routes through HTTP.
    product_enricher, preference_matcher, preference_reranker:
        The optional accepted Milestones 8 / 10A / 10B collaborators, exactly as the
        DAG takes them.  Each stage runs only when its collaborator is present, so the
        capability reproduces the DAG's conditional shape instead of forcing a stage the
        deployment did not configure.
    augment_query_with_preferences:
        Mirrors the DAG flag that appends active preferences to the retrieval query.
        It only changes which evidence fragments are selected from the metadata of the
        already-fixed candidate set; it cannot add, drop or reorder a candidate.
    """

    def __init__(
        self,
        tool: RecommendationTool,
        *,
        product_enricher: ProductEnricherLike | None = None,
        preference_matcher: PreferenceMatcherLike | None = None,
        preference_reranker: PreferenceRerankerLike | None = None,
        augment_query_with_preferences: bool = True,
    ) -> None:
        if not isinstance(tool, RecommendationTool):
            raise AgentGraphError(
                f"tool must be a RecommendationTool, got {type(tool).__name__}"
            )
        if (preference_matcher is None) != (preference_reranker is None):
            raise AgentGraphError(
                "preference_matcher and preference_reranker must be configured together; "
                "a half-configured preference stage is not supported"
            )
        if preference_matcher is not None:
            if not callable(getattr(preference_matcher, "match", None)):
                raise AgentGraphError(
                    "preference_matcher must provide a callable match(candidates=..., "
                    "preferences=...) method"
                )
            if not callable(getattr(preference_reranker, "rerank", None)):
                raise AgentGraphError(
                    "preference_reranker must provide a callable rerank(report) method"
                )
            if product_enricher is None:
                raise AgentGraphError(
                    "preference_matcher requires product_enricher: preference evidence is "
                    "read from the metadata the enrichment stage attached to the current "
                    "candidates, so matching cannot run without it"
                )
        self._tool = tool
        self._product_enricher = product_enricher
        self._preference_matcher = preference_matcher
        self._preference_reranker = preference_reranker
        self._augment_query_with_preferences = bool(augment_query_with_preferences)

    # -- metadata ---------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Stable capability identity."""
        return CAPABILITY_NAME

    @property
    def tool(self) -> RecommendationTool:
        """The accepted Tool this capability drives (exposed for inspection/tests)."""
        return self._tool

    @property
    def enriches(self) -> bool:
        """True when the enrichment stage is configured."""
        return self._product_enricher is not None

    @property
    def matches_preferences(self) -> bool:
        """True when the preference-evidence stage is configured."""
        return self._preference_matcher is not None

    @property
    def reranks_preferences(self) -> bool:
        """True when the preference-reranking stage is configured."""
        return self._preference_reranker is not None

    def stages(self) -> tuple[str, ...]:
        """The exact stage sequence this capability will execute."""
        stages = ["recommend"]
        if self._product_enricher is not None:
            stages.append("enrich")
        if self._preference_matcher is not None:
            stages.extend(("match_preferences", "rerank"))
        return tuple(stages)

    # -- execution --------------------------------------------------------- #

    def execute(
        self,
        action: ValidatedAction,
        *,
        read_trusted_history: TrustedHistoryReader,
        user_message: str,
        preference_snapshot: Any = None,
        tool_call_count: int = 0,
    ) -> RecommendationDomainResult:
        """Run the accepted pipeline and return an **unverified** domain result.

        Parameters
        ----------
        action:
            The validated action.  Only its validated ``k`` is read; the action cannot
            carry history, a tool name, a query or candidate ids.
        read_trusted_history:
            The injected reader described in the module docstring.  Called exactly once.
        user_message:
            Untrusted text, used only to select evidence among the current candidates.
        preference_snapshot:
            The turn's Milestone 9 read-only snapshot, if memory is configured.
        tool_call_count:
            How many tool calls the loop has already consumed.  Used for the opaque
            candidate-set reference only; it never changes the pipeline result.

        Raises
        ------
        PolicyActionError
            The action is not a validated recommendation action.  The capability refuses
            rather than guessing.
        MissingUserHistory
            The trusted reader returned no history.  There is no fallback.
        RecommendationToolError
            Propagated unchanged from the accepted Tool (stable domain codes).
        AgentGraphError
            An optional stage was reached without its input, or the reranker changed the
            candidate set.
        """
        if action.action is not ActionKind.RECOMMEND_FROM_HISTORY:
            raise PolicyActionError(
                f"capability {CAPABILITY_NAME} only executes "
                f"'{ActionKind.RECOMMEND_FROM_HISTORY.value}', got '{action.action.value}'"
            )

        # -- stage 1: recommend (accepted Milestone 7A) --------------------- #
        history = read_trusted_history()
        if not history:
            raise MissingUserHistory(
                "the run has no trusted user history; supply it from application state"
            )
        request = RecommendationToolRequest(k=action.k)
        context = RecommendationContext(user_history=history)
        tool_result = self._tool.run(request=request, context=context)

        enrichment: Any = None
        evidence: Any = None
        reranking: Any = None

        # -- stage 2: enrich (accepted Milestone 8) ------------------------- #
        if self._product_enricher is not None:
            query = user_message if isinstance(user_message, str) else ""
            if self._augment_query_with_preferences:
                lines = _preference_lines(preference_snapshot)
                if lines:
                    query = memory_context(query, preference_snapshot)
            enrichment = self._product_enricher.enrich(tool_result, query)

            # -- stage 3: preference evidence (accepted Milestone 10A) ------ #
            if self._preference_matcher is not None:
                preferences = () if preference_snapshot is None else preference_snapshot
                evidence = self._preference_matcher.match(
                    candidates=enrichment.items, preferences=preferences
                )

                # -- stage 4: rerank (accepted Milestone 10B) --------------- #
                if self._preference_reranker is not None:
                    reranking = self._preference_reranker.rerank(evidence)

        returned_k = len(tool_result.recommendations)
        candidate_set_ref = (
            f"{CAPABILITY_NAME}:{action.action_id}:{returned_k}:{tool_call_count}"
        )

        return RecommendationDomainResult(
            action_id=action.action_id,
            status="empty" if returned_k == 0 else "ok",
            tool_result=tool_result,
            enrichment=enrichment,
            preference_evidence=evidence,
            reranking=reranking,
            requested_k=tool_result.requested_k,
            returned_k=returned_k,
            eligible_candidates=tool_result.eligible_candidates,
            candidate_set_ref=candidate_set_ref,
        )


#: Structural interface kept visible for callers that only need the memory seam.
MemoryLike = PreferenceMemoryLike
