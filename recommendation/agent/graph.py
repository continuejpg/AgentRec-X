"""Minimal LangGraph agent orchestration around the Recommendation Tool (Milestone 7B).

The graph proves one thing: that a LangGraph workflow can drive the already
accepted Milestone 7A Recommendation Tool while keeping the trusted-history
boundary intact.

::

    START
      |
      v
    decide            (decision model: see recommendation/agent/decision.py)
      |
      +-- action == "direct_response" --> finalize --> END
      |
      +-- action == "recommend" --------> recommend -> finalize --> END
                                          (RecommendationTool)

The ``recommend`` node is the *only* component that touches the recommender, and
it does so through the accepted Tool - never through HTTP, never by reimplementing
scoring, masking or ranking:

::

    LangGraph -> RecommendationTool -> SASRecInferenceEngine -> SASRec model

Trust boundary
--------------
* The decision model receives exactly one thing: the user message, wrapped by
  :func:`~recommendation.agent.decision.build_decision_messages`.  It never
  receives ``trusted_user_history``, internal item ids, encoded model history,
  SASRec tensors, mapping internals or checkpoint internals.
* ``trusted_user_history`` is application-owned graph state.  The ``decide`` node
  returns only ``decision``, so a decision cannot add to, drop from or reorder the
  history; ``AgentDecision`` additionally forbids unknown fields, so a payload
  containing a history field is rejected outright.
* The Tool node builds a fresh :class:`RecommendationContext` from the state and
  passes it as the *separate* ``context`` argument.  The model-facing request
  (:class:`RecommendationToolRequest`) carries only ``k``.

What this milestone is not
--------------------------
No planner, intent classifier, ReAct loop, reflection, retry, summarizer, critic,
reranker, memory, RAG, vector store, embedding, product-metadata lookup, semantic
ID, multi-agent hand-off or conversation history is implemented here.  The graph
has no cycles and makes at most one Tool call per run.

No provider SDK is imported and no network call is made: the decision model is
injected, so tests and the smoke experiment supply a deterministic offline stub.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langgraph.graph import END, START, StateGraph

from recommendation.tools import (
    MissingUserHistory,
    RecommendationContext,
    RecommendationTool,
    RecommendationToolRequest,
)

from .decision import (
    AgentAction,
    AgentDecision,
    DecisionMessage,
    DecisionModel,
    MalformedDecision,
    build_decision_messages,
    parse_agent_decision,
)
from .state import (
    AgentGraphState,
    AgentInput,
    TrustedHistory,
    history_digest,
    new_agent_state,
    read_trusted_history,
)

__all__ = [
    "AGENT_GRAPH_VERSION",
    "NODE_DECIDE",
    "NODE_ENRICH",
    "NODE_FINALIZE",
    "NODE_RECOMMEND",
    "ROUTE_DIRECT",
    "ROUTE_RECOMMEND",
    "AgentAction",
    "AgentConfigurationError",
    "AgentGraph",
    "AgentGraphError",
    "ProductEnricherLike",
    "build_agent_graph",
    "history_digest",
]

#: Graph contract version, bumped when the node/route contract changes.
AGENT_GRAPH_VERSION = 1

#: Node names, exposed so tests and the smoke experiment can assert structure.
NODE_DECIDE = "decide"
NODE_RECOMMEND = "recommend"
NODE_ENRICH = "enrich"
NODE_FINALIZE = "finalize"

#: Conditional-edge labels returned by the decision router.
ROUTE_DIRECT = "direct"
ROUTE_RECOMMEND = "recommend"


class AgentGraphError(Exception):
    """Base class for agent-graph failures."""


class AgentConfigurationError(AgentGraphError):
    """The graph was constructed with an unusable collaborator."""


@runtime_checkable
class ProductEnricherLike(Protocol):
    """Structural interface for the optional Milestone 8 product enricher.

    The graph depends on this shape rather than on
    :class:`~recommendation.rag.enrichment.ProductEnricher`, exactly as it depends on
    a ``RecommendationEngine`` protocol rather than the concrete engine.  That keeps
    ``recommendation/agent`` free of a hard import of the RAG package and lets a test
    or a future backend inject any conforming object.
    """

    def enrich(self, result: Any, query: str = "") -> Any:
        """Return candidate-scoped evidence for a Tool result."""
        ...


#: Text used when the recommender legitimately has no eligible candidate left.
#: Candidate exhaustion is a normal outcome, never turned into an error or padded
#: with fabricated items.
_NO_CANDIDATES_TEXT = (
    "No unseen product is left to recommend from this interaction history, so "
    "there are no candidates to show."
)

#: Footer that stops raw model scores from being read as product evidence.  The
#: recommender's scores order candidates; they say nothing about a product.
_SCORE_DISCLAIMER = (
    "Note: these are raw sequential-model ranking scores used only to order the "
    "candidates. They are not probabilities or confidence values, and they are not "
    "evidence about a product's attributes, quality or availability."
)

#: Provenance root of the catalogue metadata used by the Milestone 8 enricher.
_METADATA_PROVENANCE_ROOT = "amazon_reviews_2023:meta_categories"

#: Field labels used when rendering grounded metadata facts.
_FIELD_LABELS: dict[str, str] = {
    "title": "Title",
    "store": "Store",
    "main_category": "Main category",
    "categories": "Category",
    "features": "Feature",
    "description": "Description",
    "details": "Detail",
}


def _build_recommendation_response(result: Any) -> str:
    """Render a Tool result as candidate lines plus an honest score disclaimer."""
    if not result.recommendations:
        return _NO_CANDIDATES_TEXT

    lines = [
        f"Top {result.returned_k} candidate(s) from the sequential recommender:",
        "",
    ]
    lines.extend(
        f"{item.rank}. {item.parent_asin} (score {item.score:+.4f})"
        for item in result.recommendations
    )
    if result.returned_k < result.requested_k:
        lines.extend(
            [
                "",
                f"Only {result.returned_k} of {result.requested_k} requested candidates "
                "were available after excluding items already in the history.",
            ]
        )
    lines.extend(["", _SCORE_DISCLAIMER])
    return "\n".join(lines)


def _shorten(text: str, limit: int = 240) -> str:
    """Trim a rendered fact for display, marking the elision explicitly.

    Only presentation is shortened; the structured evidence keeps the full verbatim
    text, so nothing is lost from the grounding record.
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _build_grounded_response(enrichment: Any) -> str:
    """Render enriched candidates with attributed catalogue facts.

    Milestone 8 is the first point where product attribute claims may appear, and
    only facts actually present in normalized metadata are printed.  A candidate the
    source does not cover is shown with its identity and an explicit
    "metadata unavailable" note rather than an invented description.  The raw SASRec
    score is labelled as a ranking score, never as a probability or a rating.
    """
    if not enrichment.items:
        return _NO_CANDIDATES_TEXT

    lines = [
        f"Top {enrichment.returned_k} candidate(s) from the sequential recommender, "
        "with catalogue facts where available:",
    ]

    for item in enrichment.items:
        lines.extend(
            ["", f"{item.rank}. {item.parent_asin} (ranking score {item.score:+.4f})"]
        )
        if item.metadata_status == "missing":
            lines.append("   metadata unavailable for this item")
            continue
        if not item.evidence:
            reason = item.fallback_reason or "no_evidence"
            lines.append(f"   no matching catalogue detail retrieved ({reason})")
            continue
        for evidence in item.evidence:
            label = _FIELD_LABELS.get(evidence.field, evidence.field)
            if evidence.detail_key:
                label = evidence.detail_key
            lines.append(f"   {label}: {_shorten(evidence.text)}")

    lines.extend(
        [
            "",
            "Facts above are quoted from Amazon Reviews 2023 product metadata for the "
            "listed item and are not present for every item.",
            _SCORE_DISCLAIMER,
            "Evidence is selected only from these candidates; the ranking itself comes "
            "from the sequential recommender.",
        ]
    )
    return "\n".join(lines)


class AgentGraph:
    """The compiled Milestone 7B workflow plus its validated entry points.

    Parameters
    ----------
    decision_model:
        Any object with ``decide(messages) -> AgentDecision``.  Injected, so the
        graph never constructs a provider client itself and works fully offline.
    tool:
        The accepted Milestone 7A :class:`~recommendation.tools.RecommendationTool`.
        The graph calls it in process; it never routes through the FastAPI service.
    """

    def __init__(
        self,
        decision_model: DecisionModel,
        tool: RecommendationTool,
        product_enricher: ProductEnricherLike | None = None,
    ) -> None:
        if decision_model is None or not callable(
            getattr(decision_model, "decide", None)
        ):
            raise AgentConfigurationError(
                "decision_model must provide a callable decide(messages) method"
            )
        if not isinstance(tool, RecommendationTool):
            raise AgentConfigurationError(
                f"tool must be a RecommendationTool, got {type(tool).__name__}"
            )
        if product_enricher is not None and not callable(
            getattr(product_enricher, "enrich", None)
        ):
            raise AgentConfigurationError(
                "product_enricher must provide a callable enrich(result, query) method"
            )
        self._decision_model = decision_model
        self._tool = tool
        self._product_enricher = product_enricher
        self._graph = self._build()

    # -- metadata ---------------------------------------------------------- #

    @property
    def version(self) -> int:
        """Graph contract version."""
        return AGENT_GRAPH_VERSION

    @property
    def decision_model(self) -> DecisionModel:
        """The injected decision model (exposed for lifecycle inspection/tests)."""
        return self._decision_model

    @property
    def tool(self) -> RecommendationTool:
        """The injected Recommendation Tool (exposed for lifecycle inspection/tests)."""
        return self._tool

    @property
    def product_enricher(self) -> ProductEnricherLike | None:
        """The injected product enricher, or ``None`` for raw M7B/M7C behaviour."""
        return self._product_enricher

    @property
    def enriches_products(self) -> bool:
        """True when the recommendation route includes an enrichment node."""
        return self._product_enricher is not None

    # -- construction ------------------------------------------------------ #

    def _build(self) -> Any:
        """Assemble and compile the state graph.

        The topology is conditional on whether a product enricher was injected:

        * **no enricher** -- the accepted Milestone 7B/7C topology is built exactly as
          before (``decide -> {finalize | recommend -> finalize}``), so existing
          behaviour and node set are preserved bit-for-bit;
        * **enricher present** -- an ``enrich`` node is inserted between ``recommend``
          and ``finalize`` on the recommendation route only.

        The graph never constructs a metadata store itself: enrichment exists only
        because a caller injected one.
        """
        builder: StateGraph = StateGraph(AgentGraphState)
        builder.add_node(NODE_DECIDE, self._decide_node)
        builder.add_node(NODE_RECOMMEND, self._recommend_node)
        builder.add_node(NODE_FINALIZE, self._finalize_node)

        builder.add_edge(START, NODE_DECIDE)
        builder.add_conditional_edges(
            NODE_DECIDE,
            self._route,
            {ROUTE_DIRECT: NODE_FINALIZE, ROUTE_RECOMMEND: NODE_RECOMMEND},
        )
        if self._product_enricher is None:
            builder.add_edge(NODE_RECOMMEND, NODE_FINALIZE)
        else:
            builder.add_node(NODE_ENRICH, self._enrich_node)
            builder.add_edge(NODE_RECOMMEND, NODE_ENRICH)
            builder.add_edge(NODE_ENRICH, NODE_FINALIZE)
        builder.add_edge(NODE_FINALIZE, END)
        return builder.compile()

    # -- nodes ------------------------------------------------------------- #

    def _decide_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Ask the decision model which route to take.

        Only ``user_message`` is passed on, and only ``decision`` is returned:
        this node is structurally unable to read or write trusted history.
        """
        user_message = state.get("user_message")
        if not isinstance(user_message, str) or not user_message.strip():
            raise MalformedDecision("the agent run has no user message")

        messages: tuple[DecisionMessage, ...] = build_decision_messages(user_message)
        try:
            raw = self._decision_model.decide(messages)
        except MalformedDecision:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any collaborator failure
            raise MalformedDecision(
                f"the decision model failed: {type(exc).__name__}"
            ) from exc
        return {"decision": parse_agent_decision(raw)}

    def _route(self, state: AgentGraphState) -> str:
        """Conditional-edge function; raises rather than defaulting to a route."""
        decision = state.get("decision")
        if not isinstance(decision, AgentDecision):
            raise MalformedDecision("no decision is present in the graph state")
        return ROUTE_RECOMMEND if decision.needs_recommendation else ROUTE_DIRECT

    def _recommend_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Call the accepted Recommendation Tool with **trusted** history.

        ``request`` carries only ``k``; ``context`` carries the application-owned
        history.  Tool domain errors propagate unchanged so callers see the Tool's
        stable error codes.
        """
        decision = state.get("decision")
        if not isinstance(decision, AgentDecision):
            raise MalformedDecision("no decision is present in the graph state")

        # The graph owns the trusted history, so a missing one is reported with the
        # Tool's own business-level error rather than continuing into the engine.
        history = state.get("trusted_user_history")
        if not history:
            raise MissingUserHistory(
                "the agent run has no trusted user history; supply it from application state"
            )

        request = RecommendationToolRequest(k=decision.requested_k)
        context = RecommendationContext(user_history=history)
        return {"tool_result": self._tool.run(request=request, context=context)}

    def _enrich_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Attach candidate-scoped catalogue evidence to the Tool result.

        This node is reached only on the recommendation route and only when an
        enricher was injected.  It receives the Tool result and the untrusted user
        message; it is never given the trusted interaction history, so it cannot
        alter it.  The Tool result and the candidate sequence pass through unchanged.
        """
        if self._product_enricher is None:  # pragma: no cover - node not built then
            raise AgentGraphError("no product enricher is configured")

        result = state.get("tool_result")
        if result is None:
            raise AgentGraphError("the recommend route produced no Tool result to enrich")

        # The user message is untrusted text used only to select evidence among the
        # current candidates; it never reaches history, ids or the candidate universe.
        user_message = state.get("user_message")
        query = user_message if isinstance(user_message, str) else ""
        return {"enrichment": self._product_enricher.enrich(result, query)}

    def _finalize_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Produce the run's final text and route tag."""
        decision = state.get("decision")
        if not isinstance(decision, AgentDecision):
            raise MalformedDecision("no decision is present in the graph state")

        if decision.needs_recommendation:
            # Re-validate the application-owned history here too: the finalize step
            # must never emit a recommendation for a run that had no trusted history.
            read_trusted_history(state)
            result = state.get("tool_result")
            if result is None:
                raise AgentGraphError("the recommend route produced no Tool result")
            enrichment = state.get("enrichment")
            if enrichment is not None:
                return {
                    "route": ROUTE_RECOMMEND,
                    "final_response": _build_grounded_response(enrichment),
                }
            return {
                "route": ROUTE_RECOMMEND,
                "final_response": _build_recommendation_response(result),
            }
        return {
            "route": ROUTE_DIRECT,
            "final_response": decision.direct_response or "",
        }

    # -- invocation -------------------------------------------------------- #

    def invoke(self, agent_input: AgentInput) -> AgentGraphState:
        """Run the graph for one validated application input.

        Raises
        ------
        MalformedDecision
            The decision model produced an unusable decision, or the run had no
            trusted history / user message.
        RecommendationToolError
            The Tool rejected or could not complete the call (propagated unchanged).
        """
        if not isinstance(agent_input, AgentInput):
            raise AgentConfigurationError(
                f"invoke expects an AgentInput, got {type(agent_input).__name__}"
            )
        return self._graph.invoke(new_agent_state(agent_input))  # type: ignore[return-value]

    def run(self, user_message: str, trusted_user_history: TrustedHistory) -> AgentGraphState:
        """Convenience wrapper around :meth:`invoke` with validated inputs."""
        return self.invoke(
            AgentInput(user_message=user_message, trusted_user_history=trusted_user_history)
        )

    # -- introspection ----------------------------------------------------- #

    def node_names(self) -> tuple[str, ...]:
        """Return the declared node names, sorted, for structural assertions."""
        return tuple(sorted(self._graph.get_graph().nodes))

    def mermaid(self) -> str:
        """Return the graph as Mermaid text (the canonical M7B structure proof)."""
        return self._graph.get_graph().draw_mermaid()


#: Public alias mirroring the module purpose.
def build_agent_graph(
    decision_model: DecisionModel,
    tool: RecommendationTool,
    product_enricher: ProductEnricherLike | None = None,
) -> AgentGraph:
    """Construct the agent graph (factory form).

    Without ``product_enricher`` this builds the accepted Milestone 7B/7C graph;
    supplying one inserts the Milestone 8 enrichment node on the recommendation route.
    """
    return AgentGraph(decision_model, tool, product_enricher=product_enricher)
