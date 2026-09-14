"""LangGraph agent orchestration around the Recommendation Tool (Milestones 7B-10D).

The graph drives the already accepted Milestone 7A Recommendation Tool while keeping
the trusted-history boundary intact, and composes the accepted Milestone 8, 9, 10A and
10B stages as *optional injected collaborators*.

Complete topology (every optional stage present)::

    START
      |
      v
    load_memory        (M9: read-only active-preference snapshot)
      |
      v
    decide             (decision model: see recommendation/agent/decision.py)
      |
      +-- action == "direct_response" --> finalize --> persist_memory --> END
      |
      +-- action == "recommend" --------> recommend -> enrich
                                            (RecommendationTool)
                                            (M8 candidate-scoped metadata)
                                                          |
                                                          v
                                                 match_preferences
                                                    (M10A evidence)
                                                          |
                                                          v
                                                       rerank
                                                  (M10B frozen policy)
                                                          |
                                                          v
                                                      finalize --> persist_memory --> END

Every stage after ``decide`` is conditional on what the caller injected.  With nothing
injected the graph is exactly the accepted Milestone 7B/7C shape; with only an enricher
it is Milestone 8; with memory as well it is Milestone 9; adding the Milestone 10D
matcher/reranker pair appends the two nodes above.  No default collaborator is ever
constructed.

The ``recommend`` node is the *only* component that touches the recommender, and
it does so through the accepted Tool - never through HTTP, never by reimplementing
scoring, masking or ranking:

::

    LangGraph -> RecommendationTool -> SASRecInferenceEngine -> SASRec model

Original order versus reranked order
------------------------------------
Milestone 10D adds derived state, never a rewrite.  ``tool_result`` and ``enrichment``
keep the upstream SASRec order and their ``rank`` values are the authoritative
``original_rank``; ``preference_evidence`` holds M10A evidence still in that order; and
``reranking`` holds the M10B result, where each candidate carries both ``original_rank``
and ``reranked_rank``.  The final text is rendered in reranked order, and the graph
itself never sorts: ordering comes from the injected reranker.

M10C (offline policy evaluation) is deliberately **not** part of this path.  A serving
request needs evidence and order, not a cohort evaluator.

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
* Preferences come from the M9 snapshot only, and that snapshot is read once per turn.

What this milestone is not
--------------------------
No planner, intent classifier, ReAct loop, reflection, retry, summarizer, critic,
learned reranker, LLM critic, vector store, embedding, semantic ID, multi-agent
hand-off or conversation history is implemented here.  The graph has no cycles and
makes at most one Tool call per run.

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
    "NODE_LOAD_MEMORY",
    "NODE_MATCH_PREFERENCES",
    "NODE_PERSIST_MEMORY",
    "NODE_RECOMMEND",
    "NODE_RERANK",
    "ROUTE_DIRECT",
    "ROUTE_RECOMMEND",
    "AgentAction",
    "AgentConfigurationError",
    "AgentGraph",
    "AgentGraphError",
    "PreferenceMatcherLike",
    "PreferenceMemoryLike",
    "PreferenceRerankerLike",
    "ProductEnricherLike",
    "build_agent_graph",
    "history_digest",
]

#: Graph contract version, bumped when the node/route contract changes.
#:
#: * ``1`` -- Milestone 7B/7C/8/9: ``decide`` routing, optional ``enrich``, optional
#:   ``load_memory`` / ``persist_memory``;
#: * ``2`` -- Milestone 10D adds the optional ``match_preferences`` and ``rerank``
#:   nodes on the recommendation route.  Graphs built without a matcher/reranker pair
#:   keep the version 1 shape and behaviour.
AGENT_GRAPH_VERSION = 2

#: Node names, exposed so tests and the smoke experiment can assert structure.
NODE_LOAD_MEMORY = "load_memory"
NODE_DECIDE = "decide"
NODE_RECOMMEND = "recommend"
NODE_ENRICH = "enrich"
NODE_MATCH_PREFERENCES = "match_preferences"
NODE_RERANK = "rerank"
NODE_FINALIZE = "finalize"
NODE_PERSIST_MEMORY = "persist_memory"

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


@runtime_checkable
class PreferenceMemoryLike(Protocol):
    """Structural interface for the optional Milestone 9 preference-memory service.

    The graph depends on this shape rather than on the concrete
    :class:`~recommendation.memory.service.PreferenceMemoryService`, so
    ``recommendation/agent`` does not hard-import the memory package and a test can
    inject any conforming object.

    Note what is **absent**: there is no method that accepts or returns trusted
    interaction history.  The Agent therefore cannot read, rewrite or extend the
    SASRec history through this seam.
    """

    def get_active_preferences(self, user_key: str) -> Any:
        """Return the user's active preference snapshot (read-only)."""
        ...

    def process_turn(self, **kwargs: Any) -> Any:
        """Extract and persist explicit preferences from one user-authored turn."""
        ...


@runtime_checkable
class PreferenceMatcherLike(Protocol):
    """Structural interface for the optional Milestone 10A evidence matcher.

    The graph depends on this shape rather than on the concrete
    :class:`~recommendation.preference_matching.PreferenceCandidateMatcher`, so
    ``recommendation/agent`` does not hard-import the matching package and a test can
    inject any conforming object.

    Note what the seam exposes: **already-enriched candidates plus a preference
    snapshot in, evidence out**.  There is no store, no retriever and no Tool here, so
    the Agent cannot use this seam to reach a product outside the candidate universe,
    and it cannot use it to read or write trusted interaction history.
    """

    def match(self, *, candidates: Any, preferences: Any) -> Any:
        """Return M10A evidence for ``candidates`` against ``preferences``."""
        ...


@runtime_checkable
class PreferenceRerankerLike(Protocol):
    """Structural interface for the optional Milestone 10B reranker.

    Evidence in, reranked report out.  The graph never sorts candidates itself: it
    calls this seam, so the canonical lexicographic policy exists in exactly one place
    (``recommendation/reranking``).
    """

    def rerank(self, report: Any) -> Any:
        """Return the reranked view of an M10A evidence report."""
        ...


def _active_preference_lines(snapshot: Any) -> list[str]:
    """Render a preference snapshot as short, attributed constraint lines."""
    if snapshot is None:
        return []
    lines: list[str] = []
    for entry in snapshot.active_entries:
        marker = "prefers" if entry.polarity.value == "prefer" else "avoids"
        lines.append(f"{entry.kind.value}: {marker} {entry.value}")
    return lines


def _memory_context(query: str, snapshot: Any) -> str:
    """Augment a retrieval query with active explicit preferences.

    Format is explicit and documented rather than implicit::

        <user query>
        preferences:
        - <kind>: <prefer|avoid> <value>
        - ...

    This only changes *which evidence fragments* are selected from the metadata of
    the already-fixed candidate set.  It cannot add, drop or reorder a candidate, and
    it never touches trusted interaction history.
    """
    lines = _active_preference_lines(snapshot)
    if not lines:
        return query
    return "\n".join([query, "preferences:", *[f"- {line}" for line in lines]])


def _preference_block(preferences: Any) -> list[str]:
    """Render the user's stored preferences as an honest, clearly-labelled block.

    The block states what the user said.  It deliberately contains **no match score,
    no "perfectly matches" claim and no ranking hint** -- Milestone 9 has no
    preference-to-product scoring, and presenting one would be fabrication.
    """
    lines = _active_preference_lines(preferences)
    if not lines:
        return []
    return [
        "Your stated preferences (stored from your own messages; not used to rank "
        "these candidates):",
        *[f"- {line}" for line in lines],
        "",
    ]


def _build_recommendation_response(result: Any, preferences: Any = None) -> str:
    """Render a Tool result as candidate lines plus an honest score disclaimer."""
    if not result.recommendations:
        return _NO_CANDIDATES_TEXT

    lines = [
        f"Top {result.returned_k} candidate(s) from the sequential recommender:",
        "",
        *_preference_block(preferences),
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


def _build_grounded_response(enrichment: Any, preferences: Any = None) -> str:
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
        "",
        *_preference_block(preferences),
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


#: Header for the final response when M10D reranking is active.  It names the policy
#: and nothing else: no "best", no "relevant", no quality word.
_RERANK_HEADER = (
    "candidate(s) from the sequential recommender, ordered by the configured "
    "explicit-preference policy (fewer supported violations first, then more supported "
    "preference matches, then the original ranking order), with catalogue facts where "
    "available:"
)

#: Footer clauses for the reranked response.  Each is a statement about *policy
#: adherence*, never about product quality or relevance.
_RERANK_FOOTER = (
    "Original SASRec ranks are shown so the upstream order stays auditable; the "
    "recommendation itself always comes from the sequential recommender.",
    "A preference match or violation describes what this candidate's catalogue "
    "metadata says about your stored explicit preferences. It is not a product-quality "
    "judgement, and policy adherence is not a relevance or satisfaction measure.",
)


def _preference_block_for_reranking(preferences: Any) -> list[str]:
    """Render the stored preferences that the reranking policy consumed.

    Unlike :func:`_preference_block` this says the preferences **were** used, because
    with M10D active they are exactly what the matching node evaluated.  It still makes
    no claim that a candidate satisfies them; that claim appears per candidate, and only
    where M10A produced it.
    """
    lines = _active_preference_lines(preferences)
    if not lines:
        return []
    return [
        "Your stated preferences (stored from your own messages; evaluated as explicit "
        "preference evidence by the reranking policy):",
        *[f"- {line}" for line in lines],
        "",
    ]


def _aligned_candidates(reranking: Any, enrichment: Any) -> list[tuple[Any, Any]]:
    """Pair each reranked candidate with the enriched candidate of the same identity.

    Alignment is by ``(parent_asin, item_id)``, never by position, so a reranked order
    of ``C, A, B`` prints ``C``'s metadata next to ``C`` rather than whatever metadata
    happened to sit at that index.  Any mismatch is a hard error: the graph must not
    print one product's facts beside another product's identity, and it must not paper
    over a reranker that changed the candidate set.
    """
    if len(reranking.candidates) != len(enrichment.items):
        raise AgentGraphError(
            "reranking changed the candidate count "
            f"({len(reranking.candidates)} != {len(enrichment.items)})"
        )

    by_identity = {
        (item.parent_asin, item.recommendation.item_id): item for item in enrichment.items
    }
    if len(by_identity) != len(enrichment.items):  # pragma: no cover - defensive
        raise AgentGraphError("the enriched candidate set contains a duplicate identity")

    pairs: list[tuple[Any, Any]] = []
    for candidate in reranking.candidates:
        item = by_identity.get((candidate.parent_asin, candidate.item_id))
        if item is None:
            raise AgentGraphError(
                "reranking produced a candidate that is not in the enriched candidate set"
            )
        pairs.append((candidate, item))
    return pairs


def _movement_line(candidate: Any) -> str | None:
    """Describe a rank movement using only directly supported facts.

    Deliberately does **not** use the M10B ``rerank_reason`` label: Milestone 10C
    established that its tail ``DETERMINISTIC_TIE_BREAK`` wording can be imprecise (the
    candidate actually lost on the original-rank key), so it is not used as a
    user-facing explanation.  Only ``original_rank``/``reranked_rank`` -- facts the
    reranker really produced -- are stated.
    """
    if not candidate.moved:
        return None
    return (
        f"   moved from rank {candidate.original_rank} to rank {candidate.reranked_rank} "
        "under the configured explicit-preference policy"
    )


def _evidence_line(candidate: Any) -> str:
    """The per-candidate M10A evidence counts, stated as counts only."""
    return (
        f"   preference evidence: {candidate.match_count} supported match(es), "
        f"{candidate.violation_count} supported violation(s), "
        f"{candidate.unknown_count} unknown"
    )


def _build_reranked_response(reranking: Any, enrichment: Any, preferences: Any = None) -> str:
    """Render the recommendation in **reranked** order, grounded and auditable.

    The candidate sequence follows ``reranking.candidates``; the catalogue facts come
    from each candidate's *own* enriched record, matched by identity.  No metadata is
    looked up again and no ranked value is recomputed -- this function only decides
    presentation.
    """
    if not reranking.candidates:
        return _NO_CANDIDATES_TEXT

    pairs = _aligned_candidates(reranking, enrichment)

    lines = [
        f"Top {len(pairs)} {_RERANK_HEADER}",
        "",
        *_preference_block_for_reranking(preferences),
    ]

    for candidate, item in pairs:
        lines.extend(
            [
                "",
                f"{candidate.reranked_rank}. {candidate.parent_asin} "
                f"(original SASRec rank {candidate.original_rank}, "
                f"ranking score {candidate.sasrec_score:+.4f})",
                _evidence_line(candidate),
            ]
        )
        movement = _movement_line(candidate)
        if movement is not None:
            lines.append(movement)

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
            *_RERANK_FOOTER,
            _SCORE_DISCLAIMER,
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
    preference_matcher, reranker:
        The optional Milestone 10D pair.  When both are supplied (which requires
        ``product_enricher``) the recommendation route becomes
        ``recommend -> enrich -> match_preferences -> rerank -> finalize``.  Neither is
        constructed by the graph, and supplying only one is a configuration error.
    """

    def __init__(
        self,
        decision_model: DecisionModel,
        tool: RecommendationTool,
        product_enricher: ProductEnricherLike | None = None,
        memory_service: PreferenceMemoryLike | None = None,
        user_key: str | None = None,
        augment_query_with_preferences: bool = True,
        preference_matcher: PreferenceMatcherLike | None = None,
        reranker: PreferenceRerankerLike | None = None,
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
        if memory_service is not None:
            for method in ("get_active_preferences", "process_turn"):
                if not callable(getattr(memory_service, method, None)):
                    raise AgentConfigurationError(
                        f"memory_service must provide a callable {method}() method"
                    )
            if not isinstance(user_key, str) or not user_key.strip():
                raise AgentConfigurationError(
                    "a non-empty user_key is required when memory_service is configured"
                )
        # Milestone 10D dependency matrix.  The two preference collaborators are a
        # **pair**: evidence without a policy (or a policy without evidence) would be a
        # half-configured reranking stage, and the graph will not guess the missing half.
        if (preference_matcher is None) != (reranker is None):
            raise AgentConfigurationError(
                "preference_matcher and reranker must be configured together; a "
                "half-configured preference reranking stage is not supported"
            )
        if preference_matcher is not None:
            if not callable(getattr(preference_matcher, "match", None)):
                raise AgentConfigurationError(
                    "preference_matcher must provide a callable "
                    "match(candidates=..., preferences=...) method"
                )
            if not callable(getattr(reranker, "rerank", None)):
                raise AgentConfigurationError(
                    "reranker must provide a callable rerank(report) method"
                )
            if product_enricher is None:
                raise AgentConfigurationError(
                    "preference_matcher requires product_enricher: Milestone 10A evidence "
                    "is read from the metadata the Milestone 8 enricher attached to the "
                    "current candidates, so matching cannot run without it"
                )
        self._decision_model = decision_model
        self._tool = tool
        self._product_enricher = product_enricher
        self._memory_service = memory_service
        self._user_key = user_key.strip() if isinstance(user_key, str) else None
        self._augment_query_with_preferences = bool(augment_query_with_preferences)
        self._preference_matcher = preference_matcher
        self._reranker = reranker
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

    @property
    def memory_service(self) -> PreferenceMemoryLike | None:
        """The injected preference-memory service, or ``None`` when memory is off."""
        return self._memory_service

    @property
    def user_key(self) -> str | None:
        """The memory namespace this graph instance operates in."""
        return self._user_key

    @property
    def uses_memory(self) -> bool:
        """True when the run loads and persists conversational preference memory."""
        return self._memory_service is not None

    @property
    def augments_query_with_preferences(self) -> bool:
        """True when active preferences are appended to the retrieval query."""
        return self._memory_service is not None and self._augment_query_with_preferences

    @property
    def preference_matcher(self) -> PreferenceMatcherLike | None:
        """The injected M10A matcher, or ``None`` when evidence is not produced."""
        return self._preference_matcher

    @property
    def reranker(self) -> PreferenceRerankerLike | None:
        """The injected M10B reranker, or ``None`` when reranking is off."""
        return self._reranker

    @property
    def matches_preferences(self) -> bool:
        """True when the recommendation route includes an M10A evidence node."""
        return self._preference_matcher is not None

    @property
    def reranks_preferences(self) -> bool:
        """True when the recommendation route includes an M10B reranking node."""
        return self._reranker is not None

    # -- construction ------------------------------------------------------ #

    def _build(self) -> Any:
        """Assemble and compile the state graph.

        Topology is conditional, and each condition preserves the previously accepted
        shape when its collaborator is absent:

        * **no enricher, no memory** -- the accepted Milestone 7B/7C topology,
          ``decide -> {finalize | recommend -> finalize}``, unchanged;
        * **enricher present** -- an ``enrich`` node is inserted between ``recommend``
          and ``finalize`` (Milestone 8);
        * **matcher and reranker present** -- ``match_preferences`` and ``rerank`` are
          appended after ``enrich``, so the recommendation route becomes
          ``recommend -> enrich -> match_preferences -> rerank -> finalize``
          (Milestone 10D);
        * **memory present** -- a ``load_memory`` entry node runs before ``decide``
          (read-only) and a ``persist_memory`` node runs after ``finalize`` on both
          routes, so the direct route can read and update conversational memory
          without ever invoking the recommender.

        The direct route is untouched by Milestone 10D: it never reaches ``recommend``,
        ``enrich``, ``match_preferences`` or ``rerank``.

        The graph never constructs a metadata store, a memory store, a matcher or a
        reranker itself: each exists only because a caller injected it.
        """
        builder: StateGraph = StateGraph(AgentGraphState)
        builder.add_node(NODE_DECIDE, self._decide_node)
        builder.add_node(NODE_RECOMMEND, self._recommend_node)
        builder.add_node(NODE_FINALIZE, self._finalize_node)

        entry_node = START
        if self._memory_service is not None:
            builder.add_node(NODE_LOAD_MEMORY, self._load_memory_node)
            builder.add_edge(START, NODE_LOAD_MEMORY)
            entry_node = NODE_LOAD_MEMORY
        builder.add_edge(entry_node, NODE_DECIDE)

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
            if self._preference_matcher is None:
                builder.add_edge(NODE_ENRICH, NODE_FINALIZE)
            else:
                # Milestone 10D: evidence over the enriched candidates, then the
                # accepted policy.  Both nodes are on the recommendation route only, so
                # the direct route still touches neither.
                builder.add_node(NODE_MATCH_PREFERENCES, self._match_preferences_node)
                builder.add_node(NODE_RERANK, self._rerank_node)
                builder.add_edge(NODE_ENRICH, NODE_MATCH_PREFERENCES)
                builder.add_edge(NODE_MATCH_PREFERENCES, NODE_RERANK)
                builder.add_edge(NODE_RERANK, NODE_FINALIZE)

        if self._memory_service is None:
            builder.add_edge(NODE_FINALIZE, END)
        else:
            builder.add_node(NODE_PERSIST_MEMORY, self._persist_memory_node)
            builder.add_edge(NODE_FINALIZE, NODE_PERSIST_MEMORY)
            builder.add_edge(NODE_PERSIST_MEMORY, END)
        return builder.compile()

    # -- nodes ------------------------------------------------------------- #

    def _load_memory_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Read the user's active preference memory **without writing anything**.

        This node is reached before the decision, so both routes see the same
        snapshot, and a read cannot change memory.  It never receives or returns
        trusted interaction history.

        Turn ordering is deliberate::

            load_memory (read)  ->  decide  ->  [recommend -> enrich]  ->  finalize
                                                                            |
                                                              persist_memory (write)

        The snapshot this node reads is what the retrieval query and the rendered
        response are built from, so memory changes take effect from the following
        turn.  A turn therefore cannot observe its own write, and no read is ever
        influenced by a write from the same turn.
        """
        if self._memory_service is None or self._user_key is None:  # pragma: no cover
            return {}
        snapshot = self._memory_service.get_active_preferences(self._user_key)
        return {"preference_snapshot": snapshot}

    def _persist_memory_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Extract and persist explicit preferences from the user's own message.

        Only ``user_message`` is passed, because only user-authored text is eligible
        for extraction.  The node writes conversational preference memory and returns
        nothing that could influence candidates.
        """
        if self._memory_service is None or self._user_key is None:  # pragma: no cover
            return {}

        user_message = state.get("user_message")
        if not isinstance(user_message, str) or not user_message.strip():
            return {}

        turn_id = state.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id.strip():
            # Idempotency needs a turn identity; without one a stable per-message id
            # is derived so re-processing the same text is still deduplicated.
            turn_id = f"auto-{abs(hash(user_message)) % (10 ** 12)}"

        result = self._memory_service.process_turn(
            user_key=self._user_key,
            user_message=user_message,
            turn_id=turn_id,
        )
        # Writes happen after the response has been produced, and this node does not
        # rewrite ``preference_snapshot``.  Consequences, all deliberate:
        #
        #   * the retrieval query and the rendered preference block are built from the
        #     snapshot loaded at the start of the turn, so a preference stated in *this*
        #     message takes effect from the next turn;
        #   * a turn can therefore never appear to have let its own statement influence
        #     the candidates it returned;
        #   * candidate identity, count, rank and score are untouched either way.
        #
        # The write is persisted before the run ends, so nothing is lost.
        return {"memory_update": result}

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
        if self.augments_query_with_preferences:
            query = _memory_context(query, state.get("preference_snapshot"))
        return {"enrichment": self._product_enricher.enrich(result, query)}

    def _match_preferences_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Produce Milestone 10A evidence over the **already-enriched** candidates.

        Inputs, and nothing else:

        * ``enrichment`` -- the M8 candidate set, whose metadata is the only evidence
          source; the matcher has no catalogue handle, so no new product can enter;
        * ``preference_snapshot`` -- the M9 snapshot loaded at the **start** of this
          turn.  It is read, never refreshed here, so one turn uses one coherent
          snapshot even if ``persist_memory`` later writes to the store.

        Preferences are never inferred from trusted interaction history, SASRec scores,
        candidate identities or retrieved evidence: they come from the snapshot alone.
        With no memory configured the snapshot is absent and an empty preference
        sequence is passed, which is the documented way the matcher runs with zero
        active preferences -- it yields an empty-evidence report rather than being
        skipped, so the reranking stage still sees a well-formed report.

        Failures propagate.  The node never substitutes fabricated or partial evidence.
        """
        if self._preference_matcher is None:  # pragma: no cover - node not built then
            raise AgentGraphError("no preference matcher is configured")

        enrichment = state.get("enrichment")
        if enrichment is None:
            raise AgentGraphError(
                "the recommendation route produced no enriched candidates to match"
            )

        # ``None`` means "no memory configured"; ``()`` reaches the matcher as the
        # explicit "no active preferences" sequence.  Both yield empty evidence.
        snapshot = state.get("preference_snapshot")
        preferences = () if snapshot is None else snapshot
        return {
            "preference_evidence": self._preference_matcher.match(
                candidates=enrichment.items, preferences=preferences
            )
        }

    def _rerank_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Apply the accepted Milestone 10B policy to the evidence report.

        The graph delegates the ordering: it never sorts candidates itself, so the
        canonical key exists in exactly one place.  The Tool result, the enrichment and
        the evidence report are left exactly as they were -- only the derived
        ``reranking`` channel is written.

        Failures propagate, so a run can never present un-reranked output as if the
        policy had been applied.
        """
        if self._reranker is None:  # pragma: no cover - node not built then
            raise AgentGraphError("no reranker is configured")

        report = state.get("preference_evidence")
        if report is None:
            raise AgentGraphError(
                "the recommendation route produced no preference evidence to rerank"
            )
        return {"reranking": self._reranker.rerank(report)}

    def _finalize_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Produce the run's final text and route tag.

        The presentation branch is chosen by **which structured stage actually ran**,
        in order of derivation:

        1. ``reranking`` present -- render the M10B order (Milestone 10D);
        2. else ``enrichment`` present -- render the M8 order (Milestone 8);
        3. else -- render the Tool's order (Milestone 7B/7C).

        It is deliberately not a string check: a candidate list is presented in the
        reranked order only when a real ``RerankingReport`` exists.
        """
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
            reranking = state.get("reranking")
            if reranking is not None:
                enrichment = state.get("enrichment")
                if enrichment is None:
                    raise AgentGraphError(
                        "reranking is present but the enriched candidate set is missing"
                    )
                return {
                    "route": ROUTE_RECOMMEND,
                    "final_response": _build_reranked_response(
                        reranking, enrichment, state.get("preference_snapshot")
                    ),
                }
            enrichment = state.get("enrichment")
            if enrichment is not None:
                return {
                    "route": ROUTE_RECOMMEND,
                    "final_response": _build_grounded_response(
                        enrichment, state.get("preference_snapshot")
                    ),
                }
            return {
                "route": ROUTE_RECOMMEND,
                "final_response": _build_recommendation_response(
                    result, state.get("preference_snapshot")
                ),
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
    memory_service: PreferenceMemoryLike | None = None,
    user_key: str | None = None,
    augment_query_with_preferences: bool = True,
    preference_matcher: PreferenceMatcherLike | None = None,
    reranker: PreferenceRerankerLike | None = None,
) -> AgentGraph:
    """Construct the agent graph (factory form).

    Without ``product_enricher`` this builds the accepted Milestone 7B/7C graph;
    supplying one inserts the Milestone 8 enrichment node on the recommendation route.
    Supplying ``memory_service`` adds the Milestone 9 read/persist nodes and requires
    ``user_key``.

    Supplying ``preference_matcher`` **and** ``reranker`` (together, and with an
    ``product_enricher``) adds the Milestone 10D evidence and reranking nodes.  Memory
    is not required for that pair: without it the matcher receives an empty preference
    sequence and the reranker is order-preserving.
    """
    return AgentGraph(
        decision_model,
        tool,
        product_enricher=product_enricher,
        memory_service=memory_service,
        user_key=user_key,
        augment_query_with_preferences=augment_query_with_preferences,
        preference_matcher=preference_matcher,
        reranker=reranker,
    )
