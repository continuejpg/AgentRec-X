"""LangGraph agent orchestration (Milestones 7B-10D).

This package proves the smallest useful orchestration contract around the already
accepted Milestone 7A Recommendation Tool, and composes the accepted Milestone 8, 9,
10A and 10B stages as optional injected collaborators:

::

    START -> load_memory -> decide
                             |-- direct_response -> finalize -> persist_memory -> END
                             |
                             `-- recommend -> enrich -> match_preferences -> rerank
                                                     -> finalize -> persist_memory -> END
                                                  |
                                        RecommendationTool
                                                  |
                                        SASRecInferenceEngine

Every stage after ``decide`` exists only when the corresponding collaborator was
injected, so the accepted Milestone 7B/7C, 8 and 9 configurations keep their exact
topology and behaviour.

Boundaries this package enforces:

* the **decision model is injected** (``decide(messages) -> AgentDecision``), so the
  agent needs no LLM provider SDK, no API key and no network;
* the decision model receives only the user message - never
  ``trusted_user_history``, internal item ids, encoded model histories, SASRec
  tensors, mapping internals or checkpoint internals;
* trusted history lives in application-owned graph state and is read only by the
  Tool node;
* the recommendation is produced only by the accepted Tool, in process.  The
  FastAPI service (Milestone 6) is an external boundary and is never called;
* preference evidence comes from the injected M10A matcher and the order from the
  injected M10B reranker.  The graph reimplements neither, and it never constructs a
  default one;
* the M10C offline evaluator is not imported by any runtime node.

Deliberately absent: planner, intent classifier, ReAct loop, reflection, retries,
summarizer, learned critic, learned reranker, LLM critic, vector store, embeddings,
semantic IDs, multi-agent hand-off.
"""

from __future__ import annotations

from .decision import (
    AGENT_DECISION_VERSION,
    AgentAction,
    AgentDecision,
    DecisionMessage,
    DecisionModel,
    MalformedDecision,
    build_decision_messages,
    parse_agent_decision,
)
from .graph import (
    AGENT_GRAPH_VERSION,
    NODE_DECIDE,
    NODE_ENRICH,
    NODE_FINALIZE,
    NODE_LOAD_MEMORY,
    NODE_MATCH_PREFERENCES,
    NODE_PERSIST_MEMORY,
    NODE_RECOMMEND,
    NODE_RERANK,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentConfigurationError,
    AgentGraph,
    AgentGraphError,
    PreferenceMatcherLike,
    PreferenceMemoryLike,
    PreferenceRerankerLike,
    ProductEnricherLike,
    build_agent_graph,
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
    "AGENT_DECISION_VERSION",
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
    "AgentDecision",
    "AgentGraph",
    "AgentGraphError",
    "AgentGraphState",
    "AgentInput",
    "DecisionMessage",
    "DecisionModel",
    "MalformedDecision",
    "PreferenceMatcherLike",
    "PreferenceMemoryLike",
    "PreferenceRerankerLike",
    "ProductEnricherLike",
    "TrustedHistory",
    "build_agent_graph",
    "build_decision_messages",
    "history_digest",
    "new_agent_state",
    "parse_agent_decision",
    "read_trusted_history",
]

__version__ = "0.1.0"
