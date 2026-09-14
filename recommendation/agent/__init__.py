"""Minimal LangGraph agent orchestration (Milestone 7B).

This package proves the smallest useful orchestration contract around the already
accepted Milestone 7A Recommendation Tool:

::

    START -> decide -> (direct_response -> finalize -> END)
                    -> (recommend ------> finalize -> END)
                                        |
                              RecommendationTool
                                        |
                              SASRecInferenceEngine

Boundaries this package enforces:

* the **decision model is injected** (``decide(messages) -> AgentDecision``), so
  Milestone 7B needs no LLM provider SDK, no API key and no network;
* the decision model receives only the user message - never
  ``trusted_user_history``, internal item ids, encoded model histories, SASRec
  tensors, mapping internals or checkpoint internals;
* trusted history lives in application-owned graph state and is read only by the
  Tool node;
* the recommendation is produced only by the accepted Tool, in process.  The
  FastAPI service (Milestone 6) is an external boundary and is never called.

Deliberately absent: planner, intent classifier, ReAct loop, reflection, retries,
summarizer, critic, reranker, memory, RAG, vector store, embeddings,
product-metadata retrieval, semantic IDs, multi-agent hand-off.
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
    NODE_RECOMMEND,
    ROUTE_DIRECT,
    ROUTE_RECOMMEND,
    AgentConfigurationError,
    AgentGraph,
    AgentGraphError,
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
    "NODE_RECOMMEND",
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
