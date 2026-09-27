"""Multi-turn web demo for AgentRec-X (Milestone 11).

Exposes the already accepted backend as a browser-usable, multi-turn shopping-agent
demo::

    browser
      -> FastAPI demo endpoints        recommendation/api/demo_routes.py
      -> DemoSessionManager            this package: sessions
      -> AgentGraph                    recommendation/agent  (M7B -> M10D, unchanged)
      -> M7A Tool -> M6 SASRec -> M8 metadata/RAG
      -> M9 memory -> M10A evidence -> M10B reranking

Milestone 11 is a **productization/integration** milestone.  It adds no scoring, no
ranking policy, no retrieval rule and no preference semantics: the web layer validates
requests, looks up a session, allocates a server-owned turn id, calls the accepted
graph and serializes the result through an explicit whitelist.

Three distinct states, never conflated
--------------------------------------
=========================  ==========================================================
trusted behavioural        application-owned chronological ``parent_asin`` history,
history                    selected from a demo profile before the chat starts and
                           read-only for the whole session.  Chat text never reaches it.
preference memory          explicit conversational preferences (Milestone 9), scoped to
                           a per-session ``user_key``, applied from the *next* turn.
browser transcript         purely visual history in the page.  Not persisted server-side
                           and not part of either of the above.
=========================  ==========================================================

This is a **local research demo**, not a production authenticated service: session ids
are opaque capability tokens, there is no real authentication, and the live session
registry is process-local.
"""

from __future__ import annotations

from .decision import DEFAULT_DEMO_K, DemoDecisionModel, looks_like_recommendation
from .profiles import (
    DEFAULT_DEMO_PROFILE_COUNT,
    DemoProfile,
    build_demo_profiles,
    demo_profiles_from_artifact,
)
from .agent_policy import (
    DEFAULT_SOURCE_PLAN,
    SOURCE_PLANS,
    DemoSourcePlanPolicy,
    normalise_source_plan,
)
from .agent_schemas import (
    AGENT_API_VERSION,
    MAX_CONSTRAINTS,
    MAX_HISTORY_ITEMS,
    AgentCandidateView,
    AgentConstraintItem,
    AgentEligibilityView,
    AgentRecommendRequest,
    AgentRecommendResponse,
    AgentRecommendationView,
    AgentTerminalView,
    AgentTimingView,
    GroundedCandidateView,
    TrajectoryStepView,
    TrajectorySummaryView,
)
from .agent_service import DemoAgentError, DemoAgentService
from .runtime import (
    DEFAULT_DEMO_MEMORY_DB,
    DemoRuntime,
    DemoRuntimeError,
    build_demo_runtime,
    catalog_metadata_path,
    memory_database_path,
)
from .schemas import (
    DECISION_MODES,
    DECISION_MODE_DETERMINISTIC,
    DECISION_MODE_LLM,
    DEFAULT_DECISION_MODE,
    DEMO_API_VERSION,
    MAX_MESSAGE_LENGTH,
    ActivePreferenceView,
    AuditView,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    DecisionMode,
    DecisionModesResponse,
    DecisionModeView,
    DemoHealthResponse,
    DemoProfileView,
    EvidenceView,
    MemoryUpdateView,
    PreferenceMutationView,
    ProductMetadataView,
    ProfileListResponse,
    RecommendationCard,
    ResetResponse,
    SessionResponse,
    SessionStateResponse,
    TracePreferenceAction,
)
from .serialization import (
    TurnDecision,
    active_preference_views,
    build_audit,
    build_cards,
    build_chat_response,
    build_trace_with_decision,
    memory_update_view,
    movement_summary,
)
from .sessions import (
    DEFAULT_MAX_SESSIONS,
    DemoError,
    DemoSession,
    DemoSessionManager,
    SessionCapacityExceeded,
    TurnAllocation,
    UnknownProfile,
    UnknownSession,
    UnsupportedDecisionMode,
)

__all__ = [
    "AGENT_API_VERSION",
    "DECISION_MODES",
    "DECISION_MODE_DETERMINISTIC",
    "DECISION_MODE_LLM",
    "DEFAULT_DECISION_MODE",
    "DEFAULT_DEMO_K",
    "DEFAULT_DEMO_MEMORY_DB",
    "DEFAULT_DEMO_PROFILE_COUNT",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_SOURCE_PLAN",
    "DEMO_API_VERSION",
    "MAX_CONSTRAINTS",
    "MAX_HISTORY_ITEMS",
    "MAX_MESSAGE_LENGTH",
    "SOURCE_PLANS",
    "ActivePreferenceView",
    "AgentCandidateView",
    "AgentConstraintItem",
    "AgentEligibilityView",
    "AgentRecommendRequest",
    "AgentRecommendResponse",
    "AgentRecommendationView",
    "AgentTerminalView",
    "AgentTimingView",
    "AuditView",
    "ChatRequest",
    "ChatResponse",
    "CreateSessionRequest",
    "DecisionMode",
    "DecisionModeView",
    "DecisionModesResponse",
    "DemoAgentError",
    "DemoAgentService",
    "DemoDecisionModel",
    "DemoError",
    "DemoHealthResponse",
    "DemoProfile",
    "DemoProfileView",
    "DemoRuntime",
    "DemoRuntimeError",
    "DemoSession",
    "DemoSessionManager",
    "DemoSourcePlanPolicy",
    "EvidenceView",
    "GroundedCandidateView",
    "MemoryUpdateView",
    "PreferenceMutationView",
    "ProductMetadataView",
    "ProfileListResponse",
    "RecommendationCard",
    "ResetResponse",
    "SessionCapacityExceeded",
    "SessionResponse",
    "SessionStateResponse",
    "TracePreferenceAction",
    "TrajectoryStepView",
    "TrajectorySummaryView",
    "TurnAllocation",
    "TurnDecision",
    "UnknownProfile",
    "UnknownSession",
    "UnsupportedDecisionMode",
    "active_preference_views",
    "build_audit",
    "build_cards",
    "build_chat_response",
    "build_demo_profiles",
    "build_demo_runtime",
    "build_trace_with_decision",
    "catalog_metadata_path",
    "demo_profiles_from_artifact",
    "looks_like_recommendation",
    "memory_database_path",
    "memory_update_view",
    "movement_summary",
    "normalise_source_plan",
]

__version__ = "1.0.0"
