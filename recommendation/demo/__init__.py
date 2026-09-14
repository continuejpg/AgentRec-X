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
from .runtime import (
    DEFAULT_DEMO_MEMORY_DB,
    DemoRuntime,
    DemoRuntimeError,
    build_demo_runtime,
    catalog_metadata_path,
    memory_database_path,
)
from .schemas import (
    DEMO_API_VERSION,
    MAX_MESSAGE_LENGTH,
    ActivePreferenceView,
    AuditView,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
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
)
from .serialization import (
    active_preference_views,
    build_audit,
    build_cards,
    build_chat_response,
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
)

__all__ = [
    "DEFAULT_DEMO_K",
    "DEFAULT_DEMO_MEMORY_DB",
    "DEFAULT_DEMO_PROFILE_COUNT",
    "DEFAULT_MAX_SESSIONS",
    "DEMO_API_VERSION",
    "MAX_MESSAGE_LENGTH",
    "ActivePreferenceView",
    "AuditView",
    "ChatRequest",
    "ChatResponse",
    "CreateSessionRequest",
    "DemoDecisionModel",
    "DemoError",
    "DemoHealthResponse",
    "DemoProfile",
    "DemoProfileView",
    "DemoRuntime",
    "DemoRuntimeError",
    "DemoSession",
    "DemoSessionManager",
    "EvidenceView",
    "MemoryUpdateView",
    "PreferenceMutationView",
    "ProductMetadataView",
    "ProfileListResponse",
    "RecommendationCard",
    "ResetResponse",
    "SessionCapacityExceeded",
    "SessionResponse",
    "SessionStateResponse",
    "TurnAllocation",
    "UnknownProfile",
    "UnknownSession",
    "active_preference_views",
    "build_audit",
    "build_cards",
    "build_chat_response",
    "build_demo_profiles",
    "build_demo_runtime",
    "catalog_metadata_path",
    "demo_profiles_from_artifact",
    "looks_like_recommendation",
    "memory_database_path",
    "memory_update_view",
    "movement_summary",
]

__version__ = "1.0.0"
