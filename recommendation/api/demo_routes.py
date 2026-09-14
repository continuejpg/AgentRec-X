"""FastAPI routes for the Milestone 11 multi-turn web demo.

The controller is deliberately thin.  It may::

    validate the request
    look up the session
    read the application-owned demo history off that session
    allocate a server-owned turn id
    invoke the accepted AgentGraph
    serialize the result through the explicit adapter
    map domain errors onto HTTP

It must never::

    score candidates        mask candidates       sort candidates
    retrieve metadata       compute BM25          extract preferences
    match preferences       rerank                resolve memory conflicts
    render grounded prose

Every one of those lives behind an accepted backend abstraction
(:mod:`recommendation.tools`, :mod:`recommendation.rag`,
:mod:`recommendation.memory`, :mod:`recommendation.preference_matching`,
:mod:`recommendation.reranking`, :mod:`recommendation.agent`).  The demo module imports
those packages **only** inside :mod:`recommendation.demo.runtime` when composing the
runtime; this controller touches just the runtime, the session manager and the
serializer.

The endpoints are synchronous ``def`` functions on purpose: the accepted graph performs
blocking CPU inference, so FastAPI runs each request in its worker thread and the event
loop stays responsive.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from recommendation.agent import AgentGraphError, MalformedDecision
from recommendation.demo import (
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    DemoHealthResponse,
    DemoProfileView,
    DemoRuntime,
    DemoRuntimeError,
    ProfileListResponse,
    ResetResponse,
    SessionCapacityExceeded,
    SessionResponse,
    SessionStateResponse,
    UnknownProfile,
    UnknownSession,
    active_preference_views,
    build_chat_response,
)
from recommendation.reranking import RerankingError
from recommendation.tools import RecommendationToolError

__all__ = [
    "DEMO_ROUTER_PREFIX",
    "DEMO_TAGS",
    "DemoHTTPError",
    "build_demo_router",
    "map_demo_exception",
]

DEMO_ROUTER_PREFIX = "/v1/demo"
DEMO_TAGS = ["demo"]

#: Detail text for a successful reset.  Kept as a constant so the API documentation and
#: the tests assert the same wording.
RESET_DETAIL = (
    "The demo session was removed from the live registry and its preference-memory "
    "namespace was retired; it is unreachable through this API. Create a new session to "
    "continue. Only this session was affected."
)

#: Detail text for an unready demo.  Never echoes a path or an exception message.
UNAVAILABLE_DETAIL = (
    "the demo backend is not ready; check the server startup log and GET /v1/demo/health"
)

#: Detail text for a full session registry.  Written here rather than taken from the
#: exception, so every 5xx detail is authored by this layer.
CAPACITY_DETAIL = (
    "the demo has reached its live-session limit; reset a session and try again"
)


class DemoHTTPError(Exception):
    """A demo failure already mapped onto a stable HTTP status and error code.

    Raised by the controller and rendered by the handler registered in
    :func:`recommendation.api.app.create_app`, so every demo error body keeps the
    accepted ``{"error", "detail"}`` shape and never carries a stack trace, a path or an
    internal identifier.
    """

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


def map_demo_exception(exc: BaseException) -> DemoHTTPError:
    """Map a domain failure onto a documented HTTP status.

    Mapping, in order of specificity:

    ==========================================  ======  ==========================
    failure                                     status  error code
    ==========================================  ======  ==========================
    unknown / expired / reset session           404     ``session_not_found``
    unknown demo profile                        404     ``unknown_profile``
    live-session capacity reached               503     ``session_capacity_exceeded``
    demo backend could not be composed          503     ``demo_unavailable``
    missing or unknown trusted history          422     ``invalid_history``
    invalid tool request                        422     ``invalid_request``
    RecommendationTool failure                  502     ``recommendation_failed``
    preference matching / reranking failure     502     ``preference_stage_failed``
    agent orchestration failure                 502     ``agent_failed``
    anything else                               502     ``demo_backend_failed``
    ==========================================  ======  ==========================

    Nothing here is reachable by client input: every branch is a backend failure, and the
    client-visible detail is written by this function rather than taken from the
    exception, so no internal text leaks.
    """
    if isinstance(exc, UnknownSession):
        return DemoHTTPError(404, "session_not_found", str(exc))
    if isinstance(exc, UnknownProfile):
        return DemoHTTPError(404, "unknown_profile", str(exc))
    if isinstance(exc, SessionCapacityExceeded):
        return DemoHTTPError(503, "session_capacity_exceeded", CAPACITY_DETAIL)
    if isinstance(exc, DemoRuntimeError):
        return DemoHTTPError(503, "demo_unavailable", UNAVAILABLE_DETAIL)
    if isinstance(exc, MalformedDecision):
        return DemoHTTPError(
            502, "agent_failed", "the agent could not produce a well-formed recommendation plan"
        )
    if isinstance(exc, AgentGraphError):
        return DemoHTTPError(502, "agent_failed", "the agent route failed to complete")
    if isinstance(exc, RerankingError):
        return DemoHTTPError(
            502, "preference_stage_failed", "the preference reranking stage failed"
        )
    # Tool domain errors carry stable codes; reuse them rather than inventing new ones.
    if isinstance(exc, RecommendationToolError):
        code = getattr(exc, "code", "recommendation_tool_error")
        if code in ("missing_user_history", "unknown_history_item"):
            return DemoHTTPError(422, "invalid_history", "the demo history is not usable")
        if code == "invalid_request":
            return DemoHTTPError(422, "invalid_request", "the recommendation request is invalid")
        return DemoHTTPError(
            502, "recommendation_failed", "the recommender could not complete this request"
        )
    return DemoHTTPError(502, "demo_backend_failed", "the demo backend failed to complete this turn")


def _runtime(request: Request) -> DemoRuntime:
    """Return the process-scoped demo runtime, or report the demo as unavailable."""
    runtime = getattr(request.app.state, "demo_runtime", None)
    if runtime is None:
        raise DemoHTTPError(503, "demo_unavailable", UNAVAILABLE_DETAIL)
    return runtime


def _profile_view(profile: Any) -> DemoProfileView:
    """Public view of a demo profile: identity and history shape, never the history."""
    return DemoProfileView(
        profile_id=profile.profile_id,
        display_name=profile.display_name,
        history_length=profile.history_length,
        history_distinct=profile.history_distinct,
    )


def build_demo_router() -> APIRouter:
    """Build the demo router.

    A factory rather than a module-level singleton so a test can build an isolated app
    without sharing router state.
    """
    router = APIRouter(prefix=DEMO_ROUTER_PREFIX, tags=DEMO_TAGS)

    @router.get("/health", response_model=DemoHealthResponse)
    def demo_health(request: Request) -> DemoHealthResponse:
        """Demo readiness: model, metadata, profiles and the live-session registry."""
        runtime = _runtime(request)
        ready = runtime.ready and runtime.model_loaded
        return DemoHealthResponse(
            status="ok" if ready else "unavailable",
            model_loaded=runtime.model_loaded,
            metadata_loaded=runtime.metadata_loaded,
            demo_ready=runtime.ready,
            profiles=len(runtime.profiles),
            active_sessions=len(runtime.sessions),
            max_sessions=runtime.sessions.max_sessions,
            detail=None if ready else "the demo backend is not fully constructed",
        )

    @router.get("/profiles", response_model=ProfileListResponse)
    def list_profiles(request: Request) -> ProfileListResponse:
        """List the server-owned demo profiles a browser may start a session with."""
        runtime = _runtime(request)
        return ProfileListResponse(
            profiles=tuple(_profile_view(profile) for profile in runtime.profiles.values())
        )

    @router.post("/sessions", response_model=SessionResponse, status_code=201)
    def create_session(payload: CreateSessionRequest, request: Request) -> SessionResponse:
        """Create a fresh, isolated demo session bound to a demo profile."""
        runtime = _runtime(request)
        try:
            session = runtime.sessions.create(payload.profile_id)
        except Exception as exc:  # noqa: BLE001 - mapped, never leaked
            raise map_demo_exception(exc) from exc
        return SessionResponse(
            session_id=session.session_id,
            profile=_profile_view(runtime.profiles[session.profile_id]),
            turn=session.turns_completed,
        )

    @router.get("/sessions/{session_id}", response_model=SessionStateResponse)
    def session_state(session_id: str, request: Request) -> SessionStateResponse:
        """Return the session's safe state: metadata, turn count and ACTIVE preferences."""
        runtime = _runtime(request)
        try:
            session = runtime.sessions.get(session_id)
            snapshot = runtime.memory_service.get_active_preferences(session.user_key)
        except Exception as exc:  # noqa: BLE001 - mapped, never leaked
            raise map_demo_exception(exc) from exc

        active = active_preference_views(snapshot)
        return SessionStateResponse(
            session_id=session.session_id,
            profile=_profile_view(runtime.profiles[session.profile_id]),
            turn=session.turns_completed,
            active_preferences=active,
            active_preference_count=len(active),
        )

    @router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
    def chat(session_id: str, payload: ChatRequest, request: Request) -> ChatResponse:
        """Run one conversational turn through the accepted agent graph.

        The whole turn -- turn-id allocation, history lookup, graph invocation and the
        Milestone 9 write -- happens while holding **this session's** lock, so two
        simultaneous messages to one session are serialised (distinct turn ids, no
        interleaved memory) while different sessions proceed independently.
        """
        runtime = _runtime(request)
        try:
            with runtime.sessions.turn(session_id) as allocation:
                session = allocation.session
                graph = runtime.graph_for(payload.k, user_key=session.user_key)
                state = graph.invoke(
                    _agent_input(payload.message, session.trusted_user_history, allocation.turn_id)
                )
                return build_chat_response(
                    state,
                    session_id=session.session_id,
                    turn_id=allocation.turn_id,
                    turn_number=allocation.turn_number,
                )
        except Exception as exc:  # noqa: BLE001 - mapped, never leaked
            raise map_demo_exception(exc) from exc

    @router.delete("/sessions/{session_id}", response_model=ResetResponse)
    def reset_session(session_id: str, request: Request) -> ResetResponse:
        """Reset a demo session.  Only the target session is affected."""
        runtime = _runtime(request)
        try:
            session = runtime.sessions.delete(session_id)
        except Exception as exc:  # noqa: BLE001 - mapped, never leaked
            raise map_demo_exception(exc) from exc
        # Drop the compiled graphs bound to this namespace: they are unreachable now.
        runtime.release_user_key(session.user_key)
        return ResetResponse(session_id=session.session_id, detail=RESET_DETAIL)

    return router


def _agent_input(message: str, history: tuple[str, ...], turn_id: str) -> Any:
    """Build the validated agent input for one turn.

    ``turn_id`` is server-owned and ``trusted_user_history`` comes from the session, not
    from the request body: the browser has no field through which either could arrive.
    """
    from recommendation.agent import AgentInput

    return AgentInput(user_message=message, trusted_user_history=history, turn_id=turn_id)
