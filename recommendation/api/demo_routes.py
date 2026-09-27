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
    DECISION_MODE_DETERMINISTIC,
    DECISION_MODE_LLM,
    DEFAULT_DECISION_MODE,
    AgentRecommendRequest,
    AgentRecommendResponse,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    DecisionModesResponse,
    DecisionModeView,
    DemoAgentError,
    DemoHealthResponse,
    DemoProfileView,
    DemoRuntime,
    DemoRuntimeError,
    ProfileListResponse,
    ResetResponse,
    SessionCapacityExceeded,
    SessionResponse,
    SessionStateResponse,
    TurnDecision,
    UnknownProfile,
    UnknownSession,
    UnsupportedDecisionMode,
    active_preference_views,
    build_chat_response,
)
from recommendation.demo.control_plane import CONTROL_PLANE_GRAPH
from recommendation.demo.llm_policy import LLMPolicyError, provider_availability
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

#: Why the LLM decision mode cannot run, in words a browser can show.  One entry per stable
#: machine-readable reason the availability probe returns, plus a fallback that never invents a
#: cause.  Each states that nothing was executed, because a mode that could not start must not
#: look like a turn that ran.
LLM_REASON_DETAIL: dict[str, str] = {
    "missing_api_key": (
        "no provider credential is configured (set AGENTRECX_LLM_API_KEY), so no LLM turn "
        "can be served"
    ),
    "not_configured": (
        "no provider is configured (set AGENTRECX_LLM_BASE_URL and AGENTRECX_LLM_MODEL), so "
        "no LLM turn can be served"
    ),
    "control_plane_unsupported": (
        "this deployment serves the alpha loop control plane; the LLM decision mode runs on "
        "the accepted graph path only"
    ),
}

#: Fallback for an availability reason this layer does not recognise.
LLM_UNAVAILABLE_FALLBACK = "the LLM decision mode is not available on this deployment"

#: The deterministic mode's availability detail: it needs nothing, so it can never be blocked
#: by configuration.
DETERMINISTIC_DETAIL = "runs offline; no provider, credential or network call is used"

#: Failure details.  Each names the failure and states the one fact that matters for trust: the
#: failed turn changed no preference memory, because the plan is obtained before anything runs.
LLM_TIMEOUT_DETAIL = (
    "the provider call timed out; the turn stopped before any trusted component ran, so no "
    "preference was changed"
)
LLM_INVALID_PLAN_DETAIL = (
    "the model did not return a valid decision plan; every candidate plan was rejected before "
    "anything ran, so no preference was changed"
)
LLM_PROVIDER_ERROR_DETAIL = (
    "the provider call failed; the turn stopped before any trusted component ran, so no "
    "preference was changed"
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
    unsupported candidate-source plan            422     ``unsupported_source_plan``
    unknown decision mode                        422     ``unsupported_decision_mode``
    LLM mode not configured                      503     ``llm_unavailable``
    LLM provider call timed out                  504     ``llm_timeout``
    LLM plan not valid                           502     ``llm_invalid_plan``
    LLM provider call failed                     502     ``llm_provider_error``
    anything else                               502     ``demo_backend_failed``
    ==========================================  ======  ==========================

    Nothing here is reachable by client input: every branch is a backend failure, and the
    client-visible detail is written by this function rather than taken from the
    exception, so no internal text leaks.
    """
    if isinstance(exc, DemoHTTPError):
        # Already mapped by this layer (for example by the chat controller's mode resolution).
        # Re-mapping it would replace a precise status with the generic catch-all.
        return exc
    if isinstance(exc, UnknownSession):
        return DemoHTTPError(404, "session_not_found", str(exc))
    if isinstance(exc, UnknownProfile):
        return DemoHTTPError(404, "unknown_profile", str(exc))
    if isinstance(exc, UnsupportedDecisionMode):
        return DemoHTTPError(
            422, "unsupported_decision_mode", "the requested decision mode does not exist"
        )
    if isinstance(exc, SessionCapacityExceeded):
        return DemoHTTPError(503, "session_capacity_exceeded", CAPACITY_DETAIL)
    if isinstance(exc, DemoRuntimeError):
        return DemoHTTPError(503, "demo_unavailable", UNAVAILABLE_DETAIL)
    if isinstance(exc, LLMPolicyError):
        return _llm_failure(exc)
    if isinstance(exc, DemoAgentError):
        # The agent service already wrote a client-safe detail; its code decides the status.
        if exc.code == "unsupported_source_plan":
            return DemoHTTPError(422, "unsupported_source_plan", str(exc))
        return DemoHTTPError(502, "agent_failed", "the agent could not complete this request")
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


# --------------------------------------------------------------------------- #
# decision modes
# --------------------------------------------------------------------------- #


def _llm_availability(runtime: DemoRuntime | None) -> dict[str, Any]:
    """Whether the LLM decision mode can run on this deployment, from execution state only.

    Two independent conditions, both read rather than assumed: a provider must be configured
    (the policy layer's own probe), and the turn must be servable by the control plane the LLM
    mode composes (the accepted graph).  Reporting a mode as available when it could not run
    would be exactly the silent-fallback failure this endpoint exists to prevent.
    """
    state = dict(provider_availability())
    if runtime is not None and runtime.control_plane != CONTROL_PLANE_GRAPH:
        return {
            "available": False,
            "reason": "control_plane_unsupported",
            "provider": state.get("provider"),
            "model": state.get("model"),
        }
    return state


def _llm_unavailable_detail(reason: Any) -> str:
    """Authored explanation for an unavailable LLM mode, or a cause-free fallback."""
    if isinstance(reason, str) and reason in LLM_REASON_DETAIL:
        return LLM_REASON_DETAIL[reason]
    return LLM_UNAVAILABLE_FALLBACK


def _llm_failure(exc: LLMPolicyError) -> DemoHTTPError:
    """Map a policy-layer failure onto a documented status, detail and error code.

    The code is the policy layer's own stable code; the detail is written here, so no provider
    prose and no prompt text can reach a client.  Every branch states the same trust fact: the
    turn stopped before a trusted component ran, so no preference was changed.
    """
    code = str(getattr(exc, "code", "llm_policy_error"))
    if code == "timeout":
        return DemoHTTPError(504, "llm_timeout", LLM_TIMEOUT_DETAIL)
    if code == "invalid_plan":
        return DemoHTTPError(502, "llm_invalid_plan", LLM_INVALID_PLAN_DETAIL)
    if code == "not_configured":
        return DemoHTTPError(503, "llm_unavailable", _llm_unavailable_detail(code))
    return DemoHTTPError(502, "llm_provider_error", LLM_PROVIDER_ERROR_DETAIL)


def _resolve_decision_mode(runtime: DemoRuntime, payload: Any, session: Any) -> str:
    """Decide which policy serves this turn, failing closed when the mode cannot run.

    The request may switch the mode; omitting it keeps the session's mode, so a client that
    never sends the field stays deterministic for the whole session.  Requesting the LLM mode
    when it cannot run is an error - never a quiet downgrade to the deterministic path.
    """
    mode = getattr(payload, "decision_mode", None) or session.decision_mode
    if mode not in (DECISION_MODE_DETERMINISTIC, DECISION_MODE_LLM):
        raise DemoHTTPError(
            422, "unsupported_decision_mode", "the requested decision mode does not exist"
        )
    if mode == DECISION_MODE_LLM:
        availability = _llm_availability(runtime)
        if not availability["available"]:
            raise DemoHTTPError(
                503, "llm_unavailable", _llm_unavailable_detail(availability.get("reason"))
            )
    return mode


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

    @router.get("/decision-modes", response_model=DecisionModesResponse)
    def decision_modes(request: Request) -> DecisionModesResponse:
        """Report which decision modes this deployment can actually serve.

        The browser reads this before offering the choice, so the LLM option is disabled with
        the real reason instead of failing on the first turn.  Availability is probed from
        execution state on every call, so a deployment that gains a credential does not need a
        restart to advertise it.
        """
        runtime = _runtime(request)
        availability = _llm_availability(runtime)
        return DecisionModesResponse(
            default=DEFAULT_DECISION_MODE,
            modes=(
                DecisionModeView(
                    mode=DECISION_MODE_DETERMINISTIC,
                    label="Deterministic",
                    available=True,
                    reason=None,
                    detail=DETERMINISTIC_DETAIL,
                    provider=None,
                    model=None,
                ),
                DecisionModeView(
                    mode=DECISION_MODE_LLM,
                    label="LLM Agent",
                    available=bool(availability["available"]),
                    reason=availability.get("reason"),
                    detail=(
                        None
                        if availability["available"]
                        else _llm_unavailable_detail(availability.get("reason"))
                    ),
                    provider=availability.get("provider"),
                    model=availability.get("model"),
                ),
            ),
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
        """Create a fresh, isolated demo session bound to a demo profile.

        The session's decision mode may be chosen here; asking for the LLM mode on a deployment
        that cannot serve it is refused, so a session never starts in a mode that would have to
        fall back on its first turn.
        """
        runtime = _runtime(request)
        if payload.decision_mode == DECISION_MODE_LLM:
            availability = _llm_availability(runtime)
            if not availability["available"]:
                raise DemoHTTPError(
                    503, "llm_unavailable", _llm_unavailable_detail(availability.get("reason"))
                )
        try:
            session = runtime.sessions.create(
                payload.profile_id, decision_mode=payload.decision_mode
            )
        except Exception as exc:  # noqa: BLE001 - mapped, never leaked
            raise map_demo_exception(exc) from exc
        return SessionResponse(
            session_id=session.session_id,
            profile=_profile_view(runtime.profiles[session.profile_id]),
            turn=session.turns_completed,
            decision_mode=session.decision_mode,
        )

    @router.get("/sessions/{session_id}", response_model=SessionStateResponse)
    def session_state(session_id: str, request: Request) -> SessionStateResponse:
        """Return the session's safe state: metadata, turn count, mode and ACTIVE preferences."""
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
            decision_mode=session.decision_mode,
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

        Two decision modes share this endpoint.  ``deterministic`` is the accepted path,
        unchanged.  ``llm`` obtains one validated plan from the policy layer **before** any
        trusted component runs, then executes that plan through the same accepted components;
        a failure there is reported as a failed turn and is never answered by the deterministic
        path instead.
        """
        runtime = _runtime(request)
        try:
            with runtime.sessions.turn(session_id) as allocation:
                session = allocation.session
                # Resolved (and therefore validated) before anything runs, and made the
                # session's mode before the turn executes: a failed LLM turn must leave the
                # session in LLM mode so the next turn fails the same way rather than quietly
                # answering from the deterministic path while the browser shows "LLM Agent".
                mode = _resolve_decision_mode(runtime, payload, session)
                session.switch_decision_mode(mode)
                # The runtime owns which control plane serves the turn.  Both return the
                # accepted AgentGraphState, so this controller stays control-plane agnostic.
                agent_input = _agent_input(
                    payload.message, session.trusted_user_history, allocation.turn_id
                )
                if mode == DECISION_MODE_LLM:
                    state, run = runtime.llm_turn(
                        payload.k, user_key=session.user_key, agent_input=agent_input
                    )
                    decision = TurnDecision(
                        decision_mode=mode,
                        provider=run.provider,
                        model=run.model,
                        proposed_route=run.proposed_route,
                        preference_actions=run.preference_actions(),
                    )
                else:
                    state = runtime.turn(
                        payload.k, user_key=session.user_key, agent_input=agent_input
                    )
                    decision = TurnDecision(decision_mode=mode)
                return build_chat_response(
                    state,
                    session_id=session.session_id,
                    turn_id=allocation.turn_id,
                    turn_number=allocation.turn_number,
                    decision=decision,
                )
        except LLMPolicyError as exc:
            # A decision-policy failure is always resolved by this layer: the failure occurred
            # before any trusted component ran, so the detail may state that no preference
            # changed, and no provider prose is used.
            raise _llm_failure(exc) from exc
        except Exception as exc:  # noqa: BLE001 - mapped, never leaked
            raise map_demo_exception(exc) from exc

    @router.post("/agent/recommend", response_model=AgentRecommendResponse)
    def agent_recommend(
        payload: AgentRecommendRequest, request: Request
    ) -> AgentRecommendResponse:
        """Run one stateless agent turn through the existing control plane.

        This endpoint adds no agent behaviour: it composes the runtime's existing engine,
        catalogue, tool and memory service with the accepted bounded loop and multi-source
        candidate plane, then projects the result.  It creates no session, so the caller
        supplies its own history and the request is the only channel history can arrive
        through - exactly as the accepted ``POST /v1/recommend`` does for the model path.

        Preference memory is touched only when ``user_key`` is supplied; without it the run
        neither reads nor writes memory, which is what makes the offline demo reproducible.
        """
        runtime = _runtime(request)
        if not runtime.ready:
            raise DemoHTTPError(503, "demo_unavailable", UNAVAILABLE_DETAIL)
        try:
            return runtime.agent_service().run(payload)
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
