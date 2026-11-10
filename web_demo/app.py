"""FastAPI BFF for the AgentRec-X Web Demo.

Endpoints (see ``docs/WEB_DEMO_CONTRACT.md``):
    GET  /api/health
    GET  /api/demo/config
    POST /api/demo/restart
    POST /api/sessions                      GET /api/sessions
    GET  /api/sessions/{session_id}         POST /api/sessions/{session_id}/reset
    GET  /api/preferences?session_id=
    POST /api/chat/stream                   (SSE)
    POST /api/products/details
    POST /api/products/compare
    POST /api/clarification/reply
    GET  /api/replay

The BFF binds 127.0.0.1 and is a local demo only.  It never imports recommendation core modules
directly: every product/preference/gate value comes from one of the seven P17 MCP tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from web_demo import demo_flow
from web_demo import viewmodels as vm
from web_demo.config import (
    DEFAULT_PROFILE_ID,
    NOTICES,
    PROFILE_BY_ID,
    VERIFIED_COMPARISON_IDS,
    Settings,
    limits_public,
    load_settings,
    modes_public,
)
from web_demo.mcp_manager import MCPManager, MCPManagerError
from web_demo.qwen_health import QwenHealthMonitor
from web_demo.sessions import DEFAULT_TITLE, Session, SessionStore, iso, utc_now

LOGGER = logging.getLogger("agentrecx.web_demo.bff")
HEARTBEAT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Request models: extra="forbid" so the browser can never inject identity or
# confirmation flags (user_id / confirmed_by_user / confirmation_source / ...).
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    selected_product_ids: list[str] | None = None
    mode: Literal["cpu_live", "qwen_live"] = "cpu_live"


class ProductDetailsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    product_ref: str = Field(min_length=1)


class ProductCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    product_refs: list[str] = Field(min_length=2, max_length=5)


class ClarificationReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1)
    clarification_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RestartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str | None = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(settings: Settings) -> list[logging.Handler]:
    """Attach real file logging for ``runs/web_demo/logs/bff.log``; return handlers to remove.

    When the one-command launcher runs the BFF with ``AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG=1`` the
    launcher already tees console output into ``bff.log``, so the BFF skips its own file handler
    to avoid duplicated lines.
    """
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agentrecx.web_demo")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate lines via the root (uvicorn/basicConfig) handler
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    bff_log = (settings.log_dir / "bff.log").resolve()
    added: list[logging.Handler] = []
    has_file = any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == bff_log
        for handler in logger.handlers
    )
    if not has_file and not settings.console_only_log:
        file_handler = logging.FileHandler(bff_log, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        added.append(file_handler)
    has_stream = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        added.append(stream_handler)
    return added


def _remove_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> None:
    for handler in handlers:
        logger.removeHandler(handler)
        with suppress(Exception):
            handler.close()


# ---------------------------------------------------------------------------
# SSE serialisation
# ---------------------------------------------------------------------------


def format_sse(item: dict[str, Any]) -> str:
    name = str(item.get("event") or "message")
    data = item.get("data") or {}
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # json.dumps escapes real newlines inside strings, so one data line per event is safe.
    return f"event: {name}\ndata: {encoded}\n\n"


def sse_response(events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    """Wrap an async event iterator in a text/event-stream response with keep-alives."""

    async def stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for item in events:
                    await queue.put(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as one real error event
                await queue.put(
                    {
                        "event": "error",
                        "data": {
                            "code": "internal_error",
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    }
                )
            finally:
                with suppress(Exception):
                    await queue.put(None)

        producer = asyncio.create_task(produce(), name="sse-producer")
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield format_sse(item)
        finally:
            producer.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await producer

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    manager: MCPManager | Any | None = None,
    sessions: SessionStore | None = None,
    qwen_health: QwenHealthMonitor | Any | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    manager = manager or MCPManager(settings)
    store = sessions or SessionStore()
    monitor = qwen_health or QwenHealthMonitor(settings)
    meta: dict[str, Any] = {"started_at": iso(utc_now()), "monotonic": time.monotonic()}
    handlers_added: list[logging.Handler] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        handlers_added.extend(configure_logging(settings))
        LOGGER.info(
            "BFF starting host=%s port=%s search_mode=%s writes=%s qwen_configured=%s",
            settings.host, settings.port, settings.search_mode, settings.allow_memory_writes,
            settings.qwen_live_configured,
        )
        manager.start()
        monitor.start()
        try:
            yield
        finally:
            LOGGER.info("BFF stopping; closing MCP clients")
            await monitor.stop()
            await manager.stop()
            _remove_handlers(logging.getLogger("agentrecx.web_demo"), handlers_added)
            handlers_added.clear()

    app = FastAPI(
        title="AgentRec-X Web Demo BFF",
        version="1.0.0",
        description="Local BFF over the P17 MCP tools; no fake data, no LLM.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.manager = manager
    app.state.sessions = store
    app.state.qwen_health = monitor
    app.state.meta = meta

    @app.exception_handler(MCPManagerError)
    async def mcp_error_handler(request: Request, exc: MCPManagerError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=exc.http_status, content=exc.as_payload())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": "request validation failed (unknown extra fields are forbidden)",
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        LOGGER.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000.0,
        )
        return response

    # -- helpers -------------------------------------------------------------
    def get_session(session_id: str) -> Session:
        session = store.get(session_id)
        if session is None:
            raise MCPManagerError(
                f"unknown session_id {session_id!r}",
                code="session_not_found",
                http_status=404,
            )
        return session

    def require_ready(session: Session) -> None:
        manager.ensure_ready(session.profile_id)

    # -- system --------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        status, ready, cpu_available = manager.aggregate_status()
        qwen = monitor.snapshot()
        return {
            "status": status,
            "ready": ready,
            "started_at": meta["started_at"],
            "uptime_s": round(time.monotonic() - meta["monotonic"], 1),
            "catalogue_size": manager.catalogue_size(),
            "servers": manager.health_servers(),
            "modes": modes_public(cpu_live_available=cpu_available, qwen_health=qwen),
            "qwen_live": qwen,
            "memory_writes_enabled": bool(settings.allow_memory_writes),
        }

    @app.get("/api/demo/config")
    async def demo_config() -> dict[str, Any]:
        _status, _ready, cpu_available = manager.aggregate_status()
        qwen = monitor.snapshot()
        return {
            "profiles": [settings.profile_public(profile_id) for profile_id in PROFILE_BY_ID],
            "default_profile_id": DEFAULT_PROFILE_ID,
            "limits": limits_public(settings),
            "verified_comparison_ids": list(VERIFIED_COMPARISON_IDS),
            "modes": modes_public(cpu_live_available=cpu_available, qwen_health=qwen),
            "qwen_live": qwen,
            "notices": list(NOTICES),
        }

    @app.post("/api/demo/restart")
    async def demo_restart(body: RestartRequest | None = None) -> dict[str, Any]:
        profile_id = body.profile_id if body is not None else None
        servers = await manager.restart(profile_id)
        return {
            "restarting": True,
            "servers": servers,
            "message": "MCP server restart scheduled; poll /api/health for real readiness.",
        }

    @app.get("/api/replay")
    async def replay() -> dict[str, Any]:
        try:
            data = json.loads(settings.replay_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise MCPManagerError(
                f"Qwen replay artifact not found: {settings.replay_path}",
                code="replay_unavailable",
                http_status=503,
            ) from exc
        except json.JSONDecodeError as exc:
            raise MCPManagerError(
                f"Qwen replay artifact is not valid JSON: {exc}",
                code="replay_invalid",
                http_status=503,
            ) from exc
        return {
            "mode": "Qwen_Agent_Replay",
            "replay": True,
            "live_call": False,
            "badge": "Historical Replay — Not a Live Model Call",
            "examples": data.get("examples") or [],
        }

    # -- sessions ------------------------------------------------------------
    @app.post("/api/sessions")
    async def create_session(body: SessionCreateRequest) -> dict[str, Any]:
        if body.profile_id not in PROFILE_BY_ID:
            raise MCPManagerError(
                f"unknown profile_id {body.profile_id!r}; use /api/demo/config",
                code="unknown_profile_id",
                http_status=400,
            )
        session = store.create(body.profile_id)
        LOGGER.info("session created id=%s profile=%s", session.session_id, session.profile_id)
        return session.view()

    @app.get("/api/sessions")
    async def list_sessions(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        return {"sessions": [session.summary() for session in store.list(limit)]}

    @app.get("/api/sessions/{session_id}")
    async def get_session_view(session_id: str) -> dict[str, Any]:
        return get_session(session_id).view()

    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session(session_id: str) -> dict[str, Any]:
        session = get_session(session_id)
        session.reset_turns()
        session.title = DEFAULT_TITLE
        LOGGER.info("session reset id=%s", session.session_id)
        return session.view()

    # -- preferences ---------------------------------------------------------
    @app.get("/api/preferences")
    async def preferences(session_id: str = Query(min_length=1)) -> dict[str, Any]:
        session = get_session(session_id)
        require_ready(session)
        payload = await manager.call_tool(session.profile_id, "get_active_preferences", {})
        return vm.preference_state(payload, profile_id=session.profile_id)

    # -- chat ----------------------------------------------------------------
    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest) -> StreamingResponse:
        session = get_session(body.session_id)
        require_ready(session)
        if body.mode == "qwen_live" and not monitor.snapshot()["available"]:
            raise MCPManagerError(
                "Qwen Base Live is unavailable: " + str(monitor.snapshot().get("detail") or "health check failed"),
                code="qwen_unavailable",
                http_status=409,
                detail={"mode": "qwen_live", "qwen_status": monitor.snapshot().get("status")},
            )
        events = demo_flow.chat_events(
            manager=manager,
            session=session,
            settings=settings,
            message=body.message,
            selected_product_ids=list(body.selected_product_ids or []),
            mode=body.mode,
            qwen_health=monitor,
        )
        return sse_response(events)

    # -- products ------------------------------------------------------------
    @app.post("/api/products/details")
    async def product_details(body: ProductDetailsRequest) -> dict[str, Any]:
        session = get_session(body.session_id)
        require_ready(session)
        return await demo_flow.product_details(
            manager=manager, session=session, product_ref=body.product_ref
        )

    @app.post("/api/products/compare")
    async def product_compare(body: ProductCompareRequest) -> dict[str, Any]:
        session = get_session(body.session_id)
        require_ready(session)
        return await demo_flow.compare_products(
            manager=manager, session=session, product_refs=list(body.product_refs)
        )

    # -- clarification -------------------------------------------------------
    @app.post("/api/clarification/reply")
    async def clarification_reply(body: ClarificationReplyRequest) -> dict[str, Any]:
        session = get_session(body.session_id)
        require_ready(session)
        return await demo_flow.clarification_reply(
            manager=manager,
            session=session,
            clarification_id=body.clarification_id,
            message=body.message,
            qwen_health=monitor,
        )

    return app
