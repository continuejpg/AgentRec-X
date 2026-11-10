"""Long-lived MCP client management for the Web Demo BFF.

One official ``mcp.Client`` (stdio transport) is held per fixed demo profile for the whole
lifetime of the BFF process.  Both servers start in the background so the HTTP API can answer
``/api/health`` while they initialise (catalogue + BGE cold start is ~35-45 s).

Design notes:
* each server has its own status (starting/ready/error/stopped), startup time, tool list and
  asyncio call lock, so calls to one server never interleave with each other;
* the MCP server's real stderr is appended to ``runs/web_demo/logs/mcp-<profile>.log`` by a
  tiny ``/bin/bash -c 'exec ... 2>>log'`` wrapper (the SDK does not expose an errlog sink);
* tool errors from the server are converted to explicit ``MCPToolError`` instances, never to a
  fabricated success.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Mapping

from mcp import Client, StdioServerParameters

from web_demo.config import (
    MCP_SERVER_MODULE,
    PROFILE_BY_ID,
    Settings,
)

LOGGER = logging.getLogger("agentrecx.web_demo.mcp")

_HTTP_400 = 400
_HTTP_409 = 409
_HTTP_503 = 503
_HTTP_504 = 504


class MCPManagerError(Exception):
    """A real MCP failure, exposed to the HTTP/SSE layer without masking."""

    code = "mcp_error"
    http_status = _HTTP_400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        tool: str | None = None,
        detail: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        if code:
            self.code = code
        if http_status is not None:
            self.http_status = int(http_status)
        self.tool = tool
        self.detail = dict(detail) if detail else None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.tool:
            payload["tool"] = self.tool
        if self.detail:
            payload["detail"] = self.detail
        return payload


class MCPUnavailableError(MCPManagerError):
    code = "mcp_not_ready"
    http_status = _HTTP_503


class MCPToolTimeout(MCPManagerError):
    code = "tool_timeout"
    http_status = _HTTP_504


class MCPToolError(MCPManagerError):
    code = "tool_error"
    http_status = _HTTP_400


def _classify_tool_error(message: str) -> str:
    text = message.strip()
    for prefix, code in (
        ("memory_writes_disabled", "memory_writes_disabled"),
        ("product_not_found", "product_not_found"),
        ("namespace mismatch", "namespace_mismatch"),
        ("invalid_or_expired_clarification_id", "invalid_or_expired_clarification_id"),
        ("clarification_id_expired", "clarification_id_expired"),
        ("clarification_already_answered", "clarification_already_answered"),
        ("user_confirmation_required", "user_confirmation_required"),
        ("comparison_failed", "comparison_failed"),
        ("invalid_product_reference", "invalid_product_reference"),
        ("qwen_unavailable", "qwen_unavailable"),
        ("qwen_planner_timeout", "qwen_planner_timeout"),
        ("qwen_planner_invalid", "qwen_planner_invalid"),
        ("qwen_planner_error", "qwen_planner_error"),
        ("qwen_planner_contract_error", "qwen_planner_error"),
        ("agent_execution_error", "agent_execution_error"),
    ):
        if text.startswith(prefix):
            return code
    return "tool_error"


def _error_text(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    texts = [str(getattr(block, "text", "")) for block in blocks if getattr(block, "text", None)]
    message = "\n".join(text for text in texts if text).strip() or "MCP tool reported an error"
    marker = "Error executing tool "
    if message.startswith(marker):
        message = message[len(marker):]
        if ":" in message:
            message = message.split(":", 1)[1].strip()
    return message


@dataclass
class ServerRuntime:
    """Mutable runtime state for one MCP server."""

    profile_id: str
    user_key: str
    writes_enabled: bool
    status: str = "stopped"  # stopped | starting | ready | error
    error: str | None = None
    startup_ms: float | None = None
    tools: list[dict[str, Any]] | None = None
    task: asyncio.Task[None] | None = None
    watchdog: asyncio.Task[None] | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: Client | None = None
    last_call: dict[str, Any] | None = None

    def health(self, search_mode: str) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "user_key": self.user_key,
            "status": self.status,
            "search_mode": search_mode,
            "writes_enabled": self.writes_enabled,
            "startup_ms": self.startup_ms,
            "error": self.error,
            "tools": [tool["name"] for tool in self.tools] if self.tools else None,
        }


class MCPManager:
    """Owns both long-lived MCP clients: start, health, serialized calls, restart, shutdown."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runtimes: dict[str, ServerRuntime] = {}
        for profile_id, profile in PROFILE_BY_ID.items():
            self.runtimes[profile_id] = ServerRuntime(
                profile_id=profile_id,
                user_key=profile.user_key,
                writes_enabled=settings.writes_enabled_for(profile_id),
            )
        self.observed_catalogue_size: int | None = None
        self.started_at: float | None = None
        self._started = False
        self._stopping = False

    # -- lifecycle -----------------------------------------------------------
    def _server_parameters(self, runtime: ServerRuntime) -> StdioServerParameters:
        log_path = self.settings.log_dir / f"mcp-{runtime.profile_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            self.settings.python_executable,
            "-m",
            MCP_SERVER_MODULE,
            "--user-key",
            runtime.user_key,
            "--memory-fixture",
            str(self.settings.memory_fixture),
            "--search-mode",
            self.settings.search_mode,
            "--clarification-ttl-seconds",
            repr(float(self.settings.clarification_ttl_s)),
        ]
        if self.settings.search_mode == "bge":
            # The Web BFF advertises ready only after the real encoder warm-up and a real
            # retrieval check completed in the child process.  This removes the old first
            # user-query wait without claiming a fake readiness.  Set
            # AGENTRECX_WEB_DEMO_BGE_WARMUP=0 only for controlled before/after measurements.
            _warmup = str(os.environ.get("AGENTRECX_WEB_DEMO_BGE_WARMUP", "1")).strip().lower()
            if _warmup not in {"0", "false", "no", "off"}:
                parts.append("--warmup")
        if runtime.writes_enabled:
            parts.append("--allow-memory-writes")
        inner = " ".join(shlex.quote(part) for part in parts)
        child_env: dict[str, str] = {
            "PYTHONPATH": str(self.settings.repo),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "AGENTRECX_WEB_DEMO_PROFILE": runtime.profile_id,
        }
        if self.settings.qwen_live_configured:
            # The Qwen endpoint is passed only to the trusted stdio child process. It is never
            # included in health/config/SSE payloads or logs.
            child_env.update({
                "AGENTRECX_QWEN_PLANNER": "1",
                "AGENTRECX_QWEN_PLANNER_URL": self.settings.qwen_model_endpoint,
                "AGENTRECX_QWEN_PLANNER_MODEL": self.settings.qwen_model_name,
                "AGENTRECX_QWEN_PLANNER_TIMEOUT_S": repr(float(self.settings.qwen_live_timeout_s)),
                "AGENTRECX_QWEN_MAX_STEPS": str(int(self.settings.qwen_max_steps)),
                "AGENTRECX_QWEN_EXPLAIN": "1" if self.settings.qwen_explain else "0",
            })
        return StdioServerParameters(
            # A shell wrapper is the only supported way to capture the server's real stderr;
            # stdout stays untouched (it carries the MCP stdio protocol).
            command="/bin/bash",
            args=["-c", f"exec {inner} 2>>{shlex.quote(str(log_path))}"],
            cwd=str(self.settings.repo),
            env=child_env,
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.started_at = time.time()
        for runtime in self.runtimes.values():
            self._start_runtime(runtime)
        LOGGER.info(
            "MCP manager starting: profiles=%s search_mode=%s writes_global=%s",
            list(self.runtimes), self.settings.search_mode, self.settings.allow_memory_writes,
        )

    def _start_runtime(self, runtime: ServerRuntime) -> None:
        runtime.stop_event = asyncio.Event()
        runtime.lock = asyncio.Lock()
        runtime.status = "starting"
        runtime.error = None
        runtime.startup_ms = None
        runtime.tools = None
        runtime.client = None
        runtime.task = asyncio.create_task(
            self._serve(runtime), name=f"mcp-server-{runtime.profile_id}"
        )
        runtime.watchdog = asyncio.create_task(
            self._startup_watchdog(runtime), name=f"mcp-startup-watchdog-{runtime.profile_id}"
        )

    async def _startup_watchdog(self, runtime: ServerRuntime) -> None:
        """Enforce the configured startup ceiling without wrapping the client's enter/exit."""
        try:
            await asyncio.sleep(self.settings.mcp_timeout_s)
        except asyncio.CancelledError:
            raise
        if runtime.status != "starting":
            return
        runtime.status = "error"
        runtime.error = (
            f"startup timed out after {self.settings.mcp_timeout_s:.0f}s "
            "(catalogue/BGE cold start)"
        )
        LOGGER.error("MCP server startup timeout profile=%s", runtime.profile_id)
        task = runtime.task
        if task is not None and not task.done():
            task.cancel()

    async def _serve(self, runtime: ServerRuntime) -> None:
        started = time.perf_counter()
        client = Client(self._server_parameters(runtime))
        runtime.client = client
        try:
            # NOTE: the client's enter/exit and the whole server lifetime stay in this same task.
            # Wrapping __aenter__ in a timeout helper (asyncio.wait_for / anyio.fail_after) breaks
            # the MCP transport's cancel-scope ownership; startup timeout is enforced by a
            # separate watchdog task that cancels THIS task instead.
            await client.__aenter__()
            listing = await client.list_tools()
            runtime.tools = [
                {"name": tool.name, "description": tool.description} for tool in listing.tools
            ]
            runtime.status = "ready"
            runtime.startup_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if runtime.watchdog is not None:
                runtime.watchdog.cancel()
                runtime.watchdog = None
            LOGGER.info(
                "MCP server ready profile=%s startup_ms=%s tools=%s",
                runtime.profile_id, runtime.startup_ms, [tool["name"] for tool in runtime.tools],
            )
            await runtime.stop_event.wait()
        except asyncio.CancelledError:
            LOGGER.info("MCP server task cancelled profile=%s", runtime.profile_id)
            raise
        except Exception as exc:  # noqa: BLE001 - the real failure is surfaced in health
            runtime.status = "error"
            runtime.error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("MCP server failed profile=%s: %s", runtime.profile_id, runtime.error)
        finally:
            watchdog = runtime.watchdog
            runtime.watchdog = None
            if watchdog is not None and not watchdog.done():
                watchdog.cancel()
                with suppress(asyncio.CancelledError):
                    await watchdog
            try:
                await client.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - shutdown best effort, logged
                LOGGER.warning(
                    "MCP server exit error profile=%s: %s: %s",
                    runtime.profile_id, type(exc).__name__, exc,
                )
            runtime.client = None
            if runtime.status != "error":
                runtime.status = "stopped"
            LOGGER.info("MCP server stopped profile=%s", runtime.profile_id)

    async def _stop_runtime(self, runtime: ServerRuntime) -> None:
        runtime.stop_event.set()
        watchdog = runtime.watchdog
        runtime.watchdog = None
        if watchdog is not None:
            watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog
        task = runtime.task
        runtime.task = None
        if task is None:
            runtime.status = "stopped"
            return
        if runtime.status == "starting" and not task.done():
            # A server still cold-starting will never observe stop_event; cancel its enter loop.
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=30.0)
        except asyncio.TimeoutError:
            LOGGER.error("MCP server did not stop in time profile=%s; cancelling", runtime.profile_id)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
            LOGGER.warning("MCP stop task error profile=%s: %s", runtime.profile_id, exc)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        LOGGER.info("MCP manager stopping")
        await asyncio.gather(
            *(self._stop_runtime(runtime) for runtime in self.runtimes.values()),
            return_exceptions=True,
        )
        self._started = False

    async def restart(self, profile_id: str | None = None) -> list[dict[str, Any]]:
        if profile_id is not None and profile_id not in self.runtimes:
            raise MCPManagerError(
                f"unknown profile_id {profile_id!r}", code="unknown_profile_id", http_status=400
            )
        targets = (
            [self.runtimes[profile_id]] if profile_id is not None else list(self.runtimes.values())
        )
        for runtime in targets:
            await self._stop_runtime(runtime)
        for runtime in targets:
            self._start_runtime(runtime)
        self._stopping = False
        return [runtime.health(self.settings.search_mode) for runtime in targets]

    # -- readiness / health --------------------------------------------------
    def ensure_ready(self, profile_id: str) -> ServerRuntime:
        runtime = self.runtimes.get(profile_id)
        if runtime is None:
            raise MCPManagerError(
                f"unknown profile_id {profile_id!r}", code="unknown_profile_id", http_status=400
            )
        if runtime.status != "ready":
            raise MCPUnavailableError(
                f"MCP server for {profile_id} is not ready (status={runtime.status})",
                detail={
                    "profile_id": profile_id,
                    "status": runtime.status,
                    "error": runtime.error,
                    "startup_ms": runtime.startup_ms,
                },
            )
        return runtime

    def health_servers(self) -> list[dict[str, Any]]:
        return [
            runtime.health(self.settings.search_mode) for runtime in self.runtimes.values()
        ]

    def aggregate_status(self) -> tuple[str, bool, bool]:
        statuses = [runtime.status for runtime in self.runtimes.values()]
        ready = all(status == "ready" for status in statuses)
        if ready:
            return "ready", True, True
        if any(status == "error" for status in statuses) and not any(
            status == "starting" for status in statuses
        ):
            if any(status == "ready" for status in statuses):
                return "degraded", False, True
            return "error", False, False
        return "starting", False, any(status == "ready" for status in statuses)

    def status(self) -> str:
        return self.aggregate_status()[0]

    def ready(self) -> bool:
        return self.aggregate_status()[1]

    def cpu_live_available(self) -> bool:
        return self.aggregate_status()[2]

    def catalogue_size(self) -> int:
        from web_demo.config import CATALOGUE_SIZE

        return int(self.observed_catalogue_size or CATALOGUE_SIZE)

    # -- tool calls ----------------------------------------------------------
    async def call_tool(
        self,
        profile_id: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        runtime = self.ensure_ready(profile_id)
        call_timeout = float(timeout_s) if timeout_s is not None else float(self.settings.mcp_timeout_s)
        async with runtime.lock:
            # Re-check after waiting for the lock: the server may have failed or been restarted
            # while this request was queued.
            runtime = self.ensure_ready(profile_id)
            client = runtime.client
            if client is None:  # pragma: no cover - guarded by status == ready
                raise MCPUnavailableError(f"MCP client for {profile_id} is not connected")
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    client.call_tool(
                        tool, dict(arguments or {}),
                        progress_callback=progress_callback,
                    ),
                    timeout=call_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise MCPToolTimeout(
                    f"tool {tool!r} on {profile_id} exceeded "
                    f"{call_timeout:.0f}s timeout",
                    tool=tool,
                    detail={"profile_id": profile_id},
                ) from exc
            except MCPManagerError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - real transport/session failure
                raise MCPToolError(
                    f"tool {tool!r} on {profile_id} failed: {type(exc).__name__}: {exc}",
                    tool=tool,
                    detail={"profile_id": profile_id},
                ) from exc
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if getattr(result, "is_error", False):
                message = _error_text(result)
                error = MCPToolError(
                    message,
                    code=_classify_tool_error(message),
                    tool=tool,
                    detail={
                        "profile_id": profile_id,
                        "latency_ms": latency_ms,
                        "content": [
                            getattr(block, "text", None) for block in result.content or []
                        ],
                    },
                )
                LOGGER.warning(
                    "tool error profile=%s tool=%s code=%s message=%s",
                    profile_id, tool, error.code, message,
                )
                raise error
            structured = getattr(result, "structured_content", None)
            if not isinstance(structured, dict):
                raise MCPToolError(
                    f"tool {tool!r} on {profile_id} returned no structured content",
                    tool=tool,
                    detail={"profile_id": profile_id, "latency_ms": latency_ms},
                )
            catalogue_size = structured.get("catalogue_size")
            if isinstance(catalogue_size, int) and catalogue_size > 0:
                self.observed_catalogue_size = catalogue_size
            runtime.last_call = {
                "tool": tool,
                "latency_ms": latency_ms,
                "at": time.time(),
                "profile_id": profile_id,
            }
            LOGGER.info(
                "tool ok profile=%s tool=%s latency_ms=%s", profile_id, tool, latency_ms
            )
            return structured
