"""Background health monitor for the optional real Qwen Base Live mode.

The monitor performs a real ``POST {endpoint}/plan`` with the existing planner contract and only
marks Qwen Live available when the response is a valid PlannerAction.  It never falls back to Rule,
never starts/stops AutoDL or a tunnel, and never exposes the endpoint to the browser or logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from web_demo.config import Settings

LOGGER = logging.getLogger("agentrecx.web_demo.qwen_health")
PROBE_QUERY = "wireless earbuds for working out"
# With no candidates yet, FINALIZE/RERANK are not contract-legal.  A health pass therefore proves
# the service returned a valid action for the current observation, not just any JSON.
PROBE_ACTIONS = ("ASK", "RETRIEVE")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class QwenHealthMonitor:
    """One shared health state for both fixed profiles (the model service is process-wide)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._status = "disabled" if not settings.qwen_live_configured else "checking"
        self._available = False
        self._planner_action: str | None = None
        self._latency_ms: float | None = None
        self._checked_at: str | None = None
        self._detail: str | None = None

    # -- public snapshot -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.settings.qwen_live_enabled),
            "configured": bool(self.settings.qwen_live_configured),
            "status": self._status,
            "available": bool(self._available),
            "model": self.settings.qwen_model_name,
            "planner_action": self._planner_action,
            "latency_ms": self._latency_ms,
            "checked_at": self._checked_at,
            "detail": self._detail,
        }

    def mark_unavailable(self, detail: str) -> None:
        """A real planner call failed after health passed: stop advertising availability."""
        self._status = "unavailable"
        self._available = False
        self._detail = str(detail)[:400]
        self._checked_at = _now_iso()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if not self.settings.qwen_live_configured:
            self._status = "disabled"
            self._detail = "QWEN_LIVE_ENABLED / QWEN_MODEL_ENDPOINT not configured"
            LOGGER.info("Qwen Live disabled (not configured)")
            return
        LOGGER.info("Qwen Live configured; starting real PlannerAction health checks")
        self._task = asyncio.create_task(self._loop(), name="qwen-live-health")

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self.check()
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.settings.qwen_health_interval_s
                )
            except asyncio.TimeoutError:
                continue

    async def check(self) -> dict[str, Any]:
        if not self.settings.qwen_live_configured:
            self._status = "disabled"
            self._available = False
            return self.snapshot()
        async with self._lock:
            self._status = "checking"
            self._detail = None
            try:
                result = await asyncio.to_thread(self._probe)
            except Exception as exc:  # noqa: BLE001 - explicit unavailable state
                self._status = "unavailable"
                self._available = False
                self._detail = f"{type(exc).__name__}: probe failed"
                self._checked_at = _now_iso()
                LOGGER.warning("Qwen Live health unavailable: %s", self._detail)
                return self.snapshot()
            self._status = "available"
            self._available = True
            self._planner_action = result["action"]
            self._latency_ms = result["latency_ms"]
            self._checked_at = _now_iso()
            self._detail = f"valid PlannerAction from {self.settings.qwen_model_name}"
            LOGGER.info(
                "Qwen Live health ok action=%s latency_ms=%s",
                self._planner_action, self._latency_ms,
            )
            return self.snapshot()

    # -- real HTTP probe -----------------------------------------------------
    def _probe(self) -> dict[str, Any]:
        """Blocking health probe; called in a worker thread.

        A valid PlannerAction is required. This is a real model call and is intentionally the only
        availability gate; no Rule output can satisfy it.
        """
        endpoint = self.settings.qwen_model_endpoint.rstrip("/")
        body = {
            "contract_version": "agentrecx.planner_action.v1",
            "step": 0,
            "query": PROBE_QUERY,
            "candidate_ids": [],
            "candidate_count": 0,
            "candidate_evidence": [],
            "allowed_actions": list(PROBE_ACTIONS),
            "remaining_steps": max(1, self.settings.qwen_max_steps),
            "needs_user_input": False,
            "reranked": False,
        }
        request = urllib.request.Request(
            endpoint + "/plan",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.qwen_health_timeout_s
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Never include the endpoint in the error detail.
            raise RuntimeError(type(exc).__name__) from exc
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if not isinstance(payload, dict):
            raise RuntimeError("health response is not a JSON object")
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            raise RuntimeError("health response reported parser errors")
        action_payload = payload.get("payload")
        if isinstance(action_payload, dict):
            payload = action_payload
        raw_action = str(payload.get("action") or "").strip().upper()
        if raw_action not in PROBE_ACTIONS:
            raise RuntimeError(f"health returned invalid PlannerAction {raw_action!r}")
        return {"action": raw_action, "latency_ms": latency_ms}
