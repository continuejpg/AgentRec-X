"""Real Agent Evaluation Runner for the Web Golden Dataset.

The runner starts the existing FastAPI BFF and real BM25 MCP stdio child with isolated
temporary SQLite databases.  It executes each Golden case through the same HTTP entry points a
browser uses, records compact tool traces and business results, and never imports a second
Agent implementation.  Phoenix is not required; tracing is disabled for the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

from agent_eval.dataset import GoldenCase, GoldenDataset, dataset_sha256, load_dataset
from agent_eval.metrics import compute_metrics, evaluate_case
from web_demo.app import create_app
from web_demo.config import SEARCH_TOP_K, Settings
from recommendation.control.catalog_search import BM25_B, BM25_K1
from web_demo.handle_lifecycle import HandleRef
from web_demo.mcp_manager import MCPManagerError

__all__ = ["AgentEvaluationRunner", "EvaluationRunError", "build_settings", "main"]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "profile_0057"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "evaluations"
CATALOG_PATH = REPO_ROOT / "data/p13r/electronics/catalog_meta.jsonl"
SIDECAR_PATH = REPO_ROOT / "data/enrichment/electronics_metadata_sidecar.jsonl"
RUN_SCHEMA = "agentrecx.eval.agent_tasks.run.v1"
REPORT_SCHEMA = "agentrecx.eval.agent_tasks.report.v1"


class EvaluationRunError(RuntimeError):
    """The real evaluation environment could not be started or driven."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=False,
            capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def build_settings(workdir: Path, *, price_mode: str = "demo_historical_fx") -> Settings:
    """Isolated settings for one price-mode pass; never touch developer DBs.

    The no-price Golden case runs in strict mode.  Mixing modes in one BFF process would
    silently change the catalogue's meaning, so the runner starts one app per mode.
    """
    if price_mode not in {"demo_historical_fx", "strict"}:
        raise EvaluationRunError(f"unsupported price mode: {price_mode!r}")
    return Settings(
        search_mode="bm25",
        allow_memory_writes=True,
        log_dir=workdir / "logs",
        repo=REPO_ROOT,
        python_executable=sys.executable,
        mcp_timeout_s=180.0,
        clarification_ttl_s=1800.0,
        web_state_db_path=workdir / "web-state.sqlite3",
        memory_db_path=workdir / "m9.sqlite3",
        price_mode=price_mode,
        demo_price_sidecar=SIDECAR_PATH if price_mode == "demo_historical_fx" else None,
        otel_enabled=False,
    )


def _wait_ready(client: TestClient, timeout_s: float = 300.0) -> float:
    started = time.perf_counter()
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        last = client.get("/api/health")
        if last.status_code == 200 and last.json().get("ready"):
            return round((time.perf_counter() - started) * 1000.0, 2)
        time.sleep(1.0)
    detail = last.text if last is not None else "no response"
    raise EvaluationRunError(f"real MCP manager did not become ready: {detail}")


def parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        events.append({"event": name or "message", "data": payload})
    return events


def _event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for item in reversed(events):
        if item.get("event") == name:
            data = item.get("data")
            return dict(data) if isinstance(data, Mapping) else None
    return None


def _selected_card(
    recommendation: Mapping[str, Any] | None, products: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    for container in (recommendation, products):
        if not isinstance(container, Mapping):
            continue
        selected = container.get("recommended_product_id")
        items = container.get("items")
        if selected and isinstance(items, list):
            for item in items:
                if isinstance(item, Mapping) and str(item.get("product_id")) == str(selected):
                    return dict(item)
    items = (recommendation or {}).get("items") if isinstance(recommendation, Mapping) else None
    if isinstance(items, list) and len(items) == 1 and isinstance(items[0], Mapping):
        return dict(items[0])
    return None


class ToolRecorder:
    """Wrap the real MCP manager and retain only compact, reportable summaries."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self._real = manager.call_tool
        self.calls: list[dict[str, Any]] = []
        self.active = True

    def start_case(self) -> None:
        self.calls = []

    async def __call__(self, profile_id: str, tool: str, arguments: Any = None, **kwargs: Any) -> Any:
        if not self.active:
            return await self._real(profile_id, tool, arguments, **kwargs)
        started = time.perf_counter()
        record: dict[str, Any] = {
            "tool": tool,
            "argument_keys": sorted(str(key) for key in dict(arguments or {})),
            "status": "running",
            "latency_ms": None,
        }
        self.calls.append(record)
        try:
            result = await self._real(profile_id, tool, arguments, **kwargs)
        except MCPManagerError as exc:
            record["status"] = "error"
            record["error_code"] = exc.code
            record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            raise
        except Exception as exc:  # noqa: BLE001 - real transport failure is recorded
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            raise
        record["status"] = "ok"
        record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        self._summarize(record, result)
        return result

    @staticmethod
    def _summarize(record: dict[str, Any], result: Any) -> None:
        if not isinstance(result, Mapping):
            return
        if record["tool"] == "search_products":
            hits = [hit for hit in (result.get("hits") or []) if isinstance(hit, Mapping)]
            record["top5"] = [str(hit.get("item_id")) for hit in hits[:5]]
            record["hit_count"] = len(hits)
            record["search_mode"] = result.get("search_mode")
        elif record["tool"] == "recommend_products":
            budget = result.get("budget_verification")
            gate = result.get("gate")
            record["recommend_status"] = result.get("status")
            record["selected_product_id"] = result.get("selected_product_id")
            record["gate_action"] = (gate or {}).get("action") if isinstance(gate, Mapping) else None
            record["gate_reason_code"] = (
                (gate or {}).get("reason_code") if isinstance(gate, Mapping) else None
            )
            record["budget"] = (
                {
                    "within": budget.get("within"),
                    "over": budget.get("over"),
                    "unknown": budget.get("unknown"),
                    "currency_mismatch": budget.get("currency_mismatch"),
                    "selected_status": budget.get("selected_status"),
                }
                if isinstance(budget, Mapping) else None
            )
            requirements = result.get("task_requirements")
            record["task_requirements"] = (
                dict(requirements) if isinstance(requirements, Mapping) else None
            )
        elif record["tool"] == "get_product_info":
            record["product_id"] = result.get("product_id")
        elif record["tool"] == "request_preference_clarification":
            record["clarification_id"] = result.get("clarification_id")
            record["has_continuation"] = isinstance(result.get("continuation"), Mapping)
        elif record["tool"] == "restore_clarification_handle":
            record["restored_clarification_id"] = result.get("clarification_id")
            record["restore_status"] = result.get("status")
        elif record["tool"] == "submit_clarification_reply":
            update = result.get("update")
            if isinstance(update, Mapping):
                record["memory_update"] = {
                    key: update.get(key) for key in ("added", "superseded", "removed")
                }


@dataclasses.dataclass
class EvaluationRunnerConfig:
    dataset_path: Path = REPO_ROOT / "agent_eval/golden/golden_agent_tasks_v1.json"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_id: str | None = None
    case_ids: list[str] | None = None
    limit: int | None = None
    keep_workdir: bool = False


class AgentEvaluationRunner:
    """Execute the Golden Dataset against the real BFF/MCP entry points."""

    def __init__(self, dataset: GoldenDataset, config: EvaluationRunnerConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._selected_cases: list[GoldenCase] = []
        self._tool_recorder: ToolRecorder | None = None

    def selected_cases(self) -> list[GoldenCase]:
        cases = self.dataset.cases
        if self.config.case_ids:
            known = {case.case_id for case in cases}
            missing = sorted(set(self.config.case_ids) - known)
            if missing:
                raise EvaluationRunError(f"unknown case ids: {missing}")
            cases = [case for case in cases if case.case_id in set(self.config.case_ids)]
        if self.config.limit is not None:
            if self.config.limit < 1:
                raise EvaluationRunError("--limit must be >= 1")
            cases = cases[: self.config.limit]
        if not cases:
            raise EvaluationRunError("no Golden cases selected")
        return cases

    def run(self) -> dict[str, Any]:
        self._selected_cases = self.selected_cases()
        run_id = self.config.run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + (_git("rev-parse", "--short", "HEAD") or "nogit")
        )
        output_dir = self.config.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="agentrecx-agent-eval-"))
        started_wall = time.time()
        setup_ms: float | None = None
        setup_by_mode: dict[str, float] = {}
        results_by_id: dict[str, dict[str, Any]] = {}
        groups: dict[str, list[GoldenCase]] = {}
        for case in self._selected_cases:
            groups.setdefault(case.price_mode, []).append(case)
        try:
            for price_mode, group in groups.items():
                app = create_app(settings=build_settings(workdir, price_mode=price_mode))
                with TestClient(app) as client:
                    group_setup = _wait_ready(client)
                    setup_by_mode[price_mode] = group_setup
                    setup_ms = round((setup_ms or 0.0) + group_setup, 2)
                    manager = app.state.manager
                    recorder = ToolRecorder(manager)
                    manager.call_tool = recorder
                    self._tool_recorder = recorder
                    try:
                        for case in group:
                            number = len(results_by_id) + 1
                            print(
                                f"[{number}/{len(self._selected_cases)}] {case.case_id} "
                                f"({case.execution_mode}, {case.price_mode})",
                                flush=True,
                            )
                            result = self._run_case(client, case)
                            results_by_id[case.case_id] = result
                            print(
                                f"  done http={result.get('http_status')} "
                                f"outcome={result.get('actual_outcome_kind')} "
                                f"end_to_end_ms={result.get('latency_ms', {}).get('end_to_end_ms')}",
                                flush=True,
                            )
                    finally:
                        manager.call_tool = recorder._real
                        self._tool_recorder = None
        finally:
            self._cleanup_workdir(workdir)

        results = [results_by_id[case.case_id] for case in self._selected_cases]
        evaluations = [evaluate_case(case, result) for case, result in zip(
            self._selected_cases, results, strict=True
        )]
        metrics = compute_metrics(
            self.dataset, results, evaluations, total_cases=len(self._selected_cases)
        )
        metadata = self._metadata(
            run_id=run_id,
            started_wall=started_wall,
            setup_ms=setup_ms,
            setup_by_mode=setup_by_mode,
            selected_count=len(self._selected_cases),
        )
        report = {
            "report_schema": REPORT_SCHEMA,
            "metadata": metadata,
            "dataset": self.dataset.as_dict(),
            "metrics": metrics,
            "results": results,
            "case_evaluations": evaluations,
        }
        _write_reports(output_dir, report)
        return {"report": report, "output_dir": output_dir}

    def _run_case(self, client: TestClient, case: GoldenCase) -> dict[str, Any]:
        assert self._tool_recorder is not None
        recorder = self._tool_recorder
        recorder.start_case()
        started = time.perf_counter()
        result: dict[str, Any] = {
            "case_id": case.case_id,
            "category": case.category,
            "execution_mode": case.execution_mode,
            "price_mode": case.price_mode,
            "http_status": None,
            "chat_http_status": None,
            "reply_http_status": None,
            "actual_outcome_kind": "empty",
            "actual_refusal_code": None,
            "selected_product_id": None,
            "recommendation_id": None,
            "selected_product": None,
            "requirements_actual": None,
            "budget_verification": None,
            "gate": None,
            "retrieval_top5": [],
            "latency_ms": {},
            "ids": {"trace_id": None},
            "http_request_ids": {},
            "tool_calls": [],
            "replay": {"performed": False, "completed": False, "same_response": False, "mcp_calls": 0},
            "cold_resume": {"performed": False, "process_changed": False, "new_handle_ok": False},
            "error": None,
            "mode_failure": None,
        }
        session_id: str | None = None
        clarification_id: str | None = None
        pending_id: str | None = None
        task_id: str | None = None
        reply_turn_id: str | None = None
        first_reply_body: Mapping[str, Any] | None = None
        last_recommend: Mapping[str, Any] | None = None

        try:
            create_started = time.perf_counter()
            session_response = client.post("/api/sessions", json={"profile_id": PROFILE})
            result["session_create_ms"] = round(
                (time.perf_counter() - create_started) * 1000.0, 2
            )
            if session_response.status_code != 200:
                result["error"] = f"session HTTP {session_response.status_code}"
                result["http_status"] = session_response.status_code
                return self._finish_case(result, started)
            session_id = str(session_response.json().get("session_id") or "")
            result["ids"]["session_id"] = session_id

            chat_started = time.perf_counter()
            chat_response = client.post(
                "/api/chat/stream",
                json={
                    "session_id": session_id,
                    "message": case.user_query,
                },
            )
            chat_ms = round((time.perf_counter() - chat_started) * 1000.0, 2)
            result["latency_ms"]["ask_ms"] = chat_ms
            result["chat_http_status"] = chat_response.status_code
            result["http_status"] = chat_response.status_code
            result["http_request_ids"]["chat"] = chat_response.headers.get(
                "X-AgentRecX-Request-Id"
            )
            events = parse_sse(chat_response.text)
            clarification = _event(events, "clarification")
            products = _event(events, "products")
            error_event = _event(events, "error")
            if error_event is not None:
                result["error"] = str(error_event.get("message") or error_event.get("code"))

            # Capture the structured requirements from the real recommend tool even when the
            # final business outcome is a refusal or a direct products event.
            for call in recorder.calls:
                if call.get("tool") == "recommend_products" and call.get("task_requirements"):
                    last_recommend = call

            if clarification is not None:
                clarification_id = str(clarification.get("clarification_id") or "")
                result["ids"]["clarification_id"] = clarification_id
                state_store = getattr(client.app.state.sessions, "state_store", None)
                if state_store is not None:
                    pending = state_store.get_pending_for_clarification(
                        session_id, clarification_id
                    )
                    if pending is not None:
                        pending_id = pending.pending_id
                        task_id = pending.task_id
                        reply_turn_id = pending.reply_turn_id
                        result["ids"].update({
                            "pending_id": pending_id,
                            "task_id": task_id,
                            "reply_turn_id": reply_turn_id,
                        })

                if case.execution_mode == "cold_resume":
                    self._cold_restart(client, result)
                reply_started = time.perf_counter()
                reply_response = client.post(
                    "/api/clarification/reply",
                    json={
                        "session_id": session_id,
                        "clarification_id": clarification_id,
                        "message": case.clarification_reply,
                    },
                )
                result["latency_ms"]["reply_ms"] = round(
                    (time.perf_counter() - reply_started) * 1000.0, 2
                )
                result["reply_http_status"] = reply_response.status_code
                result["http_status"] = reply_response.status_code
                result["http_request_ids"]["reply"] = reply_response.headers.get(
                    "X-AgentRecX-Request-Id"
                )
                try:
                    body = reply_response.json()
                except ValueError:
                    body = {}
                first_reply_body = body if isinstance(body, Mapping) else None
                result["_reply_body"] = first_reply_body
                if reply_response.status_code != 200:
                    code = (
                        first_reply_body.get("code")
                        if isinstance(first_reply_body, Mapping) else None
                    )
                    result["error"] = (
                        f"HTTP {reply_response.status_code}"
                        + (f": {code}" if code else "")
                    )

                if case.execution_mode == "replay":
                    before = len(recorder.calls)
                    replay_started = time.perf_counter()
                    replay_response = client.post(
                        "/api/clarification/reply",
                        json={
                            "session_id": session_id,
                            "clarification_id": clarification_id,
                            "message": case.clarification_reply,
                        },
                    )
                    replay_ms = round(
                        (time.perf_counter() - replay_started) * 1000.0, 2
                    )
                    result["latency_ms"]["replay_ms"] = replay_ms
                    try:
                        replay_body = replay_response.json()
                    except ValueError:
                        replay_body = {}
                    result["replay"] = {
                        "performed": True,
                        "completed": bool(
                            replay_response.status_code == 200
                            and isinstance(replay_body, Mapping)
                            and isinstance(replay_body.get("recommendation"), Mapping)
                        ),
                        "same_response": bool(
                            replay_response.status_code == 200
                            and isinstance(replay_body, Mapping)
                            and first_reply_body is not None
                            and replay_body == dict(first_reply_body)
                        ),
                        "mcp_calls": max(0, len(recorder.calls) - before),
                        "http_status": replay_response.status_code,
                    }
                    if not result["replay"]["completed"]:
                        result["error"] = (
                            result["error"]
                            or f"replay HTTP {replay_response.status_code}"
                        )
            elif case.execution_mode in {"cold_resume", "replay"}:
                result["mode_failure"] = f"mode {case.execution_mode} requires a clarification"
            elif case.execution_mode == "direct" and case.expected_outcome.kind == "recommendation":
                # A direct finalized response is a legal end state; products is the response.
                pass

            if products is not None:
                result["_direct_products"] = products
                result["ids"]["recommendation_id"] = products.get("recommendation_id")
                result["ids"]["selected_product_id"] = products.get("recommended_product_id")

            # The final recommend tool call (ASK or reply) carries the most complete
            # structured requirements, including the reply's merged use case.
            for call in reversed(recorder.calls):
                if call.get("tool") == "recommend_products" and call.get("task_requirements"):
                    last_recommend = call
                    break

            # Normalize the final business payload.
            final_body: Mapping[str, Any] = (
                first_reply_body or {}
            )
            final_products: Mapping[str, Any] | None = (
                products if isinstance(products, Mapping) else None
            )
            recommendation = final_body.get("recommendation")
            refusal = final_body.get("refusal")
            if result["reply_http_status"] is not None and result["reply_http_status"] != 200:
                result["actual_outcome_kind"] = "execution_error"
            elif isinstance(recommendation, Mapping) and recommendation.get("items"):
                result["actual_outcome_kind"] = "recommendation"
            elif isinstance(refusal, Mapping) and refusal.get("code"):
                result["actual_outcome_kind"] = "refusal"
            elif isinstance(final_products, Mapping) and final_products.get("recommended_product_id"):
                result["actual_outcome_kind"] = "recommendation"
            elif clarification is not None and case.execution_mode == "direct":
                result["actual_outcome_kind"] = "clarification"
            elif isinstance(result["_reply_body"], Mapping) and result["_reply_body"].get("recommendation") is None:
                result["actual_outcome_kind"] = "empty"
            elif isinstance(final_products, Mapping):
                result["actual_outcome_kind"] = "empty"
            else:
                result["actual_outcome_kind"] = "empty"

            if isinstance(refusal, Mapping):
                result["actual_refusal_code"] = refusal.get("code")
            selected = _selected_card(
                recommendation if isinstance(recommendation, Mapping) else None,
                final_products,
            )
            selected_id = None
            if isinstance(recommendation, Mapping):
                selected_id = recommendation.get("recommended_product_id")
            if selected_id is None and isinstance(final_products, Mapping):
                selected_id = final_products.get("recommended_product_id")
            if selected_id is None and selected is not None:
                selected_id = selected.get("product_id")
            result["selected_product_id"] = str(selected_id) if selected_id else None
            result["selected_product"] = selected
            if isinstance(recommendation, Mapping):
                result["recommendation_id"] = recommendation.get("recommendation_id")
            if result["recommendation_id"] is None and isinstance(final_products, Mapping):
                result["recommendation_id"] = final_products.get("recommendation_id")

            requirements = None
            if isinstance(final_body, Mapping):
                requirements = final_body.get("task_requirements")
            if not isinstance(requirements, Mapping) and last_recommend is not None:
                requirements = last_recommend.get("task_requirements")
            result["requirements_actual"] = (
                dict(requirements) if isinstance(requirements, Mapping) else None
            )
            budget = final_body.get("budget_verification") if isinstance(final_body, Mapping) else None
            if not isinstance(budget, Mapping) and last_recommend is not None:
                budget = last_recommend.get("budget")
            result["budget_verification"] = (
                dict(budget) if isinstance(budget, Mapping) else None
            )
            if isinstance(final_products, Mapping):
                result["gate"] = final_products.get("gate")
            elif isinstance(recommendation, Mapping):
                result["gate"] = recommendation.get("gate")
            elif last_recommend is not None:
                result["gate"] = {
                    "action": last_recommend.get("gate_action"),
                    "reason_code": last_recommend.get("gate_reason_code"),
                }

            for call in recorder.calls:
                if call.get("tool") == "search_products" and call.get("top5"):
                    result["retrieval_top5"] = list(call["top5"])
                    break

            result["tool_calls"] = [dict(call) for call in recorder.calls]
            return self._finish_case(result, started)
        except Exception as exc:  # noqa: BLE001 - one case must not abort the suite
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["actual_outcome_kind"] = "execution_error"
            return self._finish_case(result, started)

    def _cold_restart(self, client: TestClient, result: dict[str, Any]) -> None:
        manager = client.app.state.manager
        before = manager.process_identity(PROFILE)
        old_process = before.process_instance_id if before is not None else None
        old_handle = manager.check_handle(HandleRef(
            profile_id=PROFILE,
            session_id=result["ids"].get("session_id", ""),
            clarification_id=result["ids"].get("clarification_id", ""),
        )).code
        restart_started = time.perf_counter()
        response = client.post("/api/demo/restart", json={"profile_id": PROFILE})
        if response.status_code == 200:
            _wait_ready(client)
        result["latency_ms"]["restart_ms"] = round(
            (time.perf_counter() - restart_started) * 1000.0, 2
        )
        after = manager.process_identity(PROFILE)
        new_process = after.process_instance_id if after is not None else None
        old_after = manager.check_handle(HandleRef(
            profile_id=PROFILE,
            session_id=result["ids"].get("session_id", ""),
            clarification_id=result["ids"].get("clarification_id", ""),
        )).code
        result["cold_resume"] = {
            "performed": True,
            "process_changed": bool(old_process and new_process and old_process != new_process),
            "old_handle_before": old_handle,
            "old_handle_after": old_after,
            "new_handle_ok": False,
            "old_process_instance_id": old_process,
            "new_process_instance_id": new_process,
        }

    @staticmethod
    def _finish_case(result: dict[str, Any], started: float) -> dict[str, Any]:
        result["latency_ms"]["end_to_end_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 2
        )
        result.pop("_reply_body", None)
        result.pop("_direct_products", None)
        if result.get("reply_http_status") == 200 and result.get("cold_resume", {}).get("performed"):
            result["cold_resume"]["new_handle_ok"] = True
        return result

    def _metadata(
        self,
        *,
        run_id: str,
        started_wall: float,
        setup_ms: float | None,
        setup_by_mode: Mapping[str, float],
        selected_count: int,
    ) -> dict[str, Any]:
        catalog_hash = _sha256_file(CATALOG_PATH)
        sidecar_hash = _sha256_file(SIDECAR_PATH)
        return {
            "run_id": run_id,
            "generated_at": _utc_now(),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "dataset_path": str(self.config.dataset_path),
            "dataset_sha256": dataset_sha256(self.config.dataset_path),
            "dataset_version": self.dataset.dataset_version,
            "dataset_case_count": len(self.dataset.cases),
            "selected_case_count": selected_count,
            "catalog_path": str(CATALOG_PATH.relative_to(REPO_ROOT)),
            "catalog_sha256": catalog_hash,
            "catalog_bytes": CATALOG_PATH.stat().st_size,
            "catalog_size": 76263,
            "sidecar_path": str(SIDECAR_PATH.relative_to(REPO_ROOT)),
            "sidecar_sha256": sidecar_hash,
            "sidecar_bytes": SIDECAR_PATH.stat().st_size,
            "price_mode": (
                next(iter(setup_by_mode)) if len(setup_by_mode) == 1 else "mixed"
            ),
            "price_modes": sorted(setup_by_mode),
            "fx_rate": "7.00" if "demo_historical_fx" in setup_by_mode else None,
            "price_note": "演示参考价：历史美元价格按固定汇率换算，非当前市场售价。",
            "search": {
                "mode": "bm25",
                "top_k": int(SEARCH_TOP_K),
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "score_kind": "bm25",
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "packages": {
                name: _package_version(name)
                for name in ("fastapi", "pydantic", "mcp", "pytest")
            },
            "tracing": {"enabled": False, "phoenix_required": False},
            "setup_ms": setup_ms,
            "setup_by_price_mode": dict(setup_by_mode),
            "run_started_at": datetime.fromtimestamp(
                started_wall, tz=timezone.utc
            ).isoformat(),
            "run_wall_ms": round((time.time() - started_wall) * 1000.0, 2),
            "command": [sys.executable, *sys.argv],
            "environment": {
                "AGENTRECX_PRICE_MODE": os.environ.get("AGENTRECX_PRICE_MODE"),
                "AGENTRECX_WEB_DEMO_SEARCH_MODE": os.environ.get(
                    "AGENTRECX_WEB_DEMO_SEARCH_MODE"
                ),
                "AGENTRECX_OTEL_ENABLED": os.environ.get("AGENTRECX_OTEL_ENABLED"),
            },
            "notes": [
                "task success is not inferred from HTTP 200",
                "correct refusals are scored separately from execution failures",
                "demo historical FX is not a live market price",
                "sample size is small; no statistical-significance claim is made",
            ],
        }

    def _cleanup_workdir(self, workdir: Path) -> None:
        if self.config.keep_workdir:
            print(f"work directory kept: {workdir}", flush=True)
            return
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _write_reports(output_dir: Path, report: Mapping[str, Any]) -> None:
    from agent_eval.report import write_json_report, write_markdown_report
    write_json_report(output_dir / "report.json", report)
    write_markdown_report(output_dir / "report.md", report)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in report["results"]:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the AgentRec-X task-level Golden Dataset against the real CPU/BM25 Agent."
    )
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "agent_eval/golden/golden_agent_tasks_v1.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--case-id", action="append", default=None,
                        help="run only the named case(s); useful for debugging")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N selected cases (smoke runs)")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        dataset = load_dataset(args.dataset)
    except Exception as exc:  # noqa: BLE001 - invalid dataset is a command failure
        print(f"dataset validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    runner = AgentEvaluationRunner(
        dataset,
        EvaluationRunnerConfig(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            run_id=args.run_id,
            case_ids=args.case_id,
            limit=args.limit,
            keep_workdir=bool(args.keep_workdir),
        ),
    )
    started = time.perf_counter()
    try:
        outcome = runner.run()
    except Exception as exc:  # noqa: BLE001 - command must fail visibly
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    metrics = outcome["report"]["metrics"]
    print(f"report: {outcome['output_dir'] / 'report.md'}")
    print(json.dumps({
        "task_outcome_accuracy": metrics["task_outcome_accuracy"],
        "requirement_accuracy": metrics["requirement_accuracy"],
        "budget_violations": metrics["budget_violations"],
        "retrieval_precision_at_5": metrics["retrieval_precision_at_5"],
        "latency_ms": metrics["latency_ms"],
        "failure_types": metrics["failure_types"]["counts"],
    }, indent=2, ensure_ascii=False))
    print(f"wall_s={time.perf_counter() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
