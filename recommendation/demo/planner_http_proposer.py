"""Structured HTTP Planner Action proposer (explicit opt-in, no fallback).

This client asks an external HTTP model for one Planner action.  It never executes a
tool itself; ``run_planner`` passes the returned ActionProposal through the existing
contract validator and then the existing ServicePlannerExecutor.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from recommendation.demo.planner_contract import (
    ActionProposal,
    PlannerAction,
    PlannerContractError,
    PlannerObservation,
)

ENV_ENABLED = "AGENTRECX_QWEN_PLANNER"
ENV_URL = "AGENTRECX_QWEN_PLANNER_URL"
ENV_MODEL = "AGENTRECX_QWEN_PLANNER_MODEL"
ENV_TIMEOUT = "AGENTRECX_QWEN_PLANNER_TIMEOUT_S"
CONTRACT_VERSION = "agentrecx.planner_action.v1"


def planner_enabled() -> bool:
    return (os.environ.get(ENV_ENABLED, "").strip().casefold() in {"1", "true", "on", "yes", "enabled"})


class HttpPlannerError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def parse_action_payload(payload: Any) -> ActionProposal:
    if not isinstance(payload, dict):
        raise HttpPlannerError("planner_payload_not_object", "planner response is not a JSON object")
    if "operations" in payload:
        # The Preference Operation schema is not a Planner Action schema.
        raise HttpPlannerError("planner_used_preference_schema", "planner returned Preference Operation JSON")
    raw_action = str(payload.get("action") or "").strip().upper()
    try:
        action = PlannerAction(raw_action)
    except ValueError as exc:
        raise HttpPlannerError("planner_invalid_action", f"unknown planner action {raw_action!r}") from exc
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HttpPlannerError("planner_invalid_arguments", "planner arguments must be an object")
    rationale = str(payload.get("rationale") or "")
    return ActionProposal(action=action, arguments=dict(arguments), rationale=rationale)


class HttpPlannerProposer:
    def __init__(self, url: str, *, model: str | None = None, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.last_raw: str = ""
        self.last_response: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> "HttpPlannerProposer":
        if not planner_enabled():
            raise HttpPlannerError("planner_disabled", "AGENTRECX_QWEN_PLANNER is not enabled")
        url = (os.environ.get(ENV_URL) or "").strip()
        if not url:
            raise HttpPlannerError("planner_misconfigured", "AGENTRECX_QWEN_PLANNER_URL is empty")
        timeout_raw = (os.environ.get(ENV_TIMEOUT) or "").strip()
        timeout = 120.0
        if timeout_raw:
            try:
                timeout = float(timeout_raw)
            except ValueError as exc:
                raise HttpPlannerError(
                    "planner_misconfigured",
                    f"{ENV_TIMEOUT} must be a number, got {timeout_raw!r}",
                ) from exc
            if timeout <= 0:
                raise HttpPlannerError(
                    "planner_misconfigured", f"{ENV_TIMEOUT} must be positive"
                )
        return cls(url, model=(os.environ.get(ENV_MODEL) or "").strip() or None,
                   timeout=timeout)

    def propose(self, observation: PlannerObservation) -> ActionProposal:
        request_payload = {
            "contract_version": CONTRACT_VERSION,
            "step": observation.step,
            "query": observation.query,
            "candidate_ids": list(observation.candidate_ids),
            "candidate_count": len(observation.candidate_ids),
            "candidate_evidence": list(observation.candidate_evidence),
            "last_retrieve_query": observation.last_retrieve_query,
            "retrieval_feedback": observation.retrieval_feedback,
            "role_target": observation.role_target,
            "eligibility": observation.eligibility,
            "source_policy": observation.source_policy,
            "source_feedback": observation.source_feedback,
            "state_version": observation.state_version,
            "m5_state": observation.m5_state,
            "allowed_actions": list(observation.allowed_actions),
            "last_tool_observation": observation.last_tool_observation,
            "remaining_steps": observation.remaining_steps,
            "needs_user_input": observation.needs_user_input,
            "reranked": observation.reranked,
        }
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/plan",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise HttpPlannerError("planner_unavailable", "the planner model server is unavailable") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise HttpPlannerError("planner_invalid_json", "the planner model server returned invalid JSON") from exc
        if isinstance(payload, dict):
            self.last_response = dict(payload)
            self.last_raw = str(payload.get("text") or "")
            # Optional local diagnostic: record the exact planner request/response envelope.
            trace_path = (os.environ.get("AGENTRECX_QWEN_PLANNER_TRACE") or "").strip()
            if trace_path:
                try:
                    record = {
                        "request": request_payload,
                        "response": payload,
                        "raw_text": self.last_raw,
                    }
                    with Path(trace_path).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                except OSError:
                    pass
            # model_server returns an envelope {payload,text,errors,...}; a Stub server
            # may return the action object directly.  Unwrap only the envelope form.
            action_payload = payload.get("payload")
            if isinstance(action_payload, dict):
                payload = action_payload
        return parse_action_payload(payload)
