"""HTTP client implementing the benchmark PreferenceProposer protocol."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Sequence


class HttpOperationProposer:
    def __init__(self, url: str = "http://127.0.0.1:18008"):
        self.url = url
        self.last_payload: dict[str, Any] | None = None
        self.last_diagnostics: dict[str, Any] = {}

    def propose(self, *, messages: Sequence[str], current: str) -> tuple[str, list[dict[str, Any]], int, int]:
        body = json.dumps({"messages": list(messages), "current": current}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/propose",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.last_payload = payload.get("payload")
        self.last_diagnostics = {
            "text": payload.get("text", ""),
            "errors": payload.get("errors", []),
            "latency_ms": payload.get("latency_ms"),
        }
        if self.last_payload is None:
            # Keep an explicit invalid marker: it must reach the service as a rejected
            # proposal instead of looking like "no proposal this turn".
            self.last_payload = {
                "schema_version": "invalid_model_output",
                "operations": [],
                "clarification": None,
                "_model_error": list(payload.get("errors") or ()),
            }
        operations = (
            list((self.last_payload or {}).get("operations", []))
            if isinstance(self.last_payload, dict)
            else []
        )
        return current, operations, int(payload.get("input_tokens") or 0), int(payload.get("output_tokens") or 0)
