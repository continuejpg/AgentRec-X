"""Explicit opt-in Qwen proposal client for the Shopping Agent API.

Disabled unless both AGENTRECX_QWEN_PROPOSAL and AGENTRECX_QWEN_PROPOSAL_URL are set.
A transport/model failure raises; it never falls back to rule proposals.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

ENV_ENABLED = "AGENTRECX_QWEN_PROPOSAL"
ENV_URL = "AGENTRECX_QWEN_PROPOSAL_URL"
ENV_MODEL = "AGENTRECX_QWEN_PROPOSAL_MODEL"


class QwenProposalError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def qwen_proposal_enabled() -> bool:
    return (os.environ.get(ENV_ENABLED, "").strip().casefold() in {"1", "true", "on", "yes", "enabled"})


class QwenProposalClient:
    def __init__(self, url: str, model: str | None = None, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "QwenProposalClient":
        enabled = qwen_proposal_enabled()
        url = (os.environ.get(ENV_URL) or "").strip()
        if not enabled:
            raise QwenProposalError("qwen_proposal_disabled", "AGENTRECX_QWEN_PROPOSAL is not enabled")
        if not url:
            raise QwenProposalError("qwen_proposal_misconfigured", "AGENTRECX_QWEN_PROPOSAL_URL is empty")
        return cls(url, (os.environ.get(ENV_MODEL) or "").strip() or None)

    def propose(self, *, messages: Sequence[str], current: str) -> dict[str, Any]:
        body = json.dumps({"messages": list(messages), "current": current}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/propose",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QwenProposalError("qwen_proposal_unavailable", "the Qwen model server is unavailable") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise QwenProposalError("qwen_proposal_invalid_response", "the Qwen model server returned invalid JSON") from exc
        proposal = payload.get("payload") if isinstance(payload, dict) else None
        if not isinstance(proposal, dict):
            raise QwenProposalError("qwen_proposal_invalid_payload", "the Qwen model server returned no proposal payload")
        return proposal
