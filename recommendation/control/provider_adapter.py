"""Optional provider adapter for the structured-model seam (Phase 1).

This module is the **only** place a provider protocol is described, and it contains no SDK
import, no HTTP client, no API key and no network call.  It documents the contract a real
adapter must satisfy and fails clearly when the deployment has not configured one.

Why it ships without a live call
--------------------------------
The repository's dependency policy keeps provider SDKs out of the core, and this phase's
requirement is that the test suite needs no network or key.  A protocol adapter is therefore
split in two:

* :class:`OpenAICompatibleChatAdapter` - the **shape** of an adapter for any provider that
  speaks the OpenAI-compatible ``POST {base_url}/chat/completions`` contract (which covers
  several hosted providers and local servers such as vLLM or Ollama).  It performs the request
  through an **injected** transport, so no HTTP library is imported here and a deployment
  supplies one.
* :func:`build_provider_client` - constructs the adapter only when the environment is
  explicitly configured, and raises :class:`ModelCallError` with an actionable message
  otherwise.

That keeps the seam real and reviewable while leaving the dependency choice to the deployment.

Configuration
-------------
``AGENTRECX_LLM_BASE_URL``
    Base URL of the provider, e.g. ``https://api.openai.com/v1``.  Required.
``AGENTRECX_LLM_MODEL``
    Model identifier, e.g. ``gpt-4o-mini``.  Required.
``AGENTRECX_LLM_API_KEY``
    Credential.  Read from the environment at call time and **never** logged, stored or
    serialised.  A local server that needs no key may use any placeholder.

Example (not executed by any test, and not by the smoke unless explicitly enabled)::

    AGENTRECX_LLM_BASE_URL=http://localhost:8000/v1 \\
    AGENTRECX_LLM_MODEL=Qwen2.5-7B-Instruct \\
    AGENTRECX_LLM_API_KEY=EMPTY \\
    .venv/bin/python -m experiments.llm_policy_smoke --live
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Protocol, runtime_checkable

from .model_client import ModelRequest, ModelResponse, ModelCallError

__all__ = [
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "HttpTransport",
    "OpenAICompatibleChatAdapter",
    "build_provider_client",
    "provider_configured",
]

ENV_BASE_URL = "AGENTRECX_LLM_BASE_URL"
ENV_MODEL = "AGENTRECX_LLM_MODEL"
ENV_API_KEY = "AGENTRECX_LLM_API_KEY"


@runtime_checkable
class HttpTransport(Protocol):
    """The one thing an adapter needs from an HTTP client.

    Injected so this module imports no HTTP library: a deployment passes ``requests.post``,
    ``httpx.post``, or its own client, and the tests pass a stub.
    """

    def __call__(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> tuple[int, str]:
        """Perform the request and return ``(status_code, body_text)``."""
        ...


class OpenAICompatibleChatAdapter:
    """Adapt an OpenAI-compatible chat endpoint to :class:`StructuredModelClient`.

    The response body's ``choices[0].message.content`` becomes the
    :class:`~recommendation.control.model_client.ModelResponse` text.  Everything the model
    returns stays untrusted: the policy parses it, and the validator authorises the result.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        transport: HttpTransport,
        api_key: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        if not base_url.strip():
            raise ModelCallError("base_url is required", code="not_configured")
        if not model.strip():
            raise ModelCallError("model is required", code="not_configured")
        if not callable(transport):
            raise ModelCallError("transport must be callable", code="not_configured")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport
        self._api_key = api_key
        self._timeout = float(timeout)
        #: Deterministic by default: a control decision should not vary run to run merely
        #: because a sampler was warm.
        self._temperature = float(temperature)

    @property
    def endpoint(self) -> str:
        """The chat-completions URL this adapter calls."""
        return f"{self._base_url}/chat/completions"

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send the request and return the model's raw answer."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": request.as_messages(),
            "temperature": self._temperature,
        }
        try:
            status, body = self._transport(
                self.endpoint, headers=headers, payload=payload, timeout=self._timeout
            )
        except TimeoutError as exc:
            raise ModelCallError("the model request timed out", code="timeout") from exc
        except Exception as exc:  # noqa: BLE001 - transport failures are normalised
            raise ModelCallError(
                f"the model request failed: {type(exc).__name__}", code="provider_error"
            ) from exc

        if status >= 400:
            # The provider's own text is not echoed: it can contain request details.
            raise ModelCallError(
                f"the provider returned HTTP {status}", code=f"http_{status}"
            )
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ModelCallError("the provider returned non-JSON", code="bad_response") from exc

        text = _extract_text(decoded)
        usage = decoded.get("usage") or {}
        return ModelResponse(
            text=text,
            model_id=str(decoded.get("model") or self._model)[:120],
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
        )


def _extract_text(decoded: Any) -> str:
    """Pull the assistant text out of an OpenAI-compatible response, or raise."""
    if not isinstance(decoded, dict):
        raise ModelCallError("the provider response was not an object", code="bad_response")
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelCallError("the provider response carried no choices", code="empty_response")
    first = choices[0]
    if not isinstance(first, dict):
        raise ModelCallError("the provider choice was not an object", code="bad_response")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ModelCallError("the provider choice carried no message", code="bad_response")
    content = message.get("content")
    if not isinstance(content, str):
        raise ModelCallError("the provider message carried no text", code="empty_response")
    return content


def _as_int(value: Any) -> int | None:
    """Return ``value`` as a non-negative int, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def provider_configured() -> bool:
    """True when the environment names a provider and a model."""
    return bool(
        os.environ.get(ENV_BASE_URL, "").strip() and os.environ.get(ENV_MODEL, "").strip()
    )


def build_provider_client(
    *,
    transport: HttpTransport | None = None,
    timeout: float = 30.0,
    temperature: float = 0.0,
) -> OpenAICompatibleChatAdapter:
    """Build an adapter from the environment, or fail with an actionable message.

    ``transport`` must be supplied by the caller: this module deliberately imports no HTTP
    library, so choosing one is a deployment decision rather than a core dependency.

    Raises
    ------
    ModelCallError
        The environment is not configured, or no transport was supplied.  The message names the
        missing setting, so a misconfiguration is obvious rather than mysterious.
    """
    base_url = os.environ.get(ENV_BASE_URL, "").strip()
    model = os.environ.get(ENV_MODEL, "").strip()
    if not base_url or not model:
        missing = [
            name for name, value in ((ENV_BASE_URL, base_url), (ENV_MODEL, model)) if not value
        ]
        raise ModelCallError(
            "no live provider is configured; set " + ", ".join(missing),
            code="not_configured",
        )
    if transport is None:
        raise ModelCallError(
            "a transport must be supplied (this module imports no HTTP client on purpose)",
            code="not_configured",
        )
    return OpenAICompatibleChatAdapter(
        base_url=base_url,
        model=model,
        transport=transport,
        api_key=os.environ.get(ENV_API_KEY) or None,
        timeout=timeout,
        temperature=temperature,
    )
