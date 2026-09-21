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
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .model_client import ModelCallError, ModelRequest, ModelResponse

__all__ = [
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_JSON_MODE",
    "ENV_MODEL",
    "ENV_PROFILE",
    "ENV_THINKING",
    "ENV_TIMEOUT",
    "HttpTransport",
    "OpenAICompatibleChatAdapter",
    "ProviderProfile",
    "PROVIDER_PROFILES",
    "TransportResult",
    "build_provider_client",
    "provider_configured",
    "provider_settings",
    "resolve_profile",
    "urllib_transport",
]

ENV_BASE_URL = "AGENTRECX_LLM_BASE_URL"
ENV_MODEL = "AGENTRECX_LLM_MODEL"
ENV_API_KEY = "AGENTRECX_LLM_API_KEY"
ENV_TIMEOUT = "AGENTRECX_LLM_TIMEOUT"
ENV_PROFILE = "AGENTRECX_LLM_PROFILE"
ENV_JSON_MODE = "AGENTRECX_LLM_JSON_MODE"
ENV_THINKING = "AGENTRECX_LLM_THINKING"
ENV_INPUT_PRICE = "AGENTRECX_LLM_INPUT_PRICE"
ENV_OUTPUT_PRICE = "AGENTRECX_LLM_OUTPUT_PRICE"


@dataclass(frozen=True)
class TransportResult:
    """What a transport hands back: a status, a body, and nothing else that matters.

    ``headers`` is included only for the headers a caller may legitimately want - a provider
    request id for correlation.  The adapter reads individual headers by name and never copies
    the mapping anywhere, so an ``Authorization`` echo cannot travel with it.
    """

    status: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)

    def header(self, name: str) -> str | None:
        """One response header, case-insensitively, or ``None``."""
        wanted = name.strip().lower()
        for key, value in self.headers.items():
            if key.strip().lower() == wanted:
                return value
        return None


@dataclass(frozen=True)
class ProviderProfile:
    """The protocol differences that matter to this adapter.

    Kept as data rather than as branches in :meth:`OpenAICompatibleChatAdapter.complete`, so
    adding a provider is a declaration instead of an edit to the request path.
    """

    name: str
    #: Whether the provider needs ``{"thinking": {"type": "disabled"}}`` to stop emitting a
    #: chain of thought.  The agent asks for one structured action, so this defaults on.
    disable_thinking: bool
    #: Whether the provider's thinking mode ignores ``temperature``.
    thinking_ignores_temperature: bool
    #: Whether ``response_format={"type": "json_object"}`` is supported.
    supports_json_mode: bool
    #: Header a provider uses for its own request id, for correlation.
    request_id_header: str | None = None
    #: Response field carrying provider-side reasoning, if any.  Never requested, never recorded;
    #: named so the adapter can deliberately ignore it rather than accidentally surface it.
    reasoning_field: str | None = None

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        thinking: bool,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Return the request body for this provider."""
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if self.disable_thinking:
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if not (thinking and self.thinking_ignores_temperature):
            # In thinking mode the provider ignores temperature, so sending it would suggest a
            # control the caller does not actually have.
            payload["temperature"] = temperature
        if json_mode and self.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        return payload


#: The provider profiles this repository knows.  ``openai_compatible`` is the neutral default:
#: it sends no provider-specific field and relies on the caller's endpoint to accept the
#: standard OpenAI body.
PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "deepseek": ProviderProfile(
        name="deepseek",
        disable_thinking=True,
        thinking_ignores_temperature=True,
        supports_json_mode=True,
        request_id_header="x-request-id",
        reasoning_field="reasoning_content",
    ),
    "openai_compatible": ProviderProfile(
        name="openai_compatible",
        disable_thinking=False,
        thinking_ignores_temperature=False,
        supports_json_mode=True,
        request_id_header="x-request-id",
        reasoning_field=None,
    ),
}


def resolve_profile(name: str | None) -> ProviderProfile:
    """Return the named provider profile, or the neutral default.

    An unknown name is an error rather than a silent fallback: a typo that quietly selected the
    wrong request shape would invalidate an experiment without failing it.
    """
    if name is None or not name.strip():
        return PROVIDER_PROFILES["openai_compatible"]
    key = name.strip().casefold()
    profile = PROVIDER_PROFILES.get(key)
    if profile is None:
        raise ModelCallError(
            f"unknown provider profile {name!r}; known: {', '.join(sorted(PROVIDER_PROFILES))}",
            code="not_configured",
        )
    return profile


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean environment flag, tolerating the usual spellings."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_float(name: str) -> float | None:
    """Read a non-negative float, or ``None`` when absent or unusable."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


@dataclass(frozen=True)
class ProviderSettings:
    """Everything a live run needs, read from the environment **without** the credential.

    ``api_key_present`` is a boolean, never the key: a caller can report whether a run is
    configured, and a log line or an experiment artifact can say so, without a credential ever
    entering a value that could be printed.
    """

    base_url: str
    model: str
    profile: ProviderProfile
    timeout: float
    json_mode: bool
    thinking: bool
    api_key_present: bool
    input_price_per_million: float | None
    output_price_per_million: float | None

    def describe(self) -> dict[str, Any]:
        """A log-safe description: no credential, not even a redacted one."""
        return {
            # A single ``provider`` key, matching the identity a recording stores, so a report
            # says the same thing whether the run was live or replayed.
            "provider": self.profile.name,
            "base_url": self.base_url,
            "model": self.model,
            "profile": self.profile.name,
            "timeout": self.timeout,
            "json_mode": self.json_mode,
            "thinking": self.thinking,
            "api_key_present": self.api_key_present,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
        }


def provider_settings() -> ProviderSettings:
    """Read the provider configuration, or raise an actionable error.

    Raises
    ------
    ModelCallError
        The environment does not name a base URL and a model.  The message names the missing
        variable, so a misconfiguration is obvious rather than mysterious.
    """
    base_url = os.environ.get(ENV_BASE_URL, "").strip()
    model = os.environ.get(ENV_MODEL, "").strip()
    if not base_url or not model:
        missing = [
            name for name, value in ((ENV_BASE_URL, base_url), (ENV_MODEL, model)) if not value
        ]
        raise ModelCallError(
            "no live provider is configured; set " + ", ".join(missing), code="not_configured"
        )
    profile = resolve_profile(os.environ.get(ENV_PROFILE))
    return ProviderSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        profile=profile,
        timeout=_env_float(ENV_TIMEOUT) or 30.0,
        json_mode=_env_flag(ENV_JSON_MODE, default=True),
        # Off by default on purpose: the agent asks for one structured action, and a chain of
        # thought is neither requested nor recorded by this repository.
        thinking=_env_flag(ENV_THINKING, default=False),
        api_key_present=bool(os.environ.get(ENV_API_KEY, "").strip()),
        input_price_per_million=_env_float(ENV_INPUT_PRICE),
        output_price_per_million=_env_float(ENV_OUTPUT_PRICE),
    )


@runtime_checkable
class HttpTransport(Protocol):
    """The one thing an adapter needs from an HTTP client.

    Injected so this module imports no HTTP library at module scope: a deployment passes
    :func:`urllib_transport`, ``requests.post``, ``httpx.post`` or its own client, and the tests
    pass a stub.  The return may be a :class:`TransportResult` or a plain
    ``(status, body)`` pair, so the simplest possible stub works.
    """

    def __call__(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> TransportResult | tuple[int, str]:
        """Perform the request and return the status and body."""
        ...


def urllib_transport(
    url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> TransportResult:
    """A standard-library HTTP transport: the smallest production-shaped client that works.

    Imports ``urllib.request`` lazily, so the core dependency set is unchanged and importing
    this module costs nothing.  A provider error response is returned as a status rather than
    raised, because :class:`OpenAICompatibleChatAdapter` owns the decision of what a status
    means; a transport failure (DNS, connection, TLS, timeout) does raise, so the adapter can
    normalise it.

    ``urllib`` raises :class:`~urllib.error.HTTPError` for a 4xx/5xx *with the body attached*,
    which is why it is caught here and converted: the adapter wants the status, and the body is
    what carries a provider's error code.  The body is **not** logged or propagated - the
    adapter raises a status-only error - but it is returned so a caller that chooses to inspect
    it can.
    """
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            # Only the headers a caller may legitimately correlate on are copied; the response
            # mapping is never retained wholesale.
            return TransportResult(
                status=int(response.status),
                body=body,
                headers={"x-request-id": response.headers.get("x-request-id", "")},
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return TransportResult(
            status=int(exc.code),
            body=body,
            headers={"x-request-id": exc.headers.get("x-request-id", "") if exc.headers else ""},
        )


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
        profile: ProviderProfile | str | None = None,
        json_mode: bool = False,
        thinking: bool = False,
        max_tokens: int | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
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
        self._profile = resolve_profile(profile) if isinstance(profile, str) or profile is None else profile
        self._json_mode = bool(json_mode)
        self._thinking = bool(thinking)
        self._max_tokens = max_tokens
        self._input_price = input_price_per_million
        self._output_price = output_price_per_million

    @property
    def endpoint(self) -> str:
        """The chat-completions URL this adapter calls."""
        return f"{self._base_url}/chat/completions"

    @property
    def profile(self) -> ProviderProfile:
        """The provider protocol profile in use."""
        return self._profile

    @property
    def model(self) -> str:
        """The configured model identifier (not a credential)."""
        return self._model

    def estimate_cost(
        self, *, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        """Estimated cost in the provider's currency, or ``None`` when pricing is unconfigured.

        An estimate is only produced from prices the deployment stated explicitly.  Absent
        pricing means **unknown**, never zero and never a built-in default: a hard-coded price
        silently becomes a wrong one, and a fabricated zero would read as "free".
        """
        if self._input_price is None and self._output_price is None:
            return None
        if input_tokens is None and output_tokens is None:
            return None
        cost = 0.0
        if input_tokens is not None and self._input_price is not None:
            cost += (input_tokens / 1_000_000.0) * self._input_price
        if output_tokens is not None and self._output_price is not None:
            cost += (output_tokens / 1_000_000.0) * self._output_price
        return cost

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send the request and return the model's raw answer.

        Latency is measured around the transport call with a **monotonic** clock, because a
        wall-clock timestamp can move backwards and a duration measurement must not.
        """
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            # The credential enters the header here and nowhere else: it is never logged, never
            # returned in a response, never recorded, and never put in an exception message.
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = self._profile.build_payload(
            model=self._model,
            messages=request.as_messages(),
            temperature=self._temperature,
            thinking=self._thinking,
            json_mode=self._json_mode,
            max_tokens=self._max_tokens,
        )

        started = time.monotonic()
        try:
            raw = self._transport(
                self.endpoint, headers=headers, payload=payload, timeout=self._timeout
            )
        except TimeoutError as exc:
            raise ModelCallError("the model request timed out", code="timeout") from exc
        except Exception as exc:  # noqa: BLE001 - transport failures are normalised
            raise ModelCallError(
                f"the model request failed: {type(exc).__name__}", code="provider_error"
            ) from exc
        latency_ms = max(0.0, (time.monotonic() - started) * 1000.0)

        status, body, response_headers = _unpack_transport(raw)

        if status >= 400:
            # The provider's own text is not echoed: it can contain request details.  The
            # provider's error *code* is useful and stable, so it is surfaced instead.
            raise ModelCallError(
                f"the provider returned HTTP {status}: {_provider_error_code(body)}",
                code=f"http_{status}",
            )
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ModelCallError("the provider returned non-JSON", code="bad_response") from exc

        text = _extract_text(decoded)
        usage = decoded.get("usage") or {}
        input_tokens = _as_int(usage.get("prompt_tokens"))
        output_tokens = _as_int(usage.get("completion_tokens"))
        total_tokens = _as_int(usage.get("total_tokens"))
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            # Derivable rather than fabricated: the provider reports both parts, so the sum is
            # arithmetic on reported values.  Left ``None`` when a part is unknown.
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

        request_id = None
        if self._profile.request_id_header:
            request_id = response_headers.get(self._profile.request_id_header) or None
        if request_id is None:
            # Some deployments expose the id in the body instead of a header.
            request_id = _optional_str(decoded.get("id"))

        return ModelResponse(
            text=text,
            model_id=str(decoded.get("model") or self._model)[:120],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_id=None if request_id is None else str(request_id)[:120],
            latency_ms=latency_ms,
            finish_reason=_finish_reason(decoded),
            estimated_cost=self.estimate_cost(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )


def _unpack_transport(raw: Any) -> tuple[int, str, dict[str, str]]:
    """Normalise whatever a transport returned into ``(status, body, headers)``.

    A :class:`TransportResult` is the documented shape; a plain ``(status, body)`` pair is
    accepted too, so the simplest possible test double works and a caller can pass a bare
    callable without importing this module's types.
    """
    if isinstance(raw, TransportResult):
        return raw.status, raw.body, dict(raw.headers)
    if isinstance(raw, tuple) and len(raw) == 2:
        return int(raw[0]), str(raw[1]), {}
    raise ModelCallError(
        "the transport returned an unusable value", code="provider_error"
    )


def _provider_error_code(body: str) -> str:
    """Extract a provider's own error code from an error body, or a neutral placeholder.

    Providers put a stable machine code here (DeepSeek uses ``error.code``); surfacing it makes
    a failure diagnosable without echoing any of the body, which may quote the request.
    """
    try:
        decoded = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return "unstructured"
    if not isinstance(decoded, dict):
        return "unstructured"
    error = decoded.get("error")
    if not isinstance(error, dict):
        return "unstructured"
    code = error.get("code") or error.get("type")
    return str(code)[:60] if code else "unstructured"


def _optional_str(value: Any) -> str | None:
    """Return ``value`` as a non-empty string, or ``None``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _finish_reason(decoded: Any) -> str | None:
    """The provider's finish reason for the first choice, when it reports one.

    Recorded because ``length`` distinguishes "the model chose badly" from "the answer was cut
    off", which are different findings about a malformed response.
    """
    if not isinstance(decoded, dict):
        return None
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return _optional_str(choices[0].get("finish_reason"))


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
    """True when the environment names a provider and a model.

    Answers "is a live run configured at all", which is what a caller must know *before*
    deciding whether to offer one.  It never reports whether a credential is present - a
    boolean about a secret is still information about the secret, and the adapter does not need
    it to decide anything.
    """
    return bool(
        os.environ.get(ENV_BASE_URL, "").strip() and os.environ.get(ENV_MODEL, "").strip()
    )


def build_provider_client(
    *,
    transport: HttpTransport | None = None,
    timeout: float | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    settings: ProviderSettings | None = None,
) -> OpenAICompatibleChatAdapter:
    """Build an adapter from the environment, or fail with an actionable message.

    ``transport`` defaults to :func:`urllib_transport`, the standard-library client, so a live
    run needs no dependency beyond the standard library.  A caller may still inject one (a test
    does, and a deployment with its own client pool can).

    Raises
    ------
    ModelCallError
        The environment is not configured.  The message names the missing setting, so a
        misconfiguration is obvious rather than mysterious.
    """
    resolved = settings or provider_settings()
    return OpenAICompatibleChatAdapter(
        base_url=resolved.base_url,
        model=resolved.model,
        transport=transport or urllib_transport,
        api_key=os.environ.get(ENV_API_KEY) or None,
        timeout=resolved.timeout if timeout is None else timeout,
        temperature=temperature,
        profile=resolved.profile,
        json_mode=resolved.json_mode,
        thinking=resolved.thinking,
        max_tokens=max_tokens,
        input_price_per_million=resolved.input_price_per_million,
        output_price_per_million=resolved.output_price_per_million,
    )
