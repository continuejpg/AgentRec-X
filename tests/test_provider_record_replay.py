"""Phase 3 offline tests: the provider seam and the record/replay layer.

Everything here runs with **no network and no credential**.  A live provider call is exercised
only by the explicit experiment commands (``experiments.llm_policy_smoke --mode live`` and
``experiments.live_agent_evaluation --mode record``), never by ``pytest``.

What is pinned:

* provider response parsing, including the shapes a provider uses to say "no answer"
* provider error normalisation: HTTP status, transport failure, timeout, non-JSON
* the provider *profile* differences that matter (thinking mode, JSON mode, temperature)
* usage extraction, with unknown preserved as unknown
* latency measured with a monotonic clock
* cost: estimated only from explicit pricing, otherwise unknown
* the canonical request fingerprint: stable, and sensitive to every input that changes an answer
* record -> replay round trips the model text
* replay is offline by construction and never falls back to the network
* record/replay produce the same **trajectory**, and therefore the same policy decisions
* trajectory model metadata distinguishes live, replay and scripted
* no credential can reach a recording, a log or a response
* live execution takes two deliberate flags
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control.model_client import (  # noqa: E402
    ModelCallError,
    ModelRequest,
    ModelResponse,
    ScriptedModelClient,
)
from recommendation.control.model_policy import (  # noqa: E402
    LLMAgentPolicy,
    POLICY_PROMPT_VERSION,
)
from recommendation.control.model_recorder import (  # noqa: E402
    DEFAULT_RECORDINGS_DIR,
    ModelSink,
    RECORDING_FORMAT_VERSION,
    RecordingFormatError,
    RecordingModelClient,
    RecordingStore,
    ReplayMode,
    fingerprint_request,
    recording_identity,
    replay_client,
)
from recommendation.control.provider_adapter import (  # noqa: E402
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_INPUT_PRICE,
    ENV_JSON_MODE,
    ENV_MODEL,
    ENV_OUTPUT_PRICE,
    ENV_PROFILE,
    ENV_THINKING,
    ENV_TIMEOUT,
    OpenAICompatibleChatAdapter,
    ProviderProfile,
    TransportResult,
    build_provider_client,
    provider_configured,
    provider_settings,
    resolve_profile,
    urllib_transport,
)
from recommendation.control.context import CandidateState, PolicyContext  # noqa: E402
from recommendation.control.schemas import ActionKind  # noqa: E402

#: A minimal well-formed provider answer.
ANSWER = {"action": "finish", "rationale": "the request is answered"}


def _body(text: str = "", **extra: Any) -> str:
    """An OpenAI-compatible response body."""
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "model": "deepseek-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
    payload.update(extra)
    return json.dumps(payload)


class _RecordingTransport:
    """A transport stub that records what it was asked and replays a scripted result."""

    def __init__(self, *results: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._results = list(results)

    def __call__(self, url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float):
        self.calls.append({"url": url, "headers": dict(headers), "payload": payload, "timeout": timeout})
        result = self._results[min(len(self.calls) - 1, len(self._results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def _adapter(transport: Any, **kwargs: Any) -> OpenAICompatibleChatAdapter:
    defaults: dict[str, Any] = {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "transport": transport,
        "profile": "deepseek",
    }
    defaults.update(kwargs)
    return OpenAICompatibleChatAdapter(**defaults)


def _request(**kwargs: Any) -> ModelRequest:
    base: dict[str, Any] = {
        "system_prompt": "Choose one action as JSON.",
        "context_payload": {"candidates": {"candidate_refs": []}},
        "action_schema": ({"action": "finish"},),
    }
    base.update(kwargs)
    return ModelRequest(**base)


# =========================================================================== #
# A. Provider response parsing
# =========================================================================== #


def test_a_provider_answer_becomes_response_text() -> None:
    """The documented shape parses, and everything the provider reported is retained."""
    transport = _RecordingTransport(
        TransportResult(
            status=200,
            body=_body(
                json.dumps(ANSWER),
                usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
            ),
            headers={"x-request-id": "req-abc"},
        )
    )
    response = _adapter(transport).complete(_request())
    assert json.loads(response.text) == ANSWER
    assert response.model_id == "deepseek-flash"
    assert (response.input_tokens, response.output_tokens, response.total_tokens) == (120, 30, 150)
    assert response.request_id == "req-abc"
    assert response.finish_reason == "stop"
    assert response.is_empty is False


def test_a_bare_status_body_pair_is_accepted() -> None:
    """A transport may return a plain pair, so the simplest stub works."""
    transport = _RecordingTransport((200, _body(json.dumps(ANSWER))))
    response = _adapter(transport).complete(_request())
    assert json.loads(response.text) == ANSWER


def test_the_total_is_derived_only_when_both_parts_were_reported() -> None:
    """A partially reported usage is never padded into a total that looks measured."""
    both = _RecordingTransport(
        TransportResult(200, _body("{}", usage={"prompt_tokens": 10, "completion_tokens": 4}))
    )
    assert _adapter(both).complete(_request()).total_tokens == 14

    only_input = _RecordingTransport(
        TransportResult(200, _body("{}", usage={"prompt_tokens": 10}))
    )
    response = _adapter(only_input).complete(_request())
    assert response.input_tokens == 10
    assert response.output_tokens is None
    assert response.total_tokens is None


def test_absent_usage_stays_unknown() -> None:
    """A provider that reports no usage yields ``None``, not zero."""
    response = _adapter(_RecordingTransport(TransportResult(200, _body("{}")))).complete(_request())
    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.total_tokens is None
    assert response.usage()["input_tokens"] is None


def test_the_reasoning_field_is_never_surfaced() -> None:
    """A chain of thought in the response never reaches the text or the diagnostics.

    The adapter is configured knowing DeepSeek's reasoning field name, so this asserts it is
    deliberately ignored rather than accidentally forwarded.
    """
    body = json.dumps(
        {
            "model": "deepseek-flash",
            "choices": [
                {
                    "message": {"content": json.dumps(ANSWER), "reasoning_content": "SECRET THOUGHTS"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    response = _adapter(_RecordingTransport(TransportResult(200, body))).complete(_request())
    assert "SECRET THOUGHTS" not in response.text
    assert "SECRET" not in response.model_dump_json()


# =========================================================================== #
# B. Provider error normalisation
# =========================================================================== #


@pytest.mark.parametrize("status", [400, 401, 402, 429, 500, 503])
def test_an_http_error_becomes_a_normalised_model_call_error(status: int) -> None:
    """Every provider status maps onto one error type with a stable, status-derived code."""
    transport = _RecordingTransport(TransportResult(status, '{"error": {"code": "rate_limit"}}'))
    with pytest.raises(ModelCallError) as caught:
        _adapter(transport).complete(_request())
    assert caught.value.code == f"http_{status}"
    assert str(status) in str(caught.value)


def test_a_provider_error_does_not_echo_the_response_body() -> None:
    """The error carries the provider's code but none of its text, which may quote the request."""
    noisy = json.dumps({"error": {"code": "invalid_request", "message": "SECRET-REQUEST-DUMP"}})
    with pytest.raises(ModelCallError) as caught:
        _adapter(_RecordingTransport(TransportResult(400, noisy))).complete(_request())
    assert "invalid_request" in str(caught.value)
    assert "SECRET-REQUEST-DUMP" not in str(caught.value)


def test_a_transport_failure_is_normalised_without_the_message() -> None:
    """A connection error is reported by type name, never by its own text."""
    transport = _RecordingTransport(OSError("connection refused to internal-host"))
    with pytest.raises(ModelCallError) as caught:
        _adapter(transport).complete(_request())
    assert caught.value.code == "provider_error"
    assert "internal-host" not in str(caught.value)


def test_a_timeout_is_reported_as_a_timeout() -> None:
    """A timeout is distinguishable from an execution failure."""
    with pytest.raises(ModelCallError) as caught:
        _adapter(_RecordingTransport(TimeoutError("timed out"))).complete(_request())
    assert caught.value.code == "timeout"


def test_a_non_json_body_is_a_bad_response() -> None:
    """A gateway returning HTML is a response problem, not a parse problem in the policy."""
    with pytest.raises(ModelCallError) as caught:
        _adapter(_RecordingTransport(TransportResult(200, "<html>oops</html>"))).complete(_request())
    assert caught.value.code == "bad_response"


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({"choices": []}),
        json.dumps({"choices": [{"message": {}}]}),
        json.dumps({"choices": [{"message": {"content": None}}]}),
        json.dumps({"choices": "not-a-list"}),
    ],
)
def test_a_response_without_usable_text_is_an_empty_response(body: str) -> None:
    """Every "the provider answered with a shaped object but no text" case maps onto one code."""
    with pytest.raises(ModelCallError) as caught:
        _adapter(_RecordingTransport(TransportResult(200, body))).complete(_request())
    assert caught.value.code == "empty_response"


def test_a_body_that_is_not_an_object_is_a_bad_response() -> None:
    """A JSON body of the wrong type is a protocol problem, not a missing answer."""
    with pytest.raises(ModelCallError) as caught:
        _adapter(_RecordingTransport(TransportResult(200, json.dumps([1, 2, 3])))).complete(
            _request()
        )
    assert caught.value.code == "bad_response"


# =========================================================================== #
# C. Provider profiles
# =========================================================================== #


def test_the_deepseek_profile_disables_thinking_and_requests_json() -> None:
    """The two provider differences that matter to a structured control decision."""
    transport = _RecordingTransport(TransportResult(200, _body(json.dumps(ANSWER))))
    _adapter(transport, json_mode=True).complete(_request())
    payload = transport.calls[0]["payload"]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False


def test_thinking_mode_suppresses_temperature() -> None:
    """In thinking mode the provider ignores temperature, so sending it would imply control."""
    transport = _RecordingTransport(TransportResult(200, _body("{}")))
    _adapter(transport, thinking=True, temperature=0.0).complete(_request())
    assert "temperature" not in transport.calls[0]["payload"]


def test_the_neutral_profile_sends_no_provider_specific_field() -> None:
    """An unknown endpoint gets the standard body, with temperature honoured."""
    transport = _RecordingTransport(TransportResult(200, _body("{}")))
    _adapter(transport, profile="openai_compatible", temperature=0.25).complete(_request())
    payload = transport.calls[0]["payload"]
    assert "thinking" not in payload
    assert payload["temperature"] == 0.25


def test_an_unknown_profile_is_an_error_rather_than_a_default() -> None:
    """A typo must fail, not silently select a different request shape."""
    with pytest.raises(ModelCallError) as caught:
        resolve_profile("deepsek")
    assert caught.value.code == "not_configured"
    assert "deepseek" in str(caught.value)


def test_the_endpoint_is_the_openai_compatible_chat_path() -> None:
    """DeepSeek's documented base URL plus the chat path."""
    assert _adapter(lambda *a, **k: None).endpoint == "https://api.deepseek.com/chat/completions"
    assert _adapter(lambda *a, **k: None, base_url="https://api.deepseek.com/").endpoint == (
        "https://api.deepseek.com/chat/completions"
    )


# =========================================================================== #
# D. Configuration
# =========================================================================== #


def test_configuration_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every documented variable is honoured, and the credential is never read into the report."""
    monkeypatch.setenv(ENV_BASE_URL, "https://api.deepseek.com")
    monkeypatch.setenv(ENV_MODEL, "deepseek-flash")
    # Non-key-shaped, but still containing the word "secret": the last assertion below checks that
    # the *description* leaks no part of the credential, and that check is only meaningful while
    # the credential contains something worth finding.  No real credential is involved.
    monkeypatch.setenv(ENV_API_KEY, "not-a-real-secret-canary")
    monkeypatch.setenv(ENV_TIMEOUT, "12.5")
    monkeypatch.setenv(ENV_PROFILE, "deepseek")
    monkeypatch.setenv(ENV_INPUT_PRICE, "0.30")
    monkeypatch.setenv(ENV_OUTPUT_PRICE, "1.20")
    settings = provider_settings()
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"
    assert settings.timeout == 12.5
    assert settings.profile.name == "deepseek"
    assert settings.input_price_per_million == 0.30
    assert settings.output_price_per_million == 1.20
    assert settings.api_key_present is True
    # The description is what gets logged and reported: no credential, in any form.
    described = json.dumps(settings.describe())
    assert "sk-secret-value" not in described
    assert "secret" not in described.casefold()
    assert described.count("api_key_present") == 1


def test_missing_configuration_names_the_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misconfiguration is an actionable message, not a mystery."""
    for name in (ENV_BASE_URL, ENV_MODEL):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelCallError) as caught:
        provider_settings()
    assert caught.value.code == "not_configured"
    assert ENV_BASE_URL in str(caught.value) and ENV_MODEL in str(caught.value)
    assert provider_configured() is False


def test_booleans_and_floats_tolerate_absent_or_bad_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd optional setting falls back to its default instead of breaking a run."""
    monkeypatch.setenv(ENV_BASE_URL, "https://api.deepseek.com")
    monkeypatch.setenv(ENV_MODEL, "deepseek-flash")
    monkeypatch.setenv(ENV_TIMEOUT, "not-a-number")
    monkeypatch.setenv(ENV_THINKING, "maybe")
    settings = provider_settings()
    assert settings.timeout == 30.0
    # Thinking stays off: an unreadable flag must never enable chain-of-thought.
    assert settings.thinking is False

    monkeypatch.setenv(ENV_THINKING, "true")
    assert provider_settings().thinking is True
    monkeypatch.setenv(ENV_JSON_MODE, "0")
    assert provider_settings().json_mode is False


# =========================================================================== #
# E. Usage, latency and cost
# =========================================================================== #


def test_latency_is_measured_with_a_monotonic_clock() -> None:
    """The adapter times the transport, and reports a non-negative duration."""
    ticks = iter([100.0, 100.25])
    import recommendation.control.provider_adapter as adapter_module

    original = adapter_module.time.monotonic
    adapter_module.time.monotonic = lambda: next(ticks)  # type: ignore[assignment]
    try:
        response = _adapter(
            _RecordingTransport(TransportResult(200, _body("{}")))
        ).complete(_request())
    finally:
        adapter_module.time.monotonic = original  # type: ignore[assignment]
    assert response.latency_ms == pytest.approx(250.0)


def test_cost_is_estimated_only_from_configured_pricing() -> None:
    """An estimate needs explicit prices; without them the cost is unknown, never zero."""
    body = _body("{}", usage={"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
    unpriced = _adapter(_RecordingTransport(TransportResult(200, body))).complete(_request())
    assert unpriced.estimated_cost is None

    priced = _adapter(
        _RecordingTransport(TransportResult(200, body)),
        input_price_per_million=0.30,
        output_price_per_million=1.20,
    ).complete(_request())
    # 1M input at 0.30 plus 0.5M output at 1.20.
    assert priced.estimated_cost == pytest.approx(0.30 + 0.60)


def test_cost_stays_unknown_when_usage_is_unreported() -> None:
    """Pricing without usage is still unknown: the arithmetic has nothing to work from."""
    response = _adapter(
        _RecordingTransport(TransportResult(200, _body("{}"))),
        input_price_per_million=0.30,
    ).complete(_request())
    assert response.estimated_cost is None


def test_a_model_response_without_latency_reports_unknown() -> None:
    """A client that does not measure latency says so, instead of reporting zero."""
    assert ModelResponse(text="{}").latency_ms is None
    assert ModelResponse(text="{}", latency_ms=0.0).latency_ms == 0.0


# =========================================================================== #
# F. The standard-library transport
# =========================================================================== #


def test_the_urllib_transport_is_callable_and_imports_nothing_at_module_scope() -> None:
    """The transport exists without the adapter importing an HTTP client eagerly."""
    import recommendation.control.provider_adapter as adapter_module

    assert callable(urllib_transport)
    assert not hasattr(adapter_module, "urllib")


# =========================================================================== #
# G. Canonical fingerprinting
# =========================================================================== #


def _fingerprint(request: ModelRequest, **kwargs: Any) -> str:
    defaults: dict[str, Any] = {
        "provider": "deepseek",
        "model": "deepseek-flash",
        "prompt_version": 1,
        "parse_protocol_version": 2,
    }
    defaults.update(kwargs)
    return fingerprint_request(request, **defaults).digest


def test_the_fingerprint_is_stable_across_dict_insertion_order() -> None:
    """Two materially identical requests share a digest regardless of construction order."""
    first = _request(context_payload={"a": 1, "b": [1, 2], "c": {"d": True}})
    second = _request(context_payload={"c": {"d": True}, "b": [1, 2], "a": 1})
    assert _fingerprint(first) == _fingerprint(second)


def test_the_fingerprint_is_stable_across_processes() -> None:
    """A recorded digest is reproducible, not dependent on hash randomisation."""
    request = _request(context_payload={"x": [1, 2, 3]})
    expected = _fingerprint(request)
    # Recomputing in a fresh interpreter is what a later replay does.
    import subprocess

    code = (
        "import sys, json; sys.path.insert(0, '.');"
        "from recommendation.control.model_client import ModelRequest;"
        "from recommendation.control.model_recorder import fingerprint_request;"
        "r = ModelRequest(system_prompt='Choose one action as JSON.',"
        " context_payload={'x': [1, 2, 3]}, action_schema=({'action': 'finish'},));"
        "print(fingerprint_request(r, provider='deepseek', model='deepseek-flash',"
        " prompt_version=1, parse_protocol_version=2).digest)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT), check=True
    )
    assert completed.stdout.strip() == expected


@pytest.mark.parametrize(
    "label, changed",
    [
        ("the observation changed", _request(context_payload={"last_observation": {"status": "ok"}})),
        (
            "the candidate references changed",
            _request(context_payload={"candidates": {"candidate_refs": [{"parent_asin": "cand-red"}]}}),
        ),
        ("the offered actions changed", _request(action_schema=({"action": "search_catalog"},))),
        ("the correction changed", _request(correction="your previous answer was rejected")),
        ("the system prompt changed", _request(system_prompt="A different contract.")),
        ("the constraint state changed", _request(context_payload={"active_constraints": ["price_max<=35"]})),
    ],
)
def test_the_fingerprint_changes_when_the_decision_inputs_change(label: str, changed: ModelRequest) -> None:
    """Every input that can change the answer changes the key.

    This is the property that makes replay safe: a recording cannot be replayed against a
    request that differs in the observation, the candidate references, the offered menu, the
    retry correction, the prompt or the constraint state.
    """
    assert _fingerprint(_request()) != _fingerprint(changed), label


def test_the_fingerprint_is_sensitive_to_provider_model_and_prompt_version() -> None:
    """A recording made against one provider/model/prompt is not reusable against another."""
    request = _request()
    base = _fingerprint(request)
    assert _fingerprint(request, model="deepseek-v4-pro") != base
    assert _fingerprint(request, provider="openai_compatible") != base
    assert _fingerprint(request, prompt_version=2) != base
    assert _fingerprint(request, prompt_version=None) != base
    # The parse protocol is part of the key: a trace recorded under one interpretation of an
    # answer must not be replayed as though it were made under another.
    assert _fingerprint(request, parse_protocol_version=1) != _fingerprint(
        request, parse_protocol_version=2
    )


def test_the_fingerprint_does_not_depend_on_the_user_request_alone() -> None:
    """Two turns of one conversation with different state must not share a recording."""
    same_text = {"user_request": "recommend gear"}
    first = _request(context_payload={**same_text, "budget": {"step_index": 0}})
    second = _request(context_payload={**same_text, "budget": {"step_index": 1}})
    assert _fingerprint(first) != _fingerprint(second)


def test_the_fingerprint_exposes_what_it_hashed() -> None:
    """A replay miss can be explained rather than guessed at."""
    fingerprint = fingerprint_request(
        _request(), provider="deepseek", model="deepseek-flash", prompt_version=1
    )
    assert set(fingerprint.canonical) == {
        "format",
        "provider",
        "model",
        "prompt_version",
        "parse_protocol_version",
        "system_prompt_sha256",
        "context_payload",
        "action_schema",
        "correction",
    }
    assert fingerprint.short() == fingerprint.digest[:12]
    assert len(fingerprint.digest) == 64


# =========================================================================== #
# H. Record and replay
# =========================================================================== #


def _store(tmp_path: Path, name: str = "run.jsonl") -> RecordingStore:
    return RecordingStore(tmp_path / name)


def test_recording_writes_a_versioned_entry_with_the_request_and_response(tmp_path: Path) -> None:
    """A recording holds enough to reproduce the decision, and nothing sensitive."""
    store = _store(tmp_path)
    sink = ModelSink()
    client = RecordingModelClient(
        ScriptedModelClient([ANSWER]),
        store=store,
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
        sink=sink,
    )
    request = _request(context_payload={"candidates": {"candidate_refs": []}, "probe": "recording"})
    response = client.complete(request)

    assert json.loads(response.text) == ANSWER
    assert client.live_calls == 1 and client.replayed_calls == 0
    entries = store.entries_for(client.fingerprint(request).digest)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["format_version"] == RECORDING_FORMAT_VERSION
    assert entry["provider"] == "deepseek"
    assert entry["model"] == "deepseek-flash"
    assert entry["prompt_version"] == 1
    assert entry["request"]["system_prompt"] == request.system_prompt
    assert entry["request"]["context_payload"] == request.context_payload
    assert entry["response"]["text"] == response.text
    assert isinstance(entry["timestamp"], (int, float))
    assert sink.summary()["model_calls"] == 1


def test_replay_returns_the_recorded_text_without_calling_the_wrapped_client(tmp_path: Path) -> None:
    """The wrapped client is not merely unused, it is unreachable."""

    class Exploding:
        model = "deepseek-flash"

        def complete(self, request: Any) -> Any:  # pragma: no cover - must not be called
            raise AssertionError("replay reached the provider")

    store = _store(tmp_path)
    RecordingModelClient(
        ScriptedModelClient([ANSWER]),
        store=store,
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    ).complete(_request())

    replayed = RecordingModelClient(
        Exploding(),
        store=RecordingStore(store.path),
        mode=ReplayMode.REPLAY,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    response = replayed.complete(_request())
    assert json.loads(response.text) == ANSWER
    assert replayed.live_calls == 0 and replayed.replayed_calls == 1


def test_replay_without_a_recording_fails_and_never_falls_back(tmp_path: Path) -> None:
    """A miss is an error, not a silent provider call - that would destroy reproducibility."""
    store = _store(tmp_path, "empty.jsonl")
    store.path.write_text("", encoding="utf-8")

    class Exploding:
        def complete(self, request: Any) -> Any:  # pragma: no cover - must not be called
            raise AssertionError("replay fell back to the provider")

    client = RecordingModelClient(
        Exploding(),
        store=store,
        mode=ReplayMode.REPLAY,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    with pytest.raises(RecordingFormatError) as caught:
        client.complete(_request())
    assert caught.value.code == "recording_missing"
    assert "never falls back" in str(caught.value)
    assert client.live_calls == 0


def test_replay_ignores_a_different_request(tmp_path: Path) -> None:
    """A recording is keyed on the request, so a changed context misses rather than matches."""
    store = _store(tmp_path)
    RecordingModelClient(
        ScriptedModelClient([ANSWER]),
        store=store,
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    ).complete(_request())

    client = RecordingModelClient(
        ScriptedModelClient([ANSWER]),
        store=RecordingStore(store.path),
        mode=ReplayMode.REPLAY,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    with pytest.raises(RecordingFormatError):
        client.complete(_request(context_payload={"last_observation": {"status": "ok"}}))


def test_repeated_identical_requests_keep_every_response(tmp_path: Path) -> None:
    """A request that repeats consumes its own recorded occurrences in order.

    Overwriting the earlier answer would be invisible - the requests are byte-identical - so the
    store keeps both and the replay hands them back in sequence rather than reusing the first.
    """
    store = _store(tmp_path)
    answers = [{"action": "finish", "rationale": "first"}, {"action": "finish", "rationale": "second"}]
    recorder = RecordingModelClient(
        ScriptedModelClient(answers),
        store=store,
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    request = _request(context_payload={"candidates": {"candidate_refs": []}, "probe": "repeat"})
    recorder.complete(request)
    recorder.complete(request)
    digest = recorder.fingerprint(request).digest
    assert store.response_count(digest) == 2
    assert len(store) == 2

    replayer = RecordingModelClient(
        ScriptedModelClient(answers),
        store=RecordingStore(store.path),
        mode=ReplayMode.REPLAY,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    assert json.loads(replayer.complete(request).text)["rationale"] == "first"
    assert json.loads(replayer.complete(request).text)["rationale"] == "second"
    # A third occurrence has nothing to replay, and is refused rather than wrapped.
    with pytest.raises(RecordingFormatError) as caught:
        replayer.complete(request)
    assert caught.value.code == "recording_exhausted"


def test_a_recording_from_another_format_version_is_refused(tmp_path: Path) -> None:
    """An incompatible recording is rejected rather than mis-replayed."""
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps({"format_version": 99, "fingerprint": "x", "response": {"text": "{}"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RecordingFormatError) as caught:
        RecordingStore(path).response_for("x")
    assert "format" in str(caught.value)


def test_a_corrupt_recording_line_is_reported_with_its_position(tmp_path: Path) -> None:
    """A malformed file fails with a located error instead of a partial silent replay."""
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RecordingFormatError) as caught:
        RecordingStore(path).response_for("x")
    assert "line 1" in str(caught.value)


def test_a_recording_reports_the_identity_it_was_made_against(tmp_path: Path) -> None:
    """A replay keys on the recorded provider and model, not on the current environment."""
    store = _store(tmp_path)
    RecordingModelClient(
        ScriptedModelClient([ANSWER]),
        store=store,
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=3,
    ).complete(_request())
    identity = recording_identity(store.path)
    assert identity["provider"] == "deepseek"
    assert identity["model"] == "deepseek-flash"
    assert identity["prompt_version"] == 3


def test_an_empty_recording_has_no_identity(tmp_path: Path) -> None:
    """An empty recording is reported as empty rather than defaulted to something plausible."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RecordingFormatError) as caught:
        recording_identity(path)
    assert caught.value.code == "recording_empty"


def test_record_if_missing_replays_what_it_has_and_calls_for_the_rest(tmp_path: Path) -> None:
    """The combined mode is useful, and still records only what it actually called."""
    store = _store(tmp_path)
    sink = ModelSink()
    client = RecordingModelClient(
        ScriptedModelClient([ANSWER, ANSWER]),
        store=store,
        mode=ReplayMode.RECORD_IF_MISSING,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
        sink=sink,
    )
    first = _request()
    client.complete(first)
    assert client.live_calls == 1

    second = RecordingModelClient(
        ScriptedModelClient([{"action": "search_catalog", "arguments": {"terms": ["x"]}}]),
        store=RecordingStore(store.path),
        mode=ReplayMode.RECORD_IF_MISSING,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
        sink=sink,
    )
    # Recorded: replayed, provider untouched.
    assert json.loads(second.complete(first).text) == ANSWER
    assert second.live_calls == 0 and second.replayed_calls == 1
    # Not recorded: called and written.
    assert json.loads(second.complete(_request(correction="retry")).text)["action"] == "search_catalog"
    assert second.live_calls == 1


def test_replay_client_builds_the_right_mode_from_what_exists(tmp_path: Path) -> None:
    """The convenience builder records when there is nothing to replay, and replays when there is."""
    path = tmp_path / "auto.jsonl"
    recording = replay_client(
        wrapped=ScriptedModelClient([ANSWER]),
        recording=path,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    assert recording.mode is ReplayMode.RECORD
    recording.complete(_request())

    replayer = replay_client(
        wrapped=None, recording=path, provider="deepseek", model="deepseek-flash", prompt_version=1
    )
    assert replayer.mode is ReplayMode.REPLAY
    assert json.loads(replayer.complete(_request()).text) == ANSWER

    with pytest.raises(RecordingFormatError) as caught:
        replay_client(wrapped=None, recording=tmp_path / "absent.jsonl", provider="deepseek")
    assert caught.value.code == "recording_missing"


def test_a_wrapped_client_failure_is_normalised(tmp_path: Path) -> None:
    """A provider SDK's own exception type cannot escape the recorder."""

    class RudeClient:
        def complete(self, request: Any) -> Any:
            raise ValueError("internal detail")

    client = RecordingModelClient(
        RudeClient(),
        store=_store(tmp_path),
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
    )
    with pytest.raises(ModelCallError) as caught:
        client.complete(_request())
    assert caught.value.code == "provider_error"
    assert "internal detail" not in str(caught.value)


def test_no_credential_can_reach_a_recording(tmp_path: Path) -> None:
    """The strongest privacy assertion available offline: scan the whole recording for a secret.

    The credential is only ever placed in a request header by the adapter, and the recording
    stores the request *body* - so this asserts the separation rather than trusting it.
    """
    # A canary, not a credential: the assertions below scan the recording and the serialized
    # response for this exact string, so any unique value proves the same separation.  Keeping it
    # un-key-shaped avoids asking a credential scanner to triage a fixture.
    secret = "example-not-a-key"
    store = _store(tmp_path)
    adapter = _adapter(
        _RecordingTransport(
            TransportResult(200, _body(json.dumps(ANSWER), usage={"prompt_tokens": 1}))
        ),
        api_key=secret,
    )
    client = RecordingModelClient(
        adapter,
        store=store,
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    response = client.complete(_request())
    blob = store.path.read_text(encoding="utf-8")
    assert secret not in blob
    assert "Authorization" not in blob
    assert "Bearer" not in blob
    assert secret not in response.model_dump_json()


# =========================================================================== #
# I. Trajectory metadata: live vs replay vs scripted
# =========================================================================== #


def test_the_policy_reports_a_scripted_double_as_scripted() -> None:
    """A deterministic double is never reported as a live provider run."""
    policy = LLMAgentPolicy(ScriptedModelClient([ANSWER]))
    assert policy.model_metadata == {"execution_mode": "scripted"}
    policy.choose(_policy_context())
    assert policy.last_metadata["execution_mode"] == "scripted"


@pytest.mark.parametrize(
    "mode, expected",
    [(ReplayMode.LIVE, "live"), (ReplayMode.RECORD, "record"), (ReplayMode.REPLAY, "replay")],
)
def test_the_policy_reports_the_recorder_mode(tmp_path: Path, mode: ReplayMode, expected: str) -> None:
    """The execution mode identifies whether a provider was called or a trace replayed."""
    client = RecordingModelClient(
        ScriptedModelClient([ANSWER]),
        store=_store(tmp_path),
        mode=mode,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    metadata = LLMAgentPolicy(client).model_metadata
    assert metadata["execution_mode"] == expected
    assert metadata["model_provider"] == "deepseek"
    assert metadata["model_endpoint_id"] == "deepseek-flash"


def test_policy_metadata_accumulates_usage_over_a_decisions_attempts(tmp_path: Path) -> None:
    """A retried decision reports the usage of both calls, so cost is attributable per step."""
    answers = [
        {"action": "finish", "arguments": {"bogus": 1}},  # shape-invalid -> retry
        {"action": "finish"},
    ]
    client = RecordingModelClient(
        ScriptedModelClient(answers),
        store=_store(tmp_path),
        mode=ReplayMode.RECORD,
        provider="deepseek",
        model="deepseek-flash",
        prompt_version=1,
    )
    policy = LLMAgentPolicy(client, max_attempts=2)
    policy.choose(_policy_context())
    metadata = policy.last_metadata
    assert metadata["model_calls"] == 2
    assert metadata["execution_mode"] == "record"
    # The scripted double reports no tokens, so nothing is invented.
    assert "step_input_tokens" not in metadata


def _policy_context() -> PolicyContext:
    """A minimal policy context with one offered action."""
    return PolicyContext(
        user_request="anything",
        available_actions=(ActionKind.FINISH,),
        has_trusted_history=True,
        candidate_state=CandidateState(),
    )


# =========================================================================== #
# J. End-to-end: record and replay the same trajectory
# =========================================================================== #


class _DeterministicProvider:
    """A provider double that answers from the request, for full-suite record/replay.

    It is *not* a policy: it reads the payload and returns JSON, so the policy above it still has
    to parse, validate and propose.  That keeps the test a test of the recorder plus the policy,
    rather than of the double.
    """

    provider_name = "test-provider"
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        payload = request.context_payload
        refs = [
            entry["parent_asin"]
            for entry in payload.get("candidates", {}).get("candidate_refs", [])
        ]
        observation = payload.get("last_observation") or {}
        if not refs:
            answer: dict[str, Any] = {"action": "recommend_from_history", "k": 4}
        elif observation.get("kind") != "details":
            answer = {"action": "get_details", "arguments": {"parent_asins": [refs[0]]}}
        else:
            answer = {"action": "finish", "rationale": "answered"}
        return ModelResponse(
            text=json.dumps(answer, sort_keys=True),
            model_id=self.model,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            # Reported, so the record -> replay path carries a real duration and the
            # "replayed usage equals recorded usage" assertion tests something.
            latency_ms=5.0,
            finish_reason="stop",
        )


def _suite_pair(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], _DeterministicProvider]:
    """Record the whole case suite, then replay it, and return both reports."""
    from experiments.live_agent_evaluation import run

    recording = tmp_path / "suite.jsonl"
    provider = _DeterministicProvider()
    _, recorded = run(
        mode=ReplayMode.RECORD, recording=recording, output=None, wrapped=provider
    )
    calls_after_record = provider.calls
    _, replayed = run(mode=ReplayMode.REPLAY, recording=recording, output=None)
    # The replay must not have touched the provider.
    assert provider.calls == calls_after_record
    return recorded, replayed, provider


def test_record_and_replay_produce_the_same_trajectory(tmp_path: Path) -> None:
    """The reproducibility requirement, over the whole suite.

    Asserted on the things a reader compares: the executed actions, the terminal outcome, the
    failure dimensions, the constraint verdicts and the attributed components.  Excluded are the
    inherently live-only measurements - a fresh wall-clock duration and the suite's own timing -
    because a replay legitimately takes a different amount of *time* while making the same
    *decisions*.
    """
    recorded, replayed, _provider = _suite_pair(tmp_path)

    assert recorded["total"] == replayed["total"] > 0
    assert recorded["passed"] == replayed["passed"]
    assert recorded["failures_by_component"] == replayed["failures_by_component"]

    for first, second in zip(recorded["cases"], replayed["cases"]):
        assert first["case_id"] == second["case_id"]
        assert first["actions"] == second["actions"], first["case_id"]
        assert first["terminal"] == second["terminal"], first["case_id"]
        assert first["metrics"]["failures"] == second["metrics"]["failures"], first["case_id"]
        assert (
            first["metrics"]["constraint_checked"] == second["metrics"]["constraint_checked"]
        ), first["case_id"]
        assert (
            first["metrics"]["constraint_not_enforced"]
            == second["metrics"]["constraint_not_enforced"]
        ), first["case_id"]
        assert first["attribution"] == second["attribution"], first["case_id"]


def test_record_and_replay_report_their_mode_distinctly(tmp_path: Path) -> None:
    """A replayed run is never reported as a live one, at suite or trajectory level."""
    recorded, replayed, _provider = _suite_pair(tmp_path)
    assert replayed["model"]["execution_mode"] == "replay"
    assert replayed["model"]["replayed_calls"] > 0
    assert replayed["model"]["live_calls"] == 0

    modes_recorded = {case["model"]["execution_mode"] for case in recorded["cases"]}
    modes_replayed = {case["model"]["execution_mode"] for case in replayed["cases"]}
    assert "record" in modes_recorded
    assert modes_replayed == {"replay"}


def test_a_replayed_run_reports_recorded_usage_rather_than_measuring_nothing(tmp_path: Path) -> None:
    """Usage and model latency come from the recording, so efficiency is comparable offline."""
    recorded, replayed, _provider = _suite_pair(tmp_path)
    assert recorded["model"]["usage"]["live_calls"] > 0
    assert replayed["model"]["usage"]["model_calls"] == recorded["model"]["usage"]["model_calls"]
    assert replayed["model"]["usage"]["total_tokens"] == recorded["model"]["usage"]["total_tokens"]
    assert replayed["model"]["usage"]["model_latency_ms_total"] is not None


def test_the_trajectory_reports_model_identity_and_latency(tmp_path: Path) -> None:
    """Every case carries which model ran, how, and what it cost in time."""
    recorded, _replayed, _provider = _suite_pair(tmp_path)
    case = next(entry for entry in recorded["cases"] if entry["model"]["model_calls"] > 0)
    model = case["model"]
    assert model["model_provider"] == "test-provider"
    assert model["model_endpoint_id"] == "test-model"
    assert model["model_calls"] > 0
    assert model["input_tokens"] is not None
    assert model["model_latency_ms"] is not None
    assert model["wall_latency_ms"] is not None
    # Wall latency is the whole turn; model latency is the subset spent in the provider.
    assert model["wall_latency_ms"] >= model["model_latency_ms"]


def test_a_scripted_run_reports_no_model_usage() -> None:
    """The deterministic variants report zero calls and unknown tokens, not fabricated numbers."""
    from experiments._harness import run_case_suite
    from recommendation.evaluation.agent import ABLATION_ADAPTIVE
    from recommendation.evaluation.agent.runner import build_adaptive_policy

    report = run_case_suite(variant=ABLATION_ADAPTIVE, policy_factory=build_adaptive_policy)
    trajectory = report.outcomes[0].trajectory
    assert trajectory.model_calls == 0
    assert trajectory.input_tokens is None
    assert trajectory.total_tokens is None
    assert trajectory.model_latency_ms is None
    assert trajectory.execution_mode is None
    assert trajectory.wall_latency_ms is not None


# =========================================================================== #
# K. Live execution is opt-in
# =========================================================================== #


def test_the_live_evaluation_refuses_without_confirmation(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Two deliberate flags are required, so no configuration can start a paid run alone."""
    from experiments.live_agent_evaluation import main

    code = main(["--mode", "record", "--recording", str(tmp_path / "x.jsonl")])
    assert code == 2
    out = capsys.readouterr().out
    assert "--confirm-live" in out
    assert "Nothing was sent" in out


def test_the_live_evaluation_reports_a_missing_recording_clearly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A replay with nothing to replay is an actionable message, not a traceback."""
    from experiments.live_agent_evaluation import main

    code = main(["--mode", "replay", "--recording", str(tmp_path / "absent.jsonl")])
    assert code == 2
    assert "recording_missing" in capsys.readouterr().out


def test_the_live_smoke_reports_missing_configuration_clearly(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misconfigured live smoke names the missing variable and exits non-zero."""
    from experiments.llm_policy_smoke import run

    monkeypatch.delenv(ENV_BASE_URL, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    code = run(live=False, transport=None, mode="live")
    assert code == 2
    assert "NOT CONFIGURED" in capsys.readouterr().out


def test_the_dry_run_calls_nothing_and_reports_that(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default mode is inert even when a provider *is* configured."""
    from experiments.llm_policy_smoke import SUMMARY, run

    monkeypatch.setenv(ENV_BASE_URL, "https://api.deepseek.com")
    monkeypatch.setenv(ENV_MODEL, "deepseek-flash")
    monkeypatch.setenv(ENV_API_KEY, "example-not-a-key")  # configured, but must never be used
    SUMMARY.clear()
    code = run(live=False, transport=None, mode=None)
    assert code == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert SUMMARY["called_provider"] is False


def test_default_recordings_live_outside_the_repository() -> None:
    """Recordings are evaluation artifacts, not unreviewed repository fixtures."""
    assert not str(DEFAULT_RECORDINGS_DIR).startswith(str(REPO_ROOT))
    assert "tmp" in str(DEFAULT_RECORDINGS_DIR)


def test_no_provider_sdk_is_imported_by_the_control_plane() -> None:
    """The dependency policy holds: the provider seam is standard library only."""
    import recommendation.control.provider_adapter as adapter_module

    source = Path(adapter_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("import openai", "import anthropic", "from openai", "import requests", "import httpx"):
        assert forbidden not in source, forbidden


# =========================================================================== #
# L. Parse protocol 2: the JSON convention measured on the live baseline
# =========================================================================== #


def _policy_context() -> PolicyContext:
    from recommendation.control.context import CandidateState

    return PolicyContext(
        user_request="recommend gear",
        available_actions=(
            ActionKind.RECOMMEND_FROM_HISTORY,
            ActionKind.SEARCH_CATALOG,
            ActionKind.FINISH,
        ),
        has_trusted_history=True,
        candidate_state=CandidateState(),
    )


def test_an_empty_arguments_object_is_accepted_for_an_argumentless_action() -> None:
    """The measured live-baseline cause, pinned.

    DeepSeek sent ``"arguments": {}`` for RECOMMEND_FROM_HISTORY on 20 of 44 baseline calls.  The
    proposal contract rejects ``{}`` because it is not ``None``, so 7 of 14 cases failed with no
    usable action after their retry - a JSON convention, not a policy error.  ``{}`` and "no
    arguments" state the same thing for an action that declares no argument fields.
    """
    policy = LLMAgentPolicy(
        ScriptedModelClient([{"action": "recommend_from_history", "arguments": {}}])
    )
    proposal = policy.choose(_policy_context())
    assert proposal.action is ActionKind.RECOMMEND_FROM_HISTORY
    assert proposal.arguments is None
    assert policy.last_metadata["parse_ok"] is True


def test_a_foreign_payload_is_still_rejected_for_an_argumentless_action() -> None:
    """The fold is narrow: only an *empty* object is equivalent to absent.

    Anything else is the model trying to attach a payload to an action that takes none, which is
    exactly what the action's declared model must keep refusing.
    """
    policy = LLMAgentPolicy(
        ScriptedModelClient(
            [
                {"action": "recommend_from_history", "arguments": {"parent_asins": ["FAKE"]}},
                {"action": "recommend_from_history", "arguments": {"parent_asins": ["FAKE"]}},
            ]
        )
    )
    with pytest.raises(Exception):
        policy.choose(_policy_context())
    assert str(policy.last_metadata.get("last_failure", "")).startswith("invalid_arguments")


def test_an_empty_arguments_object_is_still_rejected_when_arguments_are_required() -> None:
    """An action that declares argument fields cannot be satisfied by ``{}``."""
    policy = LLMAgentPolicy(
        ScriptedModelClient(
            [
                {"action": "search_catalog", "arguments": {}},
                {"action": "search_catalog", "arguments": {}},
            ]
        )
    )
    with pytest.raises(Exception):
        policy.choose(_policy_context())
    assert str(policy.last_metadata.get("last_failure", "")).startswith("invalid_arguments")


def test_the_proposal_contract_itself_remains_strict() -> None:
    """``ActionProposal`` is untouched: the fold lives in the model seam, not in the type.

    ``ActionProposal`` is also the deterministic policies' construction contract, so it must keep
    rejecting a payload it was never meant to accept.  Only the text-model path normalises the
    JSON spelling of "absent".
    """
    from recommendation.control import ActionProposal

    with pytest.raises(Exception):
        ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4, arguments={})


def test_the_parse_protocol_version_is_reported() -> None:
    """The version is a recorded contract, not an implementation detail."""
    from recommendation.control.model_policy import (
        POLICY_PARSE_PROTOCOL_VERSION,
        POLICY_PROMPT_VERSION,
    )

    # The values advance as the contracts are revised; what matters is that both exist, are
    # positive, and are recorded - pinning literal numbers here would only make every future
    # revision edit this test.
    assert POLICY_PROMPT_VERSION >= 1
    assert POLICY_PARSE_PROTOCOL_VERSION >= 1
    from recommendation.control.model_policy import PROMPT_V1, PROMPT_V2

    assert PROMPT_V1 != PROMPT_V2
