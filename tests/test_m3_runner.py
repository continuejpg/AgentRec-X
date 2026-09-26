"""Offline tests for the M3 execution runner.

All tests are fast and offline. ``ScriptedModelClient`` (or a trivial stub client) is the only
model, no provider is contacted, no credential is read, and no recommendation metric is asserted —
the pairing statistics are exercised on synthetic ranks only, to test arithmetic, never to produce
an M3 result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from experiments.m3_agent_benchmark import (
    EXPECTED,
    FROZEN_PROVIDER,
    build_m3_evaluation_client,
    build_m3_provider_client,
    m3_provider_settings,
    provider_config_checks,
    FrozenInputs,
    HardCheckFailure,
    agent_arm_ranks,
    artifact_checks,
    cohort_checks,
    compute_statistics,
    guard,
    main,
    run_preflight,
)
from recommendation.control.model_client import ModelRequest, ModelResponse


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class _Case:
    """The minimum an EvaluationCase must expose for the cohort checks."""

    def __init__(self, target: int) -> None:
        self.test_target = target


def _frozen(*, users: int, aligned: bool = True, with_cohort: bool = True) -> FrozenInputs:
    """A synthetic frozen-input bundle: small, aligned, and cheap to build."""
    heads = [[10 * i + 1, 10 * i + 2, 10 * i + 3] for i in range(users)]
    top10 = np.asarray([(row + [-1] * 10)[:10] for row in heads], dtype=np.int64)
    if not aligned:
        top10 = top10.copy()
        top10[0, 0] = 999_999
    targets = np.arange(1, users + 1, dtype=np.int64)
    return FrozenInputs(
        heads=heads,
        head_lengths=[len(h) for h in heads],
        comparator_ranks=np.full(users, 50, dtype=np.int64),
        comparator_targets=targets,
        popularity_targets=targets.copy(),
        sasrec_top10=top10,
        cohort=[_Case(int(t)) for t in targets] if with_cohort else [],
        catalogue_records=EXPECTED["catalogue"],
    )


class _StubClient:
    """A minimal structured-model client: answers from a list, never touches a network."""

    def __init__(self, texts: list[str], *, finish_reason: str | None = "stop",
                 raises: Exception | None = None) -> None:
        self._texts = list(texts)
        self._finish_reason = finish_reason
        self._raises = raises
        self.calls = 0

    def complete(self, request: Any) -> ModelResponse:
        if self._raises is not None:
            self.calls += 1
            raise self._raises
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return ModelResponse(
            text=text, model_id="stub", input_tokens=10, output_tokens=5, latency_ms=1.5,
            finish_reason=self._finish_reason,
        )


def _probe_harness(client: Any) -> tuple[Any, Any]:
    """A fixture-backed harness factory, so preflight tests stay fast and cohort-free.

    The policy and the control plane are the real ones; only the catalogue is synthetic.
    """
    from recommendation.control.model_policy import LLMAgentPolicy
    from tests.agent_reranking_fixture import CANDIDATE_ROWS
    from tests.test_m3_agent_arm import _build_harness

    def build(case: Any) -> Any:
        return _build_harness(LLMAgentPolicy(client, max_attempts=2))

    return build, {row[0]: row[1] for row in CANDIDATE_ROWS}


def _preflight(tmp_path: Any, client: Any, *, calls: int = 2) -> dict[str, Any]:
    harness_factory, item2id = _probe_harness(client)
    return run_preflight(
        out=tmp_path,
        client_factory=lambda: client,
        calls=calls,
        harness_factory=harness_factory,
        item2id=item2id,
    )


# --------------------------------------------------------------------------- #
# CLI / wiring
# --------------------------------------------------------------------------- #


def test_cli_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["--preflight", "--execute"])
    assert exc.value.code == 2


def test_cli_stops_on_a_hard_check_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from experiments import m3_agent_benchmark as runner

    monkeypatch.setattr(runner, "artifact_checks", lambda frozen: ["synthetic failure"])
    rc = main(["--preflight", "--calls", "1"])

    assert rc == 3, "a hard-check failure must exit non-zero"
    assert "STOP" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# preflight isolation
# --------------------------------------------------------------------------- #


def test_preflight_never_reaches_the_evaluator(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The isolation guarantee: preflight must not be able to compute a metric."""
    import recommendation.evaluation.batched as batched

    def _explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("preflight reached evaluate_batched")

    monkeypatch.setattr(batched, "evaluate_batched", _explode)

    summary = _preflight(tmp_path, _StubClient(['{"action": "finish"}']), calls=2)

    assert summary["evaluator_invoked"] is False
    assert summary["cohort_loaded"] is False
    assert summary["cohort_users"] == 0
    assert summary["recommendation_metrics_computed"] is False


def test_preflight_reports_usage_and_no_recommendation_metrics(tmp_path: Any) -> None:
    summary = _preflight(tmp_path, _StubClient(['{"action": "finish"}']), calls=3)

    for key in (
        "provider_calls",
        "provider_failures",
        "failure_rate",
        "input_tokens",
        "output_tokens",
        "latency_ms_mean",
        "json_parse_successes",
    ):
        assert key in summary, key

    assert summary["provider_calls"] >= 3
    assert summary["input_tokens"] >= 30
    assert summary["json_parse_successes"] == 3
    assert summary["provider_connected"] is True

    # no recommendation metric of any kind
    blob = str(summary).lower()
    for banned in ("ndcg", "recall@", "hr@", "recall_at"):
        assert banned not in blob, banned


def test_preflight_counts_unparseable_answers(tmp_path: Any) -> None:
    summary = _preflight(tmp_path, _StubClient(["not json at all"]), calls=2)
    assert summary["provider_calls"] >= 2
    assert summary["json_parse_successes"] == 0
    assert summary["failure_rate"] == 0.0


def test_preflight_writes_its_sidecar(tmp_path: Any) -> None:
    _preflight(tmp_path, _StubClient(['{"action":"finish"}']), calls=1)
    assert (tmp_path / "preflight.json").is_file()


# --------------------------------------------------------------------------- #
# preflight drives the real policy and control plane
# --------------------------------------------------------------------------- #


def test_preflight_drives_the_real_policy_and_control_plane(tmp_path: Any) -> None:
    client = _StubClient(['{"action": "recommend_from_history", "k": 3}', '{"action": "finish"}'])
    summary = _preflight(tmp_path, client)

    control = summary["control_plane"]
    assert control["evaluator_invoked"] is False
    assert control["recommendation_metrics_computed"] is False
    assert control["action_sequence"], "the real policy proposed actions"
    assert control["termination_status"] is not None
    assert control["synthetic_case"]["cohort_member"] is False
    assert control["synthetic_case"]["target_scored"] is False
    assert client.calls > 0, "the policy actually asked the provider for decisions"


def test_preflight_reports_schema_finish_reason_and_tokens(tmp_path: Any) -> None:
    client = _StubClient(['{"action": "finish"}'], finish_reason="stop")
    summary = _preflight(tmp_path, client, calls=1)

    assert summary["schema_actions_validated"] is True
    assert summary["action_schema_offered"], "the offered actions are reported"
    assert summary["finish_reasons"] == {"stop": summary["finish_reasons"].get("stop")}
    assert summary["finish_reasons"]["stop"] >= 1
    assert summary["total_tokens"] >= 15
    assert summary["latency_ms_total"] > 0


def test_preflight_reports_retries_on_unparseable_answers(tmp_path: Any) -> None:
    summary = _preflight(tmp_path, _StubClient(["definitely not json"]), calls=1)

    control = summary["control_plane"]
    # the policy retried and the control plane still terminated — reported, not crashed
    assert control["termination_status"] is not None
    assert "control_plane_error" not in control or control["control_plane_error"]
    assert summary["json_parse_successes"] == 0


def test_preflight_reports_provider_failure_codes(tmp_path: Any) -> None:
    from recommendation.control.model_client import ModelCallError

    client = _StubClient(['{"action": "finish"}'],
                         raises=ModelCallError("boom", code="timeout"))
    summary = _preflight(tmp_path, client, calls=2)

    assert summary["provider_failures"] >= 1
    assert summary["failure_codes"].get("timeout", 0) >= 1
    assert summary["provider_connected"] is False
    assert summary["evaluator_invoked"] is False


def test_preflight_non_cohort_isolation(tmp_path: Any) -> None:
    summary = _preflight(tmp_path, _StubClient(['{"action": "finish"}']), calls=1)

    assert summary["cohort_loaded"] is False
    assert summary["cohort_users"] == 0
    assert summary["control_plane"]["synthetic_case"]["cohort_member"] is False
    blob = str(summary).lower()
    for banned in ("ndcg", "recall@", "hr@", "precision", "map@"):
        assert banned not in blob, banned


# --------------------------------------------------------------------------- #
# hard checks
# --------------------------------------------------------------------------- #


def test_artifact_checks_pass_on_the_real_frozen_artifacts() -> None:
    assert artifact_checks(_frozen(users=4)) == []


def test_cohort_checks_pass_on_aligned_inputs() -> None:
    synthetic = _frozen(users=EXPECTED["cohort"])
    assert cohort_checks(synthetic) == []


def test_cohort_checks_reject_a_wrong_cohort_size() -> None:
    problems = cohort_checks(_frozen(users=3))
    assert problems, "a 3-user cohort must not pass"
    assert any("expected 20000" in p for p in problems), problems


def test_alignment_guard_detects_head_disagreement() -> None:
    """The decisive guard: heads must match the frozen sequential evidence per user."""
    aligned = _frozen(users=EXPECTED["cohort"], aligned=True)
    assert cohort_checks(aligned) == []

    drifted = _frozen(users=EXPECTED["cohort"], aligned=False)
    problems = cohort_checks(drifted)
    assert problems and any("not aligned" in p for p in problems), problems


def test_guard_fails_fast() -> None:
    guard([])  # no problems, no raise
    with pytest.raises(HardCheckFailure):
        guard(["boom"])


# --------------------------------------------------------------------------- #
# execute plumbing
# --------------------------------------------------------------------------- #


def test_failure_fast_stops_before_the_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad cohort must stop the run before any model call and before the evaluator."""
    import recommendation.evaluation.batched as batched

    from experiments import m3_agent_benchmark as runner

    def _explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("execute reached the evaluator despite a failed hard check")

    monkeypatch.setattr(batched, "evaluate_batched", _explode)
    called = {"provider": 0}

    def _client_factory() -> Any:
        called["provider"] += 1
        return _StubClient(['{"action":"finish"}'])

    with pytest.raises(HardCheckFailure):
        runner.run_execute(
            out=runner.DEFAULT_OUT,
            client_factory=_client_factory,
            frozen=_frozen(users=3, with_cohort=False),
        )
    assert called["provider"] == 0, "no provider client may be built before the checks pass"


def test_execute_plumbing_reaches_the_evaluator() -> None:
    """The arm's batches do reach evaluate_batched — the step `--execute` depends on."""
    from tests.test_m3_agent_arm import _case, _runner

    cases = [_case(), _case()]
    ranks, result = agent_arm_ranks(cases=cases, runner=_runner(), num_items=104, records=[])
    assert len(ranks) == len(cases)
    assert result.num_cases == len(cases)
    # plumbing only: no metric value is read or asserted


def test_execute_plumbing_populates_the_ranking_sink() -> None:
    from tests.test_m3_agent_arm import _case, _runner

    rankings: list[list[int]] = []
    agent_arm_ranks(
        cases=[_case(), _case()], runner=_runner(), num_items=104, records=[], rankings=rankings
    )
    assert len(rankings) == 2, "one final ranking per user, in cohort order"
    assert all(isinstance(r, list) for r in rankings)
    assert any(r for r in rankings), "the run proposed candidates"


def test_execute_plumbing_populates_the_behavior_log() -> None:
    from tests.test_m3_agent_arm import _case, _runner

    records: list[dict[str, Any]] = []
    agent_arm_ranks(cases=[_case()], runner=_runner(), num_items=104, records=records)
    assert len(records) == 1
    assert "termination_reason" in records[0]
    assert "action_sequence" in records[0]


# --------------------------------------------------------------------------- #
# statistics arithmetic (synthetic only)
# --------------------------------------------------------------------------- #


def test_compute_statistics_arithmetic_on_synthetic_ranks() -> None:
    agent = np.asarray([1, 1, 500, 500], dtype=np.int64)      # users 0,1 hit at K=5
    comparator = np.asarray([1, 500, 1, 500], dtype=np.int64)  # users 0,2 hit at K=5
    stats = compute_statistics(agent, comparator)

    at5 = stats["per_k"]["K=5"]
    assert stats["users"] == 4
    assert at5["agent_hits"] == 2
    assert at5["comparator_hits"] == 2
    assert at5["agent_hit_comparator_miss"] == 1
    assert at5["agent_miss_comparator_hit"] == 1
    assert 0.0 <= at5["mcnemar_p"] <= 1.0
    assert at5["ci95_low"] <= at5["ci95_high"]
    assert set(stats["per_k"]) == {"K=5", "K=10", "K=20"}


def test_compute_statistics_handles_disjoint_and_identical() -> None:
    identical = np.asarray([1, 2, 3], dtype=np.int64)
    stats = compute_statistics(identical, identical)
    at5 = stats["per_k"]["K=5"]
    assert at5["agent_hit_comparator_miss"] == 0
    assert at5["agent_miss_comparator_hit"] == 0
    assert at5["mcnemar_p"] == 1.0


# --------------------------------------------------------------------------- #
# provider configuration compliance with the frozen DeepSeek amendment
# --------------------------------------------------------------------------- #

#: A placeholder, never a real credential. Tests never read the environment's real key.
_TEST_KEY = "TEST-PLACEHOLDER-NOT-A-REAL-KEY"


class _FakeTransport:
    """Captures the outgoing request payload; performs no network call."""

    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: dict[str, str] | None = None
        self.url: str | None = None

    def __call__(self, url: str, *, headers: Any, payload: Any, timeout: Any) -> Any:
        self.url = url
        self.headers = dict(headers)
        self.payload = dict(payload)
        import json as _json

        body = _json.dumps(
            {
                "choices": [
                    {"message": {"content": '{"action": "finish"}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "model": "deepseek-flash",
                "id": "req-test",
            }
        )
        return 200, body


def _set_provider_env(monkeypatch: pytest.MonkeyPatch, *, profile: str = "openai_compatible") -> None:
    """Set the endpoint, a placeholder key, and a DELIBERATELY WRONG profile."""
    monkeypatch.setenv("AGENTRECX_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("AGENTRECX_LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("AGENTRECX_LLM_API_KEY", _TEST_KEY)
    monkeypatch.setenv("AGENTRECX_LLM_PROFILE", profile)


def test_m3_settings_force_the_deepseek_profile_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment says openai_compatible; the runner must not obey it."""
    _set_provider_env(monkeypatch, profile="openai_compatible")
    settings = m3_provider_settings()

    assert settings.profile.name == FROZEN_PROVIDER["profile"] == "deepseek"
    assert settings.thinking is False
    assert settings.json_mode is True


def test_actual_request_payload_matches_the_frozen_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strongest check: inspect the payload the adapter would really send."""
    _set_provider_env(monkeypatch)
    transport = _FakeTransport()
    client = build_m3_provider_client(transport=transport)

    from recommendation.control.model_client import ModelRequest

    client.complete(ModelRequest(system_prompt="s", context_payload={}, action_schema=()))

    payload = transport.payload
    assert payload is not None
    assert transport.url is not None and transport.url.endswith("/chat/completions")
    assert payload["model"] == FROZEN_PROVIDER["model"]
    assert payload["thinking"] == {"type": "disabled"}, "thinking must be explicitly disabled"
    assert payload["temperature"] == FROZEN_PROVIDER["temperature"] == 0.0
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == FROZEN_PROVIDER["max_tokens"] == 512
    assert payload["stream"] is False
    assert "tools" not in payload and "tool_choice" not in payload
    # the placeholder key travels in the header, never in the body
    assert transport.headers is not None
    assert _TEST_KEY not in str(payload)


def test_provider_config_checks_pass_on_the_frozen_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_provider_env(monkeypatch)
    assert provider_config_checks() == []


def test_provider_config_checks_detect_profile_drift() -> None:
    """A profile that does not disable thinking must be rejected."""
    from recommendation.control.provider_adapter import PROVIDER_PROFILES

    drifted = m3_provider_settings()
    import dataclasses

    drifted = dataclasses.replace(
        drifted, profile=PROVIDER_PROFILES["openai_compatible"], thinking=False
    )
    problems = provider_config_checks(drifted)
    assert problems, "the openai_compatible profile does not send thinking=disabled"
    assert any("thinking" in p or "profile" in p for p in problems)


def test_provider_config_checks_detect_thinking_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    _set_provider_env(monkeypatch)
    drifted = dataclasses.replace(m3_provider_settings(), thinking=True)
    problems = provider_config_checks(drifted)
    assert any("thinking" in p for p in problems), problems


def test_provider_config_checks_detect_json_mode_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    _set_provider_env(monkeypatch)
    drifted = dataclasses.replace(m3_provider_settings(), json_mode=False)
    problems = provider_config_checks(drifted)
    assert any("json_mode" in p or "response_format" in p for p in problems), problems


def test_preflight_stops_when_the_provider_config_drifts(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import m3_agent_benchmark as runner

    monkeypatch.setattr(runner, "provider_config_checks", lambda settings=None: ["drifted"])
    with pytest.raises(HardCheckFailure):
        runner.run_preflight(
            out=tmp_path, client_factory=lambda: _StubClient(['{"action":"finish"}']), calls=1
        )


def test_execute_stops_when_the_provider_config_drifts(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import m3_agent_benchmark as runner

    monkeypatch.setattr(runner, "provider_config_checks", lambda settings=None: ["drifted"])
    with pytest.raises(HardCheckFailure):
        runner.run_execute(
            out=runner.DEFAULT_OUT,
            client_factory=lambda: _StubClient(['{"action":"finish"}']),
            frozen=_frozen(users=3, with_cohort=False),
        )


def test_unconfigured_provider_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No credential must be a STOP, not a traceback."""
    for name in (
        "AGENTRECX_LLM_BASE_URL",
        "AGENTRECX_LLM_MODEL",
        "AGENTRECX_LLM_API_KEY",
        "AGENTRECX_LLM_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)

    rc = main(["--preflight", "--calls", "1"])
    assert rc == 3
    assert "STOP" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the outgoing message path carries the frozen JSON-format scaffolding
# --------------------------------------------------------------------------- #


def _system_message(payload: dict[str, Any]) -> str:
    messages = payload["messages"]
    assert isinstance(messages, list) and messages
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


def test_m3_evaluation_client_sends_the_json_instruction_and_shape_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The literal `json` token and the shape example must be in the outgoing request."""
    from experiments.m3_agent_arm import M3_JSON_FORMAT_SUFFIX, JsonFormatClient

    _set_provider_env(monkeypatch)
    transport = _FakeTransport()
    client = build_m3_evaluation_client(
        client_factory=lambda: build_m3_provider_client(transport=transport)
    )
    assert isinstance(client.inner, JsonFormatClient), "the formatter sits above the provider"

    client.complete(ModelRequest(system_prompt="BASE PROMPT", context_payload={}, action_schema=()))

    payload = transport.payload
    assert payload is not None
    system = _system_message(payload)
    assert system.startswith("BASE PROMPT"), "the policy prompt itself is preserved, not replaced"
    assert "json" in system, "the literal lowercase word is required by JSON Output mode"
    assert '{"action"' in system, "the shape example is required"
    assert M3_JSON_FORMAT_SUFFIX in system

    # and the rest of the frozen config still holds in the same request
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 512
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.0


def test_preflight_message_path_carries_the_scaffolding(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through preflight: the payload that would leave the process is formatted."""
    _set_provider_env(monkeypatch)
    transport = _FakeTransport()
    # ONE evaluation client drives both phases, exactly as the runner does by default, so the
    # control-plane requests go through the same formatter the provider adapter sits behind.
    evaluation_client = build_m3_evaluation_client(
        client_factory=lambda: build_m3_provider_client(transport=transport)
    )
    harness_factory, item2id = _probe_harness(evaluation_client)
    summary = run_preflight(
        out=tmp_path,
        client_factory=lambda: evaluation_client,
        calls=1,
        harness_factory=harness_factory,
        item2id=item2id,
    )

    assert summary["json_format_scaffolding"] is True
    assert summary["evaluator_invoked"] is False
    payload = transport.payload
    assert payload is not None, "the fake transport captured a real outgoing request"
    system = _system_message(payload)
    assert "json" in system
    assert '{"action"' in system


def test_formatting_wrapper_does_not_change_the_action_space() -> None:
    """Scaffolding is presentational: the offered actions are untouched."""
    from experiments.m3_agent_arm import JsonFormatClient
    from recommendation.control.model_client import ModelRequest as _Req

    inner = _StubClient(['{"action": "finish"}'])
    schema = ({"action": "finish"}, {"action": "recommend_from_history"})
    request = _Req(system_prompt="P", context_payload={"k": 1}, action_schema=schema)

    JsonFormatClient(inner).complete(request)

    sent = inner.last_request if hasattr(inner, "last_request") else None
    assert sent is None or sent.action_schema == request.action_schema
    assert request.system_prompt == "P", "the frozen request object is not mutated"
    assert request.action_schema == schema
