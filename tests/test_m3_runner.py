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
from recommendation.control.model_client import ModelResponse


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

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def complete(self, request: Any) -> ModelResponse:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return ModelResponse(
            text=text, model_id="stub", input_tokens=10, output_tokens=5, latency_ms=1.5
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

    summary = run_preflight(
        out=tmp_path,
        client_factory=lambda: _StubClient(['{"action": "finish"}']),
        calls=2,
    )

    assert summary["evaluator_invoked"] is False
    assert summary["cohort_loaded"] is False
    assert summary["cohort_users"] == 0
    assert summary["recommendation_metrics_computed"] is False


def test_preflight_reports_usage_and_no_recommendation_metrics(tmp_path: Any) -> None:
    summary = run_preflight(
        out=tmp_path,
        client_factory=lambda: _StubClient(['{"action": "finish"}']),
        calls=3,
    )

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

    assert summary["provider_calls"] == 3
    assert summary["input_tokens"] == 30
    assert summary["json_parse_successes"] == 3

    # no recommendation metric of any kind
    blob = str(summary).lower()
    for banned in ("ndcg", "recall@", "hr@", "recall_at"):
        assert banned not in blob, banned


def test_preflight_counts_unparseable_answers(tmp_path: Any) -> None:
    summary = run_preflight(
        out=tmp_path,
        client_factory=lambda: _StubClient(["not json at all"]),
        calls=2,
    )
    assert summary["provider_calls"] == 2
    assert summary["json_parse_successes"] == 0
    assert summary["failure_rate"] == 0.0


def test_preflight_writes_its_sidecar(tmp_path: Any) -> None:
    run_preflight(out=tmp_path, client_factory=lambda: _StubClient(['{"action":"finish"}']), calls=1)
    assert (tmp_path / "preflight.json").is_file()


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
