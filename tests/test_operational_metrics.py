"""Tests for the process-local Agent operational metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.operational_metrics import OperationalMetricsCollector  # noqa: E402
from tests.demo_fixture import PROFILE_A, build_harness  # noqa: E402


def _response(
    *,
    succeeded: bool,
    tool_calls: int = 1,
    latency_ms: float = 10.0,
    recommendation_ids: tuple[str, ...] = ("a",),
    grounded_ids: tuple[str, ...] = ("a",),
    projection_presented: tuple[str, ...] | None = None,
    constraint_evaluated: bool = False,
    pending_clarification: bool = False,
    route: str = "recommend",
    source_calls: dict[str, int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        route=route,
        terminal=SimpleNamespace(succeeded=succeeded, tool_calls=tool_calls),
        recommendations=tuple(
            SimpleNamespace(parent_asin=identity) for identity in recommendation_ids
        ),
        grounded=SimpleNamespace(
            candidates=tuple(
                SimpleNamespace(parent_asin=identity, grounded=True, verification_status="grounded")
                for identity in grounded_ids
            )
        ),
        eligibility=SimpleNamespace(
            evaluated=constraint_evaluated,
            projection=(
                {"presented": list(projection_presented)}
                if projection_presented is not None
                else None
            ),
        ),
        observation=SimpleNamespace(source_calls=source_calls or {"history": 1}),
        clarification=SimpleNamespace(clarification_id="c1") if pending_clarification else None,
        timing=SimpleNamespace(total_ms=latency_ms),
    )


def test_snapshot_without_turns_reports_null_denominators() -> None:
    snapshot = OperationalMetricsCollector().snapshot().as_dict()
    assert snapshot["turns"] == 0
    assert snapshot["rates"]["task_success"] is None
    assert snapshot["rates"]["constraint_violation"] is None
    assert snapshot["latency_ms"]["p95"] is None
    assert snapshot["availability"]["tokens"] is False


def test_task_success_tools_and_latency_quantiles() -> None:
    collector = OperationalMetricsCollector(max_latency_samples=8)
    collector.record(_response(succeeded=True, latency_ms=10.0), route="shopping")
    collector.record(_response(succeeded=True, latency_ms=20.0), route="shopping")
    collector.record(_response(succeeded=False, latency_ms=100.0), route="other")
    snapshot = collector.snapshot().as_dict()
    assert snapshot["turns"] == 3
    assert snapshot["completed"] == 2
    assert snapshot["failed"] == 1
    assert snapshot["rates"]["task_success"] == 2 / 3
    assert snapshot["tools"]["calls_total"] == 3
    assert snapshot["tools"]["source_calls"] == {"history": 3}
    assert snapshot["routes"] == {"other": 1, "shopping": 2}
    assert snapshot["latency_ms"]["count"] == 3
    assert snapshot["latency_ms"]["p50"] == 20.0
    assert snapshot["latency_ms"]["p95"] == 100.0


def test_constraint_violations_and_unverified_candidates_are_separate() -> None:
    collector = OperationalMetricsCollector()
    collector.record(
        _response(
            succeeded=True,
            recommendation_ids=("a", "b"),
            grounded_ids=("a",),
            constraint_evaluated=True,
            projection_presented=(),
        )
    )
    snapshot = collector.snapshot().as_dict()
    assert snapshot["constraint"] == {"evaluated_turns": 1, "violations": 2}
    assert snapshot["rates"]["constraint_violation"] == 1.0
    assert snapshot["unverified_presented"] == 1
    assert snapshot["rates"]["unverified_presentation"] == 0.5


def test_clarification_closure_is_counted_only_after_an_answer() -> None:
    collector = OperationalMetricsCollector()
    collector.record(
        _response(succeeded=False, pending_clarification=True),
        answered_clarification=False,
    )
    first = collector.snapshot().as_dict()
    assert first["clarification"]["pending_turns"] == 1
    assert first["clarification"]["answers"] == 0
    assert first["rates"]["clarification_closure"] is None

    collector.record(
        _response(succeeded=False, pending_clarification=True),
        answered_clarification=True,
    )
    second = collector.snapshot().as_dict()
    assert second["clarification"]["answers"] == 1
    assert second["clarification"]["closures"] == 0
    assert second["rates"]["clarification_closure"] == 0.0

    collector.record(
        _response(succeeded=True, pending_clarification=False),
        answered_clarification=True,
    )
    third = collector.snapshot().as_dict()
    assert third["clarification"]["closures"] == 1
    assert third["rates"]["clarification_closure"] == 0.5


def test_tokens_are_only_reported_when_supplied_and_reset_is_clean() -> None:
    collector = OperationalMetricsCollector()
    collector.record(_response(succeeded=True), route="shopping", total_tokens=42)
    snapshot = collector.snapshot().as_dict()
    assert snapshot["tokens_total"] == 42
    assert snapshot["availability"]["tokens"] is True
    collector.reset()
    assert collector.snapshot().as_dict()["turns"] == 0


def test_metrics_endpoint_is_read_only_and_populated_by_agent_turn(tmp_path: Path) -> None:
    harness = build_harness(tmp_path)
    try:
        session = harness.create_session(PROFILE_A)["session_id"]
        history = list(harness.manager.get(session).trusted_user_history)
        before = harness.client.get("/v1/demo/metrics")
        assert before.status_code == 200
        assert before.json()["turns"] == 0

        response = harness.client.post(
            "/v1/demo/agent/recommend",
            json={"message": "find me a RedWidget", "history": history, "k": 3},
        )
        assert response.status_code == 200, response.text

        after = harness.client.get("/v1/demo/metrics")
        assert after.status_code == 200
        body = after.json()
        assert body["turns"] == 1
        assert body["routes"] == {"agent_recommend": 1}
        assert body["latency_ms"]["count"] == 1
        assert body["latency_ms"]["p50"] is not None
        assert body["rates"]["task_success"] is not None
    finally:
        harness.runtime.close()
