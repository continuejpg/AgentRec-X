"""Lightweight operational metrics for the shopping Agent.

This is the runtime counterpart of the evaluation-plane trajectory metrics.  It does not
score recommendation quality and does not replace the per-turn response trace.  It answers
the operational questions a dashboard needs:

* how many Agent turns succeeded, and with which route;
* how often a verified hard constraint was violated or an ungrounded product was presented;
* how many clarification answers actually closed a pending question;
* tool calls and p50/p95 end-to-end latency.

The collector is intentionally process-local and dependency-free.  A production deployment
can bridge the same snapshot to OpenTelemetry/Prometheus; the repository does not add a
telemetry backend or networking side effect for the local demo.
"""

from __future__ import annotations

import math
import statistics
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .agent_schemas import AgentRecommendResponse

__all__ = [
    "MAX_LATENCY_SAMPLES",
    "OperationalMetricsCollector",
    "OperationalSnapshot",
]


MAX_LATENCY_SAMPLES = 10_000


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    # Nearest-rank percentile: deterministic and simple to audit.
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class OperationalSnapshot:
    """Immutable JSON view of the collector at one point in time."""

    schema_version: str
    turns: int
    routes: Mapping[str, int]
    completed: int
    failed: int
    clarification_pending_turns: int
    clarification_answers: int
    clarification_closures: int
    clarification_failures: int
    recommendations: int
    constraint_evaluated_turns: int
    constraint_violations: int
    unverified_presented: int
    tool_calls_total: int
    source_calls: Mapping[str, int]
    fallback_events: int
    tokens_total: int | None
    latency_ms: Mapping[str, float | int | None]
    rates: Mapping[str, float | None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "turns": self.turns,
            "routes": dict(sorted(self.routes.items())),
            "completed": self.completed,
            "failed": self.failed,
            "clarification": {
                "pending_turns": self.clarification_pending_turns,
                "answers": self.clarification_answers,
                "closures": self.clarification_closures,
                "failures": self.clarification_failures,
            },
            "recommendations": self.recommendations,
            "constraint": {
                "evaluated_turns": self.constraint_evaluated_turns,
                "violations": self.constraint_violations,
            },
            "unverified_presented": self.unverified_presented,
            "tools": {
                "calls_total": self.tool_calls_total,
                "source_calls": dict(sorted(self.source_calls.items())),
            },
            "fallback_events": self.fallback_events,
            "tokens_total": self.tokens_total,
            "latency_ms": dict(self.latency_ms),
            "rates": dict(sorted(self.rates.items())),
            "availability": {
                "routing": True,
                "sources": True,
                "tools": True,
                "latency": True,
                "clarification": True,
                "constraints": True,
                # The local Agent response does not expose provider usage/cost; do not
                # fabricate them. These keys are populated only when an adapter supplies them.
                "tokens": self.tokens_total is not None,
                "cost": False,
                "fallback": True,
            },
        }


class OperationalMetricsCollector:
    """Thread-safe counters and bounded latency samples for Agent turns."""

    def __init__(self, *, max_latency_samples: int = MAX_LATENCY_SAMPLES) -> None:
        if max_latency_samples < 1:
            raise ValueError("max_latency_samples must be >= 1")
        self._lock = threading.Lock()
        self._max_latency_samples = int(max_latency_samples)
        self._turns = 0
        self._routes: Counter[str] = Counter()
        self._completed = 0
        self._failed = 0
        self._clarification_pending_turns = 0
        self._clarification_answers = 0
        self._clarification_closures = 0
        self._clarification_failures = 0
        self._recommendations = 0
        self._constraint_evaluated_turns = 0
        self._constraint_violations = 0
        self._unverified_presented = 0
        self._tool_calls_total = 0
        self._source_calls: Counter[str] = Counter()
        self._fallback_events = 0
        self._tokens_total: int | None = None
        self._latency: deque[float] = deque(maxlen=self._max_latency_samples)

    def record(
        self,
        response: AgentRecommendResponse,
        *,
        route: str = "unknown",
        answered_clarification: bool = False,
        fallback_used: bool = False,
        total_tokens: int | None = None,
    ) -> None:
        """Record one Agent response.  This is a pure observation of the response.

        ``answered_clarification`` is supplied by the HTTP route because a response alone
        cannot tell whether the caller was answering a pending question.  It is not inferred
        from prose.
        """
        terminal = response.terminal
        recommendations = tuple(response.recommendations or ())
        recommendation_ids = {item.parent_asin for item in recommendations}
        grounded_ids = {
            item.parent_asin
            for item in (response.grounded.candidates or ())
            if bool(item.grounded)
        }
        projection = (response.eligibility.projection or {}) if response.eligibility else {}
        presented_projection = set(projection.get("presented") or ())
        constrained_turn = bool(response.eligibility and response.eligibility.evaluated)
        violations = 0
        if constrained_turn:
            # The projection is the trusted feasible set.  A presented identity outside it
            # is an actual constraint violation, not an inference from a score.
            violations = len(recommendation_ids - presented_projection)
        unverified = len(recommendation_ids - grounded_ids)
        pending = response.clarification is not None
        observation = response.observation

        with self._lock:
            self._turns += 1
            self._routes[str(route)] += 1
            if terminal.succeeded:
                self._completed += 1
            else:
                self._failed += 1
            if pending:
                self._clarification_pending_turns += 1
            if answered_clarification:
                self._clarification_answers += 1
                if pending:
                    pass  # a second question is still a pending task, not closure
                elif terminal.succeeded:
                    self._clarification_closures += 1
                else:
                    self._clarification_failures += 1
            if constrained_turn:
                self._constraint_evaluated_turns += 1
                self._constraint_violations += violations
            self._recommendations += len(recommendations)
            self._unverified_presented += unverified
            self._tool_calls_total += int(terminal.tool_calls)
            if observation is not None:
                self._source_calls.update(getattr(observation, "source_calls", {}) or {})
            if fallback_used:
                self._fallback_events += 1
            if total_tokens is not None:
                if int(total_tokens) < 0:
                    raise ValueError("total_tokens must be non-negative")
                self._tokens_total = int(total_tokens) + (self._tokens_total or 0)
            self._latency.append(float(response.timing.total_ms))

    def snapshot(self) -> OperationalSnapshot:
        with self._lock:
            latency = tuple(self._latency)
            routes = dict(self._routes)
            source_calls = dict(self._source_calls)
            turns = self._turns
            completed = self._completed
            failed = self._failed
            answers = self._clarification_answers
            closures = self._clarification_closures
            failed_clarifications = self._clarification_failures
            recommendations = self._recommendations
            evaluated_turns = self._constraint_evaluated_turns
            violations = self._constraint_violations
            unverified = self._unverified_presented
            tools = self._tool_calls_total
            fallback_events = self._fallback_events
            tokens = self._tokens_total
        latency_snapshot: dict[str, float | int | None] = {
            "count": len(latency),
            "p50": _percentile(latency, 0.50),
            "p95": _percentile(latency, 0.95),
        }
        if latency:
            latency_snapshot["mean"] = statistics.fmean(latency)
        rates = {
            "task_success": _rate(completed, turns),
            "failure": _rate(failed, turns),
            "constraint_violation": _rate(violations, recommendations)
            if evaluated_turns
            else None,
            "unverified_presentation": _rate(unverified, recommendations)
            if recommendations
            else None,
            "clarification_closure": _rate(closures, answers) if answers else None,
            "tool_calls_per_turn": _rate(tools, turns),
            "fallback_per_turn": _rate(fallback_events, turns),
        }
        return OperationalSnapshot(
            schema_version="agentrecx.agent_operational_metrics.v1",
            turns=turns,
            routes=routes,
            completed=completed,
            failed=failed,
            clarification_pending_turns=self._clarification_pending_turns,
            clarification_answers=answers,
            clarification_closures=closures,
            clarification_failures=failed_clarifications,
            recommendations=recommendations,
            constraint_evaluated_turns=evaluated_turns,
            constraint_violations=violations,
            unverified_presented=unverified,
            tool_calls_total=tools,
            source_calls=source_calls,
            fallback_events=fallback_events,
            tokens_total=tokens,
            latency_ms=latency_snapshot,
            rates=rates,
        )

    def reset(self) -> None:
        """Reset the in-process snapshot.  Intended for tests and local diagnostics only."""
        with self._lock:
            self._turns = 0
            self._routes.clear()
            self._completed = 0
            self._failed = 0
            self._clarification_pending_turns = 0
            self._clarification_answers = 0
            self._clarification_closures = 0
            self._clarification_failures = 0
            self._recommendations = 0
            self._constraint_evaluated_turns = 0
            self._constraint_violations = 0
            self._unverified_presented = 0
            self._tool_calls_total = 0
            self._source_calls.clear()
            self._fallback_events = 0
            self._tokens_total = None
            self._latency.clear()
