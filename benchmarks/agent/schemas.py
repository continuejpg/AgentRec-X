"""Typed benchmark task, trace and evaluation schemas.

The benchmark deliberately keeps its data model independent of the serving schemas.
Expected answers live only in the task/evaluator side; policy adapters receive only
``task.history`` and ``task.turns[*].user_input``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class PolicyId(str, Enum):
    """The four controlled policy arms."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class PolicyStatus(str, Enum):
    """Whether a policy arm could actually be executed."""

    RUNNABLE = "runnable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BenchTurn:
    """One simulated user utterance.  No expected answer is given to the policy."""

    user_input: str


@dataclass(frozen=True)
class ExpectedOutcome:
    """Deterministic post-run assertions, evaluated after all user turns."""

    final_any: tuple[str, ...] = ()
    final_min: int = 1
    required_title_terms_any: tuple[str, ...] = ()
    must_exclude_title_terms: tuple[str, ...] = ()
    must_exclude_ids: tuple[str, ...] = ()
    active_preference_attributes: tuple[str, ...] = ()
    actions_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyTask:
    """Task projection visible to a policy adapter.

    This object intentionally has no expected outcome, expected identity or grader
    metadata.  Only this projection is passed into the four arms.
    """

    task_id: str
    history: tuple[str, ...]
    turns: tuple[BenchTurn, ...]
    preference_discovery: bool = False


@dataclass(frozen=True)
class BenchTask:
    """One real-catalogue shopping task with 3-5 user turns."""

    task_id: str
    split: str
    scenario: str
    history: tuple[str, ...]
    turns: tuple[BenchTurn, ...]
    expected: ExpectedOutcome
    preference_discovery: bool = False

    @property
    def policy_task(self) -> PolicyTask:
        """Sanitized policy input; contains no expected answer."""
        return PolicyTask(
            task_id=self.task_id,
            history=self.history,
            turns=self.turns,
            preference_discovery=self.preference_discovery,
        )

    @property
    def policy_input(self) -> Mapping[str, Any]:
        """The exact task projection a policy adapter may read.

        Expected answers intentionally do not appear here.
        """
        return {
            "task_id": self.task_id,
            "history": self.history,
            "preference_discovery": self.preference_discovery,
            "turns": tuple(turn.user_input for turn in self.turns),
        }


@dataclass
class TurnTrace:
    """Observed behaviour for one user turn."""

    turn_index: int
    user_input: str
    effective_input: str
    response_status: str = ""
    clarification_id: str | None = None
    selected_option_id: str | None = None
    clarification_question: str | None = None
    clarification_options: tuple[str, ...] = ()
    candidate_pool_ids: tuple[str, ...] = ()
    final_ids: tuple[str, ...] = ()
    retrieval_actions: tuple[str, ...] = ()
    memory_events: tuple[Mapping[str, Any], ...] = ()
    tool_calls: tuple[str, ...] = ()
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    llm_proposals: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None


@dataclass
class TaskRun:
    """One policy's run of one task."""

    task_id: str
    policy: PolicyId
    runnable: bool
    status: str
    turns: list[TurnTrace] = field(default_factory=list)
    final_ids: tuple[str, ...] = ()
    active_preference_attributes: tuple[str, ...] = ()
    retrieval_actions: tuple[str, ...] = ()
    unavailable_reason: str | None = None
    error: str | None = None
    wall_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Evaluation:
    """Deterministic evaluation of one TaskRun.  Human labels are separate."""

    task_id: str
    policy: PolicyId
    runnable: bool
    completed: bool
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]
    final_ids: tuple[str, ...]
    human_label: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "policy": self.policy.value,
            "runnable": self.runnable,
            "completed": self.completed,
            "passed": self.passed,
            "checks": self.checks,
            "failures": list(self.failures),
            "final_ids": list(self.final_ids),
            "human_label": self.human_label,
        }
