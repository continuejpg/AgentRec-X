"""M8 — decision-level evaluation for AgentRec-X (offline lab only).

This module adds a small, typed decision-trajectory record and the metrics an experiment
needs to compare interaction policies.  It is deliberately **separate from the ranking
evaluation** in :mod:`recommendation.evaluation.metrics`: that module scores item ranking,
this one scores whether the *decision process* helped a user choose.

Boundary rules
--------------
* A trajectory contains only what was actually displayed/recorded: chosen interaction
  action, displayed candidates and their trusted evidence, the user's answer, preference
  changes, the experimental ranking-refresh result and the final top-K.  It has **no**
  ground-truth target field; labels are supplied separately to the metric functions, so
  the target can never leak into the agent/ranker path.
* Team Accuracy / Conditional Selection Accuracy are computed only from a **real** user
  choice (``UserChoice.source == "user"``) and real target labels.  A scripted-user
  trajectory is a contract validation, never a quality result.
* Trajectories serialise deterministically and replay byte-identically.

The controlled arm runner (A0/A1/A2) lives here too.  It uses the accepted M7 policy
(:mod:`recommendation.demo.grounded_actions`) and the accepted preference ranking; M6.1 is
called only through its public bridge.  Scripted users are explicitly labelled as such.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

__all__ = [
    "ARM_A0",
    "ARM_A1",
    "ARM_A2",
    "CONTROLLED_ARMS",
    "DecisionArmReport",
    "DecisionCandidate",
    "DecisionEvaluationReport",
    "DecisionFeedback",
    "DecisionMetric",
    "DecisionStep",
    "DecisionTaskLabels",
    "DecisionTaskView",
    "DecisionTrajectory",
    "DecisionTrajectoryRecorder",
    "Feedback",
    "MetricResult",
    "PreferenceChange",
    "RankingRefreshRecord",
    "UserChoice",
    "compute_decision_metrics",
    "decision_trajectory_enabled",
    "run_controlled_comparison",
]

ARM_A0 = "A0"
ARM_A1 = "A1"
ARM_A2 = "A2"
CONTROLLED_ARMS: tuple[str, ...] = (ARM_A0, ARM_A1, ARM_A2)

ENV_M8_TRAJECTORY = "AGENTRECX_M8_TRAJECTORY"
TRAJECTORY_SCHEMA_VERSION = 1


def decision_trajectory_enabled(value: str | None = None) -> bool:
    """M8 trajectory recording is an explicit experiment switch, default off."""
    import os

    raw = value if value is not None else os.environ.get(ENV_M8_TRAJECTORY, "")
    return str(raw).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# trajectory records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecisionCandidate:
    """One candidate as displayed, with only the trusted evidence that was shown."""

    candidate_id: str
    rank: int
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def evidence_available(self) -> bool:
        return bool(self.evidence)


@dataclass(frozen=True)
class Feedback:
    """The user's actual answer to one interaction (never a model's guess)."""

    option_id: str
    label: str
    action: str
    attribute: str = ""
    value: Any = None
    accepted: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceChange:
    """One preference change caused by the feedback."""

    attribute: str
    action: str
    value: Any = None
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingRefreshRecord:
    """The M6.1 experimental refresh result for one step (never a model score)."""

    ran: bool
    changed: bool = False
    fallback: str | None = None
    error: str | None = None
    order_before: tuple[str, ...] = ()
    order_after: tuple[str, ...] = ()

    @property
    def change_rate(self) -> float:
        if not self.ran or not self.order_before or not self.order_after:
            return 0.0
        before = {candidate: index for index, candidate in enumerate(self.order_before)}
        shared = [item for item in self.order_after if item in before]
        if not shared:
            return 0.0
        changed = sum(
            1
            for position, candidate in enumerate(self.order_after)
            if before.get(candidate) != position
        )
        return changed / max(len(self.order_after), 1)


@dataclass(frozen=True)
class DecisionStep:
    """One ASK / COMPARE / EXPLAIN / FINALIZE turn."""

    action: str
    displayed_candidates: tuple[DecisionCandidate, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    feedback: Feedback | None = None
    preference_changes: tuple[PreferenceChange, ...] = ()
    refresh: RankingRefreshRecord | None = None
    budget_remaining: int = 0
    explanation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class UserChoice:
    """The user's final selection.  ``source`` is explicit about its provenance."""

    selected_candidate_id: str | None = None
    source: Literal["user", "scripted_contract", "missing"] = "missing"
    no_suitable_item: bool = False

    @property
    def is_real_user_choice(self) -> bool:
        return (
            self.source == "user"
            and self.selected_candidate_id is not None
            and not self.no_suitable_item
        )


@dataclass(frozen=True)
class DecisionTrajectory:
    """A complete, replayable decision trajectory for one task and one arm."""

    schema_version: int
    trajectory_id: str
    task_id: str
    user_key: str
    arm: str
    seed: int
    budget: int
    k: int
    steps: tuple[DecisionStep, ...] = ()
    final_top_k: tuple[str, ...] = ()
    terminal_status: str = "finished"
    choice: UserChoice = field(default_factory=UserChoice)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "user_key": self.user_key,
            "arm": self.arm,
            "seed": self.seed,
            "budget": self.budget,
            "k": self.k,
            "steps": [step.as_dict() for step in self.steps],
            "final_top_k": list(self.final_top_k),
            "terminal_status": self.terminal_status,
            "choice": asdict(self.choice),
        }

    def digest(self) -> str:
        return _digest(self.as_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionTrajectory":
        if int(payload.get("schema_version", 0)) != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("unsupported decision-trajectory schema version")
        steps: list[DecisionStep] = []
        for raw in payload.get("steps") or ():
            feedback_raw = raw.get("feedback")
            refresh_raw = raw.get("refresh")
            steps.append(
                DecisionStep(
                    action=str(raw.get("action") or ""),
                    displayed_candidates=tuple(
                        DecisionCandidate(
                            candidate_id=str(item.get("candidate_id") or ""),
                            rank=int(item.get("rank") or 0),
                            evidence=dict(item.get("evidence") or {}),
                        )
                        for item in raw.get("displayed_candidates") or ()
                    ),
                    evidence=dict(raw.get("evidence") or {}),
                    feedback=(
                        Feedback(
                            option_id=str(feedback_raw.get("option_id") or ""),
                            label=str(feedback_raw.get("label") or ""),
                            action=str(feedback_raw.get("action") or ""),
                            attribute=str(feedback_raw.get("attribute") or ""),
                            value=feedback_raw.get("value"),
                            accepted=bool(feedback_raw.get("accepted", True)),
                        )
                        if isinstance(feedback_raw, Mapping)
                        else None
                    ),
                    preference_changes=tuple(
                        PreferenceChange(
                            attribute=str(item.get("attribute") or ""),
                            action=str(item.get("action") or ""),
                            value=item.get("value"),
                            label=str(item.get("label") or ""),
                        )
                        for item in raw.get("preference_changes") or ()
                    ),
                    refresh=(
                        RankingRefreshRecord(
                            ran=bool(refresh_raw.get("ran", False)),
                            changed=bool(refresh_raw.get("changed", False)),
                            fallback=refresh_raw.get("fallback"),
                            error=refresh_raw.get("error"),
                            order_before=tuple(
                                str(item) for item in refresh_raw.get("order_before") or ()
                            ),
                            order_after=tuple(
                                str(item) for item in refresh_raw.get("order_after") or ()
                            ),
                        )
                        if isinstance(refresh_raw, Mapping)
                        else None
                    ),
                    budget_remaining=int(raw.get("budget_remaining") or 0),
                    explanation=raw.get("explanation"),
                )
            )
        choice_raw = payload.get("choice") or {}
        return cls(
            schema_version=TRAJECTORY_SCHEMA_VERSION,
            trajectory_id=str(payload.get("trajectory_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            user_key=str(payload.get("user_key") or ""),
            arm=str(payload.get("arm") or ""),
            seed=int(payload.get("seed") or 0),
            budget=int(payload.get("budget") or 0),
            k=int(payload.get("k") or 0),
            steps=tuple(steps),
            final_top_k=tuple(str(item) for item in payload.get("final_top_k") or ()),
            terminal_status=str(payload.get("terminal_status") or "finished"),
            choice=UserChoice(
                selected_candidate_id=choice_raw.get("selected_candidate_id"),
                source=str(choice_raw.get("source") or "missing"),
                no_suitable_item=bool(choice_raw.get("no_suitable_item", False)),
            ),
        )


class DecisionTrajectoryRecorder:
    """Mutable recorder with deterministic serialisation and no target field."""

    def __init__(
        self,
        *,
        trajectory_id: str,
        task_id: str,
        user_key: str,
        arm: str,
        seed: int = 0,
        budget: int = 2,
        k: int = 5,
    ) -> None:
        self._trajectory_id = str(trajectory_id)
        self._task_id = str(task_id)
        self._user_key = str(user_key)
        self._arm = str(arm)
        self._seed = int(seed)
        self._budget = max(1, int(budget))
        self._k = max(1, int(k))
        self._steps: list[DecisionStep] = []
        self._final_top_k: tuple[str, ...] = ()
        self._terminal_status = "finished"
        self._choice = UserChoice()

    @classmethod
    def for_service(
        cls, *, task_id: str, user_key: str, arm: str = "A2", seed: int = 0, budget: int = 2, k: int = 5
    ) -> "DecisionTrajectoryRecorder":
        return cls(
            trajectory_id=_digest(
                {"task": str(task_id), "user": str(user_key), "arm": str(arm)}
            )[:24],
            task_id=str(task_id),
            user_key=str(user_key),
            arm=str(arm),
            seed=seed,
            budget=budget,
            k=k,
        )

    def begin_action(
        self,
        *,
        action: str,
        displayed: Sequence[DecisionCandidate],
        evidence: Mapping[str, Any] | None = None,
        budget_remaining: int = 0,
        explanation: str | None = None,
    ) -> None:
        text = str(action).strip().upper()
        if text not in {"ASK", "COMPARE", "EXPLAIN", "FINALIZE"}:
            raise ValueError("unknown decision action")
        self._steps.append(
            DecisionStep(
                action=text,
                displayed_candidates=tuple(displayed),
                evidence=dict(evidence or {}),
                budget_remaining=int(budget_remaining),
                explanation=explanation,
            )
        )

    def record_feedback(
        self,
        *,
        option_id: str,
        label: str,
        action: str,
        attribute: str = "",
        value: Any = None,
        accepted: bool = True,
        preference_changes: Sequence[PreferenceChange] = (),
    ) -> None:
        if not self._steps:
            raise ValueError("feedback requires a preceding action")
        self._steps[-1] = replace(
            self._steps[-1],
            feedback=Feedback(
                option_id=str(option_id),
                label=str(label),
                action=str(action),
                attribute=str(attribute),
                value=value,
                accepted=bool(accepted),
            ),
            preference_changes=tuple(preference_changes),
        )

    def record_refresh(self, refresh: RankingRefreshRecord) -> None:
        if not self._steps:
            raise ValueError("a ranking refresh requires a preceding action")
        self._steps[-1] = replace(self._steps[-1], refresh=refresh)

    def finalize(self, final_top_k: Sequence[str], *, terminal_status: str = "finished") -> None:
        self._final_top_k = tuple(str(item) for item in final_top_k)
        self._terminal_status = str(terminal_status)

    def record_choice(
        self, selected_candidate_id: str, *, source: Literal["user", "scripted_contract"] = "user"
    ) -> None:
        identity = str(selected_candidate_id)
        if self._final_top_k and identity not in self._final_top_k:
            raise ValueError("the selected candidate is not in the recorded final top-K")
        if not self._final_top_k:
            raise ValueError("finalize the trajectory before recording a choice")
        self._choice = UserChoice(selected_candidate_id=identity, source=source)

    def record_no_suitable_choice(
        self, *, source: Literal["user", "scripted_contract"] = "user"
    ) -> None:
        """Record a real "no suitable item" choice; never a correctness label."""
        if not self._final_top_k:
            raise ValueError("finalize the trajectory before recording a choice")
        self._choice = UserChoice(
            selected_candidate_id=None, source=source, no_suitable_item=True
        )

    def trajectory(self) -> DecisionTrajectory:
        return DecisionTrajectory(
            schema_version=TRAJECTORY_SCHEMA_VERSION,
            trajectory_id=self._trajectory_id,
            task_id=self._task_id,
            user_key=self._user_key,
            arm=self._arm,
            seed=self._seed,
            budget=self._budget,
            k=self._k,
            steps=tuple(self._steps),
            final_top_k=self._final_top_k,
            terminal_status=self._terminal_status,
            choice=self._choice,
        )

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def user_key(self) -> str:
        return self._user_key

    @property
    def final_top_k(self) -> tuple[str, ...]:
        return self._final_top_k


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricResult:
    name: str
    status: Literal["MEASURED", "NOT_EXECUTED"]
    value: float | None = None
    sample_size: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionTaskLabels:
    """Ground-truth labels kept OUTSIDE the agent/ranker trajectory."""

    task_id: str
    target_ids: tuple[str, ...]
    label_source: str = "real"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_decision_metrics(
    trajectories: Sequence[DecisionTrajectory],
    labels: Mapping[str, DecisionTaskLabels],
    *,
    k: int = 5,
) -> "DecisionMetricSet":
    """Compute decision-level metrics, separating measurable from not-executed ones.

    Recall@K is measured per task when a target label exists.  Team Accuracy and
    Conditional Selection Accuracy are measured **only** when at least one trajectory
    carries a real user choice (``source == "user"``) and the corresponding target label
    exists; otherwise they are explicitly ``NOT_EXECUTED``.  Interaction cost, evidence
    coverage and ranking-change rate are always observable when steps exist.
    """
    recalled: list[float] = []
    team: list[float] = []
    conditional: list[float] = []
    interaction_counts = {"turns": 0, "ask": 0, "compare": 0, "explain": 0, "finalize": 0}
    evidence_ratios: list[float] = []
    refresh_rates: list[float] = []
    real_choices = 0
    conditional_cases = 0
    for trajectory in trajectories:
        label = labels.get(trajectory.task_id)
        if label is not None and label.target_ids:
            recalled.append(
                1.0
                if any(item in set(label.target_ids) for item in trajectory.final_top_k[:k])
                else 0.0
            )
        for step in trajectory.steps:
            interaction_counts["turns"] += 1
            interaction_counts[step.action.lower()] = (
                interaction_counts.get(step.action.lower(), 0) + 1
            )
            displayed = step.displayed_candidates
            if displayed:
                evidence_ratios.append(
                    sum(1 for candidate in displayed if candidate.evidence_available)
                    / len(displayed)
                )
            if step.refresh is not None and step.refresh.ran:
                refresh_rates.append(step.refresh.change_rate)
        choice = trajectory.choice
        if label is not None and choice.is_real_user_choice:
            real_choices += 1
            correct = 1.0 if choice.selected_candidate_id in set(label.target_ids) else 0.0
            team.append(correct)
            if any(item in set(label.target_ids) for item in trajectory.final_top_k[:k]):
                conditional_cases += 1
                conditional.append(correct)
    return DecisionMetricSet(
        recall_at_k=MetricResult(
            name=f"recall_at_{k}",
            status="MEASURED" if recalled else "NOT_EXECUTED",
            value=_mean(recalled) if recalled else None,
            sample_size=len(recalled),
            reason="target labels available" if recalled else "no target label for any task",
        ),
        team_accuracy=MetricResult(
            name="team_accuracy",
            status="MEASURED" if team else "NOT_EXECUTED",
            value=_mean(team) if team else None,
            sample_size=len(team),
            reason=(
                "real user choice + target labels"
                if team
                else "no real user final choice with a target label; scripted/missing choices are not used"
            ),
        ),
        conditional_selection_accuracy=MetricResult(
            name="conditional_selection_accuracy",
            status="MEASURED" if conditional else "NOT_EXECUTED",
            value=_mean(conditional) if conditional else None,
            sample_size=len(conditional),
            reason=(
                "real choice where target was in final top-K"
                if conditional
                else "no real user choice with the target in final top-K"
            ),
        ),
        interaction_cost=MetricResult(
            name="interaction_cost",
            status="MEASURED" if trajectories else "NOT_EXECUTED",
            value=float(interaction_counts["turns"]),
            sample_size=interaction_counts["turns"],
            reason=_canonical(interaction_counts),
        ),
        evidence_coverage=MetricResult(
            name="evidence_coverage",
            status="MEASURED" if evidence_ratios else "NOT_EXECUTED",
            value=_mean(evidence_ratios) if evidence_ratios else None,
            sample_size=len(evidence_ratios),
            reason="fraction of displayed candidates with trusted evidence",
        ),
        ranking_change_rate=MetricResult(
            name="ranking_change_rate",
            status="MEASURED" if refresh_rates else "NOT_EXECUTED",
            value=_mean(refresh_rates) if refresh_rates else None,
            sample_size=len(refresh_rates),
            reason=(
                "fraction of positions changed by the experimental M6.1 refresh"
                if refresh_rates
                else "no executed M6.1 refresh in these trajectories"
            ),
        ),
        real_choice_count=real_choices,
        conditional_case_count=conditional_cases,
        interaction_breakdown=dict(interaction_counts),
    )


@dataclass(frozen=True)
class DecisionMetricSet:
    recall_at_k: MetricResult
    team_accuracy: MetricResult
    conditional_selection_accuracy: MetricResult
    interaction_cost: MetricResult
    evidence_coverage: MetricResult
    ranking_change_rate: MetricResult
    real_choice_count: int = 0
    conditional_case_count: int = 0
    interaction_breakdown: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k.as_dict(),
            "team_accuracy": self.team_accuracy.as_dict(),
            "conditional_selection_accuracy": self.conditional_selection_accuracy.as_dict(),
            "interaction_cost": self.interaction_cost.as_dict(),
            "evidence_coverage": self.evidence_coverage.as_dict(),
            "ranking_change_rate": self.ranking_change_rate.as_dict(),
            "real_choice_count": self.real_choice_count,
            "conditional_case_count": self.conditional_case_count,
            "interaction_breakdown": dict(self.interaction_breakdown),
        }


# --------------------------------------------------------------------------- #
# controlled offline comparison (A0 / A1 / A2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecisionTaskView:
    """Public task for the agent/ranker: candidates, records, scripted answers.

    There is deliberately no target field.  Ground-truth targets live in a separate
    :class:`DecisionTaskLabels` map supplied only to the metric layer.
    """

    task_id: str
    user_key: str
    candidate_ids: tuple[str, ...]
    records: Mapping[str, Any]
    message: str = "recommend hiking gear"
    scripted_feedback: Mapping[str, str] = field(default_factory=dict)
    budget: int = 2
    seed: int = 0
    k: int = 5


@dataclass(frozen=True)
class DecisionArmReport:
    arm: str
    task_count: int
    metrics: DecisionMetricSet
    trajectories: tuple[DecisionTrajectory, ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_count": self.task_count,
            "metrics": self.metrics.as_dict(),
            "trajectories": [trajectory.as_dict() for trajectory in self.trajectories],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DecisionEvaluationReport:
    schema_version: int
    protocol: str
    seed: int
    k: int
    budget: int
    task_count: int
    label_source: str
    arms: tuple[DecisionArmReport, ...]
    quality_status: str
    notes: tuple[str, ...]
    report_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def with_digest(self) -> "DecisionEvaluationReport":
        payload = self.as_dict()
        payload.pop("report_digest", None)
        return replace(self, report_digest=_digest(payload))


def run_controlled_comparison(
    tasks: Sequence[DecisionTaskView],
    labels: Mapping[str, DecisionTaskLabels],
    *,
    seed: int = 0,
    k: int = 5,
    budget: int = 2,
    label_source: str = "synthetic_contract",
) -> DecisionEvaluationReport:
    """Run fixed A0/A1/A2 arms over the same tasks, candidates and budget.

    A0 never interacts.  A1 uses only the accepted ASK prompt.  A2 uses the full M7.1
    policy (ASK/COMPARE/EXPLAIN/FINALIZE) and the M6.1 public refresh bridge, falling back
    to the accepted task-preference ordering when the refresh does not run.  Scripted
    answers are explicitly labelled; no quality claim follows from them.
    """
    arm_reports: list[DecisionArmReport] = []
    for arm in CONTROLLED_ARMS:
        trajectories = [
            _run_arm(arm, task, k=k, budget=budget, seed=seed) for task in tasks
        ]
        metrics = compute_decision_metrics(trajectories, labels, k=k)
        arm_reports.append(
            DecisionArmReport(
                arm=arm,
                task_count=len(tasks),
                metrics=metrics,
                trajectories=tuple(trajectories),
                notes=(
                    "scripted answers are contract fixtures, not real user feedback",
                )
                if arm != ARM_A0
                else ("no interaction arm (baseline candidate order)",),
            )
        )
    report = DecisionEvaluationReport(
        schema_version=1,
        protocol="agentrecx.m8.decision_eval.v1",
        seed=int(seed),
        k=int(k),
        budget=int(budget),
        task_count=len(tasks),
        label_source=str(label_source),
        arms=tuple(arm_reports),
        quality_status="SYNTHETIC_CONTRACT_ONLY_NOT_QUALITY",
        notes=(
            f"label_source={label_source}; Team Accuracy requires a REAL user choice and "
            "is NOT_EXECUTED for scripted users",
            "same fixed task set, candidate set and interaction budget for every arm",
            "evidence values are copied from trusted metadata records; targets are not "
            "present in any decision view",
        ),
    )
    return report.with_digest()


def _scripted_option(prompt: Any, scripted: Mapping[str, str]) -> Any:
    preferred = str(scripted.get(str(prompt.attribute), "")).strip()
    if preferred:
        for option in prompt.options:
            if preferred in {option.option_id, option.action, option.label}:
                return option
    for option in prompt.options:
        if option.action not in {"none", "skip"}:
            return option
    return None


def _displayed(prompt: Any) -> tuple[DecisionCandidate, ...]:
    displayed: list[DecisionCandidate] = []
    values = dict(prompt.values or {})
    for index, identity in enumerate(prompt.products, start=1):
        evidence = (
            {str(prompt.attribute): values[identity]} if identity in values else {}
        )
        displayed.append(
            DecisionCandidate(candidate_id=str(identity), rank=index, evidence=evidence)
        )
    return tuple(displayed)


def _apply_preference(candidate_ids: Sequence[str], records: Mapping[str, Any], preferences: Sequence[Any]) -> tuple[str, ...]:
    from types import SimpleNamespace

    from recommendation.demo.preference_construction import rank_with_task_preferences

    candidates = [
        SimpleNamespace(parent_asin=str(identity), item_id=index)
        for index, identity in enumerate(candidate_ids, start=1)
    ]
    ranking = rank_with_task_preferences(candidates, records, tuple(preferences))
    if ranking is None:
        return tuple(candidate_ids)
    return ranking.ranked_order


def _run_arm(arm: str, task: DecisionTaskView, *, k: int, budget: int, seed: int) -> DecisionTrajectory:
    from types import SimpleNamespace

    from recommendation.state.state_projector import RankerStateProjection

    recorder = DecisionTrajectoryRecorder(
        trajectory_id=_digest({"task": task.task_id, "user": task.user_key, "arm": arm})[:24],
        task_id=task.task_id,
        user_key=task.user_key,
        arm=arm,
        seed=seed,
        budget=budget,
        k=k,
    )
    candidates = list(task.candidate_ids)
    preferences: list[Any] = []
    if arm in {ARM_A1, ARM_A2}:
        for round_no in range(1, budget + 1):
            if arm == ARM_A1:
                from recommendation.demo.preference_construction import (
                    choose_preference_action,
                    preference_from_option,
                )

                decision = choose_preference_action(
                    candidates,
                    task.records,
                    message=task.message,
                    top_k=k,
                    round_no=round_no,
                )
                action = "ASK"
                prompt = decision.dimension
                if decision.action == "SKIP" or prompt is None:
                    break
                option = _scripted_option(prompt, task.scripted_feedback)
                preference = (
                    preference_from_option(prompt, option) if option is not None else None
                )
                recorder.begin_action(
                    action="ASK",
                    displayed=_displayed(prompt),
                    evidence={
                        "attribute": prompt.attribute,
                        "values": dict(prompt.values),
                    },
                    budget_remaining=budget - round_no + 1,
                )
            else:
                from recommendation.demo.grounded_actions import (
                    GroundedAction,
                    choose_grounded_action,
                    refresh_candidate_order,
                )
                from recommendation.demo.preference_construction import preference_from_option

                decision = choose_grounded_action(
                    candidates,
                    task.records,
                    message=task.message,
                    top_k=k,
                    round_no=round_no,
                    budget_remaining=budget - round_no + 1,
                )
                if decision.action is GroundedAction.FINALIZE or decision.prompt is None:
                    break
                prompt = decision.prompt
                action = decision.action.value
                option = _scripted_option(prompt, task.scripted_feedback)
                preference = (
                    preference_from_option(prompt, option) if option is not None else None
                )
                recorder.begin_action(
                    action=action,
                    displayed=_displayed(prompt),
                    evidence={
                        "attribute": prompt.attribute,
                        "values": dict(prompt.values),
                    },
                    budget_remaining=budget - round_no + 1,
                    explanation=decision.explanation,
                )
                before = tuple(candidates)
                if preference is not None:
                    preferences.append(preference)
                    refreshed = refresh_candidate_order(
                        recommendations=[
                            SimpleNamespace(
                                parent_asin=str(identity),
                                item_id=index,
                                score=0.0,
                                score_kind="fixed_pool_order",
                            )
                            for index, identity in enumerate(before, start=1)
                        ],
                        records=task.records,
                        base_projection=RankerStateProjection(
                            task_id=task.task_id, state_version=0, features=()
                        ),
                        preferences=tuple(preferences),
                        user_key=task.user_key,
                        task_id=task.task_id,
                        turn_key=f"{task.task_id}:turn-{round_no}",
                    )
                    if refreshed.ran and refreshed.order:
                        candidates = list(refreshed.order)
                        after = tuple(candidates)
                    else:
                        after = _apply_preference(candidates, task.records, preferences)
                        candidates = list(after)
                    recorder.record_refresh(
                        RankingRefreshRecord(
                            ran=refreshed.ran,
                            changed=refreshed.changed,
                            fallback=None if refreshed.ran else "accepted_ordering",
                            error=refreshed.error,
                            order_before=before,
                            order_after=after,
                        )
                    )
                if preference is None:
                    break
                if option is not None:
                    recorder.record_feedback(
                        option_id=option.option_id,
                        label=option.label,
                        action=option.action,
                        attribute=prompt.attribute,
                        value=option.value,
                        accepted=True,
                        preference_changes=(
                            PreferenceChange(
                                attribute=preference.attribute,
                                action=preference.action,
                                value=preference.value,
                                label=preference.label,
                            ),
                        ),
                    )
                continue
            if preference is not None:
                preferences.append(preference)
                candidates = list(_apply_preference(candidates, task.records, preferences))
            if option is not None:
                recorder.record_feedback(
                    option_id=option.option_id,
                    label=option.label,
                    action=option.action,
                    attribute=prompt.attribute,
                    value=option.value,
                    accepted=True,
                    preference_changes=(
                        (
                            PreferenceChange(
                                attribute=preference.attribute,
                                action=preference.action,
                                value=preference.value,
                                label=preference.label,
                            ),
                        )
                        if preference is not None
                        else ()
                    ),
                )
            if preference is None:
                break
    recorder.finalize(tuple(candidates[:k]))
    return recorder.trajectory()
