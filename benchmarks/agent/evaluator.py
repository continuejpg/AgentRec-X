"""Deterministic evaluation of benchmark trajectories.

The evaluator uses only trusted catalogue facts and recorded run state.  Human
relevance labels are loaded from a separate annotation file and never change the
deterministic checks; they are reported separately.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import BenchTask, Evaluation, PolicyId, TaskRun


def _title(record: Any) -> str:
    return " ".join(str(getattr(record, "title", "") or "").lower().split())


def load_human_labels(path: Path | str | None) -> dict[str, dict[str, Any]]:
    """Load independent human annotations; never synthesize a missing label."""
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    if source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                task_id = str(row.get("task_id", "")).strip()
                if task_id:
                    labels[task_id] = {key: row.get(key) for key in row}
        return labels
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        task_id = str(payload.get("task_id", "")).strip()
        if task_id:
            labels[task_id] = dict(payload)
    return labels


def evaluate_run(
    run: TaskRun,
    task: BenchTask,
    records: Mapping[str, Any],
    *,
    human_label: str | None = None,
) -> Evaluation:
    """Evaluate one run with deterministic product/state/action assertions."""
    checks: dict[str, bool] = {}
    failures: list[str] = []
    if not run.runnable:
        return Evaluation(
            task_id=task.task_id,
            policy=run.policy,
            runnable=False,
            completed=False,
            passed=False,
            checks={"runnable": False},
            failures=(f"policy unavailable: {run.unavailable_reason or 'unknown'}",),
            final_ids=run.final_ids,
            human_label=human_label,
        )
    checks["runnable"] = True
    last_status = run.turns[-1].response_status if run.turns else ""
    complete = last_status in {"finished", "completed"} and not (run.turns and run.turns[-1].error)
    checks["completed"] = complete
    if not complete:
        failures.append(f"conversation did not finish (last status={last_status!r})")
    checks["no_turn_errors"] = not any(turn.error for turn in run.turns)
    if not checks["no_turn_errors"]:
        failures.append("one or more turns raised an error")
    final_ids = tuple(run.final_ids)
    checks["has_final_products"] = len(final_ids) >= task.expected.final_min
    if not checks["has_final_products"]:
        failures.append(
            f"final products {final_ids!r} do not meet minimum {task.expected.final_min}"
        )
    expected = set(task.expected.final_any)
    checks["expected_product_identity"] = bool(expected.intersection(final_ids))
    if task.expected.final_any and not checks["expected_product_identity"]:
        failures.append("none of the expected product identities were presented")
    titles = [_title(records[identity]) for identity in final_ids if identity in records]
    checks["trusted_products"] = len(titles) == len(final_ids)
    if not checks["trusted_products"]:
        failures.append("a final identity is not a trusted catalogue product")
    required_terms = task.expected.required_title_terms_any
    checks["title_role"] = not required_terms or any(
        term in title for term in required_terms for title in titles
    )
    if required_terms and not checks["title_role"]:
        failures.append("no final product has the required trusted title evidence")
    checks["no_forbidden_ids"] = not set(task.expected.must_exclude_ids).intersection(final_ids)
    if not checks["no_forbidden_ids"]:
        failures.append("a forbidden product identity was presented")
    forbidden_terms = task.expected.must_exclude_title_terms
    checks["no_forbidden_roles"] = not any(
        term in title for term in forbidden_terms for title in titles
    )
    if forbidden_terms and not checks["no_forbidden_roles"]:
        failures.append("a forbidden product role appeared in the final list")

    if task.expected.active_preference_attributes:
        actual = set(run.active_preference_attributes)
        checks["active_memory"] = set(task.expected.active_preference_attributes).issubset(actual)
        if not checks["active_memory"]:
            failures.append(
                "expected active preferences missing: "
                f"{sorted(set(task.expected.active_preference_attributes) - actual)}"
            )
    else:
        # A/B have no stateful memory. For C/D an empty expectation only means the task
        # does not assert a particular attribute; it is not a claim that memory is empty.
        checks["active_memory"] = True

    if task.expected.actions_any and run.policy in {PolicyId.C, PolicyId.D}:
        actual_actions = set(run.retrieval_actions)
        checks["retrieval_action"] = bool(actual_actions.intersection(task.expected.actions_any))
        if not checks["retrieval_action"]:
            failures.append(
                f"no expected retrieval action; observed {sorted(actual_actions)}"
            )
    else:
        checks["retrieval_action"] = True

    passed = all(checks.values())
    return Evaluation(
        task_id=task.task_id,
        policy=run.policy,
        runnable=True,
        completed=complete,
        passed=passed,
        checks=checks,
        failures=tuple(failures),
        final_ids=final_ids,
        human_label=human_label,
    )


def evaluate_runs(
    runs: Iterable[TaskRun],
    tasks: Sequence[BenchTask],
    records: Mapping[str, Any],
    *,
    human_labels: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Evaluation]:
    by_id = {task.task_id: task for task in tasks}
    labels = human_labels or {}
    result: list[Evaluation] = []
    for run in runs:
        task = by_id.get(run.task_id)
        if task is None:
            continue
        label = labels.get(run.task_id, {})
        human = label.get("human_relevance") or label.get("human_relevance_0_1_2")
        result.append(
            evaluate_run(
                run,
                task,
                records,
                human_label=str(human) if human not in (None, "") else None,
            )
        )
    return result


def task_success_summary(evaluations: Sequence[Evaluation]) -> dict[str, dict[str, Any]]:
    """Per-policy runnable/completed/passed counts and deterministic failures."""
    summary: dict[str, dict[str, Any]] = {}
    for policy in PolicyId:
        rows = [item for item in evaluations if item.policy is policy]
        if not rows:
            continue
        summary[policy.value] = {
            "runnable": sum(1 for row in rows if row.runnable),
            "completed": sum(1 for row in rows if row.completed),
            "passed": sum(1 for row in rows if row.passed),
            "tasks": len(rows),
            "human_labeled": sum(1 for row in rows if row.human_label not in (None, "")),
            "failures": [failure for row in rows for failure in row.failures],
        }
    return summary
