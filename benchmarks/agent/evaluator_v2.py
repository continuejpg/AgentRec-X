"""Versioned, independent evaluator V2 for the shopping benchmark.

This module is deliberately separate from the legacy evaluator and from the runtime
``product_role`` classifier.  It consumes a versioned independent label file
(``v2_labels_dev.json``) plus trusted catalogue records, and reports separate metrics.
It never rewrites the frozen task/Gold files and never changes the legacy result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import BenchTask, TaskRun

EVALUATION_V2_VERSION = "agentrecx.eval_v2.1"
LABELS_VERSION = "agentrecx.eval_v2.labels.1"
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"


def load_v2_labels(path: Path | str | None = None, *, split: str = "dev") -> dict[str, Any]:
    source = Path(path) if path is not None else Path(__file__).with_name(f"v2_labels_{split}.json")
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    labels = payload.get("labels", {}) if isinstance(payload, dict) else {}
    return dict(labels) if isinstance(labels, dict) else {}


def _text(record: Any) -> str:
    return " ".join(str(getattr(record, "title", "") or "").lower().split())


def _category(record: Any) -> str:
    return " ".join(str(value) for value in (getattr(record, "categories", ()) or ())).lower()


def _price(record: Any) -> float | None:
    raw = getattr(record, "price", None)
    if raw is None:
        details = getattr(record, "details", None)
        if isinstance(details, Mapping):
            raw = details.get("Price") or details.get("price")
        elif isinstance(details, (list, tuple)):
            for pair in details:
                if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[0]).lower() == "price":
                    raw = pair[1]
                    break
    try:
        return None if raw is None else float(str(raw).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def assess_role(record: Any, label: Mapping[str, Any]) -> str:
    """Independent category/title role rubric; never imports the runtime classifier."""
    title = _text(record)
    category = _category(record)
    if any(term in category for term in label.get("reject_category_terms", ())):
        return FAIL
    if any(term in title for term in label.get("reject_title_terms", ())):
        return FAIL
    if any(term in category for term in label.get("accept_category_terms", ())):
        if label.get("target_role") == "accessory":
            return PASS if any(term in title for term in label.get("accept_title_terms_any", ())) else UNKNOWN
        return PASS
    if any(term in title for term in label.get("accept_title_terms_any", ())):
        return PASS
    return UNKNOWN


def _role_satisfaction(final_ids: Sequence[str], records: Mapping[str, Any], label: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    evidence: list[dict[str, str]] = []
    if not final_ids:
        return UNKNOWN, evidence
    statuses: list[str] = []
    for identity in final_ids:
        record = records.get(identity)
        if record is None:
            status = UNKNOWN
        else:
            status = assess_role(record, label)
        statuses.append(status)
        evidence.append({"parent_asin": identity, "role_status": status, "title": _text(record) if record else ""})
    if FAIL in statuses:
        return FAIL, evidence
    if PASS in statuses:
        return PASS, evidence
    return UNKNOWN, evidence


def _lifecycle_correct(run: TaskRun, label: Mapping[str, Any]) -> tuple[str, list[str]]:
    actual = {str(value).lower() for value in run.active_preference_attributes}
    expected = {str(value).lower() for value in label.get("active_preference_attributes", ())}
    inactive = {str(value).lower() for value in label.get("inactive_preference_attributes", ())}
    missing = sorted(expected - actual)
    resurrected = sorted(inactive & actual)
    if missing:
        return FAIL, [f"missing active attributes: {missing}"]
    if resurrected:
        return FAIL, [f"inactive attributes still active: {resurrected}"]
    return PASS, []


def _hard_constraints(final_ids: Sequence[str], records: Mapping[str, Any], label: Mapping[str, Any]) -> tuple[str, list[str]]:
    constraints = list(label.get("hard_constraints", ()) or ())
    if not constraints:
        return PASS, []
    problems: list[str] = []
    unknown = False
    for identity in final_ids:
        record = records.get(identity)
        if record is None:
            unknown = True
            continue
        for constraint in constraints:
            kind = str(constraint.get("kind", ""))
            if kind == "price_max":
                price = _price(record)
                if price is None:
                    unknown = True
                elif price > float(constraint.get("value", 0)):
                    problems.append(f"{identity}: price {price} > {constraint.get('value')}")
            else:
                unknown = True
    if problems:
        return FAIL, problems
    if unknown:
        return UNKNOWN, ["at least one final product has no trusted value for the labelled hard constraint"]
    return PASS, []


def _tool_action_validity(run: TaskRun, role_status: str, pool_role: bool) -> tuple[str, str]:
    actions = tuple(str(value).upper() for value in run.retrieval_actions)
    if not actions:
        if role_status == PASS:
            return "NOT_REQUIRED", "final candidate snapshot satisfies the independent role rubric"
        if not pool_role:
            return "REQUIRED_MISSING", "no R1 action ran and the candidate pool has no independently acceptable candidate"
        return "NOT_EXECUTED", "no R1 action ran, but the final role evidence is insufficient"
    invalid = [action for action in actions if action not in {"RETRIEVE", "RERANK"}]
    if invalid:
        return "INVALID", f"unknown retrieval action(s): {sorted(set(invalid))}"
    return "VALID", "only trusted retrieval actions were recorded"


@dataclass
class EvaluationV2:
    task_id: str
    policy: str
    task_state_correct: str
    preference_lifecycle_correct: str
    product_role_satisfaction: str
    hard_constraint_satisfaction: str
    exact_identity_hit: bool
    retrieval_adequacy: str
    tool_action_validity: str
    clarification_correctness: str
    task_success_v2: str
    failure_class: str
    evidence: dict[str, Any] = field(default_factory=dict)
    version: str = EVALUATION_V2_VERSION

    def as_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "policy": self.policy,
            "task_state_correct": self.task_state_correct,
            "preference_lifecycle_correct": self.preference_lifecycle_correct,
            "product_role_satisfaction": self.product_role_satisfaction,
            "hard_constraint_satisfaction": self.hard_constraint_satisfaction,
            "exact_identity_hit": self.exact_identity_hit,
            "retrieval_adequacy": self.retrieval_adequacy,
            "tool_action_validity": self.tool_action_validity,
            "clarification_correctness": self.clarification_correctness,
            "task_success_v2": self.task_success_v2,
            "failure_class": self.failure_class,
            "evaluator_version": self.version,
        }


def evaluate_run_v2(
    run: TaskRun,
    task: BenchTask,
    records: Mapping[str, Any],
    label: Mapping[str, Any],
) -> EvaluationV2:
    final_ids = tuple(run.final_ids)
    expected_ids = tuple(task.expected.final_any)
    exact_hit = bool(set(expected_ids) & set(final_ids))
    pool_ids = tuple(
        dict.fromkeys(
            identity
            for turn in run.turns
            for identity in (turn.candidate_pool_ids or ())
        )
    )
    role_status, role_evidence = _role_satisfaction(final_ids, records, label)
    pool_status = UNKNOWN
    for identity in pool_ids:
        record = records.get(identity)
        status = UNKNOWN if record is None else assess_role(record, label)
        if status == PASS:
            pool_status = PASS
            break
        if status == FAIL:
            pool_status = FAIL
    lifecycle, lifecycle_reasons = _lifecycle_correct(run, label)
    hard, hard_reasons = _hard_constraints(final_ids, records, label)
    action, action_reason = _tool_action_validity(run, role_status, pool_status == PASS)
    if not run.turns:
        clarification = UNKNOWN
    elif any(turn.response_status == "waiting_for_user" for turn in run.turns):
        clarification = "CORRECT_CONTINUED" if run.turns[-1].response_status in {"finished", "completed"} else "STILL_WAITING"
    else:
        clarification = "NOT_REQUIRED"
    if role_status == PASS:
        retrieval = "ADEQUATE"
    elif pool_status == PASS:
        retrieval = "SELECTION_OMITTED"
    elif pool_ids:
        retrieval = "CANDIDATE_POOL_INSUFFICIENT"
    else:
        retrieval = UNKNOWN

    metrics = [role_status, lifecycle, hard, clarification, action, retrieval]
    if role_status == FAIL:
        success = FAIL
    elif role_status != PASS:
        success = UNKNOWN
    elif FAIL in {lifecycle, hard, clarification, action}:
        success = FAIL
    elif UNKNOWN in {lifecycle, hard, clarification, action, retrieval}:
        success = UNKNOWN
    else:
        success = PASS
    task_state = role_status if role_status in {PASS, FAIL} else UNKNOWN

    if role_status == FAIL or lifecycle == FAIL or hard == FAIL:
        failure = "product_semantics"
    elif action == "REQUIRED_MISSING":
        failure = "necessary_retrieval_missing"
    elif retrieval == "CANDIDATE_POOL_INSUFFICIENT":
        failure = "retrieval_insufficient"
    elif role_status == UNKNOWN or hard == UNKNOWN:
        failure = "evidence_unknown"
    elif success == PASS:
        failure = "none"
    else:
        failure = "completion_or_evidence"
    return EvaluationV2(
        task_id=run.task_id,
        policy=run.policy.value,
        task_state_correct=task_state,
        preference_lifecycle_correct=lifecycle,
        product_role_satisfaction=role_status,
        hard_constraint_satisfaction=hard,
        exact_identity_hit=exact_hit,
        retrieval_adequacy=retrieval,
        tool_action_validity=action,
        clarification_correctness=clarification,
        task_success_v2=success,
        failure_class=failure,
        evidence={
            "role": role_evidence,
            "pool_role": pool_status,
            "lifecycle_reasons": lifecycle_reasons,
            "hard_reasons": hard_reasons,
            "action_reason": action_reason,
            "exact_identity_required": bool(label.get("exact_identity_required", False)),
            "label_version": LABELS_VERSION,
            "candidate_pool": list(pool_ids),
            "failures": list(run.error and [run.error] or []),
        },
    )

def evaluate_runs_v2(
    runs: Sequence[TaskRun],
    tasks: Sequence[BenchTask],
    records: Mapping[str, Any],
    *,
    labels: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[EvaluationV2]:
    """Evaluate explicit runs against the versioned V2 rubric, never legacy Gold."""
    by_task = {task.task_id: task for task in tasks}
    label_map = labels or load_v2_labels()
    result: list[EvaluationV2] = []
    for run in runs:
        task = by_task.get(run.task_id)
        if task is None:
            continue
        result.append(evaluate_run_v2(run, task, records, label_map.get(run.task_id, {})))
    return result
