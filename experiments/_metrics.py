"""Phase 3.1 experiment metrics: per-case behaviour, never one aggregate score.

The v1/v2 comparison needs to report things the existing plane does not: how often the run asked
a question, how often it asked one it should not have, how often it repeated a read-only action
whose observation added nothing, and the token and latency cost.

Deliberately *not* a score.  Each dimension is reported separately and each is attributable to a
component, so a reader can tell a contract-ambiguity failure from an evidence gap.  Nothing here
is folded into a single number, and nothing here is used to decide a case's pass or fail: the
evaluator's own verdict is the verdict, and these are the explanations beside it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "READ_ONLY_ACTIONS",
    "behaviour_metrics",
    "compare_metrics",
]

#: Actions that read facts and cannot change candidate membership.  Repeating one of these on the
#: same targets is the "no progress" pattern the phase names.
READ_ONLY_ACTIONS: frozenset[str] = frozenset(
    {"get_details", "compare", "trade_off", "check_compatibility", "bundle", "verify"}
)


def _repeated_read_only(records: Any) -> list[dict[str, Any]]:
    """Every occurrence of a read-only action repeated on the same targets, after the first.

    Keyed on ``(action, targets)``, not on the action alone: reading *different* candidates is
    progress, and comparing a different pair is a different question.  A second identical read is
    only "no progress" when the targets and the action are both unchanged.
    """
    seen: set[str] = set()
    repeats: list[dict[str, Any]] = []
    for record in records:
        signature = getattr(record, "read_signature", None)
        if not signature:
            continue
        action = str(signature).split(":", 1)[0]
        if action not in READ_ONLY_ACTIONS:
            continue
        if signature in seen:
            repeats.append({"step_index": record.step_index, "signature": signature})
        else:
            seen.add(signature)
    return repeats


def behaviour_metrics(outcome: Any) -> dict[str, Any]:
    """The per-case behaviour this phase reports, alongside the evaluator's own verdict.

    ``passed`` is the evaluator's verdict and is not recomputed here.  Every other field is an
    observation about *how* the run behaved, so a failure can be classified rather than merely
    counted.
    """
    trajectory = outcome.trajectory
    metrics = outcome.metrics
    sequence = list(trajectory.action_sequence())
    failures = list(metrics.failures())
    return {
        "case_id": outcome.case.case_id,
        "variant": trajectory.variant,
        # The evaluator's verdict, verbatim.
        "passed": outcome.passed,
        "terminal": trajectory.terminal.value,
        "termination_reason": trajectory.termination_reason,
        "failures": failures,
        "attributed": dict(outcome.attribution.attributed),
        # Behaviour.
        "trajectory": sequence,
        "trajectory_length": len(sequence),
        "clarification_count": sequence.count("ask_clarification"),
        "clarification_was_expected": metrics.clarification_was_expected,
        "clarification_was_forbidden": "ask_clarification"
        in outcome.case.forbidden_actions,
        # "Unnecessary" is the evaluator's judgement, not a heuristic.  Two distinct kinds are
        # reported separately, because they are different defects:
        #
        # * ``forbidden_clarification`` - the case explicitly forbids asking, so a question is a
        #   policy-selection violation regardless of anything else the run did;
        # * ``answered_but_asked`` - the case does not forbid asking, but the run failed *only*
        #   because it did not produce the answer it required.  Without the "only" this would
        #   count a run that also broke a budget as an over-clarification, which would overstate
        #   the finding.
        "forbidden_clarification": bool(
            sequence.count("ask_clarification")
            and "ask_clarification" in outcome.case.forbidden_actions
        ),
        "answered_but_asked": bool(
            sequence.count("ask_clarification")
            and "required_action_missing" in failures
            and "ask_clarification" not in outcome.case.forbidden_actions
        ),
        "repeated_read_only_actions": _repeated_read_only(trajectory.records),
        "repeated_read_only_count": len(_repeated_read_only(trajectory.records)),
        "budget_exhausted": trajectory.terminal.value == "budget_exhausted",
        "premature_finish": "completion" in failures
        and trajectory.terminal.value
        in {"completion_refused", "budget_exhausted", "failed", "asked_clarification"},
        "unnecessary_tool_calls": metrics.unnecessary_tool_calls,
        # Cost.
        "model_calls": trajectory.model_calls,
        "input_tokens": trajectory.input_tokens,
        "output_tokens": trajectory.output_tokens,
        "total_tokens": trajectory.total_tokens,
        "model_latency_ms": trajectory.model_latency_ms,
        "wall_latency_ms": trajectory.wall_latency_ms,
        "estimated_cost": trajectory.estimated_cost,
        "execution_mode": trajectory.execution_mode,
    }


def _sum(values: Any) -> int | None:
    present = [v for v in values if isinstance(v, int) and not isinstance(v, bool)]
    return sum(present) if present else None


def _sum_float(values: Any) -> float | None:
    present = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(float(v) for v in present), 3) if present else None


def suite_metrics(outcomes: Any) -> dict[str, Any]:
    """Aggregate the per-case behaviour into the comparison table's columns.

    Still not a score: ``passed`` is a count of the evaluator's verdicts, and every other field is
    a separate observation.  Unknowns stay ``None`` rather than becoming zero.
    """
    cases = [behaviour_metrics(outcome) for outcome in outcomes]
    return {
        "total": len(cases),
        "passed": sum(1 for c in cases if c["passed"]),
        "cases": cases,
        "clarification_count": sum(c["clarification_count"] for c in cases),
        "forbidden_clarification": sum(1 for c in cases if c["forbidden_clarification"]),
        "answered_but_asked": sum(1 for c in cases if c["answered_but_asked"]),
        "unnecessary_clarification_failures": sum(
            1
            for c in cases
            if (c["forbidden_clarification"] or c["answered_but_asked"]) and not c["passed"]
        ),
        "cases_with_repeated_read_only": sum(
            1 for c in cases if c["repeated_read_only_count"]
        ),
        "repeated_read_only_total": sum(c["repeated_read_only_count"] for c in cases),
        "budget_exhausted": sum(1 for c in cases if c["budget_exhausted"]),
        "premature_finish": sum(1 for c in cases if c["premature_finish"]),
        "trajectory_length_total": sum(c["trajectory_length"] for c in cases),
        "model_calls": _sum(c["model_calls"] for c in cases),
        "input_tokens": _sum(c["input_tokens"] for c in cases),
        "output_tokens": _sum(c["output_tokens"] for c in cases),
        "total_tokens": _sum(c["total_tokens"] for c in cases),
        "model_latency_ms": _sum_float(c["model_latency_ms"] for c in cases),
        "wall_latency_ms": _sum_float(c["wall_latency_ms"] for c in cases),
        "estimated_cost": _sum_float(c["estimated_cost"] for c in cases),
        "failures_by_component": _component_counts(cases),
    }


def _component_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        for component in set(case["attributed"].values()):
            counts[component] = counts.get(component, 0) + 1
    return dict(sorted(counts.items()))


def compare_metrics(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """A side-by-side comparison, plus which cases changed and how.

    ``changed`` lists only the cases whose *verdict* or *trajectory* differs, with both
    trajectories quoted, because "the score moved" is not an explanation.
    """
    by_id_first = {c["case_id"]: c for c in first["cases"]}
    changed: list[dict[str, Any]] = []
    for case in second["cases"]:
        before = by_id_first.get(case["case_id"])
        if before is None:
            continue
        if before["passed"] != case["passed"] or before["trajectory"] != case["trajectory"]:
            changed.append(
                {
                    "case_id": case["case_id"],
                    "v1_passed": before["passed"],
                    "v2_passed": case["passed"],
                    "v1_trajectory": before["trajectory"],
                    "v2_trajectory": case["trajectory"],
                    "v1_terminal": before["terminal"],
                    "v2_terminal": case["terminal"],
                    "v1_failures": before["failures"],
                    "v2_failures": case["failures"],
                }
            )
    columns = (
        "passed",
        "clarification_count",
        "forbidden_clarification",
        "answered_but_asked",
        "unnecessary_clarification_failures",
        "cases_with_repeated_read_only",
        "repeated_read_only_total",
        "budget_exhausted",
        "premature_finish",
        "trajectory_length_total",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "model_latency_ms",
        "wall_latency_ms",
        "estimated_cost",
    )
    return {
        "changed_cases": changed,
        "table": {
            column: {"v1": first.get(column), "v2": second.get(column)}
            for column in columns
        },
    }
