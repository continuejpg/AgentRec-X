"""JSONL/CSV/Markdown report generation for the shopping benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import BenchTask, Evaluation, PolicyId, TaskRun


def _run_payload(run: TaskRun, state_dir: str | None = None) -> dict[str, Any]:
    payload = asdict(run)
    payload["policy"] = run.policy.value
    payload["state_dir"] = state_dir
    return payload


def _evaluation_payload(evaluation: Evaluation) -> dict[str, Any]:
    payload = asdict(evaluation)
    payload["policy"] = evaluation.policy.value
    return payload


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def write_reports(
    out_dir: Path | str,
    *,
    runs: Sequence[TaskRun],
    evaluations: Sequence[Evaluation],
    tasks: Sequence[BenchTask],
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the per-task trajectory, metrics, summary and failure report.

    These files are observations, not claims of shopping satisfaction.  Human labels
    remain in a separate annotation file and are never synthesized here.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_rows = [_run_payload(run, str(destination / run.policy.value)) for run in runs]
    eval_rows = [_evaluation_payload(item) for item in evaluations]

    runs_path = destination / "trajectories.jsonl"
    evals_path = destination / "evaluations.jsonl"
    _write_jsonl(runs_path, run_rows)
    _write_jsonl(evals_path, eval_rows)

    metrics_path = destination / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "task_id",
            "policy",
            "runnable",
            "completed",
            "passed",
            "final_ids",
            "checks",
            "failures",
            "human_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation in evaluations:
            row = evaluation.as_row()
            row["human_label"] = evaluation.human_label or ""
            row["final_ids"] = "|".join(evaluation.final_ids)
            row["checks"] = json.dumps(evaluation.checks, sort_keys=True)
            row["failures"] = " | ".join(evaluation.failures)
            writer.writerow(row)

    summary_path = destination / "summary.md"
    failure_path = destination / "failures.md"
    counts: dict[str, dict[str, int]] = {}
    for policy in PolicyId:
        rows = [item for item in evaluations if item.policy is policy]
        if not rows:
            continue
        counts[policy.value] = {
            "available": len(rows),
            "runnable": sum(1 for row in rows if row.runnable),
            "completed": sum(1 for row in rows if row.completed),
            "passed": sum(1 for row in rows if row.passed),
            "human_labeled": sum(1 for row in rows if row.human_label not in (None, "")),
        }
    lines = [
        "# Multi-turn shopping Agent benchmark",
        "",
        "Deterministic task completion, product role and state assertions are reported "
        "separately from human relevance labels. These are not satisfaction claims.",
        "",
        "## Policy availability",
        "",
        "| policy | available | runnable | completed | deterministic pass | human labeled |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in PolicyId:
        row = counts.get(policy.value)
        if row:
            lines.append(
                f"| {policy.value} | {row['available']} | {row['runnable']} | "
                f"{row['completed']} | {row['passed']} | {row['human_labeled']} |"
            )
    lines.extend(["", "## Control-plane facts", ""])
    if meta:
        for key, value in sorted(meta.items()):
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## What is and is not measured",
            "",
            "- Measured: task completion, final product identity/role, expected active",
            "  preference attributes, observed R1 action, token usage and latency.",
            "- Not measured: human relevance/satisfaction unless a separate human label file",
            "  exists; next-item HR/NDCG is not used as a shopping success label.",
            "- A/B/D are unavailable when the required LLM provider is not configured; the",
            "  runner records the unavailable reason rather than simulating a result.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = [
        item for item in evaluations
        if item.runnable and not item.passed
    ] + [
        item for item in evaluations if not item.runnable
    ]
    failure_lines = ["# Failures and unavailable policies", ""]
    for item in failures:
        failure_lines.append(f"## {item.policy.value} / {item.task_id}")
        if not item.runnable:
            failure_lines.append("- unavailable")
        else:
            for failure in item.failures:
                failure_lines.append(f"- {failure}")
        failure_lines.append("")
    failure_path.write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    return {
        "trajectories": runs_path,
        "evaluations": evals_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "failures": failure_path,
    }
