"""JSON and Markdown presentation for a real Golden Dataset run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["write_json_report", "write_markdown_report"]


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")
    return text


def write_json_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _metric_rows(metrics: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    task = metrics["task_outcome_accuracy"]
    req = metrics["requirement_accuracy"]
    budget = metrics["budget_violations"]
    precision = metrics["retrieval_precision_at_5"]
    latency = metrics["latency_ms"]
    refusal = metrics["correct_refusals"]
    execution = metrics["execution_failures"]
    return [
        (
            "Task Outcome Accuracy",
            _pct(task["value"]),
            f"{task['numerator']} / {task['denominator']}",
        ),
        (
            "Outcome-kind Accuracy (without hard constraints)",
            _pct(task["outcome_kind_accuracy"]),
            f"{task['outcome_kind_numerator']} / {task['denominator']}",
        ),
        (
            "Requirement Accuracy",
            _pct(req["value"]),
            f"{req['numerator']} / {req['denominator']} fields across {req['observed_cases']} observed cases",
        ),
        (
            "Budget Violations",
            _num(budget["count"]),
            f"rate {_pct(budget['rate'])}; denominator {budget['denominator']} budget-constrained cases",
        ),
        (
            "Correct Refusals",
            _pct(refusal["value"]),
            f"{refusal['numerator']} / {refusal['denominator']} expected refusals",
        ),
        (
            "Execution Failures",
            _num(execution["count"]),
            "HTTP non-200 / runner error; not counted as refusal",
        ),
        (
            "Retrieval Precision@5",
            _pct(precision["value"]),
            f"{precision['numerator']} / {precision['denominator']} judged retrieval slots; "
            f"{precision['judged_cases']} judged cases",
        ),
        (
            "Latency mean / p50 / p95 / max",
            f"{_num(latency['mean'])} / {_num(latency['p50'])} / {_num(latency['p95'])} / {_num(latency['max'])} ms",
            f"{latency['count']} measured cases; app/MCP startup excluded",
        ),
    ]


def write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    metadata = report["metadata"]
    dataset = report["dataset"]
    metrics = report["metrics"]
    evaluations = {
        row["case_id"]: row for row in report.get("case_evaluations") or []
    }
    raw_results = {row["case_id"]: row for row in report.get("results") or []}
    case_defs = {case["case_id"]: case for case in dataset.get("cases") or []}

    lines: list[str] = [
        "# AgentRec-X Golden Agent Task Evaluation",
        "",
        "> This report was generated from a real CPU/BM25 FastAPI + MCP run.",
        "> HTTP 200 is not treated as task success; refusals are scored as refusals.",
        "> The sample size is intentionally small and no statistical-significance claim is made.",
        "",
        "## Run",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Run id | `{_cell(metadata['run_id'])}` |",
        f"| Generated | {_cell(metadata['generated_at'])} |",
        f"| Git commit | `{_cell(metadata['git_sha'])}` |",
        f"| Git dirty | {_num(metadata['git_dirty'])} |",
        f"| Dataset version | `{_cell(metadata['dataset_version'])}` |",
        f"| Dataset cases (total / selected) | {metadata['dataset_case_count']} / {metadata['selected_case_count']} |",
        f"| Dataset SHA256 | `{_cell(metadata['dataset_sha256'])}` |",
        f"| Catalogue SHA256 | `{_cell(metadata['catalog_sha256'])}` |",
        f"| Catalogue records | {metadata['catalog_size']} |",
        f"| Sidecar SHA256 | `{_cell(metadata['sidecar_sha256'])}` |",
        f"| Price mode(s) / rate | `{_cell(', '.join(metadata.get('price_modes') or [metadata['price_mode']]))}` / `{_cell(metadata['fx_rate'])}` |",
        f"| Search | `{_cell(metadata['search'])}` |",
        f"| Python | `{_cell(metadata['python']['version'])}` |",
        f"| Platform | {_cell(metadata['python']['platform'])} |",
        f"| App/MCP setup | {_num(metadata['setup_ms'])} ms |",
        f"| Run wall time | {_num(metadata['run_wall_ms'])} ms |",
        f"| Tracing | disabled; Phoenix not required |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Numerator / denominator |",
        "|---|---:|---|",
    ]
    for name, value, denominator in _metric_rows(metrics):
        lines.append(f"| {name} | {value} | {_cell(denominator)} |")

    lines.extend([
        "",
        "### Requirement fields",
        "",
        "| Field | Passed | Total |",
        "|---|---:|---:|",
    ])
    for field, values in sorted(metrics["requirement_accuracy"]["by_field"].items()):
        lines.append(f"| {_cell(field)} | {values['passed']} | {values['total']} |")

    lines.extend([
        "",
        "### Latency distribution",
        "",
        f"- count: {metrics['latency_ms']['count']}",
        f"- mean: {_num(metrics['latency_ms']['mean'])} ms",
        f"- p50: {_num(metrics['latency_ms']['p50'])} ms",
        f"- p95: {_num(metrics['latency_ms']['p95'])} ms",
        f"- max: {_num(metrics['latency_ms']['max'])} ms",
        "",
        "## Case results",
        "",
        "| Case | Category | Mode | Expected | Actual | Task success | Latency (ms) | Failure types |",
        "|---|---|---|---|---|---:|---:|---|",
    ])
    for case in dataset.get("cases") or []:
        case_id = case["case_id"]
        evaluation = evaluations.get(case_id, {})
        result = raw_results.get(case_id, {})
        lines.append(
            "| `{case}` | {cat} | {mode} | {expected} | {actual} | {ok} | {lat} | {fail} |".format(
                case=_cell(case_id),
                cat=_cell(case.get("category")),
                mode=_cell(case.get("execution_mode")),
                expected=_cell(case.get("expected_outcome", {}).get("kind")),
                actual=_cell(evaluation.get("actual_outcome")),
                ok=_num(evaluation.get("task_success")),
                lat=_num((evaluation.get("latency_ms") or {}).get("end_to_end_ms")),
                fail=_cell(", ".join(evaluation.get("failure_types") or [])),
            )
        )

    failures = [
        evaluation for evaluation in evaluations.values()
        if not evaluation.get("task_success")
    ]
    lines.extend([
        "",
        "## Failure cases",
        "",
    ])
    if not failures:
        lines.append("No failed task outcomes in this run.")
    else:
        lines.extend([
            "| Case | Expected | Actual | HTTP | Failure types | Details |",
            "|---|---|---|---:|---|---|",
        ])
        for evaluation in sorted(failures, key=lambda row: row["case_id"]):
            case_id = evaluation["case_id"]
            result = raw_results.get(case_id, {})
            case = case_defs.get(case_id, {})
            detail = (
                result.get("error")
                or evaluation.get("actual_refusal_code")
                or case.get("notes")
                or ""
            )
            lines.append(
                "| `{case}` | {expected} | {actual} | {http} | {fail} | {detail} |".format(
                    case=_cell(case_id),
                    expected=_cell(evaluation.get("expected_outcome")),
                    actual=_cell(evaluation.get("actual_outcome")),
                    http=_cell(evaluation.get("http_status")),
                    fail=_cell(", ".join(evaluation.get("failure_types") or [])),
                    detail=_cell(detail),
                )
            )

    precision = metrics["retrieval_precision_at_5"]
    lines.extend([
        "",
        "## Retrieval Precision@5 detail",
        "",
        "Manual labels are binary and independent of BM25 scores. Only cases whose entire "
        "retrieved top-5 pool is manually judged are included.",
        "",
    ])
    if not precision["cases"]:
        lines.append("No judged retrieval cases in this run.")
    else:
        lines.extend([
            "| Case | Top-5 | Relevant | Precision@5 |",
            "|---|---|---:|---:|",
        ])
        for row in precision["cases"]:
            lines.append(
                f"| `{_cell(row['case_id'])}` | {_cell(', '.join(row['top5']))} | "
                f"{row['relevant']} | {_pct(row['precision'])} |"
            )
    if precision["excluded_cases"]:
        lines.append("")
        lines.append("Excluded judged cases:")
        for row in precision["excluded_cases"]:
            lines.append(f"- `{row['case_id']}`: {_cell(row['reason'])}")

    lines.extend([
        "",
        "## Reproduction",
        "",
        "```bash",
        " ".join(metadata.get("command") or []),
        "```",
        "",
        "The exact command above, the recorded Git commit, the dataset SHA256 and the "
        "catalogue/sidecar SHA256 define a reproducible run. Each case uses a fresh Web "
        "session and isolated temporary SQLite databases; no developer database is touched.",
        "",
        "## Interpretation limits",
        "",
        "- 15-20 tasks are a representative engineering fixture, not a statistically "
        "significant benchmark.",
        "- `demo_historical_fx` prices are historical USD values converted at the fixed "
        "7.00 demo rate, not live market prices.",
        "- `budget_unverifiable` / `budget_exceeded` refusals are correct business outcomes "
        "when expected; they are not execution failures.",
        "- Retrieval Precision@5 is reported only over manually judged cases; unjudged "
        "retrieval ids are excluded rather than assumed non-relevant.",
        "- The runner uses tracing disabled; Phoenix is not a runtime dependency.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
