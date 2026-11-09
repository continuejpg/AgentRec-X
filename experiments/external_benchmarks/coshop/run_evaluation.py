"""Command-line runner for the CoShop external-benchmark adapter (M6.2, offline lab).

The CLI runs the official CoShop visible-query retrieval baseline over a small
number of benchmark tasks and reuses the official CoShop metric implementation.
It does **not** simulate a user, converse, or estimate the official dialogue /
decision metrics; those are reported as ``NOT_EXECUTED``.

Exit codes:
    0 -- evaluation executed and a report (with digest) was produced;
    2 -- the external benchmark is NOT EXECUTED (CoShop checkout, its optional
         dependencies or the official Google Drive data assets are unavailable, or
         the official payload violates the required contract).  A clean
         ``NOT EXECUTED: <reason>`` line is printed and no metrics are invented;
    1 -- an unexpected internal error.

This is an offline-lab tool only.  It changes no AgentRec-X production behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python experiments/external_benchmarks/coshop/run_evaluation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.external_benchmarks.coshop.adapter import (  # noqa: E402
    ADAPTER_VERSION,
    DEFAULT_DOMAIN,
    DEFAULT_K,
    DEFAULT_LIMIT,
    DEFAULT_SPLIT,
    DEFAULT_VERSION,
    PROTOCOL_NAME,
    SUPPORTED_DOMAINS,
    SUPPORTED_SPLITS,
    SUPPORTED_VERSIONS,
    CoshopContractError,
    CoshopEvaluationReport,
    CoshopUnavailable,
    evaluate_coshop,
)

__all__ = ["build_parser", "main", "write_not_executed_json"]


def _positive_int(text: str) -> int:
    """Argparse type: a strictly positive integer."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CoShop adapter CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.external_benchmarks.coshop.run_evaluation",
        description=(
            "Run the official CoShop visible-query retrieval baseline against the "
            "external CoShop benchmark (offline lab only; no serving-path effect)."
        ),
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, choices=SUPPORTED_DOMAINS)
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=SUPPORTED_VERSIONS)
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SUPPORTED_SPLITS)
    parser.add_argument(
        "--tasks",
        type=_positive_int,
        default=DEFAULT_LIMIT,
        help=f"number of consecutive tasks to evaluate (default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        default=DEFAULT_K,
        help=f"top-k passed to the official CoShop metrics (default {DEFAULT_K})",
    )
    parser.add_argument(
        "--coshop-root",
        default=None,
        help="path to the official external CoShop checkout (defaults to AGENTRECX_COSHOP_ROOT)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="write the deterministic report JSON (including report digest) to this path",
    )
    return parser


def _format_metric(value: object) -> str:
    """Format one metric value for the human-readable summary."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6f}"
    return "n/a"


def _write_json(path: str, payload: object) -> None:
    """Write deterministic JSON to ``path`` (no NaN/inf, sorted keys)."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    target.write_text(text + "\n", encoding="utf-8")


def write_not_executed_json(path: str | None, *, reason: str, args: argparse.Namespace) -> None:
    """Write an explicit NOT_EXECUTED JSON document (never fake metrics)."""
    if not path:
        return
    _write_json(
        path,
        {
            "status": "NOT_EXECUTED",
            "benchmark": "coshop",
            "reason": reason,
            "request": {
                "domain": args.domain,
                "version": args.version,
                "split": args.split,
                "tasks": args.tasks,
                "k": args.k,
                "coshop_root": args.coshop_root,
            },
            "official_metrics": None,
            "custom_diagnostics": None,
            "decision_metrics": {
                "team_accuracy": {"status": "NOT_EXECUTED", "value": None},
                "conditional_selection_accuracy": {"status": "NOT_EXECUTED", "value": None},
                "interaction_cost": {"status": "NOT_EXECUTED", "value": None},
            },
        },
    )


def _print_summary(report: CoshopEvaluationReport) -> None:
    """Print the human-readable report summary."""
    official = report.official_metrics
    custom = report.custom_diagnostics
    provenance = report.provenance
    catalog = provenance.get("catalog", {})
    print("CoShop external benchmark -- official visible-query retrieval baseline")
    print(
        f"adapter={report.adapter_version} protocol={report.protocol} "
        f"domain={report.domain} version={report.version} split={report.split} "
        f"tasks={report.task_count} k={report.k} candidates={report.candidate_count_total}"
    )
    print(f"baseline: {report.baseline}")
    print(f"candidate source: {report.candidate_source}")
    print(
        "official metrics "
        f"({official.get('source')}, k={official.get('k')}, "
        f"compute_embedding_ustar={official.get('compute_embedding_ustar')}):"
    )
    print(f"  recall@{official.get('k')}      = {_format_metric(official.get('recall_at_k'))}")
    print(f"  recall@1       = {_format_metric(official.get('recall_at_1'))}")
    print(
        f"  ndcg_ustar@{official.get('k')} = "
        f"{_format_metric(official.get('ndcg_ustar_at_k'))}"
    )
    print(f"custom diagnostics ({custom.get('label')}; NOT official):")
    print(
        f"  target_in_candidate_pool_rate = "
        f"{_format_metric(custom.get('target_in_candidate_pool_rate'))}"
    )
    print(
        f"  mean_target_original_rank_in_pool = "
        f"{_format_metric(custom.get('mean_target_original_rank_in_pool'))}"
    )
    print(
        f"  mean_custom_recall_at_k = {_format_metric(custom.get('mean_custom_recall_at_k'))} "
        f"(k={report.k})"
    )
    print(
        f"  retrieval_calls={custom.get('retrieval_calls')} "
        f"dialogue_turns={custom.get('dialogue_turns')} "
        f"clarifying_questions={custom.get('clarifying_questions')} "
        f"tool_calls={custom.get('tool_calls')}"
    )
    print("decision metrics: NOT_EXECUTED")
    for name, status in report.decision_metrics.items():
        print(f"  {name}: {status['status']} -- {status['reason']}")
    print(
        f"provenance: catalog_ids={catalog.get('id_count')} "
        f"sha256={catalog.get('ordered_ids_sha256')} "
        f"coshop_root={provenance.get('coshop_root')} "
        f"coshop_package_version={provenance.get('coshop_package_version')} "
        f"seed={provenance.get('seed')}"
    )
    print(f"report digest: {report.report_digest}")
    for note in report.notes:
        print(f"note: {note}")


def main(argv: list[str] | None = None) -> int:
    """Run the CoShop adapter CLI; return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_coshop(
            domain=args.domain,
            version=args.version,
            split=args.split,
            limit=args.tasks,
            k=args.k,
            coshop_root=args.coshop_root,
        )
    except (CoshopUnavailable, CoshopContractError) as error:
        reason = str(error)
        print(f"NOT EXECUTED: {reason}")
        write_not_executed_json(args.json_path, reason=reason, args=args)
        return 2
    except Exception as error:  # unexpected internal failure, never a fake result
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    _print_summary(report)
    if args.json_path:
        _write_json(args.json_path, report.as_dict())
        print(f"report JSON written to {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
