"""M6.1 frozen paired evaluation runner (synthetic tasks; offline lab only).

Runs :class:`recommendation.ranking.evaluation.FrozenPairedEvaluator` under the frozen
preregistered protocol over the shared synthetic hiking-backpack tasks from
``tests/m6_1_fixture.py`` and prints:

* the protocol name and digest the run was produced under;
* per-arm **contract** metrics (identity preservation, candidate-set equality,
  determinism, eligibility violations, change rates, top-K overlaps vs A0/A1,
  explicit-evidence totals, inferred-evidence coverage);
* the quality status and, only if documented labels existed, the paired quality deltas;
* the aggregated feature-coverage / eligibility-exclusion / unsupported-feature notes;
* an explicit **no quality claim** statement.

The fixture tasks deliberately carry no preference-conditioned labels, so quality is
reported as ``NOT_EXECUTED_NO_LABELS``: historical purchase relevance is not
preference-conditioned, and ranking change is never quality evidence.  Any contract gate
failure (identity, eligibility, determinism) makes the runner fail loudly.

Everything is offline and deterministic: no checkpoint, no network, no LLM, no labels.

Usage::

    python -m experiments.m6_1_belief_aware_evaluation [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.ranking.evaluation import (  # noqa: E402
    DEFAULT_PROTOCOL,
    FrozenPairedEvaluator,
)
from recommendation.ranking.schemas import EvaluationReport, RankingArm  # noqa: E402

DISCLAIMER = (
    "offline lab only; Serving V1 not activated; no quality claim (quality status must be "
    "MEASURED under a documented preference-conditioned label protocol before any "
    "effectiveness claim)"
)


class EvaluationRunError(RuntimeError):
    """A frozen contract gate or the run itself failed."""


def _load_tasks() -> tuple[Any, ...]:
    """Load the shared synthetic task set, or stop with a clear message."""
    try:
        from tests import m6_1_fixture  # noqa: PLC0415 - clean CLI error
    except Exception as error:  # noqa: BLE001 - reported, never guessed
        raise EvaluationRunError(
            "the shared fixture tests/m6_1_fixture.py is not importable: "
            f"{type(error).__name__}: {error}"
        ) from error
    try:
        return tuple(m6_1_fixture.build_tasks())
    except Exception as error:  # noqa: BLE001 - reported, never guessed
        raise EvaluationRunError(
            f"the shared fixture could not build its tasks: {type(error).__name__}: {error}"
        ) from error


def _contract_checks(report: EvaluationReport) -> list[str]:
    """Return human-readable gate failures; empty means the frozen contract holds."""
    failures: list[str] = []
    if not report.report_digest or len(report.report_digest) != 64:
        failures.append("the report digest is missing or malformed")
    if report.protocol_digest != DEFAULT_PROTOCOL.digest:
        failures.append("the report was produced under a different protocol digest")
    if report.task_count < 1:
        failures.append("no frozen task was evaluated")
    expected_arms = list(DEFAULT_PROTOCOL.arms)
    if [arm.arm for arm in report.arms] != expected_arms:
        failures.append("the report does not contain exactly the preregistered arms")
    for arm in report.arms:
        contract = arm.contract
        if not contract.candidate_identity_preserved:
            failures.append(f"{arm.arm.value}: candidate identity was not preserved")
        if not contract.candidate_set_equal_to_expected_pool:
            failures.append(
                f"{arm.arm.value}: candidate set differs from the expected ranking pool "
                "(snapshot when nothing assessed, otherwise the verified-eligible set)"
            )
        if not contract.ordering_deterministic:
            failures.append(f"{arm.arm.value}: ordering was not deterministic")
        if contract.eligibility_invariant_violations != 0:
            failures.append(
                f"{arm.arm.value}: {contract.eligibility_invariant_violations} "
                "eligibility invariant violation(s)"
            )
        if arm.arm is RankingArm.A0_ORIGINAL:
            if contract.change_rate_vs_a0 != 0.0:
                failures.append("A0 must not change relative to itself")
        if contract.candidate_count < 0:  # pragma: no cover - defensive
            failures.append(f"{arm.arm.value}: negative candidate count")
    return failures


def _format_arm_table(report: EvaluationReport) -> str:
    """Render the per-arm contract metrics table."""
    headers = (
        "arm",
        "candidates",
        "identity",
        "set=expected_pool",
        "set=snapshot",
        "deterministic",
        "elig_violations",
        "change_vs_A0",
        "change_vs_A1",
        "top1_overlap_A0",
        "top3_overlap_A0",
        "top5_overlap_A0",
        "explicit_violations",
        "explicit_matches",
        "inferred_covered",
        "inferred_unknown",
    )
    rows: list[dict[str, Any]] = []
    for arm in report.arms:
        contract = arm.contract
        rows.append(
            {
                "arm": arm.arm.value,
                "candidates": contract.candidate_count,
                "identity": contract.candidate_identity_preserved,
                "set=expected_pool": contract.candidate_set_equal_to_expected_pool,
                "set=snapshot": contract.candidate_set_equal_to_snapshot,
                "deterministic": contract.ordering_deterministic,
                "elig_violations": contract.eligibility_invariant_violations,
                "change_vs_A0": f"{contract.change_rate_vs_a0:.4f}",
                "change_vs_A1": f"{contract.change_rate_vs_a1:.4f}",
                "top1_overlap_A0": f"{contract.top_k_overlap_vs_a0.get('1', 0.0):.4f}",
                "top3_overlap_A0": f"{contract.top_k_overlap_vs_a0.get('3', 0.0):.4f}",
                "top5_overlap_A0": f"{contract.top_k_overlap_vs_a0.get('5', 0.0):.4f}",
                "explicit_violations": contract.explicit_violations_total,
                "explicit_matches": contract.explicit_matches_total,
                "inferred_covered": contract.inference_covered_candidates,
                "inferred_unknown": contract.inference_unknown_candidates,
            }
        )
    widths = {
        header: max(len(header), max((len(str(row[header])) for row in rows), default=0))
        for header in headers
    }
    lines = ["  ".join(header.ljust(widths[header]) for header in headers)]
    for row in rows:
        lines.append("  ".join(str(row[header]).ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def _format_quality(report: EvaluationReport) -> str:
    """Render the quality status and any paired deltas (never inferred from change)."""
    lines = [f"quality_status: {report.quality_status}"]
    for arm in report.arms:
        quality = arm.quality
        if quality.status == "MEASURED":
            lines.append(
                f"  {arm.arm.value}: MRR={quality.mrr} "
                f"Recall@K={quality.recall_at_k} NDCG@K={quality.ndcg_at_k} "
                f"tasks={quality.task_count_with_labels}"
            )
        else:
            lines.append(f"  {arm.arm.value}: {quality.status} ({quality.note})")
    if report.paired_deltas:
        lines.append("paired quality deltas (A2 - baseline):")
        for delta in report.paired_deltas:
            lines.append(
                f"  {delta.comparison} {delta.metric}: mean={delta.mean_delta:.6f} "
                f"CI=[{delta.ci_low}, {delta.ci_high}] n={delta.sample_size} "
                f"status={delta.status}"
            )
    else:
        lines.append(
            "paired quality deltas: none (quality NOT_EXECUTED; change rates and "
            "overlaps are contract diagnostics, never quality evidence)"
        )
    return "\n".join(lines)


def run_evaluation() -> dict[str, Any]:
    """Run the frozen paired evaluation and return the JSON-able evidence record."""
    tasks = _load_tasks()
    if not tasks:
        raise EvaluationRunError("the fixture returned no frozen tasks")
    if any(task.labels for task in tasks):
        # The protocol allows labels, but the fixture must document the protocol and the
        # runner must then report MEASURED deltas; that is a different frozen scenario.
        raise EvaluationRunError(
            "the synthetic fixture unexpectedly carries labels; this runner expects the "
            "unlabeled preregistered scenario (quality NOT_EXECUTED_NO_LABELS)"
        )
    evaluator = FrozenPairedEvaluator(protocol=DEFAULT_PROTOCOL)
    report = evaluator.evaluate(tasks)
    failures = _contract_checks(report)
    if failures:
        raise EvaluationRunError("; ".join(failures))

    print(f"protocol: {report.protocol_name}")
    print(f"protocol_digest: {report.protocol_digest}")
    print(f"dataset_source: {report.dataset_source}")
    print(f"tasks: {report.task_count}  candidates_total: {report.candidate_count_total}")
    print(f"report_digest: {report.report_digest}")
    print()
    print("per-arm contract metrics (behavioural diagnostics, not quality):")
    print(_format_arm_table(report))
    print(
        "note: set=expected_pool is the frozen contract check (snapshot when no assessment "
        "exists, otherwise the verified-eligible set). set=snapshot is the literal snapshot "
        "comparison and is False here only because the trusted eligibility projection "
        "legitimately excludes INELIGIBLE/UNRESOLVED candidates, which are never rankable."
    )
    print()
    print(_format_quality(report))
    print()
    print("feature coverage / exclusions / unsupported features:")
    for note in report.notes:
        print(f"  - {note}")
    print()
    print(DISCLAIMER)
    return {
        "protocol_name": report.protocol_name,
        "protocol_digest": report.protocol_digest,
        "dataset_source": report.dataset_source,
        "task_count": report.task_count,
        "candidate_count_total": report.candidate_count_total,
        "quality_status": report.quality_status,
        "report": report.as_dict(),
        "disclaimer": DISCLAIMER,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and return the process exit code (0 = success)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=None, help="optional path for the report JSON"
    )
    arguments = parser.parse_args(argv)
    try:
        evidence = run_evaluation()
    except EvaluationRunError as error:
        print(f"EVALUATION: FAIL - {error}")
        return 1
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"report written to {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
