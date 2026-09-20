"""Agent evaluation-plane smoke test (AgentRec-X 2.0-alpha Stage 6).

Runs the whole evaluation case set through a **real** control plane and prints the report the
stage asks for: per-component failure attribution and an ablation comparison, never a single
aggregate score.

::

    .venv/bin/python -m experiments.agent_evaluation_smoke
    .venv/bin/python -m experiments.agent_evaluation_smoke --json /tmp/agent_eval.json

What it measures
----------------
**Trajectory quality**, not recommendation accuracy.  The recommender's own metrics live in
`recommendation/evaluation/metrics.py` under the accepted full-catalogue temporal protocol and
are not touched here; this smoke reuses none of them, because where a target item lands in a
ranking says nothing about whether an agent chose sensible actions.

The two variants it compares:

``adaptive``
    the bounded loop as built - the policy may change its next action because of the
    observation it was just given;
``decide_once``
    the historical baseline as a policy - one decision, then wrap up regardless of the result.

Fully offline: no checkpoint, no GPU, no network and no provider API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    CandidateLedger,
    CandidatePlane,
    CatalogSearchSource,
    GroundedReasoner,
    GroundingVerifier,
    ReasoningExecutor,
)
from recommendation.evaluation.agent import (  # noqa: E402
    ABLATION_ADAPTIVE,
    ABLATION_DECIDE_ONCE,
    EVALUATION_CASES,
    CaseRunner,
    DecidingOncePolicy,
    EvaluationCase,
    load_cases,
)
from recommendation.evaluation.agent.runner import build_adaptive_policy  # noqa: E402
from tests.agent_reranking_fixture import CANDIDATE_ROWS  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402


class _Map:
    """The trusted identity map the fixture catalogue is keyed by."""

    def __init__(self, rows: Any = CANDIDATE_ROWS) -> None:
        self._ids = {row[0]: row[1] for row in rows}

    def has_parent_asin(self, parent_asin: str) -> bool:
        return parent_asin in self._ids

    def parent_asin_to_item_id(self, parent_asin: str) -> int:
        return self._ids[parent_asin]

    def item_id_to_parent_asin(self, item_id: int) -> str:
        for identity, value in self._ids.items():
            if value == item_id:
                return identity
        raise KeyError(item_id)


def _reasoner() -> GroundedReasoner:
    """A reasoner over the suite catalogue."""
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )


def _factory(case: EvaluationCase, policy: Any, limits: Any) -> Any:
    """Compose a control plane for one case, with a widened catalogue."""
    empty_history = "recovery-required" in case.case_id
    harness = build_control_harness(
        policy=policy,
        limits=limits,
        rows=() if empty_history else CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
    )
    metadata = harness.parts["enricher"].metadata
    harness.controller._reasoning = ReasoningExecutor(GroundedReasoner(metadata))  # noqa: SLF001
    harness.controller._candidate_plane = CandidatePlane(  # noqa: SLF001
        ledger=CandidateLedger(),
        grounding=GroundingVerifier(_Map(), metadata),
        catalog_search=CatalogSearchSource(metadata),
    )
    return harness.controller


def run(*, json_path: Path | None = None) -> int:
    """Run the evaluation smoke and return a process exit code."""
    print("=" * 78)
    print(" AgentRec-X 2.0-alpha agent evaluation plane")
    print("=" * 78)
    print()
    print(f" cases: {len(EVALUATION_CASES)}   variants: {ABLATION_ADAPTIVE}, {ABLATION_DECIDE_ONCE}")
    print()

    reports: dict[str, Any] = {}
    for variant, policy_for in (
        (ABLATION_ADAPTIVE, build_adaptive_policy),
        (ABLATION_DECIDE_ONCE, lambda case: DecidingOncePolicy(build_adaptive_policy(case))),
    ):
        runner = CaseRunner(_factory, variant=variant)
        print(f" {variant}")
        outcomes = []
        for case in load_cases():
            outcome = runner.run(case, policy=policy_for(case), reasoner=_reasoner())
            outcomes.append(outcome)
            mark = "PASS" if outcome.passed else "FAIL"
            actions = " -> ".join(outcome.trajectory.action_sequence()) or "(none)"
            print(
                f"   [{mark}] {case.case_id:32s} {outcome.trajectory.terminal.value:19s} "
                f"tools={outcome.trajectory.tool_calls} steps={outcome.trajectory.steps}"
            )
            print(f"          {actions}")
            if not outcome.passed:
                for name, owner in outcome.attribution.attributed.items():
                    print(f"          ! {name} <- {owner}")
        passed = sum(1 for outcome in outcomes if outcome.passed)
        by_component: dict[str, int] = {}
        for outcome in outcomes:
            for component in outcome.attribution.components:
                by_component[component] = by_component.get(component, 0) + 1
        print()
        print(f"   {passed}/{len(outcomes)} passed")
        print(f"   failures by component: {by_component or '{} (none)'}")
        print()
        reports[variant] = {
            "passed": passed,
            "total": len(outcomes),
            "failures_by_component": by_component,
            "cases": [
                {
                    "case_id": outcome.case.case_id,
                    "category": outcome.case.category,
                    "passed": outcome.passed,
                    "terminal": outcome.trajectory.terminal.value,
                    "actions": list(outcome.trajectory.action_sequence()),
                    "failures": list(outcome.metrics.failures()),
                    "attributed": outcome.attribution.attributed,
                }
                for outcome in outcomes
            ],
        }

    print(" ablation: observation-conditioned control vs decide-once")
    for case in load_cases():
        adaptive = next(
            row for row in reports[ABLATION_ADAPTIVE]["cases"] if row["case_id"] == case.case_id
        )
        once = next(
            row for row in reports[ABLATION_DECIDE_ONCE]["cases"] if row["case_id"] == case.case_id
        )
        if adaptive["passed"] == once["passed"]:
            continue
        print(f"   {case.case_id}")
        print(f"     {ABLATION_ADAPTIVE:12s} {'PASS' if adaptive['passed'] else 'FAIL'}  {adaptive['actions']}")
        print(f"     {ABLATION_DECIDE_ONCE:12s} {'PASS' if once['passed'] else 'FAIL'}  {once['actions']}")
    print()

    print("Trajectory quality only. NOT a recommendation-accuracy benchmark, and no single")
    print("aggregate score is produced: NDCG does not measure orchestration.")
    payload = {"reports": reports}
    if json_path is not None:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")

    adaptive_passed = reports[ABLATION_ADAPTIVE]["passed"]
    return 0 if adaptive_passed == len(EVALUATION_CASES) else 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)
    return run(json_path=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
