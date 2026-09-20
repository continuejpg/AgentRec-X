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
    CandidateEligibilityEvaluator,
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
    ABLATION_MODEL_POLICY,
    EVALUATION_CASES,
    CaseRunner,
    DecidingOncePolicy,
    EvaluationCase,
    build_model_policy_factory,
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
    """A reasoner over the suite catalogue.

    Built from a harness that is given ``catalog_rows``, because the default harness keyed its
    catalogue to the same rows the *engine* returns - and several cases drive the engine with
    nothing, which would otherwise leave the reasoner over an empty catalogue.
    """
    return GroundedReasoner(
        build_control_harness(catalog_rows=CANDIDATE_ROWS).parts["enricher"].metadata
    )


def _factory(
    case: EvaluationCase,
    policy: Any,
    limits: Any,
    task_state: Any = None,
) -> Any:
    """Compose a control plane for one case, with a widened catalogue.

    ``task_state`` is the case's declared hard constraints, so a case that asserts enforcement
    runs against a constraint that is genuinely active.  It is supplied by the runner at run
    time, so the factory only has to accept it.
    """
    empty_history = "recovery-required" in case.case_id
    harness = build_control_harness(
        policy=policy,
        limits=limits,
        rows=() if empty_history else CANDIDATE_ROWS,
        catalog_rows=CANDIDATE_ROWS,
        engine_error=(
            RuntimeError("history source unavailable") if case.history_source_fails else None
        ),
    )
    metadata = harness.parts["enricher"].metadata
    reasoner = GroundedReasoner(metadata)
    harness.controller._reasoning = ReasoningExecutor(reasoner)  # noqa: SLF001
    # Phase 2: the same reasoner drives task-scoped constraint eligibility, so a case that
    # declares a hard constraint is run *and measured* against one enforced constraint set.
    harness.controller._eligibility = CandidateEligibilityEvaluator(reasoner)  # noqa: SLF001
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
    print(
        f" cases: {len(EVALUATION_CASES)}   variants: "
        f"{ABLATION_ADAPTIVE}, {ABLATION_DECIDE_ONCE}, {ABLATION_MODEL_POLICY}"
    )
    print()

    reports: dict[str, Any] = {}
    variants: tuple[tuple[str, Any], ...] = (
        (ABLATION_ADAPTIVE, build_adaptive_policy),
        (ABLATION_DECIDE_ONCE, lambda case: DecidingOncePolicy(build_adaptive_policy(case))),
        # Phase 1: a model-driven policy, driven by a deterministic observation-reactive double so
        # the comparison stays offline and reproducible.
        (ABLATION_MODEL_POLICY, build_model_policy_factory()),
    )
    for variant, policy_for in variants:
        # The policy is built per case and handed in explicitly.  Letting the runner build it
        # instead would silently fall back to the adaptive default whenever a factory is not
        # wired through, which would mislabel the run rather than fail it - so the smoke always
        # passes the policy it means to measure.
        runner = CaseRunner(_factory, variant=variant, policy_factory=None)
        print(f" {variant}")
        outcomes = []
        for case in load_cases():
            outcome = runner.run(case, policy=policy_for(case), reasoner=_reasoner())
            outcomes.append(outcome)
            mark = "PASS" if outcome.passed else "FAIL"
            actions = " -> ".join(outcome.trajectory.action_sequence()) or "(none)"
            # The policy that actually decided, read back from the run: a variant label is a
            # claim, and printing the recorded policy is what makes the claim checkable.
            policies = ",".join(outcome.trajectory.policy_names()) or "unrecorded"
            print(
                f"   [{mark}] {case.case_id:32s} {outcome.trajectory.terminal.value:19s} "
                f"tools={outcome.trajectory.tool_calls} steps={outcome.trajectory.steps} "
                f"policy={policies}"
            )
            # Phase 2: state the two constraint questions separately, because reporting one
            # number would make "checked and compliant" look like "never checked".
            if case.hard_constraints:
                checked = ",".join(outcome.metrics.constraint_checked) or "none"
                print(
                    f"          constraints checked={checked} "
                    f"enforced={not outcome.metrics.constraint_not_enforced} "
                    f"unresolved_in_output={outcome.metrics.constraint_unresolved_in_output}"
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
                    "policies": list(outcome.trajectory.policy_names()),
                    "constraint_checked": list(outcome.metrics.constraint_checked),
                    "constraint_enforced": not outcome.metrics.constraint_not_enforced,
                    "constraint_violations": list(outcome.metrics.constraint_violations),
                    "constraint_unresolved_in_output": (
                        outcome.metrics.constraint_unresolved_in_output
                    ),
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
