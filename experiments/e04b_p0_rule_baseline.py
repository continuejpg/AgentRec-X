"""E-04B P0: reproducible offline behaviour baseline for the B2 rule-based policy.

What this measures, and what it does not
---------------------------------------
B2 is the **deterministic tool-selection policy** (`evaluation/agent/runner.py` policy library, driven
through the accepted `LoopController`).  This script runs the repository's own controlled case
registry under that policy and records *behaviour*: which sources were planned and executed, how many
tool calls and steps were spent, what the terminal reason was, whether the declared budgets and
legality declarations held, and whether the run's own metrics agree with its trajectory.

It does **not** measure recommendation quality: the catalogue here is the synthetic fixture the
accepted agent-plane tests use, there are no neural weights and no provider calls, so no HR/NDCG,
no task-success rate and no "agent beats fixed policy" claim can come out of this run.  Every number
below is a **new measurement of behaviour on synthetic fixtures**, and is labelled as such.

Usage
-----
    python -m experiments.e04b_p0_rule_baseline --out runs/e04b_p0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Frozen identities carried into the result file so a later run can be compared like for like.
PROTOCOL_NAME = "agentrecx.eval_protocol.v1"
SEED = 20261015
VARIANT = "adaptive"
ARM = "B2.rule_tool_selection"

#: Provenance labels; every record in the output states which of these it is.
DATA_KIND = "synthetic_fixture"      # accepted agent-plane fixture catalogue, no real interactions
MEASUREMENT_KIND = "new_execution"   # measured in this run, not replayed from an archive
#: Never claimed by this script (no labels for them exist):
NOT_CLAIMED = (
    "task_success_rate",
    "preference_satisfaction_rate",
    "hr_at_k",
    "ndcg_at_k",
    "agent_beats_fixed_policy",
)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    """A rate with its own numerator and denominator, so no reader has to guess the base."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def compute_behaviour_metrics(outcomes: Sequence[Any]) -> dict[str, Any]:
    """The nine frozen behaviour metrics, each with an explicit numerator and denominator."""
    cases = list(outcomes)
    total = len(cases)
    metrics: dict[str, Any] = {"cases": total}

    def field(outcome: Any, name: str, default: Any = 0) -> Any:
        value = getattr(getattr(outcome, "metrics", None), name, default)
        return default if value is None else value

    # 1. Declared action-contract compliance, NOT a per-proposal validator rate.  The accepted
    #    metric is `tool_selection_correct = not required_missing and not forbidden_used`
    #    (recommendation/evaluation/agent/metrics.py:260): the run must have performed every action
    #    the case declared as required and no action it declared forbidden.  It says nothing about
    #    whether the chosen tool was the *most appropriate* one - that would need an annotation this
    #    repository does not have.
    compliant = sum(1 for outcome in cases if bool(field(outcome, "tool_selection_correct", False)))
    metrics["declared_action_contract_compliance"] = _rate(compliant, total)

    # 2. tool calls, 3. executed sources, 4. failures/empty results, 5. grounded candidates
    tool_calls = sum(int(field(outcome, "tool_calls", 0)) for outcome in cases)
    metrics["tool_calls_total"] = tool_calls
    metrics["tool_calls_per_case"] = round(tool_calls / total, 4) if total else None
    # The accepted outcome object does not expose a per-source call breakdown, so the executed set
    # is reported as not-exposed rather than reconstructed by guessing; the declared expectation is
    # checked separately by ``declared_expectation_match``.
    metrics["executed_sources"] = {"status": "not_exposed_by_harness",
                                   "checked_instead": "required_actions_met / forbidden_actions_used"}
    # No failure/empty counters exist on the accepted metrics object; reporting them as 0 would be a
    # phantom zero (the default of a missing attribute), so they are reported as not-exposed.
    metrics["tool_failures"] = {"status": "not_exposed_by_harness",
                                "checked_instead": "case declarations (history_source_fails, "
                                                   "recovery-required) plus the recorded terminal"}
    metrics["tool_empty_results"] = {"status": "not_exposed_by_harness",
                                     "checked_instead": "the empty-source cases' recorded "
                                                        "tool_calls/steps/terminal"}
    metrics["grounded_candidates_total"] = {"status": "not_exposed_by_harness",
                                            "checked_instead": "hallucinated_candidates == 0"}

    # 6. budget compliance: the run must stay inside the case's declared limits
    within = sum(1 for outcome in cases if bool(getattr(outcome, "within_budget", True)))
    metrics["budget_compliance"] = _rate(within, total)

    # 7. terminal distribution
    terminals: dict[str, int] = {}
    for outcome in cases:
        key = str(field(outcome, "terminal_outcome", "unknown"))
        terminals[key] = terminals.get(key, 0) + 1
    metrics["terminal_distribution"] = terminals

    # 8. rule decision vs executed trajectory: the declared expectations must match the run
    matched = sum(
        1
        for outcome in cases
        if bool(field(outcome, "required_actions_met", False))
        and not bool(field(outcome, "forbidden_actions_used", False))
    )
    metrics["declared_expectation_match"] = _rate(matched, total)

    # 9. waste and constraint hygiene, reported as raw counts (no quality claim attached)
    metrics["unnecessary_tool_calls"] = sum(
        int(field(outcome, "unnecessary_tool_calls", 0)) for outcome in cases
    )
    metrics["constraint_violations"] = sum(
        len(field(outcome, "constraint_violations", ()) or ()) for outcome in cases
    )
    metrics["constraint_not_enforced"] = sum(
        int(field(outcome, "constraint_not_enforced", 0)) for outcome in cases
    )
    metrics["hallucinated_candidates"] = sum(
        int(field(outcome, "hallucinated_candidates", 0)) for outcome in cases
    )
    metrics["not_claimed"] = list(NOT_CLAIMED)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-04B P0: B2 rule-policy behaviour baseline")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "e04b_p0")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from recommendation.evaluation.agent.cases import load_cases
    from recommendation.evaluation.agent.runner import run_suite
    from recommendation.evaluation.agent.runner import build_adaptive_policy

    # The accepted, already-tested controller factory and reasoner live in the agent-plane test
    # module; P0 reuses them rather than rebuilding a second harness.
    from tests.test_evaluation_agent_plane import build_reasoner, suite_factory

    # Protocol checks (frozen E-04B-P rules) before any measurement.
    from experiments.e04b_protocol_v2 import (
        ARMS,
        assert_same_case_ids,
        metric_is_computable,
    )

    cases = load_cases()
    case_ids = [case.case_id for case in cases]
    assert_same_case_ids({ARM: case_ids, "B2.repeat": case_ids})   # same ids across B2 instances
    assert metric_is_computable("tool_call_count", []) is True
    assert metric_is_computable("hr_at_k", []) is False, "no labels here: HR is not computable"
    budget = ARMS[ARM]["budget"]

    started = time.perf_counter()
    report = run_suite(
        cases,
        controller_factory=suite_factory,
        variant=VARIANT,
        policy_factory=build_adaptive_policy,   # deterministic rules; no model client is built
        reasoner_factory=build_reasoner,
    )
    wall = time.perf_counter() - started

    outcomes = list(getattr(report, "outcomes", ()) or ())
    metrics = compute_behaviour_metrics(outcomes)
    rows = []
    for outcome in outcomes:
        case = getattr(outcome, "case", None)
        rows.append({
            "case_id": getattr(case, "case_id", None),
            "category": getattr(case, "category", None),
            "message_kind": "declared case text" if getattr(case, "message", None) else None,
            "expected_sources": list(getattr(case, "expected_sources", ()) or ()),
            "acceptable_actions": list(getattr(case, "acceptable_actions", ()) or ()),
            "forbidden_actions": list(getattr(case, "forbidden_actions", ()) or ()),
            "max_steps": getattr(case, "max_steps", None),
            "max_tool_calls": getattr(case, "max_tool_calls", None),
            "tool_calls": getattr(getattr(outcome, "metrics", None), "tool_calls", None),
            "steps": getattr(getattr(outcome, "metrics", None), "steps", None),
            "terminal_outcome": str(
                getattr(getattr(outcome, "metrics", None), "terminal_outcome", "")
            ),
            "passed": bool(getattr(outcome, "passed", False)),
        })

    payload = {
        "arm": ARM,
        "protocol_name": PROTOCOL_NAME,
        "variant": VARIANT,
        "seed": SEED,
        "case_count": len(cases),
        "case_ids": case_ids,
        "source_menu": ["history", "catalog_search", "similar_item"],
        "declared_budget": budget,
        "llm_calls": 0,
        "data_kind": DATA_KIND,
        "measurement_kind": MEASUREMENT_KIND,
        "wall_seconds": round(wall, 2),
        "metrics": metrics,
        "cases": rows,
    }
    (args.out / "p0_baseline.json").write_text(json.dumps(payload, indent=2, default=str),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("cases", "case_ids")}, indent=2,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
