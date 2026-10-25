"""Phase 6B: fixed policy vs bounded dynamic product research, on a small scenario set.

This is a decision experiment, not a benchmark.  It reuses the existing Phase-4 synthetic
catalogue/control-plane fixture and runs the **same** scenario against:

* the fixed ``recommend once, then finish`` policy;
* the same inner policy wrapped by :class:`DynamicProductResearchPolicy`.

Everything is deterministic and offline: no external LLM, no model training, no service
start, no frozen evaluation artifact.  The result is written to a new run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    ConstraintOrigin,
    GroundedReasoner,
    LoopLimits,
    PolicyActionError,
    ReasoningExecutor,
    TaskState,
    TaskConstraint,
)
from recommendation.memory import PreferenceKind  # noqa: E402
from recommendation.demo.agent_policy import DemoSourcePlanPolicy  # noqa: E402
from recommendation.demo.research_policy import DynamicProductResearchPolicy  # noqa: E402
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.control_fixture import build_control_harness  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "p6b_research_eval"
REPETITIONS = 3
CAND_ID = re.compile(r"cand-[a-z]+")


def _fixed_policy() -> DemoSourcePlanPolicy:
    """The accepted fixed shopping policy: consult history once, then finish."""
    return DemoSourcePlanPolicy(("history",), default_k=3, use_memory=False)


class FailingResearch:
    """A controlled tool failure: the loop must observe it, not retry forever."""

    name = "failing-product-research"

    def available_actions(self) -> tuple[ActionKind, ...]:
        return (ActionKind.INSPECT_PRODUCT, ActionKind.COMPARE_PRODUCTS)

    def execute(self, action: Any, *, authorized_candidates: Any = None) -> Any:
        raise PolicyActionError("controlled research failure", code="research_failed")


def _price_constraint(limit: str = "30") -> TaskState:
    return TaskState().with_constraint(
        TaskConstraint(
            kind=PreferenceKind.PRICE_MAX,
            value=limit,
            origin=ConstraintOrigin.CURRENT_TURN,
            source_text=f"under {limit}",
        )
    )


SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "sufficient",
        "message": "recommend outdoor gear",
        "acceptable_terminal": "finished",
    },
    {
        "name": "explicit_compare",
        "message": "compare the first two by price and color",
        "acceptable_terminal": "finished",
    },
    {
        "name": "missing_material",
        "message": "is the material leather?",
        "acceptable_terminal": "waiting",
    },
    {
        "name": "compare_missing_material",
        "message": "compare the first two by material",
        "acceptable_terminal": "waiting",
    },
    {
        "name": "untrusted_identity_in_message",
        "message": "compare cand-green and cand-black by price",
        "acceptable_terminal": "finished",
    },
    {
        "name": "budget_exhausted",
        "message": "compare the first two by price and color",
        "acceptable_terminal": "finished",
        "limits": {"max_steps": 5, "max_tool_calls": 1, "max_retries": 1},
    },
    {
        "name": "research_failure",
        "message": "compare the first two by price and color",
        "acceptable_terminal": "finished",
        "failing_research": True,
    },
    {
        "name": "hard_constraint",
        "message": "compare the first two by price and color",
        "acceptable_terminal": "finished",
        "with_eligibility": True,
        "task_state": "price_max_30",
    },
)


def _limits(scenario: Mapping[str, Any]) -> LoopLimits:
    values = dict(scenario.get("limits") or {})
    return LoopLimits(
        max_steps=int(values.get("max_steps", 8)),
        max_tool_calls=int(values.get("max_tool_calls", 5)),
        max_retries=int(values.get("max_retries", 1)),
    )


def _run_one(scenario: Mapping[str, Any], policy_name: str) -> dict[str, Any]:
    harness = build_control_harness(
        policy=_fixed_policy(),
        limits=_limits(scenario),
        with_eligibility=bool(scenario.get("with_eligibility")),
    )
    reasoning: Any
    if scenario.get("failing_research"):
        reasoning = FailingResearch()
    else:
        reasoning = ReasoningExecutor(
            GroundedReasoner(harness.parts["enricher"].metadata)
        )
    harness.controller._reasoning = reasoning  # noqa: SLF001 - experiment wiring
    if policy_name == "dynamic":
        harness.controller._policy = DynamicProductResearchPolicy(  # noqa: SLF001
            harness.controller._policy
        )
    task_state = _price_constraint() if scenario.get("task_state") else None
    started = time.perf_counter()
    result = harness.controller.run(
        str(scenario["message"]),
        HISTORY,
        task_state=task_state,
        run_id=f"p6b:{scenario['name']}:{policy_name}",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    actions = tuple(result.trajectory.actions())
    observations = [
        step.observation
        for step in result.trajectory.steps
        if step.observation is not None
    ]
    research_observations = [
        observation for observation in observations
        if str(observation.get("kind", "")).startswith("product_")
    ]
    research_actions = tuple(
        action for action in actions if action in ("inspect_product", "compare_products")
    )
    waiting = result.control.termination_reason.value == "awaiting_user"
    final_ids = () if waiting else tuple(dict.fromkeys(CAND_ID.findall(result.final_response or "")))
    grounded_ids = tuple(
        item.parent_asin for item in (result.state.get("tool_result").recommendations or ())
    )
    eligibility = result.state.get("candidate_eligibility")
    hard_violations = 0
    unverified_presented = 0
    if eligibility is not None:
        for identity in final_ids:
            verdict = eligibility.eligibility_of(identity)
            value = getattr(verdict, "value", verdict)
            if value == "ineligible":
                hard_violations += 1
            elif value == "unresolved":
                unverified_presented += 1

    inspected = [
        step.action_proposal["arguments"]["parent_asin"]
        for step in result.trajectory.steps
        if step.action_proposal
        and step.action_proposal.get("action") == "inspect_product"
    ]
    duplicate_inspections = len(inspected) - len(set(inspected))
    invalid_research = 0
    for step in result.trajectory.steps:
        proposal = step.action_proposal or {}
        if proposal.get("action") == "inspect_product":
            target = proposal.get("arguments", {}).get("parent_asin")
            if target not in grounded_ids:
                invalid_research += 1
        if proposal.get("action") == "compare_products":
            targets = set(proposal.get("arguments", {}).get("parent_asins") or ())
            if not targets <= set(grounded_ids):
                invalid_research += 1

    fact_ok = _fact_check(scenario["name"], observations, result, grounded_ids)
    acceptable = (
        (not waiting and scenario["acceptable_terminal"] == "finished")
        or (waiting and scenario["acceptable_terminal"] == "waiting")
    )
    return {
        "scenario": scenario["name"],
        "policy": policy_name,
        "message": scenario["message"],
        "status": result.control.termination_reason.value,
        "waiting": waiting,
        "actions": list(actions),
        "research_actions": list(research_actions),
        "research_observations": [
            {
                "kind": observation.get("kind"),
                "tool": observation.get("tool"),
                "status": observation.get("status"),
                "products": list(observation.get("products") or ()),
                "parent_asin": observation.get("parent_asin"),
                "missing_attributes": list(observation.get("missing_attributes") or ()),
                "code": observation.get("code"),
            }
            for observation in research_observations
        ],
        "grounded_ids": list(grounded_ids),
        "final_ids": list(final_ids),
        "hard_constraint_violations": hard_violations,
        "unverified_presented": unverified_presented,
        "tool_calls": int(result.control.tool_call_count),
        "steps": int(result.control.step_count),
        "duplicate_inspections": duplicate_inspections,
        "invalid_research": invalid_research,
        "fact_check_ok": fact_ok,
        "acceptable_terminal": acceptable,
        "ms": round(elapsed_ms, 2),
    }


def _fact_check(
    name: str,
    observations: Sequence[Mapping[str, Any]],
    result: Any,
    grounded_ids: Sequence[str],
) -> bool | None:
    """A scenario-specific fact assertion; None means the scenario makes no fact claim."""
    by_kind = {str(item.get("kind")): item for item in observations}
    if name == "explicit_compare":
        row = next(
            (item for item in (by_kind.get("product_comparison") or {}).get("rows", ())
             if item.get("attribute") == "price"),
            None,
        )
        if row is None or row.get("comparison") is None:
            return False
        values = row["comparison"].get("values", {})
        return values.get("cand-red") == 25.0 and values.get("cand-blue") == 40.0
    if name in {"missing_material", "compare_missing_material"}:
        inspection = by_kind.get("product_inspection")
        return bool(inspection and "material" in (inspection.get("missing_attributes") or ()))
    if name == "untrusted_identity_in_message":
        proposals = [
            step.action_proposal
            for step in result.trajectory.steps
            if step.action_proposal
        ]
        for proposal in proposals:
            if proposal.get("action") == "compare_products":
                targets = set(proposal.get("arguments", {}).get("parent_asins") or ())
                return targets <= set(grounded_ids) and "cand-green" not in targets
        return False
    if name == "research_failure":
        return any(
            item.get("kind") == "product_research_failure"
            and item.get("status") == "failed"
            for item in observations
        )
    if name == "hard_constraint":
        return not any(
            item.get("kind") == "product_research_failure" for item in observations
        )
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))], 2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for policy in ("fixed", "dynamic"):
            for repetition in range(REPETITIONS):
                row = _run_one(scenario, policy)
                row["repetition"] = repetition
                rows.append(row)

    by_policy: dict[str, dict[str, Any]] = {}
    for policy in ("fixed", "dynamic"):
        selected = [row for row in rows if row["policy"] == policy]
        dynamic_selected = [row for row in selected if row["policy"] == "dynamic"]
        by_policy[policy] = {
            "runs": len(selected),
            "task_completion_rate": round(
                sum(1 for row in selected if not row["waiting"] and row["acceptable_terminal"])
                / len(selected), 4
            ),
            "acceptable_terminal_rate": round(
                sum(1 for row in selected if row["acceptable_terminal"]) / len(selected), 4
            ),
            "research_calls": sum(len(row["research_actions"]) for row in selected),
            "invalid_research": sum(row["invalid_research"] for row in selected),
            "duplicate_inspections": sum(row["duplicate_inspections"] for row in selected),
            "hard_constraint_violations": sum(
                row["hard_constraint_violations"] for row in selected
            ),
            "unverified_presented": sum(row["unverified_presented"] for row in selected),
            "mean_tool_calls": round(
                statistics.mean(row["tool_calls"] for row in selected), 3
            ),
            "p50_ms": _percentile([row["ms"] for row in selected], 0.50),
            "p95_ms": _percentile([row["ms"] for row in selected], 0.95),
            "research_fact_checks": {
                "checked": sum(1 for row in dynamic_selected if row["fact_check_ok"] is not None),
                "passed": sum(1 for row in dynamic_selected if row["fact_check_ok"] is True),
            },
        }

    # Research relevance and decision impact, compared scenario by scenario.
    fixed_by_scenario = {
        row["scenario"]: row for row in rows if row["policy"] == "fixed" and row["repetition"] == 0
    }
    dynamic_by_scenario = {
        row["scenario"]: row for row in rows if row["policy"] == "dynamic" and row["repetition"] == 0
    }
    impact = []
    for name, dynamic_row in dynamic_by_scenario.items():
        fixed_row = fixed_by_scenario[name]
        if dynamic_row["actions"] != fixed_row["actions"]:
            impact.append(
                {
                    "scenario": name,
                    "fixed_actions": fixed_row["actions"],
                    "dynamic_actions": dynamic_row["actions"],
                    "research_observation": dynamic_row["research_observations"],
                }
            )

    summary = {
        "task": "phase6b_dynamic_research",
        "policy_fixed": DemoSourcePlanPolicy.name,
        "policy_dynamic": DynamicProductResearchPolicy.name,
        "scenarios": [scenario["name"] for scenario in SCENARIOS],
        "repetitions": REPETITIONS,
        "by_policy": by_policy,
        "decision_impact": impact,
        "sample_example": dynamic_by_scenario.get("compare_missing_material"),
        "notes": {
            "external_llm": "none",
            "training": "none",
            "default_artifacts_modified": "none",
            "candidate_order": "unchanged; research is read-only",
            "task_completion": "completed only; waiting is reported separately as acceptable_terminal",
        },
    }
    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "scenarios": list(SCENARIOS),
                "repetitions": REPETITIONS,
                "driver_sha256": _sha256(Path(__file__)),
                "catalog": "tests.agent_reranking_fixture (synthetic Phase-4 catalogue)",
                "history": list(HISTORY),
                "output": str(args.out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        name: {"sha256": _sha256(args.out / name), "bytes": (args.out / name).stat().st_size}
        for name in ("config.json", "samples.jsonl", "summary.json")
    }
    (args.out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
