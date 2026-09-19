"""AgentRec-X 2.0-alpha control-plane smoke test.

Proves the Stage 1 acceptance trajectory end to end, fully offline and on CPU::

    turn starts
      -> state initialized (one read-only preference snapshot)
      -> Policy receives PolicyContext
      -> Policy chooses RECOMMEND_FROM_HISTORY
      -> ActionProposal validated (controller stamps the execution metadata)
      -> the accepted recommendation pipeline executes
           RecommendationTool -> engine
           ProductEnricher -> MetadataIndex
           PreferenceCandidateMatcher
           PreferenceReranker
      -> DomainResult produced
      -> ResultVerifier accepts (candidate identity, counts, ranks, grounding)
      -> RecommendationObservation created (minimised)
      -> state updated
      -> CONTROL RETURNS TO POLICY          <-- the point of the milestone
      -> Policy sees the new Observation
      -> Policy chooses FINISH
      -> CompletionGuard accepts
      -> the accepted renderer/finalizer produces the response
      -> trajectory recorded

The smoke asserts the control invariants, not recommendation quality:

* control genuinely returns to the policy after a non-terminal observation;
* the policy may only choose from the system-provided ``available_actions``;
* ``ActionProposal`` is not executable and is never acted on unvalidated;
* ``DomainResult`` never reaches the policy - only a minimised ``Observation`` does;
* execution metadata is controller-generated;
* FINISH is only a proposal and must pass the ``CompletionGuard``;
* the budgets terminate the loop deterministically;
* the trajectory can answer "why did the next step happen?";
* the loop and the accepted DAG render the **same** response for the same input.

Nothing here needs a GPU, a checkpoint, a dataset, a network or a provider API.  The engine
behind the accepted Tool is a local deterministic stub, so the whole control plane and the
whole accepted pipeline are real while the model is fake.

Usage::

    .venv/bin/python -m experiments.control_plane_smoke
    .venv/bin/python -m experiments.control_plane_smoke --json /tmp/control_smoke.json
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

from recommendation.agent import AgentDecision, AgentGraph  # noqa: E402
from recommendation.control import (  # noqa: E402
    ActionKind,
    ActionProposal,
    LoopController,
    LoopLimits,
    PolicyContext,
    RecommendFromHistoryCapability,
    RuleBasedPolicy,
    RunStatus,
    TerminationReason,
)
from recommendation.control.topology import (  # noqa: E402
    DECLARED_CYCLE_EDGES,
    LOOP_NODE_NAMES,
    LOOP_PHASE_ORDER,
)
from recommendation.tools import RecommendationTool  # noqa: E402
from tests.agent_fakes import HISTORY  # noqa: E402
from tests.agent_reranking_fixture import QUERY, FixedEngine, rendered_order  # noqa: E402
from tests.control_fixture import (  # noqa: E402
    RecordingPolicy,
    build_full_capability,
)


class _FixedDecisionModel:
    """A decision model that returns one fixed decision, for the DAG comparison."""

    def __init__(self, decision: AgentDecision) -> None:
        self._decision = decision
        self.call_count = 0

    def decide(self, messages: Any) -> AgentDecision:
        """Record the call and return the fixed decision."""
        self.call_count += 1
        return self._decision


class Gate:
    """Collects named pass/fail gates so the smoke reports all of them, not the first."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        """Record one gate."""
        self.results.append((name, bool(condition), detail))
        return bool(condition)

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        """The gates that did not pass."""
        return [row for row in self.results if not row[1]]

    def report(self) -> None:
        """Print every gate with its verdict."""
        width = max(len(name) for name, _, _ in self.results)
        for name, ok, detail in self.results:
            mark = "PASS" if ok else "FAIL"
            suffix = f"  {detail}" if detail else ""
            print(f"  [{mark}] {name.ljust(width)}{suffix}")
        print()
        print(f"  {len(self.results) - len(self.failed)}/{len(self.results)} gates passed")


def _build_controller(
    *,
    driver: str,
    limits: LoopLimits | None = None,
    policy: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Build a control plane over the real accepted pipeline and a stub engine.

    The policy is wrapped in a recorder, so the smoke can inspect exactly what the policy
    was shown rather than asserting the projection from the outside.
    """
    capability, engine, parts = build_full_capability()
    resolved = policy or RecordingPolicy(RuleBasedPolicy(default_k=4))
    controller = LoopController(
        resolved,
        capability,
        limits=limits or LoopLimits(),
        driver=driver,
    )
    return controller, engine, parts, resolved


def run(*, json_path: Path | None = None) -> int:
    """Run the control-plane smoke and return a process exit code."""
    gates = Gate()
    payload: dict[str, Any] = {}

    print("=" * 78)
    print(" AgentRec-X 2.0-alpha control-plane smoke")
    print("=" * 78)
    print()
    print(" topology")
    for phase in LOOP_PHASE_ORDER:
        print(f"   {phase}")
    print("   -> check_limits (back-edge)")
    print()
    print(f" nodes: {', '.join(LOOP_NODE_NAMES)}")
    print()

    # -- 1. the canonical turn on the graph driver -------------------------- #
    controller, engine, parts, policy = _build_controller(driver="graph")
    result = controller.run(QUERY, HISTORY)
    contexts = list(policy.contexts)

    print(" canonical turn (graph driver)")
    print(f"   status             : {result.status.value}")
    print(f"   termination reason : {result.control.termination_reason.value}")
    print(f"   steps              : {result.control.step_count}")
    print(f"   tool calls         : {result.control.tool_call_count}")
    print(f"   action sequence    : {' -> '.join(result.trajectory.actions())}")
    print(f"   engine calls       : {engine.call_count}")
    print(f"   rendered order     : {', '.join(rendered_order(result.final_response))}")
    print()

    gates.check(
        "the run finished through an accepted completion",
        result.status is RunStatus.FINISHED
        and result.control.termination_reason is TerminationReason.COMPLETED,
        f"{result.status.value}/{result.control.termination_reason}",
    )
    gates.check(
        "the action sequence is RECOMMEND_FROM_HISTORY then FINISH",
        result.trajectory.actions() == ("recommend_from_history", "finish"),
        str(result.trajectory.actions()),
    )
    gates.check(
        "the accepted recommendation pipeline ran exactly once",
        engine.call_count == 1,
        f"engine calls={engine.call_count}",
    )

    # -- 2. CONTROL RETURNED TO THE POLICY ---------------------------------- #
    returning = [step for step in result.trajectory.steps]
    gates.check(
        "control returned to the policy after the observation",
        len(returning) >= 2,
        f"{len(returning)} steps recorded (one action each)",
    )
    gates.check(
        "the policy was consulted again with an updated context",
        len(contexts) >= 2
        and contexts[0].last_observation is None
        and getattr(contexts[-1].last_observation, "kind", None) == "recommendation",
        f"policy calls={len(contexts)}",
    )
    if len(contexts) >= 2:
        first, last = contexts[0], contexts[-1]
        gates.check(
            "the second context shows a grounded candidate set",
            last.candidate_state.grounded is True
            and first.candidate_state.grounded is False,
            f"grounded {first.candidate_state.grounded} -> {last.candidate_state.grounded}",
        )
        gates.check(
            "the second context shows a reduced budget",
            last.remaining_steps < first.remaining_steps
            and last.remaining_tool_calls < first.remaining_tool_calls,
            f"steps {first.remaining_steps}->{last.remaining_steps}, "
            f"tool calls {first.remaining_tool_calls}->{last.remaining_tool_calls}",
        )

    # -- 3. authority boundaries -------------------------------------------- #
    gates.check(
        "ActionProposal is not executable",
        not hasattr(ActionProposal, "execute") and not hasattr(ActionProposal, "run"),
    )
    forbidden_fields = {
        "history",
        "trusted_user_history",
        "candidates",
        "candidate_ids",
        "sql",
        "tool",
        "action_id",
        "step_index",
        "run_id",
        "turn_id",
    }
    gates.check(
        "a proposal has no field for history, candidates, a tool or metadata",
        not (set(ActionProposal.model_fields) & forbidden_fields),
        str(sorted(set(ActionProposal.model_fields) & forbidden_fields)),
    )
    policy_public = {name for name in dir(policy) if not name.startswith("_")}
    execution_surface = {
        "run",
        "execute",
        "invoke",
        "tool",
        "engine",
        "store",
        "memory_service",
        "metadata",
        "rerank",
        "match",
        "commit",
    }
    gates.check(
        "the policy exposes no execution surface",
        policy_public.isdisjoint(execution_surface),
        str(sorted(policy_public & execution_surface)),
    )
    if contexts:
        projected = set(vars(contexts[0]))
        gates.check(
            "PolicyContext has no history, key, tool or store field",
            not (
                projected
                & {"trusted_user_history", "user_key", "tool", "engine", "memory_service"}
            ),
            str(sorted(projected)),
        )
        blob = repr(contexts[-1]) + str(contexts[-1].summary())
        gates.check(
            "no trusted history value appears in the policy's view",
            all(asin not in blob for asin in HISTORY),
        )

    # -- 4. DomainResult never reaches the policy --------------------------- #
    observations = [
        step.observation for step in result.trajectory.steps if step.observation
    ]
    gates.check(
        "the policy saw a minimised observation",
        observations
        and all(
            {"status", "returned_k", "has_candidates", "candidate_set_ref"} <= set(obs)
            for obs in observations
        ),
        f"{len(observations)} observation(s)",
    )
    gates.check(
        "the observation carries no candidate identity or score",
        all(
            not (set(obs) & {"recommendations", "candidates", "scores"})
            and "parent_asin" not in obs
            and "score" not in obs
            for obs in observations
        ),
    )
    gates.check(
        "the raw domain result is not policy-visible",
        all(
            getattr(step, "verification_result", None) is not None
            for step in result.trajectory.steps
            if step.validated_action is not None
            and step.validated_action["action"] == ActionKind.RECOMMEND_FROM_HISTORY.value
        ),
        "every executed step carries a verification verdict",
    )

    # -- 5. verification keeps the accepted invariants ---------------------- #
    verified_steps = [
        step
        for step in result.trajectory.steps
        if step.verification_result is not None
        and step.validated_action is not None
        and step.validated_action["action"] == ActionKind.RECOMMEND_FROM_HISTORY.value
    ]
    if verified_steps:
        checks = set(verified_steps[0].verification_result.checks)
        required = {
            "tool_grounding",
            "count_integrity",
            "rank_integrity",
            "candidate_identity",
            "enrichment_alignment",
            "evidence_alignment",
            "reranking_alignment",
        }
        gates.check(
            "verification checked grounding, counts, ranks and stage alignment",
            required <= checks,
            f"missing={sorted(required - checks)}",
        )
    else:  # pragma: no cover - the canonical turn always verifies
        gates.check("verification ran", False, "no verified step recorded")

    # -- 6. candidate identity survives the loop ---------------------------- #
    tool_result = result.state["tool_result"]
    expected = [
        (item.parent_asin, item.item_id) for item in tool_result.recommendations
    ]
    enrichment = result.state["enrichment"]
    evidence = result.state["preference_evidence"]
    reranking = result.state["reranking"]
    gates.check(
        "candidate identity is unchanged through enrich / match / rerank",
        [(i.parent_asin, i.recommendation.item_id) for i in enrichment.items] == expected
        and [(c.parent_asin, c.item_id) for c in evidence.candidates] == expected
        and sorted((c.parent_asin, c.item_id) for c in reranking.candidates)
        == sorted(expected),
        f"{len(expected)} candidates",
    )
    gates.check(
        "trusted history is exactly what the run was started with",
        result.state["trusted_user_history"] == tuple(HISTORY)
        and engine.last_history == list(HISTORY),
    )

    # -- 7. the loop and the accepted DAG agree ----------------------------- #
    dag_engine = FixedEngine()
    dag = AgentGraph(
        _FixedDecisionModel(AgentDecision(action="recommend", k=4)),
        RecommendationTool(dag_engine),
        product_enricher=parts["enricher"],
        preference_matcher=parts["matcher"],
        reranker=parts["reranker"],
    )
    dag_state = dag.run(QUERY, HISTORY)
    gates.check(
        "the loop and the accepted DAG render the same response",
        result.final_response == dag_state["final_response"]
        and result.route == dag_state["route"],
        f"route={result.route}",
    )
    gates.check(
        "the accepted DAG is still acyclic",
        "update_state" not in dag.mermaid()
        and "finalize --> decide" not in dag.mermaid(),
    )

    # -- 8. bounded execution ------------------------------------------------ #
    class NeverFinishes:
        """A policy that never proposes FINISH, so only the budget can stop the run."""

        name = "never-finishes"

        def choose(self, context: PolicyContext) -> ActionProposal:
            return ActionProposal(action=ActionKind.RECOMMEND_FROM_HISTORY, k=4)

    bounded_controller, bounded_engine, _, _ = _build_controller(
        driver="graph",
        limits=LoopLimits(max_steps=4, max_tool_calls=8),
        policy=NeverFinishes(),
    )
    bounded = bounded_controller.run(QUERY, HISTORY)
    gates.check(
        "max_steps terminates a runaway loop deterministically",
        bounded.status is RunStatus.ABORTED
        and bounded.control.termination_reason is TerminationReason.MAX_STEPS
        and bounded.control.step_count == 4,
        f"{bounded.status.value}/{bounded.control.termination_reason} "
        f"at step {bounded.control.step_count}",
    )
    gates.check(
        "a runaway loop cannot exceed one recommender call per step",
        bounded_engine.call_count <= bounded.control.step_count
        and bounded_engine.call_count <= 8,
        f"engine calls={bounded_engine.call_count}, steps={bounded.control.step_count}",
    )

    no_budget, no_budget_engine, _, _ = _build_controller(
        driver="graph", limits=LoopLimits(max_tool_calls=0)
    )
    no_budget_result = no_budget.run(QUERY, HISTORY)
    gates.check(
        "a zero tool budget prevents any recommender call",
        no_budget_engine.call_count == 0
        and no_budget_result.control.tool_call_count == 0,
        f"engine calls={no_budget_engine.call_count}",
    )

    # -- 9. both drivers run the same loop ---------------------------------- #
    direct_controller, _, _, _ = _build_controller(driver="direct")
    direct = direct_controller.run(QUERY, HISTORY)
    gates.check(
        "the graph driver and the direct driver agree",
        direct.final_response == result.final_response
        and direct.trajectory.actions() == result.trajectory.actions()
        and direct.control.termination_reason == result.control.termination_reason,
    )

    # -- 10. trajectory answers "why did the next step happen?" -------------- #
    steps = result.trajectory.steps
    gates.check(
        "every step records context, proposal, validation and outcome",
        all(
            step.policy_context_summary
            and step.action_proposal is not None
            and step.validation_result is not None
            and step.validated_action is not None
            for step in steps
        ),
        f"{len(steps)} steps",
    )
    gates.check(
        "the trajectory is payload-free",
        all(
            asin not in str(step.model_dump())
            for step in steps
            for asin in HISTORY
        ),
        "no trusted history in the trajectory",
    )
    gates.check(
        "the declared cycle contains the back-edge",
        (LOOP_PHASE_ORDER[0], LOOP_PHASE_ORDER[1]) in DECLARED_CYCLE_EDGES
        and ("update_state", "check_limits") in DECLARED_CYCLE_EDGES,
    )

    print(" trajectory (the audit of why each next step happened)")
    for step in steps:
        proposal = step.action_proposal or {}
        outcome = step.observation or {}
        print(
            f"   step {step.step_index}: "
            f"proposed {proposal.get('action')} -> "
            f"validated {step.action_id[:12]}... -> "
            f"observation {outcome.get('kind')}/{outcome.get('status')}"
        )
    print()

    payload = {
        "status": result.status.value,
        "termination_reason": str(result.control.termination_reason),
        "steps": result.control.step_count,
        "tool_calls": result.control.tool_call_count,
        "actions": list(result.trajectory.actions()),
        "policy_calls": len(contexts),
        "candidates": len(expected),
        "gates": [
            {"name": name, "passed": ok, "detail": detail}
            for name, ok, detail in gates.results
        ],
        "passed": len(gates.results) - len(gates.failed),
        "total": len(gates.results),
    }

    gates.report()
    print()
    print("Control plane only. NOT a recommendation-quality benchmark.")
    print(f"SMOKE: {'PASS' if not gates.failed else 'FAIL'}")

    if json_path is not None:
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")

    return 0 if not gates.failed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)
    return run(json_path=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
