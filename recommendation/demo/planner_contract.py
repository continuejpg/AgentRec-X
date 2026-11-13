"""Minimal opt-in Planner Action Contract for AgentRec-X P3D.

This is a contract/orchestration prototype, not a new recommender.  Actions are
validated and executed against the existing DemoAgentService (M5/R1/M6.1 and the
CandidateLedger are inside that service); the planner only chooses the next action.

The built-in ``StubPlanner`` is a deterministic CPU test double.  It is not Qwen and
must not be reported as model reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, Sequence


class PlannerAction(str, Enum):
    ASK = "ASK"
    RETRIEVE = "RETRIEVE"
    RERANK = "RERANK"
    FINALIZE = "FINALIZE"


@dataclass(frozen=True)
class PlannerObservation:
    step: int
    query: str
    candidate_ids: tuple[str, ...] = ()
    state_version: str | None = None
    needs_user_input: bool = False
    reranked: bool = False
    terminal: bool = False
    note: str = ""
    #: Dense, read-only context for an LLM planner.  It is never used as a candidate
    #: source and never bypasses role/price/catalogue validation.
    m5_state: dict[str, Any] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()
    last_tool_observation: dict[str, Any] = field(default_factory=dict)
    remaining_steps: int = 0
    candidate_evidence: tuple[dict[str, Any], ...] = ()
    last_retrieve_query: str = ""
    retrieval_feedback: dict[str, Any] = field(default_factory=dict)
    role_target: dict[str, Any] = field(default_factory=dict)
    eligibility: dict[str, Any] = field(default_factory=dict)
    source_policy: str = "balanced"
    source_feedback: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionProposal:
    action: PlannerAction
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


class PlannerContractError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def allowed_planner_actions(observation: PlannerObservation) -> tuple[str, ...]:
    if observation.terminal:
        return ()
    # ASK is only legal when there is no candidate set yet or the Gate/user has signalled
    # a clarification need.  Do not advertise actions that ``validate_action`` rejects.
    actions = [] if (observation.candidate_ids and not observation.needs_user_input) else [PlannerAction.ASK.value]
    retrieval_remaining = (observation.retrieval_feedback or {}).get("retrievals_remaining")
    # Do not advertise RETRIEVE after the bounded search budget is exhausted; the
    # executor raises tool_budget_exhausted for that action.
    if not observation.reranked and retrieval_remaining != 0:
        actions.append(PlannerAction.RETRIEVE.value)
    if observation.candidate_ids:
        actions.append(PlannerAction.FINALIZE.value)
        # RERANK is a one-shot inspection step: after it completes, repeating it cannot
        # add evidence, and RETRIEVE is illegal by the contract.
        if not observation.reranked:
            actions.append(PlannerAction.RERANK.value)
    return tuple(actions)


def validate_action_arguments(proposal: ActionProposal) -> None:
    if proposal.action is PlannerAction.ASK:
        if not str(proposal.arguments.get("question") or "").strip():
            raise PlannerContractError("ask_missing_question", "ASK requires a non-empty question")
    if proposal.action is PlannerAction.RETRIEVE:
        if not str(proposal.arguments.get("query") or "").strip():
            raise PlannerContractError("retrieve_missing_query", "RETRIEVE requires a non-empty query")
        policy = proposal.arguments.get("source_policy")
        if policy is not None:
            from recommendation.control.candidate_ledger import (
                SOURCE_POLICIES,
                normalize_source_policy,
            )

            try:
                normalized = normalize_source_policy(policy)
            except ValueError as exc:
                raise PlannerContractError(
                    "invalid_source_policy", str(exc)
                ) from exc
            if normalized not in SOURCE_POLICIES:  # defensive; normalize already checks
                raise PlannerContractError(
                    "invalid_source_policy",
                    f"unknown source_policy {policy!r}",
                )


def validate_action(
    proposal: ActionProposal,
    observation: PlannerObservation,
    *,
    step: int,
    max_steps: int,
) -> None:
    validate_action_arguments(proposal)
    if step > max_steps:
        raise PlannerContractError("max_steps_exceeded", f"step {step} > {max_steps}")
    if observation.terminal:
        raise PlannerContractError("terminal_observation", "no action after terminal observation")
    if proposal.action is PlannerAction.FINALIZE and not observation.candidate_ids:
        raise PlannerContractError("finalize_without_candidates", "cannot finalize an empty candidate set")
    if proposal.action is PlannerAction.RERANK and not observation.candidate_ids:
        raise PlannerContractError("rerank_without_candidates", "cannot rerank an empty candidate set")
    if proposal.action is PlannerAction.ASK and observation.candidate_ids and not observation.needs_user_input:
        raise PlannerContractError("ask_without_need", "ASK requires an ambiguity/need signal")
    if proposal.action is PlannerAction.RETRIEVE and observation.candidate_ids and observation.reranked:
        raise PlannerContractError("retrieve_after_rerank", "retrieve after a completed rerank is not legal")


class StubPlanner:
    """Deterministic CPU planner used only for the prototype trace."""

    def propose(self, observation: PlannerObservation) -> ActionProposal:
        if observation.terminal:
            raise PlannerContractError("stub_terminal", "stub planner received terminal observation")
        if observation.needs_user_input:
            return ActionProposal(PlannerAction.ASK, {"question": "Could you clarify the product type?"}, "ambiguity signal")
        if not observation.candidate_ids:
            return ActionProposal(PlannerAction.RETRIEVE, {"query": observation.query}, "no candidates yet")
        if not observation.reranked:
            return ActionProposal(PlannerAction.RERANK, {"candidates": list(observation.candidate_ids)}, "existing M6.1 rerank")
        return ActionProposal(PlannerAction.FINALIZE, {"candidates": list(observation.candidate_ids)}, "candidate set is ready")


@dataclass(frozen=True)
class PlannerStep:
    step: int
    action: str
    tool: str
    action_arguments: dict[str, Any]
    observation_before: dict[str, Any]
    tool_observation: dict[str, Any]
    observation_after: dict[str, Any]
    state_version: str | None
    terminal: bool


@dataclass
class PlannerRun:
    steps: list[PlannerStep] = field(default_factory=list)
    final_ids: tuple[str, ...] = ()
    termination: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "step": step.step,
                    "action": step.action,
                    "tool": step.tool,
                    "action_arguments": step.action_arguments,
                    "observation_before": step.observation_before,
                    "tool_observation": step.tool_observation,
                    "observation_after": step.observation_after,
                    "state_version": step.state_version,
                    "terminal": step.terminal,
                }
                for step in self.steps
            ],
            "final_ids": list(self.final_ids),
            "termination": self.termination,
        }


class PlannerExecutor(Protocol):
    def execute(self, proposal: ActionProposal, observation: PlannerObservation) -> tuple[PlannerObservation, dict[str, Any], bool]: ...


def run_planner(
    *,
    planner: Any,
    executor: PlannerExecutor,
    initial_observation: PlannerObservation,
    max_steps: int = 4,
) -> PlannerRun:
    observation = initial_observation
    run = PlannerRun()
    for step in range(1, max_steps + 1):
        observation = replace(
            observation,
            allowed_actions=allowed_planner_actions(observation),
            remaining_steps=max(0, max_steps - step),
        )
        proposal = planner.propose(observation)
        if proposal.action.value not in observation.allowed_actions:
            raise PlannerContractError("action_not_allowed", f"{proposal.action.value} is not allowed now")
        validate_action(proposal, observation, step=step, max_steps=max_steps)
        after, tool_observation, terminal = executor.execute(proposal, observation)
        after = replace(
            after,
            last_tool_observation=dict(tool_observation),
            remaining_steps=max(0, max_steps - step),
        )
        run.steps.append(
            PlannerStep(
                step=step,
                action=proposal.action.value,
                tool=proposal.action.value.lower(),
                action_arguments=dict(proposal.arguments),
                observation_before={
                    "step": observation.step,
                    "candidate_ids": list(observation.candidate_ids),
                    "state_version": observation.state_version,
                    "needs_user_input": observation.needs_user_input,
                    "reranked": observation.reranked,
                },
                tool_observation=tool_observation,
                observation_after={
                    "step": after.step,
                    "candidate_ids": list(after.candidate_ids),
                    "state_version": after.state_version,
                    "needs_user_input": after.needs_user_input,
                    "reranked": after.reranked,
                    "terminal": after.terminal,
                },
                state_version=after.state_version,
                terminal=terminal,
            )
        )
        observation = after
        if terminal:
            run.termination = proposal.action.value
            run.final_ids = tuple(after.candidate_ids)
            break
    else:
        run.termination = "max_steps"
        run.final_ids = tuple(observation.candidate_ids)
    return run
