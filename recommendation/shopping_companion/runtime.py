"""One real Shopping Companion train-task run through the accepted Agent core.

The runtime is deliberately a *composition* of accepted components, not a new
agent:

    official train task (stage_1/stage_2, oracle-isolated)
      -> stage_1 planner loop (existing Action contract) -> Memory
      -> stage_2 planner loop -> real Product Search -> real Product Info
      -> CandidateLedger -> GroundedReasoner + Eligibility -> FINALIZE / ASK
      -> structured trajectory in runs/p14a/trajectories/

Every trajectory carries an explicit execution ``mode``.  The modes are never mixed:

* ``real_full_catalogue``  - not achieved by P14-A (the 20 GiB catalogue was not indexed);
* ``real_limited_catalogue`` - real catalogue sample, real tools, real ledger;
* ``contract_only``        - synthetic test fixture; never a benchmark result;
* ``oracle_diagnostic``    - explicit diagnostic that may use released oracle memories;
                             never counted as a normal Agent success.

The only component that reads the evaluator-only ground truth is
:mod:`recommendation.shopping_companion.evaluator`, which runs *after* the
trajectory is complete.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from recommendation.control.constraint_eligibility import ConstraintRequirement
from recommendation.control.grounded_reasoning import ConstraintKind
from recommendation.demo.planner_contract import PlannerObservation, run_planner

from .catalogue import ShoppingCompanionCatalogue
from .namespace import SHOPPING_COMPANION_NAMESPACE
from .planner_bridge import (
    BridgeConfig,
    P14ARulePlanner,
    P14AToolExecutor,
    PHASE_MEMORY_RETRIEVE,
    PHASE_PRODUCT_RETRIEVE,
    StageResult,
)
from .product_info import ShoppingCompanionProductInfoTool
from .search import ShoppingCompanionSearchTool

__all__ = [
    "EXECUTION_MODES",
    "RunConfig",
    "TaskInput",
    "BlockedMemoryTool",
    "ShoppingCompanionRuntime",
    "derive_constraints",
    "compose_product_query",
]

#: Trajectory mode labels (see module docstring; never mix them in one success claim).
EXECUTION_MODES: tuple[str, ...] = (
    "real_full_catalogue",
    "real_limited_catalogue",
    "contract_only",
    "oracle_diagnostic",
)

TRAJECTORY_SCHEMA = "agentrecx.p14a.trajectory.v1"

_PRICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?P<phrase>(?:under|below|less than|no more than|at most|maximum of|max(?:imum)?)\s*"
        r"\$?\s*(?P<value>\d+(?:\.\d+)?))",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<phrase>budget(?: of| is|:)?\s*\$?\s*(?P<value>\d+(?:\.\d+)?))", re.IGNORECASE),
    re.compile(r"(?P<phrase>\$\s*(?P<value>\d+(?:\.\d+)?)\s*(?:or less|max))", re.IGNORECASE),
)


class RuntimeModeError(ValueError):
    """An unknown execution mode (labels are part of the acceptance criteria)."""


class MemoryProtocol(Protocol):
    def search(self, query: str, *, scope: str = "long_term", k: int = 10) -> dict[str, Any]: ...

    def history_status(self) -> dict[str, Any]: ...


class BlockedMemoryTool:
    """The explicit "history unavailable" memory capability.

    Used when no released history corpus exists.  It never invents preferences and
    never reads the evaluator-only ground truth; it only reports the blocked state.
    """

    name = "p14a_history_unavailable"
    reason = (
        "the released Shopping Companion benchmark ships train/test parquet and products.jsonl "
        "only; the cross-session conversation histories required by the official memory service "
        "are not released, so no real history can be retrieved for this task"
    )

    def search(self, query: str, *, scope: str = "long_term", k: int = 10) -> dict[str, Any]:
        return {
            "available": False,
            "status": "history_unavailable",
            "preferences": [],
            "evidence": [],
            "conflicts": [],
            "reason": self.reason,
            "scope": scope,
            "query": query,
        }

    def history_status(self) -> dict[str, Any]:
        return {"available": False, "status": "history_unavailable", "reason": self.reason}


def derive_constraints(question: str) -> tuple[ConstraintRequirement, ...]:
    """Extract explicit numeric budget/price ceilings from the task question.

    Deterministic and conservative: only an explicitly stated ceiling becomes a hard
    constraint.  Absence of a statement means no constraint (never a guessed one).
    The most restrictive parsed ceiling wins (smallest value).
    """
    text = str(question or "")
    found: list[tuple[float, str]] = []
    for pattern in _PRICE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = float(match.group("value"))
            except (TypeError, ValueError):
                continue
            if value < 0:
                continue
            found.append((value, match.group("phrase").strip()))
    if not found:
        return ()
    value, phrase = min(found, key=lambda item: item[0])
    return (
        ConstraintRequirement(
            kind=ConstraintKind.PRICE_MAX,
            expected=f"{value:g}",
            source_text=phrase,
        ),
    )


def _preference_terms(preferences: Sequence[Mapping[str, Any]]) -> list[str]:
    terms: list[str] = []
    for preference in preferences:
        polarity = str(preference.get("polarity") or "").casefold()
        if polarity in {"negative", "dislike", "avoid", "exclude"}:
            continue
        value = preference.get("value")
        if isinstance(value, str) and value.strip():
            terms.append(value.strip())
        elif isinstance(value, (list, tuple)):
            terms.extend(str(item).strip() for item in value if str(item).strip())
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return seen[:8]


def compose_product_query(question: str, preferences: Sequence[Mapping[str, Any]]) -> str:
    """Compose the stage_2 search query from the question + **recovered** preferences.

    Only preferences actually recovered during stage_1 are appended; a blocked stage_1
    contributes nothing, so the query is never enriched with oracle/ground-truth data.
    """
    base = str(question or "").strip()
    text = re.sub(r"^Current Date:[^\n]*\n?", "", base, flags=re.IGNORECASE).strip()
    terms = _preference_terms(preferences)
    if not terms:
        return text
    return (text + " " + " ".join(terms)).strip()


@dataclass(frozen=True)
class RunConfig:
    """Configuration recorded verbatim in every trajectory."""

    mode: str = "real_limited_catalogue"
    search_limit: int = 10
    inspect_top_k: int = 5
    memory_k: int = 10
    max_steps_stage_1: int = 3
    max_steps_stage_2: int = 5
    catalogue_version: str | None = None
    #: When stage_1 ends awaiting user clarification, the official two-stage protocol
    #: has NOT produced the preference answer, so stage_2 must not run as the normal
    #: task result.  This opt-in flag runs stage_2 as an explicitly-labelled diagnostic
    #: continuation (never counted as a completed task).
    stage2_diagnostic_without_preferences: bool = False

    def __post_init__(self) -> None:
        if self.mode not in EXECUTION_MODES:
            raise RuntimeModeError(f"unknown execution mode {self.mode!r}; allowed: {EXECUTION_MODES}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "search_limit": self.search_limit,
            "inspect_top_k": self.inspect_top_k,
            "memory_k": self.memory_k,
            "max_steps_stage_1": self.max_steps_stage_1,
            "max_steps_stage_2": self.max_steps_stage_2,
            "catalogue_version": self.catalogue_version,
            "stage2_diagnostic_without_preferences": self.stage2_diagnostic_without_preferences,
        }


@dataclass(frozen=True)
class TaskInput:
    """The agent-visible part of one official train task (no eval fields)."""

    task_id: str
    conversation_id: str
    question: str
    question_date: str = ""
    question_type: str = "single_product"
    split: str = "train"
    source_index: int | None = None
    stage_1_user: str | None = None
    stage_2_user: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TaskInput":
        stage_1 = row.get("stage_1") or {}
        stage_2 = row.get("stage_2") or {}
        return cls(
            task_id=str(row.get("task_id") or row.get("question_id") or ""),
            conversation_id=str(row.get("conversation_id") or ""),
            question=str(row.get("question") or (stage_1.get("user") if isinstance(stage_1, Mapping) else "") or ""),
            question_date=str(row.get("question_date") or ""),
            question_type=str(row.get("question_type") or "single_product"),
            split=str(row.get("split") or "train"),
            source_index=row.get("source_index") if isinstance(row.get("source_index"), int) else None,
            stage_1_user=str(stage_1.get("user")) if isinstance(stage_1, Mapping) and stage_1.get("user") else None,
            stage_2_user=str(stage_2.get("user")) if isinstance(stage_2, Mapping) and stage_2.get("user") else None,
        )


class ShoppingCompanionRuntime:
    """Run one task through stage_1 + stage_2 and return a traceable trajectory."""

    def __init__(
        self,
        *,
        catalogue: ShoppingCompanionCatalogue,
        search_tool: ShoppingCompanionSearchTool,
        info_tool: ShoppingCompanionProductInfoTool,
        memory: MemoryProtocol | None,
        config: RunConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._catalogue = catalogue
        self._search_tool = search_tool
        self._info_tool = info_tool
        self._memory = memory
        self._config = config or RunConfig()
        self._clock = clock
        self._bridge_config = BridgeConfig(
            search_limit=self._config.search_limit,
            inspect_top_k=self._config.inspect_top_k,
            memory_k=self._config.memory_k,
        )

    @property
    def config(self) -> RunConfig:
        return self._config

    def run_task(self, task: TaskInput) -> dict[str, Any]:
        started = self._clock()
        if self._config.mode not in EXECUTION_MODES:
            raise RuntimeModeError(self._config.mode)
        memory = self._memory if self._memory is not None else BlockedMemoryTool()
        executor = P14AToolExecutor(
            catalogue=self._catalogue,
            search_tool=self._search_tool,
            info_tool=self._info_tool,
            memory=memory,
            config=self._bridge_config,
            clock=self._clock,
        )
        executor.allow_oracle_diagnostic(self._config.mode == "oracle_diagnostic")
        planner = P14ARulePlanner()

        # -- stage 1: memory / preference recovery -------------------------- #
        executor.reset_stage("stage_1")
        stage_1_started = self._clock()
        stage_1_query = str(task.stage_1_user or task.question)
        observation_1 = PlannerObservation(
            step=0,
            query=stage_1_query,
            m5_state={
                "stage": "stage_1",
                "phase": PHASE_MEMORY_RETRIEVE,
                "next_query": stage_1_query,
                "conversation_id": task.conversation_id,
            },
        )
        run_1 = run_planner(
            planner=planner,
            executor=executor,
            initial_observation=observation_1,
            max_steps=self._config.max_steps_stage_1,
        )
        stage_1_seconds = self._clock() - stage_1_started
        stage_1_result = executor.stage_results["stage_1"]
        preferences = [dict(item) for item in stage_1_result.preferences]

        # -- stage 2: real product recommendation --------------------------- #
        # Add-on-deal tasks carry a bundle voucher/budget; P14-A only validates the
        # schema and tool boundary for that task type, so no bundle-level constraint is
        # silently treated as a per-product ceiling and no bundle-construction success is
        # claimed.  Single-product tasks may carry an explicit per-product price ceiling.
        bundle_task = task.question_type == "add_on_deals"
        stage_1_awaiting = bool(run_1.termination == "ASK" or stage_1_result.status in {
            "history_unavailable",
            "clarification_pending",
        })
        stage_1_failed = stage_1_result.status == "tool_error"
        stage_2_diagnostic_only = bool(
            stage_1_awaiting and self._config.stage2_diagnostic_without_preferences
        )
        run_stage_2 = (not stage_1_awaiting) or stage_2_diagnostic_only
        constraints = () if bundle_task else derive_constraints(task.question)
        stage_2_started = self._clock()
        run_2 = None
        if run_stage_2:
            executor.reset_stage("stage_2", requirements=constraints)
            stage_2_query = compose_product_query(task.question, preferences)
            observation_2 = PlannerObservation(
                step=0,
                query=stage_2_query,
                m5_state={
                    "stage": "stage_2",
                    "phase": PHASE_PRODUCT_RETRIEVE,
                    "next_query": stage_2_query,
                    "preferences": preferences,
                    "constraints": [requirement.as_dict() for requirement in constraints],
                },
            )
            run_2 = run_planner(
                planner=planner,
                executor=executor,
                initial_observation=observation_2,
                max_steps=self._config.max_steps_stage_2,
            )
            stage_2_result = executor.stage_results["stage_2"]
            stage_2_query = stage_2_query
        else:
            stage_2_query = ""
            stage_2_result = StageResult(
                stage="stage_2",
                status="not_run",
                reason=(
                    "stage_1 ended awaiting user clarification; the official two-stage "
                    "protocol has no preference answer to hand to stage_2, so stage_2 was "
                    "not executed as the task result"
                ),
            )
        stage_2_seconds = self._clock() - stage_2_started
        total_seconds = self._clock() - started

        limitations: list[str] = []
        if bundle_task:
            # P14-A validates the add-on schema and tool boundary only.  Bundle/voucher
            # construction (multi-product compatibility, voucher thresholds/caps) is
            # P14-B/C work, so a flat add-on run must not be reported as a completed
            # bundle recommendation.
            limitations.append(
                "add_on_deals structural smoke only: bundle construction and voucher "
                "optimization are not implemented in P14-A"
            )
        if stage_1_awaiting:
            limitations.append(
                "stage_1 ended awaiting user clarification (missing history/answer); "
                "stage_2 was not executed as the task result"
            )
        if stage_2_diagnostic_only:
            limitations.append(
                "stage_2 executed only as an explicitly-labelled diagnostic continuation "
                "without a user answer; this run is not a completed task"
            )
        if stage_1_failed:
            limitations.append("stage_1 tool error disqualifies the run")

        succeeded = bool(
            run_stage_2
            and not stage_1_awaiting
            and stage_2_result.status == "finalized"
            and stage_2_result.final_ids
            and not stage_1_failed
            and not bundle_task
        )
        termination = (run_2.termination if run_2 is not None else "ASK") or "max_steps"
        if stage_1_failed:
            terminal_status = "tool_error"
            succeeded = False
            reason = stage_1_result.reason or "stage_1 tool error"
        elif stage_1_awaiting:
            terminal_status = "awaiting_clarification"
            succeeded = False
            reason = (
                "stage_1 requires a user answer (preference history unavailable in the "
                "released benchmark); no user answer was available, so the task is awaiting "
                "clarification rather than completed"
            )
            if stage_2_diagnostic_only:
                reason += "; stage_2 ran only as a labelled diagnostic continuation"
        elif bundle_task:
            terminal_status = "structural_smoke_completed"
            succeeded = False
            reason = (
                "add-on deal structural smoke completed; P14-A does not implement bundle "
                "construction or voucher optimization, so this is not a completed bundle recommendation"
            )
        else:
            terminal_status = "succeeded" if succeeded else "not_succeeded"
            reason = stage_2_result.reason or (
                "stage_2 finalized a verified recommendation"
                if succeeded
                else f"stage_2 terminated with {termination} ({stage_2_result.status})"
            )
        memory_status = memory.history_status()
        # Mode isolation: a real-catalogue run must never be fed synthetic-fixture or
        # oracle-diagnostic evidence (that would make a real-mode success claim unfounded).
        if self._config.mode in {"real_limited_catalogue", "real_full_catalogue"}:
            memory_status_name = str((memory_status or {}).get("status") or "")
            if memory_status_name in {"synthetic_fixture", "oracle_diagnostic"}:
                limitations.append(
                    f"mode isolation violation: {self._config.mode} run received "
                    f"{memory_status_name} memory evidence"
                )
                succeeded = False
                terminal_status = "mode_isolation_violation"
                reason = (
                    "the runtime received memory evidence whose source is not a real history; "
                    "this run cannot be reported as a real-catalogue Agent success"
                )
        return {
            "schema": TRAJECTORY_SCHEMA,
            "mode": self._config.mode,
            "traceable": True,
            "task": {
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
                "question": task.question,
                "question_date": task.question_date,
                "question_type": task.question_type,
                "split": task.split,
                "source_index": task.source_index,
            },
            "namespace": SHOPPING_COMPANION_NAMESPACE,
            "config": self._config.as_dict(),
            "catalogue": self._catalogue.as_dict(),
            "memory": {
                "status": memory_status,
                "stage_1_query": stage_1_query,
                "preferences": preferences,
                "evidence_ids": list(stage_1_result.evidence_ids),
            },
            "constraints": [requirement.as_dict() for requirement in constraints],
            "stages": {
                "stage_1": {
                    "result": stage_1_result.as_dict(),
                    "planner_run": run_1.as_dict(),
                    "seconds": round(stage_1_seconds, 4),
                },
                "stage_2": {
                    "executed": run_stage_2,
                    "diagnostic_only": stage_2_diagnostic_only,
                    "result": stage_2_result.as_dict(),
                    "planner_run": (
                        run_2.as_dict()
                        if run_2 is not None
                        else {"steps": [], "final_ids": [], "termination": "not_run"}
                    ),
                    "search_query": stage_2_query,
                    "seconds": round(stage_2_seconds, 4),
                },
            },
            "ledger": executor.ledger.as_dict(),
            "tool_calls": list(executor.tool_calls),
            "tool_call_count": len(executor.tool_calls),
            "recommendation": [dict(item) for item in stage_2_result.recommendations],
            "awaiting_clarification": stage_1_awaiting,
            "stage_2_executed": run_stage_2,
            "stage_2_diagnostic_only": stage_2_diagnostic_only,
            "bundle_structural_only": bundle_task,
            "limitations": list(limitations),
            "terminal": {
                "status": terminal_status,
                "succeeded": succeeded,
                "termination": termination,
                "reason": reason,
            },
            "timing": {
                "stage_1_seconds": round(stage_1_seconds, 4),
                "stage_2_seconds": round(stage_2_seconds, 4),
                "total_seconds": round(total_seconds, 4),
            },
        }
