"""Unified CLI runner for the multi-turn shopping Agent benchmark.

Examples:
    python -m benchmarks.agent.run --split dev --policies C --out runs/agent_benchmark/dev
    python -m benchmarks.agent.run --split dev --policies A,B,C,D
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .adapters import PolicyUnavailable, build_adapter
from .evaluator import evaluate_runs, load_human_labels
from .report import write_reports
from .schemas import BenchTask, PolicyId, TaskRun
from .tasks import (
    TASKS_DIR,
    TaskValidationError,
    load_tasks,
    validate_tasks,
    verify_manifest,
    write_manifest,
)

#: The fixed benchmark profile. Optional BGE/ItemCF enhancements are deliberately OFF.
FIXED_BENCHMARK_ENV = {
    "AGENTRECX_M7_GROUNDED_ACTIONS": "on",
    "AGENTRECX_M7_BUDGET": "2",
    "AGENTRECX_R1_DYNAMIC_RETRIEVAL": "on",
    "AGENTRECX_M8_TRAJECTORY": "on",
    "AGENTRECX_STATE_MODE": "shadow",
    "AGENTRECX_DENSE_RETRIEVAL": "off",
    "AGENTRECX_ITEMCF_RERANK": "off",
}


def _apply_profile(out_dir: Path, state_mode: str = "shadow", soft_m5: str = "off") -> None:
    for key, value in FIXED_BENCHMARK_ENV.items():
        os.environ[key] = value
    os.environ["AGENTRECX_STATE_MODE"] = state_mode
    os.environ["AGENTRECX_M5_SOFT_RERANK"] = soft_m5
    os.environ["AGENTRECX_STATE_DB"] = str(out_dir / "decision_state.sqlite3")


def _build_runtime(out_dir: Path) -> Any:
    """Compose the existing demo runtime once, using real accepted artifacts."""
    from recommendation.demo.runtime import build_demo_runtime

    return build_demo_runtime(
        memory_db=out_dir / "preference_memory.sqlite3",
        checkpoint_db=out_dir / "checkpoint.sqlite3",
    )


def _try_llm_clients() -> tuple[Any | None, Any | None, str | None]:
    """Return provider clients or an explicit unavailable reason; never fake one."""
    from recommendation.demo.llm_runtime import build_llm_client

    try:
        client = build_llm_client()
    except Exception as exc:  # noqa: BLE001 - unavailable is an expected benchmark state
        return None, None, f"llm_not_configured: {type(exc).__name__}"
    # One client is enough for the rewrite and proposal seams; each adapter owns its own
    # small consumer object, but no second provider is constructed.
    return client, client, None


def _task_hash(tasks: Sequence[BenchTask]) -> str:
    payload = json.dumps(
        [task.policy_input for task in tasks], ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_policies(value: str) -> list[PolicyId]:
    if value.strip().lower() in {"all", "a,b,c,d"}:
        return [PolicyId.A, PolicyId.B, PolicyId.C, PolicyId.D]
    result: list[PolicyId] = []
    for part in value.split(","):
        part = part.strip().upper()
        if not part:
            continue
        try:
            policy = PolicyId(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"unknown policy {part!r}") from exc
        if policy not in result:
            result.append(policy)
    if not result:
        raise argparse.ArgumentTypeError("at least one policy is required")
    return result


def run(
    *,
    split: str,
    policies: Sequence[PolicyId],
    out_dir: Path,
    tasks_dir: Path = TASKS_DIR,
    limit: int | None = None,
    task_ids: Sequence[str] = (),
    human_labels: Path | None = None,
    state_mode: str = "shadow",
    task_order: str = "forward",
    soft_m5: str = "off",
    proposal_client: Any | None = None,
) -> tuple[list[TaskRun], list[Any]]:
    """Execute the selected policies and write trajectories, metrics and reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_profile(out_dir, state_mode, soft_m5)
    verify_manifest(tasks_dir)
    tasks = list(load_tasks(split, tasks_dir=tasks_dir))
    runtime = _build_runtime(out_dir)
    records = runtime.metadata.records
    validate_tasks(
        tasks,
        records,
        expected_split=split,
        expected_counts={"dev": 20, "frozen": 8},
    )
    if task_ids:
        wanted = set(task_ids)
        tasks = [task for task in tasks if task.task_id in wanted]
    if limit is not None:
        tasks = tasks[: max(0, int(limit))]
    if task_order == "reverse":
        tasks = list(reversed(tasks))
    if not tasks:
        raise TaskValidationError("no tasks selected")
    service = runtime.agent_service()
    rewrite_client, env_proposal_client, llm_reason = _try_llm_clients()
    if proposal_client is None:
        proposal_client = env_proposal_client
    if proposal_client is not None:
        # An explicitly injected local model proposer is a real provider; it is not a
        # fallback and must not be reported as "llm unavailable".
        llm_reason = None
    started = time.perf_counter()
    runs: list[TaskRun] = []
    for policy in policies:
        adapter = build_adapter(
            policy,
            service=service,
            rewrite_client=rewrite_client,
            proposal_client=proposal_client,
            state_db_path=str(out_dir / "decision_state.sqlite3"),
        )
        policy_dir = out_dir / policy.value
        policy_dir.mkdir(parents=True, exist_ok=True)
        reason = adapter.availability()
        if reason:
            # Do not run a fallback or a simulated LLM path: record one unavailable run per
            # task so the report has an explicit denominator.
            for task in tasks:
                runs.append(
                    TaskRun(
                        task_id=task.task_id,
                        policy=policy,
                        runnable=False,
                        status="unavailable",
                        unavailable_reason=reason,
                    )
                )
            continue
        for task in tasks:
            task_dir = policy_dir / task.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            runs.append(adapter.run(task.policy_task, state_dir=str(task_dir)))

    evaluations = evaluate_runs(
        runs,
        tasks,
        records,
        human_labels=load_human_labels(human_labels),
    )
    elapsed = time.perf_counter() - started
    meta = {
        "split": split,
        "task_count": len(tasks),
        "task_hash": _task_hash(tasks),
        "state_mode": state_mode,
        "soft_m5": soft_m5,
        "task_order": task_order,
        "policy_arms": ",".join(policy.value for policy in policies),
        "llm_available": llm_reason is None,
        "llm_unavailable_reason": llm_reason or "",
        "fixed_profile": json.dumps(FIXED_BENCHMARK_ENV, sort_keys=True),
        "wall_seconds": round(elapsed, 3),
        "optional_bge": "off",
        "optional_itemcf": "off",
        "human_labels_file": str(human_labels) if human_labels else "none",
    }
    write_reports(out_dir, runs=runs, evaluations=evaluations, tasks=tasks, meta=meta)
    return runs, evaluations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "frozen"), default="dev")
    parser.add_argument("--policies", default="C", help="A,B,C,D or all")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tasks-dir", type=Path, default=TASKS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=[], dest="task_ids")
    parser.add_argument("--human-labels", type=Path, default=None)
    parser.add_argument(
        "--task-order",
        choices=("forward", "reverse"),
        default="forward",
        help="Run selected tasks in forward or reverse task order (isolation audit).",
    )
    parser.add_argument(
        "--state-mode",
        choices=("shadow", "apply_constraints"),
        default="shadow",
        help="M5 integration mode; apply_constraints enables the controlled M5->R1 arm.",
    )
    parser.add_argument(
        "--soft-m5",
        choices=("off", "on"),
        default="off",
        help="P1B-3 M5 soft-preference -> M6.1 rerank arm (default off).",
    )
    parser.add_argument(
        "--manifest",
        choices=("verify", "write"),
        default="verify",
        help="verify the frozen hash manifest (default) or deliberately regenerate it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.manifest == "write":
        path = write_manifest(args.tasks_dir)
        print(f"manifest written: {path}")
        return 0
    out_dir = args.out or (
        Path("runs") / "agent_benchmark" / args.split / time.strftime("%Y%m%d-%H%M%S")
    )
    try:
        policies = _parse_policies(args.policies)
        runs, evaluations = run(
            split=args.split,
            policies=policies,
            out_dir=out_dir,
            tasks_dir=args.tasks_dir,
            limit=args.limit,
            task_ids=tuple(args.task_ids),
            human_labels=args.human_labels,
            state_mode=args.state_mode,
            task_order=args.task_order,
            soft_m5=args.soft_m5,
        )
    except (TaskValidationError, RuntimeError) as exc:
        print(f"benchmark unavailable: {exc}", file=sys.stderr)
        return 2
    runnable = sum(1 for item in evaluations if item.runnable)
    passed = sum(1 for item in evaluations if item.passed)
    print(f"out: {out_dir}")
    print(f"evaluations: {len(evaluations)}; runnable: {runnable}; passed: {passed}")
    for policy in policies:
        rows = [item for item in evaluations if item.policy is policy]
        if rows:
            print(
                f"  {policy.value}: runnable={sum(1 for r in rows if r.runnable)} "
                f"completed={sum(1 for r in rows if r.completed)} "
                f"passed={sum(1 for r in rows if r.passed)}/{len(rows)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
