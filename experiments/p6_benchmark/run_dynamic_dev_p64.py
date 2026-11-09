"""P6.4 controlled dynamic-DEV runner.

Fixed scope:
* six existing P6 dynamic DEV tasks, re-expressed in natural-language shopping turns
  (``data/p6_benchmark_v0/dynamic_dev_p64.jsonl``);
* one explicit initial history and initial M5 state shared by every task/arm;
* the authoritative M5 hard-constraint state is advanced by the runner's own
  deterministic state-transition table and passed to every arm through the trusted
  ``constraints`` request field, so no Planner is scored on parsing the turn text;
* the real Serving V1 Fusion / DemoAgentService / M5 / CandidateLedger / Eligibility /
  role rules are reused unchanged; scoring uses the final presented response only.

The run is labelled ``planner_only_controlled_evaluation``: natural-language turns are
still delivered to the Agent, but parser understanding is not part of the Planner score.
Expected state is used only for execution control and evaluation; no Gold product
identity is ever passed to the Agent.

This file is intentionally read-only except for writing its run artifacts.  It does not
run the Holdout split, Qwen, training or the GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TASKS = REPO / "data" / "p6_benchmark_v0" / "dynamic_dev_p64.jsonl"
PRODUCTS = REPO / "data" / "processed" / "Sports_and_Outdoors_products.jsonl"
CHECKPOINT = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
OUT_ROOT = REPO / "runs" / "p6_4_dynamic"
PROFILE_ID = "p64-fixed"
K = 5
SOURCES = ("history", "catalog_search")
STATE_MODE = "apply_constraints"
SERVING_FUSION = "rrf3"

from experiments.p6_benchmark.run_dynamic_dev import (  # noqa: E402
    _price_from_constraint_trace,
    _price_of,
    _role_target_from_diagnostics,
    catalogue_evidence,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = base.with_name(f"{base.name}_r{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free directory next to {base}")


def load_tasks(path: Path, only: Sequence[str] = ()) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tasks) != 6 or any(t.get("split") != "dev" for t in tasks):
        raise ValueError("P6.4 runs exactly the six dynamic DEV tasks")
    if only:
        wanted = set(only)
        tasks = [t for t in tasks if t["task_id"] in wanted or t["template_id"] in wanted]
    if not tasks:
        raise ValueError(f"no task matched {list(only)}")
    return tasks


def apply_operations(state: Mapping[str, Any], operations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the task's deterministic shared-state transition (planner-independent)."""
    updated = copy.deepcopy(dict(state))
    for operation in operations or ():
        op = str(operation.get("op"))
        if op == "set_goal":
            updated["goal_type"] = str(operation["goal_type"])
            updated["role"] = str(operation.get("role", "main"))
        elif op == "set_role":
            updated["role"] = str(operation["role"])
        elif op == "set_price_max":
            updated["price_max"] = float(operation["value"])
        elif op == "remove_price_max":
            updated["price_max"] = None
        elif op == "add_preference":
            updated.setdefault("preferences", []).append(
                {
                    "kind": str(operation.get("kind", "feature")),
                    "value": str(operation["value"]),
                    "polarity": str(operation.get("polarity", "prefer")),
                }
            )
        elif op == "retract_preference":
            value = str(operation.get("value", "")).casefold()
            updated["preferences"] = [
                item for item in updated.get("preferences", ())
                if value not in str(item.get("value", "")).casefold()
            ]
        else:  # pragma: no cover - the frozen task file only uses the operations above
            raise ValueError(f"unknown state operation {op!r}")
    return updated


def _lighter_active(memory: Any) -> bool:
    """Whether an active memory entry still carries the lighter-options preference."""
    for entry in tuple(getattr(memory, "active_entries", ()) or ()):
        text = f"{getattr(entry, 'value', '')} {getattr(entry, 'source_text', '')}".casefold()
        if "light" in text:
            return True
    return False


def _role_of_state(state: Mapping[str, Any]) -> tuple[str, str]:
    return str(state.get("goal_type")), str(state.get("role", "main"))


def _role_statuses(final_ids: Sequence[str], records: Mapping[str, Any], target: Any) -> list[dict[str, Any]]:
    from recommendation.demo.product_role import assess_candidate

    out = []
    for identity in final_ids:
        record = records.get(identity)
        assessment = assess_candidate(identity, record, target)
        out.append(
            {
                "parent_asin": identity,
                "status": assessment.status.value,
                "title": getattr(record, "title", None) if record is not None else None,
                "price": _price_of(record),
                "reasons": list(assessment.reasons),
            }
        )
    return out


def _diagnose(
    *,
    task: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]],
    final_body: Mapping[str, Any] | None,
    final_expected: Mapping[str, Any],
    final_diagnostics: Mapping[str, Any] | None,
    final_role_diagnostics: Mapping[str, Any] | None,
    final_memory: Any,
    records: Mapping[str, Any],
) -> dict[str, Any]:
    from recommendation.demo.product_role import ProductTarget, RoleStatus, TargetRole, assess_candidate

    checks = dict(task.get("checks") or {})
    goal_type, role = _role_of_state(final_expected)
    target = ProductTarget(goal_type or None, TargetRole(role))
    price_max = final_expected.get("price_max")
    final_ids = [
        str(item.get("parent_asin", ""))
        for item in ((final_body or {}).get("recommendations", ()) or ())
    ]
    role_evidence = _role_statuses(final_ids, records, target)
    role_correct = sum(1 for item in role_evidence if item["status"] == RoleStatus.MATCH.value)
    role_mismatch = sum(1 for item in role_evidence if item["status"] == RoleStatus.MISMATCH.value)
    role_unknown = sum(1 for item in role_evidence if item["status"] == RoleStatus.UNKNOWN.value)

    errors: list[str] = []
    unverified: list[str] = []

    # Per-turn canonical-state checks: price, goal/role target, lighter preference.
    state_checks: list[dict[str, Any]] = []
    for turn in turns:
        expected = turn["expected_effective_state"]
        trace = turn.get("constraint_trace") or {}
        actual_prices = _price_from_constraint_trace(trace)
        expected_prices = [] if expected.get("price_max") is None else [float(expected["price_max"])]
        price_ok = sorted(actual_prices) == sorted(expected_prices)
        actual_target = _role_target_from_diagnostics(turn.get("role_diagnostics"))
        expected_type, expected_role = _role_of_state(expected)
        if expected_type is None:
            target_ok: bool | None = None
        elif actual_target is None:
            target_ok = None
        else:
            target_ok = (
                str(actual_target.get("type")) == expected_type
                and str(actual_target.get("role")) == expected_role
            )
        expected_lighter = any(
            "light" in str(item.get("value", "")).casefold()
            for item in expected.get("preferences", ())
        )
        lighter_ok = bool(turn.get("lighter_active")) == expected_lighter
        state_ok = not any(value is False for value in (price_ok, target_ok, lighter_ok))
        state_checks.append(
            {
                "turn": turn["turn"],
                "message": turn["message"],
                "expected": expected,
                "actual_price_max": actual_prices,
                "actual_role_target": actual_target,
                "actual_lighter_active": turn.get("lighter_active"),
                "price_ok": price_ok,
                "target_ok": target_ok,
                "lighter_ok": lighter_ok,
                "state_ok": state_ok,
            }
        )
    state_fail = [item for item in state_checks if not item["state_ok"]]
    state_missing = [
        item for item in state_checks
        if item["state_ok"] and item["target_ok"] is None and item["expected"]["goal_type"] is not None
    ]
    if state_fail:
        errors.append(
            "shared M5 state mismatch at turn(s) "
            + ",".join(str(item["turn"]) for item in state_fail)
        )
    if state_missing:
        unverified.append(
            "role target evidence missing at turn(s) "
            + ",".join(str(item["turn"]) for item in state_missing)
        )
    preference_consistent = not state_fail and not state_missing

    # Final presentation checks.
    price_violations: list[str] = []
    price_unknown: list[str] = []
    for item in role_evidence:
        if item["price"] is None:
            price_unknown.append(item["parent_asin"])
        elif price_max is not None and item["price"] > float(price_max) + 1e-9:
            price_violations.append(f"{item['parent_asin']}:{item['price']}>{price_max}")
    if role_mismatch:
        errors.append(f"{role_mismatch} final product(s) role-MISMATCH")
    if role_unknown:
        unverified.append(f"{role_unknown} final product(s) role-UNKNOWN")
    if price_violations:
        errors.append("hard-constraint violation: " + ", ".join(price_violations))
    if price_unknown and price_max is not None:
        unverified.append("final product price unknown: " + ", ".join(price_unknown))

    catalogue = catalogue_evidence(records, target, None if price_max is None else float(price_max))
    eligible_exists = catalogue["eligible_verified_count"] > 0
    existence_verified = eligible_exists or (
        catalogue["role_match_count"] == 0
        and catalogue["eligible_price_unknown_count"] == 0
        and catalogue["role_unknown_count"] == 0
    )
    eligible_present = any(item["status"] == RoleStatus.MATCH.value for item in role_evidence) and (
        price_max is None
        or any(
            item["status"] == RoleStatus.MATCH.value
            and item["price"] is not None
            and item["price"] <= float(price_max) + 1e-9
            for item in role_evidence
        )
    )
    terminal = (final_body or {}).get("terminal") or {}
    finished = str(terminal.get("status")) == "finished" and bool(final_ids)

    if checks.get("must_report_unknown_when_no_eligible") and not eligible_exists:
        if not existence_verified:
            unverified.append(
                "catalogue eligibility cannot be decided: "
                f"{catalogue['role_match_count']} role-matching product(s), "
                f"{catalogue['eligible_price_unknown_count']} with unknown price"
            )
        if final_ids:
            errors.append("no verified eligible catalogue product but products were presented")
        else:
            message = str((final_body or {}).get("message") or "").casefold()
            if not any(
                phrase in message
                for phrase in ("no ", "none", "couldn't", "cannot", "unable", "not find", "no matching",
                               "no suitable", "no product", "no item", "budget")
            ):
                unverified.append("Agent did not report that no eligible product was verified")
    else:
        if not final_ids:
            if eligible_exists:
                errors.append("catalogue contains verified eligible products but Agent presented none")
            else:
                unverified.append("no final products and no verified catalogue-eligible product")
        elif not finished:
            unverified.append(
                f"final terminal status is {terminal.get('status')!r}, recommendations={len(final_ids)}"
            )

    if errors:
        result = "FAIL"
    elif unverified:
        result = "UNKNOWN"
    else:
        result = "PASS"
    return {
        "result": result,
        "verified_success": result == "PASS",
        "unknown": result == "UNKNOWN",
        "reasons": errors + unverified,
        "final_ids": final_ids,
        "final_requirements": [
            {"kind": "price_max", "value": price_max} for _ in ([0] if price_max is not None else [])
        ],
        "role_correct": role_correct,
        "role_total": len(role_evidence),
        "role_evidence": role_evidence,
        "hard_constraint_violations": len(price_violations),
        "hard_constraint_violation_detail": price_violations,
        "preference_update_consistent": preference_consistent,
        "per_turn_state": state_checks,
        "eligible_candidate_present": eligible_present,
        "eligible_catalogue_verified": eligible_exists,
        "catalogue_existence_verified": existence_verified,
        "catalogue_evidence": catalogue,
        "tool_calls": sum(int((turn.get("observation") or {}).get("tool_calls_used", 0)) for turn in turns),
        "final_role_diagnostics": final_role_diagnostics,
        "final_m5_diagnostics": final_diagnostics,
        "final_memory_lighter_active": _lighter_active(final_memory) if final_memory is not None else None,
    }


def _planner_turn(
    *,
    service: Any,
    proposer: Any,
    payload: Any,
    session_id: str,
    task_id: str,
    turn_prefix: str,
    message: str,
    state: Mapping[str, Any],
    max_steps: int = 6,
    source_policy: str = "balanced",
) -> dict[str, Any]:
    """Run one user turn through the existing HTTP Planner Proposer + executor.

    The per-user-turn M5 state operations have already been applied by the caller; this
    function performs only the Planner episode (RETRIEVE/RERANK/FINALIZE/ASK).  It does
    not re-apply state operations between steps.  The canonical state is exposed to the
    Planner observation (role target + effective state) and hard constraints travel in the
    trusted request payload.
    """
    from recommendation.demo.planner_contract import PlannerObservation, run_planner
    from recommendation.demo.planner_service_executor import ServicePlannerExecutor

    # Seed the deterministic role target from the real user message before the episode,
    # so a rewritten RETRIEVE query cannot change which product role this turn is about.
    service.prime_task_role_target(
        task_key=task_id, user_key=session_id, message=message
    )
    role_target = {
        "type": state.get("goal_type"),
        "role": state.get("role", "main"),
    }
    executor = ServicePlannerExecutor(
        service=service,
        payload=payload,
        session_id=session_id,
        task_id=task_id,
        turn_prefix=turn_prefix,
        effective_role_target=role_target,
        effective_state=dict(state),
        state_message=message,
        source_policy=source_policy,
    )
    initial = PlannerObservation(
        step=0,
        query=message,
        candidate_ids=(),
        terminal=False,
        needs_user_input=False,
        m5_state={"effective_state": dict(state)},
        role_target=role_target,
        remaining_steps=max_steps,
        source_policy=source_policy,
    )
    run = run_planner(
        planner=proposer,
        executor=executor,
        initial_observation=initial,
        max_steps=max_steps,
    )
    response = executor.last_response
    body = response.model_dump() if response is not None else None
    final_ids = list(run.final_ids) if run.termination == "FINALIZE" else []
    if body is None:
        body = {
            "recommendations": [],
            "message": "",
            "terminal": {
                "status": "waiting_for_user" if run.termination == "ASK" else run.termination,
                "termination_reason": run.termination,
                "succeeded": False,
            },
            "eligibility": {},
            "observation": {"tool_calls_used": executor.tool_calls_used},
        }
    else:
        body = dict(body)
        by_id = {
            str(item.get("parent_asin")): item
            for item in body.get("recommendations", ()) or ()
        }
        if final_ids:
            ordered = [by_id[identity] for identity in final_ids if identity in by_id]
            body["recommendations"] = ordered
            if len(ordered) != len(final_ids):
                body["planner_identity_mismatch"] = {
                    "final_ids": final_ids,
                    "response_ids": list(by_id),
                }
        else:
            body["recommendations"] = []
            if run.termination == "ASK":
                body["terminal"] = {
                    "status": "waiting_for_user",
                    "termination_reason": "awaiting_user",
                    "succeeded": False,
                }
            elif run.termination == "max_steps":
                body["terminal"] = {
                    "status": "max_steps",
                    "termination_reason": "max_steps",
                    "succeeded": False,
                }
    return {
        "body": body,
        "planner_run": run.as_dict(),
        "action_sequence": [step.action for step in run.steps],
        "planner_final_ids": final_ids,
        "tool_calls_used": int(executor.tool_calls_used),
        "last_response_terminal": (
            response.terminal.model_dump()
            if response is not None and getattr(response, "terminal", None) is not None
            else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6.4/P7-A controlled dynamic-DEV runner")
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--only", default="", help="comma-separated task_id/template_id selection")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--arm",
        choices=("rule", "base", "sft"),
        default="rule",
        help="rule = direct DemoAgentService; base/sft = existing HTTP Planner Proposer + ServicePlannerExecutor",
    )
    parser.add_argument("--planner-url", default=None, help="Planner HTTP base URL (proposer appends /plan)")
    parser.add_argument("--planner-model", default=None, help="optional model field sent to the planner server")
    parser.add_argument("--planner-timeout", type=float, default=120.0)
    parser.add_argument(
        "--source-policy",
        choices=("balanced", "intent_focus", "history_focus"),
        default="balanced",
        help="Agent-level history/catalog rank-fusion policy for controlled evaluation",
    )
    parser.add_argument(
        "--check-wiring",
        action="store_true",
        help="validate arm/proposer configuration only; build no runtime and issue no HTTP call",
    )
    args = parser.parse_args(argv)

    if args.arm in ("base", "sft") and not args.planner_url:
        parser.error(f"--arm {args.arm} requires --planner-url")
    if args.check_wiring:
        from recommendation.demo.planner_contract import run_planner, validate_action
        from recommendation.demo.planner_http_proposer import HttpPlannerProposer
        from recommendation.demo.planner_service_executor import ServicePlannerExecutor

        proposer = (
            HttpPlannerProposer(
                args.planner_url,
                model=args.planner_model,
                timeout=args.planner_timeout,
            )
            if args.arm in ("base", "sft")
            else None
        )
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "proposer": type(proposer).__name__ if proposer is not None else None,
                    "proposer_url": getattr(proposer, "url", None),
                    "executor": ServicePlannerExecutor.__name__,
                    "contract_runner": run_planner.__name__,
                    "validator": validate_action.__name__,
                    "network_request_issued": False,
                    "runtime_built": False,
                },
                indent=2,
            )
        )
        return 0

    only = [part.strip() for part in args.only.split(",") if part.strip()]
    tasks = load_tasks(args.tasks, only)
    history = tuple(tasks[0]["initial_history"])
    if any(tuple(task["initial_history"]) != history for task in tasks):
        raise ValueError("all P6.4 tasks must share one initial history")
    if any(task.get("evaluation_protocol") != "planner_only_controlled_evaluation" for task in tasks):
        raise ValueError("P6.4 task file is not marked planner_only_controlled_evaluation")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _unique_dir(
        args.out or (OUT_ROOT / f"{args.arm}_{args.source_policy}_{timestamp}")
    )
    out.mkdir(parents=True, exist_ok=False)
    os.environ["AGENTRECX_STATE_MODE"] = STATE_MODE
    os.environ["AGENTRECX_STATE_DB"] = str(out / "decision_state.sqlite3")
    os.environ["AGENTRECX_MEMORY_DB"] = str(out / "preference_memory.sqlite3")
    os.environ["AGENTRECX_SERVING_FUSION"] = SERVING_FUSION
    os.environ["AGENTRECX_CATALOG_METADATA_PATH"] = str(PRODUCTS)
    os.environ.pop("AGENTRECX_QWEN_PROPOSAL", None)
    os.environ.pop("AGENTRECX_AGENT_POLICY", None)

    from recommendation.api.app import ServiceSettings, build_engine
    from recommendation.demo.agent_schemas import AgentConstraintItem, AgentRecommendRequest
    from recommendation.demo.profiles import DemoProfile
    from recommendation.demo.runtime import build_demo_runtime
    from recommendation.demo.planner_contract import PlannerObservation, run_planner
    from recommendation.demo.planner_service_executor import ServicePlannerExecutor
    from experiments.p6_benchmark.metrics import dynamic_metrics

    planner_proposer = None
    if args.arm in ("base", "sft"):
        from recommendation.demo.planner_http_proposer import HttpPlannerProposer

        planner_proposer = HttpPlannerProposer(
            args.planner_url,
            model=args.planner_model,
            timeout=args.planner_timeout,
        )

    started = time.perf_counter()
    settings = ServiceSettings.from_env()
    engine, error = build_engine(settings)
    if engine is None:
        raise SystemExit(f"serving engine unavailable: {error}")
    engine_meta = engine.model_metadata()
    if engine_meta.get("active_engine") != "fusion_rrf":
        raise SystemExit(f"expected fusion_rrf, got {engine_meta.get('active_engine')!r}")
    profile = DemoProfile(
        profile_id=PROFILE_ID,
        display_name="P6.4 controlled shopper",
        trusted_user_history=history,
        source_user_int_id=0,
        source_length=len(history) + 2,
    )
    runtime = build_demo_runtime(
        settings=settings,
        engine=engine,
        memory_db=out / "preference_memory.sqlite3",
        profiles={PROFILE_ID: profile},
    )
    service = runtime.agent_service()
    if service.policy_name != "deterministic":
        raise SystemExit(f"expected deterministic Rule Agent, got {service.policy_name!r}")
    if int(service.tool_budget) != 4:
        raise SystemExit(f"expected per-turn tool budget 4, got {service.tool_budget!r}")
    records = runtime.metadata.records

    turns_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for task in tasks:
        session = runtime.sessions.create(PROFILE_ID)
        state = copy.deepcopy(task["initial_m5"])
        turn_records: list[dict[str, Any]] = []
        final_body: dict[str, Any] | None = None
        final_expected: dict[str, Any] = state
        execution_error: str | None = None
        for index, turn in enumerate(task["turns"]):
            state = apply_operations(state, turn.get("state_operations", ()))
            if state != turn["expected_effective_state"]:
                raise SystemExit(
                    f"{task['task_id']} turn {index}: transition state {state} != "
                    f"declared {turn['expected_effective_state']}"
                )
            final_expected = state
            constraints = (
                [AgentConstraintItem(kind="price_max", value=str(state["price_max"]))]
                if state.get("price_max") is not None
                else []
            )
            try:
                with runtime.sessions.turn(session.session_id) as allocation:
                    current = allocation.session
                    payload = AgentRecommendRequest(
                        message=turn["message"],
                        history=list(current.trusted_user_history),
                        k=K,
                        sources=SOURCES,
                        constraints=constraints,
                        user_key=current.user_key,
                    )
                    began = time.perf_counter()
                    planner_info: dict[str, Any] | None = None
                    if args.arm == "rule":
                        response = service.run_shopping_turn(
                            payload,
                            session_id=current.user_key,
                            task_id=current.session_id,
                            turn_id=allocation.turn_id,
                            source_policy=args.source_policy,
                        )
                        body = response.model_dump()
                    else:
                        planner_info = _planner_turn(
                            service=service,
                            proposer=planner_proposer,
                            payload=payload,
                            session_id=current.user_key,
                            task_id=current.session_id,
                            turn_prefix=f"p7a-{task['task_id']}-{index}",
                            message=turn["message"],
                            state=state,
                            source_policy=args.source_policy,
                        )
                        body = planner_info["body"]
                    seconds = round(time.perf_counter() - began, 3)
                final_body = body
                diagnostics = service.state_diagnostics(current.session_id)
                trace = service.constraint_trace(current.session_id)
                role_diag = service.role_diagnostics(current.user_key)
                memory = runtime.memory_service.get_active_preferences(current.user_key)
                record = {
                    "task_id": task["task_id"],
                    "turn": index,
                    "arm": args.arm,
                    "message": turn["message"],
                    "expected_effective_state": state,
                    "state_operations": turn.get("state_operations", []),
                    "request_constraints": [item.model_dump() for item in constraints],
                    "seconds": seconds,
                    "route": body.get("route"),
                    "terminal": body.get("terminal"),
                    "recommendations": body.get("recommendations"),
                    "observation": body.get("observation"),
                    "eligibility": body.get("eligibility"),
                    "message_text": body.get("message"),
                    "constraint_trace": trace,
                    "state_diagnostics": diagnostics,
                    "role_diagnostics": role_diag,
                    "candidate_filter": service.candidate_filter_diagnostics(current.user_key),
                    "lighter_active": _lighter_active(memory),
                }
                if planner_info is not None:
                    record.update(
                        {
                            "action_sequence": planner_info["action_sequence"],
                            "planner_run": planner_info["planner_run"],
                            "planner_final_ids": planner_info["planner_final_ids"],
                            "last_response_terminal": planner_info["last_response_terminal"],
                            "observation": {"tool_calls_used": planner_info["tool_calls_used"]},
                        }
                    )
                turns_rows.append(record)
                turn_records.append(record)
            except Exception as exc:  # noqa: BLE001 - a real execution failure is data
                execution_error = f"{type(exc).__name__}: {exc}"
                turn_records.append(
                    {"task_id": task["task_id"], "turn": index, "message": turn["message"],
                     "expected_effective_state": state, "error": execution_error}
                )
                break

        role_diag = service.role_diagnostics(session.user_key)
        memory = runtime.memory_service.get_active_preferences(session.user_key)
        if execution_error:
            evaluation = {
                "result": "FAIL", "verified_success": False, "unknown": False,
                "reasons": [f"execution error: {execution_error}"],
                "tool_calls": sum(
                    int((turn.get("observation") or {}).get("tool_calls_used", 0))
                    for turn in turn_records
                ),
                "role_correct": 0, "role_total": 0, "role_evidence": [],
                "hard_constraint_violations": 0, "preference_update_consistent": False,
                "eligible_candidate_present": False, "final_ids": [],
                "catalogue_evidence": {"error": "not evaluated because execution failed"},
            }
        else:
            evaluation = _diagnose(
                task=task,
                turns=turn_records,
                final_body=final_body,
                final_expected=final_expected,
                final_diagnostics=service.state_diagnostics(session.session_id),
                final_role_diagnostics=role_diag,
                final_memory=memory,
                records=records,
            )
        task_rows.append(
            {
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "arm": args.arm,
                "evaluation_protocol": task["evaluation_protocol"],
                "initial_history": list(history),
                "initial_m5": task["initial_m5"],
                "turns": task["turns"],
                "checks": task["checks"],
                "final_action_sequence": (
                    turn_records[-1].get("action_sequence") if turn_records else []
                ),
                "final_planner_run": (
                    turn_records[-1].get("planner_run") if turn_records else None
                ),
                "final_expected_state": final_expected,
                "final_response": final_body,
                "final_diagnostics": service.state_diagnostics(session.session_id),
                "final_role_diagnostics": role_diag,
                "final_memory_lighter_active": _lighter_active(memory),
                "evaluation": evaluation,
            }
        )
        metric_rows.append(
            {
                "verified_success": evaluation["verified_success"],
                "hard_constraint_violations": evaluation["hard_constraint_violations"],
                "role_correct": evaluation["role_correct"],
                "role_total": evaluation["role_total"],
                "preference_update_consistent": evaluation["preference_update_consistent"],
                "eligible_candidate_present": evaluation["eligible_candidate_present"],
                "tool_calls": evaluation["tool_calls"],
                "unknown": evaluation["unknown"],
            }
        )
        runtime.sessions.delete(session.session_id)

    metrics = dynamic_metrics(metric_rows)
    summary = {
        "schema_version": "agentrecx.p7a.dynamic_dev_run.v0",
        "created_utc": _utc_now(),
        "git_head": _git_head(),
        "arm": args.arm,
        "source_policy": args.source_policy,
        "evaluation_protocol": "planner_only_controlled_evaluation",
        "tasks_file": str(args.tasks.relative_to(REPO)) if args.tasks.is_relative_to(REPO) else str(args.tasks),
        "tasks_sha256": __import__("hashlib").sha256(args.tasks.read_bytes()).hexdigest(),
        "configuration": {
            "profile": PROFILE_ID,
            "initial_history": list(history),
            "state_mode": STATE_MODE,
            "serving_fusion": SERVING_FUSION,
            "k": K,
            "max_steps": 6,
            "per_turn_max_tool_calls": 4,
            "sources": list(SOURCES),
            "policy": service.policy_name,
            "arm": args.arm,
            "source_policy": args.source_policy,
            "planner_url": args.planner_url,
            "planner_model": args.planner_model,
            "planner_proposer": (
                type(planner_proposer).__name__ if planner_proposer is not None else None
            ),
            "shared_state_rule": "runner applies the task's state_operations to the canonical M5 state once per user turn and passes hard constraints through the trusted request field; a Planner episode's RETRIEVE/RERANK steps do not re-apply state operations; role/soft extraction is the deterministic trusted service, not a Planner score",
        },
        "serving": {
            "active_engine": engine_meta.get("active_engine"),
            "fusion_sources": engine_meta.get("fusion_sources"),
            "rrf_k": engine_meta.get("rrf_k"),
            "source_head_depth": engine_meta.get("source_head_depth"),
            "checkpoint": str(CHECKPOINT.relative_to(REPO)),
            "num_items": engine_meta.get("num_items"),
        },
        "resources": {"device": "cpu", "wall_seconds": round(time.perf_counter() - started, 1)},
        "metrics": metrics,
        "tasks": {row["task_id"]: row for row in task_rows},
        "metric_rows": metric_rows,
        "notes": [
            "Only Dev tasks were read/executed; Holdout was not read.",
            "Expected state controls execution/evaluation only; no Gold product identity is passed to the Agent.",
            "Final scoring uses committed response.recommendations after promotion.",
            "Planner-only controlled evaluation: parsing ability is not part of the Planner score; shared M5 state updates are runner-owned.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (out / "turns.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in turns_rows),
        encoding="utf-8",
    )
    (out / "tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in task_rows),
        encoding="utf-8",
    )
    print(json.dumps(
        {"out": str(out), "metrics": metrics,
         "tasks": {row["task_id"]: row["evaluation"]["result"] for row in task_rows}},
        indent=2, ensure_ascii=False,
    ))
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
