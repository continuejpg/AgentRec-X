"""Phase 6E: paired fixed-policy vs dynamic-research task evaluation.

The same task request is sent to:

* the fixed ``DemoSourcePlanPolicy(("history",))`` path, with no research/clarification;
* the demo shopping path with ``DynamicProductResearchPolicy`` and its existing recovery
  mechanism.

Strict completion means the **requested evidence** was obtained when the task asks for a
comparison, not merely that some terminal was reached.  Terminal-only completion is reported
separately so the distinction cannot be hidden.

No production code is changed; no external LLM is called; no training/full suite is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recommendation.demo.agent_schemas import AgentRecommendRequest  # noqa: E402
from tests.demo_fixture import build_harness, PROFILE_A  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "p6e_paired_research_eval"
BASE = "/v1/demo/sessions/{session}/agent/shopping"
MIXED = "compare the first two by price, color, and material"
WITH_DESCRIPTION = "compare the first two by price, color, material, and description"
EXPLICIT = "compare the first two by price and color"
CANDIDATES = {"cand-red", "cand-blue", "cand-black", "cand-green"}

TASKS: tuple[dict[str, Any], ...] = (
    {
        "name": "sufficient",
        "message": "recommend outdoor gear",
        "kind": "completion",
        "strict_rule": "nonempty_final",
    },
    {
        "name": "explicit_compare",
        "message": EXPLICIT,
        "kind": "completion",
        "strict_rule": "price_color_comparison",
    },
    {
        "name": "material_alt_price",
        "message": MIXED,
        "kind": "completion",
        "strict_rule": "price_comparison",
        "recovery": "compare:price",
    },
    {
        "name": "hard_constraint",
        "message": EXPLICIT,
        "kind": "completion",
        "strict_rule": "price_color_comparison_eligible",
        "constraints": [{"kind": "price_max", "value": "30"}],
    },
    {
        "name": "missing_material_known",
        "message": MIXED,
        "kind": "safety",
        "strict_rule": "known_facts_resolution",
        "recovery": "known_facts",
    },
    {
        "name": "missing_material_stop",
        "message": MIXED,
        "kind": "safety",
        "strict_rule": "stop_resolution",
        "recovery": "stop_research",
    },
    {
        "name": "description_second_stop",
        "message": WITH_DESCRIPTION,
        "kind": "safety",
        "strict_rule": "bounded_stop_resolution",
        "recovery": "compare:description",
        "second_recovery": "stop_research",
    },
    {
        "name": "untrusted_identity",
        "message": "compare cand-green and cand-black by price",
        "kind": "safety",
        "strict_rule": "safe_grounded_final",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(value: Any) -> str:
    return str(getattr(value, "value", value))


def _action_dict(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump"):
        return action.model_dump()
    if isinstance(action, Mapping):
        return dict(action)
    return {}


def _research_actions(response: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    value = getattr(response, "research", None)
    if value is None and isinstance(response, Mapping):
        value = response.get("research")
    if value is None:
        return []
    actions = getattr(value, "actions", None)
    if actions is None and isinstance(value, Mapping):
        actions = value.get("actions")
    return [_action_dict(action) for action in actions or ()]


def _final_ids(response: Mapping[str, Any] | Any) -> list[str]:
    items = getattr(response, "recommendations", None)
    if items is None and isinstance(response, Mapping):
        items = response.get("recommendations")
    ids: list[str] = []
    for item in items or ():
        if isinstance(item, Mapping):
            ids.append(str(item.get("parent_asin") or ""))
        else:
            ids.append(str(getattr(item, "parent_asin", "") or ""))
    return [identity for identity in ids if identity]


def _terminal(response: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    value = getattr(response, "terminal", None)
    if value is None and isinstance(response, Mapping):
        value = response.get("terminal")
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _compare_actions(response: Mapping[str, Any] | Any) -> list[dict[str, Any]]:
    return [
        action for action in _research_actions(response)
        if action.get("tool") == "compare_products"
    ]


def _has_attributes(actions: Sequence[Mapping[str, Any]], required: set[str]) -> bool:
    return any(required <= set(action.get("attributes") or ()) for action in actions)


def _strict(task: Mapping[str, Any], response: Mapping[str, Any] | Any) -> bool:
    if task["kind"] != "completion":
        return False
    terminal = _terminal(response)
    if _status(terminal.get("status")) != "finished":
        return False
    ids = _final_ids(response)
    if not ids:
        return False
    compare = _compare_actions(response)
    rule = task["strict_rule"]
    if rule == "nonempty_final":
        return True
    if rule == "price_color_comparison":
        return _has_attributes(compare, {"price", "color"})
    if rule == "price_comparison":
        return _has_attributes(compare, {"price"})
    if rule == "price_color_comparison_eligible":
        return (
            _has_attributes(compare, {"price", "color"})
            and "cand-blue" not in ids
            and ids == ["cand-red", "cand-black"]
        )
    raise ValueError(rule)


def _safe_final(task: Mapping[str, Any], response: Mapping[str, Any] | Any) -> bool:
    terminal = _terminal(response)
    if _status(terminal.get("status")) != "finished":
        return False
    ids = _final_ids(response)
    if not ids or not set(ids) <= CANDIDATES:
        return False
    if task["name"] == "hard_constraint" and "cand-blue" in ids:
        return False
    return True


def _waiting(response: Mapping[str, Any] | Any) -> bool:
    return _status(_terminal(response).get("status")) == "waiting_for_user"


def _post(harness: Any, session: str, body: Mapping[str, Any]) -> Any:
    response = harness.client.post(BASE.format(session=session), json=dict(body))
    return response


def _dynamic_run(harness: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    session = harness.create_session(PROFILE_A)["session_id"]
    body: dict[str, Any] = {
        "message": task["message"],
        "k": 3,
        "sources": ["history"],
    }
    if task.get("constraints"):
        body["constraints"] = list(task["constraints"])
    started = time.perf_counter()
    responses: list[dict[str, Any]] = []
    first = _post(harness, session, body)
    assert first.status_code == 200, first.text
    responses.append(first.json())
    first_waiting = _waiting(responses[-1])
    initial_clarification = responses[-1].get("clarification")
    if first_waiting and task.get("recovery"):
        cid = initial_clarification["clarification_id"]
        answer = dict(body)
        answer.update({"clarification_id": cid, "clarification_option": task["recovery"]})
        second = _post(harness, session, answer)
        assert second.status_code == 200, second.text
        responses.append(second.json())
        if _waiting(responses[-1]) and task.get("second_recovery"):
            cid2 = responses[-1]["clarification"]["clarification_id"]
            answer2 = dict(body)
            answer2.update(
                {"clarification_id": cid2, "clarification_option": task["second_recovery"]}
            )
            third = _post(harness, session, answer2)
            assert third.status_code == 200, third.text
            responses.append(third.json())
    elapsed = (time.perf_counter() - started) * 1000.0
    final = responses[-1]
    all_research = _research_actions(final)
    signatures = [
        (
            action.get("tool"),
            tuple(action.get("products") or ()),
            tuple(action.get("attributes") or ()),
        )
        for action in all_research
    ]
    duplicates = len(signatures) - len(set(signatures))
    recovery_attempted = bool(task.get("recovery")) and first_waiting
    return {
        "policy": "dynamic",
        "response_statuses": [
            _status(_terminal(response).get("status")) for response in responses
        ],
        "response_count": len(responses),
        "first_waiting": first_waiting,
        "correct_waiting": first_waiting and bool(initial_clarification),
        "recovery_attempted": recovery_attempted,
        "recovery_success": (
            None
            if not recovery_attempted
            else (
                _status(_terminal(final).get("status")) == "finished"
                and len(responses) >= 2
            )
        ),
        "strict_complete": _strict(task, final),
        "acceptable_resolution": (
            task["kind"] == "safety"
            and first_waiting
            and _status(_terminal(final).get("status")) == "finished"
            and _safe_final(task, final)
        ),
        "safe_final": _safe_final(task, final),
        "terminal_status": _status(_terminal(final).get("status")),
        "terminal_reason": str(_terminal(final).get("termination_reason") or ""),
        "final_ids": _final_ids(final),
        "research_actions": all_research,
        "research_calls": len(all_research),
        "duplicate_research": duplicates,
        "invalid_research": sum(
            1
            for action in all_research
            if any(str(item) not in CANDIDATES for item in (action.get("products") or ()))
        ),
        "hard_constraint_violations": (
            1 if task["name"] == "hard_constraint" and "cand-blue" in _final_ids(final) else 0
        ),
        "unverified_presented": sum(
            1 for identity in _final_ids(final) if identity not in CANDIDATES
        ),
        "tool_calls": sum(int(_terminal(response).get("tool_calls") or 0) for response in responses),
        "ms": round(elapsed, 2),
        "second_clarification": (
            task.get("second_recovery") is not None
            and any(_waiting(response) for response in responses[1:])
        ),
    }


def _fixed_run(harness: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    service = harness.runtime.agent_service()
    constraints = tuple(task.get("constraints") or ())
    request = AgentRecommendRequest(
        message=task["message"],
        history=list(harness.profiles[PROFILE_A].trusted_user_history),
        k=3,
        sources=("history",),
        constraints=constraints,
    )
    started = time.perf_counter()
    response = service.run(request, dynamic_research=False)
    elapsed = (time.perf_counter() - started) * 1000.0
    final = response
    return {
        "policy": "fixed",
        "response_statuses": [_status(_terminal(final).get("status"))],
        "response_count": 1,
        "first_waiting": _waiting(final),
        "correct_waiting": _waiting(final) and bool(getattr(final, "clarification", None)),
        "recovery_attempted": False,
        "recovery_success": None,
        "strict_complete": _strict(task, final),
        "acceptable_resolution": False,
        "safe_final": _safe_final(task, final),
        "terminal_status": _status(_terminal(final).get("status")),
        "terminal_reason": str(_terminal(final).get("termination_reason") or ""),
        "final_ids": _final_ids(final),
        "research_actions": _research_actions(final),
        "research_calls": len(_research_actions(final)),
        "duplicate_research": 0,
        "invalid_research": 0,
        "hard_constraint_violations": (
            1 if task["name"] == "hard_constraint" and "cand-blue" in _final_ids(final) else 0
        ),
        "unverified_presented": sum(
            1 for identity in _final_ids(final) if identity not in CANDIDATES
        ),
        "tool_calls": int(_terminal(final).get("tool_calls") or 0),
        "ms": round(elapsed, 2),
        "second_clarification": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    harness = build_harness(Path(tempfile.mkdtemp()), max_sessions=64)
    profile_history = list(harness.profiles[PROFILE_A].trusted_user_history)
    metadata = harness.runtime.metadata
    red = metadata.records["cand-red"]
    blue = metadata.records["cand-blue"]
    metadata.records["cand-red"] = red.model_copy(
        update={"description": ("red description",)}
    )
    metadata.records["cand-blue"] = blue.model_copy(update={"description": ()})

    rows: list[dict[str, Any]] = []
    try:
        for task in TASKS:
            fixed = _fixed_run(harness, task)
            dynamic = _dynamic_run(harness, task)
            fixed["task"] = task["name"]
            dynamic["task"] = task["name"]
            fixed["task_kind"] = task["kind"]
            dynamic["task_kind"] = task["kind"]
            rows.extend((fixed, dynamic))
    finally:
        harness.runtime.close()

    def aggregate(policy: str, kind: str | None = None) -> dict[str, Any]:
        selected = [
            row for row in rows
            if row["policy"] == policy and (kind is None or row["task_kind"] == kind)
        ]
        completion = [row for row in selected if row["task_kind"] == "completion"]
        return {
            "runs": len(selected),
            "strict_completion": sum(1 for row in selected if row["strict_complete"]),
            "strict_completion_rate": round(
                sum(1 for row in completion if row["strict_complete"]) / len(completion), 4
            ) if completion else None,
            "terminal_only_completion_rate": round(
                sum(1 for row in completion if row["terminal_status"] == "finished")
                / len(completion), 4
            ) if completion else None,
            "correct_waiting": sum(1 for row in selected if row["correct_waiting"]),
            "acceptable_resolution": sum(
                1 for row in selected if row["acceptable_resolution"]
            ),
            "recovery_attempted": sum(1 for row in selected if row["recovery_attempted"]),
            "recovery_success": sum(1 for row in selected if row["recovery_success"] is True),
            "recovery_success_rate": (
                round(
                    sum(1 for row in selected if row["recovery_success"] is True)
                    / sum(1 for row in selected if row["recovery_attempted"]),
                    4,
                )
                if any(row["recovery_attempted"] for row in selected)
                else None
            ),
            "second_clarifications": sum(
                1 for row in selected if row["second_clarification"]
            ),
            "research_calls": sum(row["research_calls"] for row in selected),
            "duplicate_research": sum(row["duplicate_research"] for row in selected),
            "invalid_research": sum(row["invalid_research"] for row in selected),
            "hard_constraint_violations": sum(
                row["hard_constraint_violations"] for row in selected
            ),
            "unverified_presented": sum(row["unverified_presented"] for row in selected),
            "tool_calls": sum(row["tool_calls"] for row in selected),
            "p50_ms": round(statistics.median(row["ms"] for row in selected), 2),
            "p95_ms": round(sorted(row["ms"] for row in selected)[-1], 2),
        }

    by_policy = {
        policy: {
            "all": aggregate(policy),
            "completion_tasks": aggregate(policy, "completion"),
            "safety_tasks": aggregate(policy, "safety"),
        }
        for policy in ("fixed", "dynamic")
    }
    paired = [
        {
            "task": task["name"],
            "kind": task["kind"],
            "fixed_strict": next(
                row["strict_complete"] for row in rows
                if row["task"] == task["name"] and row["policy"] == "fixed"
            ),
            "dynamic_strict": next(
                row["strict_complete"] for row in rows
                if row["task"] == task["name"] and row["policy"] == "dynamic"
            ),
            "fixed_terminal": next(
                row["terminal_status"] for row in rows
                if row["task"] == task["name"] and row["policy"] == "fixed"
            ),
            "dynamic_terminal": next(
                row["terminal_status"] for row in rows
                if row["task"] == task["name"] and row["policy"] == "dynamic"
            ),
            "dynamic_research_calls": next(
                row["research_calls"] for row in rows
                if row["task"] == task["name"] and row["policy"] == "dynamic"
            ),
        }
        for task in TASKS
    ]
    summary = {
        "task": "phase6e_paired_research_value",
        "tasks": [task["name"] for task in TASKS],
        "task_labels": {
            "completion": "requested evidence obtained; not merely a terminal",
            "safety": "correct waiting/stop/safe grounded final; not counted as completion",
        },
        "by_policy": by_policy,
        "paired": paired,
        "rows": rows,
        "decision": {
            "completion_gain": by_policy["dynamic"]["completion_tasks"]["strict_completion"]
            - by_policy["fixed"]["completion_tasks"]["strict_completion"],
            "terminal_only_gain": by_policy["dynamic"]["completion_tasks"]["terminal_only_completion_rate"]
            - by_policy["fixed"]["completion_tasks"]["terminal_only_completion_rate"],
            "research_calls_added": by_policy["dynamic"]["all"]["research_calls"],
            "safety_regressions": by_policy["dynamic"]["all"]["hard_constraint_violations"]
            + by_policy["dynamic"]["all"]["unverified_presented"],
        },
        "notes": {
            "external_llm": "none",
            "training": "none",
            "production_code_modified": False,
            "default_sources_depth_rrf": "unchanged",
            "second_clarification_max": 2,
            "task_research_budget": 4,
        },
    }
    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "tasks": list(TASKS),
                "catalog": "tests.demo_fixture (synthetic Phase-4 catalogue)",
                "history": profile_history,
                "driver_sha256": _sha256(Path(__file__)),
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
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
