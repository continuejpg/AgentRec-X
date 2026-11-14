"""Counterfactual memory-state tests for Memory Contribution Phase 1.

Each case is a pair: the same query, the same fixed candidate pool, the same
catalogue metadata and the same hard constraints; only the legal long-term memory
state changes.  The states are produced through the real M9 write path
(``ShoppingCompanionMemory.ingest_sessions``) and read through the real
``HistoryStateReader`` -- no second memory implementation is introduced.

These are constructed consistency tests.  They prove that memory-state handling and
the deterministic scorer respond as specified; they are **not** user-satisfaction
gold and are not counted as such.

Run from the repository root with the local CPU environment::

    .venv/bin/python -m experiments.memory_contribution.run_counterfactual
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Mapping, Sequence

from experiments.memory_contribution import common
from recommendation.preference.history import HistoryStateReader
from recommendation.shopping_companion.memory import (
    SOURCE_KIND_SYNTHETIC_FIXTURE,
    ShoppingCompanionMemory,
)

OUT = common.OUT_DIR
POOL_HEADPHONES = "p14e2_lite_0041"
POOL_EARBUDS = "p14e2_lite_0051"


def _sessions(case_id: str, statements: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": f"{case_id}_s{index}",
            "date": f"2025-0{index + 1}-15",
            "turns": [{"role": "user", "content": statement}],
        }
        for index, statement in enumerate(statements)
    ]


def _memory_state(case_id: str, statements: Sequence[str], query: str) -> dict[str, Any]:
    """Write the stated memory events through M9 and read the legal active state."""
    if not statements:
        return {
            "sessions": [],
            "ingest_summary": None,
            "active_raw": [],
            "inactive_raw": [],
            "active_preferences": [],
            "conflicts": [],
        }
    user_key = f"mem_contrib_{case_id}"
    with ShoppingCompanionMemory() as memory:
        ingest = memory.ingest_sessions(
            user_key,
            _sessions(case_id, statements),
            source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
            source_ref=f"memory_contribution_phase1:{case_id}",
        )
        history = HistoryStateReader(memory.preference_service).read(
            user_key, conversation_id=user_key, current_query=query
        )
    canonical = common.canonical_prefs(history["active_preferences"])
    return {
        "sessions": _sessions(case_id, statements),
        "ingest_summary": {
            "preferences_extracted": ingest["preferences_extracted"],
            "preferences_added": ingest["preferences_added"],
            "preferences_skipped_duplicates": ingest["preferences_skipped_duplicates"],
            "removal_directives": ingest["removal_directives"],
        },
        "active_raw": [
            {
                "kind": item["kind"],
                "value": item["value"],
                "polarity": item["polarity"],
                "status": item["status"],
                "source_turn_id": item["source_turn_id"],
            }
            for item in history["active_preferences"]
        ],
        "inactive_raw": [
            {
                "kind": item["kind"],
                "value": item["value"],
                "polarity": item["polarity"],
                "status": item["status"],
                "source_turn_id": item["source_turn_id"],
            }
            for item in history["inactive_preferences"]
        ],
        "active_preferences": common.preference_summary(
            [{**item, "origin": "history"} for item in canonical]
        ),
        "conflicts": common.conflicts(canonical),
    }


def _state(preferences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "active": common.preference_summary(preferences),
        "conflicts": common.conflicts(list(preferences)),
    }


def _run(pool: Mapping[str, Any], preferences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = common.score_pool(pool["row"], preferences)
    eligible = pool["eligible"]
    order = common.ordered_ids(scored, eligible)
    gate = common.gate_for_arm(
        row=pool["row"], scored_rows=scored, preferences=preferences, order=order
    )
    by_id = {item["product_ref"]["item_id"]: item for item in scored}
    hard_violations = {
        f"violations@{k}": sum(
            1
            for item_id in order[:k]
            if int((by_id.get(item_id) or {}).get("hard_tier") or 0) == 2
        )
        for k in (1, 3, 5)
    }
    return {
        "ranking": order,
        "top1": order[0] if order else None,
        "gate": gate,
        "hard_constraint_violations": hard_violations,
        "state": _state(preferences),
        "candidates": {
            item_id: {
                "rank": next(
                    (rank for rank, value in enumerate(order, start=1) if value == item_id),
                    None,
                ),
                "final_score": by_id[item_id]["final_score"],
                "query_relevance_score": by_id[item_id]["query_relevance_score"],
                "preference_match_score": by_id[item_id]["preference_match_score"],
                "preference_conflict_score": by_id[item_id]["preference_conflict_score"],
                "matched_preferences": [
                    item["value"] for item in by_id[item_id]["matched_preferences"]
                ],
                "conflicting_preferences": [
                    item["value"] for item in by_id[item_id]["conflicting_preferences"]
                ],
                "unknown_preferences": [
                    item["value"] for item in by_id[item_id]["unknown_preferences"]
                ],
                "hard_tier": by_id[item_id]["hard_tier"],
                "role_status": by_id[item_id]["role_status"],
                "presentable": by_id[item_id]["presentable"],
            }
            for item_id in order
        },
    }


def _rank(result: Mapping[str, Any], item_id: str) -> int | None:
    return (result["candidates"].get(item_id) or {}).get("rank")


def _candidate(result: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    return result["candidates"].get(item_id) or {}


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _state_checks(
    case: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    baseline_state: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected = case.get("expected_state") or {}
    active = [(item["value"], item["polarity"]) for item in state["active"]]
    inactive = [
        (item["value"], item["polarity"], item["status"])
        for item in state.get("inactive_raw") or ()
    ]
    if "active_values" in expected:
        checks.append(
            _check(
                "active_preferences_exact",
                active == [tuple(value) for value in expected["active_values"]],
                {"expected": expected["active_values"], "actual": active},
            )
        )
    if "active_contains" in expected:
        wanted = [tuple(value) for value in expected["active_contains"]]
        checks.append(
            _check(
                "active_contains",
                all(value in active for value in wanted),
                {"expected": wanted, "actual": active},
            )
        )
    if "active_empty" in expected:
        checks.append(
            _check("active_empty", not active, {"actual": active})
        )
    if "inactive_contains" in expected:
        wanted = [tuple(value) for value in expected["inactive_contains"]]
        checks.append(
            _check(
                "inactive_contains",
                all(value in inactive for value in wanted),
                {"expected": wanted, "actual": inactive},
            )
        )
    if "conflicts_min" in expected:
        checks.append(
            _check(
                "conflict_detected",
                len(state["conflicts"]) >= expected["conflicts_min"],
                {"conflicts": state["conflicts"]},
            )
        )
    if "conflicts_exact" in expected:
        checks.append(
            _check(
                "conflict_state_exact",
                len(state["conflicts"]) == expected["conflicts_exact"],
                {"conflicts": state["conflicts"]},
            )
        )
    if baseline_state is not None and "state_changed" in expected:
        checks.append(
            _check(
                "state_changed",
                (baseline_state["active"] != state["active"])
                == bool(expected["state_changed"]),
                {"baseline": baseline_state["active"], "mutated": state["active"]},
            )
        )
    return checks


def _ranking_checks(
    case: Mapping[str, Any],
    baseline: Mapping[str, Any],
    mutated: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected = case.get("expected_ranking") or {}
    if "top1_changed" in expected:
        checks.append(
            _check(
                "top1_changed",
                (baseline["top1"] != mutated["top1"]) == bool(expected["top1_changed"]),
                {"baseline_top1": baseline["top1"], "mutated_top1": mutated["top1"]},
            )
        )
    if "top1_equals" in expected:
        checks.append(
            _check(
                "top1_equals",
                mutated["top1"] == expected["top1_equals"],
                {"expected": expected["top1_equals"], "actual": mutated["top1"]},
            )
        )
    if "ranking_changed" in expected:
        checks.append(
            _check(
                "ranking_changed",
                (baseline["ranking"] != mutated["ranking"])
                == bool(expected["ranking_changed"]),
                {"baseline": baseline["ranking"], "mutated": mutated["ranking"]},
            )
        )
    if "ranking_equals_empty_state" in expected:
        empty_state = expected["ranking_equals_empty_state"]
        checks.append(
            _check(
                "ranking_equals_empty_state",
                mutated["ranking"] == empty_state,
                {"expected": empty_state, "actual": mutated["ranking"]},
            )
        )
    for item in expected.get("candidate_checks") or []:
        item_id = item["item_id"]
        candidate = _candidate(mutated, item_id)
        baseline_candidate = _candidate(baseline, item_id)
        label_prefix = f"candidate:{item_id}"
        if "matched_contains" in item:
            checks.append(
                _check(
                    f"{label_prefix}:matched_contains",
                    item["matched_contains"] in candidate.get("matched_preferences", []),
                    {"expected": item["matched_contains"], "actual": candidate.get("matched_preferences")},
                )
            )
        if "conflicting_contains" in item:
            checks.append(
                _check(
                    f"{label_prefix}:conflicting_contains",
                    item["conflicting_contains"] in candidate.get("conflicting_preferences", []),
                    {
                        "expected": item["conflicting_contains"],
                        "actual": candidate.get("conflicting_preferences"),
                    },
                )
            )
        if "unknown_contains" in item:
            checks.append(
                _check(
                    f"{label_prefix}:unknown_contains",
                    item["unknown_contains"] in candidate.get("unknown_preferences", []),
                    {"expected": item["unknown_contains"], "actual": candidate.get("unknown_preferences")},
                )
            )
        if "matched_absent" in item:
            checks.append(
                _check(
                    f"{label_prefix}:matched_absent",
                    item["matched_absent"] not in candidate.get("matched_preferences", []),
                    {"forbidden": item["matched_absent"], "actual": candidate.get("matched_preferences")},
                )
            )
        if "conflicting_absent" in item:
            checks.append(
                _check(
                    f"{label_prefix}:conflicting_absent",
                    item["conflicting_absent"] not in candidate.get("conflicting_preferences", []),
                    {
                        "forbidden": item["conflicting_absent"],
                        "actual": candidate.get("conflicting_preferences"),
                    },
                )
            )
        if "match_score" in item:
            checks.append(
                _check(
                    f"{label_prefix}:match_score",
                    abs(float(candidate.get("preference_match_score") or 0)
                        - float(item["match_score"])) < 1e-9,
                    {
                        "expected": item["match_score"],
                        "actual": candidate.get("preference_match_score"),
                    },
                )
            )
        if "conflict_score" in item:
            checks.append(
                _check(
                    f"{label_prefix}:conflict_score",
                    abs(float(candidate.get("preference_conflict_score") or 0)
                        - float(item["conflict_score"])) < 1e-9,
                    {
                        "expected": item["conflict_score"],
                        "actual": candidate.get("preference_conflict_score"),
                    },
                )
            )
        if "rank_direction" in item:
            before = baseline_candidate.get("rank")
            after = candidate.get("rank")
            direction = item["rank_direction"]
            passed = {
                "improved": before is not None and after is not None and after < before,
                "regressed": before is not None and after is not None and after > before,
                "unchanged": before == after,
            }[direction]
            checks.append(
                _check(
                    f"{label_prefix}:rank_{direction}",
                    passed,
                    {"rank_baseline": before, "rank_mutated": after},
                )
            )
        if "score_delta_sign" in item:
            before = baseline_candidate.get("final_score")
            after = candidate.get("final_score")
            delta = float(after) - float(before) if before is not None and after is not None else None
            sign = item["score_delta_sign"]
            passed = {
                "positive": delta is not None and delta > 1e-9,
                "negative": delta is not None and delta < -1e-9,
                "zero": delta is not None and abs(delta) <= 1e-9,
            }[sign]
            checks.append(
                _check(
                    f"{label_prefix}:score_delta_{sign}",
                    passed,
                    {"score_baseline": before, "score_mutated": after, "delta": delta},
                )
            )
    if "gate_action" in expected:
        checks.append(
            _check(
                "gate_action",
                mutated["gate"].get("action") == expected["gate_action"],
                {"expected": expected["gate_action"], "actual": mutated["gate"].get("action")},
            )
        )
    if "no_conflicts" in expected:
        checks.append(
            _check(
                "no_conflicts",
                not mutated["state"]["conflicts"],
                {"conflicts": mutated["state"]["conflicts"]},
            )
        )
    return checks


def _load_pool(task_id: str, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row = rows.get(task_id)
    if row is None:
        raise ValueError(f"task {task_id} not found in Ranking v1 dataset")
    eligible, excluded = common.eligibility(row)
    pool = {
        "task_id": task_id,
        "row": row,
        "eligible": eligible,
        "excluded": excluded,
        "pool_ids": [item["item_id"] for item in row["candidate_pool"]],
        "entries": {item["item_id"]: item for item in row["candidate_pool"]},
    }
    return pool


def _cases() -> list[dict[str, Any]]:
    """The 16 constructed counterfactual cases.

    Every expected value below is derived from the real product evidence in the
    stored fixed pool (e.g. ``B0B55YF7CL`` really has a rechargeable attribute and
    ``B0BR7MNJ7N`` really has no microphone attribute), not from an evaluator label.
    """
    return [
        {
            "case_id": "cf001_positive_rechargeable",
            "category": "positive_preference",
            "pool_id": POOL_HEADPHONES,
            "description": "Adding a positive rechargeable preference should reward the real rechargeable candidate.",
            "statements": [],
            "mutation": ["I prefer a rechargeable battery."],
            "expected_state": {
                "active_values": [["rechargeable", "positive"]],
                "state_changed": True,
            },
            "expected_ranking": {
                "top1_changed": True,
                "top1_equals": "B0B55YF7CL",
                "candidate_checks": [
                    {"item_id": "B0B55YF7CL", "matched_contains": "rechargeable", "match_score": 0.2},
                    {"item_id": "B0BLSQ5T75", "unknown_contains": "rechargeable", "match_score": 0.0},
                ],
            },
        },
        {
            "case_id": "cf002_positive_on_ear",
            "category": "positive_preference",
            "pool_id": POOL_HEADPHONES,
            "description": "Adding a positive on-ear preference should promote the on-ear candidate.",
            "statements": [],
            "mutation": ["I prefer on-ear headphones."],
            "expected_state": {"active_values": [["on_ear", "positive"]], "state_changed": True},
            "expected_ranking": {
                "top1_changed": True,
                "top1_equals": "B0BQ38B3NC",
                "candidate_checks": [
                    {"item_id": "B0BQ38B3NC", "matched_contains": "on_ear", "match_score": 0.2},
                ],
            },
        },
        {
            "case_id": "cf003_positive_microphone",
            "category": "positive_preference",
            "pool_id": POOL_HEADPHONES,
            "description": "A positive microphone preference should lift real microphone candidates without inventing evidence.",
            "statements": [],
            "mutation": ["I prefer a built-in microphone."],
            "expected_state": {"active_values": [["microphone", "positive"]], "state_changed": True},
            "expected_ranking": {
                "candidate_checks": [
                    {
                        "item_id": "B0B55YF7CL",
                        "matched_contains": "microphone",
                        "match_score": 0.2,
                        "rank_direction": "improved",
                    },
                    {"item_id": "B0BR7MNJ7N", "unknown_contains": "microphone", "match_score": 0.0},
                ]
            },
        },
        {
            "case_id": "cf004_negative_over_ear",
            "category": "negative_preference",
            "pool_id": POOL_HEADPHONES,
            "description": "A negative over-ear preference should penalise candidates with trusted over-ear evidence.",
            "statements": [],
            "mutation": ["I don't like over-ear."],
            "expected_state": {"active_values": [["over_ear", "negative"]], "state_changed": True},
            "expected_ranking": {
                "top1_changed": True,
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "conflicting_contains": "over_ear",
                        "conflict_score": 0.35,
                    }
                ],
            },
        },
        {
            "case_id": "cf005_negative_microphone",
            "category": "negative_preference",
            "pool_id": POOL_HEADPHONES,
            "description": "A negative microphone preference should penalise microphone candidates and change the Top-1.",
            "statements": [],
            "mutation": ["I don't like a built-in microphone."],
            "expected_state": {"active_values": [["microphone", "negative"]], "state_changed": True},
            "expected_ranking": {
                "top1_changed": True,
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "conflicting_contains": "microphone",
                        "conflict_score": 0.35,
                    },
                ],
            },
        },
        {
            "case_id": "cf006_update_over_ear_to_rechargeable",
            "category": "preference_update",
            "pool_id": POOL_HEADPHONES,
            "description": "An explicit correction should supersede the old over-ear preference rather than stack it.",
            "statements": ["I prefer over-ear."],
            "mutation": ["Actually, I prefer rechargeable instead of over-ear."],
            "expected_state": {
                "active_values": [["rechargeable", "positive"]],
                "inactive_contains": [["over-ear", "prefer", "superseded"]],
                "state_changed": True,
            },
            "expected_ranking": {
                "top1_changed": True,
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "matched_absent": "over_ear",
                        "unknown_contains": "rechargeable",
                    },
                    {
                        "item_id": "B0B55YF7CL",
                        "matched_contains": "rechargeable",
                        "match_score": 0.2,
                    },
                ],
            },
        },
        {
            "case_id": "cf007_update_microphone_to_rechargeable",
            "category": "preference_update",
            "pool_id": POOL_HEADPHONES,
            "description": "A correction from microphone to rechargeable should leave only the new preference active.",
            "statements": ["I prefer microphone."],
            "mutation": ["Actually, I prefer rechargeable instead of microphone."],
            "expected_state": {
                "active_values": [["rechargeable", "positive"]],
                "inactive_contains": [["microphone", "prefer", "superseded"]],
                "state_changed": True,
            },
            "expected_ranking": {
                "top1_changed": True,
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "matched_absent": "microphone",
                        "unknown_contains": "rechargeable",
                    },
                    {
                        "item_id": "B0B55YF7CL",
                        "matched_contains": "rechargeable",
                        "match_score": 0.2,
                    },
                ],
            },
        },
        {
            "case_id": "cf008_withdraw_microphone",
            "category": "preference_withdrawal",
            "pool_id": POOL_HEADPHONES,
            "description": "Retracting the microphone preference should remove its ranking effect entirely.",
            "statements": ["I prefer microphone."],
            "mutation": ["Forget my microphone preference."],
            "expected_state": {
                "active_empty": True,
                "inactive_contains": [["microphone", "prefer", "removed"]],
                "state_changed": True,
            },
            "expected_ranking": {
                "ranking_changed": True,
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "score_delta_sign": "negative",
                        "matched_absent": "microphone",
                    }
                ],
            },
        },
        {
            "case_id": "cf009_withdraw_negative_over_ear",
            "category": "preference_withdrawal",
            "pool_id": POOL_HEADPHONES,
            "description": "Retracting the negative over-ear preference should restore the over-ear candidate's score.",
            "statements": ["I don't like over-ear."],
            "mutation": ["I don't care about over-ear anymore."],
            "expected_state": {
                "active_empty": True,
                "inactive_contains": [["over-ear", "avoid", "removed"]],
                "state_changed": True,
            },
            "expected_ranking": {
                "ranking_changed": True,
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "score_delta_sign": "positive",
                        "conflicting_absent": "over_ear",
                    }
                ],
            },
        },
        {
            "case_id": "cf010_conflict_over_ear_in_ear",
            "category": "preference_conflict",
            "pool_id": POOL_HEADPHONES,
            "description": "Two mutually exclusive active preferences must trigger clarification, not a forced Top-1.",
            "statements": ["I prefer over-ear headphones."],
            "mutation": ["I prefer in-ear headphones."],
            "expected_state": {
                "active_contains": [["over_ear", "positive"], ["in_ear", "positive"]],
                "conflicts_min": 1,
                "state_changed": True,
            },
            "expected_ranking": {"gate_action": "NEED_USER_CLARIFICATION"},
            "requires_clarification": True,
        },
        {
            "case_id": "cf011_conflict_wireless_wired",
            "category": "preference_conflict",
            "pool_id": POOL_HEADPHONES,
            "description": "A wireless/wired contradiction is a conflict under the existing reader semantics.",
            "statements": ["I prefer wireless headphones."],
            "mutation": ["I prefer wired headphones."],
            "expected_state": {
                "active_contains": [["wireless", "positive"], ["wired", "positive"]],
                "conflicts_min": 1,
                "state_changed": True,
            },
            "expected_ranking": {"gate_action": "NEED_USER_CLARIFICATION"},
            "requires_clarification": True,
        },
        {
            "case_id": "cf012_unknown_positive_microphone",
            "category": "unknown_evidence",
            "pool_id": POOL_HEADPHONES,
            "description": "A candidate with no microphone evidence must be UNKNOWN, not a match and not a conflict.",
            "statements": [],
            "mutation": ["I prefer a built-in microphone."],
            "expected_state": {"active_values": [["microphone", "positive"]], "state_changed": True},
            "expected_ranking": {
                "candidate_checks": [
                    {
                        "item_id": "B0BR7MNJ7N",
                        "unknown_contains": "microphone",
                        "matched_absent": "microphone",
                        "conflicting_absent": "microphone",
                        "score_delta_sign": "zero",
                    }
                ]
            },
        },
        {
            "case_id": "cf013_unknown_negative_over_ear",
            "category": "unknown_evidence",
            "pool_id": POOL_HEADPHONES,
            "description": "A missing over-ear fact must not be treated as a negative-preference conflict.",
            "statements": [],
            "mutation": ["I don't like over-ear."],
            "expected_state": {"active_values": [["over_ear", "negative"]], "state_changed": True},
            "expected_ranking": {
                "candidate_checks": [
                    {
                        "item_id": "B0BR7MNJ7N",
                        "unknown_contains": "over_ear",
                        "conflicting_absent": "over_ear",
                        "conflict_score": 0.0,
                        "score_delta_sign": "zero",
                    }
                ]
            },
        },
        {
            "case_id": "cf014_unknown_positive_rechargeable",
            "category": "unknown_evidence",
            "pool_id": POOL_HEADPHONES,
            "description": "UNKNOWN rechargeable evidence earns no bonus while a real rechargeable candidate does.",
            "statements": [],
            "mutation": ["I prefer a rechargeable battery."],
            "expected_state": {"active_values": [["rechargeable", "positive"]], "state_changed": True},
            "expected_ranking": {
                "candidate_checks": [
                    {
                        "item_id": "B0BLSQ5T75",
                        "unknown_contains": "rechargeable",
                        "matched_absent": "rechargeable",
                        "score_delta_sign": "zero",
                    },
                    {
                        "item_id": "B0B55YF7CL",
                        "matched_contains": "rechargeable",
                        "score_delta_sign": "positive",
                    },
                ]
            },
        },
        {
            "case_id": "cf015_positive_microphone_0051",
            "category": "positive_preference",
            "pool_id": POOL_EARBUDS,
            "description": "In the 0051 pool the microphone preference lifts the observed-acceptable candidate but does not close the Top-1 gap.",
            "statements": [],
            "mutation": ["I prefer a built-in microphone."],
            "expected_state": {"active_values": [["microphone", "positive"]], "state_changed": True},
            "expected_ranking": {
                "top1_changed": False,
                "top1_equals": "B07BKPKXTT",
                "candidate_checks": [
                    {
                        "item_id": "B08D32JM5F",
                        "matched_contains": "microphone",
                        "match_score": 0.2,
                        "rank_direction": "improved",
                    },
                    {
                        "item_id": "B07BKPKXTT",
                        "unknown_contains": "microphone",
                        "matched_absent": "microphone",
                        "score_delta_sign": "zero",
                    },
                ],
            },
        },
        {
            "case_id": "cf016_unknown_negative_over_ear_0051",
            "category": "unknown_evidence",
            "pool_id": POOL_EARBUDS,
            "description": "Every eligible 0051 candidate is in-ear; the negative over-ear preference is UNKNOWN, not a conflict.",
            "statements": [],
            "mutation": ["I don't like over-ear."],
            "expected_state": {"active_values": [["over_ear", "negative"]], "state_changed": True},
            "expected_ranking": {
                "ranking_changed": False,
                "no_conflicts": True,
                "candidate_checks": [
                    {
                        "item_id": "B07BKPKXTT",
                        "unknown_contains": "over_ear",
                        "conflicting_absent": "over_ear",
                        "conflict_score": 0.0,
                        "score_delta_sign": "zero",
                    },
                    {
                        "item_id": "B08D32JM5F",
                        "unknown_contains": "over_ear",
                        "conflicting_absent": "over_ear",
                        "conflict_score": 0.0,
                        "score_delta_sign": "zero",
                    },
                ],
            },
        },
    ]


def _execute_case(
    case: Mapping[str, Any],
    pools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pool = pools[case["pool_id"]]
    query = pool["row"]["query"]
    baseline_memory = _memory_state(case["case_id"], case["statements"], query)
    mutated_memory = _memory_state(
        case["case_id"], list(case["statements"]) + list(case["mutation"]), query
    )
    baseline = _run(pool, baseline_memory["active_preferences"])
    mutated = _run(pool, mutated_memory["active_preferences"])
    baseline_checks_state = {
        "active": baseline["state"]["active"],
        "conflicts": baseline["state"]["conflicts"],
        "inactive_raw": baseline_memory["inactive_raw"],
    }
    mutated_checks_state = {
        "active": mutated["state"]["active"],
        "conflicts": mutated["state"]["conflicts"],
        "inactive_raw": mutated_memory["inactive_raw"],
    }
    checks = _state_checks(
        case, mutated_checks_state, baseline_state=baseline_checks_state
    ) + _ranking_checks(case, baseline, mutated)
    checks.append(
        _check(
            "no_hard_constraint_violation",
            not any(mutated["hard_constraint_violations"].values()),
            mutated["hard_constraint_violations"],
        )
    )
    requires_clarification = bool(case.get("requires_clarification", False))
    if requires_clarification:
        checks.append(
            _check(
                "clarification_required",
                mutated["gate"].get("action") == "NEED_USER_CLARIFICATION",
                {"gate_action": mutated["gate"].get("action")},
            )
        )
    passed = all(check["passed"] for check in checks)
    return {
        "schema": common.SCHEMA_COUNTERFACTUAL,
        "case_id": case["case_id"],
        "category": case["category"],
        "pool_id": case["pool_id"],
        "description": case["description"],
        "query": query,
        "hard_requirements": pool["row"].get("hard_requirements") or [],
        "candidate_pool_ids": pool["pool_ids"],
        "eligible_candidate_ids": pool["eligible"],
        "excluded_candidates": pool["excluded"],
        "same_pool_sha256": hashlib.sha256(
            json.dumps(pool["pool_ids"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "baseline_memory_sessions": baseline_memory["sessions"],
        "mutated_memory_sessions": mutated_memory["sessions"],
        "baseline_memory_state": {
            key: baseline_memory[key]
            for key in ("ingest_summary", "active_raw", "inactive_raw", "active_preferences", "conflicts")
        },
        "mutated_memory_state": {
            key: mutated_memory[key]
            for key in ("ingest_summary", "active_raw", "inactive_raw", "active_preferences", "conflicts")
        },
        "baseline": baseline,
        "mutated": mutated,
        "checks": checks,
        "checks_passed": sum(1 for check in checks if check["passed"]),
        "checks_total": len(checks),
        "passed": passed,
        "hard_constraint_violations": mutated["hard_constraint_violations"],
        "requires_clarification": requires_clarification,
        "top1_gold_allowed": not requires_clarification,
        "label_status": "constructed_state_consistency_test_not_user_satisfaction_gold",
    }


def main() -> int:
    rows = {
        row["task_id"]: row
        for row in common.read_jsonl(common.RANKING_DATASET)
        if row["expected_decision"] == "recommend"
    }
    pools = {
        POOL_HEADPHONES: _load_pool(POOL_HEADPHONES, rows),
        POOL_EARBUDS: _load_pool(POOL_EARBUDS, rows),
    }
    results = [_execute_case(case, pools) for case in _cases()]
    summary = _summary(results)

    common.write_jsonl(OUT / "counterfactual_cases.jsonl", results)
    common.write_json(OUT / "counterfactual_summary.json", summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if summary["passed_cases"] == summary["cases"] else 1


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for case in results:
        bucket = by_category.setdefault(
            case["category"], {"cases": 0, "passed": 0, "checks": 0, "checks_passed": 0}
        )
        bucket["cases"] += 1
        bucket["passed"] += int(case["passed"])
        bucket["checks"] += case["checks_total"]
        bucket["checks_passed"] += case["checks_passed"]
    failures = [
        {
            "case_id": case["case_id"],
            "failed_checks": [check for check in case["checks"] if not check["passed"]],
        }
        for case in results
        if not case["passed"]
    ]
    total_checks = sum(case["checks_total"] for case in results)
    passed_checks = sum(case["checks_passed"] for case in results)
    ranking_changed = sum(
        1 for case in results if case["baseline"]["ranking"] != case["mutated"]["ranking"]
    )
    top1_changed = sum(
        1 for case in results if case["baseline"]["top1"] != case["mutated"]["top1"]
    )
    return {
        "schema": common.SCHEMA_COUNTERFACTUAL_SUMMARY,
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cases": len(results),
        "passed_cases": sum(1 for case in results if case["passed"]),
        "failed_cases": len(failures),
        "memory_consistency_pass_rate": (
            round(sum(1 for case in results if case["passed"]) / len(results), 6)
            if results
            else 0.0
        ),
        "checks_total": total_checks,
        "checks_passed": passed_checks,
        "check_pass_rate": round(passed_checks / total_checks, 6) if total_checks else 0.0,
        "ranking_changed_cases": ranking_changed,
        "top1_changed_cases": top1_changed,
        "by_category": dict(sorted(by_category.items())),
        "requires_clarification_cases": [
            case["case_id"] for case in results if case["requires_clarification"]
        ],
        "clarification_handled": sum(
            1
            for case in results
            if case["requires_clarification"]
            and case["mutated"]["gate"].get("action") == "NEED_USER_CLARIFICATION"
        ),
        "hard_constraint_violations": sum(
            sum(case["hard_constraint_violations"].values()) for case in results
        ),
        "cases_with_hard_constraint_violation": sum(
            1 for case in results if any(case["hard_constraint_violations"].values())
        ),
        "top1_gold_withheld_cases": [
            case["case_id"] for case in results if not case["top1_gold_allowed"]
        ],
        "failures": failures,
        "disclaimer": (
            "Constructed memory-state tests only. Passing them proves the legal memory "
            "write/read path and the fixed-weight Scorer respond consistently to state "
            "changes; it does not establish real user satisfaction or recommendation quality."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
