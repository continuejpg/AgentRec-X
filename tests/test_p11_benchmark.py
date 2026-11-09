"""P11 benchmark integrity tests: schema, split, evidence levels, no leaked targets."""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK = Path("data/p11_benchmark")


def load() -> tuple[list[dict], dict, dict]:
    tasks = [json.loads(line) for line in (BENCHMARK / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads((BENCHMARK / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((BENCHMARK / "schema.json").read_text(encoding="utf-8"))
    return tasks, manifest, schema


def test_required_fields_and_no_targets() -> None:
    tasks, manifest, schema = load()
    required = set(schema["required_fields"])
    assert tasks and len(tasks) == manifest["task_count"]
    for task in tasks:
        assert required <= set(task)
        assert task["split"] == "dev"
        assert task["user_history_source"] == "train_history_only"
        assert "validation_target" not in task
        assert "test_target" not in task
        assert "answer" not in task
        assert "target" not in task


def test_evidence_levels_never_claim_a_without_review_text() -> None:
    tasks, manifest, _ = load()
    assert manifest["level_a_available"] is False
    assert {task["preference_evidence_level"] for task in tasks} <= {"B", "C", "UNKNOWN"}
    assert all(task["preference_evidence_level"] != "A" for task in tasks)


def test_same_query_pairs_are_exact_and_have_evidence_contrast() -> None:
    tasks, _, _ = load()
    pairs: dict[str, list[dict]] = {}
    for task in tasks:
        pair_id = task.get("pair_id")
        if pair_id:
            pairs.setdefault(pair_id, []).append(task)
    assert pairs
    for rows in pairs.values():
        queries = {row["conversation_turns"][0] for row in rows}
        assert len(queries) == 1
        roles = {row.get("pair_role") for row in rows}
        if {"evidence", "unknown"} <= roles:
            assert {row["preference_evidence_level"] for row in rows if row.get("pair_role") == "unknown"} == {"UNKNOWN"}


def test_no_answer_tasks_have_no_fabricated_pool() -> None:
    tasks, _, _ = load()
    for task in tasks:
        if task["task_type"] in {"clarification_required", "unsatisfiable_or_unknown"}:
            assert task["verified_candidate_pool"] == []
            assert task["expected_action_when_verifiable"] in {"ASK", "UNKNOWN"}
            assert task["adjudication_status"] == "needs_human"


def test_pool_candidates_have_catalog_evidence_fields() -> None:
    tasks, _, _ = load()
    pool_tasks = [task for task in tasks if task["verified_candidate_pool"]]
    assert pool_tasks
    for task in pool_tasks[:5]:
        for candidate in task["verified_candidate_pool"]:
            assert candidate["parent_asin"]
            assert candidate["role_status"] == "MATCH"
            assert "price_text" in candidate
            assert candidate["verification_fields"]
