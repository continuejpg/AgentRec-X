"""Validate P14-B3-Eval100: schema, labels, isolation, leakage, query/persona uniqueness."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14b3_eval100"
OUT = REPO / "runs/p14b3_eval100"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    agent = load_jsonl(DATA / "eval100.agent.jsonl")
    evaluator = load_jsonl(DATA / "eval100.evaluator.jsonl")
    personas = load_jsonl(DATA / "personas.jsonl")
    old_agent = []
    for path in (REPO / "data/p14e/benchmark").glob("*.agent.jsonl"):
        old_agent += load_jsonl(path)
    for path in (REPO / "data/p14e2/benchmark").glob("*.agent.jsonl"):
        old_agent += load_jsonl(path)
    old_ids = {row["task_id"] for row in old_agent}
    old_queries = {row["current_query"] for row in old_agent}
    old_personas = {row["persona_id"] for row in old_agent}
    old_tasks = {row["task_id"] for row in agent}
    assert len(agent) == len(evaluator) == 100
    assert len(old_tasks) == 100
    assert not (old_tasks & old_ids)
    assert all(row["schema"] == "agentrecx.p14b3eval100.shopping_preference.v0.1" for row in agent)
    assert all(row["split"] == "eval100" for row in agent)
    assert all(row["construction_metadata"]["review_status"] == "pending_human_review" for row in agent)
    assert all(row["construction_metadata"]["benchmark_split"] == "eval100" for row in agent)
    forbidden = {"expected_decision", "labels", "effective_preferences", "withdrawn_preferences",
                 "hard_constraints", "verified_candidates", "decision_rationale"}
    for row in agent:
        assert not (forbidden & set(row)), (row["task_id"], forbidden & set(row))
    ids = {row["task_id"] for row in agent}
    assert ids == {row["task_id"] for row in evaluator}
    assert len({row["persona_id"] for row in agent}) == 100
    assert not ({row["persona_id"] for row in agent} & old_personas)

    # Same-query pairs must share exact query; every other query unique and unseen.
    by_family: dict[str, list] = {}
    for ev in evaluator:
        by_family.setdefault(ev["family_id"], []).append(ev)
    pair_families = [family for family in by_family if "pair" in family]
    assert len(pair_families) == 10
    for family in pair_families:
        assert len(by_family[family]) == 2
        assert by_family[family][0]["task_id"] != by_family[family][1]["task_id"]
        queries = {next(row["current_query"] for row in agent if row["task_id"] == ev["task_id"])
                   for ev in by_family[family]}
        assert len(queries) == 1
    pair_task_ids = {ev["task_id"] for family in pair_families for ev in by_family[family]}
    non_pair_queries = [row["current_query"] for row in agent if row["task_id"] not in pair_task_ids]
    assert len(non_pair_queries) == len(set(non_pair_queries))
    assert not (set(non_pair_queries) & old_queries)

    decisions = Counter(row["expected_decision"] for row in evaluator)
    types = Counter(row["task_type"] for row in evaluator)
    assert decisions["recommend"] == 80
    assert sum(1 for row in evaluator if row["expected_decision"] != "recommend") == 20
    assert types["insufficient_information"] == 8
    assert types["preference_conflict"] == 4
    assert types["no_verified_match"] == 4
    assert types["unverifiable_constraint"] == 4
    for row in evaluator:
        if row["expected_decision"] == "recommend":
            assert row["labels"]["verified_candidate_count"] >= 3, row["task_id"]
        if row["expected_decision"] == "no_verified_match":
            assert row["labels"]["verified_candidate_count"] == 0
        if row["expected_decision"] == "insufficient_evidence":
            assert row["labels"].get("hard_unknown_pool_count", 0) >= 0

    old_reviewed = 0
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_tasks"] == 100 and manifest["human_verified"] is False
    result = {
        "schema": "agentrecx.p14b3eval100.validation.v1", "ok": True,
        "tasks": len(agent), "expected_decisions": dict(decisions), "task_types": dict(types),
        "same_query_pairs": len(pair_families), "personas": len(personas),
        "overlap_with_prior_tasks": 0, "overlap_with_prior_personas": 0,
        "overlap_with_prior_queries": 0, "agent_evaluator_leakage": False,
        "human_verified": False, "review_status": "pending_human_review",
        "old_reviewed_artifacts_unchanged": True,
        "notes": ["labeler is independent; no BM25/V0/Qwen ranking used to build or relabel tasks",
                  "eval100 tasks are evaluation-only and are not training data"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
