"""P14-E benchmark pilot contract tests (fast; no network, no LLM, no full Agent run)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import active_prefs, assess_candidate, label_task

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14e/benchmark"
FORBIDDEN_AGENT_KEYS = {
    "effective_preferences", "withdrawn_preferences", "expected_decision", "decision_rationale",
    "labels", "verified_candidates", "hard_constraints", "persona",
}
FORBIDDEN_AGENT_SUBSTRINGS = (
    "expected_decision", "verified_candidate", "acceptable_product", "no_verified_match",
    "clarification_required", "insufficient_evidence", "task_type", "intent_note",
)


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def data():
    return {
        "agent": _load("agent_tasks.jsonl"),
        "evaluator": _load("evaluator_only.jsonl"),
        "personas": _load("personas.jsonl"),
        "events": _load("history_events.jsonl"),
        "manifest": json.loads((DATA / "manifest.json").read_text(encoding="utf-8")),
        "review": json.loads((DATA / "review_checklist.json").read_text(encoding="utf-8")),
    }


@pytest.fixture(scope="module")
def catalogue() -> ElectronicsCatalogue:
    return ElectronicsCatalogue.load()


def test_counts_and_task_type_distribution(data):
    agent, evaluator = data["agent"], data["evaluator"]
    assert len(agent) == 40 and len(evaluator) == 40
    assert {task["split"] for task in agent} == {"pilot_dev"}
    assert {task["task_id"] for task in agent} == {ev["task_id"] for ev in evaluator}
    types = {}
    for ev in evaluator:
        types[ev["task_type"]] = types.get(ev["task_type"], 0) + 1
    assert types["regular_history"] == 12
    assert types["same_query_persona_pair"] == 10
    assert types["preference_update"] + types["preference_withdrawal"] + types["preference_conflict"] == 8
    assert types["insufficient_information"] == 5
    assert types["no_verified_match"] + types["unverifiable_constraint"] == 5
    assert data["manifest"]["counts"]["tasks"] == 40
    assert data["manifest"]["random_seed"] == 20261108


def test_agent_visible_is_schema_shaped_and_has_no_evaluator_data(data):
    for task in data["agent"]:
        assert task["schema"] == "agentrecx.p14e.shopping_preference.v0.1"
        assert task["catalog_namespace"] == ELECTRONICS_NAMESPACE
        assert task["allowed_tools"] and task["tool_budget"]["max_tool_calls"] > 0
        assert task["history_sessions"] and task["current_query"].strip()
        assert set(task) & FORBIDDEN_AGENT_KEYS == set()
        text = json.dumps(task, ensure_ascii=False)
        for marker in FORBIDDEN_AGENT_SUBSTRINGS:
            assert marker not in text, f"{marker!r} leaked into agent-visible task {task['task_id']}"


def test_history_provenance_replays_to_evaluator_effective_preferences(data):
    events_by_task = {}
    for event in data["events"]:
        match = re.match(r"^(p14e_pilot_\d{4})_", str(event["event_id"]))
        assert match, event["event_id"]
        events_by_task.setdefault(match.group(1), []).append(event)
    for agent, ev in zip(data["agent"], data["evaluator"]):
        task_events = sorted(events_by_task[agent["task_id"]], key=lambda item: item["sequence"])
        assert task_events, agent["task_id"]
        effective, withdrawn = active_prefs(task_events)
        assert {(p["kind"], p["value"], p["polarity"]) for p in effective} == {
            (p["kind"], p["value"], p["polarity"]) for p in ev["effective_preferences"]
        }
        assert {(p["kind"], p["value"]) for p in withdrawn} <= {
            (p["kind"], p["value"]) for p in ev["withdrawn_preferences"]
        }
        history_text = [
            turn["content"] for session in agent["history_sessions"] for turn in session["turns"]
        ]
        assert history_text == [event["raw_text"] for event in task_events]


def test_catalogue_identity_and_attribute_contract(catalogue):
    stats = catalogue.stats
    assert stats["rows"] == 76918
    assert stats["accepted"] > 70_000
    assert stats["quarantined"] > 0
    seen = set()
    for product in list(catalogue.by_type("headphones"))[:200]:
        assert product.item_id not in seen
        seen.add(product.item_id)
        assert len(product.item_id) == 10 and product.item_id.isalnum()
        assert product.namespaced_id == f"{ELECTRONICS_NAMESPACE}:{product.item_id}"
        assert product.record is not None
        assert product.record.main_category == "Electronics"
        assert product.title
    first = next(iter(catalogue.by_type("headphones")))
    attrs_a, form_a = (first.attributes, first.form_factor)
    attrs_b, form_b = (first.attributes, first.form_factor)
    assert attrs_a == attrs_b and form_a == form_b  # deterministic extraction


def test_catalogue_search_is_real_and_stable(catalogue):
    result = catalogue.search("noise cancelling headphones", limit=10)
    assert result.catalogue_size == catalogue.catalogue_size
    assert result.hits
    for hit in result.hits:
        assert hit.parent_asin.startswith(f"{ELECTRONICS_NAMESPACE}:")
        assert hit.score_kind if hasattr(hit, "score_kind") else True


def test_same_query_pairs_have_different_preferences_and_verified_sets(data, catalogue):
    agent_by_id = {task["task_id"]: task for task in data["agent"]}
    groups: dict[str, list[dict]] = {}
    for ev in data["evaluator"]:
        if ev["task_type"] == "same_query_persona_pair":
            groups.setdefault(agent_by_id[ev["task_id"]]["current_query"], []).append(ev)
    assert len(groups) == 5
    for evs in groups.values():
        assert len(evs) == 2
        left, right = evs
        left_prefs = {(p["kind"], p["value"], p["polarity"]) for p in left["effective_preferences"]}
        right_prefs = {(p["kind"], p["value"], p["polarity"]) for p in right["effective_preferences"]}
        assert left_prefs != right_prefs

        def verified(ev):
            pool = catalogue.by_type(ev["product_type_pool"])
            return {
                product.item_id for product in pool
                if assess_candidate(
                    product, hard_constraints=ev["hard_constraints"],
                    preferences=ev["effective_preferences"],
                ).acceptable
            }

        assert verified(left) != verified(right)


def test_labels_are_deterministic_and_consistent_with_stored(data, catalogue):
    for ev in data["evaluator"][:6]:
        pool = catalogue.by_type(ev["product_type_pool"])
        labels = label_task({}, {"effective_preferences": ev["effective_preferences"],
                                "hard_constraints": ev["hard_constraints"]}, pool)
        assert labels["expected_decision"] == ev["expected_decision"]
        assert labels["verified_candidate_count"] == ev["labels"]["verified_candidate_count"]


def test_review_queue_matches_pending_flags(data):
    pending = [
        ev["task_id"] for ev in data["evaluator"]
        if ev["construction_metadata"]["review_status"] == "pending_human_review"
    ]
    queued = [item["task_id"] for item in data["review"]["tasks"]]
    assert sorted(pending) == sorted(queued)
    assert all(item["reasons"] for item in data["review"]["tasks"])
