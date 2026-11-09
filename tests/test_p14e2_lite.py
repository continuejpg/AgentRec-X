"""P14-E2 Lite contract tests: counts, split/persona/family isolation, provenance, leakage, reserved guard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/p14e2/benchmark"
ORIGINAL = REPO / "data/p14e/benchmark"
VERSION = "agentrecx.p14e2.shopping_preference.v0.1"
SPLITS = ("train", "dev", "reserved_test")
TARGET_TYPES = {
    "no_history_recommend": 15, "history_influenced": 20, "same_query_persona_pair": 16,
    "preference_update": 4, "preference_withdrawal": 4, "preference_conflict": 4,
    "insufficient_information": 10, "no_verified_match": 4, "unverifiable_constraint": 3,
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_counts_splits_and_task_types():
    agent = {split: _load(DATA / f"{split}.agent.jsonl") for split in SPLITS}
    evaluator = {split: _load(DATA / f"{split}.evaluator.jsonl") for split in SPLITS}
    assert sum(len(rows) for rows in agent.values()) == 80
    assert {split: len(rows) for split, rows in agent.items()} == {"train": 40, "dev": 20, "reserved_test": 20}
    for split in SPLITS:
        assert len(agent[split]) == len(evaluator[split])
        assert {row["schema"] for row in agent[split]} == {VERSION}
        assert {row["split"] for row in agent[split]} == {split}
        assert all(re.fullmatch(r"p14e2_lite_\d{4}", row["task_id"]) for row in agent[split])
    types = {}
    for split in SPLITS:
        for row in evaluator[split]:
            types[row["task_type"]] = types.get(row["task_type"], 0) + 1
    assert types == TARGET_TYPES


def test_personas_and_families_do_not_cross_splits():
    personas = _load(DATA / "personas.jsonl")
    assert len(personas) == 80
    assert len({row["persona_id"] for row in personas}) == 80
    original = {row["persona_id"] for row in _load(ORIGINAL / "personas.jsonl")}
    assert not ({row["persona_id"] for row in personas} & original)
    family_split = {}
    for split in SPLITS:
        for task in _load(DATA / f"{split}.agent.jsonl"):
            family = _load(DATA / f"{split}.evaluator.jsonl")
            family_by_id = {ev["task_id"]: ev for ev in family}
            family_id = family_by_id[task["task_id"]]["family_id"]
            family_split.setdefault(family_id, split)
            assert family_split[family_id] == split


def test_designed_same_query_pairs_are_complete_pairs_with_different_preferences():
    eval_all = {split: {ev["task_id"]: ev for ev in _load(DATA / f"{split}.evaluator.jsonl")} for split in SPLITS}
    groups = {}
    for split in SPLITS:
        for task in _load(DATA / f"{split}.agent.jsonl"):
            groups.setdefault(task["current_query"], []).append((split, task))
    designed = 0
    for query, members in groups.items():
        evs = [eval_all[split][task["task_id"]] for split, task in members]
        pair_evs = [ev for ev in evs if ev["task_type"] == "same_query_persona_pair"]
        if not pair_evs:
            continue
        designed += 1
        assert len(members) == 2
        assert len({split for split, _ in members}) == 1
        left = {(p["kind"], p["value"], p["polarity"]) for p in pair_evs[0]["effective_preferences"]}
        right = {(p["kind"], p["value"], p["polarity"]) for p in pair_evs[1]["effective_preferences"]}
        assert left != right
    assert designed == 8


def test_provenance_replay_matches_evaluator_labels():
    from experiments.p14e.labeling import active_prefs

    events = _load(DATA / "history_events.jsonl")
    by_task = {}
    for event in events:
        match = re.match(r"^(p14e2_lite_\d{4})_", event["event_id"])
        assert match, event["event_id"]
        by_task.setdefault(match.group(1), []).append(event)
    for split in SPLITS:
        for task in _load(DATA / f"{split}.agent.jsonl"):
            ev = next(item for item in _load(DATA / f"{split}.evaluator.jsonl") if item["task_id"] == task["task_id"])
            effective, withdrawn = active_prefs(sorted(by_task[task["task_id"]], key=lambda e: e["sequence"]))
            assert {(p["kind"], p["value"], p["polarity"]) for p in effective} == {
                (p["kind"], p["value"], p["polarity"]) for p in ev["effective_preferences"]}
            assert {(p["kind"], p["value"]) for p in withdrawn} <= {
                (p["kind"], p["value"]) for p in ev["withdrawn_preferences"]}


def test_agent_visible_files_do_not_leak_evaluator_fields():
    forbidden = ("expected_decision", "verified_candidate", "persona_preferences", "effective_preferences",
                 "withdrawn_preferences", "task_type", "family_id", "label_basis", "intent_note")
    for split in SPLITS:
        for task in _load(DATA / f"{split}.agent.jsonl"):
            text = json.dumps(task, ensure_ascii=False)
            for marker in forbidden:
                assert marker not in text, f"{marker} leaked in {task['task_id']}"
            assert "expected_decision" not in task and "effective_preferences" not in task


def test_no_history_tasks_are_recommendable_with_explicit_criteria():
    for split in SPLITS:
        for ev in _load(DATA / f"{split}.evaluator.jsonl"):
            if ev["task_type"] != "no_history_recommend":
                continue
            assert ev["expected_decision"] == "recommend"
            assert ev["effective_preferences"] == []
            assert ev["hard_constraints"], ev["task_id"]
            assert ev["labels"]["verified_candidate_count"] > 0


def test_ask_and_evidence_families_have_expected_labels():
    asks = evidence = 0
    for split in SPLITS:
        for ev in _load(DATA / f"{split}.evaluator.jsonl"):
            if ev["task_type"] == "insufficient_information":
                asks += 1
                assert ev["expected_decision"] == "clarification_required"
                assert ev["effective_preferences"] == [] and ev["hard_constraints"] == []
            if ev["task_type"] in {"no_verified_match", "unverifiable_constraint"}:
                evidence += 1
                assert ev["expected_decision"] in {"no_verified_match", "insufficient_evidence"}
    assert asks == 10 and evidence == 7


def test_reserved_test_has_no_inference_artifacts():
    reserved = {task["task_id"] for task in _load(DATA / "reserved_test.agent.jsonl")}
    smoke = REPO / "runs/p14e2/smoke"
    if smoke.is_dir():
        for path in smoke.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            assert not any(task_id in text for task_id in reserved)


def test_frozen_pilot_and_original_labels_unchanged():
    # P14-E2 must not modify the frozen P14-E pilot data.
    manifest = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf-8"))
    for name in ("agent_tasks.jsonl", "evaluator_only.jsonl"):
        digest = hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest()
        assert digest == manifest["files"][name]["sha256"]
    assert (REPO / "runs/p14b0/search_preferences.jsonl").is_file()
    assert (REPO / "runs/p14b1/summary.json").is_file()


def test_review_checklist_marks_pending_but_not_human_verified():
    review = json.loads((DATA / "review_checklist.json").read_text(encoding="utf-8"))
    pending = 0
    for split in SPLITS:
        for ev in _load(DATA / f"{split}.evaluator.jsonl"):
            meta = ev["construction_metadata"]
            assert meta["review_status"] in {"automated_checked", "pending_human_review"}
            if meta["review_status"] == "pending_human_review":
                pending += 1
    assert pending == len(review["tasks"])
    assert pending > 0
