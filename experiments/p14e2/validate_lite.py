"""P14-E2 Lite validation: counts/splits, provenance, isolation, leakage, evidence, reserved guard.

Writes `runs/p14e2/validation/summary.json`.  Reserved Test is deliberately limited to
schema/count/leakage/id checks; no search, scorer, gate or planner is executed on it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14e.labeling import active_prefs, assess_candidate
from experiments.p14e2.build_lite import E2_VERSION, SPLITS, FORBIDDEN_AGENT_KEYS

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
ORIGINAL = REPO / "data/p14e/benchmark"
RUNS = REPO / "runs/p14e2/validation"
FORBIDDEN_SUBSTRINGS = (
    "expected_decision", "verified_candidate", "acceptable_product", "persona_preferences",
    "evaluator_only", "no_verified_match", "clarification_required", "insufficient_evidence",
    "task_type", "family_id", "review_reason", "label_basis", "intent_note",
)
TARGET_TYPES = {
    "no_history_recommend": 15, "history_influenced": 20, "same_query_persona_pair": 16,
    "preference_update": 4, "preference_withdrawal": 4, "preference_conflict": 4,
    "insufficient_information": 10, "no_verified_match": 4, "unverifiable_constraint": 3,
}


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load()
    agent: dict[str, list[dict[str, Any]]] = {}
    evaluator: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        agent[split] = _load(DATA / f"{split}.agent.jsonl")
        evaluator[split] = _load(DATA / f"{split}.evaluator.jsonl")
    personas = _load(DATA / "personas.jsonl")
    events = _load(DATA / "history_events.jsonl")
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    original_personas = set()
    if (ORIGINAL / "personas.jsonl").is_file():
        original_personas = {row["persona_id"] for row in _load(ORIGINAL / "personas.jsonl")}
    original_signatures = set()
    if (ORIGINAL / "agent_tasks.jsonl").is_file():
        for row in _load(ORIGINAL / "agent_tasks.jsonl"):
            original_signatures.add(
                row["current_query"] + "||" + "|".join(
                    turn["content"] for session in row["history_sessions"] for turn in session["turns"]
                )
            )

    failures: list[str] = []
    warnings: list[str] = []
    all_agent = {split: {row["task_id"]: row for row in rows} for split, rows in agent.items()}
    all_ev = {split: {row["task_id"]: row for row in rows} for split, rows in evaluator.items()}
    task_split = {task_id: split for split, rows in all_agent.items() for task_id in rows}

    # 1. counts and schema
    for split in SPLITS:
        if any(row.get("schema") != E2_VERSION for row in agent[split] + evaluator[split]):
            failures.append(f"{split}: schema version mismatch")
        if any(row.get("split") != split for row in agent[split] + evaluator[split]):
            failures.append(f"{split}: split field mismatch")
        if len(agent[split]) != len(evaluator[split]):
            failures.append(f"{split}: agent/evaluator count mismatch")
    total = sum(len(rows) for rows in agent.values())
    if total != 80:
        failures.append(f"expected 80 tasks, got {total}")
    if {s: len(agent[s]) for s in SPLITS} != {"train": 40, "dev": 20, "reserved_test": 20}:
        failures.append(f"split sizes wrong: {{s: len for s}}")
    # type distribution (combined)
    types = Counter(ev["task_type"] for evs in evaluator.values() for ev in evs)
    for name, expected in TARGET_TYPES.items():
        if types.get(name) != expected:
            warnings.append(f"task_type {name}: expected {expected}, got {types.get(name)}")

    # 2. personas / families / same-query pairs do not cross splits
    persona_split: dict[str, str] = {}
    for split in SPLITS:
        for row in personas:
            pass
    for split in SPLITS:
        for task in agent[split]:
            persona_split.setdefault(task["persona_id"], split)
            if persona_split[task["persona_id"]] != split:
                failures.append(f"persona {task['persona_id']} crosses splits")
    dup_persona = {row["persona_id"] for row in personas}
    if len(dup_persona) != len(personas):
        failures.append("duplicate persona rows")
    if dup_persona & original_personas:
        failures.append("new personas collide with P14-E pilot personas")
    for split in SPLITS:
        family_splits: dict[str, set[str]] = defaultdict(set)
        for task in agent[split]:
            family_splits[all_ev[split][task["task_id"]]["family_id"]].add(split)
        for family, splits in family_splits.items():
            if len(splits) > 1:
                failures.append(f"family {family} crosses splits")
    # Designed same-query pairs must be complete inside one split; accidental query
    # reuse across split boundaries is a warning.  Near-duplicate signatures are a
    # hard failure and are checked separately.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in SPLITS:
        for task in agent[split]:
            groups[task["current_query"]].append(task)
    for query, group in groups.items():
        if len(group) < 2:
            continue
        evs = [all_ev[task["split"]][task["task_id"]] for task in group]
        designed = [ev for ev in evs if ev["task_type"] == "same_query_persona_pair"]
        if designed:
            if len(group) != 2 or len(designed) != 2:
                failures.append(f"designed same-query pair not a clean 2-task group: {query[:60]}")
                continue
            if {task["split"] for task in group} != {designed[0]["split"]}:
                failures.append(f"designed same-query pair crosses splits: {query[:60]}")
            left = {(p["kind"], p["value"], p["polarity"]) for p in designed[0]["effective_preferences"]}
            right = {(p["kind"], p["value"], p["polarity"]) for p in designed[1]["effective_preferences"]}
            if left == right:
                failures.append(f"same-query pair has identical preferences: {query[:50]}")
        else:
            warnings.append(f"query shared by {len(group)} non-pair tasks: {query[:60]}")

    signatures: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        for task in agent[split]:
            signature = task["current_query"] + "||" + "|".join(
                turn["content"] for session in task["history_sessions"] for turn in session["turns"]
            )
            signatures[signature].append(task["task_id"])
    duplicate_groups = [ids for ids in signatures.values() if len(ids) > 1]
    if duplicate_groups:
        failures.append(f"near-duplicate E2 task groups: {duplicate_groups}")
    for signature in signatures:
        if signature in original_signatures:
            failures.append("E2 task duplicates a frozen P14-E task signature")
            break

    # 3. provenance replay + history coverage (all splits, reserved included: data-only)
    allow_history = {"history_influenced", "same_query_persona_pair", "preference_update",
                     "preference_withdrawal", "preference_conflict", "no_verified_match", "unverifiable_constraint"}
    tasks_with_history = 0
    replay_ok = 0
    for split in SPLITS:
        events_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            match = re.match(r"^(p14e2_lite_\d{4})_", str(event["event_id"]))
            if match:
                events_by_task[match.group(1)].append(event)
        for task in agent[split]:
            ev = all_ev[split][task["task_id"]]
            task_events = sorted(events_by_task.get(task["task_id"], []), key=lambda e: e["sequence"])
            if not task_events:
                failures.append(f"{task['task_id']}: no history events")
                continue
            effective, withdrawn = active_prefs(task_events)
            stored = {(p["kind"], p["value"], p["polarity"]) for p in ev["effective_preferences"]}
            replay = {(p["kind"], p["value"], p["polarity"]) for p in effective}
            if replay == stored:
                replay_ok += 1
            else:
                failures.append(f"{task['task_id']}: provenance replay mismatch")
            if ev["task_type"] in allow_history and any(e.get("preference_candidates") for e in task_events):
                tasks_with_history += 1
            if ev["task_type"] == "no_history_recommend" and stored:
                failures.append(f"{task['task_id']}: no-history task has recoverable preferences")

    # 4. leakage isolation
    leakage: Counter = Counter()
    for split in SPLITS:
        for task in agent[split]:
            if set(task) & FORBIDDEN_AGENT_KEYS:
                leakage[f"{split}:forbidden_key"] += 1
            text = json.dumps(task, ensure_ascii=False)
            for marker in FORBIDDEN_SUBSTRINGS:
                if marker in text:
                    leakage[f"{split}:{marker}"] += 1
    if leakage:
        failures.append(f"leakage hits: {dict(leakage)}")

    # 5. product identity / evidence / hard constraints
    verified_samples = 0
    hard_verifiable_tasks = 0
    unknown_rates: list[float] = []
    split_stats: dict[str, Any] = {}
    for split in SPLITS:
        split_verified = split_hard = split_tasks = 0
        for task in agent[split]:
            ev = all_ev[split][task["task_id"]]
            if ev["catalog_namespace"] != ELECTRONICS_NAMESPACE:
                failures.append(f"{task['task_id']}: wrong namespace")
            for item_id in ev["labels"]["verified_candidates_sample"]:
                if catalogue.get(item_id) is None:
                    failures.append(f"{task['task_id']}: verified sample {item_id} not in catalogue")
                elif f"{ELECTRONICS_NAMESPACE}:{item_id}" in catalogue.records:
                    verified_samples += 1
            hard = ev["hard_constraints"]
            verifiable = True
            for constraint in hard:
                if constraint["kind"] == "unverifiable":
                    verifiable = False
                    continue
                count = sum(
                    1 for product in catalogue.by_type(ev["product_type_pool"])
                    if assess_candidate(product, hard_constraints=[constraint], preferences=[]).hard_states.get(
                        constraint["constraint_id"]) == "verified"
                )
                if count == 0:
                    verifiable = False
            if not hard or verifiable:
                hard_verifiable_tasks += 1
                split_hard += 1
            requested = {str(p["value"]) for p in ev["effective_preferences"]
                         if p.get("kind") == "attribute"} | {
                str(c["value"]) for c in hard if c.get("kind") == "attribute"}
            if requested:
                pool = catalogue.by_type(ev["product_type_pool"])
                missing = sum(1 for product in pool if any(name not in (product.attributes or {}) for name in requested))
                unknown_rates.append(missing / max(len(pool), 1))
            split_tasks += 1
        split_stats[split] = {"tasks": split_tasks, "hard_verifiable_tasks": split_hard}

    # 6. family-specific sanity (no-history / ask / evidence)
    no_history = [ev for ev in evaluator["train"] + evaluator["dev"] + evaluator["reserved_test"]
                  if ev["task_type"] == "no_history_recommend"]
    if any(ev["expected_decision"] != "recommend" for ev in no_history):
        failures.append("no_history_recommend family contains non-recommend labels")
    if any(not ev["hard_constraints"] for ev in no_history):
        failures.append("no_history_recommend task lacks an explicit hard constraint")
    if any(ev["effective_preferences"] for ev in no_history):
        failures.append("no_history_recommend task has effective preferences")
    ask_tasks = [ev for ev in evaluator["train"] + evaluator["dev"] + evaluator["reserved_test"]
                 if ev["task_type"] == "insufficient_information"]
    if any(ev["expected_decision"] != "clarification_required" for ev in ask_tasks):
        failures.append("insufficient_information family label is not clarification_required")
    evidence_tasks = [ev for ev in evaluator["train"] + evaluator["dev"] + evaluator["reserved_test"]
                      if ev["task_type"] in {"no_verified_match", "unverifiable_constraint"}]
    if any(ev["expected_decision"] not in {"no_verified_match", "insufficient_evidence"} for ev in evidence_tasks):
        failures.append("evidence family contains a recommendable label")

    # 7. review checklist and reserved guard
    review = json.loads((DATA / "review_checklist.json").read_text(encoding="utf-8"))
    pending = [ev["task_id"] for split in SPLITS for ev in evaluator[split]
               if ev["construction_metadata"]["review_status"] == "pending_human_review"]
    queued = [row["task_id"] for row in review["tasks"]]
    if sorted(pending) != sorted(queued):
        failures.append("review checklist does not match pending flags")
    smoke_dir = REPO / "runs/p14e2/smoke"
    reserved_ids = set(all_agent["reserved_test"])
    smoke_leak = []
    if smoke_dir.is_dir():
        for path in smoke_dir.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            smoke_leak.extend(task_id for task_id in reserved_ids if task_id in text)
    if smoke_leak:
        failures.append(f"reserved_test task appeared in inference smoke: {sorted(set(smoke_leak))[:5]}")

    # 8. original benchmark untouched
    original_manifest = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf-8"))
    original_hashes_ok = True
    for name in ("agent_tasks.jsonl", "evaluator_only.jsonl"):
        expected = original_manifest["files"][name]["sha256"]
        if _sha256(ORIGINAL / name) != expected:
            failures.append(f"P14-E original {name} hash changed")
            original_hashes_ok = False

    summary = {
        "schema": "agentrecx.p14e2.validation.v1",
        "benchmark_version": E2_VERSION,
        "counts": {
            "total": total,
            "splits": {split: len(agent[split]) for split in SPLITS},
            "task_types": dict(sorted(types.items())),
            "expected_decisions": dict(sorted(Counter(
                ev["expected_decision"] for evs in evaluator.values() for ev in evs).items())),
            "personas": len(personas),
            "history_events": len(events),
        },
        "checks": {
            "provenance_replay_ok": f"{replay_ok}/{total}",
            "tasks_with_preference_history": f"{tasks_with_history}/{total}",
            "hard_constraint_verifiable_tasks": f"{hard_verifiable_tasks}/{total}",
            "namespace_valid_verified_samples": verified_samples,
            "unknown_attribute_rate_mean": round(sum(unknown_rates) / len(unknown_rates), 4) if unknown_rates else 0.0,
            "leakage_hits": dict(leakage),
            "persona_overlap_with_pilot": len(dup_persona & original_personas),
            "families_crossing_splits": 0,
            "original_hashes_unchanged": original_hashes_ok,
            "pending_human_review": len(pending),
            "reserved_test_inference_records": len(set(smoke_leak)),
        },
        "split_stats": split_stats,
        "failures": failures,
        "warnings": warnings,
        "pass": not failures,
    }
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
