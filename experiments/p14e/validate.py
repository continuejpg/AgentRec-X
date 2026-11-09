"""P14-E quality validation: schema, provenance consistency, leakage, contrast, reachability.

Run: ``python -m experiments.p14e.validate``
Writes ``runs/p14e/validation/summary.json`` and ``data/p14e/benchmark/review_checklist.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p14e.labeling import active_prefs, assess_candidate
from experiments.p14e.build_pilot import BENCHMARK_VERSION, MIN_RECOMMENDABLE, SPLIT

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data/p14e/benchmark"
RUNS_DIR = REPO / "runs/p14e/validation"
FORBIDDEN_AGENT_KEYS = {
    "effective_preferences",
    "withdrawn_preferences",
    "expected_decision",
    "decision_rationale",
    "labels",
    "verified_candidates",
    "verified_candidate_count",
    "hard_constraints",
    "persona",
}
FORBIDDEN_AGENT_SUBSTRINGS = (
    "expected_decision",
    "verified_candidate",
    "acceptable_product",
    "persona_preferences",
    "evaluator_only",
    "no_verified_match",
    "clarification_required",
    "insufficient_evidence",
    "intent_note",
    "task_type",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sets(prefs: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {(str(p.get("kind")), str(p.get("value")), str(p.get("polarity"))) for p in prefs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the P14-E benchmark pilot")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    warnings: list[str] = []

    agent_tasks = _load_jsonl(data_dir / "agent_tasks.jsonl")
    evaluator = _load_jsonl(data_dir / "evaluator_only.jsonl")
    personas = _load_jsonl(data_dir / "personas.jsonl")
    events = _load_jsonl(data_dir / "history_events.jsonl")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    catalogue = ElectronicsCatalogue.load()

    # 1. counts / ids / split
    task_ids = [t["task_id"] for t in agent_tasks]
    if len(task_ids) != len(set(task_ids)):
        failures.append("duplicate task_id")
    if len(agent_tasks) != len(evaluator):
        failures.append("agent/evaluator task count mismatch")
    if any(t["split"] != SPLIT for t in agent_tasks):
        failures.append("non pilot_dev split present")
    if any(t.get("schema") != BENCHMARK_VERSION for t in agent_tasks):
        failures.append("schema version mismatch")
    if manifest["counts"]["tasks"] != len(agent_tasks):
        failures.append("manifest task count mismatch")

    # 2-3. history coverage + provenance replay consistency
    events_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        match = re.match(r"^(p14e_pilot_\d{4})_", str(event["event_id"]))
        if not match:
            failures.append(f"unparseable event_id: {event['event_id']}")
            continue
        events_by_task[match.group(1)].append(event)
    history_coverage_ok = 0
    tasks_with_preference_history = 0
    replay_ok = 0
    for agent, ev in zip(agent_tasks, evaluator):
        task_events = sorted(events_by_task.get(agent["task_id"], []), key=lambda e: e["sequence"])
        if not task_events:
            failures.append(f"{agent['task_id']}: no history events")
            continue
        if any(e.get("preference_candidates") for e in task_events):
            tasks_with_preference_history += 1
        history_coverage_ok += sum(1 for e in task_events if e.get("preference_candidates"))
        effective, withdrawn = active_prefs(task_events)
        if _state_sets(effective) == _state_sets(ev["effective_preferences"]):
            replay_ok += 1
        else:
            failures.append(f"{agent['task_id']}: effective preference replay mismatch")
        withdrawn_set = _state_sets(withdrawn)
        if not withdrawn_set <= _state_sets(ev["withdrawn_preferences"]):
            failures.append(f"{agent['task_id']}: withdrawn preference provenance mismatch")
        session_texts = [
            turn["content"] for session in agent["history_sessions"] for turn in session["turns"]
        ]
        if session_texts != [e["raw_text"] for e in task_events]:
            failures.append(f"{agent['task_id']}: agent history sessions do not match frozen events")

    # 4. product id / namespace validity + 5-6. hard constraint verifiability / UNKNOWN ratio
    hard_verifiable_tasks = 0
    attr_unknown_tasks: list[float] = []
    namespace_ok = 0
    for agent, ev in zip(agent_tasks, evaluator):
        if agent["catalog_namespace"] != "amazon_electronics_c4":
            failures.append(f"{agent['task_id']}: wrong namespace")
        pool = catalogue.by_type(ev["product_type_pool"])
        if not pool:
            failures.append(f"{agent['task_id']}: empty product-type pool")
        hard = ev["hard_constraints"]
        verifiable = True
        for constraint in hard:
            if constraint["kind"] == "unverifiable":
                verifiable = False
                continue
            count = sum(
                1 for product in pool
                if assess_candidate(product, hard_constraints=[constraint], preferences=[]).hard_states.get(
                    constraint["constraint_id"]
                ) == "verified"
            )
            if count == 0:
                verifiable = False
        if verifiable or not hard:
            hard_verifiable_tasks += 1
        requested = {
            str(p["value"]) for p in ev["effective_preferences"] if p["kind"] == "attribute"
        } | {str(c["value"]) for c in hard if c["kind"] == "attribute"}
        if requested:
            missing = sum(
                1 for product in pool
                if any(name not in (product.attributes or {}) for name in requested)
            )
            attr_unknown_tasks.append(missing / max(len(pool), 1))
        for sample_id in ev["labels"]["verified_candidates_sample"]:
            if catalogue.get(sample_id) is None:
                failures.append(f"{agent['task_id']}: verified sample id not in catalogue: {sample_id}")
            elif f"{agent['catalog_namespace']}:{sample_id}" not in catalogue.records:
                failures.append(f"{agent['task_id']}: namespace record missing for {sample_id}")
            else:
                namespace_ok += 1

    # 7. leakage isolation
    leakage_hits: dict[str, int] = Counter()
    for agent in agent_tasks:
        text = json.dumps(agent, ensure_ascii=False, sort_keys=True)
        keys = set(agent)
        for key in keys & FORBIDDEN_AGENT_KEYS:
            leakage_hits[f"key:{key}"] += 1
        for marker in FORBIDDEN_AGENT_SUBSTRINGS:
            if marker in text:
                leakage_hits[f"substr:{marker}"] += 1
        for session in agent["history_sessions"]:
            for turn in session["turns"]:
                if "evaluator" in turn["content"].lower() or "expected decision" in turn["content"].lower():
                    leakage_hits["substr:history_evaluator"] += 1
    if leakage_hits:
        failures.append(f"leakage hits in agent-visible files: {dict(leakage_hits)}")

    # 8. same-query different-persona contrast
    agent_by_id = {agent["task_id"]: agent for agent in agent_tasks}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in evaluator:
        if ev["task_type"] != "same_query_persona_pair":
            continue
        groups[agent_by_id[ev["task_id"]]["current_query"]].append(agent_by_id[ev["task_id"]])
    pairs_checked = 0
    pairs_distinct_prefs = 0
    pairs_distinct_verified = 0
    pair_details: list[dict[str, Any]] = []
    evaluator_by_id = {ev["task_id"]: ev for ev in evaluator}
    verified_cache: dict[str, set[str]] = {}

    def verified_ids(ev: dict[str, Any]) -> set[str]:
        if ev["task_id"] in verified_cache:
            return verified_cache[ev["task_id"]]
        pool = catalogue.by_type(ev["product_type_pool"])
        hard = ev["hard_constraints"]
        prefs = ev["effective_preferences"]
        ids = {
            product.item_id for product in pool
            if assess_candidate(product, hard_constraints=hard, preferences=prefs).acceptable
        }
        verified_cache[ev["task_id"]] = ids
        return ids

    for query, group in groups.items():
        if len(group) < 2:
            failures.append(f"same-query pair group has fewer than 2 tasks: {query[:60]}")
            continue
        if len(group) != 2:
            failures.append(f"same-query pair group has {len(group)} tasks: {query[:60]}")
        pairs_checked += 1
        left, right = group[0], group[1]
        left_prefs = _state_sets(evaluator_by_id[left["task_id"]]["effective_preferences"])
        right_prefs = _state_sets(evaluator_by_id[right["task_id"]]["effective_preferences"])
        prefs_differ = left_prefs != right_prefs
        left_ids = verified_ids(evaluator_by_id[left["task_id"]])
        right_ids = verified_ids(evaluator_by_id[right["task_id"]])
        symmetric_diff = len(left_ids ^ right_ids)
        verified_differ = symmetric_diff > 0
        pairs_distinct_prefs += int(prefs_differ)
        pairs_distinct_verified += int(verified_differ)
        pair_details.append(
            {
                "query": query,
                "left_task": left["task_id"],
                "right_task": right["task_id"],
                "preferences_differ": prefs_differ,
                "verified_symmetric_difference": symmetric_diff,
                "left_verified": len(left_ids),
                "right_verified": len(right_ids),
            }
        )
        if not prefs_differ:
            failures.append(f"same-query pair has identical effective preferences: {left['task_id']}")
        if not verified_differ:
            failures.append(f"same-query pair has identical verified candidate sets: {left['task_id']}")

    # 9. near duplicates (query + history raw text)
    signatures: dict[str, list[str]] = defaultdict(list)
    for agent in agent_tasks:
        payload = agent["current_query"] + "||" + "|".join(
            turn["content"] for session in agent["history_sessions"] for turn in session["turns"]
        )
        signatures[hashlib.sha256(payload.encode("utf-8")).hexdigest()].append(agent["task_id"])
    near_duplicate_groups = [ids for ids in signatures.values() if len(ids) > 1]
    if near_duplicate_groups:
        failures.append(f"near-duplicate task groups: {near_duplicate_groups}")

    # 10. review queue
    review_queue = [
        {
            "task_id": ev["task_id"],
            "reasons": ev["construction_metadata"].get("review_reasons") or [],
            "expected_decision": ev["expected_decision"],
        }
        for ev in evaluator
        if ev["construction_metadata"]["review_status"] == "pending_human_review"
    ]
    (data_dir / "review_checklist.json").write_text(
        json.dumps({"schema": "agentrecx.p14e.review_checklist.v1", "tasks": review_queue}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 11. candidate reachability with the accepted BM25 source
    search = catalogue.search_source
    reachability = {"tasks": 0, "hit@10": 0, "hit@50": 0, "no_verified_tasks": 0}
    per_task_reach: list[dict[str, Any]] = []
    for agent, ev in zip(agent_tasks, evaluator):
        if ev["expected_decision"] != "recommend":
            reachability["no_verified_tasks"] += 1
            continue
        ids = verified_ids(ev)
        if not ids:
            continue
        reachability["tasks"] += 1
        result10 = search.search(agent["current_query"], limit=10)
        result50 = search.search(agent["current_query"], limit=50)
        hit10 = [hit.parent_asin.split(":", 1)[-1] for hit in result10.hits]
        hit50 = [hit.parent_asin.split(":", 1)[-1] for hit in result50.hits]
        reachability["hit@10"] += int(any(item in ids for item in hit10))
        reachability["hit@50"] += int(any(item in ids for item in hit50))
        per_task_reach.append({
            "task_id": agent["task_id"],
            "verified_candidates": len(ids),
            "top10": hit10,
            "hit@10": any(item in ids for item in hit10),
            "hit@50": any(item in ids for item in hit50),
        })
    if reachability["tasks"]:
        reachability["hit@10_rate"] = round(reachability["hit@10"] / reachability["tasks"], 4)
        reachability["hit@50_rate"] = round(reachability["hit@50"] / reachability["tasks"], 4)

    # 12. smoke results (written by a later script; absent is not a failure)
    smoke_path = args.runs_dir.parent / "smoke" / "summary.json"
    smoke: dict[str, Any] = {}
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))

    summary = {
        "schema": "agentrecx.p14e.validation.v1",
        "benchmark_version": BENCHMARK_VERSION,
        "catalogue": catalogue.stats,
        "checks": {
            "task_count": len(agent_tasks),
            "task_types": dict(Counter(t["task_type"] for t in evaluator)),
            "expected_decisions": dict(Counter(ev["expected_decision"] for ev in evaluator)),
            "history_events_with_preference_candidates": f"{history_coverage_ok}/{len(events)}",
            "tasks_with_preference_history": f"{tasks_with_preference_history}/{len(agent_tasks)}",
            "provenance_replay_consistency": f"{replay_ok}/{len(agent_tasks)}",
            "hard_constraint_verifiable_tasks": f"{hard_verifiable_tasks}/{len(agent_tasks)}",
            "namespace_valid_samples": namespace_ok,
            "unknown_attribute_rate_mean": round(sum(attr_unknown_tasks) / len(attr_unknown_tasks), 4) if attr_unknown_tasks else 0.0,
            "leakage_hits": dict(leakage_hits),
        },
        "same_query_contrast": {
            "pairs_checked": pairs_checked,
            "pairs_with_distinct_preferences": pairs_distinct_prefs,
            "pairs_with_distinct_verified_sets": pairs_distinct_verified,
            "details": pair_details,
        },
        "near_duplicate_groups": near_duplicate_groups,
        "review_required": len(review_queue),
        "reachability": reachability,
        "smoke": smoke,
        "failures": failures,
        "warnings": warnings,
        "manifest_sha256": _sha256(data_dir / "manifest.json"),
    }
    summary["pass"] = not failures
    (args.runs_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("schema", "pass", "checks", "same_query_contrast", "review_required", "reachability", "failures", "warnings")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
