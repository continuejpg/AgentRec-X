"""P14-E minimal real-tool smoke test over the pilot benchmark.

For a small representative set of tasks this script exercises the real components the
benchmark will be wired into, without any Qwen planner and without running a full Agent:

* Memory: ``recommendation.shopping_companion.memory.ShoppingCompanionMemory`` (M9+M5),
  ingesting the frozen benchmark history as ``synthetic_fixture`` (its honest source kind);
* Search: the accepted ``CatalogSearchSource`` BM25 over the real P13-R catalogue;
* Product Info: the thin P14-E Electronics adapter (facts + evidence, UNKNOWN never filled);
* CandidateLedger: ``record_from_source(CATALOG_SEARCH, ...)``;
* Eligibility: ``GroundedReasoner`` + ``CandidateEligibilityEvaluator`` with one verifiable
  CATEGORY constraint and one FEATURE constraint expected to resolve UNKNOWN.

It writes one trajectory per task plus ``summary.json`` under ``runs/p14e/smoke/``.
No reference product is injected into any tool call.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p14e.labeling import assess_candidate
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.constraint_eligibility import CandidateEligibilityEvaluator, ConstraintRequirement
from recommendation.control.grounded_reasoning import ConstraintKind, GroundedReasoner
from recommendation.shopping_companion.memory import (
    SOURCE_KIND_SYNTHETIC_FIXTURE,
    ShoppingCompanionMemory,
)

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data/p14e/benchmark"
RUNS_DIR = REPO / "runs/p14e/smoke"
SCHEMA = "agentrecx.p14e.smoke.trajectory.v1"
MAX_TASKS = 8


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _memory_sessions(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "session_id": session["session_id"],
            "date": session["date"],
            "turns": [dict(turn) for turn in session["turns"]],
        }
        for session in task["history_sessions"]
    ]


def _pref_values(preferences: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(item.get("value")), str(item.get("polarity"))) for item in preferences}


def _canonical_value(raw: str) -> str | None:
    """Canonicalise an M9 free-text preference value onto the benchmark attribute vocabulary."""
    text = str(raw or "").strip().lower().replace("_", "-")
    rules = [
        ("over-ear", "over_ear"), ("over ear", "over_ear"),
        ("in-ear", "in_ear"), ("in ear", "in_ear"), ("earbud", "in_ear"),
        ("on-ear", "on_ear"), ("on ear", "on_ear"),
        ("noise cancell", "noise_cancelling"),
        ("wireless", "wireless"), ("wired", "wired"),
        ("microphone", "microphone"), ("mic ", "microphone"),
        ("waterproof", "waterproof"), ("water resistant", "waterproof"), ("sweatproof", "waterproof"),
        ("fold", "foldable"), ("mechanical", "mechanical"),
        ("rgb", "rgb"), ("backlit", "rgb"),
        ("recharge", "rechargeable"), ("multi", "multidevice"),
        ("usb", "usb"), ("silent", "silent"), ("quiet", "silent"),
        ("ergonomic", "ergonomic"), ("gaming", "gaming"),
        ("comfort", "comfort"),
    ]
    for needle, canonical in rules:
        if needle in text:
            return canonical
    return None


def _canonical_polarity(raw: str) -> str:
    return "negative" if str(raw or "").lower() in {"avoid", "negative", "dislike"} else "positive"


def _canonical_recovered(preferences: list[dict[str, Any]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in preferences:
        canonical = _canonical_value(str(item.get("value")))
        if canonical:
            out.add((canonical, _canonical_polarity(str(item.get("polarity")))))
    return out


def _pick_tasks(agent_tasks: list[dict[str, Any]], evaluator: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {task["task_id"]: task for task in agent_tasks}
    selected: list[dict[str, Any]] = []
    pair_ids = [ev["task_id"] for ev in evaluator if ev["task_type"] == "same_query_persona_pair"][:2]
    selected.extend(by_id[task_id] for task_id in pair_ids)
    for task_type in ("regular_history", "preference_withdrawal", "preference_update",
                      "preference_conflict", "insufficient_information", "no_verified_match",
                      "unverifiable_constraint"):
        for ev in evaluator:
            if ev["task_type"] == task_type and ev["task_id"] not in {task["task_id"] for task in selected}:
                selected.append(by_id[ev["task_id"]])
                break
    return selected[:MAX_TASKS]


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "trajectories").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    catalogue = ElectronicsCatalogue.load()
    load_seconds = round(time.perf_counter() - started, 2)
    search_source = catalogue.search_source
    index_seconds = round(time.perf_counter() - started - load_seconds, 2)
    agent_tasks = _load_jsonl(DATA_DIR / "agent_tasks.jsonl")
    evaluator = _load_jsonl(DATA_DIR / "evaluator_only.jsonl")
    ev_by_id = {ev["task_id"]: ev for ev in evaluator}
    selected = _pick_tasks(agent_tasks, evaluator)

    summary: dict[str, Any] = {
        "schema": "agentrecx.p14e.smoke.summary.v1",
        "catalogue": catalogue.stats,
        "catalogue_load_seconds": load_seconds,
        "bm25_index_seconds": index_seconds,
        "peak_rss_mb": _rss_mb(),
        "tasks": [],
        "notes": [
            "no reference product is placed into any candidate pool",
            "memory ingests constructed benchmark history with source_kind=synthetic_fixture",
            "UNKNOWN product facts stay UNKNOWN; no LLM is used",
        ],
    }

    for task in selected:
        task_id = task["task_id"]
        ev = ev_by_id[task_id]
        pool = catalogue.by_type(ev["product_type_pool"])
        verified = {
            product.item_id for product in pool
            if assess_candidate(
                product,
                hard_constraints=ev["hard_constraints"],
                preferences=ev["effective_preferences"],
            ).acceptable
        }
        trajectory: dict[str, Any] = {
            "schema": SCHEMA,
            "task_id": task_id,
            "task_type": ev["task_type"],
            "mode": "real_electronics_catalogue_no_llm",
            "catalogue": catalogue.as_dict(),
            "tool_calls": [],
            "terminal": {"status": "smoke_completed", "succeeded": None,
                         "note": "smoke test only; no Agent answer is produced"},
        }
        task_started = time.perf_counter()

        # 1. Memory tool: M9+M5 over the frozen benchmark history
        memory_started = time.perf_counter()
        with ShoppingCompanionMemory() as memory:
            ingest = memory.ingest_sessions(
                task_id,
                _memory_sessions(task),
                source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE,
                source_ref=f"p14e_pilot:{task_id}",
            )
            memory_result = memory.search(
                task["current_query"], scope="long_term", k=10, conversation_id=task_id
            )
            history_query = " ".join(
                str(turn.get("content") or "")
                for session in task["history_sessions"]
                for turn in session["turns"]
            )
            history_memory_result = memory.search(
                history_query, scope="long_term", k=10, conversation_id=task_id
            )
            memory_status = memory.history_status()
        raw_preferences = list(history_memory_result.get("preferences") or [])
        recovered = _pref_values(raw_preferences)
        recovered_canonical = _canonical_recovered(raw_preferences)
        expected = _pref_values(ev["effective_preferences"])
        withdrawn_canonical = {
            _canonical_value(str(item.get("value"))) for item in ev["withdrawn_preferences"]
        } - {None}
        recovered_canonical_values = {value for value, _polarity in recovered_canonical}
        if expected:
            recall = round(len(recovered_canonical & expected) / len(expected), 4)
            precision = round(len(recovered_canonical & expected) / max(len(recovered_canonical), 1), 4)
        else:
            recall = 1.0 if not recovered_canonical else 0.0
            precision = 1.0 if not recovered_canonical else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
        stale_active = sorted(recovered_canonical - expected)
        current_query_prefs = list(memory_result.get("preferences") or [])
        trajectory["tool_calls"].append({
            "tool": "memory.search",
            "inputs": {
                "current_query": task["current_query"],
                "history_probe_query": "concatenation of the agent-visible history turns",
                "scope": "long_term",
                "k": 10,
            },
            "output": {
                "ingest": ingest,
                "status": memory_result.get("status"),
                "available": memory_result.get("available"),
                "current_query_preferences": [
                    {"kind": item.get("kind"), "value": item.get("value"),
                     "polarity": item.get("polarity")}
                    for item in current_query_prefs
                ],
                "history_probe_preferences": [
                    {"kind": item.get("kind"), "value": item.get("value"),
                     "polarity": item.get("polarity")}
                    for item in raw_preferences
                ],
                "history_status": memory_status,
            },
            "memory_recovery": {
                "expected_active": sorted(expected),
                "current_query_recall": round(
                    len(_canonical_recovered(current_query_prefs) & expected) / max(len(expected), 1), 4
                ),
                "recovered_raw": sorted(recovered),
                "recovered_canonical": sorted(recovered_canonical),
                "active_recall": recall,
                "active_precision": precision,
                "active_f1": f1,
                "stale_active": stale_active,
                "withdrawn_values_absent_from_active": sorted(
                    value for value in withdrawn_canonical if value not in recovered_canonical_values
                ),
                "withdrawn_values_still_active": sorted(
                    value for value in withdrawn_canonical if value in recovered_canonical_values
                ),
                "note": (
                    "M9's read adapter only returns preferences whose value tokens overlap the query; "
                    "the current need rarely repeats historical attribute words, so the history-derived "
                    "probe (built from agent-visible turns, no evaluator labels) is reported separately."
                ),
            },
            "latency_ms": round((time.perf_counter() - memory_started) * 1000, 2),
        })

        # 2. Search tool: accepted BM25 over the real catalogue
        search_started = time.perf_counter()
        search_result = search_source.search(task["current_query"], limit=10)
        hits = [
            {"item_id": hit.parent_asin.split(":", 1)[-1], "rank": rank + 1,
             "score": round(hit.score, 4), "score_kind": search_result.score_kind,
             "title": hit.title, "matched_fields": list(hit.matched_fields)}
            for rank, hit in enumerate(search_result.hits)
        ]
        hit_ids = {hit["item_id"] for hit in hits}
        trajectory["tool_calls"].append({
            "tool": "product_search",
            "inputs": {"query": task["current_query"], "limit": 10, "namespace": catalogue.namespace},
            "output": {"hits": hits, "matched": search_result.matched,
                       "catalogue_size": search_result.catalogue_size},
            "reachability": {
                "verified_pool_count": len(verified),
                "hit@10": sorted(hit_ids & verified),
                "hit@10_ok": bool(hit_ids & verified),
            },
            "latency_ms": round((time.perf_counter() - search_started) * 1000, 2),
        })

        # 3. Product Info tool: facts + evidence; UNKNOWN is never filled
        info_started = time.perf_counter()
        requested = sorted({
            str(p["value"]) for p in ev["effective_preferences"]
        } | {str(c["value"]) for c in ev["hard_constraints"] if c.get("kind") != "unverifiable"}
        | {"brand", "price"})
        inspected = []
        for hit in hits[:3]:
            inspection = catalogue.inspect(hit["item_id"], requested=requested)
            inspected.append({
                "item_id": inspection["product_id"],
                "status": inspection.get("status"),
                "facts_present": [name for name, fact in inspection["facts"].items()
                                  if fact.get("status") == "present"],
                "facts_unknown": inspection["unknown_fields"],
                "facts_unsupported": inspection["unsupported_fields"],
                "evidence": {name: fact.get("evidence") for name, fact in inspection["facts"].items()
                             if fact.get("status") == "present"},
            })
        trajectory["tool_calls"].append({
            "tool": "product_view",
            "inputs": {"product_ids": [hit["item_id"] for hit in hits[:3]], "requested": requested},
            "output": {"inspected": inspected},
            "latency_ms": round((time.perf_counter() - info_started) * 1000, 2),
        })

        # 4. CandidateLedger: the accepted grounded record path
        ledger_started = time.perf_counter()
        ledger = CandidateLedger()
        ledger.record_from_source(
            source=CandidateSource.CATALOG_SEARCH,
            candidates=[(f"{catalogue.namespace}:{hit['item_id']}", hit["rank"], hit["score"]) for hit in hits],
            score_kind=search_result.score_kind,
            source_query=task["current_query"],
            step_index=1,
            grounded=frozenset(f"{catalogue.namespace}:{hit['item_id']}" for hit in hits),
        )
        ranked = ledger.ranked_parent_asins()
        trajectory["tool_calls"].append({
            "tool": "candidate_ledger.record",
            "inputs": {"source": "catalog_search", "candidate_count": len(hits)},
            "output": {"entries": len(ranked), "ranked_ids": list(ranked)},
            "latency_ms": round((time.perf_counter() - ledger_started) * 1000, 2),
        })

        # 5. Eligibility: one verifiable CATEGORY and one FEATURE that stays UNKNOWN
        eligibility_started = time.perf_counter()
        reasoner = GroundedReasoner(catalogue)
        eligibility = CandidateEligibilityEvaluator(reasoner)
        requirements = [
            ConstraintRequirement(kind=ConstraintKind.CATEGORY, expected="Electronics",
                                  source_text="catalogue main category"),
            ConstraintRequirement(kind=ConstraintKind.FEATURE, expected="wireless",
                                  source_text="example feature constraint (expected UNKNOWN in this catalogue)"),
        ]
        feasibility = eligibility.assess(list(ranked[:5]), requirements=requirements)
        trajectory["tool_calls"].append({
            "tool": "eligibility.assess",
            "inputs": {"candidate_count": min(len(ranked), 5),
                       "requirements": [{"kind": r.kind.value, "expected": r.expected} for r in requirements]},
            "output": {
                "verified_eligible_count": feasibility.verified_eligible_count,
                "ineligible_count": feasibility.ineligible_count,
                "unresolved_count": feasibility.unresolved_count,
                "exclusions": [list(item) for item in feasibility.exclusions()],
            },
            "latency_ms": round((time.perf_counter() - eligibility_started) * 1000, 2),
        })

        trajectory["timing"] = {
            "task_seconds": round(time.perf_counter() - task_started, 3),
            "peak_rss_mb": _rss_mb(),
        }
        (RUNS_DIR / "trajectories" / f"{task_id}.json").write_text(
            json.dumps(trajectory, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        memory_recall = trajectory["tool_calls"][0]["memory_recovery"]["active_recall"]
        memory_f1 = trajectory["tool_calls"][0]["memory_recovery"]["active_f1"]
        summary["tasks"].append({
            "task_id": task_id,
            "task_type": ev["task_type"],
            "expected_decision": ev["expected_decision"],
            "verified_pool_count": len(verified),
            "memory_recovered": len(raw_preferences),
            "memory_active_recall": memory_recall,
            "memory_active_f1": memory_f1,
            "memory_stale_active": trajectory["tool_calls"][0]["memory_recovery"]["stale_active"],
            "hit@10": trajectory["tool_calls"][1]["reachability"]["hit@10_ok"],
            "withdrawn_values_absent": trajectory["tool_calls"][0]["memory_recovery"]["withdrawn_values_absent_from_active"],
            "withdrawn_values_still_active": trajectory["tool_calls"][0]["memory_recovery"]["withdrawn_values_still_active"],
            "eligibility": trajectory["tool_calls"][4]["output"],
            "peak_rss_mb": trajectory["timing"]["peak_rss_mb"],
        })

    summary["resources"] = {
        "wall_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_mb": _rss_mb(),
        "catalogue_load_seconds": load_seconds,
        "bm25_index_seconds": index_seconds,
    }
    recommend_tasks = [t for t in summary["tasks"] if t["expected_decision"] == "recommend"]
    if summary["tasks"]:
        summary["means"] = {
            "memory_active_recall": round(
                sum(t["memory_active_recall"] for t in summary["tasks"]) / len(summary["tasks"]), 4
            ),
            "memory_active_f1": round(
                sum(t["memory_active_f1"] for t in summary["tasks"]) / len(summary["tasks"]), 4
            ),
            "hit@10_rate_on_recommend_tasks": (
                round(sum(1 for t in recommend_tasks if t["hit@10"]) / len(recommend_tasks), 4)
                if recommend_tasks else None
            ),
            "tasks_with_withdrawn_absent": sum(
                1 for t in summary["tasks"] if t["withdrawn_values_absent"]
            ),
            "tasks_with_withdrawn_still_active": sum(
                1 for t in summary["tasks"] if t["withdrawn_values_still_active"]
            ),
        }
    (RUNS_DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": [t["task_id"] for t in summary["tasks"]], "means": summary.get("means"),
                      "resources": summary["resources"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
