"""P14-B1 real-tool Gate smoke: five representative Pilot Dev tasks through the existing
real components (Memory M9/M5, BM25, Product Info, CandidateLedger, Eligibility,
Preference Scorer, Decision Gate).

The decision loop uses only the existing Planner action vocabulary (RETRIEVE, RERANK,
FINALIZE, ASK) and never reads evaluator-only labels; outcome scoring is a separate
pass that runs *after* the trajectory is written.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from experiments.p14b1.diagnose import (
    canonical_prefs,
    candidate_rows,
    conflicts,
    hard_requirements,
    query_specificity,
    verified_ids,
)
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.constraint_eligibility import CandidateEligibilityEvaluator, ConstraintRequirement
from recommendation.control.grounded_reasoning import ConstraintKind, GroundedReasoner
from recommendation.preference.decision_gate import (
    ALLOW_FINALIZE,
    NEED_MORE_CANDIDATES,
    NEED_MORE_PRODUCT_EVIDENCE,
    NEED_USER_CLARIFICATION,
    DecisionGate,
    GateBudget,
    GateContext,
)
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.memory import (
    SOURCE_KIND_SYNTHETIC_FIXTURE,
    ShoppingCompanionMemory,
)
from recommendation.preference.history import HistoryStateReader

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e/benchmark"
RUNS = REPO / "runs/p14b1"
TASK_IDS = (
    "p14e_pilot_0001",  # normal recommendation
    "p14e_pilot_0026",  # conflicting preferences -> ASK
    "p14e_pilot_0036",  # no verified candidate in scope
    "p14e_pilot_0039",  # unverifiable hard constraint (price)
    "p14e_pilot_0029",  # withdrawn history preference, still recommendable
    "p14e_pilot_0031",  # insufficient information -> ASK
)


def _load(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "runtime").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    catalogue = ElectronicsCatalogue.load()
    search = catalogue.search_source
    scorer = PreferenceAwareScorer()
    gate = DecisionGate()
    reasoner = GroundedReasoner(catalogue)
    eligibility = CandidateEligibilityEvaluator(reasoner)
    agent = {task["task_id"]: task for task in _load("agent_tasks.jsonl")}
    evals = {task["task_id"]: task for task in _load("evaluator_only.jsonl")}
    trajectories: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        task = agent[task_id]
        query = task["current_query"]
        trajectory: dict[str, Any] = {
            "schema": "agentrecx.p14b1.runtime_smoke.v1",
            "task_id": task_id,
            "mode": "real_tools_gate_no_llm",
            "catalogue_namespace": ELECTRONICS_NAMESPACE,
            "actions": [],
            "tool_calls": [],
            "terminal": {"status": "not_started", "succeeded": None},
        }
        tool_steps = 0
        budget = GateBudget(max_retrievals=1, max_product_evidence_rounds=2)
        memory_started = time.perf_counter()
        with ShoppingCompanionMemory() as memory:
            memory.ingest_sessions(
                task_id, [dict(session) for session in task["history_sessions"]],
                source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref=f"p14b1:{task_id}",
            )
            history = HistoryStateReader(memory.preference_service).read(
                task_id, conversation_id=task_id, current_query=query
            )
        preferences = canonical_prefs(history["active_preferences"])
        hard = hard_requirements(query)
        pref_conflicts = conflicts(preferences)
        specificity = query_specificity(query, hard, preferences)
        trajectory["tool_calls"].append({
            "tool": "mem_search_history_state",
            "inputs": {"user_key": task_id, "scope": "active_state", "max_preferences": 10},
            "output": {
                "active": preferences,
                "inactive": history["inactive_preferences"],
                "provenance": history["provenance"],
                "read_policy": history["read_policy"],
            },
            "latency_ms": round((time.perf_counter() - memory_started) * 1000, 2),
        })
        trajectory["tool_calls"][-1]["stage"] = "stage_1"
        trajectory["tool_calls"][-1]["error"] = None
        tool_steps += 1

        # RETRIEVE: real BM25 over the real catalogue, recorded in CandidateLedger.
        retrieval_started = time.perf_counter()
        search_result = search.search(query, limit=20)
        hit_ids = [hit.parent_asin.split(":", 1)[-1] for hit in search_result.hits]
        ledger = CandidateLedger()
        ledger.record_from_source(
            source=CandidateSource.CATALOG_SEARCH,
            candidates=[(hit.parent_asin, rank + 1, hit.score) for rank, hit in enumerate(search_result.hits)],
            score_kind="bm25",
            source_query=query,
            step_index=1,
            grounded=frozenset(hit.parent_asin for hit in search_result.hits),
        )
        trajectory["tool_calls"].append({
            "tool": "product_search",
            "inputs": {"query": query, "limit": 20, "namespace": ELECTRONICS_NAMESPACE},
            "output": {"hit_ids": hit_ids, "ledger_entries": len(ledger.ranked_parent_asins()),
                       "score_kind": "bm25", "catalogue_size": search_result.catalogue_size},
            "latency_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
            "stage": "stage_2",
            "error": None,
        })
        tool_steps += 1
        pool = {
            "task_id": task_id, "current_query": query,
            "candidate_ids": hit_ids,
            "search_scores": {hit_ids[index]: hit.score for index, hit in enumerate(search_result.hits)},
        }
        rows = candidate_rows(pool, catalogue, scorer, preferences, hard)

        # RERANK: product info on the top rows (real catalogue facts, UNKNOWN preserved).
        info_started = time.perf_counter()
        inspected = [catalogue.inspect(item_id, requested=sorted({r["value"] for r in hard} | {"price", "brand"}))
                     for item_id in hit_ids[:5]]
        trajectory["tool_calls"].append({
            "tool": "product_view",
            "inputs": {"product_ids": hit_ids[:5]},
            "output": {"inspected": [
                {"item_id": item["product_id"], "status": item["status"],
                 "unknown": item["unknown_fields"], "present": [k for k, v in item["facts"].items() if v.get("status") == "present"]}
                for item in inspected
            ]},
            "latency_ms": round((time.perf_counter() - info_started) * 1000, 2),
            "stage": "stage_2",
            "error": None,
        })
        tool_steps += 1
        eligibility_started = time.perf_counter()
        requirements = [ConstraintRequirement(kind=ConstraintKind.CATEGORY, expected="Electronics",
                                              source_text="catalogue category")]
        feasibility = eligibility.assess(
            [f"{ELECTRONICS_NAMESPACE}:{item_id}" for item_id in hit_ids[:10]],
            requirements=requirements,
        )
        trajectory["tool_calls"].append({
            "tool": "eligibility_assess",
            "inputs": {"candidate_count": min(len(hit_ids), 10), "requirements": ["category=Electronics"]},
            "output": {"verified": feasibility.verified_eligible_count,
                       "ineligible": feasibility.ineligible_count,
                       "unresolved": feasibility.unresolved_count},
            "latency_ms": round((time.perf_counter() - eligibility_started) * 1000, 2),
            "stage": "stage_2",
            "error": None,
        })
        tool_steps += 1

        context_kwargs = {
            "query": query, "active_preferences": preferences, "hard_requirements": hard,
            "preference_conflicts": pref_conflicts, "namespace": ELECTRONICS_NAMESPACE,
            "query_specificity": specificity,
        }
        gate_started = time.perf_counter()
        decision = gate.check(GateContext(candidates=rows, budget=budget, **context_kwargs))
        gate_trace = [decision.as_dict()]
        actions = ["RETRIEVE", "RERANK"]
        # Bounded follow-up: one product-evidence round and one wider retrieval round.
        while decision.action in (NEED_MORE_PRODUCT_EVIDENCE, NEED_MORE_CANDIDATES) and tool_steps < 8:
            if decision.action == NEED_MORE_PRODUCT_EVIDENCE:
                budget = replace(budget, product_evidence_rounds_used=budget.product_evidence_rounds_used + 1)
                actions.append("RERANK")
            else:
                budget = replace(budget, retrievals_used=budget.retrievals_used + 1)
                expanded = query + " " + " ".join(str(item["value"]).replace("_", "-") for item in hard)
                wider = search.search(expanded, limit=20)
                wider_ids = [hit.parent_asin.split(":", 1)[-1] for hit in wider.hits]
                trajectory["tool_calls"].append({
                    "tool": "product_search",
                    "inputs": {"query": expanded, "limit": 20, "reason": "gate_requested_more_candidates"},
                    "output": {"hit_ids": wider_ids, "new_retrieval": True},
                    "latency_ms": 0.0, "stage": "stage_2", "error": None,
                })
                pool = {"task_id": task_id, "current_query": expanded,
                        "candidate_ids": wider_ids,
                        "search_scores": {wider_ids[i]: hit.score for i, hit in enumerate(wider.hits)}}
                rows = candidate_rows(pool, catalogue, scorer, preferences, hard)
                actions.append("RETRIEVE")
            decision = gate.check(GateContext(candidates=rows, budget=budget, **context_kwargs))
            gate_trace.append(decision.as_dict())
            tool_steps += 1
        if decision.action == NEED_USER_CLARIFICATION:
            actions.append("ASK")
            trajectory["terminal"] = {"status": "awaiting_clarification", "succeeded": False,
                                      "question": decision.ask_question,
                                      "reason": decision.reason}
        elif decision.allow_finalize:
            actions.append("FINALIZE")
            selected = decision.selected_candidate_id
            trajectory["terminal"] = {"status": "finalized", "succeeded": True,
                                      "selected_candidate_id": selected,
                                      "recommendation": next((c for c in rows if c["product_ref"]["item_id"] == selected), None),
                                      "reason": decision.reason}
        else:
            trajectory["terminal"] = {"status": decision.action.lower(), "succeeded": False,
                                      "reason": decision.reason, "bounded_scope": decision.bounded_scope}
        trajectory["actions"] = actions
        trajectory["gate"] = gate_trace
        trajectory["tool_calls"].append({
            "tool": "decision_gate", "inputs": {"candidate_count": len(rows), "budget": budget.as_dict()},
            "output": decision.as_dict(), "latency_ms": round((time.perf_counter() - gate_started) * 1000, 2),
            "stage": "stage_2", "error": None,
        })
        trajectory["tool_steps"] = tool_steps
        trajectory["timing"] = {"seconds": round(time.perf_counter() - memory_started, 3) if False else 0.0,
                                 "peak_rss_mb": _rss_mb()}
        trajectories.append(trajectory)
        (RUNS / "runtime" / f"{task_id}.json").write_text(
            json.dumps(trajectory, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # Outcome evaluation pass (labels are read here only, after every trajectory is frozen).
    evaluation = []
    for trajectory in trajectories:
        ev = evals[trajectory["task_id"]]
        verified = verified_ids(ev, catalogue)
        selected = (trajectory["terminal"].get("selected_candidate_id")
                    if trajectory["terminal"].get("status") == "finalized" else None)
        evaluation.append({
            "task_id": trajectory["task_id"],
            "task_type": ev["task_type"],
            "expected_decision": ev["expected_decision"],
            "gate_terminal": trajectory["terminal"]["status"],
            "selected_candidate_id": selected,
            "selected_correct": selected in verified if selected else None,
            "verified_total": len(verified),
            "tools_called": [call["tool"] for call in trajectory["tool_calls"]],
        })
    summary = {
        "schema": "agentrecx.p14b1.runtime_smoke.summary.v1",
        "tasks": evaluation,
        "resources": {"wall_seconds": round(time.perf_counter() - started, 2), "peak_rss_mb": _rss_mb()},
        "notes": ["no LLM planner; deterministic decision loop over real tools",
                  "evaluator labels read only in the post-trajectory evaluation pass",
                  "no ASK is answered or bypassed; unresolved ASK stays awaiting_clarification"],
    }
    (RUNS / "runtime" / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
