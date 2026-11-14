"""Counterfactual Memory Writer -> Reader -> Ranker -> Gate E2E, fixed candidate pool.

Planner is the rule planner for this focused invariant run; the P20 main run uses the real
Qwen Planner.  Memory sessions are ingested by the real ShoppingCompanionMemory writer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue, ELECTRONICS_NAMESPACE
from recommendation.control.catalog_search import CatalogSearchHit, CatalogSearchResult
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
CASES = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
POOLS = REPO / "data/p20_shadow/ranking_dataset.jsonl"
CONTROLLED = REPO / "runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json"
ADAPTER = REPO / "runs/agent_native_ranking_phase42/memory_adapter_features.jsonl"
OUT = REPO / "runs/agent_ranker_phase43"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FixedPoolSource:
    mode = "fixed_bge_pool"

    def __init__(self, ids):
        self.ids = list(ids)

    def search(self, query, limit=20):
        hits = tuple(
            CatalogSearchHit(parent_asin=f"{ELECTRONICS_NAMESPACE}:{item_id}", score=float(20 - index),
                             matched_fields=("title",), title=None, bge_rank=index + 1)
            for index, item_id in enumerate(self.ids[:limit])
        )
        return CatalogSearchResult(hits=hits, query=query, matched=bool(hits), catalogue_size=len(self.ids), score_kind="fixed_pool")


class CachedRanker:
    def __init__(self, base, scores, arm):
        self.base = base
        self.scores = scores
        self.arm = arm
        self.last_latency_ms = None

    def score(self, *, preferences, current_query, candidates, hard_requirements=(), namespace):
        started = time.perf_counter()
        rows = self.base.score(preferences=preferences, current_query=current_query,
                               candidates=candidates, hard_requirements=hard_requirements, namespace=namespace)
        for row in rows:
            item_id = row["product_ref"]["item_id"]
            if item_id in self.scores:
                row["final_score"] = float(self.scores[item_id])
        rows.sort(key=lambda row: (row["hard_tier"], -float(row["final_score"]), row["query_rank"],
                                   row["product_ref"]["namespace"], row["product_ref"]["item_id"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    cases = read_jsonl(CASES)
    pools = {row["task_id"]: row for row in read_jsonl(POOLS)}
    controlled = {row["case_id"]: row for row in json.loads(CONTROLLED.read_text(encoding="utf-8"))["counterfactual"]["details"]}
    adapter = {}
    for row in read_jsonl(ADAPTER):
        adapter[(row["source"], row["group_id"], row["candidate_id"], row["memory_snapshot_id"])] = row["features"]
    records = []
    for case in cases:
        pool = pools.get(case["pool_id"])
        if not pool:
            continue
        for state_name in ("baseline_memory_state", "mutated_memory_state"):
            sessions = case.get(state_name.replace("_state", "_sessions")) or case.get(state_name.replace("_state", "_sessions")) or []
            task = {
                "task_id": f"{case['case_id']}:{state_name}",
                "user_id": f"phase43_{case['case_id']}",
                "current_query": case["query"],
                "current_time": "2025-01-15T12:00:00",
                "history_sessions": sessions,
            }
            if sessions:
                preferences, _ = _memory_read(catalogue, task)
            else:
                preferences = []
                task["memory_source"] = "empty_active_state_no_sessions"
            hard = case.get("hard_requirements") or []
            if not isinstance(hard, list):
                hard = []
            ids = list(case.get("eligible_candidate_ids") or [])
            executor = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=case["query"],
                                                   preferences=preferences, hard=hard)
            executor.search_source = FixedPoolSource(ids)
            # Existing is the baseline. Structured/hybrid scores are cached from the GPU run.
            ctrl = controlled.get(case["case_id"])
            # The controlled JSON keeps only one state; use it for the mutated state.
            # Baseline structured/hybrid scores are unavailable in that file, so the baseline
            # uses the Existing ranker by design.
            arm_scores = {}
            if state_name == "mutated_memory_state" and ctrl:
                block = ctrl.get(state_name) or {}
                query_scores = block.get("query_scores") or {}
                structured_scores = block.get("structured_scores") or {}
                question = case["query"]
                hybrid = {}
                for item_id in ids:
                    features = adapter.get(("counterfactual", case["case_id"], item_id, f"{case['case_id']}:{state_name}")) or {}
                    semantic = float(query_scores.get(item_id, 0.0))
                    hybrid[item_id] = semantic + 0.5 * float(features.get("positive_match_rate", 0.0)) - 0.75 * float(features.get("negative_conflict_rate", 0.0))
                arm_scores = {"structured": structured_scores, "hybrid": hybrid}
            arms = [("existing", None), ("structured", arm_scores.get("structured")), ("hybrid", arm_scores.get("hybrid"))] if state_name == "mutated_memory_state" else [("existing", None)]
            for arm, scores in arms:
                if scores:
                    executor.scorer = CachedRanker(catalogue and PreferenceAwareScorer(), scores, arm)
                observation = PlannerObservation(
                    step=0, query=case["query"],
                    m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": case["query"], "preferences": preferences},
                )
                started = time.perf_counter()
                error = None
                try:
                    run = run_planner(planner=P14ARulePlanner(), executor=executor, initial_observation=observation, max_steps=MAX_STEPS)
                    status = executor.terminal_status
                    if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
                        status = "budget_exhausted"
                    steps = run.as_dict()["steps"]
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    status, steps = "error", []
                gate = executor.last_decision or {}
                hard_violation = bool((gate.get("hard_constraint_status") or {}).get("hard_constraints") == "violated")
                selected = executor.selected_candidate_id
                records.append({
                    "case_id": case["case_id"], "state": state_name, "category": case["category"],
                    "arm": arm, "status": status, "requires_clarification": bool(case.get("requires_clarification")),
                    "hard_constraints": bool(case.get("hard_requirements")), "hard_constraint_violation": hard_violation,
                    "selected_product_id": selected, "selected_in_eligible": selected in set(ids) if selected else None,
                    "gate_action": gate.get("action"), "actions": [step.get("action") for step in steps],
                    "memory_active_preferences": preferences, "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "error": error,
                })
    payload = {
        "schema": "agentrecx.agent_ranker_phase43.counterfactual_e2e.v1",
        "planner": "rule",
        "gpu": "Qwen planner loaded for P20; this focused invariant run is CPU + fixed BGE pool",
        "memory_writer": "ShoppingCompanionMemory.ingest_sessions",
        "records": records,
    }
    (OUT / "e2e_counterfactual.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "records": len(records)}, indent=1))


if __name__ == "__main__":
    main()
