"""Counterfactual Memory Writer -> Reader -> Ranker -> Gate E2E matrix.

16 cases x 2 memory states x 3 formal rankers = 96 trajectories.
Planner can be rule or real Qwen. Candidate pool is fixed BGE pool to isolate ranker effects.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue, ELECTRONICS_NAMESPACE
from recommendation.control.catalog_search import CatalogSearchHit, CatalogSearchResult
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE, P14ARulePlanner

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
CASES = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
POOLS = REPO / "data/p20_shadow/ranking_dataset.jsonl"
SHOP_SCORES = REPO / "runs/agent_ranker_phase44/cf_shop_scores.json"
ADAPTER = REPO / "runs/agent_native_ranking_phase42/memory_adapter_features.jsonl"
OUT = REPO / "runs/agent_ranker_phase44"


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
        self.scores = dict(scores)
        self.arm = arm
        self.last_latency_ms = None

    def score(self, *, preferences, current_query, candidates, hard_requirements=(), namespace):
        started = time.perf_counter()
        rows = self.base.score(preferences=preferences, current_query=current_query,
                               candidates=candidates, hard_requirements=hard_requirements, namespace=namespace)
        for row in rows:
            item_id = row["product_ref"]["item_id"]
            if item_id in self.scores:
                row["existing_final_score"] = row["final_score"]
                row["final_score"] = float(self.scores[item_id])
                row["arm_score"] = float(self.scores[item_id])
        rows.sort(key=lambda row: (row["hard_tier"], -float(row["final_score"]), row["query_rank"],
                                   row["product_ref"]["namespace"], row["product_ref"]["item_id"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return rows


def load_adapter():
    rows = {}
    for row in read_jsonl(ADAPTER):
        rows[(row["source"], row["group_id"], row["candidate_id"], row["memory_snapshot_id"])] = row["features"]
    return rows


def run_case_state(catalogue, case, state_name, shop_scores, adapter, planner_kind, planner_factory):
    sessions = case.get(f"{state_name.replace('_memory_state', '')}_sessions") or []
    # The key convention above would be baseline_sessions; use the stored explicit keys.
    sessions = case.get("baseline_memory_sessions" if state_name == "baseline_memory_state" else "mutated_memory_sessions") or []
    task = {
        "task_id": f"{case['case_id']}:{state_name}",
        "user_id": f"phase44_{case['case_id']}",
        "current_query": case["query"],
        "current_time": "2025-01-15T12:00:00",
        "history_sessions": sessions,
    }
    if sessions:
        preferences, _ = _memory_read(catalogue, task)
    else:
        preferences = []
    hard = case.get("hard_requirements") or []
    if not isinstance(hard, list):
        hard = []
    ids = list(case.get("eligible_candidate_ids") or [])
    state_scores = (shop_scores.get(case["case_id"]) or {}).get("states", {}).get(state_name) or {}
    query_scores = state_scores.get("query_scores") or {}
    natural_scores = state_scores.get("natural_scores") or {}
    hybrid = {}
    for item_id in ids:
        features = adapter.get(("counterfactual", case["case_id"], item_id, f"{case['case_id']}:{state_name}")) or {}
        semantic = float(query_scores.get(item_id, 0.0))
        hybrid[item_id] = semantic + 0.5 * float(features.get("positive_match_rate", 0.0)) - 0.75 * float(features.get("negative_conflict_rate", 0.0))
    arm_scores = {"existing": None, "natural": natural_scores, "hybrid": hybrid}
    records = []
    for arm, scores in arm_scores.items():
        # A fresh executor is mandatory: reusing one executor across arms leaks terminal
        # status/selected id/last_rows from the previous arm and causes budget_exhausted.
        executor = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=case["query"],
                                               preferences=preferences, hard=hard)
        executor.search_source = FixedPoolSource(ids)
        if scores:
            executor.scorer = CachedRanker(PreferenceAwareScorer(), scores, arm)
        planner = planner_factory()
        observation = PlannerObservation(
            step=0, query=case["query"],
            m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": case["query"], "preferences": preferences},
        )
        started = time.perf_counter()
        error = None
        steps = []
        try:
            run = run_planner(planner=planner, executor=executor, initial_observation=observation, max_steps=MAX_STEPS)
            status = executor.terminal_status
            if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
                status = "budget_exhausted"
            steps = run.as_dict().get("steps") or []
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            status = "error"
        gate = executor.last_decision or {}
        selected = executor.selected_candidate_id
        records.append({
            "case_id": case["case_id"], "state": state_name, "category": case["category"], "arm": arm,
            "planner": planner_kind, "status": status, "requires_clarification": bool(case.get("requires_clarification")),
            "hard_constraint_violation": bool((gate.get("hard_constraint_status") or {}).get("hard_constraints") == "violated"),
            "selected_product_id": selected, "selected_in_eligible": selected in set(ids) if selected else None,
            "gate_action": gate.get("action"), "gate": gate, "actions": [step.get("action") for step in steps],
            "tool_calls": [call.get("tool") for call in executor.tool_calls],
            "ranker_calls": sum(1 for call in executor.tool_calls if call.get("tool") == "rerank"),
            "memory_active_preferences": preferences,
            "memory_source": "synthetic_event_sequence_counterfactual_fixture",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "planner_errors": [call.get("errors") for call in getattr(planner, "calls", []) if call.get("errors")],
            "error": error,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("rule", "qwen"), default="rule")
    parser.add_argument("--case", default=None, help="optional single case id")
    parser.add_argument("--out", default=None, help="optional output filename")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    cases = read_jsonl(CASES)
    if args.case:
        cases = [case for case in cases if case["case_id"] == args.case]
    shop_scores = {row["case_id"]: row for row in json.loads(SHOP_SCORES.read_text(encoding="utf-8"))["rows"]}
    adapter = load_adapter()
    if args.planner == "qwen":
        if not os.environ.get("AGENTRECX_QWEN_PLANNER_URL"):
            raise SystemExit("AGENTRECX_QWEN_PLANNER_URL is required for qwen")
        base = HttpPlannerProposer.from_env()
        class Trace:
            def __init__(self):
                self.inner = base
                self.calls = []
            def propose(self, observation):
                proposal = self.inner.propose(observation)
                response = getattr(self.inner, "last_response", {}) or {}
                self.calls.append({"latency_ms": response.get("latency_ms"), "errors": response.get("errors") or []})
                return proposal
        planner_factory = lambda: Trace()
    else:
        planner_factory = lambda: P14ARulePlanner()
    records = []
    for case in cases:
        for state_name in ("baseline_memory_state", "mutated_memory_state"):
            records.extend(run_case_state(catalogue, case, state_name, shop_scores, adapter, args.planner, planner_factory))
    payload = {
        "schema": "agentrecx.agent_ranker_phase44.counterfactual_matrix.v1",
        "planner": args.planner,
        "cases": len(cases),
        "states": 2,
        "formal_rankers": ["existing", "natural", "hybrid"],
        "trajectories": len(records),
        "records": records,
    }
    output_name = args.out or f"cf_matrix_{args.planner}.json"
    (OUT / output_name).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"planner": args.planner, "records": len(records)}, indent=1))


if __name__ == "__main__":
    main()
