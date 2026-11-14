"""P100 Agent E2E: 100 tasks x 3 rankers with real Qwen Planner + Event Memory."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.control.catalog_search import CatalogSearchHit, CatalogSearchResult
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
DATA = REPO / "data/p100_extended_benchmark"
RUN = REPO / "runs/p100_extended_evaluation"
OUT = RUN / "p100_agent_e2e.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FixedPoolSource:
    mode = "p100_fixed_bge_pool"

    def __init__(self, ids):
        self.ids = list(ids)

    def search(self, query, limit=20):
        hits = tuple(CatalogSearchHit(parent_asin=f"amazon_electronics_c4:{item_id}", score=float(20 - index),
                                      matched_fields=("title",), bge_rank=index + 1, title=None)
                     for index, item_id in enumerate(self.ids[:limit]))
        return CatalogSearchResult(hits=hits, query=query, matched=bool(hits), catalogue_size=len(self.ids), score_kind="p100_fixed")


class CachedRanker:
    def __init__(self, scores, arm):
        self.base = PreferenceAwareScorer()
        self.scores = dict(scores)
        self.arm = arm
        self.last_latency_ms = None

    def score(self, *, preferences, current_query, candidates, hard_requirements=(), namespace):
        started = time.perf_counter()
        rows = self.base.score(preferences=preferences, current_query=current_query, candidates=candidates,
                               hard_requirements=hard_requirements, namespace=namespace)
        for row in rows:
            item_id = row["product_ref"]["item_id"]
            if item_id in self.scores:
                row["final_score"] = float(self.scores[item_id])
                row["arm"] = self.arm
        rows.sort(key=lambda row: (row["hard_tier"], -float(row["final_score"]), row["query_rank"],
                                   row["product_ref"]["namespace"], row["product_ref"]["item_id"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return rows


def topk(catalogue, executor, gate, k=5):
    started = time.perf_counter()
    if not (gate or {}).get("allow_finalize"):
        return {"items": [], "ask": True, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
    items = []
    for row in executor.last_rows:
        item_id = row["product_ref"]["item_id"]
        product = catalogue.get(item_id)
        attrs = list((product.attributes or {}).keys()) if product else []
        evidence = list(row.get("evidence_refs") or [])
        if int(row.get("hard_tier", 1)) != 0 or not row.get("presentable"):
            continue
        items.append({"product_id": item_id, "rank": row.get("rank"), "ranker_score": float(row.get("final_score", 0.0)),
                      "verified_evidence": evidence,
                      "memory_matches": [str(x.get("value")) for x in row.get("matched_preferences") or []],
                      "memory_conflicts": [str(x.get("value")) for x in row.get("conflicting_preferences") or []],
                      "unknown_preferences": [str(x.get("value")) for x in row.get("unknown_preferences") or []],
                      "hard_eligible": True, "evidence_supported": bool(evidence or attrs)})
        if len(items) >= k:
            break
    return {"items": items, "ask": False, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}


def main():
    tasks = read_jsonl(DATA / "p100_agent_tasks_memory_on.jsonl")
    pools = {row["base_query_id"]: row for row in read_jsonl(DATA / "p100_bge_pools.jsonl")}
    preds = {row["base_query_id"]: row for row in read_jsonl(RUN / "p100_ranker_gpu_predictions.jsonl")}
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    proposer = HttpPlannerProposer.from_env()
    records = []
    if OUT.exists():
        OUT.unlink()
    for task in tasks:
        bid = task["base_query_id"]
        pool = pools[bid]
        ids = pool["candidate_pool_ids"]
        for arm in ("existing", "natural", "hybrid"):
            executor = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=task["current_query"],
                                                   preferences=[], hard=list(task.get("hard") or []))
            executor.search_source = FixedPoolSource(ids)
            preferences, _ = _memory_read(catalogue, task)
            executor.history_preferences = list(preferences)
            executor.preferences = list(preferences)
            if arm != "existing":
                executor.scorer = CachedRanker(preds[bid]["modes"]["ON"][arm], arm)
            observation = PlannerObservation(step=0, query=task["current_query"],
                                             m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                                                       "preferences": preferences})
            calls = []
            class Trace:
                def propose(self, obs):
                    proposal = proposer.propose(obs)
                    response = getattr(proposer, "last_response", {}) or {}
                    calls.append({"latency_ms": response.get("latency_ms"), "errors": response.get("errors") or [],
                                  "action": proposal.action.value})
                    return proposal
            started = time.perf_counter()
            error = None
            steps = []
            try:
                run = run_planner(planner=Trace(), executor=executor, initial_observation=observation, max_steps=MAX_STEPS)
                status = executor.terminal_status
                if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
                    status = "budget_exhausted"
                steps = run.as_dict().get("steps") or []
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                status = "error"
            gate = executor.last_decision or {}
            presented = topk(catalogue, executor, gate, k=5)
            item_ids = [item["product_id"] for item in presented["items"]]
            record = {
                "task_id": task["task_id"], "base_query_id": bid, "query_id": task["query_id"],
                "query": task["current_query"], "scenario_type": task["scenario_type"], "arm": arm,
                "memory_mode": "ON", "status": status, "error": error,
                "actions": [step.get("action") for step in steps], "planner_calls": calls,
                "gate_action": gate.get("action"), "gate_reason": gate.get("reason_code"),
                "selected_product_id": executor.selected_candidate_id,
                "selected_in_eligible": executor.selected_candidate_id in ids if executor.selected_candidate_id else None,
                "top1": item_ids[:1], "top3": item_ids[:3], "top5": item_ids[:5],
                "ask": bool(presented["ask"]), "valid_topk": len(item_ids),
                "evidence_supported": sum(1 for item in presented["items"] if item["evidence_supported"]),
                "hard_violations": sum(1 for item in presented["items"] if not item["hard_eligible"]),
                "active_preferences": preferences,
                "ranker_latency_ms": getattr(executor.scorer, "last_latency_ms", None),
                "topk_latency_ms": presented["latency_ms"],
                "e2e_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "items": presented["items"],
            }
            records.append(record)
            with OUT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{task['task_id']} {arm} {status} {record['gate_action']} {record['e2e_latency_ms']:.0f}ms", flush=True)
    print(json.dumps({"records": len(records), "tasks": len(tasks), "arms": 3}, indent=1))


if __name__ == "__main__":
    main()
