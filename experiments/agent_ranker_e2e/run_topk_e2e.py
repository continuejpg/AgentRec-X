"""P20 Top-K Agent E2E Shadow: Qwen Planner + Memory + ranker + Gate + Top-K validation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from experiments.p14c0.electronics_executor import MAX_STEPS, ElectronicsPlannerExecutor
from experiments.p14c0.run_dev import _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
TASK_SUITE = REPO / "runs/p20_agent_eval/task_suite.json"
SCORES = REPO / "runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json"
RANKING = REPO / "data/p20_shadow/ranking_dataset.jsonl"
PER_TASK = REPO / "data/p20_shadow/per_task_results.jsonl"
OUT = REPO / "runs/agent_ranker_phase45"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Trace:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def propose(self, observation):
        proposal = self.inner.propose(observation)
        response = getattr(self.inner, "last_response", {}) or {}
        self.calls.append({"latency_ms": response.get("latency_ms"), "errors": response.get("errors") or []})
        return proposal


class CachedRanker:
    def __init__(self, scores, arm):
        self.base = PreferenceAwareScorer()
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
                row["final_score"] = float(self.scores[item_id])
                row["arm"] = self.arm
        rows.sort(key=lambda row: (row["hard_tier"], -float(row["final_score"]), row["query_rank"],
                                   row["product_ref"]["namespace"], row["product_ref"]["item_id"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return rows


def topk_presenter(catalogue, executor, gate, k=5):
    started = time.perf_counter()
    items = []
    if not (gate or {}).get("allow_finalize"):
        return {"items": [], "ask": True, "gate_action": (gate or {}).get("action"), "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
    for row in executor.last_rows:
        item_id = row["product_ref"]["item_id"]
        product = catalogue.get(item_id)
        hard_ok = int(row.get("hard_tier", 1)) == 0
        presentable = bool(row.get("presentable"))
        evidence_refs = list(row.get("evidence_refs") or [])
        attrs = list((product.attributes or {}).keys()) if product else []
        item = {
            "product_id": item_id,
            "rank": row.get("rank"),
            "ranker_score": float(row.get("final_score", 0.0)),
            "verified_evidence": evidence_refs,
            "attributes": attrs,
            "memory_matches": [str(x.get("value")) for x in row.get("matched_preferences") or []],
            "memory_conflicts": [str(x.get("value")) for x in row.get("conflicting_preferences") or []],
            "unknown_preferences": [str(x.get("value")) for x in row.get("unknown_preferences") or []],
            "hard_eligible": bool(hard_ok and presentable),
            "presentable": presentable,
            "evidence_supported": bool(evidence_refs or attrs),
        }
        if item["hard_eligible"]:
            items.append(item)
        if len(items) >= k:
            break
    return {"items": items, "ask": False, "gate_action": (gate or {}).get("action"), "latency_ms": round((time.perf_counter() - started) * 1000, 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENTRECX_METADATA_SIDECAR", str(REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"))
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    retrieval_status = configure_catalogue(catalogue, mode="bge")
    catalogue.search_source.search("wireless earbuds for working out", limit=20)
    suite = json.loads(TASK_SUITE.read_text(encoding="utf-8"))
    tasks = [task for task in suite["tasks"] if task["expected"] == "recommend"]
    ranking = {row["task_id"]: row for row in read_jsonl(RANKING)}
    per_task = {row["task_id"]: row for row in read_jsonl(PER_TASK)}
    scores = {row["task_id"]: row for row in json.loads(SCORES.read_text(encoding="utf-8"))["p20"]["details"]}
    proposer = HttpPlannerProposer.from_env()
    records = []
    for task in tasks:
        ranking_row = ranking[task["task_id"]]
        acceptable = set(ranking_row.get("acceptable_ids_in_pool") or [])
        for arm in ("existing", "natural", "hybrid"):
            executor = ElectronicsPlannerExecutor(catalogue=catalogue, current_query=task["current_query"],
                                                   preferences=[], hard=list(task["hard"]))
            preferences, _ = _memory_read(catalogue, task)
            executor.history_preferences = list(preferences)
            executor.preferences = list(preferences)
            if arm != "existing":
                arm_scores = scores[task["task_id"]]["scores"][arm]
                executor.scorer = CachedRanker(arm_scores, arm)
            planner = Trace(proposer)
            observation = PlannerObservation(step=0, query=task["current_query"],
                                             m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                                                       "preferences": preferences})
            started = time.perf_counter()
            error = None
            try:
                run = run_planner(planner=planner, executor=executor, initial_observation=observation, max_steps=MAX_STEPS)
                status = executor.terminal_status
                if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
                    status = "budget_exhausted"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                status = "error"
            gate = executor.last_decision or {}
            topk = topk_presenter(catalogue, executor, gate, k=5)
            top_items = topk["items"]
            first = next((index + 1 for index, item in enumerate(top_items) if item["product_id"] in acceptable), None)
            records.append({
                "task_id": task["task_id"], "arm": arm, "status": status,
                "gate_action": gate.get("action"), "error": error,
                "top1": [item["product_id"] for item in top_items[:1]],
                "top3": [item["product_id"] for item in top_items[:3]],
                "top5": [item["product_id"] for item in top_items[:5]],
                "hit@1": bool(first == 1), "hit@3": bool(first and first <= 3), "hit@5": bool(first and first <= 5),
                "mrr@10": round(1.0 / first, 6) if first and first <= 10 else 0.0,
                "valid_count": len(top_items),
                "evidence_supported": sum(1 for item in top_items if item["evidence_supported"]),
                "hard_violations": sum(1 for item in top_items if not item["hard_eligible"]),
                "eligible_coverage": round(len(top_items) / max(1, min(5, len(per_task[task["task_id"]]["eligible_candidate_ids"]))), 6),
                "ask": topk["ask"],
                "topk_latency_ms": topk["latency_ms"],
                "ranker_latency_ms": getattr(executor.scorer, "last_latency_ms", None),
                "planner_calls": len(planner.calls),
                "planner_latency_ms": [call["latency_ms"] for call in planner.calls],
                "e2e_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "items": top_items,
            })
    payload = {"schema": "agentrecx.agent_ranker_phase45.topk_e2e.v1",
               "retrieval": retrieval_status,
               "tasks": len(tasks), "arms": ["existing", "natural", "hybrid"], "records": records}
    (OUT / "topk_e2e.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "records": len(records)}, indent=1))


if __name__ == "__main__":
    main()
