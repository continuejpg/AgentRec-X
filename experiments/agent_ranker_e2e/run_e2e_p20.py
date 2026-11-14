"""Phase 4.3 P20 Agent E2E: real Qwen Planner + real Memory Writer/Reader + ranker + Gate.

Three experimental ranker arms:
  existing  = PreferenceAwareScorer
  natural   = frozen ShopRanker natural-memory scores (precomputed by the GPU runner)
  hybrid    = fixed ShopRanker + structured-memory hybrid scores

ShopRanker scores are injected via a precomputed ranker so the Qwen planner can occupy the
24GB GPU alone; this is the sequential-loading deployment path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from experiments.p14c0.electronics_executor import (
    MAX_STEPS,
    ElectronicsPlannerExecutor,
    prepare_hard,
)
from experiments.p14c0.run_dev import _memory_read
from experiments.p14c1.search_backend import configure_catalogue
from experiments.p14e.catalogue import ElectronicsCatalogue
from experiments.p20_metadata.corrective_eval import _is_success
from recommendation.demo.planner_contract import PlannerObservation, run_planner
from recommendation.demo.planner_http_proposer import HttpPlannerProposer
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.planner_bridge import PHASE_PRODUCT_RETRIEVE

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
TASK_SUITE = REPO / "runs/p20_agent_eval/task_suite.json"
SCORES = REPO / "runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json"
OUT = REPO / "runs/agent_ranker_phase43"
SIDECAR = REPO / "data/enrichment/electronics_metadata_sidecar.jsonl"


class TracePlanner:
    name = "qwen_trace"

    def __init__(self, inner: Any):
        self.inner = inner
        self.calls = []

    def propose(self, observation):
        proposal = self.inner.propose(observation)
        response = getattr(self.inner, "last_response", {}) or {}
        self.calls.append({
            "latency_ms": response.get("latency_ms"),
            "input_tokens": response.get("input_tokens"),
            "output_tokens": response.get("output_tokens"),
            "action": proposal.action.value,
            "errors": response.get("errors") or [],
        })
        return proposal


class CachedArmRanker:
    """Base Scorer rows + cached ShopRanker arm score ordering."""

    def __init__(self, arm_scores: dict[str, float], arm: str):
        self.base = PreferenceAwareScorer()
        self.arm_scores = dict(arm_scores)
        self.arm = arm
        self.last_latency_ms = None

    def score(self, *, preferences, current_query, candidates, hard_requirements=(), namespace):
        started = time.perf_counter()
        rows = self.base.score(
            preferences=preferences, current_query=current_query, candidates=candidates,
            hard_requirements=hard_requirements, namespace=namespace,
        )
        for row in rows:
            item_id = row["product_ref"]["item_id"]
            if item_id in self.arm_scores:
                row["existing_final_score"] = row["final_score"]
                row["final_score"] = float(self.arm_scores[item_id])
                row["arm_score"] = float(self.arm_scores[item_id])
                row["arm"] = self.arm
        rows.sort(key=lambda row: (
            row["hard_tier"], -float(row["final_score"]), row["query_rank"],
            row["product_ref"]["namespace"], row["product_ref"]["item_id"],
        ))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return rows


class TimedSearchSource:
    def __init__(self, inner: Any):
        self.inner = inner
        self.timings = []
        self.mode = getattr(inner, "mode", "unknown")

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def search(self, query: str, limit: int = 20):
        started = time.perf_counter()
        result = self.inner.search(query, limit=limit)
        self.timings.append(round((time.perf_counter() - started) * 1000.0, 3))
        return result


def final_answer(selected: str | None, catalogue: ElectronicsCatalogue, row: dict | None) -> dict:
    if not selected:
        return {"answer": None, "evidence_supported": None, "attributes": [], "unknowns": []}
    product = catalogue.get(selected)
    if product is None:
        return {"answer": None, "evidence_supported": False, "attributes": [], "unknowns": ["product_not_found"]}
    attrs = list((product.attributes or {}).keys())
    evidence = []
    for attr in attrs[:6]:
        fact = (product.attributes or {}).get(attr) or {}
        text = fact.get("evidence") if isinstance(fact, dict) else None
        evidence.append({"attribute": attr, "evidence": text})
    matched = [str(item.get("value")) for item in (row or {}).get("matched_preferences") or []]
    unknowns = ["brand", "price", "size", "material", "stock"]
    answer = f"Recommended {product.title}. Verified attributes: {', '.join(attrs) if attrs else 'none'}; matched active preferences: {', '.join(matched) if matched else 'none'}."
    return {
        "answer": answer[:800],
        "evidence_supported": bool(attrs),
        "attributes": evidence,
        "matched_preferences": matched,
        "unknown_fields": unknowns,
        "generator": "deterministic_evidence_template",
    }


def run_one(catalogue, task, arm, arm_scores, planner):
    preferences, history = _memory_read(catalogue, task)
    executor = ElectronicsPlannerExecutor(
        catalogue=catalogue, current_query=task["current_query"],
        preferences=preferences, hard=list(task["hard"]),
    )
    timed_search = TimedSearchSource(catalogue.search_source)
    executor.search_source = timed_search
    if arm != "existing":
        executor.scorer = CachedArmRanker(arm_scores, arm)
    observation = PlannerObservation(
        step=0, query=task["current_query"],
        m5_state={"phase": PHASE_PRODUCT_RETRIEVE, "next_query": task["current_query"],
                  "preferences": preferences},
    )
    started = time.perf_counter()
    error = None
    try:
        run = run_planner(planner=planner, executor=executor, initial_observation=observation, max_steps=MAX_STEPS)
        status = executor.terminal_status
        if status in {"not_started", "in_progress", "need_more_candidates", "need_more_product_evidence"}:
            status = "budget_exhausted"
        steps = run.as_dict()["steps"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        status, steps = "error", []
    selected = executor.selected_candidate_id
    row = executor._rows_by_rank.get(selected or "")
    answer_started = time.perf_counter()
    answer = final_answer(selected, catalogue, row)
    answer_latency = (time.perf_counter() - answer_started) * 1000.0
    hard_violation = False
    gate = executor.last_decision or {}
    hard_status = (gate.get("hard_constraint_status") or {})
    if hard_status.get("hard_constraints") == "violated":
        hard_violation = True
    selected_eligible = None
    if selected and executor.last_rows:
        selected_eligible = selected in {r["product_ref"]["item_id"] for r in executor.last_rows if r.get("presentable")}
    return {
        "task_id": task["task_id"], "arm": arm, "query": task["current_query"],
        "expected": task["expected"], "status": status,
        "success": _is_success(task, status, selected, catalogue, task.get("scored_evaluator")),
        "selected_product_id": selected, "selected_eligible": selected_eligible,
        "hard_constraint_violation": hard_violation,
        "gate_action": gate.get("action"), "gate": gate,
        "final_answer": answer, "final_answer_latency_ms": round(answer_latency, 3),
        "actions": [step.get("action") for step in steps],
        "planner_calls": len(planner.calls),
        "planner_latencies_ms": [call.get("latency_ms") for call in planner.calls],
        "bge_retrieval_latencies_ms": timed_search.timings,
        "memory_preferences": preferences,
        "memory_events_source": "synthetic_event_sequence_p20_fixture",
        "ranker_latency_ms": getattr(executor.scorer, "last_latency_ms", None),
        "planner_errors": [call.get("errors") for call in planner.calls if call.get("errors")],
        "e2e_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "error": error,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENTRECX_METADATA_SIDECAR", str(SIDECAR))
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    retrieval_status = configure_catalogue(catalogue, mode="bge")
    # Warm the BGE encoder outside the measured task p50; record its cold load separately.
    warmup_started = time.perf_counter()
    catalogue.search_source.search("wireless earbuds for working out", limit=20)
    retrieval_warmup_ms = round((time.perf_counter() - warmup_started) * 1000.0, 3)
    score_data = json.loads(SCORES.read_text(encoding="utf-8"))
    p20_scores = {row["task_id"]: row for row in score_data["p20"]["details"]}
    suite = json.loads(TASK_SUITE.read_text(encoding="utf-8"))
    tasks = [task for task in suite["tasks"] if task["expected"] == "recommend"]
    proposer = HttpPlannerProposer.from_env()
    arms = ("existing", "natural", "hybrid")
    records = []
    for task in tasks:
        for arm in arms:
            planner = TracePlanner(proposer)
            if arm == "existing":
                scores = {}
            elif arm == "natural":
                scores = p20_scores[task["task_id"]]["scores"]["natural"]
            else:
                scores = p20_scores[task["task_id"]]["scores"]["hybrid"]
            record = run_one(catalogue, task, arm, scores, planner)
            records.append(record)
            print(json.dumps({"task": task["task_id"], "arm": arm, "status": record["status"],
                              "success": record["success"], "selected": record["selected_product_id"],
                              "e2e_ms": record["e2e_latency_ms"], "planner_calls": record["planner_calls"],
                              "error": record["error"]}, ensure_ascii=False), flush=True)
    payload = {
        "schema": "agentrecx.agent_ranker_phase43.p20_e2e.v1",
        "gpu": "RTX 4090 D 24GB; Qwen3.5-9B planner loaded alone; ShopRanker scores injected from GPU run",
        "model_lifecycle": "sequential: ShopRanker GPU run -> unload -> Qwen planner + CPU ranker injection",
        "retrieval": retrieval_status,
        "bge_warmup_ms": retrieval_warmup_ms,
        "tasks": len(tasks),
        "arms": list(arms),
        "records": records,
    }
    (OUT / "e2e_p20.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "records": len(records)}, indent=1))


if __name__ == "__main__":
    main()
