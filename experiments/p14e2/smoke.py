"""P14-E2 Lite smoke: a few Train/Dev tasks through the real tool chain (no LLM).

Uses M9/M5 history-state read, real BM25 over P13-R, real Product Info evidence,
CandidateLedger, Eligibility, the frozen Preference Scorer, the frozen Decision Gate.
Reserved Test tasks are never selected here.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import replace
from pathlib import Path

from experiments.p14b1.diagnose import (
    canonical_prefs, candidate_rows, conflicts, hard_requirements, query_specificity, verified_ids,
)
from experiments.p14e.catalogue import ELECTRONICS_NAMESPACE, ElectronicsCatalogue
from recommendation.control.arguments import CandidateSource
from recommendation.control.candidate_ledger import CandidateLedger
from recommendation.control.constraint_eligibility import CandidateEligibilityEvaluator, ConstraintRequirement
from recommendation.control.grounded_reasoning import ConstraintKind, GroundedReasoner
from recommendation.preference.decision_gate import (
    NEED_MORE_CANDIDATES, NEED_MORE_PRODUCT_EVIDENCE, NEED_USER_CLARIFICATION,
    DecisionGate, GateBudget, GateContext,
)
from recommendation.preference.history import HistoryStateReader
from recommendation.preference.scorer import PreferenceAwareScorer
from recommendation.shopping_companion.memory import SOURCE_KIND_SYNTHETIC_FIXTURE, ShoppingCompanionMemory

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14e2/benchmark"
OUT = REPO / "runs/p14e2/smoke"
TASKS = [
    ("train", "p14e2_lite_0001"),   # no-history, explicit hard constraint -> recommend
    ("train", "p14e2_lite_0009"),   # history-influenced -> recommend
    ("train", "p14e2_lite_0019"),   # same-query pair member -> recommend / comparison
    ("train", "p14e2_lite_0031"),   # withdrawal family -> recommend
    ("dev", "p14e2_lite_0057"),     # ASK family -> clarification
    ("dev", "p14e2_lite_0060"),     # evidence family -> no verified / insufficient
]


def _load(split: str, kind: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.{kind}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trajectories").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    catalogue = ElectronicsCatalogue.load()
    search = catalogue.search_source
    scorer = PreferenceAwareScorer()
    gate = DecisionGate()
    reasoner = GroundedReasoner(catalogue)
    eligibility = CandidateEligibilityEvaluator(reasoner)
    agents, evals = {}, {}
    for split in ("train", "dev"):
        agents.update({row["task_id"]: row for row in _load(split, "agent")})
        evals.update({row["task_id"]: row for row in _load(split, "evaluator")})

    results = []
    for split, task_id in TASKS:
        assert task_id in agents, task_id
        task = agents[task_id]
        query = task["current_query"]
        with ShoppingCompanionMemory() as memory:
            memory.ingest_sessions(task_id, [dict(s) for s in task["history_sessions"]],
                                   source_kind=SOURCE_KIND_SYNTHETIC_FIXTURE, source_ref=f"p14e2:{task_id}")
            history = HistoryStateReader(memory.preference_service).read(
                task_id, conversation_id=task_id, current_query=query)
        prefs = canonical_prefs(history["active_preferences"])
        hard = hard_requirements(query)
        pref_conflicts = conflicts(prefs)
        specificity = query_specificity(query, hard, prefs)
        result = search.search(query, limit=20)
        ids = [hit.parent_asin.split(":", 1)[-1] for hit in result.hits]
        ledger = CandidateLedger()
        ledger.record_from_source(
            source=CandidateSource.CATALOG_SEARCH,
            candidates=[(hit.parent_asin, rank + 1, hit.score) for rank, hit in enumerate(result.hits)],
            score_kind="bm25", source_query=query[:200], step_index=1,
            grounded=frozenset(hit.parent_asin for hit in result.hits))
        pool = {"task_id": task_id, "current_query": query, "candidate_ids": ids,
                "search_scores": {ids[i]: hit.score for i, hit in enumerate(result.hits)}}
        rows = candidate_rows(pool, catalogue, scorer, prefs, hard)
        requirements = [ConstraintRequirement(kind=ConstraintKind.CATEGORY, expected="Electronics",
                                              source_text="catalogue category")]
        feasibility = eligibility.assess([f"{ELECTRONICS_NAMESPACE}:{i}" for i in ids[:10]],
                                          requirements=requirements)
        inspected = [catalogue.inspect(i, requested=sorted({p["value"] for p in hard} | {"price", "brand"}))
                     for i in ids[:3]]
        context = {"query": query, "active_preferences": prefs, "hard_requirements": hard,
                   "preference_conflicts": pref_conflicts, "namespace": ELECTRONICS_NAMESPACE,
                   "query_specificity": specificity}
        budget = GateBudget(max_retrievals=1, max_product_evidence_rounds=2)
        decision = gate.check(GateContext(candidates=rows, budget=budget, **context))
        trace = [decision.as_dict()]
        while decision.action in (NEED_MORE_PRODUCT_EVIDENCE, NEED_MORE_CANDIDATES) and len(trace) < 4:
            if decision.action == NEED_MORE_PRODUCT_EVIDENCE:
                budget = replace(budget, product_evidence_rounds_used=budget.product_evidence_rounds_used + 1)
            else:
                budget = replace(budget, retrievals_used=budget.retrievals_used + 1)
            decision = gate.check(GateContext(candidates=rows, budget=budget, **context))
            trace.append(decision.as_dict())
        ev = evals[task_id]
        verified = verified_ids(ev, catalogue) if ev["expected_decision"] == "recommend" else set()
        selected = decision.selected_candidate_id if decision.allow_finalize else None
        row = {
            "task_id": task_id, "split": split, "task_type": ev["task_type"],
            "expected_decision": ev["expected_decision"], "gate_action": decision.action,
            "allow_finalize": decision.allow_finalize, "selected_candidate_id": selected,
            "selected_correct": selected in verified if selected else None,
            "verified_total": len(verified),
            "unknown_fields": list(decision.unknown_fields),
            "tools": ["mem_history_state", "product_search", "product_view", "eligibility", "decision_gate"],
            "ledger_entries": len(ledger.ranked_parent_asins()),
            "eligibility": {"verified": feasibility.verified_eligible_count,
                            "ineligible": feasibility.ineligible_count,
                            "unresolved": feasibility.unresolved_count},
            "gate_trace": trace,
        }
        results.append(row)
        (OUT / "trajectories" / f"{task_id}.json").write_text(
            json.dumps({"task": task_id, "split": split, **row}, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")

    summary = {
        "schema": "agentrecx.p14e2.smoke.summary.v1",
        "tasks": results,
        "resources": {"wall_seconds": round(time.perf_counter() - started, 2), "peak_rss_mb": _rss_mb()},
        "notes": ["train/dev only; reserved_test never selected",
                  "real M9/M5 read, BM25, Product Info, CandidateLedger, Eligibility, frozen Scorer and Gate",
                  "no LLM planner; evaluator labels are read only for this post-run summary"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
