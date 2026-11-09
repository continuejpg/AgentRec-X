"""P12 retrieval/agent diagnosis over the existing P11 artifacts.

No model or index is changed.  The script separates hybrid-fusion behavior, history-cache
protocol, bounded-pool under-labelling, and Agent decision failures.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data/p11_benchmark/tasks.jsonl"
P11_EVAL = ROOT / "runs/p11/eval/per_task.jsonl"
P11_SUMMARY = ROOT / "runs/p11/eval/summary.json"
AGENT_RESULTS = ROOT / "runs/p11/agent/results.jsonl"
CACHE = ROOT / "runs/p8_d/multiuser/candidate_cache.jsonl"
BGE_META = ROOT / "runs/p8_bge/bge_base_dense.npz.json"
OUT = ROOT / "runs/p12/diagnosis"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = load_jsonl(TASKS)
    eval_rows = {row["task_id"]: row for row in load_jsonl(P11_EVAL)}
    agent_rows = load_jsonl(AGENT_RESULTS)
    cache_rows = {str(row["user_id"]): row for row in load_jsonl(CACHE)}
    bge_meta = json.loads(BGE_META.read_text(encoding="utf-8"))

    pool_tasks = [task for task in tasks if task.get("verified_candidate_pool")]
    query_consistency = {
        "tasks": len(tasks),
        "same_query_for_bm25_bge_hybrid": all(
            eval_rows[task["task_id"]]["query"] == task["conversation_turns"][0]
            for task in tasks
            if task["task_id"] in eval_rows
        ),
        "empty_pool_tasks": len(tasks) - len(pool_tasks),
        "pool_tasks": len(pool_tasks),
    }

    hybrid_lost = []
    hybrid_added = []
    for task in pool_tasks:
        row = eval_rows.get(task["task_id"])
        if not row:
            continue
        bm25 = row["catalog"]["bm25"]["top50"]["reachable"]
        hybrid = row["catalog"]["hybrid"]["top50"]["reachable"]
        if bm25 and not hybrid:
            hybrid_lost.append(task["task_id"])
        if hybrid and not bm25:
            hybrid_added.append(task["task_id"])
    hybrid_diag = {
        "pool_tasks": len(pool_tasks),
        "bm25_reachable@50": sum(
            1 for task in pool_tasks
            if eval_rows.get(task["task_id"], {}).get("catalog", {}).get("bm25", {}).get("top50", {}).get("reachable")
        ),
        "bge_reachable@50": sum(
            1 for task in pool_tasks
            if eval_rows.get(task["task_id"], {}).get("catalog", {}).get("bge", {}).get("top50", {}).get("reachable")
        ),
        "hybrid_reachable@50": sum(
            1 for task in pool_tasks
            if eval_rows.get(task["task_id"], {}).get("catalog", {}).get("hybrid", {}).get("top50", {}).get("reachable")
        ),
        "bm25_hit_hybrid_miss": len(hybrid_lost),
        "hybrid_hit_bm25_miss": len(hybrid_added),
        "bm25_hit_hybrid_miss_examples": hybrid_lost[:10],
        "root_cause_code": (
            "CatalogSearchSource hybrid builds bm25_rank from ALL scored BM25 documents "
            "(not a bounded head), while dense candidates are limited to `limit`; RRF "
            "then returns only the top `limit`. Dense candidates can therefore displace "
            "many BM25 head candidates. This is a fusion/budget-boundary issue, not "
            "evidence that BGE is semantically superior."
        ),
    }

    cache_consistent = 0
    cache_mismatch = []
    for task in tasks:
        row = cache_rows.get(str(task["user_id"]))
        if row and list(row.get("history") or ()) == list(task["visible_user_history"]):
            cache_consistent += 1
        elif row:
            cache_mismatch.append(task["task_id"])
    history_diag = {
        "tasks": len(tasks),
        "history_cache_rows_for_task_users": sum(1 for task in tasks if task["user_id"] in cache_rows),
        "exact_history_matches": cache_consistent,
        "history_mismatch_examples": cache_mismatch[:10],
        "history_reachable_at_50": sum(
            1 for row in eval_rows.values() if row.get("history_reachable")
        ),
        "scope": "P8-D history source cache is query-independent; it is the History Tool's real output, not a task-query-driven retriever.",
        "duplicate_lexical_note": (
            "Serving V1 history source itself includes catalog_lexical BM25; the external "
            "catalog_search in the Agent plan can re-introduce a second BM25 signal."
        ),
    }

    outside_pool = []
    invalid_agent = []
    for row in agent_rows:
        if row.get("catalog_validity_success") and not row.get("verified_success"):
            outside_pool.append(row["task_id"])
        if row.get("invalid_finalize"):
            invalid_agent.append(row["task_id"])
    agent_diag = {
        "tasks": len(agent_rows),
        "strict_verified_success": sum(1 for row in agent_rows if row.get("verified_success")),
        "catalog_validity_success": sum(1 for row in agent_rows if row.get("catalog_validity_success")),
        "invalid_finalize": len(invalid_agent),
        "hard_constraint_violation": sum(1 for row in agent_rows if row.get("hard_constraint_violation")),
        "catalog_valid_but_outside_bounded_pool": outside_pool,
        "strict_vs_catalog_gap_cause": "bounded pool cap/under-labelling; final IDs can be catalogue-valid without being in the bounded pool",
        "failure_categories": {
            "planner_or_finalize": len(invalid_agent),
            "retrieval_or_unverified_candidate": sum(1 for row in agent_rows if row.get("hard_constraint_violation")),
            "bounded_pool_underlabelled": len(outside_pool),
        },
    }

    diagnosis = {
        "schema_version": "agentrecx.p12.diagnosis.v1",
        "tasks": len(tasks),
        "query_consistency": query_consistency,
        "bge_integrity": {
            "model_id": bge_meta.get("model_id"),
            "model_revision": bge_meta.get("model_revision"),
            "embedding_dim": bge_meta.get("embedding_dim"),
            "index_item_count": bge_meta.get("item_count"),
            "query_instruction": bge_meta.get("query_instruction"),
            "finding": "no evidence of wrong model/index/query-instruction path; the main P11 coverage issue is candidate/pool boundary and fusion, not a proven BGE defect",
        },
        "hybrid": hybrid_diag,
        "history": history_diag,
        "agent": agent_diag,
        "bounded_pool": {
            "pool_limit": 100,
            "pool_tasks": len(pool_tasks),
            "note": "P11 verified candidate pool is capped/sorted by ASIN; it is not a full-catalogue oracle",
        },
    }
    (args.out / "diagnosis.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
