"""Post-hoc CF ranking metrics: Existing/Natural/Hybrid, both memory states.

These rankings use the exact fixed candidate pool and arm scores; they supplement the
E2E finalization statuses.  No independent preference gold exists for the CF cases.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.p14b1.diagnose import candidate_rows
from experiments.p14e.catalogue import ElectronicsCatalogue
from recommendation.preference.scorer import PreferenceAwareScorer

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
CASES = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
POOLS = REPO / "data/p20_shadow/ranking_dataset.jsonl"
SHOP = REPO / "runs/agent_ranker_phase44/cf_shop_scores.json"
ADAPTER = REPO / "runs/agent_native_ranking_phase42/memory_adapter_features.jsonl"
OUT = REPO / "runs/agent_ranker_phase44/cf_ranking_metrics.json"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    catalogue = ElectronicsCatalogue.load(metadata_mode="enriched")
    cases = read_jsonl(CASES)
    pools = {row["task_id"]: row for row in read_jsonl(POOLS)}
    shop = {row["case_id"]: row for row in json.loads(SHOP.read_text(encoding="utf-8"))["rows"]}
    adapter = {}
    for row in read_jsonl(ADAPTER):
        adapter[(row["source"], row["group_id"], row["candidate_id"], row["memory_snapshot_id"])] = row["features"]
    rows = []
    for case in cases:
        pool = pools[case["pool_id"]]
        ids = list(case.get("eligible_candidate_ids") or [])
        by_id = {item["item_id"]: item for item in pool["candidate_pool"]}
        for state in ("baseline_memory_state", "mutated_memory_state"):
            memory = (case.get(state) or {}).get("active_preferences") or []
            task = {"task_id": "cf", "current_query": case["query"], "candidate_ids": ids,
                    "search_scores": {}}
            base_rows = candidate_rows(task, catalogue, PreferenceAwareScorer(), memory,
                                       case.get("hard_requirements") or [])
            base_scores = {row["product_ref"]["item_id"]: row["final_score"] for row in base_rows}
            state_scores = shop[case["case_id"]]["states"][state]
            natural_scores = state_scores["natural_scores"]
            hybrid = {}
            for item_id in ids:
                features = adapter.get(("counterfactual", case["case_id"], item_id, f"{case['case_id']}:{state}")) or {}
                hybrid[item_id] = float(state_scores["query_scores"].get(item_id, 0.0)) + \
                    0.5 * float(features.get("positive_match_rate", 0.0)) - \
                    0.75 * float(features.get("negative_conflict_rate", 0.0))
            rankings = {}
            for arm, scores in (("existing", base_scores), ("natural", natural_scores), ("hybrid", hybrid)):
                rankings[arm] = sorted(ids, key=lambda item_id: (-float(scores.get(item_id, -1e9)), item_id))
            rows.append({
                "case_id": case["case_id"], "category": case["category"], "state": state,
                "requires_clarification": bool(case.get("requires_clarification")),
                "active_preferences": memory,
                "rankings": rankings,
            })
    # State-level summaries by arm.
    summary = {}
    for arm in ("existing", "natural", "hybrid"):
        by_case = {}
        for case_id in {row["case_id"] for row in rows}:
            pair = {row["state"]: row for row in rows if row["case_id"] == case_id}
            b, m = pair["baseline_memory_state"], pair["mutated_memory_state"]
            br, mr = b["rankings"][arm], m["rankings"][arm]
            top1_changed = br[:1] != mr[:1]
            top3_changed = set(br[:3]) != set(mr[:3])
            # Pairwise order change over all candidate pairs.
            changed_pairs = total_pairs = 0
            for i in range(len(br)):
                for j in range(i + 1, len(br)):
                    total_pairs += 1
                    if (br.index(br[i]) < br.index(br[j])) != (mr.index(br[i]) < mr.index(br[j])):
                        changed_pairs += 1
            by_case[case_id] = {
                "category": b["category"], "requires_clarification": b["requires_clarification"],
                "top1_changed": top1_changed, "top3_set_changed": top3_changed,
                "pair_order_changed": br != mr, "changed_pairs": changed_pairs, "total_pairs": total_pairs,
                "baseline_top1": br[:1], "mutated_top1": mr[:1],
            }
        summary[arm] = by_case
    payload = {
        "schema": "agentrecx.agent_ranker_phase44.cf_ranking_metrics.v1",
        "provenance": "mechanistic; no independent direction gold",
        "rows": rows,
        "summary": summary,
        "by_arm": {
            arm: {
                "cases": len(cases),
                "top1_changed": sum(1 for value in summary[arm].values() if value["top1_changed"]),
                "top3_set_changed": sum(1 for value in summary[arm].values() if value["top3_set_changed"]),
                "pair_order_changed": sum(1 for value in summary[arm].values() if value["pair_order_changed"]),
            }
            for arm in summary
        },
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["by_arm"], indent=1))


if __name__ == "__main__":
    main()
