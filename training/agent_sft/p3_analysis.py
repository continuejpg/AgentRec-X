from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from training.agent_sft.bridge_metrics import load_runs

ARMS = {
    "C": Path("runs/p2bridge_c"),
    "D0": Path("runs/p2bridge_d0"),
    "D1": Path("runs/p2bridge_d1"),
    "D1v2": Path("runs/p2bridge_d1v2"),
}
FINAL = Path("P2_FINAL_AGENT_COMPARISON.csv")
OUT = Path("training/agent_sft/reports")


def v2_rows():
    rows = list(csv.DictReader(FINAL.open(encoding="utf-8")))
    return {(row["task_id"], row["policy"]): row for row in rows}


def task_events(arm: str):
    path = ARMS[arm] / "decision_state.sqlite3"
    result = {}
    if not path.exists():
        return result
    connection = sqlite3.connect(path)
    rows = connection.execute("select task_id, event_type, payload from events").fetchall()
    connection.close()
    for task_id, event_type, payload in rows:
        task = str(task_id).split(":")[2] if str(task_id).count(":") >= 2 else str(task_id)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {}
        result.setdefault(task, []).append((event_type, data))
    return result


def main() -> int:
    v2 = v2_rows()
    runs = {arm: {run.task_id: run for run in load_runs(path)} for arm, path in ARMS.items()}
    events = {arm: task_events(arm) for arm in ARMS}
    task_ids = sorted(runs["C"])
    rows = []
    for task_id in task_ids:
        c_ids = set(runs["C"][task_id].final_ids)
        d0_ids = set(runs["D0"][task_id].final_ids)
        d1_ids = set(runs["D1"][task_id].final_ids)
        d1v2_ids = set(runs["D1v2"][task_id].final_ids)
        d1_props = [op for run in [runs["D1"][task_id]] for turn in run.turns for op in (turn.llm_proposals or ())]
        d1_memory = [e for turn in runs["D1"][task_id].turns for e in turn.memory_events if isinstance(e, dict)]
        accepted = sum(len((e.get("proposal_bridge") or {}).get("accepted_operations") or ()) for e in d1_memory)
        valid_turns = sum(1 for e in d1_memory if (e.get("proposal_bridge") or {}).get("valid"))
        proposal_events = 0
        for event_type, payload in events["D1"].get(task_id, []):
            source = str(payload.get("source_text", ""))
            if source and source in {str(op.get("evidence")) for op in d1_props}:
                proposal_events += 1
        rows.append({
            "task_id": task_id,
            "v2_c": v2[(task_id, "C")]["v2_task_success"],
            "v2_d0": v2[(task_id, "D0")]["v2_task_success"],
            "v2_d1": v2[(task_id, "D1")]["v2_task_success"],
            "v2_d1v2": v2[(task_id, "D1v2")]["v2_task_success"],
            "c_final_n": len(c_ids), "d0_final_n": len(d0_ids), "d1_final_n": len(d1_ids), "d1v2_final_n": len(d1v2_ids),
            "c_d1_jaccard": round(len(c_ids & d1_ids) / max(1, len(c_ids | d1_ids)), 3),
            "c_d0_jaccard": round(len(c_ids & d0_ids) / max(1, len(c_ids | d0_ids)), 3),
            "d1_d1v2_jaccard": round(len(d1_ids & d1v2_ids) / max(1, len(d1_ids | d1v2_ids)), 3),
            "d1_valid_turns": valid_turns,
            "d1_accepted_ops": accepted,
            "d1_proposal_m5_events": proposal_events,
            "d1_retrieve": sum(1 for turn in runs["D1"][task_id].turns for a in turn.retrieval_actions if a == "RETRIEVE"),
            "d1_rerank": sum(1 for turn in runs["D1"][task_id].turns for a in turn.retrieval_actions if a == "RERANK"),
        })
    with Path("training/agent_sft/eval/P3_TASK_CAUSAL_TRACE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # aggregate
    summary = {}
    for arm in ARMS:
        passed = {tid for tid in task_ids if v2[(tid, arm)]["v2_task_success"] == "PASS"}
        summary[arm] = {
            "pass": len(passed),
            "pass_set": sorted(passed),
            "pass_set_equal_C": passed == {tid for tid in task_ids if v2[(tid, "C")]["v2_task_success"] == "PASS"},
        }
    summary["pairwise"] = {}
    for a, b in (("C", "D0"), ("C", "D1"), ("D0", "D1"), ("D1", "D1v2")):
        changed_ids = sum(1 for r in rows if r[f"{a.lower()}_final_n"] >= 0 and (set(runs[a][r["task_id"]].final_ids) != set(runs[b][r["task_id"]].final_ids)))
        same_v2 = sum(1 for r in rows if r[f"v2_{a.lower()}"] == r[f"v2_{b.lower()}"])
        summary["pairwise"][f"{a}_vs_{b}"] = {"final_id_changed_tasks": changed_ids, "same_v2_tasks": same_v2}
    summary["d1"] = {
        "valid_turns": sum(r["d1_valid_turns"] for r in rows),
        "accepted_ops": sum(r["d1_accepted_ops"] for r in rows),
        "proposal_m5_events": sum(r["d1_proposal_m5_events"] for r in rows),
        "all_m5_events": sum(len(events["D1"].get(tid, ())) for tid in task_ids),
        "tasks_with_proposal_events": sum(1 for r in rows if r["d1_proposal_m5_events"] > 0),
    }
    summary["d0"] = {
        "proposal_turns": sum(1 for r in rows for turn in runs["D0"][r["task_id"]].turns for e in turn.memory_events if isinstance(e, dict) and isinstance(e.get("proposal_bridge"), dict)),
        "invalid_proposals": sum(1 for r in rows for turn in runs["D0"][r["task_id"]].turns for e in turn.memory_events if isinstance(e, dict) and isinstance(e.get("proposal_bridge"), dict) and not e["proposal_bridge"].get("valid")),
    }
    (OUT / "P3_BOTTLENECK_METRICS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
