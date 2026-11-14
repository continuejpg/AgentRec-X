"""Build P100 memory scenarios (event-sourced synthetic sequences) + Agent task manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p100_extended_benchmark"
TASKS = DATA / "p100_agent_tasks.jsonl"
POOLS = DATA / "p100_bge_pools.jsonl"
META = DATA / "p100_candidate_metadata.jsonl"
STATES = DATA / "p100_memory_states.jsonl"
OFF = DATA / "p100_agent_tasks_memory_off.jsonl"
ON = DATA / "p100_agent_tasks_memory_on.jsonl"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")
QUERY_HINTS = {
    "over_ear": ("over-ear", "over ear", "overear"), "on_ear": ("on-ear", "on ear", "onear"),
    "in_ear": ("in-ear", "in ear", "earbud", "inear"), "wireless": ("wireless", "bluetooth"),
    "wired": ("wired", "corded"), "microphone": ("microphone", "mic"),
    "rechargeable": ("rechargeable", "battery"), "noise_cancelling": ("noise cancel", "noise-cancel"),
    "waterproof": ("waterproof", "water resistant"), "portable": ("portable", "travel"),
    "gaming": ("gaming", "gamer"), "mechanical": ("mechanical",), "usb": ("usb",),
    "hdmi": ("hdmi",), "4k": ("4k", "uhd"), "remote": ("remote",),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    tasks = read_jsonl(TASKS)
    pools = {row["base_query_id"]: row for row in read_jsonl(POOLS)}
    meta = {row["item_id"]: row for row in read_jsonl(META)}
    old_states = {row["base_query_id"]: row for row in read_jsonl(STATES)}
    on_rows = []
    off_rows = []
    state_rows = []
    counts = Counter()
    for task in tasks:
        bid = task["base_query_id"]
        pool = pools[bid]
        ids = pool["candidate_pool_ids"]
        attrs = {}
        for item_id in ids:
            for attr in (meta.get(item_id) or {}).get("attributes") or []:
                attrs.setdefault(attr, []).append(item_id)
        counts_by_attr = {attr: len(items) for attr, items in attrs.items()}
        text = task["query"].lower()
        mentions = {attr for attr, hints in QUERY_HINTS.items() if any(h in text for h in hints)}
        available = [attr for attr in ATTRS if counts_by_attr.get(attr, 0) >= 2]
        preferred = [attr for attr in sorted(mentions, key=lambda a: -counts_by_attr.get(a, 0))
                     if counts_by_attr.get(attr, 0) >= 2]
        attr = (preferred or available or [None])[0]
        # Choose a safe second attribute, preferably non-exclusive with the first.
        exclusive = {"over_ear": {"in_ear", "on_ear"}, "in_ear": {"over_ear", "on_ear"},
                     "on_ear": {"over_ear", "in_ear"}, "wireless": {"wired"}, "wired": {"wireless"}}
        alternatives = [a for a in ATTRS if counts_by_attr.get(a, 0) >= 2 and a != attr
                        and a not in exclusive.get(attr, set())]
        second = alternatives[0] if alternatives else attr
        scenario = (old_states.get(bid) or {}).get("scenario_type") or "no_memory"
        if not attr or scenario in {"preference_conflict"} and not attr:
            scenario = "no_memory"
        if scenario == "no_memory":
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": f"I'm shopping for {task['query']}."}]}]
        elif scenario == "positive_memory":
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": f"I prefer {attr.replace('_', '-')}."}]}]
        elif scenario == "negative_memory":
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": f"I do not want {attr.replace('_', '-')}."}]}]
        elif scenario == "multiple_preferences":
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": f"I prefer {attr.replace('_', '-')} and {second.replace('_', '-')}."}]}]
        elif scenario == "preference_update":
            sessions = [
                {"session_id": f"{bid}_s1", "date": "2025-09-20",
                 "turns": [{"role": "user", "content": f"I prefer {attr.replace('_', '-')}."}]},
                {"session_id": f"{bid}_s2", "date": "2025-10-01",
                 "turns": [{"role": "user", "content": f"Forget my {attr.replace('_', '-')} preference. I prefer {second.replace('_', '-')}."}]},
            ]
        elif scenario == "preference_withdrawal":
            sessions = [
                {"session_id": f"{bid}_s1", "date": "2025-09-20",
                 "turns": [{"role": "user", "content": f"I prefer {attr.replace('_', '-')}."}]},
                {"session_id": f"{bid}_s2", "date": "2025-10-01",
                 "turns": [{"role": "user", "content": f"Forget my {attr.replace('_', '-')} preference."}]},
            ]
        elif scenario == "unknown_evidence":
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": "I prefer a verified unobtainium feature."}]}]
        elif scenario == "preference_conflict":
            # Only create a conflict when both exclusive form factors exist in the pool.
            if "over_ear" in attrs and "in_ear" in attrs:
                sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                             "turns": [{"role": "user", "content": "I prefer over-ear. I prefer in-ear."}]}]
            else:
                scenario = "no_memory"
                sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                             "turns": [{"role": "user", "content": f"I'm shopping for {task['query']}."}]}]
        elif scenario == "query_memory_competition":
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": f"I prefer {second.replace('_', '-')}."}]}]
        else:
            scenario = "no_memory"
            sessions = [{"session_id": f"{bid}_s1", "date": "2025-10-01",
                         "turns": [{"role": "user", "content": f"I'm shopping for {task['query']}."}]}]
        state = (old_states.get(bid) or {}).copy()
        state.update({"scenario_type": scenario, "memory_snapshot_id": f"{bid}:{scenario}",
                      "source": "synthetic_event_sequence", "history_sessions": sessions,
                      "selected_attribute": attr, "secondary_attribute": second,
                      "attribute_evidence_counts": counts_by_attr})
        state_rows.append(state)
        counts[scenario] += 1
        common = {
            "task_id": f"p100_{bid.split('_')[1]}", "base_query_id": bid, "query_id": task["query_id"],
            "current_query": task["query"], "query_family": task["query_family"],
            "user_id": f"p100_user_{bid}", "current_time": "2025-10-19T12:00:00",
            "scenario_type": scenario, "memory_snapshot_id": state["memory_snapshot_id"],
            "history_source": "synthetic_event_sequence", "hard": pool.get("hard_requirements") or [],
            "candidate_pool_ids": ids, "candidate_pool_hash": pool["candidate_pool_hash"],
            "expected": "recommend",
        }
        on = dict(common); on["history_sessions"] = sessions; on["memory_mode"] = "ON"; on_rows.append(on)
        off = dict(common); off["history_sessions"] = [
            {"session_id": f"{bid}_s0", "date": "2025-10-01",
             "turns": [{"role": "user", "content": f"I'm shopping for {task['query']}."}]}]
        off["memory_mode"] = "OFF"; off_rows.append(off)
    (STATES).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in state_rows) + "\n", encoding="utf-8")
    (ON).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in on_rows) + "\n", encoding="utf-8")
    (OFF).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in off_rows) + "\n", encoding="utf-8")
    print(json.dumps({"scenarios": dict(counts), "on": len(on_rows), "off": len(off_rows)}, indent=1))


if __name__ == "__main__":
    main()
