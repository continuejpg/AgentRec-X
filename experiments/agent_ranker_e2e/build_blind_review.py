"""Create a 40-pair blind review package from Phase 4.2/4.3 disagreements."""

from __future__ import annotations

import csv
import json
from pathlib import Path

B = Path("data/memory_ranking_benchmark_v2")
R = Path("runs/agent_native_ranking_phase42")
OUT = Path("runs/agent_ranker_phase43")
CONTROLLED = R / "shopranker_controlled_gpu.json"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controlled = json.loads(CONTROLLED.read_text(encoding="utf-8"))
    details = {row["scenario_id"]: row for row in controlled["benchmark_v2"]["details"]}
    pairs = {row["scenario_id"]: row for row in read_jsonl(B / "pair_tasks.jsonl")}
    scenarios = {row["scenario_id"]: row for row in read_jsonl(B / "scenarios.jsonl")}
    selected = []
    priority = {"negative_memory": 0, "preference_withdrawal": 1, "unknown_evidence": 2,
                "preference_update": 3, "query_memory_competition": 4, "positive_memory": 5}
    candidates = []
    for sid, pair in pairs.items():
        if pair["choice"] not in {"A", "B"}:
            continue
        row = details[sid]
        a, b = pair["candidate_a_id"], pair["candidate_b_id"]
        predictions = {}
        for arm in ("query", "natural", "structured", "hybrid"):
            score = row["scores"][arm]
            predictions[arm] = "A" if score[a] > score[b] else "B"
        if len(set(predictions.values())) < 2:
            continue
        candidates.append((priority.get(pair["scenario_type"], 99), sid, pair, predictions))
    candidates.sort(key=lambda item: (item[0], item[1]))
    for _, sid, pair, predictions in candidates[:40]:
        scenario = scenarios[sid]
        items = {item_id: item for item_id in (pair["candidate_a_id"], pair["candidate_b_id"])
                 for item in [pair["candidate_a"]] if item_id == pair["candidate_a_id"]}
        item_a = next((row for row in read_jsonl(B / "candidate_metadata.jsonl") if row["item_id"] == pair["candidate_a_id"]), {})
        item_b = next((row for row in read_jsonl(B / "candidate_metadata.jsonl") if row["item_id"] == pair["candidate_b_id"]), {})
        selected.append({
            "review_id": f"phase43_blind_{len(selected)+1:03d}",
            "pair_task_id": pair["pair_task_id"],
            "query": scenario["query"],
            "active_memory": scenario["memory_after"],
            "scenario_type": scenario["scenario_type"],
            "candidate_a": {"item_id": item_a.get("item_id"), "title": item_a.get("title"),
                            "attributes": item_a.get("attributes"), "evidence": item_a.get("evidence"),
                            "unknown_attributes": [a for a in ("over_ear", "on_ear", "in_ear", "wireless", "wired", "microphone", "rechargeable", "noise_cancelling", "waterproof", "gaming", "mechanical", "usb", "hdmi", "4k", "remote") if a not in set(item_a.get("attributes") or [])]},
            "candidate_b": {"item_id": item_b.get("item_id"), "title": item_b.get("title"),
                            "attributes": item_b.get("attributes"), "evidence": item_b.get("evidence"),
                            "unknown_attributes": [a for a in ("over_ear", "on_ear", "in_ear", "wireless", "wired", "microphone", "rechargeable", "noise_cancelling", "waterproof", "gaming", "mechanical", "usb", "hdmi", "4k", "remote") if a not in set(item_b.get("attributes") or [])]},
            "model_identity_hidden": True,
            "human_label": None,
            "human_label_provenance": "HUMAN_SINGLE_REVIEW_AVAILABLE_NOT_COLLECTED",
        })
    (OUT / "blind_review_tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    with (OUT / "blind_review_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["review_id", "choice", "confidence_1_5", "notes"])
        for row in selected:
            writer.writerow([row["review_id"], "", "", ""])
    (OUT / "blind_review_guidelines.md").write_text(
        "# Phase 4.3 Blind Review\n\n"
        "Each row shows the current query, active memory, and two real products. Choose A, B, BOTH, or INSUFFICIENT.\n\n"
        "- Model identities and scores are hidden.\n"
        "- Conflict scenarios must be INSUFFICIENT; do not force a winner.\n"
        "- UNKNOWN evidence is not absence.\n"
        "- This package has no human labels yet; HUMAN_GOLD=0. A single developer review would be HUMAN_SINGLE_REVIEW, not multi-review consensus.\n",
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(selected), "human_gold": 0, "human_single_review": 0}, indent=1))


if __name__ == "__main__":
    main()
