"""Import blind-review CSV and compute frozen-model pairwise agreement when labels exist."""

from __future__ import annotations

import csv
import json
from pathlib import Path

R = Path("runs/agent_ranker_phase43")
R45 = Path("runs/agent_ranker_phase45")
TASKS = R / "blind_review_tasks.jsonl"
CONTROLLED = Path("runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json")


def main():
    R45.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["review_id"]: row for row in tasks}
    template = R / "blind_review_template.csv"
    csv_path = R45 / "blind_review_results.csv"
    if not csv_path.exists():
        csv_path = template
    labels = {}
    invalid = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                review_id = str(row.get("review_id") or "").strip()
                if review_id not in by_id:
                    if review_id:
                        invalid.append({"row": review_id, "reason": "unknown_review_id"})
                    continue
                choice = str(row.get("choice") or "").strip().upper()
                if not choice:
                    continue
                if choice not in {"A", "B", "BOTH", "INSUFFICIENT"}:
                    invalid.append({"row": review_id, "reason": "invalid_choice", "choice": choice})
                    continue
                labels[review_id] = {"choice": choice, "notes": str(row.get("notes") or ""), "reviewer_id": str(row.get("reviewer_id") or "single_reviewer")}
    controlled = json.loads(CONTROLLED.read_text(encoding="utf-8"))
    details = {row["scenario_id"]: row for row in controlled["benchmark_v2"]["details"]}
    existing_path = Path("runs/memory_ranking_benchmark_v2/existing_scorer_shadow.json")
    existing_details = {}
    if existing_path.exists():
        existing_details = {row["scenario_id"]: row for row in json.loads(existing_path.read_text(encoding="utf-8"))["per_scenario"]}
    agreement = {}
    for arm in ("existing_scorer", "natural", "hybrid"):
        correct = judged = 0
        for review_id, label in labels.items():
            if label["choice"] not in {"A", "B"}:
                continue
            task = by_id[review_id]
            scenario_id = task["pair_task_id"].rsplit(":pair", 1)[0]
            a = task["candidate_a"]["item_id"]
            b = task["candidate_b"]["item_id"]
            if arm == "existing_scorer":
                row = existing_details.get(scenario_id)
                if not row:
                    continue
                scores = row.get("scores") or {}
            else:
                row = details.get(scenario_id)
                if not row:
                    continue
                scores = row.get("scores", {}).get(arm) or {}
            if a not in scores or b not in scores:
                continue
            predicted = "A" if scores[a] > scores[b] else "B"
            judged += 1
            correct += int(predicted == label["choice"])
        agreement[arm] = {"correct": correct, "judged": judged, "accuracy": round(correct / judged, 6) if judged else None}
    both = sum(1 for label in labels.values() if label["choice"] == "BOTH")
    insufficient = sum(1 for label in labels.values() if label["choice"] == "INSUFFICIENT")
    outcome = {
        "schema": "agentrecx.agent_ranker_phase45.human_preference_eval.v1",
        "source_csv": str(csv_path),
        "tasks_total": len(tasks),
        "human_review_count": len(labels),
        "review_provenance": "HUMAN_SINGLE_REVIEW" if labels else "UNJUDGED",
        "label_distribution": {
            "A": sum(1 for label in labels.values() if label["choice"] == "A"),
            "B": sum(1 for label in labels.values() if label["choice"] == "B"),
            "BOTH": both,
            "INSUFFICIENT": insufficient,
        },
        "model_agreement": agreement if labels else "unavailable_no_human_labels",
        "invalid_rows": invalid,
        "evaluation_scope": "Disagreement-Set Human Preference Evaluation; not random-user satisfaction",
        "notes": [
            "No labels were auto-generated; empty CSV rows remain UNJUDGED.",
            "BOTH and INSUFFICIENT are excluded from A/B agreement denominators but reported separately.",
            "Model outputs are frozen Phase 4.2/4.3 predictions; no tuning on these labels.",
        ],
    }
    (R45 / "human_preference_eval.json").write_text(json.dumps(outcome, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (R45 / "HUMAN_REVIEW_INSTRUCTIONS.md").write_text(
        "# Phase 4.5 Human Preference Review\n\n"
        "Open `runs/agent_ranker_phase44/blind_review_tool.html` in a browser.\n\n"
        "1. Review the query, active memory, and product A/B evidence.\n"
        "2. Choose A, B, BOTH, or INSUFFICIENT.\n"
        "3. Optionally add notes.\n"
        "4. Export CSV and place it at `runs/agent_ranker_phase45/blind_review_results.csv`.\n"
        "5. Run `python -m experiments.agent_ranker_e2e.import_blind_review`.\n\n"
        "Do not fill labels by rule or LLM. A single developer's labels are `HUMAN_SINGLE_REVIEW`, not multi-review consensus. "
        "This is a disagreement-set evaluation, not a random user-satisfaction estimate.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: outcome[key] for key in ("tasks_total", "human_review_count", "review_provenance", "label_distribution", "model_agreement")}, indent=1))


if __name__ == "__main__":
    main()
