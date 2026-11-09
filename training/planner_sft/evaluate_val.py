"""Evaluate the trained Planner adapter on P4A validation rows via the /plan HTTP API."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from recommendation.demo.planner_contract import PlannerObservation
from recommendation.demo.planner_http_proposer import HttpPlannerProposer


def _observation(row: dict[str, Any]) -> PlannerObservation:
    o = row["observation"]
    return PlannerObservation(
        step=o["step"],
        query=o["query"],
        candidate_ids=tuple(o["candidate_ids"]),
        state_version=o["state_version"],
        needs_user_input=o["needs_user_input"],
        reranked=o["reranked"],
        m5_state=o["m5_state"],
        allowed_actions=tuple(o["allowed_actions"]),
        last_tool_observation=o["last_tool_observation"],
        remaining_steps=o["remaining_steps"],
        candidate_evidence=tuple(o["candidate_evidence"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18008")
    parser.add_argument("--data", type=Path, default=Path("data/planner_sft_v0/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("P4B_VAL_EVAL.json"))
    parser.add_argument("--pred-out", type=Path, default=Path("P4B_VAL_PREDICTIONS.jsonl"))
    args = parser.parse_args()
    proposer = HttpPlannerProposer(args.url, timeout=180.0)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = []
    action_hits = 0
    valid = 0
    per_action: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        target = row["target"]["action"]
        try:
            proposal = proposer.propose(_observation(row))
            predicted = proposal.action.value
            schema_valid = True
            error = None
        except Exception as exc:
            predicted = None
            schema_valid = False
            error = f"{type(exc).__name__}: {exc}"
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "target_action": target,
                "predicted_action": predicted,
                "schema_valid": schema_valid,
                "action_exact": predicted == target,
                "raw": proposer.last_response,
                "error": error,
            }
        )
        valid += int(schema_valid)
        action_hits += int(predicted == target)
        per_action[target][predicted or "INVALID"] += 1
    args.pred_out.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions), encoding="utf-8")
    summary = {
        "samples": len(predictions),
        "action_exact": action_hits,
        "action_accuracy": action_hits / max(1, len(predictions)),
        "schema_valid": valid,
        "schema_validity": valid / max(1, len(predictions)),
        "per_target_action": {key: dict(value) for key, value in per_action.items()},
        "errors": [item for item in predictions if item["error"]],
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
