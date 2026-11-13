"""Combine frozen Qwen3-Reranker scores with the existing hard-filtered scorer.

This script is the local evaluator for the GPU handoff.  It is intentionally not run
unless a real `qwen3_scores.jsonl` exists.  It never treats the existing scorer as a
training label and never trains a model.

The conservative combination is a fixed, untuned RRF over two orderings of the same
fixed pool:

1. existing structural hard filter + scorer order (C);
2. Qwen3 order restricted to the same hard/role-filtered candidates.

No candidate is added; no Gate semantics are changed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

RRF_K = 60
ROLE_OK = {"MATCH", "NOT_REQUIRED"}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _acceptable(row: dict[str, Any]) -> set[str]:
    return set(row.get("acceptable_ids_in_pool") or [])


def _hit(ranking: list[str], acceptable: set[str]) -> dict[str, Any]:
    first = next((index + 1 for index, item_id in enumerate(ranking)
                  if item_id in acceptable), None)
    return {
        "hit@1": bool(ranking and ranking[0] in acceptable),
        "hit@5": bool(any(item_id in acceptable for item_id in ranking[:5])),
        "hit@10": bool(any(item_id in acceptable for item_id in ranking[:10])),
        "first_acceptable_rank": first,
        "reciprocal_rank": (1.0 / first) if first else 0.0,
    }


def _eligible(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for candidate in row["candidate_pool"]:
        scorer = candidate.get("scorer") or {}
        if int(scorer.get("hard_tier") or 0) != 0:
            continue
        if str(scorer.get("role_status") or "NOT_REQUIRED") not in ROLE_OK:
            continue
        result.append(candidate)
    return result


def _rrf(rankings: list[list[str]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (RRF_K + rank)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


def _summary(rows: list[dict[str, Any]], key: str, rankings: dict[str, list[str]]) -> dict[str, Any]:
    hits = [_hit(rankings[row["task_id"]], _acceptable(row)) for row in rows]
    return {
        "tasks": len(rows),
        "hit@1": sum(1 for item in hits if item["hit@1"]),
        "hit@5": sum(1 for item in hits if item["hit@5"]),
        "hit@10": sum(1 for item in hits if item["hit@10"]),
        "mrr": round(statistics.mean(item["reciprocal_rank"] for item in hits), 6),
        "cases": {row["task_id"]: _hit(rankings[row["task_id"]], _acceptable(row))
                  for row in rows},
        "key": key,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--qwen3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = [row for row in _rows(args.data) if row.get("expected_decision") == "recommend"]
    qwen_rows = {row["task_id"]: row for row in _rows(args.qwen3)}
    if set(qwen_rows) != {row["task_id"] for row in dataset}:
        raise SystemExit("Qwen3 output task ids do not exactly match recommend tasks")

    result: dict[str, Any] = {
        "schema": "agentrecx.preference_ranking.qwen3_comparison.v1",
        "rrf_k": RRF_K,
        "variants": {},
        "rankings": {},
    }
    c_rankings: dict[str, list[str]] = {}
    q_rankings: dict[str, list[str]] = {}
    rrf_rankings: dict[str, list[str]] = {}
    for row in dataset:
        task_id = row["task_id"]
        eligible = _eligible(row)
        eligible_ids = {item["item_id"] for item in eligible}
        c_ids = [item_id for item_id in row["rankings"]["C_hard_filtered_scorer"]
                 if item_id in eligible_ids]
        qwen_info = qwen_rows[task_id]
        qwen_order = [item["item_id"] for item in qwen_info["ranked"]]
        q_ids = [item_id for item_id in qwen_order if item_id in eligible_ids]
        # Preserve every eligible candidate if the model output is incomplete.
        q_ids += [item_id for item_id in c_ids if item_id not in q_ids]
        c_rankings[task_id] = c_ids
        q_rankings[task_id] = q_ids
        rrf_rankings[task_id] = _rrf([c_ids, q_ids])
        result["rankings"][task_id] = {
            "eligible_ids": list(eligible_ids),
            "C": c_ids,
            "qwen3": q_ids,
            "rrf": rrf_rankings[task_id],
        }
    result["variants"]["C_hard_filtered_scorer"] = _summary(dataset, "C", c_rankings)
    result["variants"]["qwen3_only_hard_filtered"] = _summary(dataset, "qwen3", q_rankings)
    result["variants"]["conservative_rrf"] = _summary(dataset, "rrf", rrf_rankings)
    result["note"] = "qwen3_only_hard_filtered and conservative_rrf are not used unless a real GPU output was supplied"
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
