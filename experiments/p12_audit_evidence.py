"""P12 audit of recovered review text against P11 Level-B history claims.

The output is deliberately `level_a_candidate_unreviewed`: rule-based explicit
positive/negative attribute mentions are candidates for human review, never confirmed
Level A.  The original P11 task file is preserved; an enriched copy is written separately.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data/p11_benchmark/tasks.jsonl"
MATCHED = ROOT / "runs/p12/reviews/reviews_matched.jsonl"
MATCH_REPORT = ROOT / "runs/p12/reviews/match_report.json"
OUT = ROOT / "runs/p12/evidence"
ENRICHED = ROOT / "data/p11_benchmark/tasks_p12_evidence.jsonl"

ATTRIBUTE_TERMS = {
    "filter", "filtration", "water", "camping", "cookware", "cook", "stove", "grill",
    "sleeping", "pad", "mat", "bottle", "hydration", "hiking", "lightweight", "portable",
    "compact", "durable", "insulated", "leak", "leaks", "easy", "clean", "battery",
    "warm", "comfortable", "sturdy", "titanium", "steel", "capacity", "size", "weight",
}
POSITIVE = {
    "love", "loved", "great", "perfect", "excellent", "amazing", "good", "works",
    "best", "happy", "recommend", "nice", "solid", "easy", "comfortable", "lightweight",
    "durable", "fantastic", "awesome", "ideal", "favorite",
}
NEGATIVE = {
    "hate", "bad", "poor", "broke", "broken", "leak", "leaked", "returned",
    "disappointed", "cheap", "flimsy", "difficult", "never", "avoid", "waste",
    "useless", "unusable", "wrong",
}
SPLIT = re.compile(r"[^a-z0-9]+")


def _snippet(text: str, limit: int = 220) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:limit]


def _classify(text: str) -> str:
    tokens = [token for token in SPLIT.split(text.casefold()) if token]
    if not tokens:
        return "empty_text"
    has_attribute = any(token in ATTRIBUTE_TERMS for token in tokens)
    has_positive = any(token in POSITIVE for token in tokens)
    has_negative = any(token in NEGATIVE for token in tokens)
    if has_attribute and has_positive and not has_negative:
        return "explicit_positive_attribute_candidate"
    if has_attribute and has_negative and not has_positive:
        return "explicit_negative_attribute_candidate"
    if has_positive or has_negative:
        return "product_sentiment_only"
    if has_attribute:
        return "attribute_mention_only"
    return "unrelated_or_unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--matched", type=Path, default=MATCHED)
    parser.add_argument("--match-report", type=Path, default=MATCH_REPORT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--enriched", type=Path, default=ENRICHED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    match_report = json.loads(args.match_report.read_text(encoding="utf-8"))
    reviews_by_key: dict[tuple[str, str, int], list[dict]] = {}
    for line in args.matched.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["user_id"]), str(row["parent_asin"]), int(row["timestamp_ms"]))
        reviews_by_key.setdefault(key, []).append(row)
    review_lookup = dict(reviews_by_key)

    enriched_tasks = []
    claims = []
    for task in tasks:
        refs = []
        classifications = Counter()
        for ref in task.get("history_evidence_refs") or ():
            timestamp = ref.get("timestamp_ms")
            if timestamp is None:
                continue
            key = (str(task["user_id"]), str(ref["parent_asin"]), int(timestamp))
            for review in review_lookup.get(key, ()):
                text = str(review.get("text") or "")
                classification = _classify(text)
                classifications[classification] += 1
                refs.append(
                    {
                        "parent_asin": review.get("parent_asin"),
                        "timestamp_ms": review.get("timestamp_ms"),
                        "rating": review.get("rating"),
                        "review_title": review.get("title"),
                        "review_text": text,
                        "snippet": _snippet(text),
                        "classification": classification,
                        "match_rule": "user_id + parent_asin + timestamp_ms",
                        "review_source": "amazon_reviews_2023:raw_review",
                    }
                )
        candidate = any(
            ref["classification"] == "explicit_positive_attribute_candidate" for ref in refs
        )
        had_reviews = bool(refs)
        status = (
            "level_a_candidate_unreviewed"
            if candidate
            else "insufficient_review_evidence"
            if had_reviews
            else "no_review_text_matched"
        )
        enriched = dict(task)
        enriched["p11_original_preference_evidence_level"] = task["preference_evidence_level"]
        enriched["review_evidence"] = refs
        enriched["review_evidence_summary"] = {
            "matched_review_count": len(refs),
            "classification_counts": dict(classifications),
            "level_a_status": status,
            "confirmed_level_a": False,
            "human_review_required": candidate,
            "review_text_source": "official Amazon Reviews 2023 Sports_and_Outdoors",
        }
        enriched["evidence_version"] = "p11.tasks_p12_evidence.v1"
        enriched["provenance"] = dict(task.get("provenance") or {})
        enriched["provenance"]["review_match_report"] = "runs/p12/reviews/match_report.json"
        enriched["provenance"]["review_match_sha256"] = match_report.get("reviews_sha256")
        enriched_tasks.append(enriched)
        if task["preference_evidence_level"] == "B":
            claims.append(
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "user_id": task["user_id"],
                    "matched_review_count": len(refs),
                    "explicit_candidate_count": classifications.get(
                        "explicit_positive_attribute_candidate", 0
                    ),
                    "negative_candidate_count": classifications.get(
                        "explicit_negative_attribute_candidate", 0
                    ),
                    "level_a_status": status,
                    "human_review_required": candidate,
                    "task_original_evidence_level": "B",
                }
            )
    level_b_claims = claims
    summary = {
        "schema_version": "agentrecx.p12.evidence_audit.v1",
        "tasks": len(tasks),
        "level_b_claims": len(level_b_claims),
        "claims_with_matched_reviews": sum(1 for row in level_b_claims if row["matched_review_count"] > 0),
        "claims_with_explicit_positive_candidate": sum(
            1 for row in level_b_claims if row["explicit_candidate_count"] > 0
        ),
        "claims_with_negative_candidate": sum(
            1 for row in level_b_claims if row["negative_candidate_count"] > 0
        ),
        "claims_without_review_text": sum(
            1 for row in level_b_claims if row["level_a_status"] == "no_review_text_matched"
        ),
        "confirmed_level_a": 0,
        "confirmed_level_a_reason": "no human review of recovered snippets in P12",
        "human_review_required": sum(1 for row in level_b_claims if row["human_review_required"]),
        "review_match_report": match_report,
        "original_tasks_preserved": str(args.tasks.relative_to(ROOT)),
        "enriched_tasks": str(args.enriched.relative_to(ROOT)),
        "note": "explicit_positive_attribute_candidate is a heuristic candidate only; it is not a confirmed user preference",
    }
    (args.out / "claims.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in claims), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.enriched.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in enriched_tasks),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
