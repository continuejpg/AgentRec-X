"""P13 target-review timestamp extraction (independent of user history).

Matches C4 target user+item to raw Amazon reviews by exact review text (normalized
whitespace/case) and rating.  The target review is used only for timing/label diagnosis,
never as prior history evidence.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ROOT / "data/p13/target_keys.jsonl"
OUT = ROOT / "runs/p13/reviews"
WS = re.compile(r"\s+")


def norm(text: object) -> str:
    return WS.sub(" ", str(text or "")).strip().casefold()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=KEYS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    keys = [json.loads(line) for line in args.keys.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key: dict[tuple[str, str], list[dict]] = {}
    users = {str(row["user_id"]) for row in keys}
    with gzip.open(args.reviews, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = str(record.get("user_id") or "")
            if user not in users:
                continue
            item = str(record.get("parent_asin") or record.get("asin") or "")
            by_key.setdefault((user, item), []).append(record)
    results = []
    failures = []
    for row in keys:
        candidates = by_key.get((str(row["user_id"]), str(row["parent_asin"])), [])
        wanted = norm(row.get("ori_review"))
        exact = [
            review for review in candidates
            if wanted in {
                norm(review.get("text")),
                norm(f"{review.get('title') or ''} {review.get('text') or ''}"),
            }
            and float(review.get("rating")) == float(row.get("ori_rating"))
        ]
        if len(exact) == 1:
            review = exact[0]
            results.append(
                {
                    **row,
                    "target_timestamp_ms": int(review["timestamp"]),
                    "target_rating": review.get("rating"),
                    "match_method": "exact normalized text + rating",
                    "candidate_reviews_for_user_item": len(candidates),
                }
            )
        else:
            failures.append(
                {
                    "qid": row.get("qid"),
                    "split": row.get("split"),
                    "user_id": row.get("user_id"),
                    "parent_asin": row.get("parent_asin"),
                    "candidate_reviews_for_user_item": len(candidates),
                    "exact_matches": len(exact),
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "target_timestamps.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    (args.out / "target_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "keys": len(keys),
        "matched": len(results),
        "failed": len(failures),
        "users": len(users),
        "method": "user_id + parent_asin + normalized target review text + rating; timestamp retained",
        "scope": "target timestamp only; target review is not prior-history evidence",
    }
    (args.out / "target_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
