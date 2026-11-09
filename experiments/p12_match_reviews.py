"""Stream the official Amazon review archive and retain only P11 train-history matches.

No database is built.  The matcher requires user_id + parent_asin + normalized timestamp;
rating is only a diagnostic cross-check.  A key with multiple/conflicting review records is
reported as ambiguous and is never force-associated.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ROOT / "data/p12/review_keys.jsonl"
OUT = ROOT / "runs/p12/reviews"


def normalize_ms(value: object) -> int | None:
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return None
    # Amazon 2023 timestamps are milliseconds; accept seconds defensively.
    return raw if raw > 10_000_000_000 else raw * 1000


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--keys", type=Path, default=KEYS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    key_rows = [json.loads(line) for line in args.keys.read_text(encoding="utf-8").splitlines() if line.strip()]
    users = {str(row["user_id"]) for row in key_rows}
    exact: dict[tuple[str, str, int], dict] = {}
    for row in key_rows:
        exact[(str(row["user_id"]), str(row["parent_asin"]), int(row["timestamp_ms"]))] = row
    # ``exact`` above is built in a compact loop to keep the lookup only keyed by the
    # three fields; rating is deliberately not part of the key.
    counts = Counter()
    matches: list[dict] = []
    ambiguous: list[dict] = []
    seen: dict[tuple[str, str, int], list[dict]] = {}
    with gzip.open(args.reviews, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counts["raw_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["parse_errors"] += 1
                continue
            user_id = str(record.get("user_id") or "")
            if user_id not in users:
                continue
            counts["user_records"] += 1
            parent_asin = str(record.get("parent_asin") or record.get("asin") or "")
            timestamp = normalize_ms(record.get("timestamp"))
            if timestamp is None:
                counts["timestamp_invalid"] += 1
                continue
            key = (user_id, parent_asin, timestamp)
            expected = exact.get(key)
            if expected is None:
                counts["unmatched_user_asin_timestamp"] += 1
                continue
            counts["matched"] += 1
            matched = {
                "user_id": user_id,
                "parent_asin": parent_asin,
                "timestamp_ms": timestamp,
                "rating": record.get("rating"),
                "title": record.get("title"),
                "text": record.get("text"),
                "verified_purchase": record.get("verified_purchase"),
                "helpful_vote": record.get("helpful_vote"),
                "key_rating": expected.get("rating"),
                "source": "amazon_reviews_2023:raw_review",
            }
            seen.setdefault(key, []).append(matched)
    for key, rows in seen.items():
        if len(rows) > 1:
            ratings = {row.get("rating") for row in rows}
            texts = {row.get("text") for row in rows}
            counts["duplicate_keys"] += 1
            if len(ratings) > 1 or len(texts) > 1:
                counts["conflicting_duplicate_keys"] += 1
            ambiguous.append(
                {
                    "user_id": key[0],
                    "parent_asin": key[1],
                    "timestamp_ms": key[2],
                    "record_count": len(rows),
                    "rating_values": sorted(str(value) for value in ratings),
                    "text_variants": len(texts),
                }
            )
        matches.extend(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reviews_matched.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in matches),
        encoding="utf-8",
    )
    (args.out / "ambiguous.json").write_text(
        json.dumps(ambiguous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "agentrecx.p12.review_match.v1",
        "reviews_file": str(args.reviews),
        "reviews_sha256": sha256_file(args.reviews),
        "keys_file": str(args.keys),
        "keys_sha256": sha256_file(args.keys),
        "key_count": len(key_rows),
        "users": len(users),
        "counts": dict(counts),
        "matched_rows_written": len(matches),
        "ambiguous_keys": len(ambiguous),
        "match_rule": "exact user_id + parent_asin + timestamp_ms; rating only diagnostic",
        "target_filter": "keys are train_history only; no validation/test target",
    }
    (args.out / "match_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
