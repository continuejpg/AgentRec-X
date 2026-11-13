"""Extract catalog-intersecting Electronics rating rows from the remote raw splits."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

CATALOG = Path("/root/autodl-tmp/catalog_ids.txt")
FILES = [
    Path("/root/autodl-tmp/p13r/raw/Electronics.valid.csv"),
    Path("/root/autodl-tmp/p13r/raw/Electronics.test.csv"),
]
OUT = Path("/root/autodl-tmp/electronics_catalog_ratings.csv")
SUMMARY = Path("/root/autodl-tmp/electronics_catalog_ratings_summary.json")


def main() -> int:
    catalog = {line.strip() for line in CATALOG.read_text(encoding="utf-8").splitlines()
               if line.strip()}
    counts: Counter[str] = Counter()
    ratings: Counter[str] = Counter()
    items: set[str] = set()
    rows = 0
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "parent_asin", "rating", "timestamp", "source_split"])
        for path in FILES:
            with path.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    item = str(row["parent_asin"])
                    if item not in catalog:
                        continue
                    user = str(row["user_id"])
                    writer.writerow([user, item, row["rating"], row["timestamp"], path.stem])
                    counts[user] += 1
                    ratings[str(row["rating"])] += 1
                    items.add(item)
                    rows += 1
    summary = {
        "rows": rows,
        "users": len(counts),
        "items": len(items),
        "ratings": dict(sorted(ratings.items())),
        "users_ge_2": sum(1 for value in counts.values() if value >= 2),
        "users_ge_3": sum(1 for value in counts.values() if value >= 3),
        "users_ge_5": sum(1 for value in counts.values() if value >= 5),
        "users_ge_10": sum(1 for value in counts.values() if value >= 10),
        "max_interactions": max(counts.values()) if counts else 0,
        "source": "Electronics.valid.csv + Electronics.test.csv, catalog ids only",
        "target_not_in_history": "enforced by later temporal builder",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
