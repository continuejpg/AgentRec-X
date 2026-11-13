"""Read-only Amazon Reviews 2023 Electronics rating split inventory.

The files are real ratings CSVs with columns user_id,parent_asin,rating,timestamp.
They are NOT human pairwise preference labels and do not contain a review text column.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

FILES = [
    Path("/root/autodl-tmp/p13r/raw/Electronics.valid.csv"),
    Path("/root/autodl-tmp/p13r/raw/Electronics.test.csv"),
]
CATALOG = Path("/root/autodl-tmp/catalog_ids.txt")
OUT = Path("/root/autodl-tmp/review_inventory.json")


def main() -> int:
    catalog = {line.strip() for line in CATALOG.read_text(encoding="utf-8").splitlines()
               if line.strip()}
    users: Counter[str] = Counter()
    items: set[str] = set()
    ratings: Counter[float] = Counter()
    per_file: dict[str, dict] = {}
    catalog_rows = 0
    catalog_users: set[str] = set()
    catalog_items: set[str] = set()
    timestamp_min = None
    timestamp_max = None
    file_users: dict[str, set[str]] = {}
    file_items: dict[str, set[str]] = {}
    for path in FILES:
        f_users: set[str] = set()
        f_items: set[str] = set()
        rows = 0
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                user = str(row["user_id"])
                item = str(row["parent_asin"])
                f_users.add(user)
                f_items.add(item)
                users[user] += 1
                items.add(item)
                ratings[float(row["rating"])] += 1
                ts = int(row["timestamp"])
                timestamp_min = ts if timestamp_min is None else min(timestamp_min, ts)
                timestamp_max = ts if timestamp_max is None else max(timestamp_max, ts)
                if item in catalog:
                    catalog_rows += 1
                    catalog_users.add(user)
                    catalog_items.add(item)
        per_file[path.name] = {"rows": rows, "users": len(f_users), "items": len(f_items)}
        file_users[path.name] = f_users
        file_items[path.name] = f_items
    counts = list(users.values())
    for field, value in (
        ("rows", sum(item["rows"] for item in per_file.values())),
        ("unique_users", len(users)),
        ("unique_items", len(items)),
        ("rating_counts", {str(k): v for k, v in sorted(ratings.items())}),
        ("users_ge_2", sum(1 for x in counts if x >= 2)),
        ("users_ge_3", sum(1 for x in counts if x >= 3)),
        ("users_ge_5", sum(1 for x in counts if x >= 5)),
        ("users_ge_10", sum(1 for x in counts if x >= 10)),
        ("users_ge_20", sum(1 for x in counts if x >= 20)),
        ("max_interactions_per_user", max(counts) if counts else 0),
        ("mean_interactions_per_user", round(sum(counts) / max(1, len(counts)), 4)),
        ("timestamp_min_ms", timestamp_min),
        ("timestamp_max_ms", timestamp_max),
        ("catalog_intersection_items", len(catalog_items)),
        ("catalog_intersection_users", len(catalog_users)),
        ("catalog_intersection_rows", catalog_rows),
        ("catalog_coverage_items", round(len(catalog_items) / max(1, len(catalog)), 6)),
        ("catalog_rows_coverage", round(catalog_rows / max(1, sum(v["rows"] for v in per_file.values())), 6)),
        ("per_file", per_file),
        ("user_overlap_valid_test", len(file_users.get("Electronics.valid.csv", set()) & file_users.get("Electronics.test.csv", set()))),
        ("item_overlap_valid_test", len(file_items.get("Electronics.valid.csv", set()) & file_items.get("Electronics.test.csv", set()))),
    ):
        print(f"{field}: {value}")
    OUT.write_text(json.dumps({
        "rows": sum(item["rows"] for item in per_file.values()),
        "unique_users": len(users),
        "unique_items": len(items),
        "rating_counts": {str(k): v for k, v in sorted(ratings.items())},
        "users_ge_2": sum(1 for x in counts if x >= 2),
        "users_ge_3": sum(1 for x in counts if x >= 3),
        "users_ge_5": sum(1 for x in counts if x >= 5),
        "users_ge_10": sum(1 for x in counts if x >= 10),
        "users_ge_20": sum(1 for x in counts if x >= 20),
        "max_interactions_per_user": max(counts) if counts else 0,
        "mean_interactions_per_user": round(sum(counts) / max(1, len(counts)), 4),
        "timestamp_min_ms": timestamp_min,
        "timestamp_max_ms": timestamp_max,
        "catalog_intersection_items": len(catalog_items),
        "catalog_intersection_users": len(catalog_users),
        "catalog_intersection_rows": catalog_rows,
        "catalog_coverage_items": round(len(catalog_items) / max(1, len(catalog)), 6),
        "catalog_rows_coverage": round(catalog_rows / max(1, sum(v["rows"] for v in per_file.values())), 6),
        "per_file": per_file,
        "user_overlap_valid_test": len(file_users.get("Electronics.valid.csv", set()) & file_users.get("Electronics.test.csv", set())),
        "item_overlap_valid_test": len(file_items.get("Electronics.valid.csv", set()) & file_items.get("Electronics.test.csv", set())),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
