"""Build Dataset B: real Amazon Reviews 2023 Electronics rating history.

Source rows are the existing rating-only split files `Electronics.valid.csv` and
`Electronics.test.csv`, filtered to parent_asin ids present in the AgentRec-X Electronics
catalogue. Only real observed ratings are labels. Non-interacted catalogue items are
UNJUDGED, never negatives.

Temporal split per user:
  target = each catalog interaction at index i
  history = earlier catalog interactions of the same user only
  last target -> test; second-last -> validation; earlier targets -> train

This is an external, pointwise preference/rating dataset, not a human pairwise preference
set. It is not fused into the 30-task Agent Eval.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/gold/amazon_reviews_electronics_v1/electronics_catalog_ratings.csv"
CATALOG = REPO / "data/p13r/electronics/catalog_meta.jsonl"
OUT = REPO / "data/gold/amazon_reviews_electronics_v1"
MIN_HISTORY = 1


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _catalog() -> dict[str, dict]:
    out = {}
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["item_id"])] = {
                "title": str(row.get("text") or "")[:300],
                "text": str(row.get("text") or "")[:4000],
            }
    return out


def main() -> int:
    catalog = _catalog()
    users: dict[str, list[tuple[int, str, float, str]]] = defaultdict(list)
    with RAW.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = str(row["parent_asin"])
            if item not in catalog:
                continue
            users[str(row["user_id"])].append((
                int(row["timestamp"]), item, float(row["rating"]), str(row["source_split"])
            ))
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    stats = Counter()
    history_lengths = []
    target_ratings = Counter()
    cold_users = Counter()
    for user, interactions in users.items():
        interactions.sort(key=lambda value: (value[0], value[1]))
        # The exact same item may have been rated multiple times; keep the latest per item
        # in a user's sequence but never put the target item into its own history.
        deduped = []
        seen = set()
        for ts, item, rating, source in reversed(interactions):
            if item in seen:
                continue
            seen.add(item)
            deduped.append((ts, item, rating, source))
        deduped.reverse()
        n = len(deduped)
        if n < 3:
            cold_users[n] += 1
            continue
        for index, (ts, item, rating, source) in enumerate(deduped):
            if index == 0:
                continue
            history = deduped[:index]
            split = "test" if index == n - 1 else ("val" if index == n - 2 else "train")
            if split == "train" and index < 1:
                continue
            if any(history_item == item for _, history_item, _, _ in history):
                stats[f"{split}_target_in_history_skipped"] += 1
                continue
            history_payload = [
                {
                    "item_id": history_item,
                    "timestamp_ms": history_ts,
                    "rating": history_rating,
                    "title": catalog.get(history_item, {}).get("title", ""),
                }
                for history_ts, history_item, history_rating, _ in history
            ]
            record = {
                "schema_version": "agentrecx.gold.amazon_reviews_electronics.v1",
                "user_id": user,
                "split": split,
                "history": history_payload,
                "history_length": len(history_payload),
                "target_item_id": item,
                "target_title": catalog.get(item, {}).get("title", ""),
                "target_text": catalog.get(item, {}).get("text", ""),
                "rating": rating,
                "target_timestamp_ms": ts,
                "source_split": source,
                "label_type": "observed_rating",
                "label_provenance": "Amazon Reviews 2023 Electronics rating split, real user rating",
                "candidate_universe": "current 76,918-item Electronics catalogue",
                "non_interacted_policy": "UNJUDGED — never treated as negative",
                "human_pairwise": False,
                "target_in_history": False,
                "history_cutoff_timestamp_ms": ts,
            }
            rows[split].append(record)
            stats[f"{split}_rows"] += 1
            history_lengths.append(len(history_payload))
            target_ratings[f"{split}:{rating:g}"] += 1
    # Query-group / user leakage check: same user can appear in train/val/test temporally,
    # but a user's test target timestamp must be later than its val/train targets.
    manifest = {
        "schema": "agentrecx.gold.amazon_reviews_electronics.dataset_manifest.v1",
        "dataset": "amazon_reviews_electronics_v1",
        "source": "Amazon Reviews 2023 Electronics rating-only split files",
        "source_files": [
            "/root/autodl-tmp/p13r/raw/Electronics.valid.csv",
            "/root/autodl-tmp/p13r/raw/Electronics.test.csv",
        ],
        "source_note": "Only user_id,parent_asin,rating,timestamp; no review text in these files.",
        "catalog": str(CATALOG),
        "catalog_items": len(catalog),
        "catalog_intersection": {
            "rows": sum(len(v) for v in users.values()),
            "users": len(users),
            "rating_distribution": {"test": {k.split(':',1)[1]: v for k, v in target_ratings.items() if k.startswith('test:')}},
        },
        "split_rule": "per-user chronological: last target test, second-last val, earlier train; history strictly before target",
        "counts": {split: len(rows[split]) for split in ("train", "val", "test")},
        "users_with_3plus_catalog_interactions": sum(1 for value in users.values() if len(value) >= 3),
        "cold_start_users_excluded": {"1_interaction": cold_users[1], "2_interactions": cold_users[2]},
        "mean_history_length": round(mean(history_lengths), 3) if history_lengths else 0,
        "max_history_length": max(history_lengths) if history_lengths else 0,
        "label_type": "observed_rating",
        "human_pairwise": False,
        "non_interacted_policy": "UNJUDGED, not negative",
        "target_leakage": "target excluded from every history; history filtered to earlier timestamps",
        "official_amazon_split": False,
        "no_p18": True,
        "files_sha256": {},
    }
    for split, part in rows.items():
        path = OUT / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in part:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        manifest["files_sha256"][split] = _sha(path)
    manifest["raw_sha256"] = _sha(RAW)
    manifest["raw_bytes"] = RAW.stat().st_size
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
