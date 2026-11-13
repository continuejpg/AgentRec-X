"""Build the optional Electronics metadata sidecar from Amazon Reviews 2023.

This is a one-off data build tool, not a second catalogue or runtime.  It scans the
official ``raw_meta_Electronics`` Parquet shards, keeps only records whose ``parent_asin``
is accepted by the existing C4 Electronics catalogue, and writes a JSONL sidecar plus a
machine-readable manifest.  It never modifies C4, the BGE index, or any evaluation file.

Run from the repository root:

    python scripts/build_electronics_metadata_sidecar.py \
      --metadata-dir "$TEMP/amazon_reviews_2023_raw_meta_Electronics" \
      --out-dir data/enrichment
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO / "data/p13r/electronics/catalog_meta.jsonl"
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_VERSION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
SIDECAR_SCHEMA = "agentrecx.electronics_metadata_sidecar.v1"
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
BOOK_MARKER = re.compile(
    r"\b(paperback|hardcover|novel|textbook|isbn|book review|about the author|"
    r"publisher|edition|memoir|fiction)\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_c4_ids(path: Path = DEFAULT_CATALOG) -> set[str]:
    """Return the exact accepted IDs used by ``ElectronicsCatalogue.load``.

    Keeping this parser next to the build tool makes the sidecar's denominator
    reproducible without importing torch (the catalogue itself imports it transitively).
    """
    accepted: set[str] = set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id") or "").strip()
            text = " ".join(str(row.get("text") or "").split())
            if not item_id or not ASIN_RE.match(item_id):
                continue
            if item_id in seen:
                continue
            seen.add(item_id)
            if len(text) < 20:
                continue
            # Mirrors ElectronicsCatalogue._first_sentence(limit=220) exactly.
            title = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:220].strip()
            if not title or BOOK_MARKER.search(title):
                continue
            accepted.add(item_id)
    return accepted


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _parse_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "").replace("$", "")
    if not text or text.casefold() in {"none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_details(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if str(k).strip()}, True
    if not isinstance(value, str) or not value.strip():
        return {}, True
    if value.strip().casefold() in {"none", "null"}:
        return {}, True
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}, False
    if not isinstance(parsed, dict):
        return {}, False
    return {str(k): v for k, v in parsed.items() if str(k).strip()}, True


def _parse_number(value: Any, *, integer: bool = False) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if integer else number


def iter_metadata_rows(directory: Path) -> Iterator[dict[str, Any]]:
    shards = sorted(directory.glob("full-*-of-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no raw_meta_Electronics Parquet shards under {directory}")
    columns = [
        "parent_asin", "title", "features", "description", "categories", "store",
        "details", "price", "average_rating", "rating_number", "main_category",
    ]
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32_000, columns=columns):
            for row in batch.to_pylist():
                yield row


def build(catalog: Path, metadata_dir: Path, out_dir: Path) -> dict[str, Any]:
    accepted = accepted_c4_ids(catalog)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sidecar = out_dir / "electronics_metadata_sidecar.jsonl"
    out_manifest = out_dir / "manifest.json"

    seen: set[str] = set()
    duplicates: list[str] = []
    malformed_details = 0
    missing_title = 0
    rows_read = 0
    records: list[dict[str, Any]] = []

    for row in iter_metadata_rows(metadata_dir):
        rows_read += 1
        item_id = str(row.get("parent_asin") or "").strip()
        if item_id not in accepted:
            continue
        if item_id in seen:
            duplicates.append(item_id)
            continue
        seen.add(item_id)
        title = _clean_text(row.get("title"))
        if title is None:
            missing_title += 1
        details, details_ok = _parse_details(row.get("details"))
        if not details_ok:
            malformed_details += 1
        record = {
            "schema": SIDECAR_SCHEMA,
            "item_id": item_id,
            "title": title,
            "features": _clean_list(row.get("features")),
            "description": _clean_list(row.get("description")),
            "categories": _clean_list(row.get("categories")),
            "store": _clean_text(row.get("store")),
            "details": details,
            "price": _parse_price(row.get("price")),
            "average_rating": _parse_number(row.get("average_rating")),
            "rating_number": _parse_number(row.get("rating_number"), integer=True),
            "main_category": _clean_text(row.get("main_category")),
            "source": "amazon_reviews_2023:raw_meta_Electronics",
            "dataset_version": DATASET_VERSION,
        }
        records.append(record)

    records.sort(key=lambda item: item["item_id"])
    tmp = out_sidecar.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(out_sidecar)

    matched = {record["item_id"] for record in records}
    missing = sorted(accepted - matched)
    shard_paths = sorted(metadata_dir.glob("full-*-of-*.parquet"))
    shard_info = [{"name": p.name, "bytes": p.stat().st_size,
                   "sha256": sha256_file(p)} for p in shard_paths]
    feature_nonempty = sum(1 for r in records if r["features"])
    description_nonempty = sum(1 for r in records if r["description"])
    price_valid = sum(1 for r in records if r["price"] is not None)
    store_nonempty = sum(1 for r in records if r["store"])
    categories_nonempty = sum(1 for r in records if r["categories"])
    details_nonempty = sum(1 for r in records if r["details"])
    title_nonempty = sum(1 for r in records if r["title"])
    rating_valid = sum(1 for r in records if r["average_rating"] is not None)
    rating_number_valid = sum(1 for r in records if r["rating_number"] is not None)
    detail_keys = Counter(key for record in records for key in record["details"])

    manifest = {
        "schema": "agentrecx.electronics_metadata_sidecar_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": DATASET_ID,
            "config": "raw_meta_Electronics",
            "dataset_version": DATASET_VERSION,
            "accessed_via": "hf-mirror.com (official HF endpoint unreachable from shell)",
            "license_status": "not verified in this run; do not assume redistribution rights",
            "raw_meta_rows": 1_610_012,
            "download_bytes": 1_955_009_715,
            "uncompressed_bytes": 4_603_602_269,
            "shards": shard_info,
        },
        "catalog": {
            "path": str(catalog.relative_to(REPO) if catalog.is_relative_to(REPO) else catalog),
            "sha256": sha256_file(catalog),
            "rows": sum(1 for line in catalog.read_text(encoding="utf-8").splitlines() if line.strip()),
            "accepted": len(accepted),
        },
        "match": {
            "rows_scanned": rows_read,
            "accepted_ids": len(accepted),
            "matched_ids": len(matched),
            "match_rate": round(len(matched) / len(accepted), 6) if accepted else 0.0,
            "missing_ids_count": len(missing),
            "missing_ids_sample": missing[:20],
            "duplicate_metadata_ids": len(duplicates),
            "missing_title": missing_title,
            "malformed_details": malformed_details,
        },
        "field_coverage": {
            "records": len(records),
            "title": title_nonempty,
            "features": feature_nonempty,
            "description": description_nonempty,
            "price": price_valid,
            "store": store_nonempty,
            "categories": categories_nonempty,
            "details": details_nonempty,
            "average_rating": rating_valid,
            "rating_number": rating_number_valid,
            "detail_keys_top": detail_keys.most_common(20),
        },
        "sidecar": {
            "path": str(out_sidecar.relative_to(REPO) if out_sidecar.is_relative_to(REPO) else out_sidecar),
            "rows": len(records),
            "bytes": out_sidecar.stat().st_size,
            "sha256": sha256_file(out_sidecar),
        },
        "semantics": {
            "price": "historical crawl-time price from upstream metadata; never live price/stock",
            "average_rating": "historical aggregate shown on the product page at crawl time",
            "details": "upstream JSON object; malformed details are stored as {} and counted",
            "bought_together": "not stored in this MVP; the sampled upstream field was empty",
        },
        "notes": [
            "Only accepted C4 Electronics item_ids are kept; C4 identity/search text is unchanged.",
            "This sidecar is additive and is not part of the BGE index.",
            "Field coverage is not proof of factual correctness.",
        ],
    }
    out_manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Electronics metadata sidecar")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/enrichment")
    args = parser.parse_args(argv)
    manifest = build(args.catalog, args.metadata_dir, args.out_dir)
    print(json.dumps({
        "matched": manifest["match"]["matched_ids"],
        "accepted": manifest["match"]["accepted_ids"],
        "match_rate": manifest["match"]["match_rate"],
        "sidecar": manifest["sidecar"],
        "field_coverage": manifest["field_coverage"],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
