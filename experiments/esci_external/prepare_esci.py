from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/esci_external"
OUT = ROOT / "runs/esci_external"
OUT.mkdir(parents=True, exist_ok=True)
EXAMPLES = DATA / "shopping_queries_dataset_examples.parquet"
PRODUCTS = DATA / "shopping_queries_dataset_products.parquet"
SOURCES = DATA / "shopping_queries_dataset_sources.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    examples_schema = pq.ParquetFile(EXAMPLES).schema_arrow
    products_schema = pq.ParquetFile(PRODUCTS).schema_arrow
    print("examples schema", examples_schema)
    print("products schema", products_schema)

    examples = pq.read_table(
        EXAMPLES,
        columns=["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "small_version", "split"],
        filters=[("small_version", "=", 1), ("split", "=", "test"), ("product_locale", "=", "us")],
    ).to_pylist()
    print("filtered examples", len(examples))
    needed_products = {str(row["product_id"]) for row in examples}
    print("needed products", len(needed_products))

    product_file = pq.ParquetFile(PRODUCTS)
    product_rows = []
    scanned = 0
    for batch in product_file.iter_batches(
        batch_size=65536,
        columns=["product_locale", "product_id", "product_title", "product_brand", "product_color"],
    ):
        scanned += batch.num_rows
        wanted = pc.and_(pc.equal(batch.column("product_locale"), pa.scalar("us")), pc.is_in(batch.column("product_id"), value_set=pa.array(sorted(needed_products))))
        filtered = batch.filter(wanted)
        if filtered.num_rows:
            product_rows.extend(filtered.to_pylist())
    print("scanned products", scanned, "matched product rows", len(product_rows))
    products = {str(row["product_id"]): row for row in product_rows}

    candidates = []
    missing_products = []
    duplicate_pairs = 0
    seen_pairs = set()
    for row in examples:
        query_id = str(row["query_id"])
        product_id = str(row["product_id"])
        pair = (query_id, product_id)
        if pair in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(pair)
        product = products.get(product_id)
        if product is None:
            missing_products.append({"query_id": query_id, "product_id": product_id})
        candidates.append({
            "query_id": query_id,
            "query": str(row["query"]),
            "product_id": product_id,
            "esci_label": str(row["esci_label"]),
            "product_locale": "us",
            "product_title": str(product.get("product_title") or "") if product else "",
            "product_brand": str(product.get("product_brand") or "") if product else "",
            "product_color": str(product.get("product_color") or "") if product else "",
            "product_metadata_present": product is not None,
            "small_version": 1,
            "split": "test",
        })

    # Query-level records.
    by_query = defaultdict(list)
    for row in candidates:
        by_query[row["query_id"]].append(row)
    queries = []
    label_counts = Counter()
    depth_counts = Counter()
    empty_queries = 0
    for query_id, rows in sorted(by_query.items(), key=lambda item: int(item[0])):
        labels = Counter(row["esci_label"] for row in rows)
        label_counts.update(labels)
        depth_counts[len(rows)] += 1
        if not rows:
            empty_queries += 1
        queries.append({
            "query_id": query_id,
            "query": rows[0]["query"],
            "candidate_count": len(rows),
            "labels": dict(labels),
            "has_relevant": any(row["esci_label"] in {"E", "S", "C"} for row in rows),
            "all_candidates_have_product_metadata": all(row["product_metadata_present"] for row in rows),
        })

    candidates.sort(key=lambda row: (int(row["query_id"]), row["product_id"]))
    (OUT / "esci_us_test_candidates.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in candidates) + "\n",
        encoding="utf-8",
    )
    (OUT / "esci_us_test_queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in queries) + "\n",
        encoding="utf-8",
    )

    source_meta = {}
    if SOURCES.is_file():
        source_meta = {"path": str(SOURCES.relative_to(ROOT)), "bytes": SOURCES.stat().st_size, "sha256": sha256(SOURCES)}
    manifest = {
        "schema": "agentrecx.esci_external.frozen_eval_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": "https://github.com/amazon-science/esci-data",
        "official_repo_commit": (DATA / "repo_commit.txt").read_text(encoding="utf-8").strip() if (DATA / "repo_commit.txt").is_file() else None,
        "license": "Apache-2.0 (official amazon-science/esci-data repository README/LICENSE)",
        "reference_paper": "https://arxiv.org/abs/2206.06588",
        "official_task": "Task 1 Query-Product Ranking; small_version; split=test; product_locale=us",
        "input_files": {
            "examples": {"path": str(EXAMPLES.relative_to(ROOT)), "bytes": EXAMPLES.stat().st_size, "sha256": sha256(EXAMPLES)},
            "products": {"path": str(PRODUCTS.relative_to(ROOT)), "bytes": PRODUCTS.stat().st_size, "sha256": sha256(PRODUCTS)},
            "sources": source_meta,
        },
        "filters": {"small_version": 1, "split": "test", "product_locale": "us"},
        "counts": {
            "examples_filtered": len(examples),
            "candidate_rows_after_dedupe": len(candidates),
            "unique_queries": len(queries),
            "duplicate_query_product_pairs_removed": duplicate_pairs,
            "queries_without_candidates": empty_queries,
            "missing_product_metadata_candidates": len(missing_products),
            "queries_with_all_product_metadata": sum(1 for q in queries if q["all_candidates_have_product_metadata"]),
            "label_counts": dict(label_counts),
            "depth_counts": dict(sorted(depth_counts.items())),
        },
        "query_text_used_for_retrieval": "query as provided by official examples",
        "product_text_field_used": "product_title only (same field for both BM25 and BGE)",
        "target_or_label_passed_to_retrieval": False,
        "judged_candidate_reranking": True,
        "full_catalog_retrieval": False,
        "notes": [
            "The examples file was filtered exactly by official small_version/split/product_locale columns.",
            "Candidate lists are the official judged products for each query; no unjudged products were added and no judged products were dropped for scoring.",
            "Missing product metadata candidates are retained with an empty title so the candidate set stays identical for both methods; counts are reported.",
            "The product file is read only for product_title (and brand/color for diagnostics); no target/label is passed to either scoring method.",
        ],
    }
    (OUT / "frozen_esci_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=1, ensure_ascii=False))
    print("missing example", missing_products[:3])
    if len(queries) != 8956:
        print(f"WARNING: expected 8,956 queries, got {len(queries)}", file=sys.stderr)
    if len(candidates) != 181701:
        print(f"WARNING: expected 181,701 rows, got {len(candidates)}", file=sys.stderr)


if __name__ == "__main__":
    main()
