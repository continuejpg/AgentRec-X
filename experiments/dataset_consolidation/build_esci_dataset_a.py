"""Build Dataset A: ESCI human-judged product relevance.

Official source: amazon-science/esci-data (Apache-2.0), small_version=1, product_locale=us.
The official test split is preserved. Validation is carved only from official train by
query_id, never from test.

The raw E/S/C/I label is retained. A task-specific graded gain is also stored:
E=1.0, S=0.1, C=0.01, I=0.0. This is a ranking mapping, not a claim that Complement is
universally worse than Substitute; the alternative official-script mapping is recorded in
the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ESCI = REPO / "data/esci_external"
DEFAULT_OUT = REPO / "data/gold/esci_relevance_v1"
LABEL_GAIN_PRIMARY = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
LABEL_GAIN_OFFICIAL_SCRIPT = {"E": 1.0, "S": 0.01, "C": 0.1, "I": 0.0}
LABEL_MEANING = {
    "E": "exact",
    "S": "substitute",
    "C": "complement",
    "I": "irrelevant",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _product_text(row: Any) -> str:
    title = str(row.get("product_title") or "")
    desc = str(row.get("product_description") or "")
    bullet = str(row.get("product_bullet_point") or "")
    brand = str(row.get("product_brand") or "")
    color = str(row.get("product_color") or "")
    parts = [title, brand, color, desc, bullet]
    return " ".join(part for part in parts if part and part != "nan").strip()[:4000]


def split_for_query(query_id: int) -> str:
    # Validation is the deterministic 10% of official train query groups.
    bucket = int(hashlib.sha256(f"esci-val:{query_id}".encode()).hexdigest()[:8], 16) % 10
    return "val" if bucket == 0 else "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esci", type=Path, default=DEFAULT_ESCI)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--locale", default="us")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    examples_path = args.esci / "shopping_queries_dataset_examples.parquet"
    products_path = args.esci / "shopping_queries_dataset_products.parquet"
    examples = pq.read_table(examples_path, columns=[
        "example_id", "query", "query_id", "product_id", "product_locale",
        "esci_label", "small_version", "large_version", "split",
    ]).to_pandas()
    examples = examples[
        (examples["product_locale"] == args.locale)
        & (examples["small_version"] == 1)
        & (examples["split"].isin(["train", "test"]))
    ].copy()
    products = pq.read_table(products_path, columns=[
        "product_id", "product_title", "product_description", "product_bullet_point",
        "product_brand", "product_color", "product_locale",
    ]).to_pandas()
    products = products[products["product_locale"] == args.locale].drop_duplicates(
        subset=["product_id"], keep="first"
    )
    product_map = {str(row.product_id): row for row in products.itertuples(index=False)}
    # Vectorized query split for official train.
    query_ids = examples["query_id"].astype(int)
    examples["split_final"] = [
        "test" if split == "test" else split_for_query(qid)
        for split, qid in zip(examples["split"], query_ids)
    ]
    # Ensure official test boundary and no query overlap.
    train_queries = set(examples.loc[examples["split_final"] == "train", "query_id"])
    val_queries = set(examples.loc[examples["split_final"] == "val", "query_id"])
    test_queries = set(examples.loc[examples["split_final"] == "test", "query_id"])
    if train_queries & val_queries or (train_queries | val_queries) & test_queries:
        raise RuntimeError("query leakage across ESCI splits")

    fields = [
        "example_id", "query_id", "query", "product_id", "product_locale",
        "esci_label", "split", "split_final", "small_version", "large_version",
    ]
    missing_products = 0
    counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        part = examples[examples["split_final"] == split]
        out_path = args.out / f"{split}.jsonl.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as handle:
            for row in part.itertuples(index=False):
                product = product_map.get(str(row.product_id))
                if product is None:
                    missing_products += 1
                    product_title = ""
                    product_text = ""
                    brand = ""
                    color = ""
                else:
                    product_title = str(getattr(product, "product_title", "") or "")
                    product_text = _product_text(product._asdict())
                    brand = str(getattr(product, "product_brand", "") or "")
                    color = str(getattr(product, "product_color", "") or "")
                record = {
                    "schema_version": "agentrecx.gold.esci_relevance.v1",
                    "example_id": int(row.example_id),
                    "query_id": int(row.query_id),
                    "query": str(row.query),
                    "product_id": str(row.product_id),
                    "product_locale": str(row.product_locale),
                    "product_title": product_title,
                    "product_text": product_text,
                    "product_brand": brand,
                    "product_color": color,
                    "esci_label_original": str(row.esci_label),
                    "label_meaning": LABEL_MEANING.get(str(row.esci_label), "unknown"),
                    "label_gain_primary": LABEL_GAIN_PRIMARY.get(str(row.esci_label), 0.0),
                    "label_gain_official_script": LABEL_GAIN_OFFICIAL_SCRIPT.get(str(row.esci_label), 0.0),
                    "label_mapping_version": "esci_primary_E1_S0.1_C0.01_I0",
                    "label_provenance": "Amazon ESCI human judgment; official examples.parquet",
                    "human_label": True,
                    "query_group": f"esci-us:{int(row.query_id)}",
                    "split": split,
                    "official_split": str(row.split),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts[split] += 1
                label_counts[split][str(row.esci_label)] = label_counts[split].get(
                    str(row.esci_label), 0
                ) + 1
        if split == "train":
            # Avoid writing the same query twice: train/val/test query groups are disjoint.
            pass
    if missing_products:
        raise RuntimeError(f"{missing_products} ESCI examples have no product metadata")
    manifest = {
        "schema": "agentrecx.gold.esci_relevance.dataset_manifest.v1",
        "dataset": "esci_relevance_v1",
        "source": "amazon-science/esci-data",
        "source_url": "https://github.com/amazon-science/esci-data",
        "repo_commit": (args.esci / "repo_commit.txt").read_text(encoding="utf-8").strip(),
        "license": "Apache-2.0",
        "official_task": "small_version=1, product_locale=us, query-product relevance",
        "official_test_preserved": True,
        "validation_rule": "10% of official train query_ids by sha256 bucket; never test",
        "label_map_primary": LABEL_GAIN_PRIMARY,
        "label_map_official_script": LABEL_GAIN_OFFICIAL_SCRIPT,
        "label_mapping_note": "E/S/C/I are relation labels; C is not assumed universally worse than S; primary is a declared ranking gain, alternative is retained for sensitivity.",
        "counts": counts,
        "label_counts": label_counts,
        "unique_queries": {
            "train": len(train_queries), "val": len(val_queries), "test": len(test_queries)
        },
        "missing_product_metadata": missing_products,
        "input_sha256": {
            "examples": sha256(examples_path),
            "products": sha256(products_path),
            "sources": sha256(args.esci / "shopping_queries_dataset_sources.csv"),
        },
        "files_sha256": {
            split: sha256(args.out / f"{split}.jsonl.gz")
            for split in ("train", "val", "test")
        },
        "no_catalog_fusion": "ESCI product ids are external; current Agent catalogue intersection is reported separately.",
        "no_p18": True,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
