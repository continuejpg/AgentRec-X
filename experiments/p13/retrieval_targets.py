"""P13 full-source target-rank diagnostics (same queries, no pooled truncation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.inference.dense_catalog import DenseCatalogIndex

ROOT = Path(__file__).resolve().parent.parent.parent
TASKS = ROOT / "data/p13/sports/tasks.jsonl"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
BGE_INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
BGE_MODEL = ROOT / "models/bge-base-en-v1.5"
BLAIR_MODEL = ROOT / "models/blair-roberta-base"
CACHE = ROOT / "runs/p13/retrieval/blair_cache"
OUT = ROOT / "runs/p13/retrieval"


def load_catalog(path: Path) -> Any:
    records: dict[str, Any] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            asin = row.get("parent_asin")
            if not asin:
                continue
            records[str(asin)] = SimpleNamespace(
                parent_asin=str(asin),
                title=str(row.get("title") or ""),
                store=row.get("store"),
                main_category=row.get("main_category"),
                categories=tuple(row.get("categories") or ()),
                features=tuple(row.get("features") or ()),
                description=tuple(row.get("description") or ()),
                price_text=row.get("price_text"),
            )
    return SimpleNamespace(records=records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--products", type=Path, default=PRODUCTS)
    parser.add_argument("--mappings", type=Path, default=MAPPINGS)
    parser.add_argument("--bge-index", type=Path, default=BGE_INDEX)
    parser.add_argument("--bge-model", type=Path, default=BGE_MODEL)
    parser.add_argument("--blair-model", type=Path, default=BLAIR_MODEL)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.products)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    dense = DenseCatalogIndex.load(
        args.bge_index, model_path=str(args.bge_model),
        catalogue_records=catalog.records,
        mapping_sha=__import__("hashlib").sha256(args.mappings.read_bytes()).hexdigest(),
    )
    bm25 = CatalogSearchSource(catalog, mode="bm25")
    bge = CatalogSearchSource(catalog, mode="bge", dense_index=dense)
    blair_ids = json.loads((args.cache / "blair_identities.json").read_text(encoding="utf-8"))
    blair_vectors = np.load(args.cache / "blair_vectors.npy")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.blair_model))
    blair_model = AutoModel.from_pretrained(str(args.blair_model)).to(device).eval()
    rows = []
    for task in tasks:
        query = str(task["proxy_query"])
        target = task["target_parent_asin"]
        bm25_hits = bm25.search(query, limit=100).hits
        bge_hits = bge.search(query, limit=100).hits
        encoded = tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.inference_mode():
            query_vec = torch.nn.functional.normalize(blair_model(**encoded).last_hidden_state[:, 0], dim=-1).cpu().numpy()[0]
        order = np.lexsort((np.asarray(blair_ids), -(blair_vectors @ query_vec)))[:100]
        blair_hits = [SimpleNamespace(parent_asin=blair_ids[int(i)], score=float((blair_vectors @ query_vec)[int(i)])) for i in order]
        row = {"sequence_id": task["sequence_id"], "qid": task["qid"], "split": task["split"], "user_id": task["user_id"], "target": target}
        for name, hits in (("bm25", bm25_hits), ("bge", bge_hits), ("blair", blair_hits)):
            rank = next((i + 1 for i, hit in enumerate(hits) if hit.parent_asin == target), None)
            row[name] = {"hit@10": bool(rank and rank <= 10), "hit@50": bool(rank and rank <= 50), "hit@100": bool(rank and rank <= 100), "rank": rank}
        rows.append(row)
        print(task["sequence_id"], {name: row[name]["rank"] for name in ("bm25", "bge", "blair")}, flush=True)
    (args.out / "source_target_ranks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "tasks": len(rows),
        "splits": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "dev", "test")},
        "note": "one observed positive per query; hit@K is observed-positive hit rate, not full relevance recall",
        "metrics": {
            name: {
                split: {
                    "hit@10": sum(1 for row in rows if row["split"] == split and row[name]["hit@10"]),
                    "hit@50": sum(1 for row in rows if row["split"] == split and row[name]["hit@50"]),
                    "hit@100": sum(1 for row in rows if row["split"] == split and row[name]["hit@100"]),
                    "queries": sum(1 for row in rows if row["split"] == split),
                }
                for split in ("train", "dev", "test")
            }
            for name in ("bm25", "bge", "blair")
        },
    }
    (args.out / "source_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
