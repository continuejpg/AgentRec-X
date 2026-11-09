"""P13-R Electronics retrieval baseline + history diagnostic.

BM25 and BGE-base only; no BLaIR/Qwen, no ranker training.  Test split is never retrieved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from recommendation.control.catalog_search import CatalogSearchSource

ROOT = Path(__file__).resolve().parent.parent.parent
TASKS = ROOT / "data/p13r/electronics/eval_tasks.jsonl"
CATALOG = ROOT / "data/p13r/electronics/catalog.jsonl"
BGE_MODEL = ROOT / "models/bge-base-en-v1.5"
OUT = ROOT / "runs/p13r/retrieval"
TOKEN_RE = re.compile(r"[a-z0-9]+")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text or "").casefold()))


def load_catalog(path: Path) -> Any:
    records: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item = str(row["item_id"])
            records[item] = SimpleNamespace(
                parent_asin=item,
                title=str(row.get("text") or ""),
                categories=(str(row.get("category") or ""),),
                features=(),
                description=(),
                price_text=None,
            )
    return SimpleNamespace(records=records)


def encode(texts: list[str], tokenizer, model, device, *, query: bool, limit: int = 256) -> np.ndarray:
    vectors = []
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if query:
                batch = [BGE_QUERY_INSTRUCTION + text for text in batch]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=limit, return_tensors="pt"
            ).to(device)
            output = model(**encoded).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def topk(matrix: np.ndarray, identities: list[str], query_vec: np.ndarray, k: int) -> list[Any]:
    scores = matrix @ query_vec
    order = np.lexsort((np.asarray(identities), -scores))[:k]
    return [
        SimpleNamespace(parent_asin=identities[int(index)], score=float(scores[int(index)]))
        for index in order
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--bge-model", type=Path, default=BGE_MODEL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.catalog)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    device = torch.device(args.device)
    bm25 = CatalogSearchSource(catalog, mode="bm25")
    identities = sorted(catalog.records)
    texts = [str(catalog.records[item].title or "") for item in identities]
    tokenizer = AutoTokenizer.from_pretrained(str(args.bge_model))
    model = AutoModel.from_pretrained(str(args.bge_model)).to(device).eval()
    begin = time.perf_counter()
    matrix = encode(texts, tokenizer, model, device, query=False, limit=256)
    encode_seconds = time.perf_counter() - begin
    np.save(args.out / "bge_electronics_vectors.npy", matrix)
    (args.out / "bge_electronics_identities.json").write_text(
        json.dumps(identities), encoding="utf-8"
    )
    index_bytes = matrix.nbytes
    print(f"BGE index built: {matrix.shape}, {encode_seconds:.1f}s", flush=True)

    rows = []
    for index, task in enumerate(tasks, start=1):
        query = str(task["query"])
        target = str(task["target_parent_asin"])
        bm25_hits = list(bm25.search(query, limit=100).hits)
        query_vec = encode([query], tokenizer, model, device, query=True, limit=128)[0]
        bge_hits = topk(matrix, identities, query_vec, 100)
        source = {}
        for name, hits in (("bm25", bm25_hits), ("bge", bge_hits)):
            rank = next((i + 1 for i, hit in enumerate(hits) if hit.parent_asin == target), None)
            source[name] = {"rank": rank, "hit@10": bool(rank and rank <= 10), "hit@50": bool(rank and rank <= 50), "hit@100": bool(rank and rank <= 100)}
        # Fixed Top-100 candidate pool: 50 BM25 + 50 BGE, deduplicated, deterministic fill.
        pool_ids: list[str] = []
        bm25_rank = {hit.parent_asin: i + 1 for i, hit in enumerate(bm25_hits)}
        bge_rank = {hit.parent_asin: i + 1 for i, hit in enumerate(bge_hits)}
        for hit in bm25_hits[:50]:
            if hit.parent_asin not in pool_ids:
                pool_ids.append(hit.parent_asin)
        for hit in bge_hits[:50]:
            if hit.parent_asin not in pool_ids:
                pool_ids.append(hit.parent_asin)
        for hits in (bm25_hits, bge_hits):
            for hit in hits:
                if len(pool_ids) >= 100:
                    break
                if hit.parent_asin not in pool_ids:
                    pool_ids.append(hit.parent_asin)
            if len(pool_ids) >= 100:
                break
        query_tokens = tokens(query)
        history = task["history"]
        history_tokens = [tokens(item.get("text")) for item in history if item.get("text")]
        history_categories = {item["category"] for item in history}
        target_ts = int(task["target_timestamp_ms"])
        pool = []
        for item in pool_ids:
            record = catalog.records[item]
            candidate_tokens = tokens(record.title)
            category = record.categories[0] if record.categories else ""
            category_match = 1.0 if category in history_categories else 0.0
            text_overlap = 0.0
            for tokens_hist in history_tokens:
                if tokens_hist:
                    text_overlap = max(text_overlap, len(candidate_tokens & tokens_hist) / max(1, len(tokens_hist)))
            time_decay = 0.0
            for history_item in history:
                if history_item["category"] == category:
                    age_days = max(0.0, (target_ts - int(history_item["timestamp_ms"])) / 86_400_000.0)
                    time_decay += 0.5 ** (age_days / 30.0)
            pool.append(
                {
                    "parent_asin": item,
                    "bm25_rank": bm25_rank.get(item),
                    "bge_rank": bge_rank.get(item),
                    "history_category_match": category_match,
                    "history_text_overlap": text_overlap,
                    "history_time_decay": time_decay,
                    "label_observed_positive": item == target,
                }
            )
        target_in_pool = any(row["parent_asin"] == target for row in pool)
        rows.append(
            {
                "sequence_id": task["sequence_id"], "qid": task["qid"], "split": task["split"],
                "user_id": task["user_id"], "target_parent_asin": target,
                "target_in_pool": target_in_pool, "pool_size": len(pool),
                "history_length": len(history),
                "history_text_items": sum(1 for item in history if item.get("text")),
                "source": source, "pool": pool,
            }
        )
        if index % 50 == 0:
            print(f"{index}/{len(tasks)}", flush=True)
    (args.out / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "agentrecx.p13r.retrieval.v1",
        "tasks": len(rows),
        "splits": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "dev")},
        "test_retrieved": False,
        "catalog_items": len(identities),
        "bge_index_bytes": index_bytes,
        "bge_index_seconds": encode_seconds,
        "pool_quota": "BM25 top50 + BGE top50, dedup, max 100",
        "target_in_pool": {
            split: sum(1 for row in rows if row["split"] == split and row["target_in_pool"])
            for split in ("train", "dev")
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
