from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.p14c1.search_backend import MAX_LENGTH, QUERY_INSTRUCTION

DATA = ROOT / "runs/esci_external/esci_us_test_candidates.jsonl"
OUT = ROOT / "runs/esci_external"
MODEL = ROOT / "models/bge-base-en-v1.5"


def load_rows():
    rows = []
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_texts(model, tokenizer, texts: list[str], *, prefix: str, batch_size: int, label: str):
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        chunk = [prefix + text for text in texts[start:start + batch_size]]
        inputs = tokenizer(
            chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(hidden.float(), p=2, dim=1).cpu().numpy()
        vectors.append(normalized.astype(np.float32))
        done = min(start + batch_size, len(texts))
        if done % (batch_size * 20) == 0 or done == len(texts):
            elapsed = time.perf_counter() - started
            print(
                f"{label} {done}/{len(texts)} elapsed_s={elapsed:.1f} "
                f"per_item_ms={elapsed / done * 1000:.2f}",
                flush=True,
            )
    return np.concatenate(vectors, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="smoke limit on products (0 = all)")
    args = parser.parse_args()

    rows = load_rows()
    query_texts = {}
    product_titles = {}
    for row in rows:
        query_texts.setdefault(str(row["query_id"]), str(row["query"]))
        product_titles.setdefault(str(row["product_id"]), str(row["product_title"]))
    query_ids = sorted(query_texts, key=int)
    product_ids = sorted(product_titles)
    if args.limit:
        product_ids = product_ids[: args.limit]
    queries = [query_texts[qid] for qid in query_ids]
    titles = [product_titles[pid] for pid in product_ids]
    print(f"query_count={len(queries)} product_count={len(titles)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModel.from_pretrained(str(MODEL), torch_dtype=torch.float32)
    model.eval()
    product_vectors = encode_texts(
        model, tokenizer, titles, prefix="", batch_size=args.batch_size, label="products"
    )
    query_vectors = encode_texts(
        model, tokenizer, queries, prefix=QUERY_INSTRUCTION, batch_size=args.batch_size, label="queries"
    )
    np.savez_compressed(
        OUT / "bge_product_vectors.npz",
        product_ids=np.array(product_ids, dtype=object),
        vectors=product_vectors,
    )
    np.savez_compressed(
        OUT / "bge_query_vectors.npz",
        query_ids=np.array(query_ids, dtype=object),
        vectors=query_vectors,
    )
    print(
        json.dumps(
            {
                "query_count": len(query_ids),
                "product_count": len(product_ids),
                "product_vectors_sha256": __import__("hashlib").sha256(
                    (OUT / "bge_product_vectors.npz").read_bytes()
                ).hexdigest(),
                "query_vectors_sha256": __import__("hashlib").sha256(
                    (OUT / "bge_query_vectors.npz").read_bytes()
                ).hexdigest(),
                "model": str(MODEL),
                "model_weights": str(MODEL / "pytorch_model.bin"),
                "query_prefix": QUERY_INSTRUCTION,
                "pooling": "CLS",
                "normalization": "L2",
                "max_length": MAX_LENGTH,
                "batch_size": args.batch_size,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
