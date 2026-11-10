from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_texts(path: Path, id_field: str, text_field: str):
    ids = []
    texts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row[id_field]))
            texts.append(str(row[text_field]))
    return ids, texts


def encode(model, tokenizer, texts, *, prefix: str, batch_size: int, device: torch.device, label: str, out_path: Path):
    vectors = []
    total = len(texts)
    started = time.perf_counter()
    current_batch = batch_size
    position = 0
    while position < total:
        end = min(position + current_batch, total)
        try:
            chunk = [prefix + text for text in texts[position:end]]
            inputs = tokenizer(
                chunk,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            vectors.append(normalized.cpu().numpy().astype(np.float32))
            position = end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch <= 32:
                raise
            current_batch = max(32, current_batch // 2)
            print(f"OOM: {label} retry with batch_size={current_batch} at position={position}", flush=True)
            continue
        if position % (current_batch * 100) == 0 or position == total:
            elapsed = time.perf_counter() - started
            print(
                f"{label} {position}/{total} elapsed_s={elapsed:.2f} "
                f"items_per_s={position / max(elapsed, 1e-9):.1f} "
                f"batch={current_batch} gpu_alloc_mb={torch.cuda.memory_allocated()/1024**2:.1f}",
                flush=True,
            )
    matrix = np.concatenate(vectors, axis=0)
    return matrix, current_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-jsonl", required=True)
    parser.add_argument("--query-jsonl", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/models/bge-base-en-v1.5")
    parser.add_argument("--out-dir", default="/root/esci_eval")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    product_ids, product_texts = load_texts(Path(args.product_jsonl), "product_id", "product_title")
    query_ids, query_texts = load_texts(Path(args.query_jsonl), "query_id", "query")
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": os.uname().nodename,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "product_count": len(product_ids),
                "query_count": len(query_ids),
                "product_unique_ids": len(set(product_ids)),
                "query_unique_ids": len(set(query_ids)),
                "model": args.model,
                "batch_size_requested": args.batch_size,
            },
            indent=1,
        ),
        flush=True,
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    peak_before = torch.cuda.max_memory_allocated() / 1024**2
    product_started = time.perf_counter()
    # Product vectors are the expensive set; do them first so the query cache can be
    # completed quickly if the remote session is interrupted.
    product_matrix, product_batch = encode(
        model, tokenizer, product_texts, prefix="", batch_size=args.batch_size,
        device=device, label="products", out_path=out_dir / "products.npz",
    )
    product_elapsed = time.perf_counter() - product_started
    product_peak = torch.cuda.max_memory_allocated() / 1024**2
    query_started = time.perf_counter()
    query_matrix, query_batch = encode(
        model, tokenizer, query_texts, prefix=QUERY_INSTRUCTION, batch_size=args.batch_size,
        device=device, label="queries", out_path=out_dir / "queries.npz",
    )
    query_elapsed = time.perf_counter() - query_started
    total_peak = torch.cuda.max_memory_allocated() / 1024**2
    np.savez_compressed(out_dir / "products.npz", ids=np.array(product_ids, dtype=object), vectors=product_matrix)
    np.savez_compressed(out_dir / "queries.npz", ids=np.array(query_ids, dtype=object), vectors=query_matrix)
    report = {
        "schema": "agentrecx.esci_external.gpu_encoding.v1",
        "hostname": os.uname().nodename,
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "dtype": "float32",
        "tf32": False,
        "product_count": len(product_ids),
        "query_count": len(query_ids),
        "embedding_dim": int(product_matrix.shape[1]),
        "batch_size_requested": args.batch_size,
        "batch_size_product": product_batch,
        "batch_size_query": query_batch,
        "product_elapsed_s": round(product_elapsed, 3),
        "query_elapsed_s": round(query_elapsed, 3),
        "product_items_per_s": round(len(product_ids) / product_elapsed, 2),
        "query_items_per_s": round(len(query_ids) / query_elapsed, 2),
        "peak_gpu_memory_mb_before": round(peak_before, 1),
        "peak_gpu_memory_mb_products": round(product_peak, 1),
        "peak_gpu_memory_mb_total": round(total_peak, 1),
        "products_sha256": sha256(out_dir / "products.npz"),
        "queries_sha256": sha256(out_dir / "queries.npz"),
        "products_finite": bool(np.isfinite(product_matrix).all()),
        "queries_finite": bool(np.isfinite(query_matrix).all()),
        "product_norm_min_max": [float(np.linalg.norm(product_matrix, axis=1).min()), float(np.linalg.norm(product_matrix, axis=1).max())],
        "query_norm_min_max": [float(np.linalg.norm(query_matrix, axis=1).min()), float(np.linalg.norm(query_matrix, axis=1).max())],
        "protocol": {
            "model": "BAAI/bge-base-en-v1.5",
            "query_prefix": QUERY_INSTRUCTION,
            "product_prefix": "",
            "pooling": "CLS",
            "normalization": "L2",
            "max_length": 512,
        },
        "labels_sent_to_gpu": False,
    }
    (out_dir / "gpu_encoding_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
