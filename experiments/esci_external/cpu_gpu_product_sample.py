from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.esci_external.encode_esci_bge import MAX_LENGTH, QUERY_INSTRUCTION

OUT = ROOT / "runs/esci_external"
MODEL = ROOT / "models/bge-base-en-v1.5"


def main() -> None:
    products = {}
    with (OUT / "esci_product_texts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            products[str(row["product_id"])] = str(row["product_title"])
    queries = {}
    with (OUT / "esci_query_texts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["query_id"])] = str(row["query"])
    product_ids = sorted(products)
    query_ids = sorted(queries, key=int)
    sample_products = product_ids[:: max(1, len(product_ids) // 128)][:128]
    sample_queries = query_ids[:: max(1, len(query_ids) // 32)][:32]
    print("sample products", len(sample_products), "queries", len(sample_queries), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModel.from_pretrained(str(MODEL), torch_dtype=torch.float32)
    model.eval()
    started = time.perf_counter()
    vectors = []
    with torch.inference_mode():
        for position in range(0, len(sample_products), 32):
            titles = [products[product_id] for product_id in sample_products[position:position + 32]]
            inputs = tokenizer(titles, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            hidden = model(**inputs).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(hidden.float(), p=2, dim=1).numpy().astype(np.float32))
    cpu_products = np.concatenate(vectors, axis=0)
    cpu_elapsed = time.perf_counter() - started
    np.savez_compressed(OUT / "cpu_product_sample.npz", ids=np.array(sample_products, dtype=object), vectors=cpu_products)

    gpu_npz = np.load(OUT / "bge_product_vectors_gpu.npz", allow_pickle=True)
    gpu_ids = [str(item) for item in gpu_npz["ids"]]
    gpu_matrix = gpu_npz["vectors"].astype(np.float32)
    gpu_index = {product_id: index for index, product_id in enumerate(gpu_ids)}
    gpu_products = np.stack([gpu_matrix[gpu_index[product_id]] for product_id in sample_products], axis=0)
    product_diff = np.abs(cpu_products - gpu_products)
    product_cos = (cpu_products * gpu_products).sum(axis=1)

    cpu_query_npz = np.load(OUT / "bge_query_vectors_cpu.npz", allow_pickle=True)
    gpu_query_npz = np.load(OUT / "bge_query_vectors_gpu.npz", allow_pickle=True)
    cpu_query_ids = [str(item) for item in cpu_query_npz["ids"]]
    gpu_query_ids = [str(item) for item in gpu_query_npz["ids"]]
    cpu_query_index = {query_id: index for index, query_id in enumerate(cpu_query_ids)}
    gpu_query_index = {query_id: index for index, query_id in enumerate(gpu_query_ids)}
    cpu_query_vectors = cpu_query_npz["vectors"].astype(np.float32)
    gpu_query_vectors = gpu_query_npz["vectors"].astype(np.float32)
    score_diffs = []
    top1_matches = 0
    top5_overlap = []
    for query_id in sample_queries:
        cpu_scores = cpu_products @ cpu_query_vectors[cpu_query_index[query_id]]
        gpu_scores = gpu_products @ gpu_query_vectors[gpu_query_index[query_id]]
        score_diffs.extend(np.abs(cpu_scores - gpu_scores).tolist())
        cpu_top = np.argsort(-cpu_scores, kind="stable")
        gpu_top = np.argsort(-gpu_scores, kind="stable")
        top1_matches += int(cpu_top[0] == gpu_top[0])
        top5_overlap.append(len(set(cpu_top[:5].tolist()) & set(gpu_top[:5].tolist())) / 5.0)
    result = {
        "cpu_product_sample_count": len(sample_products),
        "cpu_query_sample_count": len(sample_queries),
        "cpu_elapsed_s": round(cpu_elapsed, 3),
        "cpu_samples_per_s": round(len(sample_products) / cpu_elapsed, 2),
        "product_max_abs": float(product_diff.max()),
        "product_mean_abs": float(product_diff.mean()),
        "product_cos_min": float(product_cos.min()),
        "product_cos_mean": float(product_cos.mean()),
        "score_max_abs": float(np.max(score_diffs)),
        "score_mean_abs": float(np.mean(score_diffs)),
        "top1_matches": top1_matches,
        "top5_overlap_mean": round(float(np.mean(top5_overlap)), 6),
        "sample_product_ids": sample_products,
        "sample_query_ids": sample_queries,
        "comparison_note": "CPU sample is only 128 products/32 queries for protocol verification; evaluation uses GPU vectors for both queries and products.",
    }
    (OUT / "cpu_gpu_product_diff.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "cpu_product_sample_count", "cpu_query_sample_count", "cpu_elapsed_s",
        "product_max_abs", "product_mean_abs", "product_cos_min", "product_cos_mean",
        "score_max_abs", "score_mean_abs", "top1_matches", "top5_overlap_mean",
    )}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
