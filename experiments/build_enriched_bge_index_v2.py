"""Build the versioned Enriched BGE index for Advanced Retrieval v2.

The output NPZ is compatible with ``LocalBGEDenseIndex`` (identities/vectors/titles) and
is additive: the original ``runs/p14d2/bge_index.npz`` is never overwritten.

Phase 1 may run a small CPU smoke build; the full 76,263-document encode is a Phase 2 GPU
task.  The script supports bounded batches and a resumable partial checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_docs(path: Path, max_items: int | None) -> tuple[list[str], list[str], list[str]]:
    identities: list[str] = []
    titles: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not item_id or not text:
                raise ValueError("retrieval document is missing item_id/text")
            identities.append(item_id)
            titles.append(str(row.get("title") or ""))
            texts.append(text)
            if max_items is not None and len(identities) >= max_items:
                break
    if not identities:
        raise ValueError("no retrieval documents found")
    if len(set(identities)) != len(identities):
        raise ValueError("retrieval documents contain duplicate item_ids")
    return identities, titles, texts


def _save_index(path: Path, identities: list[str], vectors, titles: list[str]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, identities=np.asarray(identities, dtype=np.str_),
        vectors=np.asarray(vectors, dtype=np.float32),
        titles=np.asarray(titles, dtype=np.str_),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-id", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    if args.max_length < 1 or args.batch_size < 1:
        raise SystemExit("batch-size and max-length must be positive")

    started = time.perf_counter()
    identities, titles, texts = _load_docs(args.docs, args.max_items)
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.eval()
    model_max_length = min(int(args.max_length), int(getattr(tokenizer, "model_max_length", 512)))
    partial = args.out.with_suffix(args.out.suffix + ".partial.npz")
    vectors = np.empty((0, 768), dtype=np.float32)
    start = 0
    if args.resume and partial.is_file():
        saved = np.load(partial, allow_pickle=False)
        saved_ids = [str(item) for item in saved["identities"]]
        if saved_ids != identities[:len(saved_ids)]:
            raise SystemExit("partial checkpoint identities do not prefix the current documents")
        vectors = np.asarray(saved["vectors"], dtype=np.float32)
        if vectors.shape != (len(saved_ids), 768):
            raise SystemExit("partial checkpoint has an invalid vector shape")
        start = len(saved_ids)
        print(f"resume from {start}/{len(identities)} documents", flush=True)

    checkpoint_batches = max(1, int(args.checkpoint_every))
    batches = 0
    token_lengths: list[int] = []
    truncated_docs = 0
    for offset in range(start, len(identities), int(args.batch_size)):
        batch_texts = texts[offset:offset + int(args.batch_size)]
        # Measure effective document length before truncation.  This is recorded in the
        # manifest so Phase 2 can see whether long enriched documents lose important
        # specifications at the 512-token BGE limit.
        raw_tokenized = tokenizer(batch_texts, add_special_tokens=True, truncation=False)
        for ids in raw_tokenized["input_ids"]:
            token_lengths.append(len(ids))
            if len(ids) > model_max_length:
                truncated_docs += 1
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=model_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state[:, 0]
            batch_vectors = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
        batch_vectors = batch_vectors.cpu().numpy().astype(np.float32)
        if batch_vectors.ndim != 2 or batch_vectors.shape[1] != 768:
            raise SystemExit(f"model returned invalid shape {batch_vectors.shape}")
        vectors = np.vstack([vectors, batch_vectors]) if len(vectors) else batch_vectors
        batches += 1
        print(f"encoded {len(vectors)}/{len(identities)}", flush=True)
        if batches % checkpoint_batches == 0 and len(vectors) < len(identities):
            _save_index(partial, identities[:len(vectors)], vectors, titles[:len(vectors)])

    if vectors.shape != (len(identities), 768):
        raise SystemExit(f"embedding/identity misalignment: {vectors.shape}")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise SystemExit("embeddings are not L2 normalized")
    import statistics
    sorted_lengths = sorted(token_lengths)
    token_stats = {
        "documents": len(sorted_lengths),
        "model_max_length": model_max_length,
        "mean": round(sum(sorted_lengths) / len(sorted_lengths), 2) if sorted_lengths else 0.0,
        "p50": int(statistics.median(sorted_lengths)) if sorted_lengths else 0,
        "p95": sorted_lengths[int(0.95 * (len(sorted_lengths) - 1))] if sorted_lengths else 0,
        "max": max(sorted_lengths) if sorted_lengths else 0,
        "truncated_documents": truncated_docs,
        "truncated_rate": round(truncated_docs / len(sorted_lengths), 6) if sorted_lengths else 0.0,
    }
    _save_index(args.out, identities, vectors, titles)
    if partial.is_file():
        partial.unlink()
    manifest = {
        "schema": "agentrecx.advanced_retrieval.v2_bge_manifest.v1",
        "builder_version": "advanced_retrieval_v2",
        "index_format": "agentrecx.dense_catalog.v3.compatible_npz",
        "docs_path": str(args.docs),
        "docs_sha256": _sha256_file(args.docs),
        "index_path": str(args.out),
        "index_sha256": _sha256_file(args.out),
        "documents": len(identities),
        "embedding_dim": 768,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_path": args.model,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "normalize_embeddings": True,
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "token_stats": token_stats,
        "max_items": args.max_items,
        "resume_used": bool(args.resume),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "alignment_check": "identities and vectors have equal row counts; every vector is L2-normalized",
        "load_contract": "runs via experiments.p14c1.search_backend.LocalBGEDenseIndex.load",
        "full_index_required_for_phase2": args.max_items is None and len(identities) == 76263,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
