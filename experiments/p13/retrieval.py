"""P13 retrieval feature builder: BM25 / BGE / BLaIR / fixed candidate pool.

Isolated under experiments/p13.  Uses the current Sports catalogue and the public C4
proxy queries; no default Serving path is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from recommendation.control.catalog_search import CatalogSearchSource
from recommendation.demo.product_role import infer_target
from recommendation.inference.dense_catalog import DenseCatalogIndex

ROOT = Path(__file__).resolve().parent.parent.parent
TASKS = ROOT / "data/p13/sports/tasks.jsonl"
PRODUCTS = ROOT / "data/processed/Sports_and_Outdoors_products.jsonl"
MAPPINGS = ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
BGE_INDEX = ROOT / "runs/p8_bge/bge_base_dense.npz"
BGE_MODEL = ROOT / "models/bge-base-en-v1.5"
BLAIR_MODEL = ROOT / "models/blair-roberta-base"
OUT = ROOT / "runs/p13/retrieval"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _normalise(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, dim=-1)


def _encode_blair_items(model, tokenizer, catalog: Any, device: torch.device) -> tuple[list[str], np.ndarray]:
    identities = sorted(catalog.records)
    texts = []
    for asin in identities:
        record = catalog.records[asin]
        title = str(getattr(record, "title", "") or "").strip()
        categories = " ".join(str(value) for value in (getattr(record, "categories", ()) or ()))
        texts.append((title + " " + categories).strip() or asin)
    vectors = []
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            output = model(**encoded).last_hidden_state[:, 0]
            vectors.append(_normalise(output).cpu().numpy())
            if (start // batch_size) % 100 == 0:
                print(f"blair encode {start}/{len(texts)}", flush=True)
    return identities, np.concatenate(vectors, axis=0)


def _blair_cache(
    model_dir: Path,
    catalog: Any,
    cache_dir: Path,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    id_path = cache_dir / "blair_identities.json"
    vec_path = cache_dir / "blair_vectors.npy"
    if id_path.exists() and vec_path.exists():
        identities = json.loads(id_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path)
        if len(identities) == len(vectors):
            print("loaded BLaIR cache", flush=True)
            return identities, vectors
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(device).eval()
    print("encoding BLaIR catalogue", flush=True)
    identities, vectors = _encode_blair_items(model, tokenizer, catalog, device)
    np.save(vec_path, vectors)
    id_path.write_text(json.dumps(identities), encoding="utf-8")
    return identities, vectors


def _topk_scores(scores: np.ndarray, identities: list[str], k: int) -> list[dict]:
    # Exact deterministic top-k; ties by smaller identity.
    order = np.lexsort((np.asarray(identities), -scores))[:k]
    return [
        SimpleNamespace(
            parent_asin=identities[int(i)],
            score=float(scores[int(i)]),
        )
        for i in order
    ]


def _history_features(
    task: dict,
    pool: list[dict],
    *,
    bge_vectors: np.ndarray,
    bge_identities: list[str],
    blair_vectors: np.ndarray,
    blair_identities: list[str],
    catalog: Any,
    bge_row: dict[str, int],
    blair_row: dict[str, int],
) -> dict[str, dict[str, float]]:
    history = task["history"]
    target_ts = int(task["target_timestamp_ms"])
    pool_asins = [row["parent_asin"] for row in pool]

    def features(hist: list[dict]) -> dict[str, dict[str, float]]:
        in_catalog = [item for item in hist if item.get("in_current_catalogue")]
        h_bge = np.asarray(
            [bge_vectors[bge_row[item["parent_asin"]]] for item in in_catalog if item["parent_asin"] in bge_row],
            dtype=np.float32,
        ) if in_catalog else np.zeros((0, 768), dtype=np.float32)
        h_blair = np.asarray(
            [blair_vectors[blair_row[item["parent_asin"]]] for item in in_catalog if item["parent_asin"] in blair_row],
            dtype=np.float32,
        ) if in_catalog else np.zeros((0, 768), dtype=np.float32)
        categories = [str(item["category"]) for item in hist]
        output = {}
        for row in pool:
            asin = row["parent_asin"]
            record = catalog.records.get(asin)
            candidate_categories = set(str(v) for v in (getattr(record, "categories", ()) or ()))
            candidate_bge = bge_vectors[bge_row[asin]] if asin in bge_row else np.zeros(768, dtype=np.float32)
            candidate_blair = blair_vectors[blair_row[asin]] if asin in blair_row else np.zeros(768, dtype=np.float32)
            if len(h_bge):
                bge_sim = h_bge @ candidate_bge
                blair_sim = h_blair @ candidate_blair
                bge_max = float(np.max(bge_sim))
                bge_mean = float(np.mean(bge_sim))
                blair_max = float(np.max(blair_sim))
                blair_mean = float(np.mean(blair_sim))
            else:
                bge_max = bge_mean = blair_max = blair_mean = 0.0
            category_matches = sum(
                1 for category in categories if category in candidate_categories
            )
            decayed_matches = 0.0
            for item in hist:
                if str(item["category"]) in candidate_categories:
                    age_days = max(0.0, (target_ts - int(item["timestamp_ms"])) / 86_400_000.0)
                    decayed_matches += 0.5 ** (age_days / 30.0)
            output[asin] = {
                "history_len": float(len(hist)),
                "history_in_catalog": float(len(in_catalog)),
                "history_category_match": float(category_matches),
                "history_category_fraction": float(category_matches / max(1, len(categories))),
                "history_time_decay_match": float(decayed_matches),
                "history_bge_max": bge_max,
                "history_bge_mean": bge_mean,
                "history_blair_max": blair_max,
                "history_blair_mean": blair_mean,
            }
        return output

    true_features = features(history)
    # Donor history: the nearest other task by history length and category set; never the
    # same user.  This is a controlled unrelated-history ablation, not a fake preference.
    return {"true": true_features, "unrelated": {}, "removed": {}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--products", type=Path, default=PRODUCTS)
    parser.add_argument("--mappings", type=Path, default=MAPPINGS)
    parser.add_argument("--bge-index", type=Path, default=BGE_INDEX)
    parser.add_argument("--bge-model", type=Path, default=BGE_MODEL)
    parser.add_argument("--blair-model", type=Path, default=BLAIR_MODEL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the BLaIR/BGE P13 retrieval build")
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.products)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    mapping = json.loads(args.mappings.read_text(encoding="utf-8"))
    item2id = mapping["item2id"]
    dense = DenseCatalogIndex.load(
        args.bge_index,
        model_path=str(args.bge_model),
        catalogue_records=catalog.records,
        mapping_sha=sha(args.mappings),
    )
    bm25 = CatalogSearchSource(catalog, mode="bm25")
    bge = CatalogSearchSource(catalog, mode="bge", dense_index=dense)
    device = torch.device(args.device)
    blair_identities, blair_vectors = _blair_cache(
        args.blair_model, catalog, args.out / "blair_cache", device
    )
    bge_payload = np.load(args.bge_index, allow_pickle=False)
    bge_identities = [str(value) for value in bge_payload["identities"].tolist()]
    bge_vectors = np.asarray(bge_payload["vectors"], dtype=np.float32)
    bge_row = {asin: index for index, asin in enumerate(bge_identities)}
    blair_row = {asin: index for index, asin in enumerate(blair_identities)}
    tokenizer = AutoTokenizer.from_pretrained(str(args.blair_model))
    blair_model = AutoModel.from_pretrained(str(args.blair_model)).to(device).eval()

    # Donor task map, built once.
    donors: dict[str, dict] = {}
    for task in tasks:
        candidates = [
            other for other in tasks
            if other["split"] == task["split"] and other["user_id"] != task["user_id"]
        ]
        if not candidates:
            continue
        task_cats = set(item["category"] for item in task["history"])
        donor = min(
            candidates,
            key=lambda other: (
                abs(len(other["history"]) - len(task["history"])),
                len(task_cats ^ set(item["category"] for item in other["history"])),
            ),
        )
        donors[task["sequence_id"]] = donor

    output = []
    for index, task in enumerate(tasks, start=1):
        query = str(task["proxy_query"])
        bm25_hits = list(bm25.search(query, limit=100).hits)
        bge_hits = list(bge.search(query, limit=100).hits)
        encoded_query = tokenizer(
            [query], padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            query_vec = blair_model(**encoded_query).last_hidden_state[:, 0]
            query_vec = _normalise(query_vec).cpu().numpy()[0]
        blair_scores = blair_vectors @ query_vec
        blair_hits = _topk_scores(blair_scores, blair_identities, 100)
        # Fixed candidate pool: 50 BM25 + 25 BGE + 25 BLaIR; deduplicate, then fill from
        # BM25/BGE/BLaIR heads until 100.  The final pool never exceeds 100.
        pool_asins: list[str] = []
        source_rank: dict[str, dict[str, int]] = {"bm25": {}, "bge": {}, "blair": {}}
        source_score: dict[str, dict[str, float]] = {"bm25": {}, "bge": {}, "blair": {}}
        for source, hits in (("bm25", bm25_hits), ("bge", bge_hits), ("blair", blair_hits)):
            for rank, hit in enumerate(hits, start=1):
                source_rank[source][hit.parent_asin] = rank
                source_score[source][hit.parent_asin] = float(hit.score)
        quota = [("bm25", 50), ("bge", 25), ("blair", 25)]
        for source, count in quota:
            for hit in {"bm25": bm25_hits, "bge": bge_hits, "blair": blair_hits}[source][:count]:
                if hit.parent_asin not in pool_asins:
                    pool_asins.append(hit.parent_asin)
        # Fill deterministically from BM25, then BGE, then BLaIR.
        for source_hits in (bm25_hits, bge_hits, blair_hits):
            for hit in source_hits:
                if len(pool_asins) >= 100:
                    break
                if hit.parent_asin not in pool_asins:
                    pool_asins.append(hit.parent_asin)
            if len(pool_asins) >= 100:
                break
        pool = [
            {
                "parent_asin": asin,
                "bm25_rank": source_rank["bm25"].get(asin),
                "bm25_score": source_score["bm25"].get(asin),
                "bge_rank": source_rank["bge"].get(asin),
                "bge_score": source_score["bge"].get(asin),
                "blair_rank": source_rank["blair"].get(asin),
                "blair_score": source_score["blair"].get(asin),
                "label_observed_positive": asin == task["target_parent_asin"],
            }
            for asin in pool_asins
        ]
        true_hist = _history_features(
            task, pool, bge_vectors=bge_vectors, bge_identities=bge_identities,
            blair_vectors=blair_vectors, blair_identities=blair_identities, catalog=catalog,
            bge_row=bge_row, blair_row=blair_row,
        )["true"]
        donor = donors.get(task["sequence_id"], {"history": []})
        donor_task = dict(task)
        donor_task["history"] = donor.get("history", [])
        unrelated_hist = _history_features(
            donor_task, pool, bge_vectors=bge_vectors, bge_identities=bge_identities,
            blair_vectors=blair_vectors, blair_identities=blair_identities, catalog=catalog,
            bge_row=bge_row, blair_row=blair_row,
        )["true"]
        removed_hist = {
            asin: {
                "history_len": 0.0, "history_in_catalog": 0.0, "history_category_match": 0.0,
                "history_category_fraction": 0.0, "history_time_decay_match": 0.0,
                "history_bge_max": 0.0, "history_bge_mean": 0.0,
                "history_blair_max": 0.0, "history_blair_mean": 0.0,
            }
            for asin in pool_asins
        }
        target = task["target_parent_asin"]
        target_rank = next(
            (i + 1 for i, row in enumerate(pool) if row["parent_asin"] == target), None
        )
        output.append(
            {
                "sequence_id": task["sequence_id"],
                "qid": task["qid"],
                "split": task["split"],
                "user_id": task["user_id"],
                "query": query,
                "target_parent_asin": target,
                "pool_target_rank_by_source_fusion": target_rank,
                "pool": pool,
                "history_features": {
                    "true": true_hist,
                    "unrelated": unrelated_hist,
                    "removed": removed_hist,
                },
            }
        )
        if index % 10 == 0:
            (args.out / "queries.partial.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
                encoding="utf-8",
            )
            print(f"{index}/{len(tasks)}", flush=True)
    (args.out / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "tasks": len(output),
                "splits": {split: sum(1 for row in output if row["split"] == split) for split in ("train", "dev", "test")},
                "pool_quota": {"bm25": 50, "bge": 25, "blair": 25},
                "blair_identities": len(blair_identities),
                "bge_identities": len(bge_identities),
                "target_in_pool": sum(1 for row in output if row["pool_target_rank_by_source_fusion"] is not None),
                "note": "Pool is fixed before ranking; one observed positive per query; unjudged items are not judged negatives.",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
