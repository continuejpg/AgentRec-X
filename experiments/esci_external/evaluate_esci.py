from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recommendation.rag.retrieval import BM25_B, BM25_K1, _idf, tokenize

OUT = ROOT / "runs/esci_external"
CANDIDATES = OUT / "esci_us_test_candidates.jsonl"
QUERIES = OUT / "esci_us_test_queries.jsonl"
PRODUCT_VECTORS = OUT / "bge_product_vectors.npz"
QUERY_VECTORS = OUT / "bge_query_vectors.npz"
K_VALUES = (10,)

# Semantic gain口径 (main result): task-specified ESCI Task 1 gain.  The official repository's prepare_trec_eval_files.py
# currently maps relevance positions E:4, S:2, C:3, I:1 and passes trec_eval
# ndcg.1=0,2=0.01,3=0.1,4=1, which is a different S/C assignment.  Both are reported as
# primary and sensitivity; the task-specified gain is primary.
PRIMARY_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}  # semantic gain口径 (main)
OFFICIAL_SCRIPT_GAINS = {"E": 1.0, "S": 0.01, "C": 0.1, "I": 0.0}  # official script口径 (S/C reversed)

NEGATION_RE = re.compile(r"\b(not|no|without|except|excluding|avoid|non)\b", re.I)
ACCESSORY_TERMS = {
    "case", "cover", "charger", "cable", "adapter", "mount", "holder", "protector",
    "protection", "replacement", "accessory", "accessories", "stand", "dock", "skin",
    "sleeve", "bag", "pouch", "strap", "bracket", "clip", "tripod", "attachment",
}
ATTRIBUTE_TERMS = {
    "wireless", "wired", "bluetooth", "waterproof", "water-resistant", "noise",
    "canceling", "cancelling", "battery", "rechargeable", "portable", "compact",
    "lightweight", "color", "black", "white", "red", "blue", "green", "pink",
    "purple", "silver", "gold", "size", "small", "large", "xl", "inch", "inches",
    "gb", "tb", "mah", "watts", "volt", "usb", "hdmi", "4k", "8k", "1080p",
}
MODEL_RE = re.compile(r"(?i)\b(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]{2,}\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dcg(gains: list[float]) -> float:
    # Official trec_eval nDCG (m_ndcg.c) uses linear gain / log2(rank+1),
    # not the exponential (2^gain - 1) variant.
    return sum(gain / math.log2(position + 2.0) for position, gain in enumerate(gains))


def ndcg(labels: list[str], order: list[int], gains: dict[str, float], k: int | None = None) -> float | None:
    ranked = [labels[int(position)] for position in order]
    if k is not None:
        ranked = ranked[:k]
    ideal = sorted((gains[label] for label in labels), reverse=True)
    if k is not None:
        ideal = ideal[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0.0:
        return None
    return dcg([gains[label] for label in ranked]) / ideal_dcg


def bm25_scores(query: str, titles: list[str], product_ids: list[str]) -> list[float]:
    terms = tuple(dict.fromkeys(tokenize(query)))
    documents = [tokenize(title) for title in titles]
    if not documents:
        return [0.0] * len(titles)
    if not terms:
        return [0.0] * len(titles)
    document_count = len(documents)
    total_length = sum(len(document) for document in documents)
    average_length = total_length / document_count if document_count else 1.0
    frequencies = []
    document_frequency = Counter()
    for document in documents:
        counts = Counter(document)
        frequencies.append(counts)
        for term in set(document):
            document_frequency[term] += 1
    scores = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        for term in terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = _idf(document_frequency.get(term, 0), document_count)
            if idf <= 0.0:
                continue
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * len(documents[index]) / average_length
            )
            score += idf * (frequency * (BM25_K1 + 1.0)) / denominator
        scores.append(score)
    return scores


def query_features(query: str, candidate_brands: list[str]) -> dict[str, bool]:
    lowered = query.lower()
    tokens = set(tokenize(query))
    brands = []
    for brand in candidate_brands:
        normalized = str(brand or "").strip().lower()
        if len(normalized) >= 2:
            brands.append(normalized)
    brand_hit = any(re.search(r"(?<![a-z0-9])" + re.escape(brand) + r"(?![a-z0-9])", lowered) for brand in brands)
    accessory_hit = bool(tokens & ACCESSORY_TERMS)
    attribute_hit = bool(tokens & ATTRIBUTE_TERMS)
    return {
        "negation": bool(NEGATION_RE.search(lowered)),
        "brand": brand_hit,
        "model_token": bool(MODEL_RE.search(lowered)),
        "attribute": attribute_hit,
        "accessory": accessory_hit,
        "main_product": not accessory_hit,
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000):
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    means = np.empty(n_boot)
    for index in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[index] = float(np.mean(sample))
    return {
        "mean": round(float(np.mean(values)), 6),
        "ci95": [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)],
        "n": int(n),
    }


def paired_bootstrap(bge: np.ndarray, bm25: np.ndarray, rng: np.random.Generator, n_boot: int = 10000):
    n = len(bge)
    diffs = bge - bm25
    means = np.empty(n_boot)
    for index in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        means[index] = float(np.mean(sample))
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    return {
        "mean_difference": round(float(np.mean(diffs)), 6),
        "ci95": [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)],
        "bge_wins": wins,
        "bm25_wins": losses,
        "ties": ties,
        "n": int(n),
    }


def main() -> None:
    candidates = load_jsonl(CANDIDATES)
    query_rows = load_jsonl(QUERIES)
    by_query = defaultdict(list)
    for row in candidates:
        by_query[str(row["query_id"])].append(row)
    query_ids = sorted(by_query, key=int)
    print(f"queries={len(query_ids)} candidates={len(candidates)}", flush=True)
    if len(query_ids) != 8956:
        print(f"WARNING expected 8956 queries, got {len(query_ids)}", file=sys.stderr)

    product_npz = np.load(PRODUCT_VECTORS, allow_pickle=True)
    query_npz = np.load(QUERY_VECTORS, allow_pickle=True)
    product_ids = [str(item) for item in product_npz["product_ids" if "product_ids" in product_npz else "ids"]]
    product_matrix = product_npz["vectors"].astype(np.float32)
    product_index = {product_id: index for index, product_id in enumerate(product_ids)}
    query_ids_from_npz = [str(item) for item in query_npz["query_ids" if "query_ids" in query_npz else "ids"]]
    query_matrix = query_npz["vectors"].astype(np.float32)
    query_index = {query_id: index for index, query_id in enumerate(query_ids_from_npz)}
    print(f"product_vectors={product_matrix.shape} query_vectors={query_matrix.shape}", flush=True)
    if product_matrix.shape[0] != len(product_ids) or query_matrix.shape[0] != len(query_ids_from_npz):
        raise SystemExit("vector/id shape mismatch")
    missing_products = [row["product_id"] for row in candidates if str(row["product_id"]) not in product_index]
    missing_queries = [qid for qid in query_ids if qid not in query_index]
    if missing_products or missing_queries:
        raise SystemExit(f"missing embeddings: products={len(missing_products)} queries={len(missing_queries)}")

    started = time.perf_counter()
    per_query = []
    for position, query_id in enumerate(query_ids, start=1):
        rows = by_query[query_id]
        query = str(rows[0]["query"])
        product_ids_for_query = [str(row["product_id"]) for row in rows]
        labels = [str(row["esci_label"]) for row in rows]
        titles = [str(row["product_title"]) for row in rows]
        brands = [str(row["product_brand"]) for row in rows]

        bm25_values = bm25_scores(query, titles, product_ids_for_query)
        bm25_order = sorted(range(len(rows)), key=lambda index: (-bm25_values[index], product_ids_for_query[index]))
        bge_values = [
            float(product_matrix[product_index[product_id]] @ query_matrix[query_index[query_id]])
            for product_id in product_ids_for_query
        ]
        bge_order = sorted(range(len(rows)), key=lambda index: (-bge_values[index], product_ids_for_query[index]))

        primary_10 = {
            "bm25": ndcg(labels, bm25_order, PRIMARY_GAINS, 10),
            "bge": ndcg(labels, bge_order, PRIMARY_GAINS, 10),
        }
        primary_all = {
            "bm25": ndcg(labels, bm25_order, PRIMARY_GAINS, None),
            "bge": ndcg(labels, bge_order, PRIMARY_GAINS, None),
        }
        repo_10 = {
            "bm25": ndcg(labels, bm25_order, OFFICIAL_SCRIPT_GAINS, 10),
            "bge": ndcg(labels, bge_order, OFFICIAL_SCRIPT_GAINS, 10),
        }
        repo_all = {
            "bm25": ndcg(labels, bm25_order, OFFICIAL_SCRIPT_GAINS, None),
            "bge": ndcg(labels, bge_order, OFFICIAL_SCRIPT_GAINS, None),
        }
        top_bm25 = [{
            "product_id": product_ids_for_query[index],
            "label": labels[index],
            "title": titles[index],
            "score": round(bm25_values[index], 8),
        } for index in bm25_order[:10]]
        top_bge = [{
            "product_id": product_ids_for_query[index],
            "label": labels[index],
            "title": titles[index],
            "score": round(bge_values[index], 8),
        } for index in bge_order[:10]]
        label_counts = Counter(labels)
        features = query_features(query, brands)
        record = {
            "query_id": query_id,
            "query": query,
            "candidate_count": len(rows),
            "label_counts": dict(label_counts),
            "bm25_ndcg10_primary": primary_10["bm25"],
            "bge_ndcg10_primary": primary_10["bge"],
            "bm25_ndcg_all_primary": primary_all["bm25"],
            "bge_ndcg_all_primary": primary_all["bge"],
            "bm25_ndcg10_repo_script_gain": repo_10["bm25"],
            "bge_ndcg10_repo_script_gain": repo_10["bge"],
            "bm25_ndcg_all_repo_script_gain": repo_all["bm25"],
            "bge_ndcg_all_repo_script_gain": repo_all["bge"],
            "query_features": features,
            "top10_bm25": top_bm25,
            "top10_bge": top_bge,
        }
        per_query.append(record)
        if position % 1000 == 0:
            print(f"evaluated {position}/{len(query_ids)} elapsed_s={time.perf_counter()-started:.1f}", flush=True)

    (OUT / "esci_per_query.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in per_query) + "\n",
        encoding="utf-8",
    )

    def values(field):
        return np.array([row[field] for row in per_query if row[field] is not None], dtype=np.float64)

    rng = np.random.default_rng(20261110)
    metric_names = [
        "bm25_ndcg10_primary", "bge_ndcg10_primary",
        "bm25_ndcg_all_primary", "bge_ndcg_all_primary",
        "bm25_ndcg10_repo_script_gain", "bge_ndcg10_repo_script_gain",
        "bm25_ndcg_all_repo_script_gain", "bge_ndcg_all_repo_script_gain",
    ]
    aggregate = {}
    for name in metric_names:
        aggregate[name] = bootstrap_mean(values(name), rng, n_boot=10000)
    aggregate["paired_ndcg10_primary"] = paired_bootstrap(
        values("bge_ndcg10_primary"), values("bm25_ndcg10_primary"), rng
    )
    aggregate["paired_ndcg_all_primary"] = paired_bootstrap(
        values("bge_ndcg_all_primary"), values("bm25_ndcg_all_primary"), rng
    )
    aggregate["paired_ndcg10_repo_script_gain"] = paired_bootstrap(
        values("bge_ndcg10_repo_script_gain"), values("bm25_ndcg10_repo_script_gain"), rng
    )
    aggregate["paired_ndcg_all_repo_script_gain"] = paired_bootstrap(
        values("bge_ndcg_all_repo_script_gain"), values("bm25_ndcg_all_repo_script_gain"), rng
    )

    feature_summary = {}
    for feature in ("negation", "brand", "model_token", "attribute", "accessory", "main_product"):
        subset = [row for row in per_query if row["query_features"][feature]]
        if not subset:
            feature_summary[feature] = {"n": 0}
            continue
        feature_summary[feature] = {
            "n": len(subset),
            "bm25_ndcg10_mean": round(float(np.mean([row["bm25_ndcg10_primary"] for row in subset])), 6),
            "bge_ndcg10_mean": round(float(np.mean([row["bge_ndcg10_primary"] for row in subset])), 6),
            "paired_difference": round(float(np.mean([row["bge_ndcg10_primary"] - row["bm25_ndcg10_primary"] for row in subset])), 6),
            "bm25_ndcg_all_mean": round(float(np.mean([row["bm25_ndcg_all_primary"] for row in subset])), 6),
            "bge_ndcg_all_mean": round(float(np.mean([row["bge_ndcg_all_primary"] for row in subset])), 6),
        }

    # Failure definition is transparent and retained: every query remains in per_query;
    # this subset only makes the failure analysis convenient.  A failure is either both
    # methods at zero nDCG@10, or a material BGE loss (BGE < BM25 by >= 0.10).
    failures = []
    for row in per_query:
        bm25 = row["bm25_ndcg10_primary"]
        bge = row["bge_ndcg10_primary"]
        reasons = []
        if bm25 == 0.0 and bge == 0.0:
            reasons.append("both_zero_ndcg10")
        if bge + 1e-12 < bm25 - 0.10:
            reasons.append("bge_material_loss")
        if reasons:
            failures.append({**row, "failure_reasons": reasons})
    (OUT / "esci_failures.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in failures) + "\n",
        encoding="utf-8",
    )

    # Representative failure samples: deterministic, one per query id, sorted by BGE loss.
    loss_samples = sorted(
        [row for row in per_query if row["bge_ndcg10_primary"] + 1e-12 < row["bm25_ndcg10_primary"] - 0.10],
        key=lambda row: (row["bge_ndcg10_primary"] - row["bm25_ndcg10_primary"], row["query_id"]),
    )[:15]
    both_zero_samples = sorted(
        [row for row in per_query if row["bge_ndcg10_primary"] == 0.0 and row["bm25_ndcg10_primary"] == 0.0],
        key=lambda row: int(row["query_id"]),
    )[:15]
    (OUT / "esci_failure_samples.json").write_text(
        json.dumps({"bge_material_loss": loss_samples, "both_zero_ndcg10": both_zero_samples}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    query_type_counts = {
        feature: sum(1 for row in per_query if row["query_features"][feature])
        for feature in ("negation", "brand", "model_token", "attribute", "accessory", "main_product")
    }
    result = {
        "schema": "agentrecx.esci_external.eval.v1",
        "task": "ESCI English-US Task 1 small_version test, judged-candidate reranking",
        "not_full_catalog_recall": True,
        "data": {
            "queries": len(query_ids),
            "candidate_rows": len(candidates),
            "catalog": "official ESCI product records, external; separate from AgentRec-X 76,263 catalogue",
        },
        "config": {
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "bm25_tokenizer": "lowercase alphanumeric runs; no stemming/stopwords; candidate-set local document frequency",
            "bm25_tie_break": "score desc, product_id asc",
            "bge_model": "BAAI/bge-base-en-v1.5 (local weights)",
            "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            "bge_pooling": "CLS",
            "bge_normalization": "L2",
            "bge_max_length": 512,
            "bge_tie_break": "score desc, product_id asc",
            "product_text_field": "product_title only; same field for BM25 and BGE",
            "query_text": "official ESCI query, unchanged",
            "gains_primary": PRIMARY_GAINS,
            "gains_repo_script_sensitivity": OFFICIAL_SCRIPT_GAINS,
            "ndcg_formula": "official trec_eval ndcg: sum(gain / log2(rank+1)); ideal DCG from the same candidate labels",
            "gains_note": "Primary follows the task-specified E=1,S=0.1,C=0.01,I=0 with the official trec_eval linear nDCG formula. The official repo prepare_trec_eval_files.py uses relevance positions E:4,S:2,C:3,I:1 and trec_eval gains 0,0.01,0.1,1, i.e. S=0.01/C=0.1; the sensitivity columns use that mapping and are not mixed with primary.",
            "test_tuning": False,
            "target_injection": False,
            "training": False,
            "index_used_for_bm25": "candidate-set local BM25 only; no global catalogue index",
            "bge_product_vectors": str(PRODUCT_VECTORS.relative_to(ROOT)),
            "bge_query_vectors": str(QUERY_VECTORS.relative_to(ROOT)),
        },
        "aggregate": aggregate,
        "query_type_counts": query_type_counts,
        "query_type_summary": feature_summary,
        "failure_counts": {
            "both_zero_ndcg10": sum(1 for row in per_query if row["bm25_ndcg10_primary"] == 0.0 and row["bge_ndcg10_primary"] == 0.0),
            "bge_material_loss": sum(1 for row in per_query if row["bge_ndcg10_primary"] + 1e-12 < row["bm25_ndcg10_primary"] - 0.10),
            "failure_rows_written": len(failures),
        },
        "per_query": str(OUT / "esci_per_query.jsonl"),
        "failures": str(OUT / "esci_failures.jsonl"),
        "hashes": {
            "candidates_sha256": sha256(CANDIDATES),
            "product_vectors_sha256": sha256(PRODUCT_VECTORS),
            "query_vectors_sha256": sha256(QUERY_VECTORS),
        },
    }
    (OUT / "esci_metrics.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=1, ensure_ascii=False))
    print("failure_counts", json.dumps(result["failure_counts"], ensure_ascii=False))
    print("query_type_summary", json.dumps(feature_summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
