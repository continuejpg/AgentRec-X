"""Phase 5.1: select 100 new ESCI test queries and build the two tracks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "p100_extended_benchmark"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261114
EXAMPLES = REPO / "data/esci_external/shopping_queries_dataset_examples.parquet"
PRODUCTS = REPO / "data/esci_external/shopping_queries_dataset_products.parquet"
BENCH = REPO / "data/memory_ranking_benchmark_v2"
PHASE41 = REPO / "runs/public_preference_ranking_v1/esci_test_predictions.jsonl"
CATALOG = REPO / "data/p13r/electronics/catalog_meta.jsonl"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")
ELEC = re.compile(r"headphone|earbud|earphone|speaker|soundbar|charger|cable|adapter|phone|tablet|laptop|computer|monitor|keyboard|mouse|router|camera|tv|television|battery|usb|hdmi|bluetooth|wireless|gaming|console|printer|ssd|hard drive|watch|smartwatch|power bank|audio|microphone|webcam|projector|vacuum|coffee|blender|kitchen|appliance|electronics|receiver|amplifier|antenna|stylus|screen|display|hub|dock|tripod|light|bulb|smart home", re.I)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def main():
    examples = pd.read_parquet(EXAMPLES, columns=["query_id", "query", "product_id", "esci_label",
                                                   "product_locale", "small_version", "split"])
    examples = examples[(examples.small_version == 1) & (examples.split == "test") &
                        (examples.product_locale == "us")].copy()
    products = pd.read_parquet(PRODUCTS, columns=["product_id", "product_title", "product_description",
                                                   "product_bullet_point", "product_brand", "product_color",
                                                   "product_locale"])
    products = products.drop_duplicates("product_id").set_index("product_id")
    catalog_ids = set()
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            catalog_ids.add(json.loads(line)["item_id"])
    used_ids = set()
    used_text = set()
    for line in (BENCH / "base_queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            used_ids.add(str(row["query_id"]))
            used_text.add(norm(row["query"]))
    for line in PHASE41.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            used_ids.add(str(row["query_id"]))
            used_text.add(norm(row["query"]))
    groups = []
    overlap_funnel = Counter()
    for qid, group in examples.groupby("query_id"):
        text = str(group["query"].iloc[0]).strip()
        text_norm = norm(text)
        if str(qid) in used_ids:
            overlap_funnel["excluded_query_id"] += 1
            continue
        if text_norm in used_text:
            overlap_funnel["excluded_normalized_text"] += 1
            continue
        p = group.drop_duplicates("product_id")
        labels = Counter(p["esci_label"].tolist())
        judged = set(p["product_id"])
        overlap = judged & catalog_ids
        if len(p) < 2 or len(labels) < 2:
            overlap_funnel["insufficient_judgments"] += 1
            continue
        if not ELEC.search(text):
            overlap_funnel["not_electronics_keyword"] += 1
            continue
        if not (6 <= len(text) <= 90):
            overlap_funnel["query_length"] += 1
            continue
        groups.append({
            "query_id": str(qid), "query": text, "norm": text_norm,
            "judged": p, "labels": labels,
            "catalog_overlap": sorted(overlap),
            "n_judged": int(len(p)), "n_labels": len(labels),
            "has_positive": labels.get("E", 0) + labels.get("S", 0) >= 2,
        })
    # Prefer queries with catalog overlap and a positive judged example; do not use model scores.
    preferred = [g for g in groups if g["catalog_overlap"] and g["has_positive"]]
    fallback = [g for g in groups if g not in preferred]
    rng = random.Random(SEED)
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    selected = preferred[:100]
    if len(selected) < 100:
        selected += fallback[:100 - len(selected)]
    selected = sorted(selected, key=lambda g: (int(g["query_id"]), g["norm"]))
    # Normalized-text de-duplication after selection.
    seen = set()
    unique = []
    for g in selected:
        if g["norm"] in seen:
            continue
        seen.add(g["norm"])
        unique.append(g)
    selected = unique
    # Deterministic scenario assignment: one scenario per query, balanced across the 100.
    types = (["positive_memory"] * 22 + ["negative_memory"] * 15 + ["multiple_preferences"] * 12 +
             ["preference_update"] * 10 + ["preference_withdrawal"] * 10 + ["unknown_evidence"] * 10 +
             ["preference_conflict"] * 6 + ["query_memory_competition"] * 8 + ["no_memory"] * 7)
    while len(types) < len(selected):
        types.append("no_memory")
    rng = random.Random(SEED + 1)
    rng.shuffle(types)
    base_rows = []
    judged_rows = []
    memory_rows = []
    for index, g in enumerate(selected):
        bid = f"p100_{index + 1:03d}"
        qid = g["query_id"]
        base_rows.append({
            "base_query_id": bid, "query_id": qid, "query": g["query"],
            "normalized_query": g["norm"],
            "query_family": f"esci_q{qid}",
            "source": "amazon_esci_us_test_small_version",
            "source_commit": "7916cdf6ab75a462e77f20ab40428a10923998d5",
            "n_judged_products": g["n_judged"],
            "n_relevance_levels": g["n_labels"],
            "catalog_overlap_count": len(g["catalog_overlap"]),
            "catalog_overlap_ids": g["catalog_overlap"],
            "selection_seed": SEED,
        })
        judged_rows.append({
            "base_query_id": bid, "query_id": qid, "query": g["query"],
            "source": "amazon_esci_us_test_small_version",
            "candidate_source": "official_esci_judged_products",
            "products": [
                {"product_id": str(row["product_id"]), "esci_label": str(row["esci_label"]),
                 "title": str(products.loc[row["product_id"], "product_title"]) if row["product_id"] in products.index else "",
                 "brand": str(products.loc[row["product_id"], "product_brand"]) if row["product_id"] in products.index else "",
                 "category": "Electronics",
                 "label_provenance": "ESCI_HUMAN_JUDGED"}
                for _, row in g["judged"].iterrows()
            ],
        })
        scenario = types[index]
        memory_rows.append({
            "base_query_id": bid, "query_id": qid, "scenario_type": scenario,
            "memory_snapshot_id": f"{bid}:{scenario}",
            "source": "synthetic_event_sequence",
            "positive": [],
            "negative": [],
        })
    # Save selection manifest and tracks.
    funnel = {
        "esci_test_us_queries": int(examples["query_id"].nunique()),
        "after_exclusion_and_filters": len(groups),
        "excluded_funnel": dict(overlap_funnel),
        "selected": len(selected),
        "selected_with_catalog_overlap": sum(1 for g in selected if g["catalog_overlap"]),
        "selected_with_positive_judgment": sum(1 for g in selected if g["has_positive"]),
    }
    (OUT / "p100_selection_manifest.json").write_text(json.dumps({
        "schema": "agentrecx.p100.selection_manifest.v1",
        "created_utc": "2026-11-14T18:00:00+08:00",
        "random_seed": SEED,
        "official_source": "Amazon ESCI shopping_queries_dataset examples small_version=1 split=test product_locale=us",
        "official_commit": "7916cdf6ab75a462e77f20ab40428a10923998d5",
        "exclusions": {"benchmark_v2_query_ids": str(BENCH / "base_queries.jsonl"),
                        "phase41_query_ids": str(PHASE41)},
        "funnel": funnel,
        "queries": base_rows,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "p100_agent_tasks.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in base_rows) + "\n", encoding="utf-8")
    (OUT / "p100_judged_tasks.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in judged_rows) + "\n", encoding="utf-8")
    (OUT / "p100_memory_states.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in memory_rows) + "\n", encoding="utf-8")
    coverage = {
        "schema": "agentrecx.p100.label_coverage.v1",
        "source": "official ESCI Task 1 human relevance labels; Agent BGE candidates separate",
        "queries": len(selected),
        "agent_pools": "pending BGE retrieval",
        "judged_products": sum(len(g["judged"]) for g in selected),
        "judged_product_rows": sum(len(g["judged"]) for g in selected),
        "queries_with_at_least_2_judged": sum(1 for g in selected if len(g["judged"]) >= 2),
        "queries_with_at_least_2_levels": sum(1 for g in selected if len(g["labels"]) >= 2),
        "queries_with_catalog_overlap": sum(1 for g in selected if g["catalog_overlap"]),
        "catalog_overlap_products": sum(len(g["catalog_overlap"]) for g in selected),
        "agent_esci_exact_overlap": "to be computed after BGE pools",
        "human_relevance_labels": "ESCI labeled examples only; unjudged Agent candidates remain UNJUDGED",
        "human_memory_preference_gold": 0,
    }
    (OUT / "p100_label_coverage.json").write_text(json.dumps(coverage, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"funnel": funnel, "scenarios": dict(Counter(r["scenario_type"] for r in memory_rows))}, indent=1))


if __name__ == "__main__":
    main()
