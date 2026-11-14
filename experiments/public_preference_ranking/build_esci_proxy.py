"""Build the ESCI Query-Derived Preference Proxy dataset.

The preference is always a *current-query* clause, never historical memory:
    preference_source = query_derived_proxy

Official ESCI E/S/C/I labels are preserved unchanged.  We only select queries whose
original text contains a removable, product-relevant preference clause.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/public_preference_ranking_v1"
EXAMPLES = REPO / "data/esci_external/shopping_queries_dataset_examples.parquet"
PRODUCTS = REPO / "data/esci_external/shopping_queries_dataset_products.parquet"
BENCH_QUERIES = REPO / "data/memory_ranking_benchmark_v2/base_queries.jsonl"

PRODUCT_TERMS = (
    "headphone", "earbud", "earphone", "speaker", "keyboard", "mouse", "laptop", "monitor",
    "camera", "watch", "tv", "television", "phone", "charger", "cable", "adapter", "printer",
    "tablet", "router", "microphone", "webcam", "drive", "ssd", "gpu", "graphics", "projector",
    "dock", "hub", "case", "stand", "tripod", "lens", "battery", "power", "headset", "console",
    "controller", "vr", "smartwatch", "earbuds", "monitors", "keyboards",
)
ATTR_PATTERNS = {
    "wireless": r"\bwireless\b",
    "wired": r"\bwired\b",
    "bluetooth": r"\bbluetooth\b",
    "microphone": r"\b(?:microphone|mic)\b",
    "rechargeable": r"\brechargeable\b",
    "noise_cancelling": r"\bnoise[- ]?cancell?ing\b",
    "waterproof": r"\bwaterproof\b|\bwater[- ]resistant\b",
    "over_ear": r"\bover[- ]?ear\b",
    "on_ear": r"\bon[- ]?ear\b",
    "in_ear": r"\bin[- ]?ear\b|\bearbuds?\b",
    "gaming": r"\bgaming\b",
    "mechanical": r"\bmechanical\b",
    "portable": r"\bportable\b",
    "foldable": r"\bfoldable\b",
    "usb": r"\busb(?:-?c)?\b",
    "hdmi": r"\bhdmi\b",
    "4k": r"\b4k\b|\buhd\b",
    "remote": r"\bremote\b",
    "led": r"\bled\b",
    "rgb": r"\brgb\b",
    "touch": r"\btouch\b",
    "ergonomic": r"\bergonomic\b",
    "silent": r"\bsilent\b",
    "lightweight": r"\blightweight\b",
    "compact": r"\bcompact\b",
    "durable": r"\bdurable\b",
    "adjustable": r"\badjustable\b",
    "fast_charging": r"\bfast[- ]charg(?:ing|e)\b",
    "long_battery": r"\blong[- ](?:lasting )?battery\b",
    "professional": r"\bprofessional\b",
    "travel": r"\btravel\b",
    "running": r"\brunning\b",
    "workout": r"\bworkout\b",
    "office": r"\boffice\b",
    "home": r"\bhome\b",
    "kids": r"\bkids?\b",
    "women": r"\bwomen(?:'s)?\b",
    "men": r"\bmen(?:'s)?\b",
    "black": r"\bblack\b",
    "white": r"\bwhite\b",
    "blue": r"\bblue\b",
    "red": r"\bred\b",
    "pink": r"\bpink\b",
    "silver": r"\bsilver\b",
    "gold": r"\bgold\b",
    "small": r"\bsmall\b",
    "large": r"\blarge\b",
    "mini": r"\bmini\b",
    "premium": r"\bpremium\b",
    "cheap": r"\bcheap(?:er)?\b",
    "affordable": r"\baffordable\b",
}
ATTR_RE = {attr: re.compile(pattern, re.I) for attr, pattern in ATTR_PATTERNS.items()}
EVIDENCE_RE = {attr: re.compile(pattern, re.I) for attr, pattern in ATTR_PATTERNS.items()}


def _extract(query: str) -> dict[str, Any] | None:
    text = str(query or "").strip()
    if not any(term in text.lower() for term in PRODUCT_TERMS):
        return None
    best = None
    for attr, pattern in ATTR_RE.items():
        match = pattern.search(text)
        if not match:
            continue
        start, end = match.span()
        before = text[:start]
        negative = bool(re.search(r"\b(?:without|no|not|non[- ]?)\s*$", before, re.I))
        remove_start = start
        if negative:
            connector = re.search(r"(?:without|no|not|non[- ]?)\s*$", before, re.I)
            if connector:
                remove_start = connector.start()
        else:
            connector = re.search(
                r"(?:with|has|have|featuring|including|built[- ]in)\s+(?:a |an |the )?$",
                before, re.I,
            )
            if connector:
                remove_start = connector.start()
        base = re.sub(r"\s+", " ", (text[:remove_start] + " " + text[end:]).strip(" ,;"))
        if len(base.split()) < 2 or not any(term in base.lower() for term in PRODUCT_TERMS):
            continue
        candidate = {
            "attribute": attr,
            "polarity": "negative" if negative else "positive",
            "clause": text[remove_start:end].strip(),
            "base_query": base,
            "extraction_rule": "negated_or_connective_clause_removal_v1",
            "match_start": remove_start,
            "match_end": end,
        }
        if best is None or (end - start) > (best["match_end"] - best["match_start"]):
            best = candidate
    return best


def _evidence_fields(title: str, description: str, bullets: str, color: str) -> dict[str, Any]:
    text = " ".join(str(value or "") for value in (title, description, bullets, color))
    return {f"evidence_{attr}": bool(pattern.search(text)) for attr, pattern in EVIDENCE_RE.items()}


def _split_for(query_id: str) -> str:
    bucket = int(hashlib.md5(str(query_id).encode()).hexdigest(), 16) % 10
    return "train" if bucket < 9 else "val"


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    examples = pd.read_parquet(EXAMPLES)
    us = examples[(examples["small_version"] == 1) & (examples["product_locale"] == "us")].copy()
    benchmark = [json.loads(line) for line in BENCH_QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    benchmark_ids = {str(row["query_id"]) for row in benchmark}
    benchmark_texts = {str(row["query"]).strip().lower() for row in benchmark}

    query_rows = us[["query_id", "query"]].drop_duplicates().copy()
    extracted = []
    for row in query_rows.itertuples():
        item = _extract(row.query)
        if item:
            extracted.append({"query_id": str(row.query_id), "query": str(row.query), **item})
    extracted = pd.DataFrame(extracted)

    # Build labels from official examples, preserving E/S/C/I exactly.
    selected_ids = set(extracted["query_id"])
    label_rows = us[us["query_id"].astype(str).isin(selected_ids)][
        ["query_id", "query", "product_id", "esci_label", "split", "example_id"]
    ].copy()
    label_rows["query_id"] = label_rows["query_id"].astype(str)
    label_rows["product_id"] = label_rows["product_id"].astype(str)
    label_rows = label_rows.rename(columns={"split": "official_split"})
    labels = extracted.merge(label_rows, on=["query_id", "query"], how="inner")
    labels["esci_label"] = labels["esci_label"].astype(str)
    labels["grade"] = labels["esci_label"].map({"E": 3, "S": 2, "C": 1, "I": 0})
    labels = labels[labels["grade"].notna()].copy()

    # Product content joined from official product table.
    product_ids = set(labels["product_id"])
    products = pd.read_parquet(
        PRODUCTS,
        columns=["product_id", "product_title", "product_description", "product_bullet_point", "product_brand", "product_color"],
    )
    products["product_id"] = products["product_id"].astype(str)
    products = products[products["product_id"].isin(product_ids)].drop_duplicates("product_id")
    labels = labels.merge(products, on="product_id", how="left")
    labels["product_title"] = labels["product_title"].fillna("")
    labels["product_description"] = labels["product_description"].fillna("")
    labels["product_bullet_point"] = labels["product_bullet_point"].fillna("")
    labels["product_brand"] = labels["product_brand"].fillna("")
    labels["product_color"] = labels["product_color"].fillna("")
    labels["has_product_text"] = (labels["product_title"].str.len() > 0).astype(int)

    # Query family split: official test is final test, excluding Benchmark v2 queries.
    official_test_ids = set(labels.loc[labels["official_split"] == "test", "query_id"])
    excluded_benchmark_ids = official_test_ids & benchmark_ids
    labels["split"] = labels["official_split"]
    labels.loc[
        (labels["official_split"] == "test")
        & (labels["query_id"].isin(benchmark_ids) | labels["query"].str.strip().str.lower().isin(benchmark_texts)),
        "split",
    ] = "excluded_benchmark_v2"
    labels.loc[(labels["official_split"] == "train") & labels["query_id"].map(
        lambda value: int(hashlib.md5(str(value).encode()).hexdigest(), 16) % 10 == 9
    ), "split"] = "val"
    labels = labels[labels["split"] != "excluded_benchmark_v2"].copy()

    # Query-derived preference is explicitly not historical memory.
    labels["preference_source"] = "query_derived_proxy"
    labels["task_type"] = "esci_query_derived_preference_proxy"
    evidence = labels.apply(
        lambda row: _evidence_fields(
            row["product_title"], row["product_description"],
            row["product_bullet_point"], row["product_color"],
        ),
        axis=1,
    )
    evidence = pd.DataFrame(list(evidence))
    labels = pd.concat([labels.reset_index(drop=True), evidence.reset_index(drop=True)], axis=1)

    # Keep only groups with >=2 judged candidates and at least one grade difference.
    group_sizes = labels.groupby("query_id")["product_id"].nunique()
    valid_ids = set(group_sizes[group_sizes >= 2].index)
    labels = labels[labels["query_id"].isin(valid_ids)].copy()
    labels["group_id"] = "esci:" + labels["query_id"]
    labels = labels.sort_values(["split", "query_id", "grade"], ascending=[True, True, False]).reset_index(drop=True)
    labels.to_parquet(OUT / "esci_preference_dataset.parquet", index=False, compression="zstd")

    all_attrs = list(ATTR_RE)
    query_meta = labels[[
        "query_id", "query", "base_query", "attribute", "polarity", "clause",
        "extraction_rule", "split", "official_split",
    ]].drop_duplicates("query_id").copy()
    query_meta.to_json(OUT / "query_preferences.jsonl", orient="records", lines=True, force_ascii=False)

    counts = {
        "queries_total_extractable": int(extracted["query_id"].nunique()),
        "queries_used": int(labels["query_id"].nunique()),
        "candidate_rows": int(len(labels)),
        "train_queries": int(query_meta[query_meta["split"] == "train"]["query_id"].nunique()),
        "val_queries": int(query_meta[query_meta["split"] == "val"]["query_id"].nunique()),
        "test_queries": int(query_meta[query_meta["split"] == "test"]["query_id"].nunique()),
        "train_rows": int((labels["split"] == "train").sum()),
        "val_rows": int((labels["split"] == "val").sum()),
        "test_rows": int((labels["split"] == "test").sum()),
        "positive_preferences": int((labels["polarity"] == "positive").sum()),
        "negative_preferences": int((labels["polarity"] == "negative").sum()),
        "product_text_coverage": round(float(labels["has_product_text"].mean()), 6),
        "label_distribution": {key: int(value) for key, value in labels["esci_label"].value_counts().items()},
        "excluded_benchmark_v2_queries": int(len(excluded_benchmark_ids)),
    }
    manifest = {
        "schema": "agentrecx.public_preference_ranking_v1.manifest.v1",
        "source": {
            "dataset": "Amazon ESCI",
            "examples": str(EXAMPLES),
            "products": str(PRODUCTS),
            "small_version": 1,
            "product_locale": "us",
            "official_splits": ["train", "test"],
            "license": "See amazon-science/esci-data; Amazon ESCI terms apply.",
        },
        "preference_source": "query_derived_proxy",
        "not_historical_memory": True,
        "candidate_pool": "official ESCI judged products only; no BGE-retrieved unjudged products",
        "label_policy": {
            "official_grades": {"E": 3, "S": 2, "C": 1, "I": 0},
            "complement_policy": "C kept as grade 1; never collapsed into S",
            "no_unjudged_negatives": True,
        },
        "split_policy": {
            "train_val": "official ESCI train, query_id hash 90/10",
            "test": "official ESCI test, excluding Benchmark v2 query IDs/texts",
            "grouping": "query_id; all variants of a query stay in one split",
        },
        "counts": counts,
        "seed": 42,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
