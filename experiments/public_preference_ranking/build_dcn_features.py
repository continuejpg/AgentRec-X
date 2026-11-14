"""Build DCN-V2 features for the ESCI query-derived preference proxy.

Three feature views are declared, never mixed:
  A base_query_only    = base query lexical + product
  B preference_on      = base query lexical + product + structured preference
  C full_query         = original full query lexical + product (no structured preference)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/public_preference_ranking_v1"
DATA = OUT / "esci_preference_dataset.parquet"
ATTRS = (
    "wireless", "wired", "bluetooth", "microphone", "rechargeable", "noise_cancelling",
    "waterproof", "over_ear", "on_ear", "in_ear", "gaming", "mechanical", "portable",
    "foldable", "usb", "hdmi", "4k", "remote", "led", "rgb", "touch", "ergonomic", "silent",
    "lightweight", "compact", "durable", "adjustable", "fast_charging", "long_battery",
    "professional", "travel", "running", "workout", "office", "home", "kids", "women", "men",
    "black", "white", "blue", "red", "pink", "silver", "gold", "small", "large", "mini",
    "premium", "cheap", "affordable",
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "for", "with", "without", "and", "or", "of", "to", "in", "on", "by", "is", "are"}


def tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(str(text or "").lower()) if token not in STOP}


def overlap(query: str, product_text: str) -> dict[str, float]:
    q = tokens(query)
    p = tokens(product_text)
    intersection = len(q & p)
    return {
        "query_token_count": float(len(q)),
        "product_token_count": float(len(p)),
        "overlap_count": float(intersection),
        "overlap_rate": float(intersection / len(q)) if q else 0.0,
        "jaccard": float(intersection / len(q | p)) if (q | p) else 0.0,
    }


def row_features(row: pd.Series) -> dict[str, Any]:
    title = str(row["product_title"] or "")
    description = str(row["product_description"] or "")
    bullets = str(row["product_bullet_point"] or "")
    brand = str(row["product_brand"] or "")
    color = str(row["product_color"] or "")
    combined = " ".join((title, description, bullets, color))
    base = overlap(str(row["base_query"]), title + " " + description + " " + bullets)
    full = overlap(str(row["query"]), title + " " + description + " " + bullets)
    attrs = {attr: int(bool(row.get(f"evidence_{attr}", False))) for attr in ATTRS}
    pref_attr = str(row["attribute"])
    pref_present = int(bool(row.get(f"evidence_{pref_attr}", False)))
    positive = str(row["polarity"]) == "positive"
    preference_match = int(positive and pref_present)
    preference_conflict = int((not positive) and pref_present)
    preference_unknown = int(not pref_present)
    result = {
        "group_id": str(row["group_id"]),
        "query_id": str(row["query_id"]),
        "split": str(row["split"]),
        "product_id": str(row["product_id"]),
        "esci_label": str(row["esci_label"]),
        "grade": float(row["grade"]),
        "query": str(row["query"]),
        "base_query": str(row["base_query"]),
        "preference_attr": pref_attr,
        "preference_polarity": "positive" if positive else "negative",
        "preference_clause": str(row["clause"]),
        "preference_source": "query_derived_proxy",
        "title_len": float(len(title)),
        "description_len": float(len(description)),
        "bullet_len": float(len(bullets)),
        "has_brand": 1.0 if brand else 0.0,
        "has_color": 1.0 if color else 0.0,
        "product_token_count": float(len(tokens(combined))),
        "preference_present": float(pref_present),
        "preference_match": float(preference_match),
        "preference_conflict": float(preference_conflict),
        "preference_unknown": float(preference_unknown),
        "preference_coverage": float(1.0 if pref_present else 0.0),
        "preference_is_negative": 0.0 if positive else 1.0,
    }
    for prefix, values in (("base", base), ("full", full)):
        for key, value in values.items():
            result[f"{prefix}_{key}"] = value
    for attr in ATTRS:
        result[f"product_attr_{attr}"] = float(attrs[attr])
    result[f"pref_attr_{pref_attr}"] = 1.0
    return result


def build() -> dict[str, Any]:
    frame = pd.read_parquet(DATA)
    rows = [row_features(row) for _, row in frame.iterrows()]
    features = pd.DataFrame(rows)
    features.to_parquet(OUT / "dcn_features.parquet", index=False, compression="zstd")
    id_cols = {"group_id", "query_id", "split", "product_id", "esci_label", "grade", "query", "base_query", "preference_attr", "preference_polarity", "preference_clause", "preference_source"}
    product_features = ["title_len", "description_len", "bullet_len", "has_brand", "has_color",
                        "product_token_count"] + [f"product_attr_{attr}" for attr in ATTRS]
    base_features = [name for name in features.columns if name.startswith("base_") and name != "base_query"]
    full_features = [name for name in features.columns if name.startswith("full_")]
    preference_features = ["preference_present", "preference_match", "preference_conflict",
                           "preference_unknown", "preference_coverage", "preference_is_negative"] + [
        f"pref_attr_{attr}" for attr in ATTRS if f"pref_attr_{attr}" in features.columns
    ]
    schema = {
        "schema": "agentrecx.public_preference_ranking_v1.dcn_feature_schema.v1",
        "id_columns": sorted(id_cols),
        "label": "grade (E=3, S=2, C=1, I=0)",
        "views": {
            "A_base_query_only": base_features + product_features,
            "B_preference_on": base_features + product_features + preference_features,
            "C_full_query": full_features + product_features,
        },
        "base_query_features": base_features,
        "full_query_features": full_features,
        "product_features": product_features,
        "preference_features": preference_features,
        "leakage_controls": [
            "no query_id/product_id input",
            "labels only official ESCI E/S/C/I",
            "A never receives preference features or full-query lexical features",
            "C receives full query but never structured preference features",
            "preference is query_derived_proxy, not historical memory",
        ],
    }
    (OUT / "dcn_feature_schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "rows": int(len(features)),
        "groups": int(features["group_id"].nunique()),
        "queries": int(features["query_id"].nunique()),
        "splits": {key: int(value) for key, value in features.groupby("split")["group_id"].nunique().items()},
        "feature_views": {key: len(value) for key, value in schema["views"].items()},
        "positive_preferences": int((features["preference_polarity"] == "positive").sum()),
        "negative_preferences": int((features["preference_polarity"] == "negative").sum()),
        "preference_present_rate": round(float(features["preference_present"].mean()), 6),
        "preference_unknown_rate": round(float(features["preference_unknown"].mean()), 6),
        "label_distribution": {key: int(value) for key, value in features["esci_label"].value_counts().items()},
    }
    (OUT / "dcn_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
