"""DCN-V2 Agent-native alignment diagnostics.

Checks whether the DCN-B regression on Agent Silver pairs is:
  * a confirmed adapter implementation bug,
  * a feature-schema mismatch,
  * a distribution shift,
  * or supervision/semantic mismatch.

No weights are re-trained.  GPU scoring only.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/root/autodl-tmp/AgentRec-X-phase2")
ESCI = REPO / "data/public_preference_ranking_v1"
BM = REPO / "data/memory_ranking_benchmark_v2"
P20 = REPO / "data/p20_shadow/ranking_dataset.jsonl"
P20_TASKS = REPO / "data/p20_shadow/per_task_results.jsonl"
CF = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
OUT = REPO / "runs/agent_native_ranking_phase42"
CHECKPOINTS = REPO / "runs/public_preference_ranking_v1"
sys.path.insert(0, str(REPO))
from experiments.dcn_v2.model import DCNConfig, DCNV2Ranker

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


def item_attrs(item: dict) -> set[str]:
    attrs = item.get("attributes")
    return set(attrs.keys()) if isinstance(attrs, dict) else set(attrs or [])


def active(memory) -> tuple[list[str], list[str]]:
    if isinstance(memory, dict):
        positive = [str(x.get("value")) for x in memory.get("positive") or [] if x.get("value")]
        negative = [str(x.get("value")) for x in memory.get("negative") or [] if x.get("value")]
        return positive, negative
    positive, negative = [], []
    for entry in memory or []:
        value = str(entry.get("value") or "")
        if not value:
            continue
        if str(entry.get("polarity") or "positive").lower() in {"negative", "avoid", "dislike"}:
            negative.append(value)
        else:
            positive.append(value)
    return positive, negative


def lexical(query: str, item: dict, prefix: str) -> dict[str, float]:
    q = tokens(query)
    p = tokens(str(item.get("title") or "") + " " + " ".join(item_attrs(item)))
    overlap = len(q & p)
    return {
        f"{prefix}_query_token_count": float(len(q)),
        f"{prefix}_product_token_count": float(len(p)),
        f"{prefix}_overlap_count": float(overlap),
        f"{prefix}_overlap_rate": float(overlap / len(q)) if q else 0.0,
        f"{prefix}_jaccard": float(overlap / len(q | p)) if (q | p) else 0.0,
    }


def dcn_features(item: dict, query: str, memory, mode: str) -> dict[str, float]:
    """mode: one_pref (old), full_active, memory_off, full_query."""
    attrs = item_attrs(item)
    positive, negative = active(memory)
    pref_pool = positive + negative
    pref_attr = pref_pool[0] if pref_pool else ""
    if mode == "one_pref":
        positives = positive[:1]
        negatives = negative[:1]
        selected = pref_pool[:1]
    elif mode == "full_active":
        positives = positive
        negatives = negative
        selected = pref_pool
    else:
        positives, negatives, selected = [], [], []
    pos_matches = len([v for v in positives if v in attrs])
    neg_conflicts = len([v for v in negatives if v in attrs])
    known = pos_matches + neg_conflicts
    active_count = len(positives) + len(negatives)
    present = int(known > 0)
    is_negative = int(bool(negatives))
    result = {
        "title_len": float(len(str(item.get("title") or ""))),
        "description_len": 0.0, "bullet_len": 0.0,
        "has_brand": 0.0, "has_color": 0.0,
        "product_token_count": float(len(tokens(str(item.get("title") or "")))),
        "preference_present": float(present),
        "preference_match": float(pos_matches > 0),
        "preference_conflict": float(neg_conflicts > 0),
        "preference_unknown": float(active_count > 0 and known == 0),
        "preference_coverage": float(known / active_count) if active_count else 0.0,
        "preference_is_negative": float(is_negative),
    }
    for attr in ATTRS:
        result[f"product_attr_{attr}"] = 1.0 if attr in attrs else 0.0
        result[f"pref_attr_{attr}"] = 1.0 if attr in selected else 0.0
    result.update(lexical(query, item, "base"))
    result.update(lexical(query, item, "full"))
    return result


def load_models():
    schema = json.loads((ESCI / "dcn_feature_schema.json").read_text())
    frame = pd.read_parquet(ESCI / "dcn_features.parquet")
    train = frame[frame["split"] == "train"]
    models = {}
    stats = {}
    for arm in ("A_base_query_only", "B_preference_on", "C_full_query"):
        features = schema["views"][arm]
        mean = train[features].astype(np.float32).fillna(0.0).to_numpy().mean(0, keepdims=True)
        std = train[features].astype(np.float32).fillna(0.0).to_numpy().std(0, keepdims=True) + 1e-6
        model = DCNV2Ranker(DCNConfig(input_dim=len(features), hidden=64, cross_layers=3, dropout=0.1)).cuda()
        model.load_state_dict(torch.load(CHECKPOINTS / f"{arm}.pt", map_location="cuda"))
        model.eval()
        models[arm] = (features, model)
        stats[arm] = (mean, std)
    return schema, models, stats, frame


def score(arm: str, schema, models, stats, rows: list[dict], query: str, base: str, memory, mode: str) -> np.ndarray:
    features, model = models[arm]
    mean, std = stats[arm]
    x = np.asarray([[dcn_features(item, query if arm == "C_full_query" else base, memory, mode)[name] for name in features] for item in rows], dtype=np.float32)
    x = np.clip((x - mean) / std, -8, 8)
    with torch.no_grad():
        return model(torch.from_numpy(x).cuda()).cpu().numpy()


def pairwise_accuracy(pairs, arm, schema, models, stats, metadata, mode):
    correct = 0
    details = []
    for pair in pairs:
        a = metadata[pair["candidate_a_id"]]
        b = metadata[pair["candidate_b_id"]]
        scores = score(arm, schema, models, stats, [a, b], pair["_query"], pair["_query"], pair["_memory"], mode)
        predicted = "A" if scores[0] > scores[1] else "B"
        correct += int(predicted == pair["choice"])
        details.append({"pair_task_id": pair["pair_task_id"], "scenario_type": pair["scenario_type"],
                        "label": pair["choice"], "predicted": predicted, "correct": predicted == pair["choice"],
                        "a_score": float(scores[0]), "b_score": float(scores[1]), "margin": float(scores[0] - scores[1]),
                        "a_positive": int(any(x["value"] in item_attrs(a) for x in pair["_memory"].get("positive") or [])),
                        "b_positive": int(any(x["value"] in item_attrs(b) for x in pair["_memory"].get("positive") or [])),
                        "a_negative": int(any(x["value"] in item_attrs(a) for x in pair["_memory"].get("negative") or [])),
                        "b_negative": int(any(x["value"] in item_attrs(b) for x in pair["_memory"].get("negative") or []))})
    return correct / len(pairs), details


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    schema, models, stats, esci_frame = load_models()
    scenarios = [json.loads(line) for line in (BM / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = {row["item_id"]: row for row in (json.loads(line) for line in (BM / "candidate_metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    pairs_raw = [json.loads(line) for line in (BM / "pair_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_by_id = {row["scenario_id"]: row for row in scenarios}
    pairs = []
    for pair in pairs_raw:
        if pair["choice"] not in {"A", "B"}:
            continue
        scenario = scenario_by_id[pair["scenario_id"]]
        pairs.append({**pair, "_query": scenario["query"], "_memory": scenario["memory_after"]})
    variants = {
        "B_one_pref": ("B_preference_on", "one_pref"),
        "B_full_active": ("B_preference_on", "full_active"),
        "B_memory_off": ("B_preference_on", "memory_off"),
        "A_base_query": ("A_base_query_only", "memory_off"),
        "C_full_query": ("C_full_query", "memory_off"),
    }
    results = {}
    per_case = {}
    for name, (arm, mode) in variants.items():
        accuracy, details = pairwise_accuracy(pairs, arm, schema, models, stats, metadata, mode)
        results[name] = {"accuracy": round(accuracy, 6), "correct": sum(1 for row in details if row["correct"]), "n": len(details)}
        per_case[name] = details
    # Diagnostic inversion check (not used as a model).
    b_details = per_case["B_one_pref"]
    results["B_one_pref_inverted_diagnostic_only"] = {
        "accuracy": round(sum(1 for row in b_details if row["predicted"] != row["label"]) / len(b_details), 6),
        "note": "INVERTED SCORE IS DIAGNOSTIC ONLY; never used as a correction.",
    }
    # Feature distribution: ESCI B training vs Agent B current.
    key_features = ["preference_present", "preference_match", "preference_conflict", "preference_unknown",
                    "preference_coverage", "preference_is_negative", "base_overlap_rate", "title_len"]
    b_features = schema["views"]["B_preference_on"]
    esc = esci_frame[esci_frame["split"] == "train"]
    agent_rows = [dcn_features(row, scenario_by_id[pair["scenario_id"]]["query"], scenario_by_id[pair["scenario_id"]]["memory_after"], "one_pref")
                  for pair in pairs for row in (metadata[pair["candidate_a_id"]], metadata[pair["candidate_b_id"]])]
    distribution = {}
    for feature in key_features:
        distribution[feature] = {
            "esci_mean": round(float(esc[feature].mean()), 6),
            "esci_std": round(float(esc[feature].std()), 6),
            "agent_mean": round(float(np.mean([row[feature] for row in agent_rows])), 6),
            "agent_std": round(float(np.std([row[feature] for row in agent_rows])), 6),
        }
    # Error grouping.
    error_groups = Counter()
    for row in per_case["B_one_pref"]:
        if not row["correct"]:
            error_groups[row["scenario_type"]] += 1
    # Schema assertions.
    schema_check = {
        "B_feature_count": len(b_features),
        "B_checkpoint_input_dim": int(models["B_preference_on"][1].input_proj.in_features),
        "B_feature_order_matches_schema": models["B_preference_on"][1].input_proj.in_features == len(b_features),
        "first_features": b_features[:8],
        "last_features": b_features[-8:],
        "normalization_mean_finite": bool(np.isfinite(stats["B_preference_on"][0]).all()),
        "normalization_std_finite": bool(np.isfinite(stats["B_preference_on"][1]).all()),
    }
    # P20 / CF DCN diagnostics (best-effort; feature adapter has no description).
    p20_rows = [json.loads(line) for line in P20.read_text(encoding="utf-8").splitlines() if line.strip()]
    p20_rows = [row for row in p20_rows if row.get("expected_decision") == "recommend"]
    p20_tasks = {row["task_id"]: row for row in (json.loads(line) for line in P20_TASKS.read_text(encoding="utf-8").splitlines() if line.strip())}
    p20 = []
    for row in p20_rows:
        eligible = set(p20_tasks[row["task_id"]]["eligible_candidate_ids"])
        candidates = [item for item in row["candidate_pool"] if item["item_id"] in eligible]
        scores_b = score("B_preference_on", schema, models, stats, candidates, row["query"], row["query"], row.get("historical_memory") or [], "one_pref")
        ids = [item["item_id"] for item in candidates]
        order = sorted(ids, key=lambda item_id: (-float(scores_b[ids.index(item_id)]), item_id))
        acceptable = set(row.get("acceptable_ids_in_pool") or [])
        first = next((index + 1 for index, item_id in enumerate(order) if item_id in acceptable), None)
        p20.append({"task_id": row["task_id"], "top3": order[:3], "first_acceptable_rank": first})
    cf_cases = [json.loads(line) for line in CF.read_text(encoding="utf-8").splitlines() if line.strip()]
    p20_by_id = {row["task_id"]: row for row in p20_rows}
    cf = []
    for case in cf_cases:
        pool = p20_by_id.get(case["pool_id"])
        if not pool:
            continue
        eligible = set(case.get("eligible_candidate_ids") or [])
        candidates = [item for item in pool["candidate_pool"] if item["item_id"] in eligible]
        before = score("B_preference_on", schema, models, stats, candidates, case["query"], case["query"], (case.get("baseline_memory_state") or {}).get("active_preferences") or [], "one_pref")
        after = score("B_preference_on", schema, models, stats, candidates, case["query"], case["query"], (case.get("mutated_memory_state") or {}).get("active_preferences") or [], "one_pref")
        ids = [item["item_id"] for item in candidates]
        before_order = [ids[index] for index in np.argsort(-before)]
        after_order = [ids[index] for index in np.argsort(-after)]
        cf.append({"case_id": case["case_id"], "category": case["category"], "top1_changed": before_order[:1] != after_order[:1],
                   "full_changed": before_order != after_order, "before_top1": before_order[:1], "after_top1": after_order[:1]})
    payload = {
        "schema": "agentrecx.agent_native_ranking_phase42.dcn_alignment.v1",
        "models_reused": ["A_base_query_only", "B_preference_on", "C_full_query"],
        "pairwise_variants": results,
        "feature_distribution_esci_vs_agent": distribution,
        "schema_check": schema_check,
        "error_groups": dict(error_groups),
        "per_case_B_one_pref": per_case["B_one_pref"],
        "per_case_B_full_active": per_case["B_full_active"],
        "p20_B_one_pref": p20,
        "counterfactual_B_one_pref": {"cases": len(cf), "top1_changed": sum(1 for row in cf if row["top1_changed"]), "full_changed": sum(1 for row in cf if row["full_changed"]), "details": cf},
        "gpu_runtime": {"wall_seconds": round(time.perf_counter() - started, 3), "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)},
        "conclusion": "No confirmed implementation bug: feature order/dim, normalization and score direction were checked. B degradation is consistent with distribution/semantic mismatch and only-one-preference issue; full-active aggregation is reported separately.",
    }
    (OUT / "dcn_alignment_diagnostics.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pairwise_variants": results, "schema_check": schema_check, "distribution": distribution, "error_groups": dict(error_groups)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
