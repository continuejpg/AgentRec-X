"""AgentRec-X Shadow adapter for the ESCI-trained DCN-V2 arms.

The adapter maps the real Agent candidate pool into the ESCI feature schema.  Missing
Agent metadata is represented as zero/UNKNOWN, never as confirmed negative evidence.
No production component is modified; this is shadow-only.
"""

from __future__ import annotations

import json
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
OUT = REPO / "runs/public_preference_ranking_v1"
P20 = REPO / "data/p20_shadow/ranking_dataset.jsonl"
P20_TASKS = REPO / "data/p20_shadow/per_task_results.jsonl"
BM = REPO / "data/memory_ranking_benchmark_v2"
CF = REPO / "data/p20_shadow/counterfactual_cases.jsonl"
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


def overlap(query: str, item: dict, prefix: str) -> dict[str, float]:
    q = tokens(query)
    title = str(item.get("title") or "")
    attrs = " ".join(str(attr) for attr in item_attrs(item))
    p = tokens(title + " " + attrs)
    intersection = len(q & p)
    return {
        f"{prefix}_query_token_count": float(len(q)),
        f"{prefix}_product_token_count": float(len(p)),
        f"{prefix}_overlap_count": float(intersection),
        f"{prefix}_overlap_rate": float(intersection / len(q)) if q else 0.0,
        f"{prefix}_jaccard": float(intersection / len(q | p)) if (q | p) else 0.0,
    }


def item_attrs(item: dict) -> set[str]:
    """Agent item attributes are canonical keys; missing keys are never decoded as absent."""
    if isinstance(item.get("attributes"), dict):
        return set(item["attributes"].keys())
    if isinstance(item.get("attributes"), list):
        return set(item["attributes"])
    return set()


def candidate_features(item: dict, query: str, base_query: str, memory: dict) -> dict[str, float]:
    attrs = item_attrs(item)
    positive = [str(entry.get("value")) for entry in memory.get("positive") or [] if entry.get("value")]
    negative = [str(entry.get("value")) for entry in memory.get("negative") or [] if entry.get("value")]
    pref_attr = (positive + negative + [""])[0]
    pref_present = int(pref_attr in attrs)
    pref_positive = int(pref_attr in positive)
    pref_negative = int(pref_attr in negative)
    result = {
        "title_len": float(len(str(item.get("title") or ""))),
        "description_len": 0.0,
        "bullet_len": 0.0,
        "has_brand": 1.0 if item.get("brand") else 0.0,
        "has_color": 1.0 if item.get("color") else 0.0,
        "product_token_count": float(len(tokens(str(item.get("title") or "")))),
        "preference_present": float(pref_present),
        "preference_match": float(pref_positive and pref_present),
        "preference_conflict": float(pref_negative and pref_present),
        "preference_unknown": float(not pref_present),
        "preference_coverage": 1.0 if pref_present else 0.0,
        "preference_is_negative": 1.0 if pref_negative else 0.0,
    }
    for attr in ATTRS:
        result[f"product_attr_{attr}"] = 1.0 if attr in attrs else 0.0
        if f"pref_attr_{attr}" in FEATURE_NAMES:
            result[f"pref_attr_{attr}"] = 1.0 if attr == pref_attr else 0.0
    result.update(overlap(base_query, item, "base"))
    result.update(overlap(query, item, "full"))
    return result


def memory_from_entries(entries: list[dict]) -> dict:
    return {
        "positive": [entry for entry in entries if str(entry.get("polarity") or "positive").lower() not in {"negative", "avoid", "dislike"}],
        "negative": [entry for entry in entries if str(entry.get("polarity") or "").lower() in {"negative", "avoid", "dislike"}],
    }


FEATURE_NAMES: list[str] = []
MODELS = {}
MEANS = {}
STDS = {}


def load_models():
    global FEATURE_NAMES, MODELS, MEANS, STDS
    schema = json.loads((ESCI / "dcn_feature_schema.json").read_text())
    frame = pd.read_parquet(ESCI / "dcn_features.parquet")
    train = frame[frame["split"] == "train"]
    for arm in ("A_base_query_only", "B_preference_on", "C_full_query"):
        features = schema["views"][arm]
        if arm == "B_preference_on":
            FEATURE_NAMES = features
        mean = train[features].astype(np.float32).fillna(0.0).to_numpy().mean(axis=0, keepdims=True)
        std = train[features].astype(np.float32).fillna(0.0).to_numpy().std(axis=0, keepdims=True) + 1e-6
        model = DCNV2Ranker(DCNConfig(input_dim=len(features), hidden=64, cross_layers=3, dropout=0.1)).cuda()
        model.load_state_dict(torch.load(OUT / f"{arm}.pt", map_location="cuda"))
        model.eval()
        MODELS[arm] = (features, model)
        MEANS[arm] = mean
        STDS[arm] = std


def score_pool(arm: str, rows: list[dict], query: str, base_query: str, memory: dict) -> tuple[list[str], dict]:
    features, model = MODELS[arm]
    x = np.asarray([
        [candidate_features(item, query, base_query, memory)[name] for name in features]
        for item in rows
    ], dtype=np.float32)
    x = np.clip((x - MEANS[arm]) / STDS[arm], -8, 8)
    with torch.no_grad():
        scores = model(torch.from_numpy(x).cuda()).cpu().numpy()
    ids = [str(item["item_id"]) for item in rows]
    score_map = {item_id: float(score) for item_id, score in zip(ids, scores)}
    order = sorted(ids, key=lambda item_id: (-score_map[item_id], item_id))
    return order, score_map


def metrics(ranking: list[str], acceptable: set[str]) -> dict:
    first = next((index + 1 for index, item_id in enumerate(ranking) if item_id in acceptable), None)
    return {
        "hit@1": bool(first == 1), "hit@3": bool(first and first <= 3), "hit@5": bool(first and first <= 5),
        "mrr@10": round(1.0 / first, 6) if first and first <= 10 else 0.0, "first_acceptable_rank": first,
        "top1": ranking[:1], "top3": ranking[:3],
    }


def load_p20():
    rows = [json.loads(line) for line in P20.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {row["task_id"]: row for row in (json.loads(line) for line in P20_TASKS.read_text(encoding="utf-8").splitlines() if line.strip())}
    return [row for row in rows if row.get("expected_decision") == "recommend"], tasks


def load_benchmark():
    scenarios = [json.loads(line) for line in (BM / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = {row["scenario_id"]: row for row in (json.loads(line) for line in (BM / "pair_tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    metadata = {row["item_id"]: row for row in (json.loads(line) for line in (BM / "candidate_metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    return scenarios, pairs, metadata


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load_models()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    # P20 shadow.
    p20_rows, p20_tasks = load_p20()
    p20 = []
    for row in p20_rows:
        task = p20_tasks[row["task_id"]]
        pool = row["candidate_pool"]
        eligible = set(task["eligible_candidate_ids"])
        acceptable = set(row.get("acceptable_ids_in_pool") or [])
        memory = memory_from_entries(row.get("historical_memory") or [])
        item = {"task_id": row["task_id"], "query": row["query"], "eligible": sorted(eligible), "acceptable": sorted(acceptable)}
        for arm in ("A_base_query_only", "B_preference_on", "C_full_query"):
            started = time.perf_counter()
            ranking, scores = score_pool(arm, pool, row["query"], row["query"], memory)
            latencies.append((time.perf_counter() - started) * 1000)
            ranking = [item_id for item_id in ranking if item_id in eligible]
            item[arm] = metrics(ranking, acceptable)
            item[f"{arm}_scores"] = {item_id: round(value, 6) for item_id, value in scores.items() if item_id in eligible}
        p20.append(item)
    p20_agg = {arm: {
        "hit@1": sum(1 for row in p20 if row[arm]["hit@1"]),
        "hit@3": sum(1 for row in p20 if row[arm]["hit@3"]),
        "hit@5": sum(1 for row in p20 if row[arm]["hit@5"]),
        "mrr@10": round(statistics.mean(row[arm]["mrr@10"] for row in p20), 6),
    } for arm in ("A_base_query_only", "B_preference_on", "C_full_query")}

    # Benchmark v2 shadow.
    scenarios, pairs, metadata = load_benchmark()
    bm = []
    pair_accuracy = {arm: {"correct": 0, "n": 0} for arm in ("A_base_query_only", "B_preference_on", "C_full_query")}
    for scenario in scenarios:
        pool = [metadata[item_id] for item_id in scenario["candidate_pool_ids"]]
        memory = scenario["memory_after"]
        pair = pairs.get(scenario["scenario_id"])
        item = {"scenario_id": scenario["scenario_id"], "scenario_type": scenario["scenario_type"], "split": scenario["split"], "requires_clarification": scenario["requires_clarification"]}
        for arm in ("A_base_query_only", "B_preference_on", "C_full_query"):
            started = time.perf_counter()
            ranking, scores = score_pool(arm, pool, scenario["query"], scenario["query"], memory)
            latencies.append((time.perf_counter() - started) * 1000)
            item[arm] = {"ranking": ranking[:20], "scores": {item_id: round(value, 6) for item_id, value in scores.items()}}
            if pair and pair["choice"] in {"A", "B"}:
                a, b = pair["candidate_a_id"], pair["candidate_b_id"]
                predicted = a if scores[a] > scores[b] else b
                pair_accuracy[arm]["correct"] += int(predicted == (a if pair["choice"] == "A" else b))
                pair_accuracy[arm]["n"] += 1
        bm.append(item)
    bm_pair = {arm: {"accuracy": round(values["correct"] / values["n"], 6) if values["n"] else None, "n": values["n"]} for arm, values in pair_accuracy.items()}
    bm_top1 = {arm: sum(1 for row in bm if row[arm]["ranking"][0] != row["A_base_query_only"]["ranking"][0]) for arm in ("B_preference_on", "C_full_query")}

    # Counterfactual cases: score baseline/mutated with Preference ON; no labels / no gold.
    ranking_rows = {row["task_id"]: row for row in p20_rows}
    cases = [json.loads(line) for line in CF.read_text(encoding="utf-8").splitlines() if line.strip()]
    cf = []
    for case in cases:
        row = ranking_rows.get(case["pool_id"])
        if row is None:
            continue
        pool = row["candidate_pool"]
        eligible = set(case.get("eligible_candidate_ids") or [])
        baseline = memory_from_entries((case.get("baseline_memory_state") or {}).get("active_preferences") or [])
        mutated = memory_from_entries((case.get("mutated_memory_state") or {}).get("active_preferences") or [])
        before, _ = score_pool("B_preference_on", pool, case["query"], case["query"], baseline)
        after, _ = score_pool("B_preference_on", pool, case["query"], case["query"], mutated)
        before = [item_id for item_id in before if item_id in eligible]
        after = [item_id for item_id in after if item_id in eligible]
        cf.append({
            "case_id": case["case_id"], "category": case["category"],
            "before_top1": before[:1], "after_top1": after[:1],
            "top1_changed": before[:1] != after[:1],
            "full_changed": before != after,
            "requires_clarification": bool(case.get("requires_clarification")),
            "top1_gold_allowed": not bool(case.get("requires_clarification")),
        })

    payload = {
        "schema": "agentrecx.public_preference_ranking_v1.agent_shadow.v1",
        "adapter": "ESCI-trained DCN; Agent title/attrs mapped to ESCI feature schema; missing description/bullets set to zero/UNKNOWN",
        "feature_source": "ESCI train mean/std; no Agent labels used",
        "p20": {"tasks": len(p20), "aggregate": p20_agg, "details": p20},
        "benchmark_v2": {"scenarios": len(bm), "silver_pairwise": bm_pair, "top1_changed_vs_A": bm_top1, "details": bm},
        "counterfactual": {
            "cases": len(cf),
            "top1_changed": sum(1 for row in cf if row["top1_changed"]),
            "full_changed": sum(1 for row in cf if row["full_changed"]),
            "requires_clarification": sum(1 for row in cf if row["requires_clarification"]),
            "details": cf,
        },
        "latency_ms": {
            "mean": round(float(statistics.mean(latencies)), 4),
            "p50": round(float(statistics.median(latencies)), 4),
            "max": round(float(max(latencies)), 4),
            "scope": "one 20-candidate pool DCN forward, GPU; includes Python feature construction",
        },
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
    }
    (OUT / "agent_shadow.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("p20", "benchmark_v2", "counterfactual", "latency_ms", "gpu_peak_memory_mb")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
