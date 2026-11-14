"""Phase D shadow: P20 fixed-pool ranking with existing Scorer vs learned neural ranker.

The neural model was trained on rating histories.  Event-Sourced active preferences
are converted into a **pseudo-history** (one synthetic item per preference, rating 5
for positive / 1 for negative) and a matching memory feature vector.  This is a
shadow compatibility probe, not proof that the model understands explicit memory.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

DATA = Path("/root/autodl-tmp/neural_memory_v1")
DATASET = Path("/root/autodl-tmp/AgentRec-X-phase2/data/p20_shadow/ranking_dataset.jsonl")
PHASE1 = Path("/root/autodl-tmp/AgentRec-X-phase2/data/p20_shadow/per_task_results.jsonl")
OUT = Path("/root/autodl-tmp/AgentRec-X-phase2/runs/neural_memory_v1")
MS = 86_400_000.0


def _load():
    item_features = np.load(DATA / "item_features.npy").astype(np.float32)
    item_ids = json.loads((DATA / "item_ids.json").read_text())
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    item_names = json.loads((DATA / "item_feature_names.json").read_text())
    memory_names = json.loads((DATA / "memory_feature_names.json").read_text())
    raw_rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_rows = [row for row in raw_rows if row["expected_decision"] == "recommend"]
    phase1_rows = {
        row["task_id"]: row
        for row in (
            json.loads(line)
            for line in PHASE1.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return item_features, item_index, item_names, memory_names, raw_rows, phase1_rows


def _attr_indices(item_names, memory_names):
    attrs = ["over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth",
             "microphone", "rechargeable", "noise_cancelling", "waterproof", "foldable",
             "portable", "gaming", "mechanical", "silent", "ergonomic", "touch", "rgb",
             "led", "usb", "hdmi", "4k", "remote"]
    return (
        [item_names.index(f"meta_attr_{attr}") for attr in attrs],
        [memory_names.index(f"h_high_attr_{attr}") for attr in attrs],
        [memory_names.index(f"h_low_attr_{attr}") for attr in attrs],
        attrs,
    )


def _pseudo_history(preferences, item_dim, memory_dim, item_names, memory_names, attr_indices, attrs, item_features):
    """Return feature tensors for one event-sourced memory state."""
    n = len(preferences)
    hist_features = np.zeros((1, max(n, 1), item_dim), dtype=np.float32)
    hist_ratings = np.zeros((1, max(n, 1)), dtype=np.float32)
    hist_days = np.zeros((1, max(n, 1)), dtype=np.float32)
    hist_known = np.zeros((1, max(n, 1)), dtype=np.float32)
    hist_mask = np.zeros((1, max(n, 1)), dtype=bool)
    memory = np.zeros((1, memory_dim), dtype=np.float32)
    if n == 0:
        return hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory
    attr_indexes = {name: idx for idx, name in enumerate(attrs)}
    attr_positions = [item_names.index(f"meta_attr_{attr}") for attr in attrs]
    high_counts = {attr: 0 for attr in attrs}
    low_counts = {attr: 0 for attr in attrs}
    all_counts = {attr: 0 for attr in attrs}
    ratings = []
    for position, preference in enumerate(preferences):
        value = str(preference.get("value") or "").lower()
        attr = value if value in attr_indexes else None
        polarity = str(preference.get("polarity") or "positive").lower()
        rating = 1.0 if polarity in {"avoid", "negative", "dislike"} else 5.0
        ratings.append(rating)
        hist_ratings[0, position] = rating
        hist_known[0, position] = 1.0
        hist_mask[0, position] = True
        if attr is not None:
            hist_features[0, position, item_names.index(f"meta_attr_{attr}")] = 1.0
            hist_features[0, position, item_names.index("meta_attr_coverage")] = 1.0 / len(attrs)
            hist_features[0, position, item_names.index("meta_attr_evidence_count")] = 1.0
            all_counts[attr] += 1
            if rating >= 4:
                high_counts[attr] += 1
            else:
                low_counts[attr] += 1
    ratings_array = np.asarray(ratings, dtype=np.float32)
    memory[0, memory_names.index("h_count")] = n
    memory[0, memory_names.index("h_count_log")] = np.log1p(n)
    memory[0, memory_names.index("h_mean")] = ratings_array.mean()
    memory[0, memory_names.index("h_std")] = ratings_array.std()
    memory[0, memory_names.index("h_min")] = ratings_array.min()
    memory[0, memory_names.index("h_max")] = ratings_array.max()
    memory[0, memory_names.index("h_high_share")] = float((ratings_array >= 4).mean())
    memory[0, memory_names.index("h_low_share")] = float((ratings_array <= 2).mean())
    memory[0, memory_names.index("h_neutral_share")] = 0.0
    memory[0, memory_names.index("h_last_rating")] = ratings_array[-1]
    memory[0, memory_names.index("h_last3_mean")] = ratings_array[-3:].mean()
    memory[0, memory_names.index("h_last5_mean")] = ratings_array[-5:].mean()
    memory[0, memory_names.index("h_last_minus_prev")] = ratings_array[-1] - ratings_array[:-1].mean() if n > 1 else 0.0
    memory[0, memory_names.index("h_days_since_last")] = 0.0
    memory[0, memory_names.index("h_tenure_days")] = 0.0
    memory[0, memory_names.index("h_avg_gap_days")] = 0.0
    memory[0, memory_names.index("h_decay_mean")] = ratings_array.mean()
    memory[0, memory_names.index("h_decay_high_share")] = float((ratings_array >= 4).mean())
    memory[0, memory_names.index("h_decay_low_share")] = float((ratings_array <= 2).mean())
    memory[0, memory_names.index("h_catalog_count")] = n
    memory[0, memory_names.index("h_catalog_coverage")] = 0.0
    memory[0, memory_names.index("h_catalog_attr_coverage_mean")] = 0.0
    for attr in attrs:
        memory[0, memory_names.index(f"h_all_attr_{attr}")] = all_counts[attr]
        memory[0, memory_names.index(f"h_high_attr_{attr}")] = high_counts[attr]
        memory[0, memory_names.index(f"h_low_attr_{attr}")] = low_counts[attr]
    return hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory


def _components(row):
    pool = row["candidate_pool"]
    eligible = [
        item for item in pool
        if int((item.get("scorer") or {}).get("hard_tier") or 0) == 0
        and str((item.get("scorer") or {}).get("role_status") or "NOT_REQUIRED") in {"MATCH", "NOT_REQUIRED"}
    ]
    eligible_ids = {item["item_id"] for item in eligible}
    acceptable = set(row.get("acceptable_ids_in_pool") or [])
    return pool, eligible_ids, acceptable


def _hits(ranking, acceptable):
    return {
        "top1": ranking[:1],
        "hit@1": bool(ranking and ranking[0] in acceptable),
        "hit@3": bool(any(item in acceptable for item in ranking[:3])),
        "hit@5": bool(any(item in acceptable for item in ranking[:5])),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("shadow script expects GPU")
    device = torch.device("cuda")
    item_features, item_index, item_names, memory_names, rows, phase1_rows = _load()
    attr_indices, high_idx, low_idx, attrs = _attr_indices(item_names, memory_names)
    from experiments.neural_memory_ranking.model import ModelConfig, NeuralMemoryRanker

    models = {}
    for name, file in (("memory_off", "memory_off_hard.pt"), ("memory_on", "memory_on_hard.pt")):
        checkpoint = torch.load(OUT / file, map_location="cpu")
        config = ModelConfig(**checkpoint["config"])
        model = NeuralMemoryRanker(config, checkpoint["attr_indices"], checkpoint["high_attr_indices"], checkpoint["low_attr_indices"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models[name] = model
    item_matrix = torch.from_numpy(item_features).cuda()
    n_items = item_matrix.shape[0] - 1

    per_task = []
    latencies = []
    for row in rows:
        pool, eligible, acceptable = _components(row)
        candidate_ids = [item["item_id"] for item in pool]
        candidate_indices = np.asarray([item_index.get(item_id, n_items) for item_id in candidate_ids], dtype=np.int64)
        bge_rel = np.asarray([1.0 - (float(item["original_bge_rank"]) - 1.0) / max(1, len(pool) - 1) for item in pool])
        result = {"task_id": row["task_id"], "eligible_ids": sorted(eligible), "acceptable_ids": sorted(acceptable)}
        phase1 = phase1_rows.get(row["task_id"]) or {}
        # Existing Scorer arms from the accepted Phase-1 fixed-pool benchmark.
        for arm, key in (
            ("relevance_only", "A_relevance_only"),
            ("existing_scorer_memory_off", "B_scorer_memory_off"),
            ("existing_scorer_memory_on", "C_scorer_memory_on"),
        ):
            ranking = [item_id for item_id in (phase1.get("rankings") or {}).get(key, []) if item_id in eligible]
            result[arm] = _hits(ranking, acceptable)
        # Neural models with pseudo-history from event-sourced memory.
        preferences = row.get("historical_memory") or []
        hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory = _pseudo_history(
            preferences, item_features.shape[1], len(memory_names), item_names, memory_names,
            attr_indices, attrs, item_features,
        )
        candidates = torch.from_numpy(candidate_indices).cuda().unsqueeze(0)
        cand_features = item_matrix[candidates]
        hist_features_t = torch.from_numpy(hist_features).cuda()
        hist_ratings_t = torch.from_numpy(hist_ratings).cuda()
        hist_days_t = torch.from_numpy(hist_days).cuda()
        hist_known_t = torch.from_numpy(hist_known).cuda()
        hist_mask_t = torch.from_numpy(hist_mask).cuda()
        memory_t = torch.from_numpy(memory).cuda()
        started = time.perf_counter()
        with torch.no_grad():
            scores_off, _ = models["memory_off"](
                cand_features, hist_features_t, hist_ratings_t, hist_days_t, hist_known_t, hist_mask_t, memory_t
            )
            scores_on, _ = models["memory_on"](
                cand_features, hist_features_t, hist_ratings_t, hist_days_t, hist_known_t, hist_mask_t, memory_t
            )
        latencies.append((time.perf_counter() - started) * 1000.0)
        score_off = scores_off[0].cpu().numpy()
        score_on = scores_on[0].cpu().numpy()
        z_off = (score_off - score_off.mean()) / (score_off.std() + 1e-6)
        z_on = (score_on - score_on.mean()) / (score_on.std() + 1e-6)
        z_bge = (bge_rel - bge_rel.mean()) / (bge_rel.std() + 1e-6)
        combined = z_on + z_bge
        # rank among eligible only
        eligible_positions = [index for index, item in enumerate(pool) if item["item_id"] in eligible]
        for label, values in (
            ("neural_off", z_off),
            ("neural_on", z_on),
            ("neural_on_plus_bge", combined),
        ):
            order = sorted(eligible_positions, key=lambda index: (-float(values[index]), candidate_ids[index]))
            ranking = [candidate_ids[index] for index in order]
            result[label] = _hits(ranking, acceptable)
            result[label]["top1"] = ranking[:1]
        result["bge_top1"] = [candidate_ids[int(np.argmax(bge_rel))]]
        result["active_preferences"] = preferences
        per_task.append(result)

    def aggregate(arm):
        return {
            "tasks": len(per_task),
            "hit@1": sum(1 for row in per_task if row[arm]["hit@1"]),
            "hit@3": sum(1 for row in per_task if row[arm]["hit@3"]),
            "hit@5": sum(1 for row in per_task if row[arm]["hit@5"]),
        }

    # Counterfactual shadow on 0051: mutate event-sourced memory and compare scores.
    task_0051 = next(row for row in per_task if row["task_id"] == "p14e2_lite_0051")
    row_0051 = next(row for row in rows if row["task_id"] == "p14e2_lite_0051")
    # Recompute with a few explicit states for diagnosis; reuse helper inside main scope.
    def score_0051(preferences):
        nonlocal item_features, item_index, item_names, memory_names, attr_indices, attrs
        pool = row_0051["candidate_pool"]
        eligible = [
            item for item in pool
            if int((item.get("scorer") or {}).get("hard_tier") or 0) == 0
            and str((item.get("scorer") or {}).get("role_status") or "NOT_REQUIRED") in {"MATCH", "NOT_REQUIRED"}
        ]
        candidate_ids = [item["item_id"] for item in pool]
        candidate_indices = np.asarray([item_index.get(item_id, n_items) for item_id in candidate_ids], dtype=np.int64)
        hist_features, hist_ratings, hist_days, hist_known, hist_mask, memory = _pseudo_history(
            preferences, item_features.shape[1], len(memory_names), item_names, memory_names,
            attr_indices, attrs, item_features,
        )
        candidates = torch.from_numpy(candidate_indices).cuda().unsqueeze(0)
        with torch.no_grad():
            scores, _ = models["memory_on"](
                item_matrix[candidates], torch.from_numpy(hist_features).cuda(),
                torch.from_numpy(hist_ratings).cuda(), torch.from_numpy(hist_days).cuda(),
                torch.from_numpy(hist_known).cuda(), torch.from_numpy(hist_mask).cuda(),
                torch.from_numpy(memory).cuda(),
            )
        scores = scores[0].cpu().numpy()
        order = sorted(range(len(candidate_ids)), key=lambda index: (-float(scores[index]), candidate_ids[index]))
        return {
            "ranking": [candidate_ids[index] for index in order],
            "scores": {candidate_ids[index]: round(float(scores[index]), 6) for index in range(len(candidate_ids))},
            "eligible": [item["item_id"] for item in eligible],
        }
    base_prefs = row_0051.get("historical_memory") or []
    counterfactual = {
        "baseline": score_0051(base_prefs),
        "withdraw_microphone": score_0051([p for p in base_prefs if str(p.get("value")) != "microphone"]),
        "add_negative_microphone": score_0051(base_prefs + [{"value": "microphone", "polarity": "negative", "kind": "attribute"}]),
        "add_positive_in_ear": score_0051(base_prefs + [{"value": "in_ear", "polarity": "positive", "kind": "form_factor"}]),
    }

    report = {
        "schema": "agentrecx.neural_memory_ranking.p20_shadow.v1",
        "tasks": len(per_task),
        "protocol": {
            "fixed_pool": "Original BGE Top-20 from runs/preference_ranking_v1/ranking_dataset.jsonl",
            "labels": "provisional observed-acceptable labels; not used in training",
            "eligibility": "existing hard_tier==0 and role MATCH/NOT_REQUIRED",
            "neural_memory_source": "event-sourced active preferences converted to pseudo-history (rating 5 positive / 1 negative); not faithful rating history",
            "bge_blend": "z(neural_on) + z(normalized BGE rank), fixed lambda=1.0, no label tuning",
        },
        "aggregate": {
            "relevance_only": aggregate("relevance_only"),
            "existing_scorer_memory_off": aggregate("existing_scorer_memory_off"),
            "existing_scorer_memory_on": aggregate("existing_scorer_memory_on"),
            "neural_memory_off": aggregate("neural_off"),
            "neural_memory_on": aggregate("neural_on"),
            "neural_memory_on_plus_bge": aggregate("neural_on_plus_bge"),
        },
        "latency_ms": {
            "mean": round(float(statistics.mean(latencies)), 3),
            "p50": round(float(statistics.median(latencies)), 3),
            "max": round(float(max(latencies)), 3),
            "scope": "20 candidates + pseudo-history, GPU, batch=1 including Python overhead",
        },
        "per_task": per_task,
        "counterfactual_0051": counterfactual,
        "limitations": [
            "The neural model was trained on inferred rating histories, not explicit event-sourced preferences.",
            "Pseudo-history ratings 5/1 are a compatibility probe, not user-generated ratings.",
            "The BGE blend uses a fixed pre-set weight and was not tuned on P20 labels.",
            "P20 labels are provisional and are not user-satisfaction gold.",
        ],
    }
    (OUT / "shadow_p20.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("aggregate", "latency_ms", "counterfactual_0051")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
