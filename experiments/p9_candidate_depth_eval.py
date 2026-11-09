"""P9 candidate-depth / source-coverage evaluation on the fixed P8-D 300 cases.

Reuses the 300 P8-D Dev cases and their cached Hybrid Catalog Top-50 lists.  Evaluates:
  * actual Serving V1 History-source reachability at budget 50 and 100 (the legal max);
  * SASRec and Two-Tower source-level reachability at 50/100/200 using their accepted
    full score matrices (200 is an offline headroom probe, not a serving change);
  * union reachability with the cached catalog Top-50;
  * final Top-10 ranking under balanced and history_focus with history budget 50 vs 100;
  * candidates lost by fusion/dedup/filtering despite being ranked by a component.

No model, index, policy, Serving default, Gold, Holdout, M5, Role Gate or Eligibility is
modified.  The 200-deep component heads are explicitly labelled offline probes; the
accepted serving engines still refuse k > MAX_K (100).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "runs/p8_d/multiuser/cases.jsonl"
CACHE = ROOT / "runs/p8_d/multiuser/candidate_cache.jsonl"
OUT = ROOT / "runs/p9/candidate_depth"

import os

os.environ["AGENTRECX_SERVING_FUSION"] = "rrf3"
os.environ["AGENTRECX_MAPPINGS_PATH"] = str(
    ROOT / "data/processed/Sports_and_Outdoors_mappings.json"
)
os.environ["AGENTRECX_TWOTOWER_DIR"] = str(ROOT / "runs/twotower_public_2026_full")
os.environ["AGENTRECX_STATE_MODE"] = "off"

from recommendation.api.app import ServiceSettings, build_engine
from recommendation.control.candidate_ledger import (
    SOURCE_POLICIES,
    weighted_reciprocal_rank_fusion,
)
from recommendation.datasets.sasrec import encode_inference_history


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rank_component(
    scores: list[float],
    *,
    num_items: int,
    seen_ids: list[int],
    limit: int,
) -> list[tuple[int, float]]:
    """Exact score ranking for an offline probe, same top-k ordering rule as serving."""
    array = np.asarray(scores, dtype=np.float64)
    if array.shape[0] != num_items + 1:
        raise ValueError("score vector shape mismatch")
    eligible = np.ones(num_items + 1, dtype=bool)
    eligible[0] = False
    if seen_ids:
        seen = np.asarray([int(value) for value in seen_ids], dtype=np.int64)
        seen = seen[(seen >= 1) & (seen <= num_items)]
        eligible[seen] = False
    candidate_ids = np.flatnonzero(eligible)
    if candidate_ids.size == 0 or limit <= 0:
        return []
    selected = candidate_ids[np.lexsort((candidate_ids, -array[candidate_ids]))[:limit]]
    return [(int(item_id), float(array[item_id])) for item_id in selected]


def _rank_of(ranked: list[tuple[int, float]], item_id: int) -> int | None:
    for position, (candidate, _) in enumerate(ranked, start=1):
        if candidate == item_id:
            return position
    return None


def _rank_of_asin(ranked: list[str] | list[dict], identity: str) -> int | None:
    for position, item in enumerate(ranked, start=1):
        value = item.get("parent_asin") if isinstance(item, dict) else item
        if value == identity:
            return position
    return None


def _metrics(ranked: list[tuple[str, float]], target: str, k: int = 10) -> dict:
    top = [identity for identity, _ in ranked[:k]]
    full = [identity for identity, _ in ranked]
    rank_top = None
    rank_full = None
    for position, identity in enumerate(top, start=1):
        if identity == target:
            rank_top = position
            break
    for position, identity in enumerate(full, start=1):
        if identity == target:
            rank_full = position
            break
    return {
        "top10": top,
        "target_rank_top10": rank_top,
        "target_rank_full": rank_full,
        "recall@10": 1.0 if rank_top is not None else 0.0,
        "ndcg@10": 1.0 / math.log2(rank_top + 1) if rank_top else 0.0,
        "mrr@10": 1.0 / rank_top if rank_top else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-budget", type=int, default=100)
    parser.add_argument("--probe-limit", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="0 = all remaining cases")
    parser.add_argument("--start", type=int, default=0, help="0-based case offset for resumable runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = _load_jsonl(CASES)
    if args.start:
        cases = cases[args.start :]
    if args.limit > 0:
        cases = cases[: args.limit]
    cache = {str(row["case_id"]): row for row in _load_jsonl(CACHE)}
    settings = ServiceSettings.from_env()
    started = time.perf_counter()
    engine, error = build_engine(settings)
    load_seconds = time.perf_counter() - started
    if engine is None:
        raise SystemExit(f"engine unavailable: {error}")
    if not hasattr(engine, "sequential") or not hasattr(engine, "two_tower"):
        raise SystemExit("expected a FusionInferenceEngine with sequential/two_tower sources")
    seq = engine.sequential
    tt = engine.two_tower
    print(
        f"engine loaded in {load_seconds:.2f}s; sources={engine.config.sources}; "
        f"head_depth={engine.config.head_depth}",
        flush=True,
    )

    per_case = []
    started = time.perf_counter()
    target_item_ids: dict[str, int] = {}
    for index, case in enumerate(cases, start=1):
        history = [str(value) for value in case["history"]]
        target = str(case["target"])
        cache_row = cache.get(str(case["case_id"]))
        if cache_row is None:
            per_case.append({"case_id": case["case_id"], "status": "skipped_no_cache"})
            continue
        # Exact identity encoding for component probes.
        seq_encoded, seq_truncated = seq.encode_history(history)
        seq_seen = [seq.parent_asin_to_item_id(value) for value in history]
        tt_ids = [tt.parent_asin_to_item_id(value) for value in history]
        tt_encoded = encode_inference_history(tt_ids, tt.max_seq_len, tt.num_items)
        target_seq_id = seq.parent_asin_to_item_id(target)
        target_tt_id = tt.parent_asin_to_item_id(target)
        target_item_ids[case["case_id"]] = target_seq_id

        # Actual Serving V1 history source at the legal max depth 100.
        began = time.perf_counter()
        history_result_100 = engine.recommend(history, k=args.history_budget)
        history_seconds = time.perf_counter() - began
        history_rows_100 = [
            {"parent_asin": item.parent_asin, "rank": item.rank, "score": item.score}
            for item in history_result_100.recommendations
        ]
        history_rows_50 = history_rows_100[:50]
        history_ids_50 = {row["parent_asin"] for row in history_rows_50}
        history_ids_100 = {row["parent_asin"] for row in history_rows_100}

        # Component probes: one full score matrix each, rank to probe limit.
        began = time.perf_counter()
        seq_scores = seq.score_catalog(seq_encoded)
        seq_ranked = _rank_component(
            seq_scores, num_items=seq.num_items, seen_ids=seq_seen, limit=args.probe_limit
        )
        seq_ranked_100 = _rank_component(
            seq_scores, num_items=seq.num_items, seen_ids=seq_seen, limit=100
        )
        seq_seconds = time.perf_counter() - began
        began = time.perf_counter()
        tt_scores = tt.score_catalog(tt_encoded)
        tt_ranked = _rank_component(
            tt_scores, num_items=tt.num_items, seen_ids=tt_ids, limit=args.probe_limit
        )
        tt_ranked_100 = _rank_component(
            tt_scores, num_items=tt.num_items, seen_ids=tt_ids, limit=100
        )
        tt_seconds = time.perf_counter() - began
        seq_rank_50 = _rank_of(seq_ranked_100, target_seq_id)
        tt_rank_50 = _rank_of(tt_ranked_100, target_tt_id)
        seq_rank_100 = _rank_of(seq_ranked_100, target_seq_id)
        tt_rank_100 = _rank_of(tt_ranked_100, target_tt_id)
        seq_rank_probe = _rank_of(seq_ranked, target_seq_id)
        tt_rank_probe = _rank_of(tt_ranked, target_tt_id)
        target_in_seq_50 = _rank_of(seq_ranked[:50], target_seq_id) is not None
        target_in_tt_50 = _rank_of(tt_ranked[:50], target_tt_id) is not None
        target_in_seq_100 = _rank_of(seq_ranked_100, target_seq_id) is not None
        target_in_tt_100 = _rank_of(tt_ranked_100, target_tt_id) is not None
        target_in_seq_probe = _rank_of(seq_ranked, target_seq_id) is not None
        target_in_tt_probe = _rank_of(tt_ranked, target_tt_id) is not None

        catalog_rows = [
            {"parent_asin": row["parent_asin"], "rank": int(row["rank"]), "score": row["score"]}
            for row in (cache_row.get("catalog_rows") or ())
        ]
        catalog_ids = {row["parent_asin"] for row in catalog_rows}
        catalog_target = target in catalog_ids
        history_target_50 = target in history_ids_50
        history_target_100 = target in history_ids_100
        # Final Agent RRF replay uses actual history source rows and cached catalog rows.
        final_replay = {}
        for label, history_rows in (("history50", history_rows_50), ("history100", history_rows_100)):
            for policy in ("balanced", "history_focus"):
                fused = weighted_reciprocal_rank_fusion(
                    {
                        "history": [row["parent_asin"] for row in history_rows],
                        "catalog_search": [row["parent_asin"] for row in catalog_rows],
                    },
                    source_policy=policy,
                )
                final_replay[f"{label}:{policy}"] = _metrics(fused, target)
        per_case.append(
            {
                "case_id": case["case_id"],
                "user_id": case["user_id"],
                "status": "evaluated",
                "is_canonical": bool(case.get("is_canonical")),
                "history_len": len(history),
                "target": target,
                "catalog_target": catalog_target,
                "history_target_50": history_target_50,
                "history_target_100": history_target_100,
                "sasrec_target_50": target_in_seq_50,
                "sasrec_target_100": target_in_seq_100,
                "sasrec_target_probe": target_in_seq_probe,
                "sasrec_rank_50": _rank_of(seq_ranked_100[:50], target_seq_id),
                "sasrec_rank_100": seq_rank_100,
                "sasrec_rank_probe": seq_rank_probe,
                "two_tower_target_50": target_in_tt_50,
                "two_tower_target_100": target_in_tt_100,
                "two_tower_target_probe": target_in_tt_probe,
                "two_tower_rank_50": _rank_of(tt_ranked_100[:50], target_tt_id),
                "two_tower_rank_100": tt_rank_100,
                "two_tower_rank_probe": tt_rank_probe,
                "history_rank_50": _rank_of_asin(history_rows_50, target),
                "history_rank_100": _rank_of_asin(history_rows_100, target),
                "component_lost_by_history": {
                    "sasrec_100_not_history": bool(target_in_seq_100 and not history_target_100),
                    "two_tower_100_not_history": bool(target_in_tt_100 and not history_target_100),
                    "sasrec_50_not_history": bool(target_in_seq_50 and not history_target_50),
                    "two_tower_50_not_history": bool(target_in_tt_50 and not history_target_50),
                },
                "timing_seconds": {
                    "history100": round(history_seconds, 4),
                    "sasrec_scores_plus_200": round(seq_seconds, 4),
                    "two_tower_scores_plus_200": round(tt_seconds, 4),
                },
                "final_replay": final_replay,
            }
        )
        if index % 10 == 0:
            (OUT / "per_case.partial.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_case),
                encoding="utf-8",
            )
            print(f"{index}/{len(cases)} {case['case_id']}", flush=True)

    evaluated = [row for row in per_case if row.get("status") == "evaluated"]
    seconds = time.perf_counter() - started
    summary: dict[str, Any] = {
        "schema_version": "agentrecx.p9.candidate_depth_eval.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": len(evaluated),
        "source": "P8-D 300 dev cases; validation target only; Holdout untouched",
        "config": {
            "history_k": args.history_budget,
            "component_probe_limit": args.probe_limit,
            "serving_fusion": getattr(engine.config, "profile", "serving_v1"),
            "serving_sources": list(engine.config.sources),
            "serving_head_depth": engine.config.head_depth,
            "serving_max_k": 100,
            "probe_label": (
                "100 is the legal serving MAX_K; 200 is an offline score-matrix headroom probe, "
                "not a Serving change"
            ),
            "catalog_budget": 50,
            "catalog_cache": str(CACHE.relative_to(ROOT)),
            "target": "validation_parent_asin",
        },
        "time": {
            "engine_load_seconds": round(load_seconds, 4),
            "eval_seconds": round(seconds, 4),
            "mean_history_source_seconds": (
                sum(row["timing_seconds"]["history100"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "mean_sasrec_seconds": (
                sum(row["timing_seconds"]["sasrec_scores_plus_200"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
            "mean_two_tower_seconds": (
                sum(row["timing_seconds"]["two_tower_scores_plus_200"] for row in evaluated) / len(evaluated)
                if evaluated else 0.0
            ),
        },
        "reachability": {},
        "newly_reachable": {},
        "lost_by_fusion": {},
        "final_top10": {},
        "notes": {},
    }
    count_fields = [
        "catalog_target", "history_target_50", "history_target_100",
        "sasrec_target_50", "sasrec_target_100", "sasrec_target_probe",
        "two_tower_target_50", "two_tower_target_100", "two_tower_target_probe",
    ]
    for field in count_fields:
        summary["reachability"][field] = sum(1 for row in evaluated if row[field])
    summary["newly_reachable"] = {
        "history_50_to_100": [
            {"case_id": row["case_id"], "target": row["target"]}
            for row in evaluated
            if not row["history_target_50"] and row["history_target_100"]
        ],
        "sasrec_50_to_100": [
            {"case_id": row["case_id"], "target": row["target"]}
            for row in evaluated
            if not row["sasrec_target_50"] and row["sasrec_target_100"]
        ],
        "two_tower_50_to_100": [
            {"case_id": row["case_id"], "target": row["target"]}
            for row in evaluated
            if not row["two_tower_target_50"] and row["two_tower_target_100"]
        ],
        "sasrec_100_to_probe": [
            {"case_id": row["case_id"], "target": row["target"]}
            for row in evaluated
            if not row["sasrec_target_100"] and row["sasrec_target_probe"]
        ],
        "two_tower_100_to_probe": [
            {"case_id": row["case_id"], "target": row["target"]}
            for row in evaluated
            if not row["two_tower_target_100"] and row["two_tower_target_probe"]
        ],
    }
    summary["lost_by_fusion"] = {
        field: sum(1 for row in evaluated if row["component_lost_by_history"][field])
        for field in (
            "sasrec_100_not_history", "two_tower_100_not_history",
            "sasrec_50_not_history", "two_tower_50_not_history",
        )
    }
    summary["notes"]["lost_by_fusion"] = (
        "target appears in a component source ranking but not in the actual History source"
    )
    for label in (
        "history50:balanced", "history50:history_focus",
        "history100:balanced", "history100:history_focus",
    ):
        rows = [row["final_replay"][label] for row in evaluated]
        summary["final_top10"][label] = {
            "recall@10": sum(row["recall@10"] for row in rows) / len(rows),
            "ndcg@10": sum(row["ndcg@10"] for row in rows) / len(rows),
            "mrr@10": sum(row["mrr@10"] for row in rows) / len(rows),
            "target_reachable": sum(1 for row in rows if row["target_rank_top10"] is not None),
            "target_in_union": sum(
                1 for row in rows if row["target_rank_full"] is not None
            ),
        }
    summary["notes"]["final_top10"] = (
        "actual history source Top-50/Top-100 fused with cached Hybrid Catalog Top-50; "
        "balanced and history_focus policy; target in Top-10 only"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "per_case.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_case),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
