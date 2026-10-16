"""E-04B-2 Gate B-1: real SASRec load, real inference, restricted quality baseline.

Frozen conditions (docs/V2_E04B_DUAL_EVALUATION_PROTOCOL.md, experiments/e04b_protocol_v2.py):
temporal_leave_two_out / agentrecx.eval_protocol.v1, seed 20261015, K = 5/10/20, full catalogue,
PAD excluded, the accepted seen mask, deterministic case selection (artifact order - no sampling, no
per-case tuning), and a strict separation between the request input (test_history) and the label
(test_target).

Everything is reused, not re-implemented: the split comes from recommendation/evaluation/split.py,
the ranking and the mask from recommendation/evaluation/metrics.py, and the model from
recommendation/inference/sasrec.py.  This script never changes the model, the algorithm or the
protocol; it measures one model on real data and records the cost.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CHECKPOINT = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
RUN_JSON = REPO / "runs" / "sasrec_canonical_2026" / "run.json"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"

PROTOCOL = "agentrecx.eval_protocol.v1"
SPLIT_NAME = "temporal_leave_two_out"
SEED = 20261015
SELECTION_RULE = "artifact order: the first N users with a valid protocol case (no sampling, no tuning)"
BATCH_SIZE = 1


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _available_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("inf")


def _unique_dir(base: Path) -> Path:
    """Never overwrite an existing result directory."""
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = base.with_name(f"{base.name}_r{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free result directory next to {base}")


def phase1_assets() -> dict[str, Any]:
    """Phase 1: checkpoint identity, model metadata, load cost.  No metrics here."""
    from recommendation.inference.sasrec import InferenceConfig, SASRecInferenceEngine, sha256_file

    run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "checkpoint_path": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "recorded_sha256": run_json.get("checkpoints", {}).get("best", {}).get("sha256"),
        "model_config": run_json.get("model_config"),
        "protocol_recorded": run_json.get("cohort", {}).get("protocol"),
        "protocol_version_recorded": run_json.get("cohort", {}).get("protocol_version"),
        "min_sequence_length_recorded": run_json.get("cohort", {}).get("min_sequence_length"),
        "historical_formal_test_metrics": run_json.get("formal_test_metrics"),
        "test_sealed": run_json.get("test_sealed"),
        "rss_mb_before_load": round(_rss_mb(), 1),
        "mem_available_mb_before_load": round(_available_mb(), 1),
    }
    record["sha256_matches_run_json"] = (
        record["checkpoint_sha256"] == record["recorded_sha256"]
    )
    began = time.perf_counter()
    engine = SASRecInferenceEngine(
        InferenceConfig(
            checkpoint_path=CHECKPOINT,
            mappings_path=MAPPINGS,
            device="cpu",
            expected_checkpoint_sha256=record["recorded_sha256"],
        )
    )
    record["load_seconds"] = round(time.perf_counter() - began, 2)
    record["model_metadata"] = engine.model_metadata()
    record["is_ready"] = bool(engine.is_ready())
    record["rss_mb_after_load"] = round(_rss_mb(), 1)
    record["mem_available_mb_after_load"] = round(_available_mb(), 1)
    return record


def build_cases(limit: int, num_items: int) -> list[Any]:
    """Phase 2 input: protocol cases, deterministic order, target never used as input."""
    from experiments.e04b2_gate_a import stream_sequences
    from recommendation.evaluation.split import split_sequence

    cases: list[Any] = []
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is not None:
            cases.append(case)
        if len(cases) >= limit:
            break
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-04B-2 Gate B-1: real SASRec evaluation")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "e04b_gate_b1_sasrec")
    parser.add_argument("--smoke", type=int, default=5)
    parser.add_argument("--exploratory", type=str, default="",
                        help="extra stages, e.g. 20,100 (comma separated)")
    parser.add_argument("--k", default="5,10,20")
    args = parser.parse_args(argv)
    k_values = tuple(int(part) for part in args.k.split(",") if part.strip())
    stages = [("smoke", args.smoke)]
    stages += [("exploratory", int(part)) for part in str(args.exploratory).split(",") if part.strip()]

    out = _unique_dir(args.out)
    out.mkdir(parents=True)
    results: dict[str, Any] = {
        "protocol": PROTOCOL, "split": SPLIT_NAME, "seed": SEED, "k_values": list(k_values),
        "selection_rule": SELECTION_RULE, "batch_size": BATCH_SIZE,
        "mask_rule": "recommendation/evaluation/metrics.excluded_seen(test_history, test_target)",
        "ranking_rule": "recommendation/evaluation/metrics.sorted_ranking (canonical tie rule)",
        "scoring": "SASRecInferenceEngine.score_catalog (full catalogue, one forward per case)",
        "data": {"sequences": str(SEQUENCES.relative_to(REPO)), "mappings": str(MAPPINGS.relative_to(REPO))},
        "output_dir": str(out),
    }

    from recommendation.evaluation.metrics import (
        excluded_seen, hit_at_k, ndcg_at_k, rank_of_target, sorted_ranking,
    )
    from recommendation.inference.sasrec import SASRecInferenceEngine

    assets = phase1_assets()
    results["phase1_assets"] = assets

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    mapping_size = len(mappings["item2id"])
    results["identity"] = {
        "num_items": num_items, "mapping_entries": mapping_size,
        "pad_id": int(mappings["padding"]["pad_id"]),
        "first_real_id": int(mappings["padding"]["first_real_id"]),
        "engine_num_items": assets["model_metadata"].get("num_items")
        if isinstance(assets["model_metadata"], dict) else None,
        "mapping_and_data_agree": mapping_size == num_items,
        "id_base_note": "1-based real ids, 0 reserved for PAD (both tables); no 0-based offset",
    }

    from recommendation.inference.sasrec import InferenceConfig

    engine = SASRecInferenceEngine(
        InferenceConfig(checkpoint_path=CHECKPOINT, mappings_path=MAPPINGS, device="cpu",
                        expected_checkpoint_sha256=assets["recorded_sha256"])
    )

    cases = build_cases(max(size for _name, size in stages), num_items)
    results["cases_available"] = len(cases)

    for name, size in stages:
        selected = cases[:size]
        stage: dict[str, Any] = {
            "stage": name, "cases_selected": len(selected), "cases_valid": 0, "cases_scored": 0,
            "cases_skipped": 0, "model_failures": 0, "invalid_candidates": 0, "pad_candidates": 0,
            "seen_candidates": 0, "hits": {str(k): 0 for k in k_values},
            "HR": {}, "NDCG": {}, "output_candidates_total": 0, "target_ids_fed_to_model": 0,
            "seconds": 0.0, "rss_mb_peak": round(_rss_mb(), 1), "per_case": [],
        }
        began = time.perf_counter()
        for case in selected:
            history = list(case.test_history)                      # the only model input
            combined_mask = excluded_seen(case.test_history, case.test_target)
            # Runtime isolation proof: every id handed to the model is a history id.
            if case.test_target in history and case.test_target in combined_mask:
                stage["target_ids_fed_to_model"] += 1              # repeat-interaction bookkeeping
            try:
                encoded, _truncated = engine.encode_history(
                    [engine.item_id_to_parent_asin(item) for item in history]
                )
                scores = engine.score_catalog(encoded)
            except Exception as exc:  # noqa: BLE001 - counted, never hidden
                stage["model_failures"] += 1
                stage["per_case"].append({"case": case.user_int_id, "error": type(exc).__name__})
                continue
            stage["cases_valid"] += 1
            ranking = sorted_ranking(scores, history=case.test_history, num_items=num_items)
            top = ranking[: max(k_values)]
            stage["output_candidates_total"] += len(top)
            stage["invalid_candidates"] += sum(1 for i in top if i < 1 or i > num_items)
            stage["pad_candidates"] += sum(1 for i in top if i == 0)
            stage["seen_candidates"] += sum(1 for i in top if i in case.test_seen)
            rank = rank_of_target(scores, case.test_target, history=case.test_history,
                                  num_items=num_items)
            for k in k_values:
                stage["hits"][str(k)] += int(hit_at_k(rank, k))
            stage["cases_scored"] += 1
            stage["per_case"].append({"case": case.user_int_id, "rank": int(rank),
                                      "returned": len(top)})
            if _available_mb() < 900:
                stage["aborted"] = "mem_available below 900 MB"
                break
        stage["seconds"] = round(time.perf_counter() - began, 2)
        stage["rss_mb_peak"] = round(_rss_mb(), 1)
        scored = max(stage["cases_scored"], 1)
        stage["HR"] = {str(k): round(stage["hits"][str(k)] / scored, 6) for k in k_values}
        stage["NDCG"] = {
            str(k): round(
                sum(ndcg_at_k(row["rank"], k) for row in stage["per_case"] if "rank" in row) / scored,
                6,
            )
            for k in k_values
        }
        (out / f"stage_{name}_{size}.json").write_text(json.dumps(stage, indent=2), encoding="utf-8")
        results.setdefault("stages", {})[f"{name}_{size}"] = {
            key: value for key, value in stage.items() if key != "per_case"
        }
        print(json.dumps({k: v for k, v in stage.items() if k not in ("per_case",)}, indent=2))

    results["resources"] = {
        "rss_mb_end": round(_rss_mb(), 1), "mem_available_mb_end": round(_available_mb(), 1),
        "note": "CPU-only; no CUDA; per-case full-catalogue scoring at batch size 1",
    }
    (out / "gate_b1_summary.json").write_text(json.dumps(results, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
