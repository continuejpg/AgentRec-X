"""E-04B-2 Gate B-2: real Two-Tower load, real inference, same-case comparison with Gate B-1.

Frozen conditions are those of Gate B-1 (docs/V2_E04B_DUAL_EVALUATION_PROTOCOL.md,
experiments/e04b_protocol_v2.py): temporal_leave_two_out / agentrecx.eval_protocol.v1, seed 20261015,
K = 5/10/20, full catalogue, PAD excluded, the accepted seen mask, and the *same* case rule
("first N valid cases in artifact order"), which this script re-derives with the same reader so the
case-id sequence can be compared one by one against the archived SASRec run.

Scoring uses the production path only: `TwoTowerInferenceEngine.score_catalog` over raw model scores.
`runs/b1_evidence/two_tower_head.npz` is deliberately **not** used - it is a ranking cache written by
`experiments/b1_fusion_paired_evidence.py` for the B1 paired-evidence experiment (arms A-D), it is not
referenced anywhere under `recommendation/`, and it is therefore not part of inference.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TT_DIR = REPO / "runs" / "twotower_public_2026_full"
CHECKPOINT = TT_DIR / "checkpoint.pt"
CATEGORIES = TT_DIR / "metadata_categories.json"
RUN_JSON = TT_DIR / "run.json"
HEAD_CACHE = REPO / "runs" / "b1_evidence" / "two_tower_head.npz"
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
B1_DIR = REPO / "runs" / "e04b_gate_b1_sasrec"

PROTOCOL = "agentrecx.eval_protocol.v1"
SPLIT_NAME = "temporal_leave_two_out"
SEED = 20261015
SELECTION_RULE = "artifact order: the first N users with a valid protocol case (identical to Gate B-1)"
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


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = base.with_name(f"{base.name}_r{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free result directory next to {base}")


def exact_mcnemar(b: int, c: int) -> dict[str, Any]:
    """Two-sided exact McNemar (binomial) over discordant pairs, no external dependency."""
    n = b + c
    if n == 0:
        return {"discordant": 0, "p_value": None, "note": "no discordant pairs"}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"discordant": n, "b_only_first": b, "c_only_second": c,
            "p_value": round(min(1.0, 2 * tail), 6), "test": "exact binomial McNemar (two-sided)"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-04B-2 Gate B-2: real Two-Tower evaluation")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "e04b_gate_b2_twotower")
    parser.add_argument("--b1", type=Path, default=B1_DIR)
    parser.add_argument("--smoke", type=int, default=5)
    parser.add_argument("--exploratory", type=str, default="")
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
        "scoring": "TwoTowerInferenceEngine.score_catalog (raw full-catalogue scores, production path)",
        "output_dir": str(out),
    }

    from recommendation.datasets.sasrec import encode_inference_history
    from recommendation.evaluation.metrics import (
        excluded_seen, hit_at_k, ndcg_at_k, rank_of_target, sorted_ranking,
    )
    from recommendation.inference.twotower import TwoTowerInferenceConfig, TwoTowerInferenceEngine

    hashes_before = {
        "checkpoint": _sha256(CHECKPOINT),
        "categories": _sha256(CATEGORIES),
        "head_cache": _sha256(HEAD_CACHE) if HEAD_CACHE.exists() else None,
    }
    run_json = json.loads(RUN_JSON.read_text(encoding="utf-8"))
    results["phase1_assets"] = {
        "checkpoint_path": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": hashes_before["checkpoint"],
        "categories_path": str(CATEGORIES.relative_to(REPO)),
        "categories_sha256": hashes_before["categories"],
        "head_cache_path": str(HEAD_CACHE.relative_to(REPO)) if HEAD_CACHE.exists() else None,
        "head_cache_sha256": hashes_before["head_cache"],
        "run_id": run_json.get("run_id"),
        "architecture": run_json.get("architecture"),
        "model_config": run_json.get("model_config"),
        "objective": run_json.get("objective"),
        "dataset": run_json.get("dataset"),
        "metadata": run_json.get("metadata"),
        "rss_mb_before_load": round(_rss_mb(), 1),
        "mem_available_mb_before_load": round(_available_mb(), 1),
    }
    results["calibration_decision"] = {
        "asset": "runs/b1_evidence/two_tower_head.npz",
        "used_for_scoring": False,
        "why": [
            "written by experiments/b1_fusion_paired_evidence.py:484-511 as the B1 paired-evidence "
            "ranking cache for arms A-D, not by the training or serving path",
            "no reference to it under recommendation/ (grep): inference/twotower.py never loads it",
            "its manifest (runs/b1_evidence/B1_EXECUTION_MANIFEST.json) is a COMPLETE experiment "
            "record with classification B_minus_A/D_minus_C, i.e. an experiment artifact",
            "using a post-hoc ranking cache would change the ranking path and the comparison basis",
        ],
        "consequence": "scoring follows the production path only (raw full-catalogue model scores)",
    }

    began = time.perf_counter()
    engine = TwoTowerInferenceEngine(
        TwoTowerInferenceConfig(checkpoint_path=CHECKPOINT, categories_path=CATEGORIES,
                                mappings_path=MAPPINGS, device="cpu")
    )
    results["load"] = {
        "seconds": round(time.perf_counter() - began, 2), "is_ready": bool(engine.is_ready()),
        "model_metadata": engine.model_metadata(), "rss_mb_after_load": round(_rss_mb(), 1),
    }

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    results["identity"] = {
        "num_items": num_items, "mapping_entries": len(mappings["item2id"]),
        "pad_id": int(mappings["padding"]["pad_id"]),
        "first_real_id": int(mappings["padding"]["first_real_id"]),
        "engine_num_items": getattr(engine, "num_items", None),
        "mapping_and_data_agree": len(mappings["item2id"]) == num_items,
    }

    from experiments.e04b2_gate_a import stream_sequences
    from recommendation.evaluation.split import split_sequence

    cases = []
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is not None:
            cases.append(case)
        if len(cases) >= max(size for _name, size in stages):
            break

    # Same-case alignment against the archived SASRec run (case ids and their order).
    b1_path = args.b1 / "stage_exploratory_100.json"
    alignment: dict[str, Any] = {"b1_stage": str(b1_path), "checked": False}
    if b1_path.exists():
        b1_stage = json.loads(b1_path.read_text(encoding="utf-8"))
        b1_ids = [row.get("case") for row in b1_stage.get("per_case", [])]
        mine = [case.user_int_id for case in cases[: len(b1_ids)]]
        alignment.update({
            "checked": True, "b1_cases": len(b1_ids), "rebuilt_cases": len(mine),
            "same_ids_in_same_order": b1_ids == mine,
            "first_mismatch": next((index for index, (a, b) in enumerate(zip(b1_ids, mine)) if a != b),
                                   None),
            "note": "history/target/mask are re-derived by the same frozen rule and split function; "
                    "the archived run stores case ids and ranks, not histories",
        })
    results["case_alignment"] = alignment

    for name, size in stages:
        selected = cases[:size]
        stage: dict[str, Any] = {
            "stage": name, "cases_selected": len(selected), "cases_valid": 0, "cases_scored": 0,
            "cases_skipped": 0, "model_failures": 0, "invalid_candidates": 0, "pad_candidates": 0,
            "seen_candidates": 0, "target_ids_fed_to_model": 0,
            "hits": {str(k): 0 for k in k_values}, "output_candidates_total": 0,
            "seconds": 0.0, "rss_mb_peak": round(_rss_mb(), 1), "per_case": [],
        }
        began = time.perf_counter()
        for case in selected:
            history = list(case.test_history)
            if case.test_target in history:
                stage["target_ids_fed_to_model"] += 1     # repeat-interaction bookkeeping
            try:
                encoded = encode_inference_history(history, engine.max_seq_len, num_items)
                scores = engine.score_catalog(encoded)
            except Exception as exc:  # noqa: BLE001 - counted, never hidden
                stage["model_failures"] += 1
                stage["per_case"].append({"case": case.user_int_id, "error": type(exc).__name__})
                continue
            stage["cases_valid"] += 1
            top = sorted_ranking(scores, history=case.test_history, num_items=num_items)[: max(k_values)]
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
            str(k): round(sum(ndcg_at_k(row["rank"], k) for row in stage["per_case"]
                              if "rank" in row) / scored, 6)
            for k in k_values
        }
        (out / f"stage_{name}_{size}.json").write_text(json.dumps(stage, indent=2), encoding="utf-8")
        results.setdefault("stages", {})[f"{name}_{size}"] = {
            key: value for key, value in stage.items() if key != "per_case"
        }
        print(json.dumps({k: v for k, v in stage.items() if k != "per_case"}, indent=2))

    # Paired comparison on the largest common stage, using the archived SASRec per-case ranks.
    common = stages[-1][1]
    b1_stage_path = args.b1 / f"stage_{stages[-1][0]}_{common}.json"
    if b1_stage_path.exists():
        b1_stage = json.loads(b1_stage_path.read_text(encoding="utf-8"))
        tt_stage = json.loads((out / f"stage_{stages[-1][0]}_{common}.json").read_text(encoding="utf-8"))
        b1_ranks = [row["rank"] for row in b1_stage["per_case"] if "rank" in row]
        tt_ranks = [row["rank"] for row in tt_stage["per_case"] if "rank" in row]
        comparison: dict[str, Any] = {"cases": min(len(b1_ranks), len(tt_ranks)),
                                      "same_case_order": alignment.get("same_ids_in_same_order"),
                                      "table": {}, "paired": {}}
        for k in k_values:
            sas_hits = sum(1 for rank in b1_ranks if rank <= k)
            tt_hits = sum(1 for rank in tt_ranks if rank <= k)
            comparison["table"][f"HR@{k}"] = {
                "sasrec": round(sas_hits / max(len(b1_ranks), 1), 6),
                "two_tower": round(tt_hits / max(len(tt_ranks), 1), 6),
                "difference": round((tt_hits - sas_hits) / max(len(b1_ranks), 1), 6),
                "hits": {"sasrec": sas_hits, "two_tower": tt_hits, "denominator": len(b1_ranks)},
            }
            b = sum(1 for a, c in zip(b1_ranks, tt_ranks) if a <= k and c > k)
            c_only = sum(1 for a, c in zip(b1_ranks, tt_ranks) if a > k and c <= k)
            comparison["paired"][f"HR@{k}"] = exact_mcnemar(b, c_only)
        results["comparison_vs_sasrec"] = comparison

    results["hash_check_after"] = {
        "checkpoint": _sha256(CHECKPOINT), "categories": _sha256(CATEGORIES),
        "head_cache": _sha256(HEAD_CACHE) if HEAD_CACHE.exists() else None,
        "unchanged": hashes_before["checkpoint"] == _sha256(CHECKPOINT),
    }
    results["resources"] = {
        "rss_mb_end": round(_rss_mb(), 1), "mem_available_mb_end": round(_available_mb(), 1),
        "note": "CPU-only; one model instance; per-case full-catalogue scoring at batch size 1",
    }
    (out / "gate_b2_summary.json").write_text(json.dumps(results, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
