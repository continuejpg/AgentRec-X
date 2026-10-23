"""E-04B-2 Gate B-3 CONTINUE: real TIGER generative retrieval on the frozen 100 cases.

The accepted path is reused throughout: `TigerServingEngine` (one local subprocess worker, JSONL IPC,
no network port) with `mode=CERTIFIED` and `allow_mode_downgrade=False`, so the mode the worker
actually ran is reported back and can be checked per case.  Decode parameters come from
`TigerWorkerConfig` defaults (the native inference defaults); nothing is tuned on the test cases.

Metrics are named `generative_topk_*`: TIGER returns a generated top-k, not a full-catalogue exact
scoring over 156,746 items, so its numbers are only a descriptive contrast with SASRec/Two-Tower.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
CKPT_DIR = REPO / "runs" / "tiger_public_2026" / "checkpoint"
SID_DIR = REPO / "runs" / "tiger_public_2026" / "sid"
HANDOFF = REPO / "runs" / "tiger_backend_handoff_prod"

PROTOCOL = "agentrecx.eval_protocol.v1"
SPLIT = "temporal_leave_two_out"
SEED = 20261015
SELECTION_RULE = "artifact order: the first N users with a valid protocol case (identical to Gates B-1/B-2)"
# Declared before running, from the native defaults - never chosen from hit rates.
FROZEN_DECODE = {"mode": "CERTIFIED", "allow_mode_downgrade": False, "k_max": 20,
                 "max_hist_items": 20, "beam": 512, "prefix_budget": 200_000,
                 "batch_prefixes": 256, "device": "cpu",
                 "source": "TigerWorkerConfig default values in recommendation/backends/tiger_engine.py:130-142"}


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


def _tree_rss_mb(pid: int) -> float:
    """RSS of the main process plus every descendant (the worker must not be invisible)."""
    total = 0.0
    try:
        pids = [pid]
        for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
            pids.append(int(child))
        for candidate in pids:
            with open(f"/proc/{candidate}/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        total += float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return total or _rss_mb()


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = base.with_name(f"{base.name}_r{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free directory next to {base}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-04B-2 Gate B-3 CONT: real TIGER retrieval")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "e04b_gate_b3_tiger")
    parser.add_argument("--stages", default="1", help="comma-separated stage sizes, e.g. 1,5,20,100")
    parser.add_argument("--k", default="5,10,20")
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--per-case-seconds", type=float, default=120.0)
    parser.add_argument("--mem-floor-mb", type=float, default=900.0)
    args = parser.parse_args(argv)
    k_values = tuple(int(part) for part in args.k.split(",") if part.strip())
    stages = [int(part) for part in str(args.stages).split(",") if part.strip()]

    out = _unique_dir(args.out)
    out.mkdir(parents=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    from recommendation.inference.tiger_engine import (
        _MappingIdentity, TigerServingEngine, TigerWorkerConfig,
    )
    from recommendation.evaluation.metrics import (
        excluded_seen, hit_at_k, ndcg_at_k,
    )

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    identity = _MappingIdentity({str(k): int(v) for k, v in mappings["item2id"].items()})

    from experiments.e04b2_gate_a import stream_sequences
    from recommendation.evaluation.split import split_sequence

    cases = []
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is not None:
            cases.append(case)
        if len(cases) >= max(stages):
            break

    manifest = {
        "protocol": PROTOCOL, "split": SPLIT, "seed": SEED, "k_values": list(k_values),
        "selection_rule": SELECTION_RULE, "stages": stages,
        "decode_config_frozen": FROZEN_DECODE, "declared_before_scoring": True,
        "case_manifest": {"rebuilt_cases": len(cases), "case_ids": [c.user_int_id for c in cases]},
        "paths": {"checkpoint_dir": str(CKPT_DIR.relative_to(REPO)),
                  "sid_dir": str(SID_DIR.relative_to(REPO)),
                  "handoff": str(HANDOFF.relative_to(REPO))},
        "mem_available_mb_start": round(_available_mb(), 1),
        "note": "generative top-k; not full-catalogue exact scoring",
    }

    worker = None
    results: dict[str, Any] = {"manifest": manifest}
    try:
        began = time.perf_counter()
        engine = TigerServingEngine(
            identity=identity,
            config=TigerWorkerConfig(checkpoint_dir=CKPT_DIR, sid_dir=SID_DIR,
                                     catalogue_dir=HANDOFF, device="cpu"),
            mode="CERTIFIED", allow_mode_downgrade=False,
            startup_timeout=args.startup_timeout, request_timeout=args.request_timeout,
            verify_assets_on_start=True,
        )
        results["load"] = {"seconds": round(time.perf_counter() - began, 2),
                           "rss_mb_after_load": round(_tree_rss_mb(os.getpid()), 1),
                           "handshake_keys": sorted(getattr(engine, "_handshake", {}) or {})[:12]}
        print("engine ready in %.1fs" % results["load"]["seconds"])

        for size in stages:
            stage: dict[str, Any] = {
                "stage": size, "cases_selected": len(cases[:size]), "cases_attempted": 0,
                "cases_completed": 0, "cases_failed": 0,
                "cases_with_zero_valid_candidates": 0, "cases_with_fewer_than_5_candidates": 0,
                "cases_with_fewer_than_10_candidates": 0, "cases_with_fewer_than_20_candidates": 0,
                "sid_generated_total": 0, "sid_parse_failures": 0, "sid_mapping_failures": 0,
                "duplicate_items_filtered": 0, "seen_items_filtered": 0,
                "final_valid_candidates_total": 0, "modes": {}, "invalid_ids": 0, "pad_ids": 0,
                "target_ids_fed_to_model": 0, "hits": {str(k): 0 for k in k_values},
                "per_case": [], "seconds": 0.0, "rss_mb_peak": 0.0,
            }
            stage_started = time.perf_counter()
            for case in cases[:size]:
                stage["cases_attempted"] += 1
                if case.test_target in list(case.test_history):
                    stage["target_ids_fed_to_model"] += 1
                case_started = time.perf_counter()
                try:
                    result = engine.recommend(list(case.test_history), k=max(k_values),
                                              mode="CERTIFIED")
                except Exception as exc:  # noqa: BLE001 - counted, never hidden
                    stage["cases_failed"] += 1
                    stage["per_case"].append({"case": case.user_int_id, "error": type(exc).__name__,
                                              "seconds": round(time.perf_counter() - case_started, 2)})
                    continue
                elapsed = time.perf_counter() - case_started
                mode = str(getattr(result, "retrieval_mode", "UNKNOWN"))
                stage["modes"][mode] = stage["modes"].get(mode, 0) + 1
                rows = list(getattr(result, "recommendations", ()) or ())
                stage["sid_generated_total"] += len(rows)
                seen = excluded_seen(case.test_history, case.test_target)
                masked = set(case.test_seen)
                candidates: list[int] = []
                for row in rows:
                    item_id = int(getattr(row, "item_id", 0) or 0)
                    if item_id < 1 or item_id > num_items:
                        stage["invalid_ids"] += 1
                        continue
                    if item_id == 0:
                        stage["pad_ids"] += 1
                        continue
                    if item_id in candidates:
                        stage["duplicate_items_filtered"] += 1
                        continue
                    if item_id in masked:
                        stage["seen_items_filtered"] += 1
                        continue
                    if item_id in seen:      # the rankable positive is never filtered out
                        continue
                    candidates.append(item_id)
                stage["final_valid_candidates_total"] += len(candidates)
                if not candidates:
                    stage["cases_with_zero_valid_candidates"] += 1
                if len(candidates) < 5:
                    stage["cases_with_fewer_than_5_candidates"] += 1
                if len(candidates) < 10:
                    stage["cases_with_fewer_than_10_candidates"] += 1
                if len(candidates) < 20:
                    stage["cases_with_fewer_than_20_candidates"] += 1
                rank = candidates.index(case.test_target) + 1 if case.test_target in candidates else None
                if rank is not None:
                    for k in k_values:
                        stage["hits"][str(k)] += int(hit_at_k(rank, k))
                stage["cases_completed"] += 1
                stage["per_case"].append({
                    "case": case.user_int_id, "mode": mode, "generated": len(rows),
                    "valid_candidates": len(candidates), "rank": rank,
                    "seconds": round(elapsed, 2),
                    "timeout_risk": elapsed > args.per_case_seconds,
                })
                stage["rss_mb_peak"] = max(stage["rss_mb_peak"], round(_tree_rss_mb(os.getpid()), 1))
                if _available_mb() < args.mem_floor_mb:
                    stage["aborted"] = f"mem_available below {args.mem_floor_mb} MB"
                    break
            stage["seconds"] = round(time.perf_counter() - stage_started, 2)
            scored = stage["cases_attempted"] or 1
            stage["generative_topk_hr_at_k"] = {
                str(k): round(stage["hits"][str(k)] / scored, 6) for k in k_values}
            stage["generative_topk_ndcg_at_k"] = {
                str(k): round(sum(ndcg_at_k(row["rank"], k) for row in stage["per_case"]
                                  if row.get("rank")) / scored, 6) for k in k_values}
            stage["denominator"] = {"all_preselected_cases": stage["cases_attempted"],
                                    "completed_only": stage["cases_completed"]}
            (out / f"stage_{size}.json").write_text(json.dumps(stage, indent=2), encoding="utf-8")
            results.setdefault("stages", {})[str(size)] = {
                key: value for key, value in stage.items() if key != "per_case"}
            print(json.dumps({k: v for k, v in stage.items() if k != "per_case"}, indent=2))
    finally:
        for name in ("close", "shutdown"):
            closer = getattr(locals().get("engine", None), name, None)
            if callable(closer):
                try:
                    closer()
                    results["worker_closed_via"] = name
                except Exception:  # noqa: BLE001
                    results["worker_closed_via"] = f"{name} raised"
                break
        worker = None

    results["resources"] = {"rss_tree_mb_end": round(_tree_rss_mb(os.getpid()), 1),
                            "mem_available_mb_end": round(_available_mb(), 1),
                            "worker_instances": 1 if results.get("load") else 0}
    results["hash_after"] = {"checkpoint_bytes": (CKPT_DIR / "tiger.pt").stat().st_size,
                             "sid_bytes": (SID_DIR / "semantic_ids.json").stat().st_size}
    (out / "gate_b3_summary.json").write_text(json.dumps(results, indent=2, default=str),
                                             encoding="utf-8")
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
