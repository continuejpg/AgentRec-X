"""E-04B-2 Gate B-3 final fixed-configuration diagnostic: full candidate chain per case.

Frozen exactly as Gate B-3: same 100 protocol cases in the same order, same checkpoint/SID artifacts,
CERTIFIED mode, beam=512, prefix_budget=200000, max_hist_items=20, k_max=20, no tuning, no APPROXIMATE.

For every case it records the raw worker candidates (item_id / parent_asin / SID / rank / score), the
candidate list before and after the frozen `excluded_seen` mask, the final top-20, whether the target
appears in the raw or masked list, its rank at each stage, and the per-case timing.  Fields the engine
does not expose are written as `UNAVAILABLE`, never invented.

Outputs (new directory, never overwriting an archive):
    GATE_B3_FULL_CANDIDATES.jsonl   one record per case
    final_diag_summary.json         aggregates and the case-29 structural check
"""

from __future__ import annotations

import argparse
import json
import os
import resource
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

FROZEN = {"mode": "CERTIFIED", "allow_mode_downgrade": False, "k_max": 20, "max_hist_items": 20,
          "beam": 512, "prefix_budget": 200_000, "batch_prefixes": 256, "device": "cpu",
          "k_requested": 20, "cases": 100, "seed": 20261015}


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = base.with_name(f"{base.name}_r{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free directory next to {base}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate B-3 final fixed-config diagnostic")
    parser.add_argument("--out", type=Path, default=REPO / "runs" / "e04b_gate_b3_final_diag")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    out = _unique_dir(args.out)
    out.mkdir(parents=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    from recommendation.inference.tiger_engine import _MappingIdentity, TigerServingEngine, TigerWorkerConfig
    from recommendation.evaluation.metrics import excluded_seen, hit_at_k

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
        if len(cases) >= args.cases:
            break

    records: list[dict[str, Any]] = []
    engine = None
    summary: dict[str, Any] = {"frozen": FROZEN, "cases_rebuilt": len(cases),
                               "engine_start_wall": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        t0 = time.perf_counter()
        engine = TigerServingEngine(
            identity=identity,
            config=TigerWorkerConfig(checkpoint_dir=CKPT_DIR, sid_dir=SID_DIR,
                                     catalogue_dir=HANDOFF, device="cpu"),
            mode="CERTIFIED", allow_mode_downgrade=False,
            startup_timeout=args.startup_timeout, request_timeout=args.request_timeout,
            verify_assets_on_start=True,
        )
        summary["engine_construct_seconds"] = round(time.perf_counter() - t0, 3)

        hits = {5: 0, 10: 0, 20: 0}
        raw_contains_target = masked_contains_target = 0
        case29: dict[str, Any] | None = None
        for case in cases:
            started = time.perf_counter()
            result = engine.recommend(list(case.test_history), k=int(FROZEN["k_requested"]),
                                      mode="CERTIFIED")
            elapsed = time.perf_counter() - started
            rows = list(getattr(result, "recommendations", ()) or [])
            raw = [{"item_id": int(getattr(r, "item_id", 0) or 0),
                    "parent_asin": getattr(r, "parent_asin", None),
                    "sid": getattr(r, "semantic_id", getattr(r, "sid", None)),
                    "rank": getattr(r, "rank", None),
                    "score": getattr(r, "score", None)} for r in rows]
            mask = excluded_seen(case.test_history, case.test_target)
            seen_mask = set(case.test_seen)
            before = [row["item_id"] for row in raw if 1 <= row["item_id"] <= num_items]
            after_mask: list[int] = []
            for item_id in before:
                if item_id in seen_mask and item_id != case.test_target:
                    continue
                if item_id in after_mask:
                    continue
                after_mask.append(item_id)
            top20 = after_mask[:20]
            raw_ids = [row["item_id"] for row in raw]
            in_raw = case.test_target in raw_ids
            in_after = case.test_target in after_mask
            if in_raw:
                raw_contains_target += 1
            if in_after:
                masked_contains_target += 1
            rank_after = after_mask.index(case.test_target) + 1 if in_after else None
            if rank_after:
                for k in hits:
                    hits[k] += int(hit_at_k(rank_after, k))
            record = {
                "case_id": case.user_int_id,
                "test_target": case.test_target,
                "history_items": list(case.test_history),
                "history_length": len(case.test_history),
                "target_in_history": case.test_target in case.test_history,
                "target_in_seen_mask": case.test_target in seen_mask,
                "target_in_mask_used_for_ranking": case.test_target in mask,
                "raw_candidates": raw,
                "candidates_before_mask": before,
                "candidates_after_mask": after_mask,
                "final_top20": top20,
                "target_in_raw_candidates": in_raw,
                "target_rank_before_mask": (before.index(case.test_target) + 1) if case.test_target in before else None,
                "target_rank_after_mask": rank_after,
                "candidate_count": len(after_mask),
                "retrieval_mode": str(getattr(result, "retrieval_mode", "UNKNOWN")),
                "prefix_budget_exhausted": getattr(result, "prefix_budget_exhausted", "UNAVAILABLE"),
                "execution_time": round(elapsed, 3),
            }
            records.append(record)
            if case.user_int_id == 29:
                case29 = {k: v for k, v in record.items() if k != "raw_candidates"}
        (out / "GATE_B3_FULL_CANDIDATES.jsonl").write_text(
            "\n".join(json.dumps(record, default=str) for record in records) + "\n", encoding="utf-8")
        summary.update({
            "cases_executed": len(records),
            "modes": {mode: sum(1 for r in records if r["retrieval_mode"] == mode)
                      for mode in {r["retrieval_mode"] for r in records}},
            "raw_contains_target": raw_contains_target,
            "after_mask_contains_target": masked_contains_target,
            "generative_topk_hits": {str(k): hits[k] for k in sorted(hits)},
            "generative_topk_hr_at_k": {str(k): round(hits[k] / max(len(records), 1), 6) for k in sorted(hits)},
            "total_seconds": round(sum(r["execution_time"] for r in records), 2),
            "per_case_seconds_min_median_max": [
                min(r["execution_time"] for r in records),
                sorted(r["execution_time"] for r in records)[len(records) // 2],
                max(r["execution_time"] for r in records)],
            "case29_check": case29,
            "rss_mb_end": round(_rss_mb(), 1),
        })
    finally:
        closer = getattr(engine, "close", None)
        if callable(closer):
            try:
                closer()
                summary["worker_closed"] = True
            except Exception:  # noqa: BLE001
                summary["worker_closed"] = "close raised"
    (out / "final_diag_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "case29_check"}, indent=2, default=str))
    print("case29:", json.dumps(summary.get("case29_check"), default=str)[:600])
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
