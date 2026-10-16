"""Gate C-1 / C-2: single-backend acceptance through the real /v1/recommend route.

Run once per backend with AGENTRECX_FUSION_PROFILE=serving_v1_sequential_only (SASRec) or
serving_v1_two_tower_only (Two-Tower), in its **own process** so model lifetimes are isolated.

Parity follows the *API ordering contract*: the route knows only the caller's history, so the
reference is `sorted_ranking(scores, history=test_history, num_items=N)[:K]` - the same exclusion
rule, no target-based mask.  (Gate B-1's `rank_of_target` used the frozen evaluation mask, which is
a different question; both are kept and labelled.)
"""
from __future__ import annotations

import argparse, json, os, resource, statistics, sys, time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
SASREC_CKPT = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
TT_DIR = REPO / "runs" / "twotower_public_2026_full"
OUT = REPO / "runs" / "gate_c12_service_validation"
PROFILES = {"sasrec": "serving_v1_sequential_only", "two_tower": "serving_v1_two_tower_only",
            "fusion": "serving_v1"}  # P3-C: the existing default profile (sequential+two_tower+lexical)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(PROFILES), required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    backend = args.backend
    os.environ["AGENTRECX_FUSION_PROFILE"] = PROFILES[backend]
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    root = args.out if not args.out.exists() else args.out.with_name(f"{args.out.name}_r2")
    (root / backend).mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient
    from recommendation.api.app import app
    from recommendation.datasets.sasrec import encode_inference_history
    from recommendation.evaluation.metrics import hit_at_k, ndcg_at_k, sorted_ranking
    from recommendation.evaluation.split import split_sequence
    from recommendation.inference.sasrec import InferenceConfig as SasrecConfig, SASRecInferenceEngine
    from recommendation.inference.twotower import TwoTowerInferenceConfig, TwoTowerInferenceEngine
    from experiments.e04b2_gate_a import stream_sequences

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    id_to_asin = {int(v): str(k) for k, v in mappings["item2id"].items()}
    cases = []
    for element in stream_sequences(SEQUENCES):
        items = [int(v) for v in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is not None:
            cases.append(case)
        if len(cases) >= args.cases:
            break

    # --- offline reference by the API ordering contract --------------------- #
    # The fusion path is multi-source by definition, so no single-backend reference exists for it:
    # parity is only meaningful for the two single-source profiles.
    single_source = backend in ("sasrec", "two_tower")
    began = time.perf_counter()
    engine = (SASRecInferenceEngine(SasrecConfig(checkpoint_path=SASREC_CKPT, mappings_path=MAPPINGS,
                                                 device="cpu"))
              if backend == "sasrec" else
              TwoTowerInferenceEngine(TwoTowerInferenceConfig(
                  checkpoint_path=TT_DIR / "checkpoint.pt", categories_path=TT_DIR / "metadata_categories.json",
                  mappings_path=MAPPINGS, device="cpu")) if single_source else None)
    load_seconds = round(time.perf_counter() - began, 2)
    reference: dict[int, list[int]] = {}
    for case in (cases if single_source else ()):
        if backend == "sasrec":
            encoded, _ = engine.encode_history([id_to_asin[i] for i in case.test_history])
        else:
            encoded = encode_inference_history(list(case.test_history), engine.max_seq_len, num_items)
        scores = engine.score_catalog(encoded)
        reference[case.user_int_id] = sorted_ranking(
            scores, history=case.test_history, num_items=num_items)[: args.k]
    del engine

    metrics = {"label": f"exploratory service-level evaluation, n={len(cases)}", "backend": backend,
               "profile": PROFILES[backend], "hits": {}, "HR": {}, "NDCG": {}}
    rounds: list[list[dict[str, Any]]] = []
    latencies: list[float] = []
    with TestClient(app) as client:
        for round_index in (1, 2):
            started = time.perf_counter()
            rows = []
            for case in cases:
                history = [id_to_asin[i] for i in case.test_history]
                t0 = time.perf_counter()
                response = client.post("/v1/recommend", json={"history": history, "k": args.k})
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)
                body = response.json() if response.status_code == 200 else {}
                cards = body.get("recommendations", []) or []
                ids = [int(c["item_id"]) for c in cards]
                rows.append({"case_id": case.user_int_id, "status": response.status_code,
                             "ordered_item_ids": ids,
                             "parent_asins": [c.get("parent_asin") for c in cards],
                             "ranks": [c.get("rank") for c in cards],
                             "scores": [c.get("score") for c in cards],
                             "returned_k": body.get("returned_k"), "test_target": case.test_target,
                             "target_rank": (ids.index(case.test_target) + 1) if case.test_target in ids else None,
                             "schema_ok": all(isinstance(c.get("rank"), int) and isinstance(c.get("item_id"), int)
                                              and isinstance(c.get("parent_asin"), str) for c in cards),
                             "invalid_ids": sum(1 for i in ids if i < 1 or i > num_items),
                             "pad_ids": sum(1 for i in ids if i == 0),
                             "duplicates": len(ids) - len(set(ids)),
                             "history_echo": sum(1 for i in ids if i in set(case.test_history)),
                             "seconds": round(elapsed, 4)})
            (root / backend / f"responses_run{round_index}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            rounds.append(rows)
            metrics[f"round{round_index}_seconds"] = round(time.perf_counter() - started, 2)
        profile_seen = getattr(getattr(app, "state", None), "engine", None)
        metrics["engine_object_profile"] = getattr(profile_seen, "config", None) and getattr(
            profile_seen.config, "profile", None)

    parity_rows, parity_ok, repeat_ok = [], 0, 0
    for index, case in enumerate(cases):
        api_ids = rounds[0][index]["ordered_item_ids"]
        ref_ids = reference.get(case.user_int_id, [])
        same = (api_ids == ref_ids) if single_source else None
        parity_ok += int(bool(same))
        repeat_ok += int(api_ids == rounds[1][index]["ordered_item_ids"])
        parity_rows.append({"case_id": case.user_int_id, "parity": same, "api_top20": api_ids,
                            "offline_top20": ref_ids,
                            "first_difference": next((i for i, (a, b) in enumerate(zip(api_ids, ref_ids))
                                                      if a != b), None)})
    route = rounds[0]
    hits = {k: sum(1 for r in route if r["target_rank"] and r["target_rank"] <= k) for k in (5, 10, 20)}
    metrics["hits"] = {str(k): hits[k] for k in hits}
    metrics["HR"] = {str(k): round(hits[k] / len(cases), 6) for k in hits}
    metrics["NDCG"] = {str(k): round(sum(ndcg_at_k(r["target_rank"], k) for r in route
                                           if r["target_rank"] and r["target_rank"] <= k) / len(cases), 6)
                       for k in hits}
    # the same metric against the *reference* ordering, for an apples-to-apples sanity check
    ref_hits = ({k: sum(1 for c in cases if c.test_target in reference[c.user_int_id][:k]) for k in (5, 10, 20)}
                if single_source else {})
    metrics["reference_hits"] = {str(k): ref_hits[k] for k in ref_hits}
    summary = {
        "backend": backend, "profile": PROFILES[backend], "cases": len(cases), "k": args.k,
        "offline_load_seconds": load_seconds, "hits": metrics["hits"], "HR": metrics["HR"],
        "NDCG": metrics["NDCG"], "reference_hits": metrics["reference_hits"],
        "successes": sum(1 for r in route if r["status"] == 200),
        "schema_ok": sum(1 for r in route if r["schema_ok"]),
        "invalid_ids": sum(r["invalid_ids"] for r in route), "pad_ids": sum(r["pad_ids"] for r in route),
        "duplicates": sum(r["duplicates"] for r in route),
        "history_echo": sum(r["history_echo"] for r in route),
        "non_empty": sum(1 for r in route if r["ordered_item_ids"]),
        "candidate_counts": sorted({r["returned_k"] for r in route}),
        "parity_identical_top20": parity_ok if single_source else "NOT_APPLICABLE (multi-source path)", "repeat_identical_top20": repeat_ok,
        "latency_p50_ms": round(statistics.median(latencies) * 1000, 2),
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000, 2),
        "latency_max_ms": round(max(latencies) * 1000, 2),
        "round_seconds": [metrics["round1_seconds"], metrics["round2_seconds"]],
        "rss_mb_peak": round(_rss_mb(), 1), "exceptions": 0,
        "ordering_contract": "reference = sorted_ranking(score_catalog, history=test_history)[:K] "
                             "(no target-based mask, matching what the route can know)",
    }
    (root / backend / "parity.json").write_text(json.dumps({"rows": parity_rows}, indent=2), encoding="utf-8")
    if single_source:
        (root / backend / "offline_reference.jsonl").write_text(
            "\n".join(json.dumps({"case_id": c.user_int_id, "offline_top20": reference[c.user_int_id]})
                      for c in cases) + "\n", encoding="utf-8")
    else:
        # A multi-source path has no single-backend reference: say so rather than writing an empty file.
        (root / backend / "offline_reference.jsonl").write_text(
            "\n".join(json.dumps({"case_id": c.user_int_id, "offline_top20": None,
                                   "note": "NOT_APPLICABLE: serving_v1 is a multi-source fusion path"})
                      for c in cases) + "\n", encoding="utf-8")
    (root / backend / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / backend / "repeatability.json").write_text(json.dumps({
        "rounds": 2, "cases": len(cases), "identical_ordered_top20": repeat_ok}, indent=2), encoding="utf-8")
    (root / backend / "resources.json").write_text(json.dumps({
        "rss_mb_peak": summary["rss_mb_peak"], "offline_load_seconds": load_seconds,
        "round_seconds": summary["round_seconds"], "latency": {
            "p50_ms": summary["latency_p50_ms"], "p95_ms": summary["latency_p95_ms"],
            "max_ms": summary["latency_max_ms"]}}, indent=2), encoding="utf-8")
    (root / f"{backend}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"archive: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
