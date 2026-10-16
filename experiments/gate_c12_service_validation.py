"""Gate C-1 / C-2: real /v1/recommend service validation for SASRec and Two-Tower.

What is exercised
-----------------
* The real FastAPI application is started through its lifespan (models are loaded by the accepted
  startup path); requests go through `POST /v1/recommend` with the official schema.
* The production serving profile is `serving_v1` (sequential + two-tower + lexical fusion); the app
  builds **no** single-backend profile, so the route's own ranking is a fusion ranking and is
  recorded as such - never attributed to one backend.
* Per-backend evidence comes from the **app's own engines** (real weights, no mock): the fusion
  engine's `_sequential_ranking` / `_two_tower_ranking` are read-only accessors of the accepted
  pipeline, and each backend additionally has a full-catalogue offline reference computed with the
  frozen protocol (`score_catalog` + `sorted_ranking` + `excluded_seen`).

Nothing here modifies production code, model weights, the queue, the id mapping or the schema.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
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
K = 20


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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
    raise RuntimeError(f"no free directory next to {base}")


def _normalise_items(raw: Any) -> list[dict[str, Any]]:
    """Accept either objects or tuples from an internal ranking accessor."""
    items: list[dict[str, Any]] = []
    for row in raw or ():
        if isinstance(row, dict):
            items.append({"item_id": int(row.get("item_id", 0) or 0),
                          "parent_asin": row.get("parent_asin"),
                          "score": row.get("score"), "rank": row.get("rank")})
        elif isinstance(row, (tuple, list)):
            items.append({"item_id": int(row[0]), "parent_asin": row[1] if len(row) > 1 else None,
                          "score": row[2] if len(row) > 2 else None, "rank": None})
        else:
            items.append({"item_id": int(getattr(row, "item_id", 0) or 0),
                          "parent_asin": getattr(row, "parent_asin", None),
                          "score": getattr(row, "score", None),
                          "rank": getattr(row, "rank", None)})
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate C-1/C-2 real service validation")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args(argv)
    out = _unique_dir(OUT)
    for sub in ("route", "sasrec", "two_tower"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    from fastapi.testclient import TestClient
    from recommendation.api.app import app
    from recommendation.evaluation.metrics import excluded_seen, hit_at_k, ndcg_at_k, sorted_ranking
    from recommendation.evaluation.split import split_sequence
    from recommendation.inference.sasrec import InferenceConfig as SasrecConfig
    from recommendation.inference.sasrec import SASRecInferenceEngine
    from recommendation.inference.twotower import TwoTowerInferenceConfig, TwoTowerInferenceEngine
    from recommendation.datasets.sasrec import encode_inference_history
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

    manifest = {
        "gate": "C-1 + C-2", "cases": len(cases), "k": args.k,
        "protocol": "agentrecx.eval_protocol.v1", "split": "temporal_leave_two_out",
        "seed": 20261015, "served_profile_env": os.environ.get("AGENTRECX_SERVING_FUSION", "auto"),
        "queue_sha256": _sha256(SEQUENCES), "mappings_sha256": _sha256(MAPPINGS),
        "sasrec_sha256": _sha256(SASREC_CKPT), "two_tower_sha256": _sha256(TT_DIR / "checkpoint.pt"),
        "versions": {"python": sys.version.split()[0]},
        "route": "POST /v1/recommend (official schema); the served profile is a fusion profile, "
                 "so route-level output is never attributed to a single backend",
        "per_backend_evidence": "the app's own engines (real weights) + frozen offline reference",
    }
    try:
        import torch
        manifest["versions"]["torch"] = torch.__version__
        manifest["versions"]["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        import fastapi
        manifest["versions"]["fastapi"] = fastapi.__version__
    except Exception:  # noqa: BLE001
        pass
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {"manifest": manifest}
    offline = {}
    offline_seconds = {}
    for name, builder in (
        ("sasrec", lambda: SASRecInferenceEngine(
            SasrecConfig(checkpoint_path=SASREC_CKPT, mappings_path=MAPPINGS, device="cpu"))),
        ("two_tower", lambda: TwoTowerInferenceEngine(
            TwoTowerInferenceConfig(checkpoint_path=TT_DIR / "checkpoint.pt",
                                    categories_path=TT_DIR / "metadata_categories.json",
                                    mappings_path=MAPPINGS, device="cpu"))),
    ):
        began = time.perf_counter()
        engine = builder()
        offline_seconds[name] = {"load_seconds": round(time.perf_counter() - began, 2),
                                 "is_ready": bool(engine.is_ready())}
        tops = {}
        for case in cases:
            if name == "sasrec":
                # SASRec's accessor returns (encoded, truncated); Two-Tower's encoder returns the
                # encoded history itself.  Branch explicitly instead of guessing the shape.
                encoded, _truncated = engine.encode_history(
                    [id_to_asin[i] for i in case.test_history])
            else:
                encoded = encode_inference_history(list(case.test_history), engine.max_seq_len,
                                                   num_items)
            scores = engine.score_catalog(encoded)
            ranking = sorted_ranking(scores, history=case.test_history, num_items=num_items)
            tops[case.user_int_id] = rankings_tail(
                ranking, case, args.k, excluded_seen(case.test_history, case.test_target))
        offline[name] = tops
        del engine
    summary["offline_reference"] = {name: {"load": offline_seconds[name],
                                           "cases": len(offline[name])} for name in offline}

    with TestClient(app) as client:
        state = app.state
        fusion = getattr(state, "engine", None)
        summary["service"] = {
            "model_loaded": bool(getattr(state, "model_loaded", False)),
            "engine_class": type(fusion).__name__ if fusion is not None else None,
            "single_backend_profile_supported": False,
            "note": "no single-backend serving profile exists; per-backend evidence uses the app's "
                    "own engines, route-level metrics describe the served fusion profile",
        }
        route_rounds, per_source = [], {"sasrec": [], "two_tower": []}
        latencies: list[float] = []
        for round_index in (1, 2):
            round_started = time.perf_counter()
            route_rows = []
            for case in cases:
                history = [id_to_asin[i] for i in case.test_history]
                began = time.perf_counter()
                response = client.post("/v1/recommend", json={"history": history, "k": args.k})
                elapsed = time.perf_counter() - began
                latencies.append(elapsed)
                body = response.json() if response.status_code == 200 else None
                cards = (body or {}).get("recommendations", []) or []
                ids = [int(card["item_id"]) for card in cards]
                route_rows.append({
                    "case_id": case.user_int_id, "status": response.status_code,
                    "history_length_api": (body or {}).get("history_length"),
                    "requested_k": (body or {}).get("requested_k"),
                    "returned_k": (body or {}).get("returned_k"),
                    "ordered_item_ids": ids,
                    "parent_asins": [card.get("parent_asin") for card in cards],
                    "ranks": [card.get("rank") for card in cards],
                    "scores": [card.get("score") for card in cards],
                    "test_target": case.test_target,
                    "target_rank": (ids.index(case.test_target) + 1) if case.test_target in ids else None,
                    "seconds": round(elapsed, 4),
                    "schema_ok": all(isinstance(card.get("rank"), int)
                                     and isinstance(card.get("item_id"), int)
                                     and isinstance(card.get("parent_asin"), str)
                                     for card in cards),
                    "invalid_ids": sum(1 for i in ids if i < 1 or i > num_items),
                    "pad_ids": sum(1 for i in ids if i == 0),
                    "duplicates": len(ids) - len(set(ids)),
                    "seen_violations": sum(1 for i in ids if i in case.test_seen),
                })
            (out / "route" / f"responses_run{round_index}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in route_rows) + "\n", encoding="utf-8")
            route_rounds.append({"round": round_index, "rows": route_rows,
                                 "seconds": round(time.perf_counter() - round_started, 2)})
            for name, accessor in (("sasrec", "_sequential_ranking"), ("two_tower", "_two_tower_ranking")):
                method = getattr(fusion, accessor, None)
                rows = []
                for case in cases:
                    history = [id_to_asin[i] for i in case.test_history]
                    raw = method(history) if callable(method) else None
                    items = _normalise_items(raw)[: args.k]
                    rows.append({"case_id": case.user_int_id,
                                 "ordered_item_ids": [item["item_id"] for item in items],
                                 "accessor_available": callable(method)})
                (out / name / f"engine_top20_run{round_index}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                per_source[name].append(rows)

    for name in ("sasrec", "two_tower"):
        ref = offline[name]
        parity_rows, repeat_ok, parity_ok = [], 0, 0
        for index, case in enumerate(cases):
            api_ids = per_source[name][0][index]["ordered_item_ids"]
            ids2 = per_source[name][1][index]["ordered_item_ids"]
            ref_ids = ref[case.user_int_id]
            if api_ids == ids2:
                repeat_ok += 1
            same = api_ids == ref_ids
            if same:
                parity_ok += 1
            parity_rows.append({"case_id": case.user_int_id, "parity": same,
                                "engine_top20": api_ids, "offline_top20": ref_ids,
                                "first_difference": next((i for i, (a, b) in
                                                          enumerate(zip(api_ids, ref_ids)) if a != b),
                                                         None)})
        available = per_source[name][0][0]["accessor_available"] if cases else False
        (out / name / "offline_reference.jsonl").write_text(
            "\n".join(json.dumps({"case_id": c.user_int_id, "offline_top20": ref[c.user_int_id]})
                      for c in cases) + "\n", encoding="utf-8")
        (out / name / "parity.json").write_text(json.dumps({
            "accessor": "_sequential_ranking" if name == "sasrec" else "_two_tower_ranking",
            "accessor_available": available,
            "cases": len(cases), "identical_top20": parity_ok,
            "protocol_note": "same weights / id namespace / history / K / seen mask; the engine "
                             "ranking is the app-built accepted pipeline, not a route response",
            "rows": parity_rows}, indent=2), encoding="utf-8")
        (out / name / "repeatability.json").write_text(json.dumps({
            "rounds": 2, "cases": len(cases), "identical_ordered_top20": repeat_ok,
            "note": "two independent calls through the app's own engine"}, indent=2), encoding="utf-8")
        hits = {k: sum(1 for c in cases
                       if ref[c.user_int_id] and c.test_target in ref[c.user_int_id][:k])
                for k in (5, 10, 20)}
        (out / name / "metrics.json").write_text(json.dumps({
            "label": "exploratory service-level evaluation, n=%d" % len(cases),
            "basis": "offline full-catalogue reference (frozen protocol)",
            "hits": {str(k): hits[k] for k in hits},
            "HR": {str(k): round(hits[k] / len(cases), 6) for k in hits},
            "NDCG": {str(k): round(sum(ndcg_at_k(ref[c.user_int_id].index(c.test_target) + 1, k)
                                       for c in cases if c.test_target in ref[c.user_int_id][:k])
                                   / len(cases), 6) for k in hits},
            "denominator": len(cases)}, indent=2), encoding="utf-8")

    route_rows = route_rounds[0]["rows"]
    metrics = {
        "label": "exploratory service-level evaluation, n=%d" % len(cases),
        "basis": "real /v1/recommend responses (served fusion profile)",
        "successes": sum(1 for row in route_rows if row["status"] == 200),
        "schema_ok": sum(1 for row in route_rows if row["schema_ok"]),
        "invalid_ids": sum(row["invalid_ids"] for row in route_rows),
        "pad_ids": sum(row["pad_ids"] for row in route_rows),
        "duplicates": sum(row["duplicates"] for row in route_rows),
        "seen_violations": sum(row["seen_violations"] for row in route_rows),
        "hits": {str(k): sum(1 for row in route_rows if row["target_rank"] and row["target_rank"] <= k)
                 for k in (5, 10, 20)},
        "non_empty_responses": sum(1 for row in route_rows if row["ordered_item_ids"]),
        "candidate_counts": sorted({row["returned_k"] for row in route_rows}),
    }
    metrics["HR"] = {k: round(v / len(cases), 6) for k, v in metrics["hits"].items()}
    repeat_route = sum(1 for a, b in zip(route_rounds[0]["rows"], route_rounds[1]["rows"])
                       if a["ordered_item_ids"] == b["ordered_item_ids"])
    latencies.sort()
    summary.update({
        "route_metrics": metrics,
        "route_repeatability": {"cases": len(cases), "identical_ordered_item_ids": repeat_route,
                                "round_seconds": [route_rounds[0]["seconds"], route_rounds[1]["seconds"]]},
        "resources": {"rss_mb_end": round(_rss_mb(), 1),
                      "latency_p50_ms": round(statistics.median(latencies) * 1000, 2),
                      "latency_p95_ms": round(latencies[int(len(latencies) * 0.95) - 1] * 1000, 2),
                      "latency_max_ms": round(max(latencies) * 1000, 2)},
    })
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "offline_reference"}, indent=2,
                     default=str)[:2600])
    print(f"\narchive: {out}")
    return 0


def rankings_tail(ranking: Sequence[int], case: Any, k: int, excluded: Any) -> list[int]:
    """The offline top-k after the frozen mask (the positive stays rankable)."""
    out: list[int] = []
    for item_id in ranking:
        if item_id in excluded:
            continue
        out.append(item_id)
        if len(out) >= k:
            break
    return out


if __name__ == "__main__":
    raise SystemExit(main())
