"""R-LR-01: materialise grouped ranking candidates for LightGBM LambdaRank.

One query group per prediction point, built ONLY from data visible at that point:
  history = items[:-1]  (frozen temporal_leave_two_out; target = items[-1])
  candidates = union of the three accepted sources' own top-`head` (raw scores kept),
               mapped to the trusted identity namespace, history items removed, deduplicated.
  label = 1 iff the candidate is the next interaction; the target is NEVER inserted.
Cases are taken AFTER the first `--skip` valid cases so the P3/acceptance queue is excluded.

Writes: runs/lr_v1/rlr01_groups.jsonl (one row per candidate), rlr01_groups.npz (X/y/group),
        rlr01_manifest.json (protocol, hashes, coverage, group statistics, feature spec).
No model is trained here; lightgbm is reported as a dependency, not imported.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
MAPPINGS = REPO / "data" / "processed" / "Sports_and_Outdoors_mappings.json"
SEQUENCES = REPO / "data" / "processed" / "Sports_and_Outdoors_sequences.json"
SASREC = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
TT = REPO / "runs" / "twotower_public_2026_full"
OUT = REPO / "runs" / "lr_v1"
HEAD = 100          # per-source recall depth, matching the serving fusion pool
FEATURES = ["seq_mask", "seq_rank", "seq_inv_rank", "seq_score",
            "tt_mask", "tt_rank", "tt_inv_rank", "tt_score",
            "lex_mask", "lex_rank", "lex_inv_rank", "lex_score",
            "rrf_score", "rrf_rank", "source_count"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=1000, help="skip the P3/acceptance prefix")
    ap.add_argument("--cases", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    out = args.out if not args.out.exists() else args.out.with_name(args.out.name + "_r2")
    out.mkdir(parents=True, exist_ok=True)

    from recommendation.control.candidate_ledger import reciprocal_rank_fusion
    from recommendation.datasets.sasrec import encode_inference_history
    from recommendation.evaluation.split import split_sequence
    from recommendation.inference.lexical import CatalogueLexicalIndex, query_text_for
    from recommendation.inference.sasrec import InferenceConfig as SC, SASRecInferenceEngine
    from recommendation.inference.twotower import TwoTowerInferenceConfig as TC, TwoTowerInferenceEngine
    from recommendation.catalog import MetadataIndex
    from recommendation.api.app import DEFAULT_CATALOG
    from experiments.e04b2_gate_a import stream_sequences

    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    id_to_asin = {int(v): str(k) for k, v in mappings["item2id"].items()}
    asin_to_id = {a: i for i, a in id_to_asin.items()}

    began = time.perf_counter()
    seq = SASRecInferenceEngine(SC(checkpoint_path=SASREC, mappings_path=MAPPINGS, device="cpu"))
    tt = TwoTowerInferenceEngine(TC(checkpoint_path=TT / "checkpoint.pt",
                                    categories_path=TT / "metadata_categories.json",
                                    mappings_path=MAPPINGS, device="cpu"))
    records = MetadataIndex.load(DEFAULT_CATALOG).records   # a mapping asin -> record
    lexical = CatalogueLexicalIndex.build(records)
    catalogue = {}
    for key, value in (records.items() if hasattr(records, "items") else ()):
        asin = getattr(value, "parent_asin", None) or (value if isinstance(value, str) else None) or key
        catalogue[str(asin)] = value
    load_seconds = round(time.perf_counter() - began, 2)

    # --- cases: valid, after the P3 prefix ---------------------------------- #
    cases, seen_valid = [], 0
    for element in stream_sequences(SEQUENCES):
        items = [int(v) for v in element.get("item_ids", ())]
        case = split_sequence(str(element.get("user_id", "")), items, num_items,
                              int(element.get("user_int_id", -1)))
        if case is None: continue
        seen_valid += 1
        if seen_valid <= args.skip: continue
        cases.append(case)
        if len(cases) >= args.cases: break
    skipped_prefix = min(seen_valid, args.skip)

    rows_path = out / "rlr01_groups.jsonl"
    stats: dict[str, Any] = {"protocol": "agentrecx.eval_protocol.v1 / temporal_leave_two_out",
                             "head_depth": HEAD, "features": FEATURES,
                             "p3_excluded_prefix": args.skip, "cases_requested": args.cases,
                             "cases_available": len(cases), "load_seconds": load_seconds,
                             "queues": {"sequences_sha256": sha(SEQUENCES), "mappings_sha256": sha(MAPPINGS)},
                             "checkpoints": {"sasrec_sha256": sha(SASREC),
                                             "two_tower_sha256": sha(TT / "checkpoint.pt")},
                             "feature_rules": {"missing": "0 (mask=0, rank=0, inv_rank=0, score=0.0)",
                                               "label": "1 iff candidate is the next interaction; never inserted",
                                               "history": "removed from candidates before features"},
                             "dependency": {"lightgbm": "ABSENT (training dependency, not installed here)",
                                            "numpy": "present"}}
    X, y, group_sizes, groups_with_pos, rows = [], [], [], 0, 0
    with rows_path.open("w", encoding="utf-8") as fh:
        for qi, case in enumerate(cases):
            history = list(case.test_history)
            hist_asins = [id_to_asin[i] for i in history if i in id_to_asin]
            hist_set = set(history)
            per_source: dict[str, dict[str, tuple[int, float]]] = {}
            try:
                s_res = seq.recommend(hist_asins, k=HEAD)
                per_source["seq"] = {it.parent_asin: (it.rank, float(it.score))
                                     for it in s_res.recommendations}
            except Exception:
                per_source["seq"] = {}
            try:
                encoded = encode_inference_history(history, tt.max_seq_len, num_items)
                t_res = tt.recommend(history, k=HEAD)
                per_source["tt"] = {it.parent_asin: (it.rank, float(it.score))
                                    for it in t_res.recommendations}
            except Exception:
                per_source["tt"] = {}
            try:
                usable = [catalogue[a] for a in hist_asins[-5:]
                          if a in catalogue and getattr(catalogue[a], "title", None)]
                q = query_text_for(usable) if usable else ""
                per_source["lex"] = {asin: (i + 1, float(score))
                                     for i, (asin, score) in enumerate(lexical.scored(q, limit=HEAD))}
            except Exception:
                per_source["lex"] = {}
            rankings = [[a for a, _ in sorted(d.items(), key=lambda kv: kv[1][0])]
                        for d in (per_source["seq"], per_source["tt"], per_source["lex"])]
            fused = dict(reciprocal_rank_fusion([r for r in rankings if r], k=60))
            pool: list[str] = []
            for d in (per_source["seq"], per_source["tt"], per_source["lex"]):
                for a in sorted(d, key=lambda a: d[a][0]):
                    if a in asin_to_id and asin_to_id[a] not in hist_set and a not in pool:
                        pool.append(a)
            if not pool:
                group_sizes.append(0); continue
            fused_order = [a for a, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
            has_pos = False
            for asin in pool:
                iid = asin_to_id[asin]
                row = {"query_id": qi, "case": case.user_int_id, "item_id": iid,
                       "parent_asin": asin, "label": int(iid == case.test_target)}
                for name, prefix in (("seq", "seq"), ("tt", "tt"), ("lex", "lex")):
                    rank, score = per_source[name].get(asin, (0, 0.0))
                    row[f"{prefix}_mask"] = int(rank > 0)
                    row[f"{prefix}_rank"] = rank
                    row[f"{prefix}_inv_rank"] = round(1.0 / rank, 6) if rank else 0.0
                    row[f"{prefix}_score"] = round(score, 6)
                row["rrf_score"] = round(float(fused.get(asin, 0.0)), 6)
                row["rrf_rank"] = (fused_order.index(asin) + 1) if asin in fused_order else 0
                row["source_count"] = sum(row[f"{p}_mask"] for p in ("seq", "tt", "lex"))
                has_pos = has_pos or bool(row["label"])
                X.append([float(row[f]) for f in FEATURES]); y.append(row["label"])
                fh.write(json.dumps(row) + "\n"); rows += 1
            group_sizes.append(len(pool))
            groups_with_pos += int(has_pos)
            if qi % 500 == 0:
                print(f"  query {qi}: groups={len(group_sizes)} rows={rows} "
                      f"groups_with_pos={groups_with_pos}", flush=True)
    import numpy as np
    np.savez_compressed(out / "rlr01_groups.npz", X=np.asarray(X, dtype=np.float32),
                        y=np.asarray(y, dtype=np.int8), group=np.asarray(group_sizes, dtype=np.int32))
    stats.update({"candidate_rows": rows, "groups": len(group_sizes),
                  "groups_with_positive": groups_with_pos,
                  "groups_without_positive": len(group_sizes) - groups_with_pos,
                  "positive_rows": int(sum(y)),
                  "sum_group_sizes": int(sum(group_sizes)),
                  "sum_equals_rows": int(sum(group_sizes)) == rows,
                  "coverage_groups_with_positive": round(groups_with_pos / max(len(group_sizes), 1), 6),
                  "elapsed_seconds": round(time.perf_counter() - began, 1),
                  "artifacts": {"jsonl": str(rows_path.relative_to(REPO)),
                                "npz": str((out / "rlr01_groups.npz").relative_to(REPO))}})
    (out / "rlr01_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k not in ("feature_rules",)}, indent=2)[:2200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
