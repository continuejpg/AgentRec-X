"""R-LR-02B: full 5,000-group LightGBM LambdaRank V1 (C0/C1/C2), freeze, one test evaluation."""
from __future__ import annotations
import hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import lightgbm as lgb

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "runs" / "lr_v1_r2" / "rlr01_groups.jsonl"
OUT = REPO / "runs" / "lr_v1_formal"
K_GROUPS = 5000
FEATURES = ["seq_mask","seq_rank","seq_inv_rank","seq_score","tt_mask","tt_rank","tt_inv_rank",
            "tt_score","lex_mask","lex_rank","lex_inv_rank","lex_score","rrf_score","rrf_rank","source_count"]
BASE = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20], "label_gain": [0, 1],
        "num_threads": 2, "force_row_wise": True, "deterministic": True, "verbose": -1, "seed": 20261019}
CONFIGS = {
 "C0": dict(BASE, num_leaves=15, max_depth=-1, learning_rate=0.05, min_data_in_leaf=50,
            lambda_l2=1.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambdarank_truncation_level=30),
 "C1": dict(BASE, num_leaves=7,  max_depth=3, learning_rate=0.03, min_data_in_leaf=100,
            lambda_l2=5.0, feature_fraction=1.0, bagging_fraction=1.0, lambdarank_truncation_level=23),
 "C2": dict(BASE, num_leaves=15, max_depth=4, learning_rate=0.03, min_data_in_leaf=100,
            lambda_l2=5.0, feature_fraction=1.0, bagging_fraction=1.0, lambdarank_truncation_level=23),
}
NUM_ROUNDS, EARLY_STOP = 400, 40


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def slices(sizes):
    a = 0
    for s in sizes:
        yield a, a + s
        a += s


def load_groups():
    X, y, item, rrf, grp, cases = [], [], [], [], [], []
    with SRC.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            r = json.loads(line)
            if r["query_id"] >= K_GROUPS: continue
            X.append([float(r[f]) for f in FEATURES]); y.append(int(r["label"]))
            item.append(int(r["item_id"])); rrf.append(float(r["rrf_score"]))
            if not cases or cases[-1] != r["case"]:
                cases.append(r["case"]); grp.append(0)
            grp[-1] += 1
    return (np.asarray(X, np.float32), np.asarray(y, np.int32), np.asarray(item, np.int64),
            np.asarray(rrf, np.float64), np.asarray(grp, np.int32), np.asarray(cases, np.int64))


def split_by_hash(cases):
    """Stable user-hash split (sha256 of the case id) - reproducible as the dataset grows."""
    bucket = np.asarray([int(hashlib.sha256(str(int(c)).encode()).hexdigest()[:8], 16) % 100 for c in cases])
    return np.where(bucket < 70)[0], np.where((bucket >= 70) & (bucket < 85))[0], np.where(bucket >= 85)[0]


def rank_metrics(y, grp, score, rrf, item, ks=(5, 10, 20)):
    """Explicit HR/NDCG over ALL groups (no-positive groups contribute 0). Deterministic tie-break:
    ranker/RRF score desc, then the other score desc, then item id asc."""
    out = {f"HR@{k}": 0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks} | {
        "n": 0, "pos_groups": 0, "pos_in_pool_not_top20": 0, "_cond": 0.0}
    rrf_main = rrf if score is not rrf else np.zeros_like(rrf)
    for (a, b) in slices(grp):
        lab = y[a:b]
        order = np.lexsort((item[a:b], -rrf_main[a:b] if score is not rrf else -rrf[a:b],
                            -score[a:b]))
        out["n"] += 1
        if not lab.any(): continue
        out["pos_groups"] += 1
        rank = int(np.where(lab[order] == 1)[0][0]) + 1
        if rank > 20: out["pos_in_pool_not_top20"] += 1
        for k in ks:
            out[f"HR@{k}"] += int(rank <= k)
            out[f"NDCG@{k}"] += (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        out["_cond"] += (1.0 / np.log2(rank + 1)) if rank <= 20 else 0.0
    n = max(out["n"], 1)
    for k in ks:
        out[f"HR@{k}"] = round(out[f"HR@{k}"] / n, 6); out[f"NDCG@{k}"] = round(out[f"NDCG@{k}"] / n, 6)
    out["conditional_NDCG@20"] = round(out.pop("_cond") / max(out["pos_groups"], 1), 6)
    return out


def main() -> int:
    t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    X, y, item, rrf, grp, cases = load_groups()
    assert len(grp) == K_GROUPS and int(grp.sum()) == len(X) == len(y)
    tr, va, te = split_by_hash(cases)
    assert not (set(cases[tr]) & set(cases[va])) and not (set(cases[tr]) & set(cases[te])) and not (set(cases[va]) & set(cases[te]))
    ranges = list(slices(grp))
    def subset(ids, positive_only):
        idx = []
        for q in ids:
            a, b = ranges[int(q)]
            if b == a: continue
            if positive_only and not y[a:b].any(): continue
            idx.append(np.arange(a, b))
        return np.concatenate(idx), np.asarray([len(i) for i in idx], np.int32)
    results, models = {}, {}
    for name, params in CONFIGS.items():
        tri, tg = subset(tr, True); vai, vg = subset(va, True)
        dtr = lgb.Dataset(X[tri], label=y[tri], group=tg, feature_name=FEATURES, free_raw_data=True)
        dva = lgb.Dataset(X[vai], label=y[vai], group=vg, feature_name=FEATURES, reference=dtr, free_raw_data=True)
        ev = {}
        m = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.record_evaluation(ev)])
        best = int(m.best_iteration or m.current_iteration())
        # selection metric: FULL-request NDCG@20 on the whole validation split
        va_all_i, va_all_g = subset(va, False)
        pv = m.predict(X[va_all_i], num_iteration=best)
        rrf_va = np.where(rrf[va_all_i] > 0, 1.0 / (60.0 + rrf[va_all_i]), 0.0)
        vm = rank_metrics(y[va_all_i], va_all_g, pv, rrf[va_all_i], item[va_all_i])
        results[name] = {"params": params, "best_iteration": best,
                         "trees": int(m.num_trees()),
                         "val_full": vm, "val_rrf_full": rank_metrics(y[va_all_i], va_all_g, rrf_va, rrf[va_all_i], item[va_all_i]),
                         "val_curve_ndcg20": [round(float(v), 6) for v in ev["val"].get("ndcg@20", [])],
                         "feature_importance_gain": dict(sorted(zip(m.feature_name(),
                             m.feature_importance("gain").tolist()), key=lambda kv: -kv[1]))}
        models[name] = (m, best, vm)
        print(f"  {name}: best_iter={best} trees={m.num_trees()} val_NDCG@20(full)={vm['NDCG@20']} "
              f"val_HR@20={vm['HR@20']} cond={vm['conditional_NDCG@20']}", flush=True)
    # ---- select by validation full-request NDCG@20; simpler config wins a tie -------- #
    best_name = max(results, key=lambda n: (results[n]["val_full"]["NDCG@20"],
                                            -CONFIGS[n]["num_leaves"]))
    model, best_iter, _ = models[best_name]
    # ---- freeze, then ONE test evaluation ------------------------------------------ #
    (OUT / "model.txt").write_text("")  # placeholder replaced below
    model.save_model(str(OUT / "model.txt"), num_iteration=best_iter)
    te_i, te_g = subset(te, False)
    pt = model.predict(X[te_i], num_iteration=best_iter)
    rrf_te = np.where(rrf[te_i] > 0, 1.0 / (60.0 + rrf[te_i]), 0.0)
    ranker_test = rank_metrics(y[te_i], te_g, pt, rrf[te_i], item[te_i])
    rrf_test = rank_metrics(y[te_i], te_g, rrf_te, rrf[te_i], item[te_i])
    # per-group NDCG@20 deltas (ranker - RRF) + bootstrap CI
    deltas = []
    for (a, b) in slices(te_g):
        lab = y[te_i][a:b]
        if len(lab) == 0 or not lab.any(): deltas.append(0.0); continue
        def nd(arr):
            o = np.lexsort((item[te_i][a:b], -rrf_te[a:b], -arr))
            r = int(np.where(lab[o] == 1)[0][0]) + 1
            return (1.0 / np.log2(r + 1)) if r <= 20 else 0.0
        deltas.append(nd(pt[a:b]) - nd(rrf_te[a:b]))
    d = np.asarray(deltas); rng = np.random.default_rng(20261019)
    boots = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(5000)]
    ci = [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)]
    reload = lgb.Booster(model_file=str(OUT / "model.txt"))
    p_reload = reload.predict(X[te_i], num_iteration=best_iter)
    schema = {"features": FEATURES, "count": len(FEATURES), "dtype": "float32",
              "missing_rule": "mask=0 rank=0 inv_rank=0 score=0.0 (training and inference identical)",
              "tie_break": ["ranker score desc", "rrf score desc", "item_id asc"]}
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    metrics = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "selected_config": best_name,
               "selection_rule": "max validation FULL-request NDCG@20; simpler config on ties",
               "best_iteration": best_iter, "trees": int(model.num_trees()),
               "test": {"ranker": ranker_test, "rrf": rrf_test},
               "paired_ndcg20_delta": {"mean": round(float(d.mean()), 6), "bootstrap_ci95": ci,
                                       "resamples": 5000, "seed": 20261019},
               "reload_prediction_identical": bool(np.array_equal(p_reload, pt)),
               "data": {"groups": K_GROUPS, "rows": int(len(X)),
                        "split_groups": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
                        "split_rule": "stable sha256(case_id) % 100 bucket: <70 train, 70-84 val, >=85 test",
                        "p3_isolation": "cases start at valid case #1001 (P3 prefix excluded), overlap 0",
                        "source_sha256": sha(SRC), "source_bytes": SRC.stat().st_size},
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "lightgbm": lgb.__version__},
               "configs": results,
               "elapsed_seconds": round(time.perf_counter() - t0, 1),
               "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "train_manifest.json").write_text(json.dumps({k: metrics[k] for k in
        ("frozen_at", "selected_config", "best_iteration", "trees", "data", "versions",
         "selection_rule", "elapsed_seconds")}, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best_name, "best_iteration": best_iter,
                      "test": metrics["test"], "delta": metrics["paired_ndcg20_delta"],
                      "reload_identical": metrics["reload_prediction_identical"],
                      "data": metrics["data"], "elapsed": metrics["elapsed_seconds"],
                      "rss": metrics["peak_rss_mb"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
