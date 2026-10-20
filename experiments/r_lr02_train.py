"""R-LR-02: train a real LightGBM LambdaRank on the frozen 2,000-group snapshot.

Reads ONLY runs/lr_v1_r2/snapshot_2000.npz (immutable) - never the growing materialiser output.
Runs under /tmp/rlr_lgbm (lightgbm 4.7.0 + numpy); imports nothing from the repo venv.
Ranking loss uses groups WITH a positive; validation NDCG@20 is tracked on those same groups.
Test groups are NOT evaluated here (reserved for the frozen final evaluation).
"""
from __future__ import annotations
import json, resource, sys, time
from pathlib import Path
import numpy as np
import lightgbm as lgb

REPO = Path(__file__).resolve().parents[1]
NPZ = REPO / "runs" / "lr_v1_r2" / "snapshot_2000.npz"
OUT = REPO / "runs" / "lr_v1_r2"
PARAMS = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20],
          "label_gain": [0, 1], "num_leaves": 15, "learning_rate": 0.05,
          "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
          "bagging_freq": 1, "lambda_l2": 1.0, "lambdarank_truncation_level": 30,
          "num_threads": 2, "force_row_wise": True, "seed": 20261019, "verbose": -1,
          "deterministic": True}
NUM_ROUNDS, EARLY_STOP = 400, 40


def group_slices(sizes):
    start = 0
    for s in sizes:
        yield start, start + s
        start += s


def rank_metrics(X, y, group, scores, ks=(5, 10, 20)):
    """Explicit HR/NDCG: a group without a positive contributes 0 to the denominator."""
    out = {f"HR@{k}": 0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks} | {"n": 0, "pos_groups": 0,
                                                                            "cond_ndcg@20_sum": 0.0}
    for (a, b), size in zip(group_slices(group), group):
        s = scores[a:b]; lab = y[a:b]
        if not lab.any():
            out["n"] += 1
            continue
        out["n"] += 1; out["pos_groups"] += 1
        order = np.argsort(-s, kind="stable")           # deterministic tie-handling
        rank = int(np.where(lab[order] == 1)[0][0]) + 1
        for k in ks:
            out[f"HR@{k}"] += int(rank <= k)
            out[f"NDCG@{k}"] += (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        out["cond_ndcg@20_sum"] += (1.0 / np.log2(rank + 1)) if rank <= 20 else 0.0
    n = max(out["n"], 1)
    for k in ks:
        out[f"HR@{k}"] = round(out[f"HR@{k}"] / n, 6)
        out[f"NDCG@{k}"] = round(out[f"NDCG@{k}"] / n, 6)
    out["conditional_NDCG@20"] = round(out["cond_ndcg@20_sum"] / max(out["pos_groups"], 1), 6)
    out.pop("cond_ndcg@20_sum")
    return out


def main() -> int:
    t0 = time.perf_counter()
    z = np.load(NPZ)
    X, y, sizes = z["X"], z["y"].astype(np.int32), z["group"]
    train_ids, val_ids, test_ids = z["train"], z["val"], z["test"]
    rrf = X[:, 13]                      # rrf_rank column (0 = not fused)
    assert len(sizes) == 2000 and int(sizes.sum()) == len(X) == len(y)
    # group-level index ranges, then select by split ---------------------------------
    ranges = list(group_slices(sizes))
    def subset(ids, positive_only: bool):
        idx, grp = [], []
        for q in ids:
            a, b = ranges[int(q)]
            if b == a:                       # a group with no legal candidate at all
                continue
            if positive_only and not y[a:b].any():
                continue
            idx.append(np.arange(a, b)); grp.append(b - a)
        return np.concatenate(idx), np.asarray(grp)
    tr_idx, tr_grp = subset(train_ids, True)
    va_idx, va_grp = subset(val_ids, True)
    dtr = lgb.Dataset(X[tr_idx], label=y[tr_idx], group=tr_grp.astype(int), free_raw_data=True)
    dva = lgb.Dataset(X[va_idx], label=y[va_idx], group=va_grp.astype(int), reference=dtr,
                      free_raw_data=True)
    evals, curve = {}, []
    model = lgb.train(PARAMS, dtr, num_boost_round=NUM_ROUNDS, valid_sets=[dva],
                      valid_names=["val"],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                 lgb.record_evaluation(evals),
                                 lgb.log_evaluation(period=25)])
    best_iter = int(model.best_iteration or model.current_iteration())
    model.save_model(str(OUT / "model.txt"), num_iteration=best_iter)
    # ---- validation metrics: ranker vs RRF on the SAME candidate pool ------------- #
    va_all_idx, va_all_grp = subset(val_ids, False)
    Xv, yv = X[va_all_idx], y[va_all_idx]
    pred = model.predict(Xv, num_iteration=best_iter)
    ranker_m = rank_metrics(Xv, yv, va_all_grp, pred)
    rrf_score = np.where(rrf[va_all_idx] > 0, 1.0 / (60.0 + rrf[va_all_idx]), 0.0)
    rrf_m = rank_metrics(Xv, yv, va_all_grp, rrf_score)
    # per-group deltas at NDCG@20 (explicit, same groups)
    deltas = []
    rrf_col = np.where(rrf[va_all_idx] > 0, 1.0 / (60.0 + rrf[va_all_idx]), 0.0)
    for (a, b), size in zip(group_slices(va_all_grp), va_all_grp):
        lab = yv[a:b]
        if size == 0 or not lab.any():
            deltas.append(0.0)
            continue
        def ndcg_at20(score: np.ndarray) -> float:
            order = np.argsort(-score, kind="stable")
            rank = int(np.where(lab[order] == 1)[0][0]) + 1
            return (1.0 / np.log2(rank + 1)) if rank <= 20 else 0.0
        deltas.append(ndcg_at20(pred[a:b]) - ndcg_at20(rrf_col[a:b]))
    rng = np.random.default_rng(20261019); d = np.asarray(deltas)
    boots = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(5000)]
    ci = [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)]
    importance = sorted(zip(model.feature_name(), model.feature_importance("gain").tolist()),
                        key=lambda kv: -kv[1])[:10]
    manifest = {
        "kind": "R-LR-02 FUNCTIONAL model on the 2,000-group snapshot (test reserved)",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": PARAMS, "num_boost_round": NUM_ROUNDS, "early_stopping_rounds": EARLY_STOP,
        "best_iteration": best_iter,
        "data": {"snapshot": str(NPZ.name), "groups": int(len(sizes)), "rows": int(len(X)),
                 "train_groups_with_positive": int(len(tr_grp)),
                 "val_groups_with_positive": int(len(va_grp)),
                 "val_groups_all": int(len(va_all_grp)),
                 "rows_train": int(len(tr_idx)), "rows_val": int(len(va_idx)),
                 "sum_group_sizes_train": int(tr_grp.sum()), "sum_equals_rows_train": int(tr_grp.sum()) == len(tr_idx)},
        "val_curve_ndcg20": [round(float(v), 6) for v in evals["val"].get("ndcg@20", [])],
        "val_metrics_same_pool": {"ranker": ranker_m, "rrf": rrf_m},
        "paired_ndcg20_delta": {"mean": round(float(d.mean()), 6), "bootstrap_ci95": ci,
                                "resamples": 5000, "seed": 20261019,
                                "note": "per-group delta over val groups; no-positive groups contribute 0"},
        "feature_importance_gain_top10": importance,
        "prediction_sanity": {"min": float(pred.min()), "max": float(pred.max()),
                              "std": float(pred.std()),
                              "non_constant": bool(float(pred.std()) > 1e-9)},
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "lightgbm": lgb.__version__},
    }
    (OUT / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("best_iteration", "data", "val_metrics_same_pool",
                                              "paired_ndcg20_delta", "feature_importance_gain_top10",
                                              "prediction_sanity", "elapsed_seconds", "peak_rss_mb")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
