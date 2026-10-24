"""Phase 2C: independent evaluation of the frozen Learned Ranker against full-precision RRF.

Question
--------
Does the frozen LightGBM LambdaRank V1 (22 trees) order the *same* candidate pool better than
the accepted full-precision RRF baseline?  The two earlier baselines disagree because the
training script sorted the RRF candidates by ``1/(60 + rrf_score)`` -a strictly decreasing
transform of the RRF score -which **inverts** the order (3/766 HR@20 instead of 30/766).
This driver fixes the baseline explicitly and evaluates on a cohort that never touched the
model's training, selection, or the Phase 2B samples.

Protocol (fixed, no tuning)
---------------------------
* task: ``agentrecx.eval_protocol.v1`` / temporal_leave_two_out;
  ``history = case.test_history = items[:-1]``, ``target = items[-1]``;
* candidate pool: union of SASRec / Two-Tower / catalogue-lexical top-100 heads, history
  removed, known identities only -exactly ``experiments/r_lr01_materialize.py``;
* baseline: full-precision ``sum 1/(60 + source_rank)`` over the **full fused universe**,
  pool order ``(-rrf_score, item_id)``;
* treatment: frozen model + schema, the production ``build_feature_rows`` /
  ``order_by_ranker`` (ranker score desc, rrf 6dp desc, item_id asc);
* metrics: target rank inside the pool; HR/NDCG at 5/10/20; primary NDCG@20;
* cohort: 2,000 valid cases from valid case #9001 (artifact index 9000+), outside P3
  (#1-1000), LR training (#1001-6000), 2B.5 (#6500-6709) and 2B.6 (#6710-7791); users in the
  frozen 20k M3/M4/Phase5/B1 cohort are excluded.

Gates (pre-registered in docs/LR_INDEPENDENT_EVAL_PLAN.md, not adjustable after seeing results)
---------------------------------------------------------------------------------------------
G1  paired ΔNDCG@20 95% CI lower bound > 0
G2  ΔHR@5 >= 0, or its 95% CI lower bound > -0.002
G3  model/schema hashes, 15 features, pool identity, no prediction fallback
G4  cohort independence (position + frozen-cohort exclusion + one case per user)

    .venv-ranker/bin/python -m experiments.r_lr05_independent_eval --cases 2000 --skip 9000 \
        --out runs/lr_v2_independent_eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from recommendation import config as project_config
from recommendation.evaluation.metrics import hit_at_k, ndcg_at_k

REPO = project_config.REPO_ROOT
SEQUENCES = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_sequences.json"
MAPPINGS = project_config.DATA_DIR / "processed" / "Sports_and_Outdoors_mappings.json"
SASREC = REPO / "runs" / "sasrec_canonical_2026" / "best.pt"
TWO_TOWER = REPO / "runs" / "twotower_public_2026_full"
MODEL = REPO / "runs" / "lr_v1_formal" / "model.txt"
SCHEMA = REPO / "runs" / "lr_v1_formal" / "feature_schema.json"
METRICS = REPO / "runs" / "lr_v1_formal" / "metrics.json"
M4_HEADS = REPO / "runs" / "m4_evidence" / "heads.npz"
DEFAULT_OUT = REPO / "runs" / "lr_v2_independent_eval"

#: Frozen artifacts (docs/LEARNED_RANKER_DEPLOYMENT.md §1).
EXPECTED_MODEL_SHA = "c1b37d4a2afe9187000c73bc7cb7bb22cc811dda3a2fe9bfe2118fa513db6eb9"
EXPECTED_SCHEMA_SHA = "ad1d0be168dc9f183c45683c26e24a0086e882632570e2feffa8f9971598a422"
EXPECTED_FEATURES = (
    "seq_mask", "seq_rank", "seq_inv_rank", "seq_score",
    "tt_mask", "tt_rank", "tt_inv_rank", "tt_score",
    "lex_mask", "lex_rank", "lex_inv_rank", "lex_score",
    "rrf_score", "rrf_rank", "source_count",
)

HEAD_DEPTH = 100
K_VALUES: tuple[int, ...] = (5, 10, 20)
DEFAULT_SKIP = 9000
DEFAULT_CASES = 2000
BOOTSTRAP_DRAWS = 10000
SEED = 20261019
RRF_K = 60

#: Valid-case windows used by earlier experiments (1-based, contiguous by position).
COHORT_WINDOWS = {
    "p3": (1, 1000),
    "lr_training": (1001, 6000),
    "p2b5": (6500, 6709),
    "p2b6": (6710, 7791),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001 - a missing git must not fail the evaluation
        return None
    return completed.stdout.strip() or None


def _frozen_cohort_users() -> set[int]:
    import numpy as np

    data = np.load(M4_HEADS, allow_pickle=True)
    return {int(value) for value in data["user_int_ids"].tolist()}


def _rank_of(order: Sequence[str], target: str) -> int | None:
    try:
        return list(order).index(target) + 1
    except ValueError:
        return None


def _metrics(rank: int | None) -> dict[str, float]:
    if rank is None:
        return {f"hr@{k}": 0.0 for k in K_VALUES} | {f"ndcg@{k}": 0.0 for k in K_VALUES}
    return {
        **{f"hr@{k}": hit_at_k(rank, k) for k in K_VALUES},
        **{f"ndcg@{k}": ndcg_at_k(rank, k) for k in K_VALUES},
    }


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    return ordered[int(fraction * (len(ordered) - 1))]


def _paired_bootstrap(
    left: Sequence[float], right: Sequence[float], *, seed: int, draws: int
) -> dict[str, Any]:
    diffs = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(draws):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return {
        "delta": round(sum(diffs) / n, 6) if n else 0.0,
        "ci95": [round(_percentile(means, 0.025), 6), round(_percentile(means, 0.975), 6)],
        "n": n,
        "seed": seed,
        "draws": draws,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite an existing result directory: {args.out}")

    import numpy as np

    from recommendation.api.app import DEFAULT_CATALOG
    from recommendation.catalog import MetadataIndex
    from recommendation.control.candidate_ledger import reciprocal_rank_fusion
    from recommendation.evaluation.split import split_sequence
    from recommendation.inference.learned_ranker import (
        build_feature_rows,
        order_by_ranker,
    )
    from recommendation.inference.lexical import CatalogueLexicalIndex, query_text_for
    from recommendation.inference.sasrec import InferenceConfig as SASRecConfig
    from recommendation.inference.sasrec import SASRecInferenceEngine
    from recommendation.inference.twotower import TwoTowerInferenceConfig, TwoTowerInferenceEngine
    from experiments.e04b2_gate_a import stream_sequences

    import lightgbm as lgb

    # -- frozen artifact checks (G3) --------------------------------------- #
    model_sha, schema_sha = _sha256(MODEL), _sha256(SCHEMA)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if model_sha != EXPECTED_MODEL_SHA or schema_sha != EXPECTED_SCHEMA_SHA:
        raise SystemExit("frozen model/schema hash mismatch; refusing to evaluate")
    if tuple(schema.get("features", ())) != EXPECTED_FEATURES or int(schema.get("count", 0)) != 15:
        raise SystemExit("feature schema disagrees with the frozen 15-feature contract")
    booster = lgb.Booster(model_file=str(MODEL))
    if int(booster.num_feature()) != 15:
        raise SystemExit("model expects a different feature count")
    best_iteration = int(json.loads(METRICS.read_text(encoding="utf-8"))["best_iteration"])

    # -- engines and catalogue --------------------------------------------- #
    began = time.perf_counter()
    mappings = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    num_items = int(mappings["num_items"])
    id_to_asin = {int(value): str(key) for key, value in mappings["item2id"].items()}
    asin_to_id = {asin: item_id for item_id, asin in id_to_asin.items()}
    sequential = SASRecInferenceEngine(
        SASRecConfig(checkpoint_path=SASREC, mappings_path=MAPPINGS, device="cpu")
    )
    two_tower = TwoTowerInferenceEngine(
        TwoTowerInferenceConfig(
            checkpoint_path=TWO_TOWER / "checkpoint.pt",
            categories_path=TWO_TOWER / "metadata_categories.json",
            mappings_path=MAPPINGS,
            device="cpu",
        )
    )
    records = MetadataIndex.load(DEFAULT_CATALOG).records
    lexical = CatalogueLexicalIndex.build(records)
    catalogue: dict[str, Any] = {}
    for key, value in records.items():
        asin = getattr(value, "parent_asin", None) or key
        catalogue[str(asin)] = value
    load_seconds = round(time.perf_counter() - began, 2)

    # -- cohort ------------------------------------------------------------- #
    cohort_users = _frozen_cohort_users()
    cases: list[tuple[int, Any]] = []
    scanned = 0
    excluded_cohort = 0
    valid_index = 0
    first_selected: int | None = None
    for element in stream_sequences(SEQUENCES):
        items = [int(value) for value in element.get("item_ids", ())]
        case = split_sequence(
            str(element.get("user_id", "")), items, num_items,
            int(element.get("user_int_id", -1)),
        )
        if case is None:
            continue
        valid_index += 1
        if valid_index <= args.skip:
            continue
        scanned += 1
        if int(case.user_int_id) in cohort_users:
            excluded_cohort += 1
            continue
        if first_selected is None:
            first_selected = valid_index
        cases.append((valid_index, case))
        if len(cases) >= args.cases:
            break
    if not cases:
        raise SystemExit("no usable cases")

    engine_meta = {
        "model_sha256": model_sha,
        "schema_sha256": schema_sha,
        "best_iteration": best_iteration,
        "features": list(EXPECTED_FEATURES),
        "head_depth": HEAD_DEPTH,
    }
    config = {
        "task": "next_item_prediction",
        "protocol": "agentrecx.eval_protocol.v1 / temporal_leave_two_out",
        "history_semantics": "case.test_history (train_history + validation_target)",
        "cohort": {
            "skip_valid_cases": args.skip,
            "requested": args.cases,
            "selected": len(cases),
            "scanned_after_skip": scanned,
            "excluded_frozen_20k_cohort": excluded_cohort,
            "first_valid_case": first_selected,
            "last_valid_case": valid_index,
            "windows": {name: list(window) for name, window in COHORT_WINDOWS.items()},
            "disjoint_by_position": first_selected is not None
            and first_selected > max(end for _, end in COHORT_WINDOWS.values()),
            "distinct_users": len({case.user_int_id for _, case in cases}),
            "frozen_cohort_size": len(cohort_users),
        },
        "engines": engine_meta,
        "baseline": {
            "definition": "full-precision RRF: sum 1/(60 + source_rank) over the full fused "
                          "universe; pool order (-rrf_score, item_id)",
            "why": "the training script's baseline used 1/(60 + rrf_score), which inverts the "
                   "RRF order; see docs/2C baseline root-cause section",
        },
        "treatment": {
            "model": "runs/lr_v1_formal/model.txt",
            "schema": "runs/lr_v1_formal/feature_schema.json",
            "ordering": "ranker score desc, rrf score (6dp) desc, item_id asc",
            "retrained": False,
        },
        "metrics": {"k_values": list(K_VALUES), "primary": "NDCG@20", "secondary": "HR@5"},
        "gates": {
            "G1": "paired dNDCG@20 CI lower > 0",
            "G2": "dHR@5 >= 0 or its CI lower > -0.002",
            "G3": "model/schema hashes, 15 features, pool identity, zero fallback",
            "G4": "cohort disjoint by position, frozen cohort excluded, one case per user",
        },
        "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "load_seconds": load_seconds,
        "inputs": {
            "sequences_sha256": _sha256(SEQUENCES),
            "mappings_sha256": _sha256(MAPPINGS),
            "sasrec_sha256": _sha256(SASREC),
            "two_tower_sha256": _sha256(TWO_TOWER / "checkpoint.pt"),
        },
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "driver_path": str(Path(__file__).resolve().relative_to(REPO)),
        "driver_sha256": _sha256(Path(__file__)),
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # -- one pass: recall once, then compare two orderings ------------------ #
    samples: list[dict[str, Any]] = []
    fallback = 0
    pool_identity_failures = 0
    feature_failures = 0
    empty_pools = 0
    source_failures = 0
    began = time.perf_counter()
    for ordinal, (case_index, case) in enumerate(cases, start=1):
        started = time.perf_counter()
        history = list(case.test_history)
        hist_asins = [id_to_asin[item] for item in history if item in id_to_asin]
        hist_set = set(history)
        per_source: dict[str, dict[str, tuple[int, float]]] = {}
        try:
            sequential_result = sequential.recommend(hist_asins, k=HEAD_DEPTH)
            per_source["seq"] = {
                item.parent_asin: (int(item.rank), float(item.score))
                for item in sequential_result.recommendations
            }
        except Exception:  # noqa: BLE001 - one source failing is recorded, not fatal
            per_source["seq"] = {}
            source_failures += 1
        try:
            two_tower_result = two_tower.recommend(history, k=HEAD_DEPTH)
            per_source["tt"] = {
                item.parent_asin: (int(item.rank), float(item.score))
                for item in two_tower_result.recommendations
            }
        except Exception:  # noqa: BLE001
            per_source["tt"] = {}
            source_failures += 1
        try:
            usable_records = [
                catalogue[asin] for asin in hist_asins[-5:]
                if asin in catalogue and getattr(catalogue[asin], "title", None)
            ]
            query = query_text_for(usable_records) if usable_records else ""
            per_source["lex"] = {
                asin: (index + 1, float(score))
                for index, (asin, score) in enumerate(
                    lexical.scored(query, limit=HEAD_DEPTH)
                )
            }
        except Exception:  # noqa: BLE001
            per_source["lex"] = {}
            source_failures += 1

        rankings = [
            [asin for asin, _ in sorted(per_source[name].items(), key=lambda kv: kv[1][0])]
            for name in ("seq", "tt", "lex")
        ]
        fused = dict(reciprocal_rank_fusion([ranking for ranking in rankings if ranking], k=RRF_K))
        fused_order = [asin for asin, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
        pool: list[str] = []
        for name in ("seq", "tt", "lex"):
            for asin in sorted(per_source[name], key=lambda key: per_source[name][key][0]):
                if asin in asin_to_id and asin_to_id[asin] not in hist_set and asin not in pool:
                    pool.append(asin)
        target_asin = id_to_asin.get(int(case.test_target), "")

        if not pool:
            empty_pools += 1
            samples.append({
                "valid_index": case_index, "ordinal": ordinal,
                "user_int_id": int(case.user_int_id), "target_asin": target_asin,
                "pool_size": 0, "target_in_pool": False,
                "rrf_rank": None, "lr_rank": None,
                "rrf": _metrics(None), "lr": _metrics(None), "ms": 0.0,
            })
            continue

        item_ids = [asin_to_id[asin] for asin in pool]
        rrf_full = {asin: float(fused.get(asin, 0.0)) for asin in pool}
        baseline_order = sorted(pool, key=lambda asin: (-rrf_full[asin], asin_to_id[asin]))
        rows = build_feature_rows(
            item_ids, pool,
            {name: {asin: pair[0] for asin, pair in per_source[name].items()}
             for name in ("seq", "tt", "lex")},
            {name: {asin: pair[1] for asin, pair in per_source[name].items()}
             for name in ("seq", "tt", "lex")},
            fused,
            fused_order,
        )
        if len(rows) != len(pool) or any(len(row) != 15 for row in rows):
            feature_failures += 1
        try:
            predictions = booster.predict(np.asarray(rows, dtype=np.float32),
                                          num_iteration=best_iteration)
            lr_indices = order_by_ranker(
                list(predictions),
                [round(rrf_full[asin], 6) for asin in pool],
                item_ids,
            )
            lr_order = [pool[index] for index in lr_indices]
        except Exception:  # noqa: BLE001 - counted; G3 will fail if it happens
            fallback += 1
            lr_order = list(baseline_order)
        if set(lr_order) != set(baseline_order):
            pool_identity_failures += 1

        rrf_rank = _rank_of(baseline_order, target_asin)
        lr_rank = _rank_of(lr_order, target_asin)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        samples.append({
            "valid_index": case_index, "ordinal": ordinal,
            "user_int_id": int(case.user_int_id), "target_asin": target_asin,
            "pool_size": len(pool), "target_in_pool": rrf_rank is not None,
            "rrf_rank": rrf_rank, "lr_rank": lr_rank,
            "rrf": _metrics(rrf_rank), "lr": _metrics(lr_rank),
            "ms": round(elapsed_ms, 2),
        })
        if ordinal % 100 == 0:
            print(json.dumps({"cases_done": ordinal, "elapsed_s": round(time.perf_counter() - began, 1)}),
                  flush=True)

    # -- aggregates and gates ---------------------------------------------- #
    def mean_metric(key: str, arm: str) -> float:
        return _mean([float(row[arm][key]) for row in samples]) or 0.0

    aggregates = {
        "cases": len(samples),
        "candidate_recall": round(sum(1 for row in samples if row["target_in_pool"]) / len(samples), 6),
        "mean_pool": round(sum(row["pool_size"] for row in samples) / len(samples), 2),
        "rrf": {f"{metric}@{k}": mean_metric(f"{metric}@{k}", "rrf")
                for metric in ("hr", "ndcg") for k in K_VALUES},
        "lr": {f"{metric}@{k}": mean_metric(f"{metric}@{k}", "lr")
               for metric in ("hr", "ndcg") for k in K_VALUES},
        "mean_rrf_rank_when_pooled": _mean(
            [float(row["rrf_rank"]) for row in samples if row["rrf_rank"]]
        ),
        "mean_lr_rank_when_pooled": _mean(
            [float(row["lr_rank"]) for row in samples if row["lr_rank"]]
        ),
        "p50_ms": round(sorted(row["ms"] for row in samples)[len(samples) // 2], 2),
        "p95_ms": round(sorted(row["ms"] for row in samples)[int(0.95 * (len(samples) - 1))], 2),
        "empty_pools": empty_pools,
    }
    paired = {
        "ndcg@20": _paired_bootstrap(
            [float(row["lr"]["ndcg@20"]) for row in samples],
            [float(row["rrf"]["ndcg@20"]) for row in samples],
            seed=SEED, draws=BOOTSTRAP_DRAWS,
        ),
        "hr@5": _paired_bootstrap(
            [float(row["lr"]["hr@5"]) for row in samples],
            [float(row["rrf"]["hr@5"]) for row in samples],
            seed=SEED, draws=BOOTSTRAP_DRAWS,
        ),
    }
    gates = {
        "G1": {
            "satisfied": paired["ndcg@20"]["ci95"][0] > 0,
            "detail": paired["ndcg@20"],
        },
        "G2": {
            "satisfied": paired["hr@5"]["delta"] >= 0 or paired["hr@5"]["ci95"][0] > -0.002,
            "detail": paired["hr@5"],
        },
        "G3": {
            "satisfied": (
                model_sha == EXPECTED_MODEL_SHA
                and schema_sha == EXPECTED_SCHEMA_SHA
                and feature_failures == 0
                and fallback == 0
                and pool_identity_failures == 0
                and int(booster.num_feature()) == 15
            ),
            "model_sha256": model_sha,
            "schema_sha256": schema_sha,
            "feature_failures": feature_failures,
            "fallback_count": fallback,
            "pool_identity_failures": pool_identity_failures,
        },
        "G4": {
            "satisfied": (
                len(cases) == args.cases
                and first_selected is not None
                and first_selected > max(end for _, end in COHORT_WINDOWS.values())
                and len({case.user_int_id for _, case in cases}) == len(cases)
            ),
            "first_valid_case": first_selected,
            "min_independent_case": max(end for _, end in COHORT_WINDOWS.values()) + 1,
            "distinct_users": len({case.user_int_id for _, case in cases}),
            "excluded_frozen_20k_cohort": excluded_cohort,
        },
    }
    gates["all_satisfied"] = all(gate["satisfied"] for gate in gates.values())

    summary = {
        "task": "next_item_prediction",
        "primary": "NDCG@20",
        "secondary": "HR@5",
        "cohort": config["cohort"],
        "engines": engine_meta,
        "baseline": config["baseline"],
        "treatment": config["treatment"],
        "aggregates": aggregates,
        "paired": paired,
        "gates": gates,
    }

    (args.out / "cases.json").write_text(
        json.dumps([
            {
                "valid_index": row["valid_index"], "user_int_id": row["user_int_id"],
                "target_asin": row["target_asin"], "pool_size": row["pool_size"],
                "target_in_pool": row["target_in_pool"],
            }
            for row in samples
        ], indent=2),
        encoding="utf-8",
    )
    (args.out / "samples.jsonl").write_text(
        "\n".join(json.dumps(row) for row in samples), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {}
    for name in ("config.json", "cases.json", "samples.jsonl", "summary.json"):
        payload = (args.out / name).read_bytes()
        manifest[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (args.out / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
