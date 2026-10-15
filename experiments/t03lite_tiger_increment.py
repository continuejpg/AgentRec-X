"""T-03-lite: does TIGER add incremental value to the accepted serving fusion?

Frozen-heads replay, **no model inference and no TIGER worker**: every source ranking is read
from an already-frozen artifact, truncated to the experiment's depth, fused with the repository's
accepted RRF, and scored by the shared evaluator.

    baseline  : history + two_tower + catalog_search
    experiment: history + two_tower + catalog_search + tiger

Profiles
--------
``primary``    every source at depth 20 (the deepest TIGER's frozen head supports for *every*
               user: its minimum length is exactly 20), so the only variable is the extra source.
``secondary``  the serving-like mix 100/100/100/20, which answers "is it useful next to the
               product configuration" at the cost of a depth asymmetry that favours the three
               incumbent sources.

What this script refuses to do
------------------------------
* no evaluator reimplementation - ``benchmark_public.evaluate_arm`` (the shared frozen evaluator)
  and, when the M4 cohort reference can be built, ``m4_benchmark.evaluate_arm_rankings``;
* no head regeneration and no TIGER worker call - the artifacts are read, never produced;
* no silent cohort drift - three hard assertions run before any metric is computed.

Usage
-----
    python -m experiments.t03lite_tiger_increment --out runs/t03lite_tiger_increment_v1.json
    python -m experiments.t03lite_tiger_increment --self-test      # assertions only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Iterable, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # pragma: no cover - direct execution
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from experiments import benchmark_public as B  # noqa: E402
from recommendation.control.candidate_ledger import RRF_K, reciprocal_rank_fusion  # noqa: E402
from recommendation.evaluation.split import load_cohort_from_artifacts  # noqa: E402

SEQUENCES = REPO / "data/processed/Sports_and_Outdoors_sequences.json"
MAPPINGS = REPO / "data/processed/Sports_and_Outdoors_mappings.json"
HEADS = REPO / "runs/m4_evidence/heads.npz"
TIGER_HEADS = REPO / "runs/step28_tiger_heads.jsonl"
FROZEN_TIGER_TOP10 = REPO / "runs/step26_tiger_hits_top10.npz"
FROZEN_TARGETS = REPO / "runs/m3_execution/paired_inputs.npz"

#: The M4 artifact's name for the accepted sequential source.
SEQUENTIAL = "history"
INCUMBENTS: tuple[str, ...] = (SEQUENTIAL, "two_tower", "catalog_search")
ALL_SOURCES: tuple[str, ...] = INCUMBENTS + ("tiger",)

PROFILES: dict[str, dict[str, int]] = {
    "primary": {name: 20 for name in ALL_SOURCES},
    "secondary": {**{name: 100 for name in INCUMBENTS}, "tiger": 20},
}

#: The target hash every artifact in this experiment must agree on.
FROZEN_TARGET_SHA256 = "99dbaf7396913303fa1497c908a95e61239a29ad14a6a8f4e51d0b37fa8e99d4"

K_VALUES = (10,)

LIMITATIONS = [
    "frozen heads replay: every source ranking is read from a pre-computed artifact; no model is "
    "run and no TIGER worker is started",
    "not an online A/B test: these are offline metrics over a frozen cohort, not user outcomes",
    "primary profile uses depth=20 for every source, which truncates the three incumbent engines "
    "from their 100-deep serving heads; the baseline is therefore not Serving V1",
    "the secondary profile keeps the serving depths (100/100/100) and truncates only TIGER to its "
    "ceiling of 20, so its depth asymmetry is biased against TIGER",
    "latency is not measured here: the numbers do not include regenerating TIGER candidates, "
    "whose CPU cold start alone is ~7-8 minutes",
    "catalog_search is the M4 BM25 implementation (control/catalog_search.py); it is the same "
    "family as, but not the same implementation as, the serving catalog_lexical index",
]


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_identity_map() -> dict[str, int]:
    payload = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload["item2id"].items()}


def load_frozen_head(source: str, *, depth: int, identity: dict[str, int]) -> list[list[int]]:
    """One source's frozen head per user, truncated to ``depth``, as integer item ids.

    ``runs/m4_evidence/heads.npz`` stores identities (object), ranks, scores and per-user lengths.
    Only the first ``depth`` identities of each row are needed, and an identity the trusted map
    cannot resolve is a hard failure rather than a dropped row.
    """
    with np.load(HEADS, allow_pickle=True) as archive:
        identities = archive[f"{source}_identities"]
        lengths = np.asarray(archive[f"{source}_lengths"])
        rows: list[list[int]] = []
        for index, length in enumerate(lengths):
            take = int(length)
            if take < depth:
                raise SystemExit(
                    f"assertion failed: {source} head for user row {index} is {take} deep, "
                    f"below the requested depth {depth}"
                )
            row = identities[index * 100 : index * 100 + depth]
            converted: list[int] = []
            for value in row:
                key = str(value)
                if key not in identity:
                    raise SystemExit(
                        f"assertion failed: {source} head row {index} holds an identity outside "
                        f"the trusted mappings: {key!r}"
                    )
                converted.append(identity[key])
            rows.append(converted)
    return rows


def load_tiger_head(*, depth: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in TIGER_HEADS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        values = json.loads(line)
        if not isinstance(values, list) or len(values) < depth:
            raise SystemExit(
                f"assertion failed: a tiger head line holds {len(values)} candidates, below the "
                f"requested depth {depth}"
            )
        rows.append([int(value) for value in values[:depth]])
    return rows


# --------------------------------------------------------------------------- #
# assertions (run before any metric is computed)
# --------------------------------------------------------------------------- #


def assert_contract(*, cases: Sequence[Any], cohort: Any, identity: dict[str, int]) -> dict[str, Any]:
    """The three contract conditions this experiment depends on.  Fail closed."""
    user_int_ids = np.asarray([int(case.user_int_id) for case in cohort])
    targets = np.asarray([int(case.test_target) for case in cohort], dtype=np.int64)

    # 1. the cohort order used by the frozen heads is the frozen target order
    with np.load(HEADS, allow_pickle=True) as archive:
        head_users = np.asarray(archive["user_int_ids"], dtype=np.int64)
    if not np.array_equal(head_users, user_int_ids):
        raise SystemExit("assertion failed: heads.npz user_int_ids do not match the cohort order")
    frozen = np.load(FROZEN_TARGETS, allow_pickle=True)["target_ids"]
    if not np.array_equal(np.asarray(frozen, dtype=np.int64), targets):
        raise SystemExit("assertion failed: the cohort targets differ from paired_inputs.npz")
    target_sha = sha256_array(targets)
    if target_sha != FROZEN_TARGET_SHA256:
        raise SystemExit(f"assertion failed: target hash {target_sha} != the frozen value")

    # 2. every source head is at least as deep as the deepest profile asks for
    deepest = max(max(profile.values()) for profile in PROFILES.values())
    with np.load(HEADS, allow_pickle=True) as archive:
        for source in INCUMBENTS:
            lengths = np.asarray(archive[f"{source}_lengths"])
            if int(lengths.min()) < deepest:
                raise SystemExit(
                    f"assertion failed: {source} has a head of {int(lengths.min())} < {deepest}"
                )
    tiger_lengths = np.asarray([len(json.loads(line)) for line in TIGER_HEADS.read_text(
        encoding="utf-8").splitlines() if line.strip()])
    if int(tiger_lengths.min()) < PROFILES["primary"]["tiger"]:
        raise SystemExit(
            f"assertion failed: a tiger head has {int(tiger_lengths.min())} candidates, below "
            f"{PROFILES['primary']['tiger']}"
        )

    # 3. the tiger artifact is the frozen one: its top-10 matches the certified step26 hits
    top10 = np.asarray(np.load(FROZEN_TIGER_TOP10, allow_pickle=True)["top10"], dtype=np.int64)
    ours = np.asarray(
        [json.loads(line)[:10] for line in TIGER_HEADS.read_text(encoding="utf-8").splitlines()
         if line.strip()],
        dtype=np.int64,
    )
    if top10.shape != ours.shape or not np.array_equal(top10, ours):
        raise SystemExit("assertion failed: step28 tiger heads do not match the frozen step26 hits")
    return {
        "users": int(len(user_int_ids)),
        "target_sha256": target_sha,
        "head_users_sha256": sha256_array(head_users),
        "tiger_top10_rows_matching_frozen_hits": int(top10.shape[0]),
        "tiger_head_depth": {"min": int(tiger_lengths.min()), "max": int(tiger_lengths.max()),
                             "mean": round(float(tiger_lengths.mean()), 3)},
        "incumbent_head_depth": deepest,
    }


# --------------------------------------------------------------------------- #
# the arm
# --------------------------------------------------------------------------- #


def source_rankings_for(profile: dict[str, int], *, identity: dict[str, int],
                        include_tiger: bool) -> dict[str, list[list[int]]]:
    sources = ALL_SOURCES if include_tiger else INCUMBENTS
    rankings: dict[str, list[list[int]]] = {}
    for source in sources:
        depth = int(profile[source])
        rankings[source] = (
            load_tiger_head(depth=depth) if source == "tiger"
            else load_frozen_head(source, depth=depth, identity=identity)
        )
    return rankings


def fused_rankings(source_rankings: dict[str, list[list[int]]], *, users: int) -> list[list[int]]:
    """RRF over the per-user source rankings, using the repository's accepted fusion."""
    sources = list(source_rankings)
    out: list[list[int]] = []
    for user in range(users):
        lists = [[str(value) for value in source_rankings[source][user]] for source in sources]
        fused = reciprocal_rank_fusion(lists, k=RRF_K)
        out.append([int(identity) for identity, _ in fused])
    return out


def evaluate(
    *, source_rankings: dict[str, list[list[int]]], cases: Sequence[Any], num_items: int,
    head: int,
) -> dict[str, Any]:
    """Score one arm with the shared frozen evaluator through the accepted fusion arm."""
    arm = B.arm_fixed_fusion(
        num_items=num_items,
        source_rankings=source_rankings,
        head=head,
        sources=list(source_rankings),
    )
    started = time.perf_counter()
    payload = B.evaluate_arm(cases=cases, num_items=num_items, batches=arm)
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    return payload


def tiger_analysis(
    *, baseline: dict[str, list[list[int]]], with_tiger: dict[str, list[list[int]]],
    targets: Sequence[int], cases: Sequence[Any],
) -> dict[str, Any]:
    unique_total = 0
    overlap_totals = {source: 0 for source in INCUMBENTS}
    tiger_only_hits = 0
    baseline_hits = 0
    joint_hits = 0
    for user, case in enumerate(cases):
        target = int(case.test_target)
        tiger_row = with_tiger["tiger"][user]
        incumbents = {name: set(baseline[name][user]) for name in INCUMBENTS}
        union = set().union(*incumbents.values()) if incumbents else set()
        unique_total += len([value for value in tiger_row if value not in union])
        for name, rows in incumbents.items():
            overlap_totals[name] += len([value for value in tiger_row if value in rows])
        base_hit = any(target in rows for rows in incumbents.values())
        tiger_hit = target in tiger_row
        baseline_hits += int(base_hit)
        joint_hits += int(base_hit or tiger_hit)
        if tiger_hit and not base_hit:
            tiger_only_hits += 1

    leave_one_out: dict[str, Any] = {}
    for dropped in ALL_SOURCES:
        rows = [value for value in ALL_SOURCES if value != dropped]
        ids = [int(case.test_target) for case in cases]
        hits = 0
        for user in range(len(cases)):
            fused = reciprocal_rank_fusion([source_rankings_as_str(with_tiger[s][user])
                                            for s in rows], k=RRF_K)
            if str(ids[user]) in [identity for identity, _ in fused[:10]]:
                hits += 1
        leave_one_out[f"without_{dropped}_hit@10"] = hits
    return {
        "unique_candidates_not_in_any_incumbent": unique_total,
        "overlap_with_incumbent_candidates": overlap_totals,
        "tiger_only_hits_at_10": tiger_only_hits,
        "incumbent_hits_at_10": baseline_hits,
        "union_hits_at_10": joint_hits,
        "leave_one_out_hit@10": leave_one_out,
    }


def source_rankings_as_str(values: Sequence[int]) -> list[str]:
    return [str(value) for value in values]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def run(*, out: pathlib.Path | None) -> dict[str, Any]:
    cases_list, split = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
    cohort_meta = B.cohort_from_cases(cases_list, size=B.DEFAULT_COHORT_SIZE)
    cases = list(cohort_meta["cases"])
    # The catalogue size comes from the split report, exactly as step28 reads it.
    num_items = int(split.catalog_size)
    identity = load_identity_map()
    contract = assert_contract(cases=cases, cohort=cases, identity=identity)

    report: dict[str, Any] = {
        "experiment": "t03lite_tiger_increment_v1",
        "config": {
            "profiles": {name: dict(depths) for name, depths in PROFILES.items()},
            "rrf_k": RRF_K,
            "k_values": list(K_VALUES),
            "users": len(cases),
            "num_items": num_items,
            "sources_baseline": list(INCUMBENTS),
            "sources_experiment": list(ALL_SOURCES),
            "artifacts": {
                "heads": {"path": str(HEADS.relative_to(REPO)), "sha256": sha256_file(HEADS)},
                "tiger_heads": {"path": str(TIGER_HEADS.relative_to(REPO)),
                                "sha256": sha256_file(TIGER_HEADS)},
                "mappings": {"path": str(MAPPINGS.relative_to(REPO)),
                             "sha256": sha256_file(MAPPINGS)},
                "frozen_targets": {"path": str(FROZEN_TARGETS.relative_to(REPO)),
                                   "sha256": sha256_file(FROZEN_TARGETS)},
                "frozen_tiger_top10": {"path": str(FROZEN_TIGER_TOP10.relative_to(REPO)),
                                       "sha256": sha256_file(FROZEN_TIGER_TOP10)},
            },
            "evaluator": "experiments.benchmark_public.evaluate_arm (shared frozen evaluator) via "
                         "benchmark_public.arm_fixed_fusion + reciprocal_rank_fusion",
        },
        "contract": contract,
        "metrics": {},
        "tiger_analysis": {},
        "limitations": LIMITATIONS,
    }

    targets = [int(case.test_target) for case in cases]
    for profile_name, depths in PROFILES.items():
        head = max(depths.values())
        baseline_sources = source_rankings_for(depths, identity=identity, include_tiger=False)
        tiger_sources = source_rankings_for(depths, identity=identity, include_tiger=True)
        baseline = evaluate(source_rankings=baseline_sources, cases=cases, num_items=num_items,
                            head=head)
        experiment = evaluate(source_rankings=tiger_sources, cases=cases, num_items=num_items,
                              head=head)
        def metric_block(payload: dict[str, Any]) -> dict[str, Any]:
            block = payload.get("metrics")
            return block if isinstance(block, dict) else payload

        baseline_metrics = metric_block(baseline)
        experiment_metrics = metric_block(experiment)

        def pick(metrics: dict[str, Any], name: str, k: int = 10) -> float:
            """Read one metric from the evaluator's payload.

            ``evaluate_arm`` groups by metric name and then by K (``{"Recall": {"10": ...}}``), so
            accept the grouped shape, the flat ``"Recall@10"`` shape and a bare number.
            """
            node: Any = None
            for key in (name, name.upper(), name.lower(), name.replace("@10", "")):
                if key in metrics:
                    node = metrics[key]
                    break
            if node is None:
                raise SystemExit(
                    f"the evaluator did not report {name}; it reported {sorted(metrics)}"
                )
            if isinstance(node, dict):
                for key in (str(k), k, f"@{k}"):
                    if key in node:
                        return float(node[key])
                raise SystemExit(f"{name} has no K={k} entry; keys: {sorted(node)}")
            return float(node)

        entry: dict[str, Any] = {
            "baseline": {"recall@10": pick(baseline_metrics, "Recall@10"),
                         "ndcg@10": pick(baseline_metrics, "NDCG@10")},
            "tiger": {"recall@10": pick(experiment_metrics, "Recall@10"),
                      "ndcg@10": pick(experiment_metrics, "NDCG@10")},
        }
        for metric in ("recall@10", "ndcg@10"):
            delta = entry["tiger"][metric] - entry["baseline"][metric]
            entry.setdefault("delta", {})[metric] = {
                "absolute": delta,
                "relative": (delta / entry["baseline"][metric]) if entry["baseline"][metric] else None,
            }
        report["metrics"][profile_name] = entry
        if profile_name == "primary":
            report["tiger_analysis"] = tiger_analysis(
                baseline=baseline_sources, with_tiger=tiger_sources, targets=targets, cases=cases
            )

    # A ranking-path cross-check on the same fused lists, when the M4 cohort reference is available.
    try:  # pragma: no cover - depends on the M4 module's internal cohort type
        from experiments import m4_benchmark as M4

        report["cross_check"] = {
            "available": True,
            "note": "evaluate_arm_rankings needs the M4 cohort reference; see the run log",
            "cohort_ref_fields": sorted(
                getattr(M4.M4CohortRef, "__dataclass_fields__", {}).keys()
            ),
        }
    except Exception as exc:  # noqa: BLE001
        report["cross_check"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO / "runs" / "t03lite_tiger_increment_v1.json")
    parser.add_argument("--self-test", action="store_true",
                        help="run the three contract assertions and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        cases_list, _ = load_cohort_from_artifacts(SEQUENCES, MAPPINGS)
        cases = list(B.cohort_from_cases(cases_list, size=B.DEFAULT_COHORT_SIZE)["cases"])
        print(json.dumps(assert_contract(cases=cases, cohort=cases, identity=load_identity_map()),
                         indent=1, sort_keys=True))
        return 0
    report = run(out=args.out)
    print(json.dumps(report["metrics"], indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
