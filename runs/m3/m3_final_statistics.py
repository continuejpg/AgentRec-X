"""M3 final offline statistics: the preregistered primary endpoint, NDCG@10.

Offline only. Reads the already-landed M3 execution evidence and the frozen fixed-fusion
evidence; calls no provider, re-runs no Agent user, and changes no policy, prompt, tool budget,
routing, comparator, cohort, K value or evaluator.

NDCG uses the repository's existing semantics — recommendation.evaluation.metrics.ndcg_at_k —
rather than a parallel metric implementation. The paired bootstrap reuses the Step-2.8b /
M3 preregistered protocol verbatim: 10 000 resamples, seed 20260201, percentile CI.

Fail-closed: every consistency figure the frozen evidence already published is recomputed here
from the same paired arrays, and any disagreement stops the run rather than being adjusted.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from recommendation.evaluation.metrics import ndcg_at_k  # noqa: E402

EXEC = REPO / "runs/m3_execution"
K_VALUES = (5, 10, 20)
COHORT = 20_000
RESAMPLES = 10_000
SEED = 20260201

#: Values already published by the frozen execution. Recomputed, never assumed.
EXPECTED = {
    "comparator_hits": {5: 172, 10: 287, 20: 465},
    "agent_hits": {5: 133, 10: 133, 20: 133},
    "recall_delta": {5: -0.00195, 10: -0.00770, 20: -0.01660},
    "mcnemar_p": {5: 0.0016313112527077745, 10: 7.516255230127073e-29,
                  20: 1.5784654489008472e-90},
    "gained": {5: 54, 10: 28, 20: 7},
    "lost": {5: 93, 10: 182, 20: 339},
}

ARTIFACTS = (
    "agent_ranks.npz", "paired_inputs.npz", "agent_behavior.jsonl", "agent_rankings.jsonl",
    "m3_statistics.json", "run_summary.json", "run_provenance.json",
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mcnemar(b: int, c: int) -> float:
    import math
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(diff: np.ndarray, seed: int = SEED, resamples: int = RESAMPLES):
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        take = min(500, resamples - done)
        idx = rng.integers(0, n, size=(take, n))
        means[done:done + take] = diff[idx].mean(axis=1)
        done += take
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ndcg_vector(ranks: np.ndarray, k: int) -> np.ndarray:
    return np.array([ndcg_at_k(int(r), k) if int(r) >= 1 else 0.0 for r in ranks], dtype=np.float64)


def main() -> int:
    print("=" * 100)
    print("M3 FINAL OFFLINE STATISTICS — preregistered primary endpoint NDCG@10")
    print("=" * 100)

    # ---- 1. freeze the landed artifacts by hash (never overwritten) ---------------- #
    print("\n--- 1. artifact SHA256 (originals untouched) ---")
    hashes = {}
    for name in ARTIFACTS:
        path = EXEC / name
        if not path.is_file():
            print(f"  STOP: missing execution artifact {name}")
            return 2
        hashes[name] = sha256(path)
        print(f"  {name:26s} {path.stat().st_size:>10,} B  {hashes[name]}")
    (EXEC / "ARTIFACT_SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    # ---- 2. paired evidence ------------------------------------------------------- #
    paired = np.load(EXEC / "paired_inputs.npz")
    agent_rank = np.asarray(paired["agent_rank"], dtype=np.int64)
    comp_rank = np.asarray(paired["comparator_rank"], dtype=np.int64)
    if agent_rank.size != COHORT or comp_rank.size != COHORT:
        print(f"  STOP: cohort size {agent_rank.size}/{comp_rank.size} != {COHORT}")
        return 2
    print(f"\n--- 2. paired evidence: {agent_rank.size} users, identical cohort order ---")

    # comparator must be the frozen fixed_fusion artifact, not a recomputation
    frozen = np.load(REPO / "runs/m3_evidence/phase5_fixed_fusion_hits.npz")
    if not np.array_equal(comp_rank, np.asarray(frozen["target_rank"], dtype=np.int64)):
        print("  STOP: comparator ranks differ from the frozen fixed_fusion evidence")
        return 2
    print("  comparator ranks identical to the frozen fixed_fusion artifact: True")

    # ---- 3. consistency check (must pass before any new statistic) ---------------- #
    print("\n--- 3. consistency check against the published M3 statistics ---")
    problems = []
    per_k: dict[str, dict] = {}
    for k in K_VALUES:
        a_hit = agent_rank <= k
        c_hit = comp_rank <= k
        b = int((a_hit & ~c_hit).sum())
        c = int((~a_hit & c_hit).sum())
        got = {
            "agent_hits": int(a_hit.sum()), "comparator_hits": int(c_hit.sum()),
            "gained": b, "lost": c,
            "recall_delta": float(a_hit.mean() - c_hit.mean()),
            "mcnemar_p": mcnemar(b, c),
        }
        for key in ("agent_hits", "comparator_hits", "gained", "lost"):
            if got[key] != EXPECTED[key][k]:
                problems.append(f"K={k} {key}: recomputed {got[key]} != published {EXPECTED[key][k]}")
        if abs(got["recall_delta"] - EXPECTED["recall_delta"][k]) > 1e-12:
            problems.append(f"K={k} recall_delta: {got['recall_delta']} != {EXPECTED['recall_delta'][k]}")
        if abs(got["mcnemar_p"] - EXPECTED["mcnemar_p"][k]) > 1e-300:
            problems.append(f"K={k} mcnemar_p: {got['mcnemar_p']} != {EXPECTED['mcnemar_p'][k]}")
        print(f"  K={k:<2d} agent_hits={got['agent_hits']:<4d} comparator_hits={got['comparator_hits']:<4d} "
              f"gained/lost={got['gained']}/{got['lost']} delta={got['recall_delta']:+.6f} "
              f"p={got['mcnemar_p']:.6e}")
        per_k[f"K={k}"] = got
    if problems:
        print("\n" + "!" * 100)
        print("STOP — alignment bug; the evidence does not reproduce its own published figures.")
        for p in problems:
            print("   ", p)
        print("!" * 100)
        return 2
    print("  CONSISTENCY: PASS — every published figure reproduced exactly")

    # ---- 4. primary endpoint: NDCG, existing semantics, preregistered bootstrap --- #
    print("\n--- 4. NDCG (recommendation.evaluation.metrics.ndcg_at_k) ---")
    ndcg: dict[str, dict] = {}
    for k in K_VALUES:
        a = ndcg_vector(agent_rank, k)
        c = ndcg_vector(comp_rank, k)
        obs, lo, hi = paired_bootstrap(a - c)
        ndcg[f"K={k}"] = {
            "agent_ndcg": float(a.mean()), "comparator_ndcg": float(c.mean()),
            "delta": obs, "ci95_low": lo, "ci95_high": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "direction": "agent_below" if obs < 0 else "agent_above",
        }
        excl = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  K={k:<2d} agent={a.mean():.10f} comparator={c.mean():.10f} "
              f"delta={obs:+.10f} CI=[{lo:+.10f},{hi:+.10f}] ({excl})")

    primary = ndcg["K=10"]
    print(f"\n  PRIMARY ΔNDCG@10 = {primary['delta']:+.10f} "
          f"CI [{primary['ci95_low']:+.10f}, {primary['ci95_high']:+.10f}]")

    # ---- 5. behaviour ------------------------------------------------------------ #
    print("\n--- 5. behaviour from agent_behavior.jsonl ---")
    records = [json.loads(line) for line in (EXEC / "agent_behavior.jsonl").read_text().splitlines() if line]
    if len(records) != COHORT:
        print(f"  STOP: {len(records)} behaviour records != {COHORT}")
        return 2
    actions: collections.Counter = collections.Counter()
    sequences: collections.Counter = collections.Counter()
    terminations: collections.Counter = collections.Counter()
    ranking_sizes: collections.Counter = collections.Counter()
    for r in records:
        seq = tuple(r.get("action_sequence") or ())
        sequences[",".join(seq) or "none"] += 1
        for action in seq:
            actions[action] += 1
        terminations[r.get("termination_reason") or "none"] += 1
        ranking_sizes[int(r.get("ranking_size", 0))] += 1
    behavior = {
        "users": len(records),
        "action_frequency": dict(actions.most_common()),
        "action_sequence_distribution": dict(sequences.most_common()),
        "recommend_from_history_count": int(actions.get("recommend_from_history", 0)),
        "search_catalog_count": int(actions.get("search_catalog", 0)),
        "select_source_count": int(actions.get("select_source", 0)),
        "termination_reason_distribution": dict(terminations.most_common()),
        "ranking_size_distribution": {str(k): v for k, v in sorted(ranking_sizes.items())},
        "completed": int(sum(1 for r in records if r.get("succeeded"))),
        "failure_count": int(sum(1 for r in records if r.get("failure"))),
        "fallback_count": int(sum(1 for r in records if r.get("budget_exhausted"))),
        "empty_ranking_count": int(sum(1 for r in records if r.get("empty_ranking"))),
        "mean_tool_calls": float(np.mean([int(r.get("tool_calls", 0)) for r in records])),
        "total_tool_calls": int(sum(int(r.get("tool_calls", 0)) for r in records)),
        "mean_ranking_size": float(np.mean([int(r.get("ranking_size", 0)) for r in records])),
        "source_selection_distribution_note": (
            "source_selection is empty for every user because SELECT_SOURCE is structurally "
            "unreachable in the frozen control plane (_plane_actions offers it only when "
            "SIMILAR_ITEM is registered, which it is not). This does NOT mean no retrieval tool "
            "ran: recommend_from_history executes through the candidate plane and the capability."
        ),
    }
    print(f"  action frequency       : {behavior['action_frequency']}")
    print(f"  sequence distribution  : {behavior['action_sequence_distribution']}")
    print(f"  recommend_from_history : {behavior['recommend_from_history_count']}")
    print(f"  search_catalog         : {behavior['search_catalog_count']}")
    print(f"  select_source          : {behavior['select_source_count']} (unreachable by design)")
    print(f"  completed/failure/fallback: {behavior['completed']}/{behavior['failure_count']}/{behavior['fallback_count']}")
    print(f"  ranking sizes          : {behavior['ranking_size_distribution']}")
    print(f"  mean tool calls        : {behavior['mean_tool_calls']}")

    out = {
        "primary_endpoint": {"name": "NDCG@10", **primary},
        "ndcg": ndcg,
        "recall_hit_consistency": per_k,
        "consistency_passed": True,
        "behavior": behavior,
        "artifact_sha256": hashes,
        "protocol": {"k_values": list(K_VALUES), "cohort": COHORT,
                     "resamples": RESAMPLES, "seed": SEED,
                     "ndcg_semantics": "recommendation.evaluation.metrics.ndcg_at_k",
                     "evaluator": "recommendation.evaluation.batched.evaluate_batched"},
    }
    (EXEC / "m3_ndcg_statistics.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\n  wrote {EXEC / 'm3_ndcg_statistics.json'} (originals untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
