"""M2 — paired uncertainty for the accepted core empirical comparisons.

Scope (strict): characterize uncertainty on ALREADY-ACCEPTED comparisons.
No training, no tuning, no model change, no new fusion search, no re-run of any arm.

Evidence policy:
  * frozen per-user artifacts only, each SHA256-verified against the value the
    producing milestone recorded before use;
  * nothing is regenerated; no arm is re-executed;
  * Claim 3 reuses the frozen Step-2.8b statistics verbatim rather than recomputing.

Method (mirrors the Step-2.8b preregistration exactly):
  * hit@K difference  -> EXACT paired McNemar, no continuity correction
        two-sided exact p = min(1, 2 * sum_{i=0}^{min(b,c)} C(n,i) / 2^n)
  * Recall/NDCG delta -> PAIRED USER-LEVEL BOOTSTRAP
        resample the 20 000 users with replacement, 10 000 resamples, seed 20260201,
        report the observed paired mean difference and the 2.5th/97.5th percentiles as a 95% CI
  * per-user NDCG@K = recommendation.evaluation.metrics.ndcg_at_k(rank, K)  (frozen primitive)
  * same frozen 20k cohort, same shared evaluator outputs

CLASSIFICATION RULE — DECLARED BEFORE ANY RESULT IS COMPUTED
  SUPPORTED                : at every K in {5,10,20}: exact McNemar p < 0.05 AND the paired
                             bootstrap 95% CI excludes 0 (for both Recall@K and NDCG@K delta)
  DIRECTIONAL ONLY         : the point estimates share one sign across all K, but the
                             SUPPORTED conditions are not all met (some CI includes 0 or
                             some p >= 0.05)
  NO DETECTABLE DIFFERENCE : the point estimates do not share a sign across K
  EVIDENCE MISSING         : no frozen per-user evidence exists for the claim
  Any claim whose CI crosses 0 is worded as DIRECTIONAL ONLY; no deterministic-superiority
  wording is permitted for it.  This rule is a REPORTING classification, not a new acceptance
  criterion: no accepted result is re-graded by it, and thresholds are not applied post hoc to
  any comparison other than the ones enumerated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np

from recommendation.evaluation.metrics import ndcg_at_k

REPO = pathlib.Path(__file__).resolve().parents[2]
COHORT = 20_000
K_VALUES = (5, 10, 20)
SEED = 20260201
RESAMPLES = 10_000
ALPHA_P = 0.05

# ---------------------------------------------------------------- frozen evidence
EVIDENCE = {
    "sequential": (
        "runs/step26_baseline_evidence/phase5_sasrec_hits.npz",
        "3072c86055cd4d093147ef10e19ed8121fb70215d581198b3261380df83d0252",
    ),
    "two_tower": (
        "runs/step26_baseline_evidence/phase5_twotower_hits.npz",
        "71516247b7685bee34e161a9d3c81efd57cb8c2a51c94e0efb7937b2eaf34229",
    ),
    "genrecv0": (
        "runs/step26_baseline_evidence/phase5_genrecv0_hits.npz",
        "54f59ea45d8c38ba234124c61c60b7e30914dde3494c3cac18112b86870b5e7d",
    ),
    "tiger_fp32": (
        # NOTE: runs/step26_tiger_hits.npz is the SUPERSEDED v1 artifact (sha256 68810cbd...).
        # The canonical artifact the frozen Step-2.8 / 2.8b producers consumed is this file,
        # whose hash matches phase5-complementarity-canonical.json's "tiger_artifact_sha256".
        # Its target_rank / hit5 / hit10 / hit20 arrays are bit-identical to the v1 file; it
        # additionally carries case_index, target_ids and top10.
        "runs/step26_tiger_hits_top10.npz",
        "6e60a12a056b2c7263dd52535722f803b9100905a2d08c905c9e0433eaa98641",
    ),
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p, exactly as the Step-2.8b preregistration defines it."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def ndcg_vector(ranks: np.ndarray, k: int) -> np.ndarray:
    return np.array([ndcg_at_k(int(r), k) if int(r) >= 1 else 0.0 for r in ranks], dtype=np.float64)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = SEED, resamples: int = RESAMPLES):
    """Percentile bootstrap of the paired mean difference a - b (chunked for memory)."""
    d = (a - b).astype(np.float64)
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 500
    done = 0
    while done < resamples:
        take = min(chunk, resamples - done)
        idx = rng.integers(0, n, size=(take, n))
        means[done : done + take] = d[idx].mean(axis=1)
        done += take
    return (
        float(d.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def ci_excludes_zero(ci: tuple[float, float]) -> bool:
    return (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)


def classify(per_k: list[dict]) -> str:
    """Apply the predeclared rule to one claim's per-K statistics.

    SUPPORTED requires, at every K: exact McNemar p < 0.05 AND both the Recall@K and the
    NDCG@K paired-bootstrap 95% CI exclude 0.  Otherwise a single consistent sign across K
    gives DIRECTIONAL ONLY; inconsistent signs give NO DETECTABLE DIFFERENCE.
    """
    supported = all(
        r["mcnemar_p"] < ALPHA_P
        and ci_excludes_zero(r["recall_ci"])
        and ci_excludes_zero(r["ndcg_ci"])
        for r in per_k
    )
    if supported:
        return "SUPPORTED"
    signs = {math.copysign(1.0, r["recall_delta"]) for r in per_k}
    if len(signs) == 1:
        return "DIRECTIONAL ONLY"
    return "NO DETECTABLE DIFFERENCE"


def main() -> int:
    print(__doc__)
    print("=" * 100)
    print("EVIDENCE INTEGRITY — every per-user artifact verified against its recorded SHA256")
    print("=" * 100)

    ranks: dict[str, np.ndarray] = {}
    for name, (rel, want) in EVIDENCE.items():
        path = REPO / rel
        got = sha256(path)
        ok = got == want
        print(f"  {name:12s} {rel}")
        print(f"      sha256 {'MATCH' if ok else 'MISMATCH'}  {got}")
        if not ok:
            print(f"      expected {want}")
            return 2
        z = np.load(path, allow_pickle=True)
        r = z["target_rank"].astype(np.int64)
        if r.size != COHORT:
            print(f"      cohort size {r.size} != {COHORT}")
            return 2
        ranks[name] = r

    # all arms are evaluated on the identical frozen cohort, in the same user order
    assert all(r.size == COHORT for r in ranks.values())

    # ---------------------------------------------------------------- validation
    print()
    print("=" * 100)
    print("VALIDATION — per-user NDCG recomputed from frozen ranks vs the frozen aggregate metrics")
    print("=" * 100)
    anchors = json.loads((REPO / "runs/twotower_benchmark_full/comparison.json").read_text())["arms"]
    h8 = json.loads((REPO / "runs/step26-h8-run.json").read_text())
    frozen_ndcg = {
        "sequential": anchors["sequential"]["metrics"]["NDCG"],
        "two_tower": anchors["two_tower"]["metrics"]["NDCG"],
        "tiger_fp32": h8["metrics"]["metrics"]["NDCG"],
    }
    validation_ok = True
    for arm, frozen in frozen_ndcg.items():
        for k in K_VALUES:
            got = float(ndcg_vector(ranks[arm], k).mean())
            want = float(frozen[f"@{k}"])
            match = abs(got - want) < 1e-12
            validation_ok &= match
            print(f"  {arm:11s} NDCG@{k:<2d} recomputed={got:.15f} frozen={want:.15f} "
                  f"{'OK' if match else 'DIFF'}")
    # hit counts vs frozen Recall (single positive => Recall@K == hit@K)
    print()
    for arm, frozen in (("sequential", anchors["sequential"]["metrics"]["Recall"]),
                        ("two_tower", anchors["two_tower"]["metrics"]["Recall"]),
                        ("tiger_fp32", h8["metrics"]["metrics"]["Recall"])):
        for k in K_VALUES:
            hits = int((ranks[arm] <= k).sum())
            want = round(float(frozen[f"@{k}"]) * COHORT)
            ok = hits == want
            validation_ok &= ok
            print(f"  {arm:11s} hits@{k:<2d} recomputed={hits:<4d} frozen={want:<4d} "
                  f"{'OK' if ok else 'DIFF'}")
    print()
    print(f"  VALIDATION: {'PASS' if validation_ok else 'FAIL'} — the per-user pipeline reproduces the "
          f"frozen aggregate metrics exactly")

    # ---------------------------------------------------------------- claims 1 & 2
    def pair_stats(left: str, right: str) -> list[dict]:
        out = []
        for k in K_VALUES:
            lh = ranks[left] <= k
            rh = ranks[right] <= k
            b = int((lh & ~rh).sum())      # left hit, right miss
            c = int((~lh & rh).sum())      # left miss, right hit
            both = int((lh & rh).sum())
            p = exact_mcnemar(b, c)
            ln = ndcg_vector(ranks[left], k)
            rn = ndcg_vector(ranks[right], k)
            rd = float(lh.mean() - rh.mean())
            rci = paired_bootstrap(lh.astype(np.float64), rh.astype(np.float64))
            npt = paired_bootstrap(ln, rn)
            out.append({
                "K": k,
                "left_hits": int(lh.sum()), "right_hits": int(rh.sum()), "both": both,
                "b_left_only": b, "c_right_only": c, "discordant": b + c,
                "mcnemar_p": p,
                "recall_delta": rd,
                "recall_ci": (rci[1], rci[2]),
                "ndcg_delta": npt[0],
                "ndcg_ci": (npt[1], npt[2]),
            })
        return out

    claims = {}
    print()
    print("=" * 100)
    print("CLAIM 1 — Two-Tower vs SASRec (standalone)")
    print("=" * 100)
    claims["C1_twotower_vs_sasrec"] = pair_stats("two_tower", "sequential")
    print_pair(claims["C1_twotower_vs_sasrec"], "two_tower", "sequential")

    print()
    print("=" * 100)
    print("CLAIM 2 — TIGER-FP32 vs SASRec and vs Two-Tower (standalone)")
    print("=" * 100)
    claims["C2_tiger_vs_sasrec"] = pair_stats("tiger_fp32", "sequential")
    print_pair(claims["C2_tiger_vs_sasrec"], "tiger_fp32", "sequential")
    print()
    claims["C2_tiger_vs_twotower"] = pair_stats("tiger_fp32", "two_tower")
    print_pair(claims["C2_tiger_vs_twotower"], "tiger_fp32", "two_tower")

    for name, per_k in claims.items():
        print(f"  -> {name}: {classify(per_k)}")

    # ---------------------------------------------------------------- claim 3 (reuse)
    print()
    print("=" * 100)
    print("CLAIM 3 — TIGER's increment inside fixed equal-weight RRF (REUSED from Step 2.8b)")
    print("=" * 100)
    b28 = json.loads((REPO / "runs/step28b-depth-sensitivity.json").read_text())
    pu = b28["paired_uncertainty"]
    print(f"  source: runs/step28b-depth-sensitivity.json (frozen Step-2.8b output, not recomputed)")
    print(f"  RRF constant {b28['rrf_constant']}, weights {b28['weights']}, head 1000")
    claim3 = {}
    for policy in ("asymmetric", "common_depth_20"):
        left_arm, right_arm = (("D", "A") if policy == "asymmetric" else ("D20", "A20"))
        per_k = []
        for k in K_VALUES:
            e = pu[policy][f"K={k}"]
            mc, nd = e["mcnemar"], e["ndcg"]
            lh = int(b28["configs"][left_arm]["hits"][str(k)])
            rh = int(b28["configs"][right_arm]["hits"][str(k)])
            per_k.append({
                "K": k,
                "left_hits": lh, "right_hits": rh,
                "both": lh - mc["A_miss_D_hit"],
                "b_left_only": mc["A_hit_D_miss"], "c_right_only": mc["A_miss_D_hit"],
                "discordant": mc["discordant"], "mcnemar_p": mc["exact_two_sided_p"],
                "recall_delta": (mc["A_miss_D_hit"] - mc["A_hit_D_miss"]) / COHORT,
                "recall_ci": (float("nan"), float("nan")),
                "ndcg_delta": nd["observed_paired_mean_difference"],
                "ndcg_ci": (nd["ci95_low"], nd["ci95_high"]),
                "ci_excludes_zero": nd["ci_excludes_zero"],
            })
        claim3[policy] = per_k
        print(f"\n  policy = {policy}   (left={left_arm}, right={right_arm})")
        print_pair(per_k, left_arm, right_arm)
    for pol, per_k in claim3.items():
        # Rule application for claim 3: McNemar governs the binary (Recall) side; the frozen
        # bootstrap CI governs the NDCG side.  Recall CI is unavailable from the frozen output.
        supported = all(
            r["mcnemar_p"] < ALPHA_P and (r["ndcg_ci"][0] > 0 and r["ndcg_ci"][1] > 0)
            for r in per_k
        )
        signs = {math.copysign(1, r["ndcg_delta"]) for r in per_k}
        status = "SUPPORTED" if supported else ("DIRECTIONAL ONLY" if len(signs) == 1
                                                else "NO DETECTABLE DIFFERENCE")
        print(f"  -> C3_rrf_increment[{pol}]: {status}")

    # ---------------------------------------------------------------- claim 4
    print()
    print("=" * 100)
    print("CLAIM 4 — adaptive / rule policy vs fixed_fusion")
    print("=" * 100)
    arms = json.loads((REPO / "runs/phase5_benchmark_public/arms.json").read_text())["arms"]
    print(f"  available arms in the frozen phase-5 output: {sorted(arms)}")
    for arm in ("agent_selected", "fixed_fusion"):
        m = arms[arm]["metrics"]
        print(f"    {arm:16s} Recall@10={m['Recall']['@10']} NDCG@10={m['NDCG']['@10']:.12f} "
              f"mean_target_rank={arms[arm]['mean_target_rank']}")
    print("  per-user arrays for these two arms: NONE FROZEN")
    print("    the frozen phase-5 output stores aggregate metrics only; there is no per-user hit,")
    print("    rank or score array for agent_selected or fixed_fusion anywhere in runs/ .")
    print("    A paired test needs per-user pairs; reconstructing them would require re-executing")
    print("    the phase-5 arms (popularity / metadata_retrieval per-user rankings are not frozen),")
    print("    which is out of M2 scope.  => EVIDENCE MISSING")

    # ---------------------------------------------------------------- seed robustness
    print()
    print("=" * 100)
    print("CI STABILITY CHECK — same protocol under 4 further bootstrap seeds")
    print("=" * 100)
    seed_check = {}
    for name in ("C1_twotower_vs_sasrec", "C2_tiger_vs_sasrec", "C2_tiger_vs_twotower"):
        left, right = {
            "C1_twotower_vs_sasrec": ("two_tower", "sequential"),
            "C2_tiger_vs_sasrec": ("tiger_fp32", "sequential"),
            "C2_tiger_vs_twotower": ("tiger_fp32", "two_tower"),
        }[name]
        seed_check[name] = {}
        for k in K_VALUES:
            ln = ndcg_vector(ranks[left], k)
            rn = ndcg_vector(ranks[right], k)
            los, his = [], []
            for s in (1, 7, 12345, 999983):
                _, lo, hi = paired_bootstrap(ln, rn, seed=s)
                los.append(lo); his.append(hi)
            crosses = any(lo <= 0 <= hi for lo, hi in zip(los, his))
            seed_check[name][k] = {
                "other_seeds": [1, 7, 12345, 999983],
                "cis": [[lo, hi] for lo, hi in zip(los, his)],
                "any_other_seed_includes_zero": crosses,
            }
            print(f"  {name:24s} K={k:<2d} seed20260201 CI=({claims[name][K_VALUES.index(k)]['ndcg_ci'][0]:+.7f},"
                  f"{claims[name][K_VALUES.index(k)]['ndcg_ci'][1]:+.7f})  "
                  f"other seeds include 0: {crosses}")

    out = {
        "classification_rule": "see module docstring (declared before results)",
        "cohort": COHORT, "k_values": list(K_VALUES), "seed": SEED, "resamples": RESAMPLES,
        "validation_passed": bool(validation_ok),
        "claims": {k: {"per_k": v, "status": classify(v)} for k, v in claims.items()},
        "claim3_reused_from_step28b": claim3,
        "claim4": {"status": "EVIDENCE MISSING",
                   "reason": "no frozen per-user evidence for agent_selected / fixed_fusion"},
        "seed_check": seed_check,
        "evidence_sha256": {k: v[1] for k, v in EVIDENCE.items()},
    }
    (REPO / "runs/m2-paired-uncertainty.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"  wrote runs/m2-paired-uncertainty.json")
    return 0 if validation_ok else 2


def print_pair(per_k: list[dict], left: str, right: str) -> None:
    print(f"  left={left}  right={right}   (delta = left - right)")
    print(f"    {'K':>3} {'L_hits':>7} {'R_hits':>7} {'both':>6} {'b(L only)':>10} {'c(R only)':>10} "
          f"{'McNemar p':>11} {'Recall d':>11} {'Recall 95% CI':>26} {'NDCG d':>12} {'NDCG 95% CI':>26}")
    for r in per_k:
        rci = r["recall_ci"]
        rcis = ("n/a (not stored)" if math.isnan(rci[0])
                else f"[{rci[0]:>+10.6f},{rci[1]:>+10.6f}]")
        print(f"    {r['K']:>3} {r['left_hits']:>7} {r['right_hits']:>7} {r['both']:>6} "
              f"{r['b_left_only']:>10} {r['c_right_only']:>10} {r['mcnemar_p']:>11.6f} "
              f"{r['recall_delta']:>+11.6f} "
              f"{rcis:>26} "
              f"{r['ndcg_delta']:>+12.8f} "
              f"[{r['ndcg_ci'][0]:>+10.8f},{r['ndcg_ci'][1]:>+10.8f}]")


if __name__ == "__main__":
    raise SystemExit(main())
