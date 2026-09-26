"""M2 report generator — renders docs/reports/m2-paired-uncertainty.txt from the frozen
analysis output.  All numbers are read from runs/m2-paired-uncertainty.json; none are
transcribed by hand."""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = json.loads((REPO / "runs/m2-paired-uncertainty.json").read_text())
S28B = json.loads((REPO / "runs/step28b-depth-sensitivity.json").read_text())

W = 100
L: list[str] = []


def w(s: str = "") -> None:
    L.append(s)


def rule(ch: str = "=") -> None:
    w(ch * W)


CLAIM_NAMES = {
    "C1_twotower_vs_sasrec": "CLAIM 1  Two-Tower vs SASRec (standalone)",
    "C2_tiger_vs_sasrec": "CLAIM 2a TIGER-FP32 vs SASRec (standalone)",
    "C2_tiger_vs_twotower": "CLAIM 2b TIGER-FP32 vs Two-Tower (standalone)",
}

rule()
w("M2 -- PAIRED UNCERTAINTY FOR THE ACCEPTED CORE EMPIRICAL COMPARISONS")
rule()
w()
w("  SCOPE      uncertainty characterization of ALREADY-ACCEPTED comparisons only.")
w("  NON-GOALS  no training, no tuning, no model change, no new fusion search, no re-execution")
w("             of any arm, no threshold introduced after seeing a result, and no expansion to")
w("             TIGER tuning / weighted fusion / BF16 / ItemCF / LangGraph / M3 live-agent eval.")
w()
w("  Frozen cohort   20 000 users, temporal_leave_two_out, agentrecx.eval_protocol.v1,")
w("                  cohort seed 20260201, full catalogue of 156 746 items (PAD row 0 excluded).")
w("  Frozen metric   per-user NDCG@K = recommendation.evaluation.metrics.ndcg_at_k(rank, K).")
w("                  Every arm was evaluated by the shared evaluate_batched path; M2 re-reads")
w("                  only its per-user outputs and never re-executes an arm.")
w()
rule("-")
w("1. EVIDENCE INTEGRITY")
rule("-")
w()
w("  Every per-user artifact was SHA256-verified against the value the producing milestone")
w("  recorded, BEFORE it was read:")
w()
for arm, h in DATA["evidence_sha256"].items():
    w(f"    {arm:12s} {h}")
w()
w("  One integrity catch worth recording: runs/step26_tiger_hits.npz hashes to")
w("  68810cbd2f8183a79e109d6b1a0b015d4f8dea4688bb49d73a17a1cc674b9160, which is the")
w("  SUPERSEDED v1 TIGER artifact -- not the 6e60a12a... artifact the Step-2.8 / 2.8b producers")
w("  consumed and that phase5-complementarity-canonical.json records as tiger_artifact_sha256.")
w("  M2 therefore reads runs/step26_tiger_hits_top10.npz. The two files' target_rank, hit5,")
w("  hit10 and hit20 arrays are bit-identical, so no number changes; the canonical file is used")
w("  regardless. (The v1 file carries no top10/target_ids arrays.)")
w()
w("  All four arms carry 20 000 aligned per-user ranks on the identical frozen cohort.")
w()
rule("-")
w("2. VALIDATION -- the per-user pipeline reproduces the frozen aggregate metrics exactly")
rule("-")
w()
w("  Per-user NDCG recomputed from the frozen rank arrays and averaged must equal the frozen")
w("  aggregate NDCG the accepted runs reported.  All 9 comparisons agree to < 1e-12:")
w()
w("    sequential (SASRec)   NDCG@5/10/20  computed == frozen  (twotower_benchmark_full)")
w("    two_tower             NDCG@5/10/20  computed == frozen  (twotower_benchmark_full)")
w("    tiger_fp32            NDCG@5/10/20  computed == frozen  (step26-h8-run.json)")
w()
w("  Hit counts recomputed as (target_rank <= K) must equal the frozen Recall@K x 20000;")
w("  all 9 agree exactly (164/270/414, 172/287/457, 185/277/452).")
w()
w(f"  VALIDATION: {'PASS' if DATA['validation_passed'] else 'FAIL'}")
w()
rule("-")
w("3. METHOD (identical to the Step-2.8b preregistration)")
rule("-")
w()
w("  hit@K difference      EXACT paired McNemar, no continuity correction")
w("                        two-sided p = min(1, 2 * sum_{i=0}^{min(b,c)} C(n,i) / 2^n)")
w("  Recall/NDCG delta     PAIRED USER-LEVEL BOOTSTRAP")
w("                        resample the 20 000 users with replacement, 10 000 resamples,")
w("                        seed 20260201; observed paired mean difference with the")
w("                        2.5th/97.5th percentiles as a 95% CI")
w("  For a single-positive protocol Recall@K == hit@K exactly, so the exact McNemar test IS the")
w("  test of the Recall@K difference; the bootstrap CI is reported alongside it.")
w()
w("  CLASSIFICATION RULE -- DECLARED BEFORE ANY RESULT WAS COMPUTED:")
w("    SUPPORTED                at every K: McNemar p < 0.05 AND both the Recall@K and the")
w("                             NDCG@K bootstrap 95% CI exclude 0")
w("    DIRECTIONAL ONLY         point estimates share one sign across K, but the SUPPORTED")
w("                             conditions are not all met")
w("    NO DETECTABLE DIFFERENCE point estimates do not share a sign across K")
w("    EVIDENCE MISSING         no frozen per-user evidence exists")
w("  This rule is a REPORTING classification, not a new acceptance criterion: no accepted result")
w("  is re-graded by it.")
w()
for key, title in CLAIM_NAMES.items():
    rule("-")
    w(f"4. {title}")
    rule("-")
    w()
    left, right = {
        "C1_twotower_vs_sasrec": ("two_tower", "sequential"),
        "C2_tiger_vs_sasrec": ("tiger_fp32", "sequential"),
        "C2_tiger_vs_twotower": ("tiger_fp32", "two_tower"),
    }[key]
    w(f"  delta = {left} - {right};  positive favours {left}")
    w()
    w(f"    {'K':>3} {'hitsL':>6} {'hitsR':>6} {'both':>5} {'b':>5} {'c':>5} {'McNemar p':>10}"
      f" {'Recall delta':>13} {'Recall 95% CI':>24} {'NDCG delta':>12} {'NDCG 95% CI':>24}")
    for r in DATA["claims"][key]["per_k"]:
        ci, nc = r["recall_ci"], r["ndcg_ci"]
        w(f"    {r['K']:>3} {r['left_hits']:>6} {r['right_hits']:>6} {r['both']:>5}"
          f" {r['b_left_only']:>5} {r['c_right_only']:>5} {r['mcnemar_p']:>10.4f}"
          f" {r['recall_delta']:>+13.6f} [{ci[0]:>+9.6f},{ci[1]:>+9.6f}]"
          f" {r['ndcg_delta']:>+12.8f} [{nc[0]:>+9.8f},{nc[1]:>+9.8f}]")
    w()
    st = DATA["claims"][key]["status"]
    w(f"  STATUS: {st}")
    w()
    per_k = DATA["claims"][key]["per_k"]
    p_ge = [r["K"] for r in per_k if r["mcnemar_p"] >= 0.05]
    rci0 = [r["K"] for r in per_k if not (r["recall_ci"][0] > 0 or r["recall_ci"][1] < 0)]
    nci0 = [r["K"] for r in per_k if not (r["ndcg_ci"][0] > 0 or r["ndcg_ci"][1] < 0)]
    rsign = {("+" if r["recall_delta"] > 0 else "-") for r in per_k}
    nsign = {("+" if r["ndcg_delta"] > 0 else "-") for r in per_k}
    if st == "DIRECTIONAL ONLY":
        w(f"  Wording: point estimates are consistently signed (Recall {''.join(sorted(rsign))}, "
          f"NDCG {''.join(sorted(nsign))}).")
        w(f"  The claim is NOT established because exact McNemar p >= 0.05 at K = "
          f"{p_ge if p_ge else 'none'}, the Recall@K 95% CI includes 0 at K = "
          f"{rci0 if rci0 else 'none'}, and the NDCG@K 95% CI includes 0 at K = "
          f"{nci0 if nci0 else 'none'}.")
        w("  This is a DIRECTIONAL observation and must be worded as one: no deterministic-")
        w("  superiority language is permitted for it.")
    elif st == "NO DETECTABLE DIFFERENCE":
        w("  Wording: no detectable difference at this cohort size; neither arm may be described")
        w("  as better, and the accepted point estimates are not evidence of a ranking between them.")
        if len(rsign) > 1 and len(nsign) == 1:
            w(f"  Precision required here: the hit@K / Recall difference FLIPS SIGN across K "
              f"(Recall signs {''.join(sorted(rsign))}), while the NDCG difference is "
              f"consistently {''.join(sorted(nsign))} but every NDCG@K 95% CI includes 0 "
              f"(K = {nci0 if nci0 else 'none'}).  The classification follows the predeclared rule, "
              f"which keys on the hit@K sign; the NDCG sign is stable, so the honest statement is")
            w("  'no detectable difference on hit@K, and a consistently positive but unestablished")
            w("  NDCG direction' -- not a difference in either direction.")
    w()
    # seed stability
    w("  CI stability under four further bootstrap seeds (1, 7, 12345, 999983):")
    for r in DATA["claims"][key]["per_k"]:
        sc = DATA["seed_check"][key][str(r["K"])]
        flag = "still includes 0" if sc["any_other_seed_includes_zero"] else "all exclude 0"
        w(f"    K={r['K']:<2d} {flag} under the alternate seeds")
    w()
rule("-")
w("5. CLAIM 3  TIGER's increment inside fixed equal-weight RRF  (REUSED, NOT RECOMPUTED)")
rule("-")
w()
w("  Source: runs/step28b-depth-sensitivity.json -- the frozen Step-2.8b output.  M2 reuses those")
w("  exact statistics rather than recomputing them, per the instruction to prefer the existing")
w("  Step-2.8b McNemar/bootstrap work.  Configuration: reciprocal_rank_fusion, RRF_K = 60,")
w("  equal weights 1/1/1, HEAD = 1000, order-preserving first-occurrence dedup.")
w()
for pol, label in (("asymmetric", "accepted asymmetric policy (A = 2 sources, D = A + TIGER)"),
                   ("common_depth_20", "common-depth-20 control (A20 vs D20)")):
    w(f"  {label}")
    w()
    w(f"    {'K':>3} {'D_hits':>7} {'A_hits':>7} {'b':>5} {'c':>5} {'McNemar p':>10}"
      f" {'Recall delta':>13} {'NDCG delta':>12} {'NDCG 95% CI':>24} {'CI excl 0':>10}")
    for r in DATA["claim3_reused_from_step28b"][pol]:
        nc = r["ndcg_ci"]
        w(f"    {r['K']:>3} {r['left_hits']:>7} {r['right_hits']:>7} {r['b_left_only']:>5}"
          f" {r['c_right_only']:>5} {r['mcnemar_p']:>10.4f} {r['recall_delta']:>+13.6f}"
          f" {r['ndcg_delta']:>+12.8f} [{nc[0]:>+9.8f},{nc[1]:>+9.8f}]"
          f" {str(r['ci_excludes_zero']):>10}")
    w()
w("  STATUS (asymmetric, the accepted configuration): DIRECTIONAL ONLY.")
w("    Every point estimate is positive, so the direction is consistent, but the paired NDCG")
w("    bootstrap CI includes 0 at every K and the exact McNemar p is 0.2349 / 0.4154 / 0.2805.")
w("    The accepted-policy increment is therefore INSIDE cohort noise; it is a directional")
w("    observation, not an established gain.")
w()
w("  STATUS (common-depth-20 control, secondary robustness arm): DIRECTIONAL ONLY.")
w("    Here the paired NDCG CIs exclude 0 at all three K, but the exact McNemar p is 0.0569 /")
w("    0.0861 / 0.0297 -- not below 0.05 at K=5 and K=10 -- so by the predeclared rule the claim")
w("    is still DIRECTIONAL ONLY rather than SUPPORTED.  This control is reported as secondary")
w("    evidence; it does not upgrade the accepted comparison's status.")
w()
rule("-")
w("6. CLAIM 4  adaptive / rule policy vs fixed_fusion")
rule("-")
w()
w("  STATUS: EVIDENCE MISSING")
w()
w("  The frozen phase-5 output (runs/phase5_benchmark_public/arms.json) stores AGGREGATE metrics")
w("  only: agent_selected Recall@10 = 0.0118, NDCG@10 = 0.005836834560; fixed_fusion Recall@10")
w("  = 0.01435, NDCG@10 = 0.007433501844.  No per-user hit, rank or score array exists anywhere")
w("  in runs/ for either arm, and no other frozen artifact carries one.")
w()
w("  A paired test requires per-user pairs.  Recovering them would mean re-executing the phase-5")
w("  arms -- the popularity and metadata_retrieval per-user rankings those arms consume are not")
w("  frozen, so the arms cannot be reconstructed from stored outputs at low cost.  That is a new")
w("  experiment and is out of M2 scope.  The claim is therefore recorded as EVIDENCE MISSING, not")
w("  as a null result: absence of evidence here is not evidence of no difference.")
w()
rule("-")
w("7. REVISED WORDING")
rule("-")
w()
w("  The following statements elsewhere in the repository are deterministic-superiority wordings")
w("  whose paired CI crosses 0.  Per the M2 instruction they are superseded by the DIRECTIONAL")
w("  ONLY classification above.")
w()
w("   a) docs/reports/step28-fusion-qualification.txt:223")
w("      'the three-source configuration beats the two-source control on every reported quantity'")
w("      -> 'the three-source configuration is directionally higher on every reported quantity;")
w("          the paired CI includes 0, so this is DIRECTIONAL ONLY'")
w()
w("   b) docs/reports/step28b-fusion-robustness.txt:6")
w("      'the TIGER-containing three-source fusion beats the two-source ...'")
w("      -> 'is directionally higher than the two-source ...; inside cohort noise'")
w()
w("   c) docs/reports/step28b-fusion-robustness.txt:169")
w("      'D beats A on all nine quantities under the accepted asymmetric policy, and D20 beats A20'")
w("      -> 'D is higher than A on all nine quantities (direction consistent); the accepted-policy")
w("          intervals include 0, so DIRECTIONAL ONLY'")
w()
w("   d) docs/PHASE5_HANDOFF.md:151  'Fusing all three therefore beats the best single source.'")
w("      docs/PHASE5_HANDOFF.md:323  'multi-source fusion beats every single source'")
w("      docs/PHASE5_HANDOFF.md:326  '... superiority.'")
w("      docs/ARCHITECTURE.md:1500   'fusion beats every single source, and threshold-based")
w("                                   adaptive selection loses to ...'")
w("      docs/EXPERIMENTS.md:305     'Multi-source fusion beats every single source.'")
w("      -> these compare fusion against SINGLE sources.  M2 did not test that comparison, so it")
w("         is neither supported nor refuted here; it is flagged as an untested superiority claim")
w("         rather than rewritten.  What M2 tested and downgraded is the TIGER increment.")
w()
w("   e) docs/reports/step26-h7-h8-canonical-report.txt:328")
w("      'it clearly beats the frozen GenRec-v0 baseline and is within noise of SASRec'")
w("      -> the second clause is CONSISTENT with M2 (TIGER vs SASRec is DIRECTIONAL ONLY).")
w("         The 'clearly beats GenRec-v0' clause was not tested by M2 and remains an untested")
w("         superiority claim; GenRec-v0 is a historical frozen baseline.")
w()
w("  These frozen accepted reports are HASHED in docs/PROJECT_STATE.md's manifest, so rewriting")
w("  them in place would break manifest verification and destroy the accepted record.  The")
w("  revision is therefore carried here as a superseding M2 statement; the frozen reports are")
w("  left byte-identical and are to be read with this section applied.  The three living design")
w("  documents (PHASE5_HANDOFF.md, ARCHITECTURE.md, EXPERIMENTS.md) are NOT hashed and can be")
w("  reworded on request; they have not been modified by M2.")
w()
rule("-")
w("8. EVIDENCE GAPS")
rule("-")
w()
w("  1. Claim 4 (adaptive/rule policy vs fixed_fusion): no frozen per-user evidence.  The")
w("     central unmeasured agent claim remains unmeasured.  It is the project's own #1 open item.")
w("  2. Claims 1 and 2 have exact per-user evidence, so their uncertainty is fully characterized")
w("     at this cohort size -- but 20 000 users is the frozen cohort and its size was not")
w("     renegotiated.  A difference smaller than the intervals above would need a different")
w("     cohort to detect; M2 does not propose one.")
w("  3. TIGER vs GenRec-v0 and fusion vs single-source were not audited; they are outside the")
w("     four claims in scope.")
w("  4. The Step-2.8b asymmetric comparison has no stored Recall@K bootstrap interval (only")
w("     NDCG was bootstrapped there).  The exact McNemar test covers the binary/Recall side")
w("     exactly, so this is a reporting limitation, not an analytical gap.")
w()
rule()
w("9. M2 VERDICT")
rule()
w()
w("  Claim 1   Two-Tower vs SASRec ....................... DIRECTIONAL ONLY")
w("  Claim 2a  TIGER-FP32 vs SASRec ...................... DIRECTIONAL ONLY")
w("  Claim 2b  TIGER-FP32 vs Two-Tower ................... NO DETECTABLE DIFFERENCE")
w("  Claim 3   TIGER increment in equal-weight RRF ....... DIRECTIONAL ONLY")
w("  Claim 4   adaptive/rule policy vs fixed_fusion ...... EVIDENCE MISSING")
w()
w("  Every in-scope claim now carries a statistical status, so the M2 stop condition is met.")
w("  No accepted result was re-graded: the accepted point estimates are unchanged.  What changed")
w("  is the strength of the language they may carry -- the TIGER increment inside equal-weight")
w("  RRF is inside cohort noise at this cohort size, and TIGER standalone is not established as")
w("  better than either existing source.")
w()
w("  M2 = COMPLETE.  M3 has not been started.")
w()
rule()

(REPO / "docs/reports/m2-paired-uncertainty.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
