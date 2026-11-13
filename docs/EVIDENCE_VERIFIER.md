# Independent Evidence Verifier — Reliability, GPU Comparison, Shadow E2E

## 1. Existing Evidence Audit

P15 already provides:

- `EvidenceBundle` with product spans, preference provenance, unknown fields and Gate context;
- `validate_claims()` for claim types, evidence refs, product/user ownership, exact spans,
  inactive/retracted preferences, UNKNOWN promotion and unsupported recommendation refs;
- deterministic `render_final_answer()`;
- explicit `semantic_truth_verified: false`.

P15 is therefore **structural/lexical validation**. It can prove that a citation exists and
belongs to the selected product/user; it cannot prove that the cited text entails the claim.

The independent verifier is a second role over the existing Qwen `/explain` service. It gets
only product id, claim text/refs and evidence; it does not get P15 validation, Planner hidden
state, Gate decision or final-answer rationale.

## 2. Four False-Negative Root Causes and Fixes

| Case | Claim / citation | P15 | Old verifier | Root cause | Fix |
|---|---|---|---|---|---|
| `d05` | “over-ear”, cites metadata categories `pev_08` | accepted | SUPPORTED | C4 categories say over-ear while metadata details say on-ear; the verifier saw only the cited hint and did not resolve the cross-source conflict | deterministic form-factor consistency precheck across **all** product evidence |
| `p06` | “prefer water resistance”, cites wireless preference `mev_487...` | accepted | SUPPORTED | the verifier used the otherwise-present water preference instead of the claim's own cited ref | deterministic preference canonical-value check; only cited preference + product context supplied to the model |
| `p07` | “prefer wireless”, cites water preference `mev_461...` | accepted | SUPPORTED | same wrong-reference substitution | same preference provenance precheck |
| `p08` | “prefer wireless”, no evidence refs | rejected | SUPPORTED | model inferred from surrounding bundle evidence although the claim cited nothing | no-ref precheck returns INSUFFICIENT before any model call |

Additional issues found during the repair:

- `x03`: cited the wrong product field for a mechanical-switch claim → attribute precheck;
- `p02`: “want to avoid over-ear” was misread as positive polarity → negated-preference parser;
- `d02`: brand claim was a false rejection because the brand evidence had no `attribute` key →
  brand exception + attribute alias handling;
- a real Qwen shadow claim “IPX5 Sweatproof” was initially a false rejection because the
  deterministic attribute matcher did not know IPX/water aliases → alias support.

No case-specific hardcoding was added. All fixes are generic citation, attribute, polarity,
and source-consistency rules.

## 3. Claim ID and Evidence Ref Reliability

Changes:

- one claim per verifier call for evaluation, shadow and any future opt-in path;
- multi-claim batches must preserve every `claim_id` exactly; a renamed/missing/reordered
  multi-claim response is `invalid_model_payload`, never silently reordered;
- a single-claim call may map one renamed id to its one expected claim, recorded as
  `id_alignment=single_claim`;
- missing verdict objects are not invented;
- Qwen's occasional missing `]` / extra brace is recovered only by parsing complete verdict
  objects or one complete JSON object; invalid verdict enums remain invalid;
- product-fact claims receive only their own cited evidence, so unrelated evidence cannot
  substitute for a wrong citation;
- preference claims receive the cited preference **and product evidence**, because matching a
  preference requires both sides;
- no refs, foreign-product refs, wrong preference refs, inactive preferences, conflicting
  form-factor evidence and absent attributes are rejected by deterministic prechecks.

Unit tests in `tests/test_evidence_verifier.py` cover exact mapping, single-claim mapping,
multi-claim mismatch, missing-array repair, no-ref claims, wrong preference source, metadata
conflict, IPX alias, transport failure and prompt independence.

## 4. Evaluation Data and Review Status

- 49 real-catalogue Claim–Evidence cases;
- 4 products (`B004K6R43G`, `B07WZFL74B`, `B09H2JQD48`, `B00005USA4`);
- 41 dev / 8 holdout;
- 19 SUPPORTED / 11 CONTRADICTED / 19 INSUFFICIENT_EVIDENCE;
- label status: **provisional_rule_authored_not_human_verified**;
- engineer review checklist exists under `runs/p20_metadata/verifier_review_checklist.json`.

**Independent human review was not obtained.** We did not upgrade any label to human gold.
The holdout was not used for prompt tuning, but it remains provisional. All claims are judged
only against the supplied evidence, not against real-world product truth.

## 5. P15 vs Qwen-only vs Combined

Three separate arms, same 49 cases and same Qwen checkpoint:

- **A** = P15 deterministic structural validator;
- **B** = Qwen-only verifier: no deterministic prechecks, one claim per call;
- **C** = P15 structural rejection + deterministic citation/conflict prechecks + Qwen verifier.

| Metric | A: P15 | B: Qwen-only | C: P15 + guards + Qwen |
|---|---:|---:|---:|
| Unsupported detection recall | 2/30 = 6.7% | 27/30 = **90.0%** | **30/30 = 100%** |
| Unsupported detection precision | — | **100%** | **100%** |
| Supported accuracy | — | **19/19 = 100%** | **19/19 = 100%** |
| Supported false rejection | 0% | **0%** | **0%** |
| INSUFFICIENT correct | — | 16/19 | 19/19 |
| Citation/conflict cases detected | — | 4/7 | **7/7** |
| Invalid JSON/verdict | — | **0** | **0** |
| Model calls | 0 | 49 | 34 |

B misses exactly the cases requiring cross-source/attribute guards: `d05`, `d06` (C4 vs
metadata form-factor conflict) and `x03` (mechanical-switch claim citing a waterproof field).
C catches those three deterministically and remains supported-claim safe.

**C vs A:** 6.7% → 100% unsupported recall on this provisional pilot.
**C vs B:** C adds the three guard-required cases; it is also cheaper (34 vs 49 model calls,
36.8k vs 64.1k input tokens) because deterministic checks skip calls.
B is not a superset of C, and C is not claimed to be a production-grade replacement for
human semantic review.

Dev: supported 16/16, unsupported 25/25.
Holdout: supported 3/3, unsupported 5/5 (only 8 cases).

The perfect C pilot score must not be read as general production accuracy: labels are
provisional, the set is small, and no human gold exists.

## 6. GPU Execution Evidence

- GPU: AutoDL RTX 4090 24 GB
- existing checkpoint: Qwen3.5-9B-ms, same `/explain` service
- no second 9B model, no retraining, no new dependency
- service baseline VRAM: 17,549 MiB
- observed global verifier peak: **21,975 MiB**
- total VRAM: 24,564 MiB; peak free: **2,118 MiB**
- B Qwen-only peak during its 49-call run: 19,335 MiB
- no OOM; the margin is still modest

### Pilot

- C: 49 cases; 34 model calls; 15 deterministic precheck-only cases
- C input: **36,821 tokens**; output: **2,100 tokens**
- C verifier call latency: P50 **2,697 ms**, P95 **3,428 ms**
- C total verifier call latency: **91,922 ms**
- B Qwen-only: 49 calls, **64,140 input / 3,136 output tokens**, total 118,591 ms;
  P50 2,382 ms, P95 3,538 ms
- ID alignment: C 34 `single_claim`, 15 precheck-only; B 49 `single_claim`; no positional fallback
- invalid JSON/verdict: **0** in both B and C

### Shadow E2E

- 6 real Qwen tasks (`0041`, `0042`, `0043`, `0044`, `0049`, `0055`)
- 23 claims / 23 verifier calls
- verifier verdicts: **23/23 SUPPORTED**
- verifier tokens: 36,557 input / 1,404 output
- verifier call time: about 59.1 s total
- E2E total: 261.6 s including main Qwen Planner/Explain
- Gate and final answers unchanged; no unsupported claim occurred

A previous run produced one natural `INSUFFICIENT_EVIDENCE` for the claim
“These earbuds are IPX5 sweatproof” because the first attribute guard did not recognize IPX
as a waterproof alias; that was a verifier false positive and was fixed with a generic IPX /
water-resistance alias. It was not counted as a success.

No opt-in interception was enabled, so there is no interception trace. The reason is in
Section 8.

## 7. Regression and Tests

Relevant suite:

```text
149 passed, 1 skipped
```

P18 remains the known pre-existing issue:

```text
1 failed, 5 passed
```

P18 Frozen Manifest was not modified; no P18 Reserved tuning was used.

Git:

- previous verifier MVP commit: `0963020`
- this round changes are committed after the final test run
- no push

## 8. Adoption Decision

**Keep the verifier as a shadow/experimental module; do not enable opt-in interception yet.**

Reasons:

1. No human gold labels; 49 cases are provisional and small.
2. The 100% result is measured on authored cases and may overstate production performance.
3. The six-task shadow run found no natural unsupported claim after the fixes; there is no
   real interception trace to validate deletion/regeneration UX.
4. A single Qwen checkpoint still has correlated semantic risk.
5. Peak VRAM is 21,975 MiB with roughly 2.1 GiB free; adding generation fallback/concurrency
   would need careful resource control.
6. A bounded opt-in path can be designed next, but only after independent human review and more
   real shadow cases; it must not hard-block candidate selection or Memory.

Correct current behavior:

- Verifier disabled: original P15/Gate/answer behavior unchanged;
- Shadow: verifier verdicts are recorded only;
- Gate remains authoritative;
- Verifier cannot modify selection, Memory, constraints or final answer.
