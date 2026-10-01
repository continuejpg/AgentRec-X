# B1 — Two-Tower fusion paired evidence: PREREGISTRATION

**Status: FROZEN BEFORE EXECUTION.** No part of the protocol below has been executed. This document
freezes the comparison; B1-Run executes it once.

Governing rules: [`AGENTS.md`](../AGENTS.md) §6 (unified evaluation protocol) and §14
(reproducibility). Method is reused from [`docs/M3_PREREGISTRATION.md`](M3_PREREGISTRATION.md) §5
rather than reinvented, so B1 stays comparable with M2/M3/M4.

---

## 0. Why this exists (the gap being closed)

`docs/TWOTOWER.md` §7 reports a controlled fusion contrast:

```text
A = popularity + SASRec + metadata              Recall@10 = 0.01435
B = A + Two-Tower                               Recall@10 = 0.01900   (+32.4% relative)
C = SASRec + metadata                           Recall@10 = 0.01620
D = C + Two-Tower                               Recall@10 = 0.02075   (+28.1% relative)
```

Those are **aggregates with no paired uncertainty**, and
[`docs/reports/m2-paired-uncertainty.txt`](reports/m2-paired-uncertainty.txt) §8 (EVIDENCE GAPS)
records explicitly that *"fusion vs single-source were not audited"*. The per-user evidence behind
B−A and D−C was never persisted: `runs/twotower_fusion_controls/controls.json` (8 KB) holds metrics
only, and the depth-1000 Two-Tower head the Step-2.8 run materialised (recorded in
`runs/step28-fusion-results.json` gate F3) was not written to disk.

B1 closes exactly that gap. It is an **evidence-completion** milestone, not a new modelling claim:
no checkpoint is trained, no head is recomputed except the one that was never saved, and no fusion
rule is touched.

---

## 1. Frozen dataset, protocol and cohort

| field | frozen value | how it is bound |
|---|---|---|
| dataset | Amazon Reviews 2023 — *Sports & Outdoors* | processed artifacts below |
| mappings | `data/processed/Sports_and_Outdoors_mappings.json` | SHA256 `dca7815a2482ab200f2cf1e3dd62eec9035bf6c5386e76cce920325360f01912` |
| sequences | `data/processed/Sports_and_Outdoors_sequences.json` | SHA256 `d3d834260c867ff39ac1b32bb08fd17ea1dc394e9a6fb7af8d19aa046c1eab81` |
| products | `data/processed/Sports_and_Outdoors_products.jsonl` | SHA256 `175c83caa3523704319f5a5adf1531f0d839d26ba0604be646c9dfdfb21e71dd` |
| protocol | `agentrecx.eval_protocol.v1` · `temporal_leave_two_out` | `recommendation/evaluation/split.py` SHA256 `054211a685c8653237fa7a6602832de07607bf2345930cffa25fcea9af20ef08` |
| evaluator | `recommendation.evaluation.batched.evaluate_batched` | `batched.py` SHA256 `1721ae7bf60ef3dfb43f8d0f96b49cb3562865ac4eca9e178d8bb7b277832f11` |
| metrics | `recommendation/evaluation/metrics.py` | SHA256 `5f84e87dff75f159218cae3e270ac331f12c08237472d229766a0c4a9db6ae72` |
| benchmark version | `1` (`BENCHMARK_VERSION`) | `experiments/benchmark_public.py` |
| cohort size | **20,000** | `DEFAULT_COHORT_SIZE` |
| cohort seed | **20260201** | `COHORT_SEED`, used by `cohort_from_cases(size=20000, seed=20260201)` → `stratified_sample` |
| **cohort identity anchor** | `target_ids` SHA256 **`99dbaf7396913303fa1497c908a95e61239a29ad14a6a8f4e51d0b37fa8e99d4`** | see below |
| catalogue | **156,746** items, full-catalogue ranking | `num_items`, `ranking = full_catalogue` |
| identity | `parent_asin` ↔ `item_id`; `PAD = 0` | `PAD` is never a candidate and never appears in a head |
| K values | **`K_VALUES = (5, 10, 20)`**, primary **K = 10** | `K_VALUES` |
| batch size | 256 | `DEFAULT_BATCH_SIZE` |

**Cohort identity is verified, not assumed.** The same `target_ids` array (SHA256
`99dbaf73…`) is byte-identical in all of:

```text
runs/m3_evidence/phase5_fixed_fusion_hits.npz   target_ids
runs/m3_evidence/phase5_popularity_hits.npz     target_ids
runs/m3_execution/paired_inputs.npz             target_ids
```

and that same digest is the one `docs/M4_EXECUTION_MANIFEST.md` records as its frozen cohort anchor.
The heads therefore belong to the same evaluated cohort as the historical comparison — established
by identity of the target array, not by filename.

### Evaluator ordering and masking (frozen semantics)

```text
ordering      higher score ranks first
tie-break     equal scores -> LOWER item_id ranks first (numeric); rank is 1-based
target rank   1 + count(score > target_score) + count(score == target_score AND item_id < target)
masking       PAD (index 0) is never rankable; every history item is excluded EXCEPT the target
```

The evaluator is used unchanged. **No arm may be scored by truncating a raw fused head**: the
accepted pipeline applies seen-history masking before top-K, and
`M3_PREREGISTRATION.md` §8 records that an unmasked rebuild does *not* reproduce hit@K.

---

## 2. Frozen existing source heads (no recomputation)

All three come from one artifact:

```text
runs/m3_evidence/phase5_fusion_source_heads.npz
SHA256 c4c0fdd6bc1e70fd4d1ffc0387c3afbe6e0642d2d0c6c0184ee440bdef5165d4
```

| source | shape | lengths | note |
|---|---|---|---|
| `popularity` | (20000, 1000) | 1000 for every user | `popularity_all_users_identical = True` (a constant list) |
| `sequential` (SASRec) | (20000, 1000) | 1000 for every user | frozen; **not recomputed** |
| `metadata` | (20000, **496**) | min 99 · mean 456.5279 · max 496 | **ragged; used exactly as stored** |

Binding evidence that these heads are the ones behind the historical comparison:

- `runs/m3_evidence/M3_REPLAY_PROVENANCE.json` SHA256
  `891cd0999f1f7ed09dc027872981fbd4a894b8ec4433b0f5bc67dfa48bc7806c` records
  `fusion_config = {rrf_k: 60, head: 1000, sources: [popularity, sequential, metadata]}`,
  cohort `{size: 20000, seed: 20260201, num_items: 156746, k_values: [5,10,20], batch_size: 256}`,
  and the same input digests listed in §1;
- `M3_PREREGISTRATION.md` §8 records that round-tripping this heads file through
  `arm_fixed_fusion` + `evaluate_batched` reproduced `target_rank` identically for all 20,000 users,
  with max |Recall delta| = 0 against the accepted values.

**Hard rules.** The `metadata` head must be used as stored: it is **not** recomputed, **not** padded
to 1000, **not** re-seeded. The `popularity` and `sequential` heads are **not** recomputed either;
they are verified by hash and shape only.

---

## 3. The one new computation: the Two-Tower head

| field | frozen value |
|---|---|
| checkpoint | `runs/twotower_public_2026_full/checkpoint.pt` |
| checkpoint SHA256 | `0e56be01f872ed0cc46eca69ef54770b1ea7dbd3e4ef92aa6eb290b605d8b1ab` |
| engine | `recommendation.inference.twotower.TwoTowerInferenceEngine` (accepted, unchanged) |
| device / dtype | CPU, float32 (this host has no CUDA device) |
| users | the frozen 20,000 cohort of §1 |
| depth | **1000** |
| retrieval | **exact full-catalogue** scoring (no ANN, no IVF-PQ, no approximation) |
| legality | `PAD = 0` excluded; identity resolved through the canonical mappings artifact |
| prohibited | retraining, quantisation, depth change, cohort change, tie-break change |

The head is ranked by the evaluator's own ordering rule (§1), so a Two-Tower head and a SASRec head
mean the same thing to the fusion: an ordering, never a score.

Per-user Two-Tower **scores are not fused with any other source's scores.** The fusion consumes
orderings only; the §4 RRF is the repository's accepted rule and the reason is unchanged — the
sources score on incomparable scales.

---

## 4. Frozen arms and the frozen fusion rule

```text
A = Popularity + SASRec + Metadata
B = Popularity + SASRec + Metadata + Two-Tower
C = SASRec + Metadata
D = SASRec + Metadata + Two-Tower
```

| field | frozen value |
|---|---|
| rule | `recommendation.control.candidate_ledger.reciprocal_rank_fusion` |
| constant | **`RRF_K = 60`** (the accepted default; not tuned, not searched) |
| head | 1000 (`top_n`) |
| score | `sum over sources of 1 / (RRF_K + rank)` |
| identity in fusion | each item id is **stringified on the way in and parsed back on the way out** (`fused_head`) |
| **tie-break** | **by identity string, lexicographically** — i.e. the *decimal string* of the item id (`"10" < "9"`), because the rule sorts on `(-score, identity)`. This is **not** numeric order and must not be "corrected" |
| dedup | order-preserving first occurrence (`dict.fromkeys`) |
| source order | `B = (popularity, sequential, metadata, two_tower)`; `D = (sequential, metadata, two_tower)` — frozen for reproducibility even though an RRF sum is order-independent |
| implementation | `experiments/benchmark_public.py::arm_fixed_fusion(sources=...)` — the accepted arm, called with an explicit source list; **no fusion code is re-implemented for B1** |

Prohibited: adding or dropping a source, weighting, calibration, learned fusion, changing
`RRF_K`, changing the depth, or changing the rule after seeing any result.

---

## 5. Frozen paired statistics

Reused verbatim from `M3_PREREGISTRATION.md` §5 (itself the Step-2.8b protocol):

| item | frozen value |
|---|---|
| unit | **user** (one paired observation per evaluated user, aligned by cohort position) |
| resampling | paired user-level **percentile bootstrap**, users resampled **with replacement** |
| resamples | **10,000** (`RESAMPLES`) |
| seed | **20260201** (`SEED`) |
| interval | **95%**, reported as the 2.5th / 97.5th percentiles |
| reported | observed paired mean difference **and** the interval |
| McNemar | **exact paired McNemar, no continuity correction**: `two-sided p = min(1, 2·Σ_{i≤min(b,c)} C(n,i) / 2^n)` |
| McNemar binary outcome | **`hit@K`, per user** — `1.0` iff the target's rank ≤ K. Under this single-positive protocol `Recall@K ≡ hit@K ≡ HR@K`, and that equivalence is documented, not a bug |
| paired metrics | `ΔNDCG@K`, `ΔRecall@K` (= `Δhit@K`), plus the McNemar 2×2 table (`b` = A-hit only, `c` = B-hit only) |

**Primary endpoint.** The primary contrast is **B − A at K = 10**, on **paired ΔNDCG@10**.

**Secondary endpoints.** D − C at all K (the popularity-independent replication that
`docs/TWOTOWER.md` §7 uses to argue the effect belongs to Two-Tower); B − A and D − C at K = 5 and
K = 20; per-user sign stability across K.

**Multiplicity.** None is applied: one primary contrast is predeclared and everything else is
supporting. This is a documented limitation of B1, matching how M2/M3 handled their audited claims;
it is stated here so a reader does not mistake a secondary interval for a corrected one.

---

## 6. Verdict rules, declared before any result exists

Two independent verdicts. **They are not the same thing.**

### 6.1 EXECUTION PASS

`EXECUTION PASS` means the protocol was executed completely and the statistics were produced:

```text
all gates of §8 pass
AND the Two-Tower head was produced for all 20,000 cohort users
AND all four arms were scored through the shared evaluator
AND every artifact of §7 was written and hashed
```

`EXECUTION PASS` is **independent of direction**. A negative result can be, and normally will be,
an `EXECUTION PASS`.

### 6.2 Classification of a contrast

Identical to M2's and M3's predeclared rule, so the milestones stay comparable:

| class | condition |
|---|---|
| `SUPPORTED` | at **every** K: McNemar p < 0.05 **and** both the ΔRecall@K and ΔNDCG@K bootstrap CIs exclude 0 |
| `DIRECTIONAL ONLY` | point estimates share one sign across K, but the `SUPPORTED` conditions are not all met |
| `NO DETECTABLE DIFFERENCE` | point estimates do not share a sign across K |
| `EVIDENCE MISSING` | no usable per-user evidence |

Wording rule: any CI that crosses 0 forces `DIRECTIONAL ONLY`; no deterministic-superiority language
is permitted for such a result. `SUPPORTED` is **not** a claim of practical superiority — only that
the difference is distinguishable from 0 at this cohort size.

### 6.3 PROMOTION SUPPORT

```text
PROMOTION SUPPORT = YES  iff
    the primary contrast (B - A at K = 10) is classified SUPPORTED
    AND its ΔNDCG@10 point estimate is POSITIVE (adding Two-Tower helps)
    AND the replication contrast (D - C at K = 10) has the SAME SIGN as the primary
otherwise
PROMOTION SUPPORT = NO
```

Consequences, fixed now:

- `EXECUTION PASS` with `PROMOTION SUPPORT = NO` is a **valid and expected outcome**. It is recorded
  as such, and the follow-up P1′ audit stays **NO-GO** for Serving Fusion V1.
- `PROMOTION SUPPORT = YES` is necessary but **not sufficient** for promotion: P1′ must also be
  re-run, and the serving-candidate audit (latency, artifact distribution, fusion semantics as a
  serving contract, reranking over a fused candidate set) remains outstanding.
- **The verdict rule is not modifiable after seeing a result.** Any change is a preregistration
  amendment, recorded and hashed *before* execution.

---

## 7. Output artifacts (all per-user evidence is persisted)

Aggregate-only storage is the exact failure this milestone exists to fix, so the per-user arrays are
the deliverable, not a by-product.

| artifact | contents |
|---|---|
| `runs/b1_evidence/two_tower_head.npz` | `two_tower` (20000, 1000) int32 + `two_tower_lengths` |
| `runs/b1_evidence/arm_A_hits.npz` | `case_index`, `target_ids`, `target_rank`, `hit5/hit10/hit20`, `top10` |
| `runs/b1_evidence/arm_B_hits.npz` | same schema |
| `runs/b1_evidence/arm_C_hits.npz` | same schema |
| `runs/b1_evidence/arm_D_hits.npz` | same schema |
| `runs/b1_evidence/paired_inputs.npz` | per-user `target_rank` and `hit@K` for both contrasts, aligned by cohort position |
| `runs/b1_evidence/paired_bootstrap.npz` | per-resample paired deltas (or the resample index matrix) so the CI is recomputable |
| `runs/b1_evidence/mcnemar.json` | 2×2 tables (`b`, `c`, `n`) and p per K per contrast |
| `runs/b1_evidence/B1_EXECUTION_MANIFEST.json` | every input hash, the frozen config, and the SHA256 of every output above |
| `docs/reports/b1-fusion-paired-evidence.txt` | the frozen report, hash-recorded like the M3/M4 reports |

**Persistence policy (amendment A1, recorded before execution).** Every B1 output must be written to
disk and hashed; nothing is optional. Where it lives afterwards differs by size:

| class | examples | Git policy |
|---|---|---|
| small text / JSON evidence | `B1_EXECUTION_MANIFEST.json`, `mcnemar.json`, the paired-statistics JSON, the final report | **committed to Git** |
| large binary evidence | the `.npz` files: Two-Tower head, per-arm hits, paired inputs, bootstrap draws | **not required to enter normal Git**; must exist on disk, and the manifest must record `path`, `size_bytes` and `SHA256` for each |

Rules that hold either way:

- every output is hashed, and `B1_EXECUTION_MANIFEST.json` records those hashes;
- **existing B1 evidence is never overwritten or deleted** — a re-run would be a new milestone, not
  an edit of this one;
- a large binary that does not suit normal Git is marked `durable_storage_pending: true` in the
  manifest, to be resolved together with the B0 durable-storage work **before P1′** — not as part of
  B1;
- **storage status cannot influence the statistical verdict.** `PROMOTION SUPPORT = NO` stays
  `PROMOTION SUPPORT = NO`; it is never re-worded as a failure, and no verdict field is conditional
  on where the bytes live.

This amendment is recorded **before execution** and changes no part of §1–§6: cohort, checkpoint,
source heads, arms, depth, RRF rule, evaluator, bootstrap, McNemar and the promotion criterion are
untouched.

---

## 8. Instrument gates (run before the statistics; a failure stops the run)

| gate | check | on failure |
|---|---|---|
| **G1** | arm **A** reproduces `runs/m3_evidence/phase5_fixed_fusion_hits.npz` **exactly** (`target_rank` identical for all 20,000 users, max \|Recall delta\| = 0) | EXECUTION FAIL — the instrument does not agree with the frozen pipeline |
| **G2** | every SHA256 in §1–§3 matches on disk | EXECUTION FAIL |
| **G3** | the rebuilt cohort's `target_ids` SHA256 == `99dbaf73…` | EXECUTION FAIL |
| **G4** | `two_tower` head shape (20000, 1000); no `PAD`; no empty row | EXECUTION FAIL |
| **G5** | evaluator files match the hashes in §1 | EXECUTION FAIL |
| **G6** | the metadata head was **not** recomputed (only read from the frozen file) | EXECUTION FAIL |

G1 is the strongest available instrument check: arms A and C share no new computation with the
historical run, so if A does not reproduce the frozen hits byte-for-byte, the difference is in the
instrument, and no number produced afterwards would be comparable.

---

## 9. Stop rule, failure policy, anti-tuning

- **One execution.** The protocol runs once over the frozen 20,000-user cohort. There is no re-run,
  no re-tune, and no second pass after a result is seen.
- **Failure handling.** Per the frozen M4 precedent, a user whose arm produces an empty or failed
  ranking **stays in the primary analysis** and is scored on the ranking it actually produced.
  Excluding users is not permitted.
- **Partial cohort is not permitted.** If the head cannot be produced for all 20,000 users, that is
  `EXECUTION FAIL`; it is reported as such and no subset is analysed.
- **Determinism failure.** If the Two-Tower head is not reproducible (e.g. checkpoint digest drift,
  non-finite scores, device-dependent results), stop and report — do not substitute a version.
- **Amendments.** Anything that changes §1–§6 after this document's freeze hash is recorded is an
  amendment, written and hashed before execution. An amendment made after a result has been seen
  invalidates B1.
- **No scope creep.** Prohibited in B1: retraining Two-Tower, recomputing the metadata/SASRec/
  popularity heads, weighted or calibrated fusion, ANN/IVF-PQ, TIGER tuning, any serving change,
  any B0 upload, and P2.

---

## 10. Freeze record

- **Amendment A1** (artifact persistence policy, §7) is the only amendment to this protocol. It was
  recorded **before execution**, on the owner's instruction, and changes no part of §1–§6: cohort,
  checkpoint, source heads, arms, depth, RRF rule, evaluator, bootstrap, McNemar and the promotion
  criterion are exactly as originally written.
- This document is frozen by its own SHA256, computed **after** the amendment and recorded in
  `runs/b1_evidence/B1_EXECUTION_MANIFEST.json` — not inside this file. A document cannot record the
  hash of the commit that contains it; this is the same rule M4 applied to its seal commit.
- The freeze manifest is written **before** the run with `status: FROZEN_NOT_STARTED`, and the run
  completes it in place. The runner refuses to start if that file already records a completed run,
  so a second execution cannot overwrite this milestone's record.
- Cost, recorded for planning only and not part of the protocol: the M3 replay of popularity +
  metadata took `wall_seconds = 1006.2` (`M3_REPLAY_PROVENANCE.json`); the frozen arms evaluate in
  ~32 s each at 79 batches. The new work is one full-catalogue scoring pass over 20,000 users
  (measured in P1 at 0.184 s per single-user request, batch size 1, on CPU), which batching reduces.
