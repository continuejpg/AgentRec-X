# Two-Tower retrieval baseline (post-Phase-5 model expansion)

A standard dual-encoder (DSSM-style) retrieval model, added as a new benchmark arm and an
optional candidate source. The question it answers:

> Does Two-Tower provide useful and complementary retrieval signal beyond the existing SASRec
> baseline?

It is **not** a replacement for SASRec, and it adds no agent, planner or ranking architecture. It
is a controlled discriminative-retrieval expansion measured inside the existing Phase-5
benchmark.

---

## 1. Why a second model family

Every arm measured before this point was either the accepted SASRec, a lexical retriever, a
popularity prior, or a fusion of those. Adding a second *learned* retrieval family is the
cheapest way to test whether the sequential model's signal is exhaustive. Two-Tower is the
standard industrial retrieval architecture for that test: item vectors are a pure function of
static item features, so the whole catalogue can be scored by one matrix multiply, and the
training objective (in-batch sampled softmax) is a different statistical task from SASRec's
per-position binary logistic loss.

---

## 2. Architecture

```
history (left-padded item ids)
   → item embedding
   → GRU (or masked mean, pooling="mean")
   → hidden state at the LAST REAL position
   → Linear → L2 normalise            ══ user vector (128)

item id (+ 4 categorical metadata fields)
   → embeddings                       ┐
   → concat → Linear → L2 normalise   ══ item vector (128)

score(user, item) = temperature · <user, item>
```

| Choice | Value | Why |
|---|---|---|
| User encoder | 1-layer GRU, hidden 128 | Order-aware but far smaller than SASRec; a mean pool is available and is *not* the default, because throwing order away would confound "two-tower is weak" with "order was ignored" |
| Item encoder | id embedding (128) + 4 categorical embeddings, concatenated and projected | Keeps the item tower a pure function of static features so item vectors are independently precomputable |
| Shared space | 128-d, L2-normalised | Cosine similarity; the retrieval contract is a dot product |
| Temperature | learned `log_temperature`, clamped to `exp([-2, 4])` | Sharpness is a loss-space parameter; ranking is invariant to it and a test pins that |
| Metadata | `main_category`, `store`, `details:Color`, `details:Material`, floor of 20 occurrences | The fields the normalised catalogue carries cleanly. Free text is deliberately excluded — the repository already has a lexical text retriever as its own arm |

**Metadata coverage** on the public catalogue (share of records with a value above the floor):
`main_category` 0.905, `store` 0.464, `Color` 0.483, `Material` 0.518. Cardinalities 22 / 1235 /
256 / 256.

Item vectors are precomputable independently: `TwoTower.precompute_item_embeddings()` returns
`[num_items + 1, 128]` in catalogue order, deterministically, and serving is a matrix multiply.

---

## 3. Training objective and negative sampling

**In-batch sampled softmax.**

```
logits[u, i] = temperature · <user_u, item_i>  −  log q(item_i)
loss         = mean_u  −log softmax_i( logits[u, :] )[i]
```

* **Negatives are the batch's other positives** — free, and they scale with batch size (2047 per
  row at batch 2048). Batches are made **duplicate-target-free** by the iterator, because a
  repeated target would make row `r`'s positive another row's false negative; repeated pairs are
  skipped, not mislabelled.
* **logQ correction.** In-batch negatives are drawn from a popularity-skewed distribution, so
  the naive estimator is biased toward popular items. Subtracting `log q(i)`, with `q` the
  smoothed empirical training-target frequency, is the standard unbiased correction. It is a
  property of the data, not a tuned parameter.
* **Single-tower supervision.** Only the batch's own positives carry gradients on the item side.
  No second full-catalogue forward pass is made, so training cost is independent of catalogue
  size.

| Configuration | Value |
|---|---|
| Pairs | 236 730 (from 20 000 users' `train_history`) |
| Window | `max_seq_len` 50, left-padded |
| Batch size | 2048 |
| Epochs | 3 (fixed, never selected on an evaluation split) |
| Optimiser | AdamW, lr 0.01, weight decay 0.0, grad-norm clip 5.0 |
| Seed | 2026 (initialisation, batch order, subsample cap) |
| Device | CPU, 24 cores, ~212 s wall |

**Leakage discipline.** Training pairs are built from `EvaluationCase.train_history` **only**.
`validation_target` and `test_target` are never read by any part of the pipeline; there is no
early stopping on validation, because selecting an epoch on an evaluation split is exactly what
the accepted protocol forbids. `run.json` records `labels_from = "EvaluationCase.train_history
only"`.

**Training trace** (loss / in-batch accuracy, chance ≈ 1/2048 = 0.05%):

| Epoch | Loss | In-batch accuracy | Temperature |
|---|---|---|---|
| 0 | 7.572 | 0.0009 | 2.93 |
| 1 | 6.882 | 0.0051 | 7.06 |
| 2 | 6.046 | 0.0238 | 10.77 |

---

## 3.1 Training-exposure parity with SASRec (Step 1.1)

The Step-1 model was trained on the 20 000-user **evaluation** cohort while the accepted SASRec
was trained on every eligible user. Nothing in a metrics table reveals that, so it is audited and
corrected here.

### What produced `runs/sasrec_canonical_2026/best.pt`

| Quantity | Value | Source |
|---|---|---|
| Users | **412 445** (all eligible; 0 with zero transitions) | `run.json` `trainable_users`, and recomputed with the accepted builder |
| Raw next-item transitions | **2 263 252** | `run.json` `train_transitions` |
| Effective transitions (50-item window) | 2 223 283 | recomputed |
| Training interactions in window | 2 635 728 | recomputed |
| Item coverage | 154 781 distinct target items of 156 746 | recomputed |
| Split source | `temporal_leave_two_out`, `agentrecx.eval_protocol.v1` | `run.json` |
| Validation/test targets excluded | **yes** — `build_dataset` reads `case.train_history` only | `recommendation/datasets/sasrec.py` |
| Epochs / batch / seed | 17 completed (patience exhausted) / 256 / 2026 | `run.json` |

### The mismatch

| | SASRec (accepted) | Two-Tower (Step 1) |
|---|---|---|
| Training users | 412 445 | **20 000** |
| Training units | 2 223 283 effective transitions | **236 730 pairs** |
| Item coverage | 154 781 | **86 002** |
| **Exposure ratio** | | **9.4×** |

The *rules* matched — both builders read `train_history` only, and both exclude single-item
histories — but the **cohorts did not**. The exact relation between the two builders is

```
two_tower_pairs == sasrec_raw_transitions − sasrec_trainable_users
```

which holds at full scale (2 263 252 − 412 445 = 1 850 807) and is asserted by
`tests/test_training_parity.py`. So the gap was entirely a cohort restriction, not a different
training rule.

### Correction

Two-Tower was **retrained on the same eligible train-only corpus** — all 412 445 users'
`train_history`, with the architecture, loss, negative-sampling design, seed (2026) and every
other hyper-parameter unchanged. The evaluation cohort stays frozen at the same 20 000 users with
`COHORT_SEED = 20260201`.

| | Step 1 | Step 1.1 (corrected) |
|---|---|---|
| Training users | 20 000 | **412 445** |
| Pairs | 236 730 | **1 850 807** |
| Distinct target items | 86 002 | **154 781** |
| Epochs × batch | 3 × 2048 | 8 × 2048 |
| Steps | 336 | 6 952 |
| Wall (CPU) | 212 s | 3 982 s |
| In-batch accuracy (final) | 0.0238 | **0.1055** (chance 1/2048 = 0.00049) |
| Artifact | `runs/twotower_public_2026` | `runs/twotower_public_2026_full` |

The Step-1 checkpoint is preserved unchanged and is reported as a **historical** result. Both
checkpoints remain on disk.

### Residual, unavoidable differences

Even at matched exposure the two models do not see identical tensors, and the difference is
architectural rather than a confound:

* SASRec's sample packs the whole window into one sequence and scores every position with a binary
  logistic loss (5.49 transitions per user on average); Two-Tower emits one pair per transition
  and scores them with an in-batch softmax over 2048 candidates. That is the point of comparing
  two model families.
* With a short window SASRec's *effective* transition count drops (transitions outside the window
  are lost), while Two-Tower still holds one pair per transition. At the canonical
  `max_seq_len = 50` the loss is small (2 263 252 → 2 223 283, 1.8%) and is recorded above.

---

## 4. Dataset and split reuse

Nothing about the benchmark protocol changed. Two-Tower is trained and evaluated through the
accepted loader on:

* **Amazon Reviews 2023**, `Sports_and_Outdoors`;
* the accepted `parent_asin` ↔ integer-id mapping;
* `temporal_leave_two_out` (`agentrecx.eval_protocol.v1`);
* the **same deterministic 20 000-user cohort** (`COHORT_SEED = 20260201`) the Phase-5 arms use;
* the full catalogue (156 746 items) and `k ∈ {5, 10, 20}`;
* the accepted evaluator, which owns PAD exclusion, seen-item masking, tie-breaking and ranking.
  Both model arms are scored from raw score matrices and neither masks.

Reproduce:

```bash
python -m experiments.train_twotower --out runs/twotower_public_2026 \
    --cohort 20000 --epochs 3 --batch-size 2048 --seed 2026
python -m experiments.benchmark_twotower --out runs/twotower_benchmark --cohort 20000
```

The comparison command **re-runs** the accepted arms rather than quoting stored numbers, and
records a cross-check against the Phase-5 artifact. In the measured run every re-run matched the
stored value exactly (`sequential` 0.01350, `metadata_retrieval` 0.00925, `fixed_fusion` 0.01435).

---

## 5. Standalone metrics (20 000 users, full-catalogue)

Two checkpoints exist. The **Step-1.1 (corrected)** one is the current model; the Step-1 one is
preserved as a historical result and is what the exposure audit above replaces.

| Arm | Recall@5 | **Recall@10** | Recall@20 | NDCG@10 | NDCG@20 | HR@10 | mean rank |
|---|---|---|---|---|---|---|---|
| `metadata_retrieval` | 0.00505 | 0.00925 | 0.01485 | 0.00441 | 0.00581 | 0.00925 | 93 379 |
| `sequential` (SASRec, accepted) | 0.00820 | 0.01350 | 0.02070 | 0.00685 | 0.00865 | 0.01350 | 20 598 |
| **`two_tower`** (Step 1.1, corrected) | 0.00860 | **0.01435** | 0.02285 | 0.00737 | 0.00951 | 0.01435 | 21 944 |
| `fixed_fusion` (Phase-5) | 0.00860 | 0.01435 | 0.02325 | 0.00743 | 0.00968 | 0.01435 | 80 590 |
| *`two_tower` (Step 1, 20k exposure — historical)* | *0.00550* | *0.00965* | *0.01650* | *0.00482* | *—* | *0.00965* | *36 319* |

**Conclusion change.** At matched training exposure, Two-Tower is **no longer a weaker single
source**: its Recall@10 rises from 0.00965 to **0.01435**, level with the accepted Phase-5 fusion
and above SASRec's 0.01350. The Step-1 statement "Two-Tower is a real but weaker single source than
SASRec" was an artifact of the 9.4× exposure gap, not of the architecture.

Neither model is converged in any strong sense — Two-Tower's in-batch accuracy at the end of
training was 10.6% against 0.05% chance, and SASRec early-stopped on patience — so these are
matched-exposure baselines, not tuned endpoints.

---

## 6. Complementarity — the question that matters

Standalone Recall is not the point. Per-user hit analysis over each source's top-1000 head
(**Step 1.1, corrected exposure**):

| Pair | both | left only | right only | neither | union | Jaccard | lift |
|---|---|---|---|---|---|---|---|
| SASRec vs **Two-Tower** | 2 559 | 1 541 | **1 872** | 14 028 | **5 972** | 0.429 | **1.348** |
| SASRec vs metadata | 407 | 3 693 | 1 227 | 14 673 | 5 327 | 0.076 | 1.299 |
| Two-Tower vs metadata | 581 | 3 850 | 1 053 | 14 516 | 5 484 | 0.106 | 1.238 |

Head-level overlap stays **low**: mean Jaccard between the SASRec and Two-Tower top-10 heads is
**0.083**, and **0.103** at top-1. Correcting exposure made the two models *more* different at the
head, not less.

| | Step 1 (20k exposure) | Step 1.1 (full exposure) |
|---|---|---|
| Two-Tower head coverage | 0.1558 | **0.2216** |
| SASRec head coverage | 0.2050 | 0.2050 |
| **Two-Tower-only hits** | 1 120 | **1 872** |
| union coverage | 0.2610 | **0.2986** |
| lift over best single source | 1.273 | **1.348** |
| head Jaccard @10 / @1 | 0.101 / 0.118 | **0.083 / 0.103** |

**Training exposure materially changed the complementarity conclusion**, in both directions at
once: the corrected model retrieves *more* on its own, and it still misses a large set SASRec
covers, so the two sources remain genuinely complementary. 1 872 users (0.0936 of the cohort) have
a target that Two-Tower's head contains and SASRec's does not.

Caveat on scale: "misses" means *not in the top-1000 head* — a statement about retrieval, not about
ranking. The fusion controls in §7 are the end-to-end test of whether this converts into metric.

---

## 7. Fusion controls — the incremental value of Two-Tower

The Step-1 comparison was **not a controlled experiment**. It compared
``SASRec + Two-Tower + metadata`` against the Phase-5 ``popularity + SASRec + metadata``, which
changed two things at once — it added Two-Tower *and* removed popularity — so its +12.2% could not
be attributed to Two-Tower. That artifact is preserved unchanged and is no longer used for
attribution.

Four frozen source sets now isolate one variable. The RRF rule, its constant, the head size
(1000), the cohort, the catalogue, the `k` values and the evaluator are identical everywhere; the
only difference between a contrast's two arms is Two-Tower's presence.

| | Sources |
|---|---|
| **A** | popularity + SASRec + metadata *(the accepted Phase-5 fusion)* |
| **B** | popularity + SASRec + metadata + **Two-Tower** |
| **C** | SASRec + metadata |
| **D** | SASRec + metadata + **Two-Tower** |

| Arm | Recall@5 | **Recall@10** | Recall@20 | NDCG@10 | HR@10 |
|---|---|---|---|---|---|
| A (Phase-5 fixed fusion) | 0.00860 | 0.01435 | 0.02325 | 0.00743 | 0.01435 |
| **B (A + Two-Tower)** | **0.01125** | **0.01900** | **0.02995** | **0.00982** | **0.01900** |
| C (SASRec + metadata) | 0.01015 | 0.01620 | 0.02525 | 0.00887 | 0.01620 |
| **D (C + Two-Tower)** | **0.01280** | **0.02075** | **0.03295** | **0.01105** | **0.02075** |

### Controlled contrasts

Every previous source is held constant; the only change is adding Two-Tower. No source is removed
and none is substituted.

| Contrast | Held constant | Added | Recall@10 | Δ absolute | Δ relative | Δ Recall@20 | Δ NDCG@10 |
|---|---|---|---|---|---|---|---|
| **B − A** | popularity, SASRec, metadata | **two_tower** | 0.01435 → 0.01900 | **+0.00465** | **+32.4%** | +28.8% | +32.0% |
| **D − C** | SASRec, metadata | **two_tower** | 0.01620 → 0.02075 | **+0.00455** | **+28.1%** | +30.5% | +24.5% |

Both contrasts agree in magnitude (+28–32% relative Recall@10, +24–32% NDCG@10), which is the
expected pattern: the incremental value of Two-Tower is similar whether or not popularity is in the
fused set. **That agreement is the evidence that the effect belongs to Two-Tower rather than to the
source set around it.** "Adding Two-Tower" is therefore a claim this repository can now make,
because both comparisons hold every other source constant.

### Note on popularity

Popularity is retained in A and B deliberately. It is a weak *ranker* (Recall@10 0.00525) but a
non-trivial *retriever* (target in head for 10.7% of users), and dropping it while adding Two-Tower
is precisely the confound this step corrects. C and D exist to show the effect without it.

The accepted Phase-5 numbers are unchanged and are re-checked in every run: in the corrected run
the re-run `sequential`, `metadata_retrieval` and `fixed_fusion` arms reproduced their stored
Recall@10 values exactly (0.01350 / 0.00925 / 0.01435).

---

## 8. Candidate-source integration

Two-Tower is exposed to the agent runtime through the **existing** candidate-source abstraction,
with no change to candidate identity authority:

* `CandidateSource.TWO_TOWER` is a new enum member, and
  `CANDIDATE_PRODUCING_SOURCES` lists it. The ledger refuses any source not on that list, so the
  allow-list remains the enforcement point.
* `TwoTowerSourceTool` implements `CandidateSourceTool`: it returns
  `(parent_asin, rank, score)` triples and **never grounds an identity** — the plane's
  `GroundingVerifier` confirms each product before the ledger records it, and an unknown identity
  is recorded as an UNGROUNDED audit entry rather than dropped.
* `score_kind` is `two_tower_cosine`, recorded on every ledger provenance record. A Two-Tower
  cosine is never compared against a SASRec logit or a BM25 value; the only cross-source
  combination in the repository is rank-based fusion, which uses ordering alone.
* Registration is opt-in (`AGENTRECX_TWO_TOWER_CHECKPOINT`, plus
  `AGENTRECX_TWO_TOWER_MAPPINGS_PATH` when the model's catalogue differs from the served one),
  and the source is only offered to a policy when a tool is registered.
* **Reachability, stated precisely.** The plane grounds and records Two-Tower candidates, and
  `TwoTowerSourceTool` is exercised against the real registry in `tests/test_twotower_source.py`.
  However the controller currently offers `SELECT_SOURCE` - the action that consults a source by
  name - only when a similar-item tool is registered; that condition predates this step, and
  changing it was measured to alter the action sequence of an archived Phase-3 recording, so it
  was deliberately left alone (the same finding is recorded in `docs/PHASE5_HANDOFF.md` §6).
  Consequence: with Two-Tower registered but no similar-item tool, the source is *available* and
  *grounded* but not yet *offered* to the demo policy. Wiring it into a selectable action is a
  control-plane change and therefore out of scope here; the benchmark arm, which is what this
  step was asked to measure, is unaffected.
* **No policy change.** The demo's request-driven plan may name `two_tower`; whether an LLM
  should choose it is a separate, unmeasured question that this step does not answer.

---

## 9. Known limitations

1. **Two-Tower is still not converged.** Eight epochs on 1 850 807 pairs, in-batch accuracy 10.6%
   against 0.05% chance, and the loss was still falling. Its standalone Recall is now level with
   the accepted Phase-5 fusion, but a longer run could move both it and the marginal value of the
   source. *This limitation replaced the Step-1 wording, which said the model was under-trained on
   a 9.4× smaller corpus; that was true and is now corrected.*
2. **The item tower sees only `train_history` targets** — 154 781 of 156 746 items appear as a
   training label at least once after the exposure correction (86 002 before it), so coverage is
   now near-total but the per-item signal is still extremely thin: 1 850 807 pairs over 154 781
   targets is ~12 examples per item on average, with a long tail at one.
3. **The user tower is trained on prefixes, not the full test history.** Training pairs are
   prefix→next pairs, while evaluation encodes the entire test history; the distribution shift is
   the standard two-tower compromise and is not corrected here.
4. **Metadata is categorical only.** Colour and material are present for roughly half the
   catalogue (0.48 / 0.52), and "unknown" is a single shared category, so a missing attribute is
   indistinguishable from a genuinely absent one.
5. **`logQ` uses the training-target frequency as the sampling prior.** That is the standard
   estimator's assumption (negatives drawn like positives), not a measurement of the true
   sampling distribution.
6. **Exact dot-product scoring only.** No ANN index, no negative sampling redesign, no hard
   negatives — all explicitly out of scope for this step.
7. **One seed.** Run-to-run determinism is verified, but a multi-seed variance estimate was not
   produced, so differences of a few percent between arms should not be over-read.
8. **The fusion gain is not attributable to Two-Tower alone.** The new fusion also drops
   popularity, which Phase-5 measured as near-chance as a ranker; the experiment isolates a
   *source set*, not a single variable.
