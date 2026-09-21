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

| Arm | Recall@5 | **Recall@10** | Recall@20 | NDCG@10 | HR@10 | mean rank |
|---|---|---|---|---|---|---|
| `metadata_retrieval` | 0.00505 | 0.00925 | 0.01485 | 0.00441 | 0.00925 | 93 379 |
| **`two_tower`** | 0.00550 | **0.00965** | 0.01650 | 0.00482 | 0.00965 | 36 319 |
| `sequential` (SASRec) | 0.00820 | 0.01350 | 0.02070 | 0.00685 | 0.01350 | 20 598 |
| `fixed_fusion` (Phase-5) | 0.00860 | 0.01435 | 0.02325 | 0.00743 | 0.01435 | 80 590 |
| **`sasrec_two_tower_metadata`** (new) | **0.00940** | **0.01610** | **0.02695** | **0.00846** | **0.01610** | 78 753 |

**Two-Tower is a real but weaker single source than SASRec** (0.00965 vs 0.01350 Recall@10) and
beats the lexical metadata retriever. Trained for 3 epochs on 236 730 pairs, it is plainly not
converged; see limitations.

---

## 6. Complementarity — the question that matters

Standalone Recall is not the point. Per-user hit analysis over each source's top-1000 head:

| Pair | both | left only | right only | neither | union | Jaccard |
|---|---|---|---|---|---|---|
| SASRec vs **Two-Tower** | 1 995 | 2 105 | **1 120** | 14 780 | **5 220** | 0.382 |
| SASRec vs metadata | 407 | 3 693 | 1 227 | 14 673 | 5 327 | 0.076 |
| Two-Tower vs metadata | 356 | 2 759 | 1 278 | 15 607 | 4 393 | 0.081 |

Head-level overlap is **low**: mean Jaccard between the SASRec and Two-Tower top-10 heads is
**0.101**, and **0.118** at top-1. The two models retrieve substantially different candidates.

Read the SASRec-vs-Two-Tower row carefully, because it is the answer to the phase's question:

* Two-Tower's head contains the target for **3 115** users (0.15575), SASRec's for **4 100**
  (0.205);
* **1 120 of Two-Tower's hits are users SASRec misses at head-1000**;
* the union covers **5 220** users (0.261) versus SASRec's 4 100 — a **1.27× lift over the best
  single source**.

So yes: **Two-Tower contributes candidates SASRec does not retrieve.** That is the precondition
for it to be worth anything in a fusion, and it is measured rather than assumed.

Caveat on scale: "misses" here means *not in the top-1000 head*, which is a statement about
retrieval, not about ranking. SASRec may still rank a target highly that Two-Tower retrieved, and
vice versa; the fusion results below are the end-to-end test of whether the complementarity
converts into metric.

---

## 7. Fusion ablation

One controlled experiment, run **after** the standalone result was frozen. Same source set as
Phase-5 fusion plus Two-Tower in place of popularity:

| | Sources |
|---|---|
| `fixed_fusion` (accepted Phase-5) | popularity + sequential + metadata |
| `sasrec_two_tower_metadata` (new) | sequential + **two_tower** + metadata |

Both use the repository's accepted reciprocal-rank fusion with the **accepted default constant**
and the **same head size (1000)**. Nothing was tuned on the test split: the rule, the constant and
the head are codebase defaults, and the only choice made is which three sources to fuse.

| Metric | `fixed_fusion` | `sasrec_two_tower_metadata` | change |
|---|---|---|---|
| Recall@5 | 0.00860 | **0.00940** | +9.3% |
| Recall@10 | 0.01435 | **0.01610** | +12.2% |
| Recall@20 | 0.02325 | **0.02695** | +15.9% |
| NDCG@10 | 0.00743 | **0.00846** | +13.9% |
| head-1000 target coverage | 0.2066 | **0.2242** | — |

The Phase-5 arm remains in the artifact unchanged; the two are reported side by side so the
comparison is explicit rather than a silent replacement.

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

1. **Two-Tower is under-trained.** Three epochs on 236 730 pairs, in-batch accuracy 2.4% against
   0.05% chance. Its standalone Recall is ~29% below SASRec's. The complementarity and fusion
   findings are measured at this training level; a longer run could move both the standalone
   metric and the marginal value of the source.
2. **The item tower sees only `train_history` targets** — 86 002 of 156 746 items appear as a
   training label at least once, so most catalogue items are never a positive. Cold-start items
   can only be retrieved through their metadata categories.
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
