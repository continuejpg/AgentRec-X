# AgentRec-X — Model Expansion Handoff (Steps 1, 1.1, 2)

Compressed context for a fresh conversation. Source of truth: the repository, Git history, tests and
`docs/ARCHITECTURE.md`. This file states **what was measured, what it means, and what must not be
reopened**. It deliberately carries no implementation history.

Read this after [`PHASE5_HANDOFF.md`](PHASE5_HANDOFF.md). Phase 5 (the public benchmark) and its
handoff remain valid and unchanged; this document covers the three model-expansion steps layered on
top of it.

---

## 1. Current State

```
branch         master
HEAD           80f81b1  milestone23: Semantic-ID generative retrieval baseline (step 2)
parent         2123c66  milestone22: Two-Tower experimental control closure (step 1.1)
               8431ce7  milestone21: Two-Tower retrieval baseline (step 1)
               6f71a4f  milestone20: demo, Docker and CI packaging
               f3307e3  milestone19: public-data benchmark (Phase 5)
origin/master  6845e78  (20 commits behind; nothing has been pushed)
worktree       clean
offline suite  2047 passed, 33 skipped
```

| Step | Commit | What it added |
|---|---|---|
| 1 — Two-Tower retrieval baseline | `8431ce7` | RQ-free DSSM dual encoder, in-batch sampled softmax, benchmark arm, candidate source |
| 1.1 — experimental control closure | `2123c66` | training-exposure parity audit + retrain; frozen A/B/C/D fusion controls |
| 2 — Semantic-ID generative retrieval | `80f81b1` | RQ-VAE tokenizer, autoregressive generator, constrained decoding, deterministic resolver |

**Nothing here is wired into Agent policy.** Two-Tower is exposed through the existing
`CandidateSource` abstraction (opt-in, and see §7 for its reachability gap). Semantic-ID generation
is **standalone only** — deliberately not exposed as an Agent tool.

---

## 2. Frozen protocol (do not change)

Every arm in every step was scored by the **same** machinery:

| | Value |
|---|---|
| Dataset | Amazon Reviews 2023, `Sports_and_Outdoors` |
| Identity | `parent_asin` ↔ int id, PAD = 0 |
| Split | `temporal_leave_two_out`, `agentrecx.eval_protocol.v1` |
| Cohort | deterministic 20 000 users, `COHORT_SEED = 20260201` |
| Catalogue | full, 156 746 items |
| Cut-offs | `k ∈ {5, 10, 20}` |
| Evaluator | `recommendation.evaluation.batched` — owns PAD exclusion, seen-item masking, tie-breaking (higher score first, then lower item id), ranking |

The evaluator ranks; **arms supply raw scores and never mask**. Retrieval arms express their ordering
as a strictly decreasing score matrix with a documented ascending-id tail for unretrieved items.

---

## 3. Training-exposure discipline (the Step-1.1 lesson)

The accepted SASRec and the corrected Two-Tower and the generator are all trained on the **same
eligible train-only corpus**: all 412 445 users' `train_history`.

| | SASRec (accepted) | Two-Tower (corrected) | GenRec |
|---|---|---|---|
| Training users | 412 445 | 412 445 | 412 445 |
| Training units | 2 223 283 effective transitions | 1 850 807 pairs | 1 850 807 examples |
| Distinct target items | 154 781 | 154 781 | 151 505 (code space limits it) |
| Labels from | `train_history` only | `train_history` only | `train_history` only |
| Epoch selection | patience on **validation** (accepted run) | none — fixed config | none — fixed config |

The exact relation between the SASRec and Two-Tower corpora, asserted by
`tests/test_training_parity.py`:

```
two_tower_pairs == sasrec_raw_transitions − sasrec_trainable_users
2 263 252 − 412 445 = 1 850 807        ✓
```

**Step 1's Two-Tower was trained on the 20 000-user *evaluation* cohort — a 9.4× exposure gap that no
metrics table reveals.** That checkpoint and its numbers are preserved as historical results in
`runs/twotower_public_2026`; the corrected one is `runs/twotower_public_2026_full`. Never quote the
Step-1 numbers as the model's performance.

---

## 4. Measured results (frozen 20 000-user cohort, full catalogue)

### Standalone arms

| Arm | Recall@5 | **Recall@10** | Recall@20 | NDCG@10 | HR@10 |
|---|---|---|---|---|---|
| popularity (floor) | 0.00370 | 0.00525 | 0.00785 | 0.00269 | 0.00525 |
| metadata_retrieval (BM25) | 0.00505 | 0.00925 | 0.01485 | 0.00441 | 0.00925 |
| agent_selected (deterministic rule) | 0.00690 | 0.01180 | 0.01820 | 0.00584 | 0.01180 |
| sequential (SASRec) | 0.00820 | 0.01350 | 0.02070 | 0.00685 | 0.01350 |
| two_tower (corrected) | 0.00860 | **0.01435** | 0.02285 | 0.00737 | 0.01435 |
| **semantic_id_genrec** | 0.00175 | **0.00185** | 0.00295 | 0.00142 | 0.00185 |
| fixed_fusion (Phase-5) | 0.00860 | 0.01435 | 0.02325 | 0.00743 | 0.01435 |
| **SASRec + Two-Tower + metadata** | **0.01280** | **0.02075** | **0.03295** | **0.01105** | **0.02075** |
| *two_tower (Step 1, historical)* | *0.00550* | *0.00965* | *0.01650* | *0.00482* | *0.00965* |

Accepted full-cohort SASRec reference: Recall@10 **0.01357** on 412 445 users; the 20 k sample
reproduces it at 0.01350 (0.5% relative), which is the check that sampling and wiring are correct.

### Controlled fusion (the only attributable comparison)

RRF rule, its constant, head size (1000), cohort, catalogue and evaluator all frozen. Each contrast
holds **every** previous source constant and varies only Two-Tower's presence.

| Arm | Sources | Recall@10 | Recall@20 |
|---|---|---|---|
| A | popularity + SASRec + metadata | 0.01435 | 0.02325 |
| **B** | A + **Two-Tower** | **0.01900** | 0.02995 |
| C | SASRec + metadata | 0.01620 | 0.02525 |
| **D** | C + **Two-Tower** | **0.02075** | 0.03295 |

| Contrast | Δ relative Recall@10 | Δ NDCG@10 |
|---|---|---|
| **B − A** | **+32.4%** | +32.0% |
| **D − C** | **+28.1%** | +24.5% |

The two contrasts agree in magnitude, which is the evidence that the gain belongs to Two-Tower and
not to the source set around it. The Step-1 comparison (`+12.2%`) changed two things at once and is
**superseded for attribution**; its artifact is preserved unchanged.

### Complementarity (per-user hits inside each source's top-1000 head)

| Pair | both | left only | right only | union | Jaccard |
|---|---|---|---|---|---|
| SASRec vs Two-Tower | 2 559 | 1 541 | **1 872** | 0.2986 | 0.429 |
| SASRec vs metadata | 407 | 3 693 | 1 227 | 0.2664 | 0.076 |
| SASRec vs GenRec | 134 | 3 966 | 97 | 0.2099 | 0.032 |
| Two-Tower vs GenRec | 118 | 4 313 | 113 | 0.2272 | 0.026 |

Three-way partition: all three 106 · exactly two 2 493 · exactly one 3 458
(`sequential` 1 513, `two_tower` 1 860, **`semantic_id_genrec` 85**) · none 13 943 · union 6 057
(0.3029).

**Two-Tower is strongly complementary** (head Jaccard 0.083 vs SASRec, 1 872 exclusive hits).
**GenRec is the most orthogonal source measured but contributes little** (Jaccard 0.020 / 0.011,
85 exclusive hits, union 0.2986 → 0.3029).

---

## 5. Semantic-ID tokenizer and generator, stated separately

These are **two different training runs with two different objectives**, and neither is a
recommendation metric. Do not merge them in a report.

**Tokenizer** (`RQ-VAE`, trained on **catalogue features only** — no interactions, so nothing to leak):
40 epochs, reconstruction objective, loss 0.646 / reconstruction MSE 0.544 at the end.

| Audit | Value |
|---|---|
| Coverage | **1.000** (156 746 / 156 746) |
| Distinct Semantic IDs | 153 391 |
| Items in a collision group | 6 520 (**4.16%**), largest group 6 |
| Codebook utilisation (level 0/1/2) | 0.871 / 0.961 / 1.000 |
| Length / codebook size | 3 / 256 |

Coverage is total **by construction**: item features are seeded tables indexed by item id and are
**not trained**, so every item is tokenisable from its id alone and metadata sparsity cannot make an
item unretrievable.

**Generator** (autoregressive Transformer, trained on `train_history` only): 5 epochs × 3 615 steps,
batch 512, 135 425 parameters, 6-item window, per-position code cross-entropy.

| Epoch | Loss | Code accuracy |
|---|---|---|
| 0 | 5.231 | 0.0264 |
| 2 | 5.058 | 0.0523 |
| 4 | **4.982** | **0.0666** |

Chance = 1/257 ≈ 0.0039, so 6.7% is ~17× chance — real structure, but it compounds to ≈0.03% for a
full 3-code sequence, which is the same order as the observed Recall.

---

## 6. Why GenRec is weak (the honest failure analysis)

Not a tokenizer failure. Two measured limits interact:

1. **The bounded prefix search is binding.** 512 expanded prefixes per level reached **534 of
   156 746 items (0.34%)**; the evaluator still ranks the full catalogue, so 99.66% of items sat at
   the documented tail score. The mean target rank of 100 118 is that convention showing through.
2. **Per-code accuracy compounds.** 6.7%³ ≈ 0.03% for a correct 3-code sequence.

Plus: 4.16% collisions share a score between items, and the item features are untrained (random
projections + metadata categories, not behavioural similarity).

**Invalid-generation rate = 0.0000** across 20 000 generations. This is a *structural* result: it
shows the identity boundary holds, not that the generations are good.

---

## 7. Trust boundaries touched by these steps

Do not weaken any of these.

1. **Two-Tower is a normal candidate source.** `CandidateSource.TWO_TOWER` was added to the enum and
   to `CANDIDATE_PRODUCING_SOURCES` (the ledger's allow-list — an unlisted source is refused). Its
   tool returns `(parent_asin, rank, score)` and **never grounds**: the plane's `GroundingVerifier`
   confirms each identity, an unknown one is recorded as UNGROUNDED, and `score_kind` is
   `two_tower_cosine` so its scores are never compared with a SASRec logit or a BM25 value.
2. **Reachability gap, stated rather than worked around.** The controller offers `SELECT_SOURCE`
   only when a similar-item tool is registered. Changing that was measured to alter the action
   sequence of an archived Phase-3 recording, so it was left alone. Consequence: with Two-Tower
   registered but no similar-item tool, the source is *available and grounded* but not yet *offered*
   to the demo policy. Its benchmark arm is unaffected.
3. **Generation cannot create identity.** Decoding walks a prefix trie built from the **catalogue
   assignment** (never from model output); `semantic_id/resolver.py` is a separate pure function that
   maps code → item and **fails closed** on an unoccupied code — never a nearest neighbour, partial
   match or default. Collisions break by the documented `COLLISION_RULE`; an eligibility filter that
   excludes every member yields nothing; PAD is never resolvable.

---

## 8. Reproducing (all offline, no credentials)

```bash
# Two-Tower: corrected full-exposure training (~66 min CPU) then the comparison + controls
python -m experiments.train_twotower --out runs/twotower_public_2026_full \
    --cohort 412445 --epochs 8 --batch-size 2048 --seed 2026
python -m experiments.benchmark_twotower --out runs/twotower_benchmark_full --with-controls
python -m experiments.fusion_controls  --out runs/twotower_fusion_controls

# Semantic ID: tokenizer (~35 s) then generator (~32 min CPU)
python -m experiments.train_semantic_id --out runs/semantic_id_public_2026 --tokenizer-only
python -m experiments.train_semantic_id --out runs/semantic_id_public_2026 --generator-only \
    --epochs 5 --batch-size 512 --max-items 6 --d-model 64 --n-heads 4 --n-layers 2
python -m experiments.benchmark_genrec --out runs/genrec_benchmark --cohort 20000
```

`runs/` is git-ignored: every artifact above is regenerable and none is committed. Every `run.json`
records the configuration, the seed, the per-epoch diagnostics and the processed-artifact SHA-256s.

---

## 9. Known limitations

1. **Absolute quality is low.** The best arm is Recall@10 = 0.02075 on a 156 746-item catalogue with
   one positive per user. This is a controlled comparison, not evidence of a production recommender.
2. **Two-Tower is not converged** (loss still falling at epoch 8); GenRec is not converged either.
3. **GenRec's prefix-search breadth is the binding constraint** (0.34% catalogue coverage). Widening
   it is an engineering/ANN change and was explicitly out of scope.
4. **Single seed for every arm.** Run-to-run determinism is verified bit-exactly, but no multi-seed
   variance or per-user bootstrap interval was produced, so the +28% vs +32% agreement should be read
   as consistent direction, not as a significant difference.
5. **`popularity` is retained in A/B deliberately.** It is a weak ranker but a non-trivial retriever;
   dropping it while adding Two-Tower is exactly the confound Step 1.1 corrected. C/D exist to show
   the effect without it.
6. **The tokenizer's item features are untrained**, so code groups reflect random projections plus
   metadata categories rather than behavioural similarity.
7. **No fusion includes GenRec.** Adding it would be a new controlled experiment, not a follow-on.

---

## 10. Do Not Reopen

Do not redesign the evaluator, the split, the cohort or the tokenizer/generator architectures. In
particular, do not:

* retrain Two-Tower on a subset cohort again (that is the Step-1 confound returning);
* quote the Step-1 Two-Tower numbers as the model's performance;
* claim "adding Two-Tower" without holding every other source constant;
* tune thresholds, beam widths, epochs or RRF parameters against the evaluation cohort;
* merge the tokenizer's reconstruction loss with the generator's cross-entropy in one "training
  metric";
* expose GenRec as an Agent tool before its retrieval capability is shown to be usable;
* change `SELECT_SOURCE`'s offering rule without re-checking the archived Phase-3 replay tests.

**Next candidate steps, in evidence order:** (a) multi-seed / bootstrap intervals, since the current
attribution is directional only; (b) widen GenRec's prefix search or give it a generation budget —
a bounded engineering experiment; (c) re-evaluate whether a `NoProgressGuard` is now justified
(Phase-4 repetition was 2 of 8 failures, both already bounded by existing budgets); (d) only then
consider exposing generative retrieval to the Agent.

---

## 11. Step 2.1 audit outcome and the TIGER backend decision

**Step 2.1 (public-implementation audit) is complete.** Two repositories were inspected:
`mclwu22/amazon-genrec` and `snap-research/GRID`. Both were audited from source, not from their
READMEs. Neither is vendored, and the reasons are legal as well as technical: `amazon-genrec`
ships **no LICENSE file at all**, and `GRID` is **Snap Inc., non-commercial research purposes
only** with a mandatory notice-retention clause. Absorbed elements are therefore *reimplemented*
and labelled per file in `backends/tiger_public/PROVENANCE.md`.

**The decision that supersedes "widen GenRec's prefix search" above.** Custom GenRec v0 stays
frozen at `80f81b1` and is **not** upgraded with content embeddings, k-means++ initialisation or a
collision-deduplication digit. Those would have produced a "GenRec v0.5" — an unattributable
mixture of representation and search changes. Instead, the next generative-retrieval work is a
**separate public-TIGER backend** behind an explicit adapter boundary, specified in
[`TIGER_BACKEND.md`](TIGER_BACKEND.md).

| Fact that drove the decision | Where it is measured |
| --- | --- |
| GenRec v0's item features are **seeded random**, not item content | this document §6; `docs/SEMANTIC_ID.md` §1 |
| The binding constraint is the **bounded prefix search** (0.34 % catalogue coverage) | this document §6 |
| GenRec v0's dead-code counts (`33/10/0`) are 12.9 %, 3.9 %, 0 % — i.e. **not** a catastrophic collapse | `runs/semantic_id_public_2026/semantic_ids.json` |
| GenRec v0 detects collisions but does not resolve them (4.16 % of items share a score) | same artifact |

**What must not be reopened by this decision:** the frozen evaluator, split, cohort, identity
mapping and trust boundaries are unchanged, and GenRec v0's artifacts and numbers stay valid as a
historical baseline. The new backend supplies **raw scores** in the frozen arm contract and owns
nothing else; it never sees `parent_asin`, a target, or a seen-item identity.

**Milestone sequence:** 2.2 specification (done) → 2.3 backend skeleton and handoff bridge, no ML
→ 2.4 content embedding + RQ-VAE + collision-free Semantic IDs → 2.5 TIGER generator →
2.6 certified catalogue retrieval and the canonical benchmark → 2.7 Agent integration, only after
2.6 establishes value.

**One item from the list above remains open and is unaffected:** multi-seed / bootstrap intervals
(a) are still the cheapest way to make the current attribution non-directional, and remain a
candidate step independent of the TIGER backend work.
