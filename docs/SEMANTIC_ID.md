# Semantic-ID generative retrieval (model expansion step 2)

A semantic tokenizer (RQ-VAE style) plus a small autoregressive generator, added as a third
benchmark arm. The question it answers:

> Does autoregressive Semantic-ID generation provide useful and complementary recommendation
> signal beyond SASRec and Two-Tower?

It adds **no** Agent capability, no fusion change and no selector policy. Semantic-ID generation is
measured as a retrieval source first; exposing it as an Agent tool is a later, separate decision.

> **Status: complete for this baseline.** The tokenizer and generator are trained, the frozen
> 20 000-user benchmark has run, and the sections below carry the measured numbers from
> `runs/semantic_id_public_2026` and `runs/genrec_benchmark`. Nothing here rewrites the accepted
> Phase-5 numbers or the Step-1/1.1 Two-Tower results, both of which remain frozen.

---

## 1. Tokenizer design and statistics

```
item id  -> deterministic embedding (seeded, NOT trained)   ┐
metadata -> deterministic per-category embeddings           ├─> encoder MLP
             (main_category, store, Color, Material)        ┘        |
                                                                    v
                                          residual quantisation over L codebooks of size K
                                                                    |
                                                                    v
                                                  Semantic ID = (c1, c2, ..., cL)
                                                                    |
                                          decoder MLP <- sum of chosen codebook vectors
```

| Choice | Value | Why |
|---|---|---|
| Shape | RQ-VAE: residual quantisation, straight-through gradients | Standard, defensible, and small enough to train on CPU; the residual hierarchy is what lets a short code express a coarse-to-fine distinction |
| Levels `L` | 3 | Three codes is the TIGER-style convention and keeps the generator's target short |
| Codebook size `K` | 256 | `K**L` = 16 777 216 outweighs the catalogue, so collisions are a modelling outcome rather than a hard cap |
| Item features | seeded id embedding (64-d) ++ 4 categorical metadata fields (16-d each) | Identity alone is enough to be **total**; metadata lets the codes group similar products |
| Trained parameters | encoder, codebooks, decoder | The feature tables are deliberately **not** trained, so the code assignment is reproducible from the seed plus the catalogue |

**Total coverage by construction.** Features come from tables indexed by item id, so every catalogue
item gets a vector and therefore a code. No item needs a text field, which is what prevents the
usual "items without metadata are unretrievable" hole.

### Measured audit

Recorded in `runs/semantic_id_public_2026/semantic_ids.json`:

| Statistic | Value |
|---|---|
| Catalogue items | 156 746 |
| Items with a Semantic ID | **156 746** (coverage **1.000**) |
| Distinct Semantic IDs | 153 391 |
| Collision groups / items affected | 3 165 / **6 520** |
| Collision rate | **4.16%** of items |
| Largest collision group | 6 |
| Codebook utilisation (level 0 / 1 / 2) | 223/256 = 0.871 · 246/256 = 0.961 · 256/256 = 1.000 |
| Codes not used anywhere | 33 / 10 / 0 |

A 4.16% collision rate means the codes cannot distinguish about one item in twenty-four from its
neighbour. That is a genuine ceiling on the arm and it is reported with the tokenizer rather than
hidden in the generator's metrics.

---

## 2. Generative model architecture

```
history Semantic IDs (one item = L codes)
    -> flatten to [SEP, c1 c2 c3, SEP, c1 c2 c3, ...]   (fixed window of whole items)
    -> causal Transformer encoder (own embedding per codebook LEVEL)
    -> P(next code token) at every position
    -> generated Semantic ID (c1, c2, c3)
```

| Component | Value |
|---|---|
| Positional/layout | one flat vocabulary = `K` codes + 1 separator; a level embedding distinguishes the levels, so code `5` at level 1 is not code `5` at level 2 |
| Size | `d_model` 64, 2 layers, 4 heads, FF×4, pre-norm, GELU, dropout 0.1 |
| History window | 6 items (`(6 + 1) × 4 = 28` tokens, with headroom for a condition prefix) |
| Parameters | ~0.2 M |
| Objective | per-position cross-entropy over the three target code tokens |

**A different objective from SASRec and Two-Tower on purpose.** SASRec fits a per-position binary
logistic loss over sampled negatives; Two-Tower fits an in-batch softmax over 2 048 candidates; this
arm fits a next-code softmax over the code vocabulary. Making it contrastive too would blur what is
being compared.

---

## 3. Training protocol

| | Value |
|---|---|
| Corpus | `EvaluationCase.train_history` **only** — `labels_from = "train_history only"` |
| Training users | all eligible users (same exposure rule as accepted SASRec and corrected Two-Tower) |
| Split | `temporal_leave_two_out`, `agentrecx.eval_protocol.v1`, unchanged |
| Epoch selection | none — the epoch count is a recorded configuration value, never chosen on an evaluation split |
| Seed | 2026 |

The tokenizer is trained on **catalogue features**, which contain no interaction data at all, so
there is nothing for it to leak. Only the generator reads histories.

---

## 4. Decoding and identity resolution

Two independent mechanisms, deliberately in different modules:

```
generated codes ──> constrained decode ──> Semantic ID ──> resolver ──> catalogue item
   (model)          (prefix trie from        (data)      (deterministic)  (or NOTHING)
                     the CATALOGUE)
```

* **Constrained decoding** (`semantic_id/generator.py`): the walk only extends prefixes that some
  catalogue item occupies. The trie is built from the catalogue assignment, never from the model's
  own output distribution, so a hallucinated prefix cannot widen the legal set.
* **Deterministic resolution** (`semantic_id/resolver.py`): a pure function of the Semantic ID and
  the trusted assignment. An unoccupied code resolves to **nothing** — never a nearest neighbour, a
  partial match or a default item. Collisions break by the documented rule
  (`COLLISION_RULE`): *eligible items first, then ascending item id*.
* **Fail-closed.** An eligibility filter that excludes every member of a collision group yields
  nothing rather than an ineligible member; PAD is never resolvable.

The invalid-generation rate is reported with the benchmark results precisely because a rate that is
*asserted* to be zero is worth measuring.

---

## 5. Standalone metrics

Frozen protocol: 20 000-user deterministic cohort, full catalogue (156 746 items), accepted
evaluator owning PAD exclusion, seen-item masking, tie-breaking and ranking, `k ∈ {5,10,20}`.

| Arm | Recall@5 | **Recall@10** | Recall@20 | NDCG@10 | NDCG@20 | HR@10 | mean target rank |
|---|---|---|---|---|---|---|---|
| `sequential` (SASRec) | 0.00820 | **0.01350** | 0.02070 | 0.00685 | 0.00865 | 0.01350 | 20 598 |
| `two_tower` | 0.00860 | **0.01435** | 0.02285 | 0.00737 | 0.00951 | 0.01435 | 21 944 |
| **`semantic_id_genrec`** | 0.00175 | **0.00185** | 0.00295 | 0.00142 | 0.00169 | 0.00185 | 100 118 |

| Diagnostic | Value |
|---|---|
| Invalid generation rate | **0.000** (0 / 20 000) |
| Items the prefix search reached | **534 of 156 746** (`scored_share` 0.0034) |
| Prefixes expanded | 512 leaves over 3 levels |
| Trie size | 188 415 nodes (223 / 34 800 / 153 391 by depth) |

**GenRec is far weaker standalone — 7.3× below SASRec and 7.8× below Two-Tower on Recall@10.** The
honest reading is not "generative retrieval is bad" but "this bounded implementation cannot reach
enough of the catalogue":

* the score of an item is its Semantic-ID log-probability, computed by a breadth-limited search that
  expands **512 prefixes per level**. On a trie with 153 391 level-3 nodes, 512 leaves cover
  **534 items, 0.34% of the catalogue**. Everything else keeps the tail score.
* the evaluator then ranks over the whole catalogue, so an arm that only scores 0.34% of items
  starts with a 0.34% ceiling on where it can place a target. Its mean target rank of 100 118 is the
  tail convention showing through, not a ranking.
* the generator's **6.7% per-code accuracy** compounds to roughly `0.067³ ≈ 0.03%` for a full
  three-code sequence, which is the same order as the observed Recall.

So the ceiling is the interaction of a bounded search with a short, low-accuracy generation — not a
property of Semantic IDs as such. Widening the search is a *different* experiment (and an
ANN/engineering one), and it was explicitly out of scope for this step.

**Invalid generation rate is 0.000, and that is a structural result worth stating precisely.**
Generation is constrained to catalogue code prefixes, so every one of the 20 000 generations landed
on at least one real catalogue item and the resolver never had to fail closed. The guard is a
*guard*: a zero here shows the boundary holds, not that the generations are good.

---

## 6. Complementarity analysis

Per-user hits inside each source's top-1000 head, on the frozen cohort:

| Source | head-1000 target coverage |
|---|---|
| SASRec | 0.2050 |
| Two-Tower | 0.2216 |
| **GenRec** | **0.0116** |

### Pairwise

| Pair | both | left only | right only | neither | union | Jaccard | lift |
|---|---|---|---|---|---|---|---|
| SASRec vs GenRec | 134 | 3 966 | **97** | 15 803 | 4 197 | 0.032 | 1.024 |
| Two-Tower vs GenRec | 118 | 4 313 | **113** | 15 456 | 4 544 | 0.026 | 1.026 |
| SASRec vs Two-Tower *(reference)* | 2 559 | 1 541 | 1 872 | 14 028 | 5 972 | 0.429 | 1.348 |

### Three-way partition

| | Users |
|---|---|
| Hit by all three | 106 |
| Hit by exactly two | 2 493 |
| **Hit by exactly one** — `sequential` 1 513 · `two_tower` 1 860 · **`semantic_id_genrec` 85** | 3 458 |
| Hit by none | 13 943 |
| Union (share) | 6 057 (0.3029) |

### What this says

**GenRec retrieves a genuinely different set — and very little of it.** Its overlap with either
model is the lowest in the table (mean top-10 head Jaccard **0.020** vs SASRec, **0.011** vs
Two-Tower, against 0.083 for SASRec-vs-Two-Tower). Of its 231 head hits, 85 are hits **no other
source** found. So the answer to "does it contribute distinct candidates?" is *yes, distinctly, but
marginally*: adding it to the SASRec ∪ Two-Tower union moves coverage from 0.2986 to 0.3029, a gain
of **85 users (0.0043 of the cohort)**. That is a real contribution and a small one, and it should
be read as "a weak, orthogonal source" rather than as a promising new arm at this training level.

---

## 7. Known limitations

1. **The prefix search is the binding constraint.** With 512 expanded prefixes per level the arm
   scored **534 of 156 746 items (0.34%)**. Items outside that set keep the documented tail score, so
   the reported metric is a retrieval metric over the expanded prefixes. This — not the tokenizer —
   is the dominant reason the standalone number is low.
2. **The generator's per-code accuracy is low.** 6.7% after 5 epochs, which compounds to ~0.03% for
   a full three-code sequence. More epochs would very likely improve it; that is a training-budget
   change and was not made, because tuning against the evaluation cohort is exactly what this
   step's rules forbid.
3. **Semantic-ID collisions cap the arm.** With a 4.16% collision rate, items sharing a code share a
   score and the resolver can present only one of them.
4. **Untrained feature tables.** The item features are seeded, not learned, so the codes group items
   by *random* projections plus metadata categories rather than by any behavioural signal. A learned
   item representation would be a different (and larger) design.
5. **Small model, short window.** 0.2 M parameters and a 6-item window, sized for CPU
   reproducibility. Nothing here is OneRec-scale, and the step's non-goals forbid pretending
   otherwise.
6. **No fusion.** The arm is measured standalone only; adding it to the accepted fusion is a later
   decision, and Step 1.1's lesson is that a fusion change must be controlled before it is claimed.
7. **Invalid-generation rate is a floor, not a quality measure.** Constrained decoding makes it
   structurally zero; a zero here says the guard works, not that the generations are good.
